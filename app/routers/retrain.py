"""
Эндпоинты реального переобучения модели.

Запускает обучение U-Net в фоновом потоке, отдаёт прогресс по эпохам,
по завершении атомарно подменяет рабочую модель в app.state.model,
чтобы инференс сразу использовал новые веса — без перезапуска сервиса.
Каждое переобучение логируется в MLflow (параметры, метрики, модель).

Эндпоинты:
    POST /api/v1/retrain         — запустить (тело: {"epochs": N})
    GET  /api/v1/retrain/status  — текущий статус/прогресс
"""

import os
import copy
import shutil
import logging
import threading
import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

# --- MLflow: трекинг экспериментов (необязательная зависимость) --------------
try:
    import mlflow
    _MLFLOW_OK = True
except Exception:
    _MLFLOW_OK = False

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Общее состояние процесса обучения ────────────────────────────────────────
_train_lock = threading.Lock()

_status = {
    "state": "idle",
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "val_loss": None,
    "metric": None,
    "message": "Готово к запуску",
    "started_at": None,
    "finished_at": None,
}
_status_guard = threading.Lock()


def _set_status(**kwargs):
    with _status_guard:
        _status.update(kwargs)


def get_status() -> dict:
    with _status_guard:
        return dict(_status)


class RetrainRequest(BaseModel):
    epochs: int = Field(1, ge=1, le=200, description="Сколько эпох обучать")


class RetrainStatus(BaseModel):
    state: str
    epoch: int
    total_epochs: int
    loss: Optional[float] = None
    val_loss: Optional[float] = None
    metric: Optional[float] = None
    message: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None


def _make_progress_callback(total_epochs: int):
    """Keras-колбэк, который пишет прогресс в общий _status после каждой эпохи."""
    import tensorflow as tf

    class _Progress(tf.keras.callbacks.Callback):
        def on_epoch_begin(self, epoch, logs=None):
            _set_status(
                epoch=epoch + 1,
                message=f"Обучение: эпоха {epoch + 1} из {total_epochs}",
            )

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            metric = None
            for k, v in logs.items():
                if "io_u" in k and not k.startswith("val_"):
                    metric = float(v)
                    break
            _set_status(
                epoch=epoch + 1,
                loss=float(logs.get("loss")) if "loss" in logs else None,
                val_loss=float(logs.get("val_loss")) if "val_loss" in logs else None,
                metric=metric,
            )

    return _Progress()


def _run_training(app, epochs: int, model_path: str):
    """Тело фоновой задачи. Выполняется в отдельном потоке."""
    import tensorflow as tf
    from src.data import (
        SegmentationDataGenerator,
        load_and_split_data,
        get_augmentations,
    )
    from src.models.unet import create_unet_model

    mlflow_run = None
    try:
        _set_status(
            state="running",
            epoch=0,
            total_epochs=epochs,
            loss=None,
            val_loss=None,
            metric=None,
            message="Подготовка данных…",
            started_at=datetime.datetime.now().isoformat(timespec="seconds"),
            finished_at=None,
        )

        cfg = copy.deepcopy(app.state.cfg)
        cfg["training"]["epochs"] = epochs

        # ── старт записи в MLflow ────────────────────────────────────────────
        if _MLFLOW_OK:
            try:
                uri = os.getenv("MLFLOW_TRACKING_URI")
                if uri:
                    mlflow.set_tracking_uri(uri)
                mlflow.set_experiment("horizon-detection")
                mlflow_run = mlflow.start_run(run_name=f"retrain-{epochs}ep")
                mlflow.log_params({
                    "epochs": epochs,
                    "learning_rate": cfg["training"]["learning_rate"],
                    "n_encoder_decoder": cfg["training"]["n_encoder_decoder"],
                    "initial_filters": cfg["training"]["initial_filters"],
                    "batch_size": cfg["data"]["batch_size"],
                    "image_size": str(cfg["data"]["image_size"]),
                    "source": "ui-retrain",
                })
            except Exception as e:
                logger.warning("MLflow init failed: %s", e)
                mlflow_run = None

        paths = cfg["paths"]
        dataset_dir = Path(paths["dataset"])
        img_size = tuple(cfg["data"]["image_size"])

        train_list, val_list = load_and_split_data(dataset_dir, cfg["data"])
        if not train_list:
            raise RuntimeError("Нет данных для обучения в dataset/.")

        train_gen = SegmentationDataGenerator(
            file_list=train_list, data_cfg=cfg, augmentation=get_augmentations()
        )
        val_gen = SegmentationDataGenerator(file_list=val_list, data_cfg=cfg)
        val_gen.shuffle = False

        _set_status(message="Построение модели U-Net…")
        model = create_unet_model(
            image_size=img_size,
            num_classes=cfg["data"]["num_classes"],
            learning_rate=cfg["training"]["learning_rate"],
            n_encoder_decoder=cfg["training"]["n_encoder_decoder"],
            initial_filters=cfg["training"]["initial_filters"],
        )

        _set_status(message=f"Старт обучения на {epochs} эпох…")
        model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            callbacks=[_make_progress_callback(epochs)],
            verbose=0,
        )

        _set_status(message="Сохранение новой модели…")
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = model_path.parent / (model_path.stem + ".tmp.keras")
        backup_path = model_path.parent / (model_path.stem + ".bak.keras")
        model.save(str(tmp_path))

        if model_path.exists():
            shutil.copy2(str(model_path), str(backup_path))
        os.replace(str(tmp_path), str(model_path))

        app.state.model = model

        metric = get_status().get("metric")
        _set_status(
            state="done",
            message="Переобучение завершено. Модель обновлена.",
            finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        logger.info("Retrain finished. New model live. IoU=%s", metric)

        # ── запись метрик и модели в MLflow ──────────────────────────────────
        if _MLFLOW_OK and mlflow_run is not None:
            try:
                st = get_status()
                for key in ("loss", "val_loss", "metric"):
                    if st.get(key) is not None:
                        mlflow.log_metric(key, float(st[key]))
                mlflow.log_artifact(str(model_path), artifact_path="model")
                logger.info("[MLflow] retrain run logged")
            except Exception as e:
                logger.warning("MLflow logging failed: %s", e)

    except Exception as e:  # noqa: BLE001
        logger.exception("Retrain failed")
        _set_status(
            state="error",
            message=f"Ошибка обучения: {e}",
            finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
    finally:
        # Закрываем запись MLflow, если открыта.
        if _MLFLOW_OK and mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass
        if _train_lock.locked():
            _train_lock.release()


@router.post("/retrain", summary="Запустить переобучение", tags=["training"])
def start_retrain(request: Request, body: RetrainRequest):
    """Запускает обучение в фоне и сразу возвращает ответ."""
    if not _train_lock.acquire(blocking=False):
        raise HTTPException(status_code=409, detail="Обучение уже выполняется.")

    model_path = os.getenv("MODEL_PATH", "checkpoints/model.keras")
    app = request.app

    thread = threading.Thread(
        target=_run_training,
        args=(app, body.epochs, model_path),
        daemon=True,
    )
    thread.start()

    return {"status": "started", "epochs": body.epochs}


@router.get(
    "/retrain/status",
    response_model=RetrainStatus,
    summary="Статус переобучения",
    tags=["training"],
)
def retrain_status():
    """Возвращает текущее состояние процесса обучения для опроса из UI."""
    return RetrainStatus(**get_status())
