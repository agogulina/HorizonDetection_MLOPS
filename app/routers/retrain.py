"""
Эндпоинты реального переобучения модели.

Запускает обучение U-Net в фоновом потоке, отдаёт прогресс по эпохам,
по завершении атомарно подменяет рабочую модель в app.state.model,
чтобы инференс сразу использовал новые веса — без перезапуска сервиса.

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

logger = logging.getLogger(__name__)
router = APIRouter()

# ── Общее состояние процесса обучения ────────────────────────────────────────
# Один лок гарантирует, что одновременно не запустятся два обучения.
_train_lock = threading.Lock()

_status = {
    "state": "idle",          # idle | running | done | error
    "epoch": 0,
    "total_epochs": 0,
    "loss": None,
    "val_loss": None,
    "metric": None,           # последнее значение основной метрики (IoU)
    "message": "Готово к запуску",
    "started_at": None,
    "finished_at": None,
}
_status_guard = threading.Lock()   # защищает _status от гонок при чтении/записи


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
            # Имя метрики IoU в этом проекте — max_mean_io_u (см. callbacks.py)
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

        # Берём конфиг приложения и переопределяем число эпох из запроса.
        cfg = copy.deepcopy(app.state.cfg)
        cfg["training"]["epochs"] = epochs

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

        # ── Атомарное сохранение и горячая подмена ──────────────────────────
        _set_status(message="Сохранение новой модели…")
        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # Сначала пишем во временный файл, затем заменяем — чтобы при сбое
        # не остаться без рабочего чекпойнта.
        # ВАЖНО: Keras определяет формат по расширению, поэтому временный и
        # резервный файлы тоже должны оканчиваться на .keras, иначе model.save
        # падает с "Invalid filepath extension for saving".
        tmp_path = model_path.parent / (model_path.stem + ".tmp.keras")
        backup_path = model_path.parent / (model_path.stem + ".bak.keras")
        model.save(str(tmp_path))

        if model_path.exists():
            shutil.copy2(str(model_path), str(backup_path))
        os.replace(str(tmp_path), str(model_path))

        # Горячая подмена модели в памяти сервиса — инференс сразу новый.
        app.state.model = model

        metric = get_status().get("metric")
        _set_status(
            state="done",
            message="Переобучение завершено. Модель обновлена.",
            finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
        logger.info("Retrain finished. New model live. IoU=%s", metric)

    except Exception as e:  # noqa: BLE001 — хотим поймать любую ошибку обучения
        logger.exception("Retrain failed")
        _set_status(
            state="error",
            message=f"Ошибка обучения: {e}",
            finished_at=datetime.datetime.now().isoformat(timespec="seconds"),
        )
    finally:
        # Освобождаем лок в любом случае.
        if _train_lock.locked():
            _train_lock.release()


@router.post("/retrain", summary="Запустить переобучение", tags=["training"])
def start_retrain(request: Request, body: RetrainRequest):
    """
    Запускает обучение в фоне и сразу возвращает ответ.
    Прогресс отслеживается через GET /retrain/status.
    """
    # Пытаемся захватить лок без блокировки: если занят — обучение уже идёт.
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
