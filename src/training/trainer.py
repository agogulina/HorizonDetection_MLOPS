import os
import datetime
import random
import numpy as np
import tensorflow as tf
from time import time as timer
from pathlib import Path

from src.data import SegmentationDataGenerator, load_and_split_data, get_augmentations
from src.models.unet import create_unet_model
from src.training.callbacks import create_callbacks

try:
    import mlflow
    _MLFLOW_OK = True
except Exception:
    _MLFLOW_OK = False


class Trainer:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._set_seed(cfg.get("seed", 42))

    def _set_seed(self, seed: int):
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        tf.random.set_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def train(self):
        paths = self.cfg["paths"]
        dataset_dir = Path(paths["dataset"])
        log_dir = Path(paths["logs"])
        ckpt_dir = Path(paths["checkpoints"])

        run_ctx = None
        if _MLFLOW_OK:
            try:
                mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "file:./mlruns"))
                mlflow.set_experiment("horizon-detection")
                run_ctx = mlflow.start_run()
                t = self.cfg["training"]
                d = self.cfg["data"]
                mlflow.log_params({
                    "epochs": t["epochs"],
                    "learning_rate": t["learning_rate"],
                    "n_encoder_decoder": t["n_encoder_decoder"],
                    "initial_filters": t["initial_filters"],
                    "batch_size": d["batch_size"],
                    "image_size": str(d["image_size"]),
                    "num_classes": d["num_classes"],
                    "seed": self.cfg.get("seed", 42),
                })
            except Exception as e:
                print(f"[MLflow] не удалось инициализировать трекинг: {e}")
                run_ctx = None

        print("Loading data")
        train_list, val_list = load_and_split_data(dataset_dir, self.cfg["data"])

        img_size = tuple(self.cfg["data"]["image_size"])
        train_gen = SegmentationDataGenerator(
            file_list=train_list,
            data_cfg=self.cfg,
            augmentation=get_augmentations()
        )
        val_gen = SegmentationDataGenerator(
            file_list=val_list,
            data_cfg=self.cfg,
        )
        val_gen.shuffle = False

        print("Building UNet")
        model = create_unet_model(
            image_size=img_size,
            num_classes=self.cfg["data"]["num_classes"],
            learning_rate=self.cfg["training"]["learning_rate"],
            n_encoder_decoder=self.cfg["training"]["n_encoder_decoder"],
            initial_filters=self.cfg["training"]["initial_filters"]
        )

        print("Setting up callbacks")
        callbacks = create_callbacks(
            model=model,
            val_file_list=val_list,
            log_dir=log_dir,
            ckpt_dir=ckpt_dir,
            cfg=self.cfg
        )

        print(f"Starting training for {self.cfg['training']['epochs']} epochs")
        start = timer()
        model.fit(
            train_gen,
            epochs=self.cfg["training"]["epochs"],
            validation_data=val_gen,
            callbacks=callbacks,
        )

        print("Evaluating")
        eval_results = model.evaluate(val_gen, return_dict=True)
        elapsed = timer() - start

        self._log_results(eval_results, elapsed)

        # запись метрик и модели в MLflow 
        if _MLFLOW_OK and run_ctx is not None:
            try:
                mlflow.log_metric("elapsed_sec", float(elapsed))
                for k, v in eval_results.items():
                    mlflow.log_metric(k, float(v))
                # логируем обученную модель как артефакт
                ckpt_file = ckpt_dir / "model.keras"
                if ckpt_file.exists():
                    mlflow.log_artifact(str(ckpt_file), artifact_path="model")
                print("[MLflow] запуск залогирован")
            except Exception as e:
                print(f"[MLflow] ошибка при логировании: {e}")
            finally:
                mlflow.end_run()

    def _log_results(self, metrics, elapsed):
        print(f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed: {elapsed:.2f} sec")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("=" * 50)
