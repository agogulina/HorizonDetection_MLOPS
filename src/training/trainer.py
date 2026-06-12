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
        val_gen.shuffle=False

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
            #max_queue_size=self.cfg["training"]["max_queue_size"],
            #workers=self.cfg["training"]["workers"],
            #use_multiprocessing=self.cfg["training"]["use_multiprocessing"]
        )

        print("Evaluating")
        eval_results = model.evaluate(val_gen, return_dict=True)
        elapsed = timer() - start

        self._log_results(eval_results, elapsed)

    def _log_results(self, metrics, elapsed):
        print(f"Completed at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Elapsed: {elapsed:.2f} sec")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
        print("="*50)