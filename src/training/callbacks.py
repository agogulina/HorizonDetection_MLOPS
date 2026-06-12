import os
import datetime
import numpy as np
import cv2
import tensorflow as tf
from pathlib import Path


def create_callbacks(model, val_file_list, log_dir, ckpt_dir, cfg):
    """Callback for logging, saving ckpt, visualizations"""
    log_dir = Path(log_dir)
    ckpt_dir = Path(ckpt_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    predict_log_dir = log_dir / "predict_output"
    if predict_log_dir.exists() and not predict_log_dir.is_dir():
        predict_log_dir.unlink()
    predict_log_dir.mkdir(parents=True, exist_ok=True)
    predict_log_dir = (log_dir / "predict_output").resolve()
    print(Path(predict_log_dir).is_dir())
    tf.io.gfile.makedirs(str(predict_log_dir))
    file_writer = tf.summary.create_file_writer(str(predict_log_dir))

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tb_callback = tf.keras.callbacks.TensorBoard(
        log_dir=str(log_dir / "train"),
        histogram_freq=1
    )

    ckpt_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(ckpt_dir / f"best_{timestamp}.keras"),
        monitor='max_mean_io_u',
        verbose=1,
        save_best_only=True,
        mode='max'
    )

    data_cfg = cfg.get("data", {})
    images_dir = Path(data_cfg.get("paths", {}).get("images", "dataset/images"))
    image_size = tuple(data_cfg.get("image_size", (128, 128)))

    def predict_epoch(epoch, logs):
        if not val_file_list:
            return

        # Random sample
        sample_filename = np.random.choice(val_file_list)
        img_path = images_dir / f"{sample_filename}.jpg"
        img = cv2.imread(str(img_path))
        if img is None:
            return
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img, image_size)
        img_norm = cv2.normalize(img_resized, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

        # Inference
        pred = model.predict(np.expand_dims(img_norm, 0), verbose=0).squeeze()

        # Binary
        pred_binary = (pred >= 0.5).astype(np.float32)

        # Draw image
        img_vis = np.clip(img_resized.astype(np.float32), 0, 1)
        land_mask = np.stack([pred_binary[:, :, 0]] * 3, axis=-1)
        sky_mask = np.stack([pred_binary[:, :, 1]] * 3, axis=-1)

        class_land = np.concatenate([img_vis, land_mask, img_vis * land_mask], axis=1)
        class_sky = np.concatenate([img_vis, sky_mask, img_vis * sky_mask], axis=1)

        # Write to TensorBoard
        with file_writer.as_default():
            tf.summary.image("class_land", np.expand_dims(class_land, 0), step=epoch)
            tf.summary.image("class_sky", np.expand_dims(class_sky, 0), step=epoch)

    pred_callback = tf.keras.callbacks.LambdaCallback(on_epoch_end=predict_epoch)

    return [tb_callback, ckpt_callback, pred_callback]