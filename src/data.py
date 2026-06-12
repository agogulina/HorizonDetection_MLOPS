import os
import cv2
import numpy as np
import math
import albumentations as albu
from pathlib import Path
from sklearn.model_selection import train_test_split
import tensorflow as tf


def get_augmentations() -> albu.Compose:
    """Augmentation pipline"""
    return albu.Compose([
        albu.HorizontalFlip(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.ShiftScaleRotate(p=0.5, border_mode=cv2.BORDER_CONSTANT),
        albu.RandomBrightnessContrast(p=0.25),
        albu.RandomGamma(p=0.25)
    ])


class SegmentationDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, file_list: list, data_cfg: dict, augmentation=None):
        self.file_list = file_list
        self.batch_size = data_cfg.get("batch_size", 32)
        self.image_size = tuple(data_cfg.get("image_size", (128, 128)))
        self.shuffle = True
        self.augmentation = augmentation

        paths = data_cfg.get("paths", {})
        self.images_dir = Path(paths.get("images", "dataset/images"))
        self.masks_land_dir = Path(paths.get("masks_land", "dataset/masks/land"))
        self.masks_sky_dir = Path(paths.get("masks_sky", "dataset/masks/sky"))
        
        self.on_epoch_end() # shuffle after every epoch

    def __len__(self) -> int:
        return math.ceil(len(self.file_list) / self.batch_size)

    def on_epoch_end(self) -> None:
        self.indices = np.arange(len(self.file_list))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __getitem__(self, index: int):
        batch_indices = self.indices[index * self.batch_size:(index + 1) * self.batch_size]
        batch_filenames = [self.file_list[k] for k in batch_indices]

        images_batch, masks_batch = [], []

        for filename in batch_filenames:
            # Load Image
            img_path = self.images_dir / f"{filename}.jpg"
            img = cv2.imread(str(img_path))
            if img is None:
                raise FileNotFoundError(f"Image not found: {img_path}")
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            # Load Masks
            mask_land_path = self.masks_land_dir / f"{filename}.png"
            mask_sky_path = self.masks_sky_dir / f"{filename}.png"
            mask_land = cv2.imread(str(mask_land_path), cv2.IMREAD_GRAYSCALE)
            mask_sky = cv2.imread(str(mask_sky_path), cv2.IMREAD_GRAYSCALE)
            if mask_land is None or mask_sky is None:
                raise FileNotFoundError(f"Mask missing for: {filename}")
            mask = np.dstack((mask_land, mask_sky))

            # Resize
            img = cv2.resize(img, self.image_size)
            mask = cv2.resize(mask, self.image_size, interpolation=cv2.INTER_NEAREST)

            # Augmentation
            if self.augmentation is not None:
                augmented = self.augmentation(image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]

            # Normalize to [0, 1]
            img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)
            mask = cv2.normalize(mask, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F)

            images_batch.append(img)
            masks_batch.append(mask)

        return np.array(images_batch), np.array(masks_batch)


def load_and_split_data(dataset_dir: str, data_cfg: dict):
    """Load files and split on train/val"""
    images_dir = Path(dataset_dir) / "images"
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    file_list = [p.stem for p in images_dir.iterdir() if p.suffix.lower() in [".jpg", ".jpeg", ".png"]]
    
    return train_test_split(
        file_list,
        test_size=data_cfg.get("test_size", 0.2),
        shuffle=True,
        random_state=data_cfg.get("random_seed", 42)
    )