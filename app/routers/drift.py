"""
Эндпоинты для мониторинга дрейфа данных.
Считает реальные признаки из папок с изображениями и возвращает в UI.
"""

import os
import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import APIRouter

logger = logging.getLogger(__name__)
router = APIRouter()

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

REFERENCE_DIR = os.getenv("REFERENCE_DIR", "dataset/images")
CURRENT_DIR   = os.getenv("CURRENT_DIR",   "monitoring/drift/current")


def extract_features(img_path: str) -> Optional[dict]:
    """Извлекает числовые признаки из одного изображения."""
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return None

    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    return {
        "brightness": float(img_gray.mean()),
        "contrast":   float(img_gray.std()),
        "saturation": float(img_hsv[:, :, 1].mean()),
        "blur_score": float(cv2.Laplacian(img_gray, cv2.CV_64F).var()),
    }


def compute_folder_stats(folder: str) -> Optional[dict]:
    """
    Считает средние признаки по всем изображениям в папке.
    Возвращает None если папка пустая или не найдена.
    """
    folder = Path(folder)
    if not folder.exists():
        logger.warning(f"Папка не найдена: {folder}")
        return None

    files = [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]
    if not files:
        logger.warning(f"Нет изображений в: {folder}")
        return None

    all_feats = [extract_features(str(f)) for f in files]
    all_feats = [f for f in all_feats if f is not None]
    if not all_feats:
        return None

    # Усредняем по всем изображениям
    keys = all_feats[0].keys()
    return {
        k: round(float(np.mean([f[k] for f in all_feats])), 2)
        for k in keys
    }


@router.get("/drift/stats", summary="Реальные статистики дрейфа", tags=["monitoring"])
def drift_stats():
    """
    Возвращает реальные средние признаки эталонных и текущих изображений.
    UI использует эти данные вместо захардкоженных значений.
    """
    ref_stats = compute_folder_stats(REFERENCE_DIR)
    cur_stats = compute_folder_stats(CURRENT_DIR)

    ref_count = len([
        f for f in Path(REFERENCE_DIR).iterdir()
        if f.suffix.lower() in IMG_EXTENSIONS
    ]) if Path(REFERENCE_DIR).exists() else 0

    cur_count = len([
        f for f in Path(CURRENT_DIR).iterdir()
        if f.suffix.lower() in IMG_EXTENSIONS
    ]) if Path(CURRENT_DIR).exists() else 0

    return {
        "reference": ref_stats,
        "current":   cur_stats,
        "ref_count": ref_count,
        "cur_count": cur_count,
        "reference_dir": REFERENCE_DIR,
        "current_dir":   CURRENT_DIR,
    }
