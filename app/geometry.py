"""
Geometric utilities for horizon line estimation.

Given a binary sky/land mask the module:
1. Cleans the predicted mask (removes speckle, keeps the dominant sky region).
2. Extracts the sky/land boundary pixels.
3. Fits a line robustly (cv2.fitLine, Huber loss) — стабильно при любом наклоне,
   включая почти вертикальный горизонт, где y = slope*x + b ломается.
4. Converts the line direction to roll and pitch angles.

Roll  (bank)  = angle of the horizon line relative to the image x-axis [degrees]
Pitch (tilt)  = angle of horizon centre offset from image centre        [degrees]
  – positive pitch means the horizon is *below* image centre (nose up).
"""

import math
from typing import Optional, Tuple

import cv2
import numpy as np


SKY_CLASS = 0   # channel index in the mask (softmax output order)
LAND_CLASS = 1

# Минимальная доля пикселей неба/земли, при которой вообще считаем,
# что в кадре есть граница (а не сплошное небо / сплошная земля).
MIN_CLASS_RATIO = 0.02


def clean_sky_mask(sky_mask: np.ndarray) -> np.ndarray:
    """
    Убрать шум сегментации перед поиском границы.

    Шаги:
    - морфологическое открытие → удаляет одиночные «крошки»;
    - морфологическое закрытие → заполняет мелкие дыры;
    - оставляем только крупнейшую связную область неба.

    Без этого разрозненные ложные пиксели тянут регрессию к горизонтали
    и угол крена занижается (например, реальные ~80° превращаются в ~0°).
    """
    m = (sky_mask > 0).astype(np.uint8)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    if num <= 1:
        return m  # нет ни одной области неба

    # компонента 0 — это фон; берём крупнейшую из остальных
    biggest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return (labels == biggest).astype(np.uint8)


def extract_boundary_pixels(mask: np.ndarray) -> np.ndarray:
    """
    Return (N, 2) array of (x, y) boundary pixels between sky and land.

    mask shape: (H, W, num_classes), dtype uint8, values 0 or 1.
    """
    sky_mask = (mask[:, :, SKY_CLASS] == 1).astype(np.uint8)
    sky_mask = clean_sky_mask(sky_mask)

    # Morphological gradient = dilation - erosion → thin boundary ring
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    boundary = cv2.morphologyEx(sky_mask, cv2.MORPH_GRADIENT, kernel)

    ys, xs = np.where(boundary > 0)
    if len(xs) == 0:
        return np.empty((0, 2), dtype=np.float32)
    return np.column_stack([xs, ys]).astype(np.float32)


def fit_horizon_line(
    boundary_pts: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Подобрать прямую к пикселям границы устойчивым методом.

    Возвращает (vx, vy) — единичный направляющий вектор линии — или None.

    В отличие от np.polyfit(y ~ x) этот метод (cv2.fitLine с Huber) корректно
    работает при крутом и вертикальном наклоне и менее чувствителен к выбросам.
    """
    if len(boundary_pts) < 10:
        return None

    vx, vy, _x0, _y0 = cv2.fitLine(
        boundary_pts, cv2.DIST_HUBER, 0, 0.01, 0.01
    ).ravel()
    return float(vx), float(vy)


def compute_angles(
    vx: float,
    vy: float,
    boundary_pts: np.ndarray,
    image_height: int,
    image_width: int,
) -> Tuple[float, float]:
    """
    Convert horizon line direction to roll and pitch in degrees.

    roll  = угол направляющего вектора линии относительно оси x,
            приведённый к диапазону (-90°, 90°].
    pitch = отклонение середины горизонта от центра кадра,
            выраженное углом через приближение камеры-обскуры
            (квадратные пиксели, вертикальный FoV ~60°).
    """
    # Угол линии. atan2 не имеет проблемы вертикали (в отличие от atan(slope)).
    roll_deg = math.degrees(math.atan2(vy, vx))
    # Приводим к (-90, 90]: линия не имеет направления, наклон ±180° эквивалентен.
    if roll_deg > 90:
        roll_deg -= 180
    elif roll_deg <= -90:
        roll_deg += 180

    # y-координата горизонта в центре кадра берётся как медиана y граничных
    # пикселей около вертикального центра — устойчивее, чем экстраполяция линии.
    x_mid = image_width / 2.0
    near_centre = boundary_pts[np.abs(boundary_pts[:, 0] - x_mid) < max(2.0, image_width * 0.1)]
    if len(near_centre) > 0:
        y_horizon = float(np.median(near_centre[:, 1]))
    else:
        y_horizon = float(np.median(boundary_pts[:, 1]))

    y_centre = image_height / 2.0
    delta_y = y_centre - y_horizon  # пиксели, положит. = горизонт ниже центра

    half_vfov_rad = math.radians(30.0)
    pitch_rad = math.atan(delta_y / (image_height / 2.0) * math.tan(half_vfov_rad))
    pitch_deg = math.degrees(pitch_rad)

    return round(roll_deg, 3), round(pitch_deg, 3)


def estimate_horizon(
    mask: np.ndarray,
) -> Tuple[bool, Optional[float], Optional[float]]:
    """
    High-level entry point.

    Returns:
        (horizon_detected, roll_deg, pitch_deg)
    """
    h, w = mask.shape[:2]
    total = float(h * w)

    # Если в кадре практически только небо или только земля — границы нет.
    sky_ratio = float((mask[:, :, SKY_CLASS] == 1).sum()) / total
    land_ratio = float((mask[:, :, LAND_CLASS] == 1).sum()) / total
    if sky_ratio < MIN_CLASS_RATIO or land_ratio < MIN_CLASS_RATIO:
        return False, None, None

    boundary = extract_boundary_pixels(mask)
    line = fit_horizon_line(boundary)
    if line is None:
        return False, None, None

    vx, vy = line
    roll, pitch = compute_angles(vx, vy, boundary, h, w)
    return True, roll, pitch
