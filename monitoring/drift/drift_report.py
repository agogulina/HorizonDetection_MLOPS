"""
Генератор отчётов о дрейфе данных с помощью Evidently.

Что делает скрипт:
1. Берёт все изображения из dataset/images/ как ЭТАЛОН (reference).
2. Берёт последние N изображений из monitoring/drift/current/ как ТЕКУЩИЕ данные.
3. Считает признаки каждого изображения: яркость, контраст, насыщенность,
   размытость, sky_ratio, roll_deg, pitch_deg.
4. Генерирует HTML отчёт с графиками дрейфа.

Запуск:
    python monitoring/drift/drift_report.py

Или с указанием папок:
    python monitoring/drift/drift_report.py \
        --reference dataset/images \
        --current   monitoring/drift/current \
        --output    monitoring/drift/reports/report.html
"""

import argparse
import os
import sys
import json
import logging
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

# Добавляем корень проекта в путь чтобы импортировать src/
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


# ── Признаки изображения ─────────────────────────────────────────────────────

def extract_features(img_path: str) -> dict:
    """
    Извлекает числовые признаки из изображения.
    Возвращает словарь {feature_name: float}.
    """
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        logger.warning(f"Не удалось прочитать: {img_path}")
        return None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    h, w = img_gray.shape

    # Яркость (mean intensity)
    brightness = float(img_gray.mean())

    # Контраст (std intensity)
    contrast = float(img_gray.std())

    # Насыщенность (saturation channel в HSV)
    saturation = float(img_hsv[:, :, 1].mean())

    # Размытость (Laplacian variance — чем ниже, тем более размыто)
    blur = float(cv2.Laplacian(img_gray, cv2.CV_64F).var())

    # Соотношение сторон
    aspect_ratio = float(w) / float(h)

    # Среднее значение каждого RGB канала
    r_mean = float(img_rgb[:, :, 0].mean())
    g_mean = float(img_rgb[:, :, 1].mean())
    b_mean = float(img_rgb[:, :, 2].mean())

    # Доля "синих" пикселей (грубая эвристика для неба)
    blue_mask = (img_rgb[:, :, 2] > img_rgb[:, :, 0]) & (img_rgb[:, :, 2] > img_rgb[:, :, 1])
    sky_ratio_color = float(blue_mask.mean())

    # Энтропия верхней и нижней половины (небо обычно однородное)
    top_half = img_gray[: h // 2, :]
    bot_half = img_gray[h // 2 :, :]
    top_std = float(top_half.std())
    bot_std = float(bot_half.std())

    return {
        "brightness": brightness,
        "contrast": contrast,
        "saturation": saturation,
        "blur_score": blur,
        "aspect_ratio": aspect_ratio,
        "r_mean": r_mean,
        "g_mean": g_mean,
        "b_mean": b_mean,
        "sky_ratio_color": sky_ratio_color,
        "top_half_std": top_std,
        "bottom_half_std": bot_std,
        "filename": os.path.basename(img_path),
    }


def load_features_from_folder(folder: str) -> pd.DataFrame:
    """Загружает признаки всех изображений из папки в DataFrame."""
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Папка не найдена: {folder}")

    records = []
    files = [f for f in folder.iterdir() if f.suffix.lower() in IMG_EXTENSIONS]

    if not files:
        raise ValueError(f"В папке {folder} нет изображений ({IMG_EXTENSIONS})")

    logger.info(f"Загружаю признаки из {folder} ({len(files)} изображений)...")
    for f in sorted(files):
        feat = extract_features(str(f))
        if feat:
            records.append(feat)

    df = pd.DataFrame(records)
    logger.info(f"  → {len(df)} изображений обработано")
    return df


# ── Генерация отчёта ─────────────────────────────────────────────────────────

def generate_report(
    reference_folder: str,
    current_folder: str,
    output_path: str,
):
    """Генерирует HTML отчёт о дрейфе данных."""

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import ColumnDriftMetric
        EVIDENTLY_AVAILABLE = True
    except ImportError:
        EVIDENTLY_AVAILABLE = False
        logger.warning("Evidently не установлен. Генерирую упрощённый HTML отчёт.")

    ref_df = load_features_from_folder(reference_folder)
    cur_df = load_features_from_folder(current_folder)

    feature_cols = [c for c in ref_df.columns if c != "filename"]

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    if EVIDENTLY_AVAILABLE:
        _generate_evidently_report(ref_df[feature_cols], cur_df[feature_cols], output_path)
    else:
        _generate_simple_html_report(ref_df, cur_df, feature_cols, output_path)

    logger.info(f"✅ Отчёт сохранён: {output_path}")


def _generate_evidently_report(ref_df, cur_df, output_path):
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset, DataQualityPreset

    report = Report(metrics=[
        DataDriftPreset(),
        DataQualityPreset(),
    ])
    report.run(reference_data=ref_df, current_data=cur_df)
    report.save_html(output_path)


def _generate_simple_html_report(ref_df, cur_df, feature_cols, output_path):
    """Упрощённый HTML отчёт без Evidently — только статистика и таблицы."""
    from scipy import stats as scipy_stats

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    drift_results = []
    for col in feature_cols:
        ref_vals = ref_df[col].dropna().values
        cur_vals = cur_df[col].dropna().values
        if len(ref_vals) < 2 or len(cur_vals) < 2:
            continue
        stat, pvalue = scipy_stats.ks_2samp(ref_vals, cur_vals)
        drifted = pvalue < 0.05
        drift_results.append({
            "feature": col,
            "ref_mean": round(ref_vals.mean(), 3),
            "ref_std": round(ref_vals.std(), 3),
            "cur_mean": round(cur_vals.mean(), 3),
            "cur_std": round(cur_vals.std(), 3),
            "ks_stat": round(stat, 4),
            "p_value": round(pvalue, 4),
            "drifted": drifted,
        })

    drifted_count = sum(1 for r in drift_results if r["drifted"])
    total = len(drift_results)

    rows = ""
    for r in drift_results:
        color = "#ffcccc" if r["drifted"] else "#ccffcc"
        label = "⚠️ ДРЕЙФ" if r["drifted"] else "✅ норма"
        rows += f"""
        <tr style="background:{color}">
            <td>{r['feature']}</td>
            <td>{r['ref_mean']} ± {r['ref_std']}</td>
            <td>{r['cur_mean']} ± {r['cur_std']}</td>
            <td>{r['ks_stat']}</td>
            <td>{r['p_value']}</td>
            <td><b>{label}</b></td>
        </tr>"""

    html = f"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="UTF-8">
<title>Drift Report — Horizon Detection</title>
<style>
  body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
  h1 {{ color: #333; }}
  .summary {{ background: white; padding: 20px; border-radius: 8px; margin-bottom: 20px;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  .badge {{ display: inline-block; padding: 4px 12px; border-radius: 12px;
            font-weight: bold; font-size: 14px; }}
  .ok {{ background: #ccffcc; color: #006600; }}
  .warn {{ background: #ffcccc; color: #cc0000; }}
  table {{ width: 100%; border-collapse: collapse; background: white;
           border-radius: 8px; overflow: hidden;
           box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  th {{ background: #4a90d9; color: white; padding: 12px; text-align: left; }}
  td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
  .footer {{ margin-top: 20px; color: #888; font-size: 12px; }}
</style>
</head>
<body>
<h1>📊 Отчёт о дрейфе данных — Horizon Detection API</h1>

<div class="summary">
  <p><b>Дата генерации:</b> {timestamp}</p>
  <p><b>Эталонных изображений:</b> {len(ref_df)}</p>
  <p><b>Текущих изображений:</b> {len(cur_df)}</p>
  <p><b>Признаков с дрейфом:</b>
     <span class="badge {'warn' if drifted_count > 0 else 'ok'}">
       {drifted_count} из {total}
     </span>
  </p>
  <p><i>Метод: тест Колмогорова–Смирнова, порог p &lt; 0.05</i></p>
</div>

<table>
  <thead>
    <tr>
      <th>Признак</th>
      <th>Эталон (mean ± std)</th>
      <th>Текущие (mean ± std)</th>
      <th>KS статистика</th>
      <th>p-value</th>
      <th>Статус</th>
    </tr>
  </thead>
  <tbody>{rows}</tbody>
</table>

<div class="footer">
  Отчёт сгенерирован автоматически скриптом monitoring/drift/drift_report.py
</div>
</body>
</html>"""

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Генерация отчёта о дрейфе данных")
    parser.add_argument("--reference", default="dataset/images",
                        help="Папка с эталонными изображениями")
    parser.add_argument("--current", default="monitoring/drift/current",
                        help="Папка с текущими (новыми) изображениями")
    parser.add_argument("--output", default="monitoring/drift/reports/drift_report.html",
                        help="Путь для сохранения HTML отчёта")
    args = parser.parse_args()

    generate_report(
        reference_folder=args.reference,
        current_folder=args.current,
        output_path=args.output,
    )
