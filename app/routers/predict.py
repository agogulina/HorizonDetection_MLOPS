"""
Inference endpoints с Prometheus метриками.
"""

import io
import time
import logging
from typing import Tuple

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException, Request, UploadFile, File
from fastapi.responses import StreamingResponse

from app.schemas import PredictMetaResponse
from app.geometry import estimate_horizon
from app.metrics import (
    REQUESTS_TOTAL,
    REQUEST_DURATION,
    ROLL_GAUGE,
    PITCH_GAUGE,
    SKY_RATIO_HISTOGRAM,
    HORIZON_DETECTED_TOTAL,
    IMAGE_SIZE_BYTES,
)
from src.inference.predict import postprocess_prediction, visualize_prediction

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_CONTENT_TYPES = {"image/jpeg", "image/png", "image/jpg"}


def _decode_upload(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="Cannot decode image. Send a valid JPEG or PNG.")
    return img


def _preprocess(img_bgr: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, image_size)
    img_norm = cv2.normalize(
        img_resized.astype(np.float32), None, 0, 1, cv2.NORM_MINMAX, cv2.CV_32F
    )
    return np.expand_dims(img_norm, axis=0)


def _run_inference(request: Request, img_bgr: np.ndarray) -> np.ndarray:
    model = request.app.state.model
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    cfg = request.app.state.cfg
    image_size = tuple(cfg["data"]["image_size"])
    tensor = _preprocess(img_bgr, image_size)
    pred = model.predict(tensor, verbose=0)
    mask = postprocess_prediction(pred, threshold=cfg["inference"]["threshold"])
    return mask


@router.post("/predict", summary="Segmentation mask as PNG", response_class=StreamingResponse)
async def predict_mask(
    request: Request,
    file: UploadFile = File(..., description="JPEG or PNG image from UAV camera"),
):
    """Запустить U-Net сегментацию и вернуть PNG с оверлеем маски."""
    start = time.time()
    endpoint = "/predict"

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="400").inc()
        raise HTTPException(status_code=400, detail=f"Unsupported content type '{file.content_type}'.")

    raw = await file.read()
    IMAGE_SIZE_BYTES.observe(len(raw))
    img_bgr = _decode_upload(raw)

    try:
        mask = _run_inference(request, img_bgr)
    except HTTPException as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=str(e.status_code)).inc()
        raise

    cfg = request.app.state.cfg
    image_size = tuple(cfg["data"]["image_size"])
    img_rgb = cv2.cvtColor(cv2.resize(img_bgr, image_size), cv2.COLOR_BGR2RGB)
    viz_rgb = visualize_prediction(img_rgb, mask)
    viz_bgr = cv2.cvtColor(viz_rgb, cv2.COLOR_RGB2BGR)

    _, buffer = cv2.imencode(".png", viz_bgr)

    duration = time.time() - start
    REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
    REQUESTS_TOTAL.labels(endpoint=endpoint, status="200").inc()

    return StreamingResponse(
        io.BytesIO(buffer.tobytes()),
        media_type="image/png",
        headers={"Content-Disposition": f'inline; filename="result_{file.filename}"'},
    )


@router.post("/predict/meta", summary="Segmentation metadata + roll/pitch", response_model=PredictMetaResponse)
async def predict_meta(
    request: Request,
    file: UploadFile = File(..., description="JPEG or PNG image from UAV camera"),
):
    """Запустить U-Net сегментацию и вернуть JSON с roll, pitch, sky/land ratio."""
    start = time.time()
    endpoint = "/predict/meta"

    if file.content_type not in ALLOWED_CONTENT_TYPES:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status="400").inc()
        raise HTTPException(status_code=400, detail=f"Unsupported content type '{file.content_type}'.")

    raw = await file.read()
    IMAGE_SIZE_BYTES.observe(len(raw))
    img_bgr = _decode_upload(raw)

    try:
        mask = _run_inference(request, img_bgr)
    except HTTPException as e:
        REQUESTS_TOTAL.labels(endpoint=endpoint, status=str(e.status_code)).inc()
        raise

    total_pixels = mask.shape[0] * mask.shape[1]
    sky_ratio = float(mask[:, :, 0].sum()) / total_pixels
    land_ratio = float(mask[:, :, 1].sum()) / total_pixels

    horizon_detected, roll_deg, pitch_deg = estimate_horizon(mask)

    # Записываем метрики
    SKY_RATIO_HISTOGRAM.observe(sky_ratio)
    HORIZON_DETECTED_TOTAL.labels(detected=str(horizon_detected).lower()).inc()
    if roll_deg is not None:
        ROLL_GAUGE.set(roll_deg)
    if pitch_deg is not None:
        PITCH_GAUGE.set(pitch_deg)

    duration = time.time() - start
    REQUEST_DURATION.labels(endpoint=endpoint).observe(duration)
    REQUESTS_TOTAL.labels(endpoint=endpoint, status="200").inc()

    return PredictMetaResponse(
        filename=file.filename or "unknown",
        mask_shape=list(mask.shape),
        horizon_detected=horizon_detected,
        roll_deg=roll_deg,
        pitch_deg=pitch_deg,
        sky_ratio=round(sky_ratio, 4),
        land_ratio=round(land_ratio, 4),
    )
