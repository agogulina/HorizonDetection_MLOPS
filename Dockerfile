# ── Stage 1: base image ──────────────────────────────────────────────────────
# Official TensorFlow CPU image keeps us from fighting CUDA/cuDNN installs.
# Pin to a specific digest in production for reproducibility.
# FROM tensorflow/tensorflow:2.15.0 AS base
FROM arm64v8/python:3.11-slim

WORKDIR /app

# System deps (OpenCV needs libGL)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# ── Stage 2: Python dependencies ─────────────────────────────────────────────
# Copy only requirement files first so Docker can cache this layer.
COPY requirements.txt requirements-api.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-api.txt

# ── Stage 3: application code ────────────────────────────────────────────────
COPY . .

# The model checkpoint is NOT baked into the image.
# Mount it at runtime via:
#   docker run  -v ./checkpoints:/app/checkpoints ...
#   k8s:        PersistentVolumeClaim or init-container download from S3
#
# Default paths (overridable via env vars):
ENV CONFIG_PATH=configs/train.yaml \
    MODEL_PATH=checkpoints/model.keras \
    # Tell TF to be quiet; remove TF_CPP_MIN_LOG_LEVEL=3 if you want full logs
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Use exec form so that SIGTERM from Docker/k8s reaches uvicorn directly
CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
