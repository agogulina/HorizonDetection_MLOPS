import os
import logging
import yaml
import tensorflow as tf

from fastapi import FastAPI
from fastapi.responses import Response, FileResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.routers import predict, drift, retrain

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_on_startup() -> tf.keras.Model:
    model_path = os.getenv("MODEL_PATH", "checkpoints/model.keras")
    if not os.path.exists(model_path):
        logger.warning(f"Checkpoint not found at '{model_path}'.")
        return None
    logger.info(f"Loading model from '{model_path}' ...")
    model = tf.keras.models.load_model(model_path, compile=False)
    logger.info("Model loaded successfully.")
    return model


def load_config() -> dict:
    config_path = os.getenv("CONFIG_PATH", "configs/train.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cfg = load_config()
    app.state.model = load_model_on_startup()
    yield


app = FastAPI(
    title="Horizon Detection API",
    description=(
        "U-Net based horizon line detection for UAV imagery.\n\n"
        "Upload a JPEG or PNG frame from a drone camera and receive "
        "a segmentation mask (sky / land) together with roll and pitch estimates."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(predict.router, prefix="/api/v1", tags=["inference"])
app.include_router(drift.router,   prefix="/api/v1", tags=["monitoring"])
app.include_router(retrain.router, prefix="/api/v1", tags=["training"])

# ── Статические файлы (веб UI) ────────────────────────────────────────────────
_static_dir = os.path.join(os.path.dirname(__file__), "static")
if os.path.isdir(_static_dir):
    app.mount("/ui", StaticFiles(directory=_static_dir, html=True), name="static")


@app.get("/", include_in_schema=False)
def root():
    index = os.path.join(_static_dir, "index.html")
    if os.path.exists(index):
        return FileResponse(index)
    return {"message": "Horizon Detection API", "docs": "/docs"}


@app.get("/health", summary="Liveness probe", tags=["ops"])
def health():
    return {"status": "ok"}


@app.get("/ready", summary="Readiness probe", tags=["ops"])
def ready():
    if app.state.model is None:
        return Response(status_code=503, content="model not loaded")
    return {"status": "ready"}


@app.get("/metrics", summary="Prometheus metrics", tags=["ops"], include_in_schema=False)
def metrics():
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
