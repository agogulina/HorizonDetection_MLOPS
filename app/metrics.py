"""
Prometheus metrics для FastAPI сервиса.
Все счётчики и гистограммы регистрируются один раз при импорте.
"""

from prometheus_client import Counter, Histogram, Gauge, CollectorRegistry, REGISTRY

# ── Счётчики запросов ────────────────────────────────────────────────────────

REQUESTS_TOTAL = Counter(
    "horizon_requests_total",
    "Общее количество запросов к API",
    ["endpoint", "status"],  # labels: /predict или /predict/meta, 200/400/503
)

# ── Время обработки ──────────────────────────────────────────────────────────

REQUEST_DURATION = Histogram(
    "horizon_request_duration_seconds",
    "Время обработки одного запроса (секунды)",
    ["endpoint"],
    buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
)

# ── Метрики предсказаний ─────────────────────────────────────────────────────

ROLL_GAUGE = Gauge(
    "horizon_roll_degrees",
    "Последнее предсказанное значение крена (roll) в градусах",
)

PITCH_GAUGE = Gauge(
    "horizon_pitch_degrees",
    "Последнее предсказанное значение тангажа (pitch) в градусах",
)

SKY_RATIO_HISTOGRAM = Histogram(
    "horizon_sky_ratio",
    "Распределение доли неба в кадре",
    buckets=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
)

HORIZON_DETECTED_TOTAL = Counter(
    "horizon_detected_total",
    "Количество кадров где горизонт был / не был найден",
    ["detected"],  # "true" / "false"
)

# ── Размер входного изображения ──────────────────────────────────────────────

IMAGE_SIZE_BYTES = Histogram(
    "horizon_image_size_bytes",
    "Размер загруженного изображения в байтах",
    buckets=[10_000, 50_000, 100_000, 500_000, 1_000_000, 5_000_000],
)
