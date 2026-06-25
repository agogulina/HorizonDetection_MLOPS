# base image 
FROM arm64v8/python:3.11-slim

WORKDIR /app

# System deps: build-essential (компилятор для сборки колёс типа stringzilla),
# libGL/glib нужны OpenCV.
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Python dependencies 
COPY requirements.txt requirements-api.txt ./

RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir -r requirements-api.txt

# application code 
COPY . .

ENV CONFIG_PATH=configs/train.yaml \
    MODEL_PATH=checkpoints/model.keras \
    TF_CPP_MIN_LOG_LEVEL=2 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
