# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt /app/requirements.txt
RUN pip install -r requirements.txt

# Copy app
COPY app /app/app
COPY ingestion /app/ingestion

# Create non-root user
RUN useradd -ms /bin/bash appuser
USER appuser

EXPOSE 8000

# Default envs (override in runtime)
ENV MILVUS_HOST=milvus-standalone \
    MILVUS_PORT=19530 \
    MODEL_DIR=/models/all-MiniLM-L6-v2-onnx \
    EMBEDDING_DIM=384 \
    INDEX_TYPE=IVF_FLAT \
    METRIC_TYPE=L2 \
    ORT_PROVIDER=AUTO

# Entrypoint
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]