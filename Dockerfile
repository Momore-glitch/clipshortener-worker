# ClipShortener Render backend — production reliability + speed build
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    CLIPSHORTENER_WORKERS=1 \
    CLIPSHORTENER_FFMPEG_THREADS=0 \
    CLIPSHORTENER_FILTER_THREADS=0 \
    CLIPSHORTENER_FFMPEG_PRESET=ultrafast \
    CLIPSHORTENER_FFMPEG_TUNE= \
    CLIPSHORTENER_X264_PARAMS= \
    CLIPSHORTENER_CRF_ORIGINAL=21 \
    CLIPSHORTENER_CRF_STANDARD=23 \
    CLIPSHORTENER_UPLOAD_CHUNK_MB=16 \
    CLIPSHORTENER_UPLOAD_FSYNC=0 \
    CLIPSHORTENER_NO_UPSCALE=1 \
    CLIPSHORTENER_PROGRESS_PERIOD=2 \
    HF_HUB_DISABLE_XET=1 \
    HF_HOME=/opt/huggingface

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg ca-certificates fonts-dejavu fonts-liberation \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN python -m pip install --no-cache-dir --prefer-binary -r requirements.txt

COPY app.py .

EXPOSE 10000
CMD ["sh", "-c", "exec python -m uvicorn app:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1"]
