# CPU-only container for DriveDocs AI (FastAPI API + web UI in one process).
FROM python:3.14-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    DEVICE=cpu \
    PORT=8000 \
    HF_HOME=/app/.hf-cache \
    HF_HUB_DISABLE_TELEMETRY=1 \
    TOKENIZERS_PARALLELISM=false \
    DRIVEDOCS_HOME=/app \
    DRIVEDOCS_STORAGE_DIR=/data/drivedocs

RUN useradd --create-home --uid 1000 appuser
WORKDIR /app

# Dependencies first for better layer caching (ONNX Runtime; no PyTorch).
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY pyproject.toml README.md ./
COPY src/ ./src/
RUN pip install --no-deps -e .

COPY configs/ ./configs/
COPY data/sample/ ./data/sample/
COPY reports/ ./reports/
COPY scripts/download_models.py scripts/build_demo_index.py ./scripts/

# Bake the models into the image so containers never download them at start-up.
# Build with --build-arg ENABLE_IMAGE_SEARCH=false for a text-only image (~0.25 GB RSS at runtime).
ARG ENABLE_IMAGE_SEARCH=true
ENV DRIVEDOCS_ENABLE_IMAGE_SEARCH=${ENABLE_IMAGE_SEARCH}
RUN python scripts/download_models.py \
    && HF_HUB_OFFLINE=1 python scripts/build_demo_index.py \
    && mkdir -p /data/drivedocs \
    && chown -R appuser:appuser /data /app/.hf-cache
ENV HF_HUB_OFFLINE=1

USER appuser
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import os, urllib.request; urllib.request.urlopen('http://127.0.0.1:%s/health' % os.environ.get('PORT', '8000'), timeout=4)" || exit 1

# Single worker: each worker would load its own copy of the models. PORT is provided by Render (default 10000).
CMD ["sh", "-c", "exec uvicorn drivedocs.api:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 30"]
