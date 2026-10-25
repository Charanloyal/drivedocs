# Multi-stage / lightweight CPU container for DriveDocs
FROM python:3.11-slim

# Prevent Python from writing .pyc files & enable unbuffered stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEVICE=cpu \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1

# Install system dependencies (build essentials, curl for health check)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN useradd -m -u 1000 appuser

WORKDIR /app

# Install Python dependencies
COPY requirements.txt pyproject.toml ./
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy source code and configurations
COPY configs/ ./configs/
COPY data/sample/ ./data/sample/
COPY src/ ./src/
COPY README.md ./

# Install project package in editable mode
RUN pip install --no-cache-dir -e .

# Create storage directory structure and assign ownership to appuser
RUN mkdir -p data/storage/images data/storage/indexes data/storage/uploads && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "drivedocs.api:app", "--host", "0.0.0.0", "--port", "8000"]
