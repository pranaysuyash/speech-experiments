# Model Lab Backend Dockerfile
# Multi-stage build with uv for Python dependencies

# =============================================================================
# Stage 1: Build environment with uv
# =============================================================================
FROM python:3.12-slim AS builder

# System deps for audio processing libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy lock files first for caching
COPY pyproject.toml uv.lock ./

# Sync dependencies (frozen = exact versions from lock)
RUN uv sync --frozen --no-dev

# =============================================================================
# Stage 2: Runtime environment
# =============================================================================
FROM python:3.12-slim AS runtime

# Runtime deps for audio
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy virtual environment from builder
COPY --from=builder /app/.venv /app/.venv

# Copy application code
COPY . .

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

USER appuser

# Ensure .venv binaries are in PATH
ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# Runtime configuration
ENV MODEL_CACHE_MAX_MB=4096
ENV LOG_LEVEL=INFO
ENV PORT=8000

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run production API server
CMD ["python", "scripts/deploy_api.py"]
