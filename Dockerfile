# Dockerfile
# ============================================================
# Multi-stage build for ViHSD Hate Speech Detection API
#
# Stage 1 (builder): install all Python dependencies into a
#   virtual environment to avoid polluting the final image.
# Stage 2 (runtime): copy only the venv + source code.
#   Final image has NO build tools — smaller and more secure.
#
# Base image: python:3.10-slim  (~130 MB)
#   - Debian Bookworm slim — well maintained, reproducible
#   - Python 3.10 matches Colab training environment
#
# Model: loaded from HuggingFace Hub at container startup.
#   NOT baked into the image (keeps image < 600 MB).
#   Model weights are cached in a Docker volume between restarts.
# ============================================================

# ---- Stage 1: dependency installer -------------------------
FROM python:3.10-slim AS builder

# Prevent interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive

# Install only the minimal OS packages required to build wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment so we can copy it cleanly to the next stage
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Upgrade pip once (speeds up subsequent installs)
RUN pip install --no-cache-dir --upgrade pip==24.0

# Copy only the requirements file first (layer-cache optimisation).
# Docker re-runs pip install only if this file changes.
COPY requirements-deploy.txt /tmp/requirements-deploy.txt

# Install CPU-only PyTorch from the official index (avoids pulling the
# full CUDA wheel, which would add ~2 GB to the layer).
RUN pip install --no-cache-dir \
        torch==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu

# Install all remaining deployment dependencies
RUN pip install --no-cache-dir -r /tmp/requirements-deploy.txt

# ---- Stage 2: runtime image --------------------------------
FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime OS packages (libgomp is required by PyTorch CPU on Debian)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the fully-built virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Set working directory — all paths in the code are relative to /app
WORKDIR /app

# Copy project source code
# Order: less-frequently-changed files first (better layer caching)
COPY setup.py            ./setup.py
COPY configs/            ./configs/
COPY src/                ./src/
COPY main.py             ./main.py

# Install the project itself in editable mode so that
# "from api.app import app" and "from data.dataset import ..." work correctly
RUN pip install --no-cache-dir -e .

# ---------------------------------------------------------------------------
# Runtime environment variables (overridable in docker-compose.yml)
# ---------------------------------------------------------------------------
# HuggingFace model source
ENV HF_MODEL_ID="thong7d/vihsd-xlmr-hate-speech"
ENV HF_TOKEN=""

# HF cache: write to /tmp/hf_cache inside the container.
# Mount the 'hf_model_cache' volume here to persist across restarts.
ENV TRANSFORMERS_CACHE="/tmp/hf_cache"
ENV HF_HOME="/tmp/hf_cache"
ENV HF_DATASETS_CACHE="/tmp/hf_datasets_cache"

# API settings
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"
ENV GRADIO_PORT="7860"
ENV MAX_LENGTH="128"
ENV BORDERLINE_LOW="0.35"
ENV BORDERLINE_HIGH="0.65"
ENV ENABLE_GRADIO="true"

# Disable tokenizer parallelism warning (single-threaded inference server)
ENV TOKENIZERS_PARALLELISM="false"

# ---------------------------------------------------------------------------
# Expose ports
# ---------------------------------------------------------------------------
EXPOSE 8000   
# FastAPI REST API
EXPOSE 7860   
# Gradio web demo

# ---------------------------------------------------------------------------
# Health check — Docker will restart the container if the API is unhealthy
# ---------------------------------------------------------------------------
HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=90s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
# Run main.py which starts FastAPI (blocking) + Gradio (daemon thread)
CMD ["python", "main.py"]
