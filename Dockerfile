# Dockerfile
# ============================================================
# Multi-stage build for ViHSD Hate Speech Detection API.
#
# Stage 1 (builder): install Python dependencies into a venv.
# Stage 2 (runtime): copy venv + source code. No build tools.
#
# Model is loaded from HuggingFace Hub at startup (NOT baked in).
# Model cache is persisted via a Docker volume.
# ============================================================

# ---- Stage 1: dependency installer -------------------------
FROM python:3.10-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

RUN pip install --no-cache-dir --upgrade pip==24.0

COPY requirements-deploy.txt /tmp/requirements-deploy.txt

# CPU-only PyTorch (avoids ~2 GB CUDA wheel)
RUN pip install --no-cache-dir \
        torch==2.2.2 \
        --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r /tmp/requirements-deploy.txt

# ---- Stage 2: runtime image --------------------------------
FROM python:3.10-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

COPY setup.py            ./setup.py
COPY configs/            ./configs/
COPY src/                ./src/
COPY main.py             ./main.py

RUN pip install --no-cache-dir -e .

# --- Runtime environment variables ---
ENV HF_MODEL_ID="thong7d/vihsd-xlmr-hate-speech"
ENV HF_TOKEN=""
ENV TRANSFORMERS_CACHE="/tmp/hf_cache"
ENV HF_HOME="/tmp/hf_cache"
ENV API_HOST="0.0.0.0"
ENV API_PORT="8000"
ENV MAX_LENGTH="128"
ENV BORDERLINE_LOW="0.35"
ENV BORDERLINE_HIGH="0.65"
ENV TOKENIZERS_PARALLELISM="false"

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=10s \
    --start-period=90s \
    --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "main.py"]
