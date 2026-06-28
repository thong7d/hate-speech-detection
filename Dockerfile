FROM python:3.10-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY configs/ ./configs/
COPY src/ ./src/
COPY main.py ./main.py
COPY setup.py ./setup.py

RUN pip install --no-cache-dir -e .

ENV API_HOST=0.0.0.0 \
    API_PORT=8000 \
    MODEL_SOURCE=huggingface \
    HF_REPO_ID=thong7d/vihsd-xlmr-base-hate-speech \
    MODEL_LOCAL_PATH=artifacts/hate_speech_model/model \
    TOKENIZERS_PARALLELISM=false

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]
