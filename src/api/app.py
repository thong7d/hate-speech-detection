# src/api/app.py
"""
FastAPI application for Vietnamese Hate Speech Detection.

The REST API uses the same shared inference function as evaluation,
manual robustness tests, and the agent layer.
"""
import gc
import logging
import os
import time
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse

try:
    from .schemas import HealthResponse, PredictRequest, PredictResponse
except ImportError:
    from api.schemas import HealthResponse, PredictRequest, PredictResponse

try:
    from evaluation.classifier import ModelArtifacts, load_hf_artifacts, predict_with_artifacts
except ImportError:
    from src.evaluation.classifier import ModelArtifacts, load_hf_artifacts, predict_with_artifacts


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("vihsd.api")


HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "thong7d/vihsd-xlmr-hate-speech")
HF_TOKEN = os.environ.get("HF_TOKEN", None)
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))
BORDERLINE_LOW = float(os.environ.get("BORDERLINE_LOW", "0.35"))
BORDERLINE_HIGH = float(os.environ.get("BORDERLINE_HIGH", "0.65"))


_artifacts: ModelArtifacts | None = None


def _load_model() -> None:
    """Download and initialise the model + tokenizer from Hugging Face Hub."""
    global _artifacts

    logger.info("Loading model and tokenizer from HF Hub: %s", HF_MODEL_ID)
    _artifacts = load_hf_artifacts(
        HF_MODEL_ID,
        token=HF_TOKEN,
        device="auto",
        max_length=MAX_LENGTH,
    )

    gc.collect()
    if _artifacts.device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(
        "Model ready | device=%s | parameters=%.1fM | max_length=%d",
        _artifacts.device,
        _artifacts.model.num_parameters() / 1e6,
        MAX_LENGTH,
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once before accepting requests; clean up on shutdown."""
    _load_model()
    yield
    global _artifacts
    _artifacts = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded. Server shutting down.")


app = FastAPI(
    title="ViHSD - Vietnamese Hate Speech Detection API",
    description=(
        "REST API serving a fine-tuned XLM-RoBERTa model for detecting "
        "hate speech in Vietnamese text. Labels: CLEAN (0), OFFENSIVE (1), HATE (2)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """Return service health status."""
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return HealthResponse(
        status="ok",
        model=HF_MODEL_ID,
        device=str(_artifacts.device),
        max_length=MAX_LENGTH,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """Classify a single text for hate speech."""
    if _artifacts is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.perf_counter()

    detected_lang = request.language or "vi"
    if not request.language:
        try:
            from langdetect import detect as _detect

            detected_lang = _detect(request.text)
        except Exception:
            detected_lang = "unknown"

    result = predict_with_artifacts(
        _artifacts,
        request.text,
        borderline_low=BORDERLINE_LOW,
        borderline_high=BORDERLINE_HIGH,
        preprocess=True,
    )
    latency_ms = (time.perf_counter() - start) * 1000

    return PredictResponse(
        text=request.text,
        processed_text=result["processed_text"],
        language=detected_lang,
        label=result["label"],
        label_id=result["label_id"],
        confidence=result["confidence"],
        probabilities=result["probabilities"],
        scores=result["scores"],
        is_borderline=result["is_borderline"],
        latency_ms=round(latency_ms, 2),
    )
