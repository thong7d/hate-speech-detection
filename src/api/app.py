# src/api/app.py
"""
FastAPI application for Vietnamese Hate Speech Detection.

Loads the fine-tuned XLM-RoBERTa model from HuggingFace Hub at startup
and serves predictions via a REST API.

Endpoints:
    POST /predict  — Classify a single text sample.
    GET  /health   — Service health check.
    GET  /         — Redirect to /docs.
    GET  /docs     — Interactive OpenAPI documentation (auto-generated).

Environment variables (set in docker-compose.yml or .env):
    HF_MODEL_ID   — HuggingFace Hub model repo, e.g. "thong7d/vihsd-xlmr-hate-speech"
    HF_TOKEN      — (Optional) HuggingFace read token for private repos.
    MAX_LENGTH    — Token sequence length (default: 128).
    BORDERLINE_LOW  — Lower confidence threshold for borderline flag (default: 0.35).
    BORDERLINE_HIGH — Upper confidence threshold for borderline flag (default: 0.65).
"""
import gc
import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from api.schemas import HealthResponse, PredictRequest, PredictResponse

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("vihsd.api")

# ---------------------------------------------------------------------------
# Configuration (sourced entirely from environment variables)
# ---------------------------------------------------------------------------
HF_MODEL_ID     = os.environ.get("HF_MODEL_ID", "thong7d/vihsd-xlmr-hate-speech")
HF_TOKEN        = os.environ.get("HF_TOKEN", None)          # None → public repo
MAX_LENGTH      = int(os.environ.get("MAX_LENGTH", "128"))
BORDERLINE_LOW  = float(os.environ.get("BORDERLINE_LOW",  "0.35"))
BORDERLINE_HIGH = float(os.environ.get("BORDERLINE_HIGH", "0.65"))

LABEL_MAP: dict[int, str] = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}

# ---------------------------------------------------------------------------
# Global model state — lazy-loaded once at startup via lifespan
# ---------------------------------------------------------------------------
_model:     Optional[AutoModelForSequenceClassification] = None
_tokenizer: Optional[AutoTokenizer] = None
_device:    Optional[torch.device]  = None


def _load_model() -> None:
    """
    Download and initialise the model + tokenizer from HuggingFace Hub.
    Caches to the directory pointed to by TRANSFORMERS_CACHE env var
    (set to /tmp/hf_cache in docker-compose.yml to avoid layer bloat).
    """
    global _model, _tokenizer, _device

    logger.info("Loading tokenizer from HF Hub: %s", HF_MODEL_ID)
    _tokenizer = AutoTokenizer.from_pretrained(
        HF_MODEL_ID, token=HF_TOKEN, trust_remote_code=False
    )

    logger.info("Loading model from HF Hub: %s", HF_MODEL_ID)
    _model = AutoModelForSequenceClassification.from_pretrained(
        HF_MODEL_ID, token=HF_TOKEN, trust_remote_code=False
    )

    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = _model.to(_device)
    _model.eval()

    # Release any leftover allocations from the download phase
    gc.collect()
    if _device.type == "cuda":
        torch.cuda.empty_cache()

    logger.info(
        "Model ready | device=%s | parameters=%.1fM | max_length=%d",
        _device,
        _model.num_parameters() / 1e6,
        MAX_LENGTH,
    )


# ---------------------------------------------------------------------------
# Lifespan: replaces deprecated @app.on_event("startup")
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model once before accepting requests; clean up on shutdown."""
    _load_model()
    yield
    # Shutdown: free GPU memory
    global _model, _tokenizer
    del _model, _tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("Model unloaded. Server shutting down.")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------
app = FastAPI(
    title="ViHSD — Vietnamese Hate Speech Detection API",
    description=(
        "REST API serving a fine-tuned XLM-RoBERTa model for detecting "
        "hate speech in Vietnamese text. "
        "Labels: **CLEAN** (0) · **OFFENSIVE** (1) · **HATE** (2)."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    """Redirect root URL to the interactive API docs."""
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health_check() -> HealthResponse:
    """
    Return service health status.

    Confirms that the model is loaded and the device is available.
    Used by Docker HEALTHCHECK and load balancers.
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return HealthResponse(
        status="ok",
        model=HF_MODEL_ID,
        device=str(_device),
        max_length=MAX_LENGTH,
    )


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(request: PredictRequest) -> PredictResponse:
    """
    Classify a single text for hate speech.

    **Input:** JSON body `{"text": "...", "language": "vi" | null}`

    **Output:**
    - `label`        — CLEAN / OFFENSIVE / HATE
    - `confidence`   — Softmax probability of the predicted class
    - `scores`       — Full probability distribution over all 3 classes
    - `is_borderline`— True when confidence is in [0.35, 0.65]; triggers agent clarification
    - `latency_ms`   — End-to-end wall-clock time in milliseconds
    """
    if _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")

    start = time.perf_counter()

    # --- Language detection (best-effort; failures are non-fatal) ---
    detected_lang = request.language or "vi"
    if not request.language:
        try:
            from langdetect import detect as _detect
            detected_lang = _detect(request.text)
        except Exception:
            detected_lang = "unknown"

    # --- Tokenization ---
    encoding = _tokenizer(
        request.text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    input_ids      = encoding["input_ids"].to(_device)
    attention_mask = encoding["attention_mask"].to(_device)

    # --- Inference (no gradient computation needed) ---
    with torch.no_grad():
        logits = _model(input_ids=input_ids, attention_mask=attention_mask).logits
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()[0]   # shape: (3,)

    pred_id    = int(probs.argmax())
    confidence = float(probs.max())
    scores     = {LABEL_MAP[i]: round(float(probs[i]), 4) for i in range(3)}
    latency_ms = (time.perf_counter() - start) * 1000

    logger.debug(
        "predict | lang=%s label=%s conf=%.3f latency=%.1fms",
        detected_lang, LABEL_MAP[pred_id], confidence, latency_ms,
    )

    return PredictResponse(
        text=request.text,
        language=detected_lang,
        label=LABEL_MAP[pred_id],
        label_id=pred_id,
        confidence=round(confidence, 4),
        scores=scores,
        is_borderline=(BORDERLINE_LOW <= confidence <= BORDERLINE_HIGH),
        latency_ms=round(latency_ms, 2),
    )
