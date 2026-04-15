import os
import time
import torch
import gc
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# --- Configuration ---
# These can be overridden by environment variables
HF_MODEL_ID = os.environ.get("HF_MODEL_ID", "")
LOCAL_MODEL_DIR = os.environ.get("LOCAL_MODEL_DIR", "")
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "128"))

LABEL_MAP = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}

# --- Pydantic schemas ---
class PredictRequest(BaseModel):
    """Request schema for /predict endpoint."""
    text: str = Field(..., min_length=1, max_length=5000,
                      description="Text to classify")
    language: Optional[str] = Field(None, description="Language hint (vi/en)")

class PredictResponse(BaseModel):
    """Response schema for /predict endpoint."""
    text: str
    language: str
    label: str
    label_id: int
    confidence: float
    scores: dict
    is_borderline: bool
    latency_ms: float

class HealthResponse(BaseModel):
    """Response schema for /health endpoint."""
    status: str
    model: str
    device: str

# --- App initialization ---
app = FastAPI(
    title="ViHSD — Vietnamese Hate Speech Detection API",
    version="1.0.0",
    description="REST API for detecting hate speech in Vietnamese text.",
)

# Global model/tokenizer (loaded once at startup)
_model = None
_tokenizer = None
_device = None


def get_model():
    """Lazy-load model on first request. Supports HF Hub and local directory."""
    global _model, _tokenizer, _device

    if _model is not None:
        return _model, _tokenizer, _device

    source = HF_MODEL_ID if HF_MODEL_ID else LOCAL_MODEL_DIR
    if not source:
        raise RuntimeError(
            "No model source configured. Set HF_MODEL_ID or LOCAL_MODEL_DIR."
        )

    print(f"Loading model from: {source}")
    _tokenizer = AutoTokenizer.from_pretrained(source)
    _model = AutoModelForSequenceClassification.from_pretrained(source)
    _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _model = _model.to(_device)
    _model.eval()

    # Cleanup after loading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"✅ Model loaded on {_device}")
    return _model, _tokenizer, _device


@app.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    model, _, device = get_model()
    return HealthResponse(
        status="ok",
        model=HF_MODEL_ID or LOCAL_MODEL_DIR,
        device=str(device),
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Classify a single text for hate speech.

    Returns label (CLEAN/OFFENSIVE/HATE), confidence score,
    per-class probabilities, and borderline flag.
    """
    model, tokenizer, device = get_model()

    start_time = time.time()

    # Detect language if not provided
    detected_lang = request.language or "vi"
    if not request.language:
        try:
            from langdetect import detect
            detected_lang = detect(request.text)
        except Exception:
            detected_lang = "unknown"

    # Tokenize
    encoding = tokenizer(
        request.text,
        max_length=MAX_LENGTH,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )

    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    # Inference
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        probs = torch.softmax(outputs.logits, dim=-1).cpu().numpy()[0]

    pred_id = int(probs.argmax())
    confidence = float(probs.max())
    scores = {LABEL_MAP[i]: round(float(probs[i]), 4) for i in range(3)}
    is_borderline = 0.35 <= confidence <= 0.65

    latency_ms = (time.time() - start_time) * 1000

    return PredictResponse(
        text=request.text,
        language=detected_lang,
        label=LABEL_MAP[pred_id],
        label_id=pred_id,
        confidence=round(confidence, 4),
        scores=scores,
        is_borderline=is_borderline,
        latency_ms=round(latency_ms, 2),
    )