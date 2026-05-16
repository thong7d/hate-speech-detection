from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import RedirectResponse

try:
    from src.api.dependencies import get_classifier, load_classifier, readiness
    from src.api.schema import (
        BatchPredictionResponse,
        BatchPredictRequest,
        HealthResponse,
        MetadataResponse,
        PredictionResponse,
        PredictRequest,
        ReadyResponse,
    )
    from src.monitoring.logging_config import configure_logging
    from src.models.classifier import HateSpeechClassifier
except ImportError:
    from api.dependencies import get_classifier, load_classifier, readiness
    from api.schema import (
        BatchPredictionResponse,
        BatchPredictRequest,
        HealthResponse,
        MetadataResponse,
        PredictionResponse,
        PredictRequest,
        ReadyResponse,
    )
    from monitoring.logging_config import configure_logging
    from models.classifier import HateSpeechClassifier


configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_classifier()
    yield


app = FastAPI(
    title="Hate Speech Detection API",
    description="Production API for hate speech detection from Hugging Face Hub or local artifacts.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/", include_in_schema=False)
def root() -> RedirectResponse:
    return RedirectResponse(url="/docs")


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.get("/ready", response_model=ReadyResponse, tags=["Monitoring"])
def ready() -> ReadyResponse:
    return ReadyResponse(**readiness())


@app.get("/metadata", response_model=MetadataResponse, tags=["Model"])
def metadata(model: HateSpeechClassifier | None = Depends(get_classifier)) -> MetadataResponse:
    if model is None:
        raise HTTPException(status_code=503, detail=readiness()["error"] or "Model not loaded.")
    return MetadataResponse(metadata=model.metadata)


@app.post("/predict", response_model=PredictionResponse, tags=["Inference"])
def predict(
    request: PredictRequest,
    model: HateSpeechClassifier | None = Depends(get_classifier),
) -> dict:
    if model is None:
        raise HTTPException(status_code=503, detail=readiness()["error"] or "Model not loaded.")
    return model.predict(request.text)


@app.post("/predict-batch", response_model=BatchPredictionResponse, tags=["Inference"])
def predict_batch(
    request: BatchPredictRequest,
    model: HateSpeechClassifier | None = Depends(get_classifier),
) -> BatchPredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail=readiness()["error"] or "Model not loaded.")
    return BatchPredictionResponse(results=model.predict_batch(request.texts))
