from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = "ok"


class ReadyResponse(BaseModel):
    ready: bool
    model_source: str | None = None
    model_version: str | None = None
    error: str | None = None


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=0, max_length=5000)
    language: str | None = Field(None, description="Optional language hint (vi, en, etc.)")


class BatchPredictRequest(BaseModel):
    texts: list[str] = Field(default_factory=list, max_length=128)


class PredictionResponse(BaseModel):
    text: str
    label: str
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    language: str | None = None


class BatchPredictionResponse(BaseModel):
    results: list[PredictionResponse]


class MetadataResponse(BaseModel):
    metadata: dict[str, Any]
