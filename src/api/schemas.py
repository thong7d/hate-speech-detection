# src/api/schemas.py
"""
Pydantic schemas for the ViHSD FastAPI endpoints.

All field names and descriptions are in English per project convention.
These schemas serve as the public contract for /predict and /health.
"""
from typing import Dict, Optional
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for POST /predict."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=5000,
        description="Vietnamese (or other language) text to classify.",
        examples=["Bài viết rất hay, cảm ơn tác giả!"],
    )
    language: Optional[str] = Field(
        default=None,
        description=(
            "Optional ISO 639-1 language hint (e.g. 'vi', 'en'). "
            "If omitted, language is auto-detected via langdetect."
        ),
        examples=["vi"],
    )


class PredictResponse(BaseModel):
    """Response body for POST /predict."""

    text: str = Field(..., description="The original input text.")
    language: str = Field(..., description="Detected or provided language code.")
    label: str = Field(
        ..., description="Predicted toxicity label: CLEAN, OFFENSIVE, or HATE."
    )
    label_id: int = Field(..., description="Integer encoding: 0=CLEAN, 1=OFFENSIVE, 2=HATE.")
    confidence: float = Field(..., description="Softmax probability of the predicted class.")
    scores: Dict[str, float] = Field(
        ..., description="Softmax probability for each class."
    )
    is_borderline: bool = Field(
        ...,
        description=(
            "True when confidence is in [borderline_low, borderline_high]. "
            "Signals that the agent should ask a clarifying question."
        ),
    )
    latency_ms: float = Field(..., description="End-to-end inference latency in milliseconds.")


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: str = Field(..., description="Always 'ok' when the service is healthy.")
    model: str = Field(..., description="Model identifier (HF Hub ID or local path).")
    device: str = Field(..., description="Compute device used for inference (cpu or cuda).")
    max_length: int = Field(..., description="Token sequence length the model accepts.")
