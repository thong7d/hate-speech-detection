# main.py
"""
Entry point for the ViHSD deployment server.

Architecture (single-process, zero-network-overhead):
  1. FastAPI app loads the model at startup via its lifespan handler.
  2. Gradio UI is mounted directly onto FastAPI at /ui using
     gr.mount_gradio_app(). It calls /predict internally via
     TestClient (in-process, no HTTP round-trip).
  3. Uvicorn serves everything on a single port (default: 8000).

Endpoints after startup:
  http://localhost:8000/docs  — Swagger UI (API documentation)
  http://localhost:8000/ui    — Gradio web demo
  http://localhost:8000/predict — REST API
  http://localhost:8000/health  — Health check
"""
import os
import logging

import gradio as gr
import uvicorn
from fastapi.testclient import TestClient

# Import the FastAPI app (model loads automatically via lifespan)
try:
    from api.app import app
except ImportError:
    from src.api.app import app

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vihsd.main")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# ---------------------------------------------------------------------------
# Gradio UI — uses TestClient to call FastAPI in-process (zero latency)
# ---------------------------------------------------------------------------
client = TestClient(app)

LABEL_EMOJI = {"CLEAN": "✅ CLEAN", "OFFENSIVE": "⚠️ OFFENSIVE", "HATE": "🚫 HATE"}


def classify(text: str):
    """Send text to the in-process FastAPI /predict endpoint."""
    if not text or not text.strip():
        return "Please enter some text.", "", ""
    try:
        resp = client.post("/predict", json={"text": text})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"❌ Error: {exc}", "", ""

    label_display = LABEL_EMOJI.get(data["label"], data["label"])
    conf_display  = f"{data['confidence']:.2%}"
    scores_text   = (
        f"CLEAN:     {data['scores'].get('CLEAN', 0):.4f}\n"
        f"OFFENSIVE: {data['scores'].get('OFFENSIVE', 0):.4f}\n"
        f"HATE:      {data['scores'].get('HATE', 0):.4f}\n"
        f"\nLatency: {data['latency_ms']:.1f} ms"
    )
    if data.get("is_borderline"):
        label_display += "  ← borderline (low confidence)"

    return label_display, conf_display, scores_text


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        label="Input text",
        lines=4,
        placeholder="Enter Vietnamese text here...",
    ),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Score breakdown"),
    ],
    title="🛡️ Vietnamese Hate Speech Detector",
    description=(
        "Powered by **XLM-RoBERTa** fine-tuned on the ViHSD dataset (~33K comments). "
        "Single-process FastAPI + Gradio deployment."
    ),
    examples=[
        ["Bài viết rất hay, cảm ơn tác giả!"],
        ["Mày ngu quá, biết gì mà nói chuyện"],
        ["Loại người như mày không xứng đáng sống trên đời này"],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

# Mount Gradio at /ui, API docs remain at /docs
app = gr.mount_gradio_app(app, demo, path="/ui")

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    logger.info("Starting integrated server (FastAPI + Gradio) on port %d", API_PORT)
    logger.info("  API docs : http://localhost:%d/docs", API_PORT)
    logger.info("  Gradio UI: http://localhost:%d/ui", API_PORT)
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False)