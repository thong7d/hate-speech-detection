# main.py
"""
Entry point for the ViHSD deployment server.

This file is placed at the project root so that Docker's WORKDIR (/app)
resolves all src/ imports correctly via the installed package (pip install -e .).

Starts two services in the same process using threading:
  1. FastAPI  (port 8000) — REST API for programmatic access
  2. Gradio   (port 7860) — Web demo for interactive presentations

Both ports are exposed in docker-compose.yml.
"""
import os
import sys
import threading
import time
import logging

import uvicorn

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("vihsd.main")

# ---------------------------------------------------------------------------
# Configuration — read from environment variables set in docker-compose.yml
# ---------------------------------------------------------------------------
API_HOST       = os.environ.get("API_HOST",    "0.0.0.0")
API_PORT       = int(os.environ.get("API_PORT", "8000"))
GRADIO_PORT    = int(os.environ.get("GRADIO_PORT", "7860"))
HF_MODEL_ID    = os.environ.get("HF_MODEL_ID", "thong7d/vihsd-xlmr-hate-speech")
MAX_LENGTH     = int(os.environ.get("MAX_LENGTH", "128"))
ENABLE_GRADIO  = os.environ.get("ENABLE_GRADIO", "true").lower() == "true"


# ---------------------------------------------------------------------------
# FastAPI server (runs in the main thread via uvicorn)
# ---------------------------------------------------------------------------
def run_fastapi() -> None:
    """Start the uvicorn ASGI server for the FastAPI application."""
    logger.info("Starting FastAPI on %s:%d", API_HOST, API_PORT)
    uvicorn.run(
        "api.app:app",          # module path: src/api/app.py → app object
        host=API_HOST,
        port=API_PORT,
        log_level="info",
        access_log=True,
        reload=False,           # Reload must be False inside Docker
    )


# ---------------------------------------------------------------------------
# Gradio demo (runs in a daemon thread so it dies when FastAPI exits)
# ---------------------------------------------------------------------------
def run_gradio() -> None:
    """
    Start a Gradio web interface that calls the local FastAPI endpoint.

    Design: Gradio sends HTTP requests to FastAPI's /predict endpoint
    rather than loading the model a second time. This avoids running
    two copies of the 278M-parameter model simultaneously and prevents OOM.

    Gradio is started 15 seconds after FastAPI to allow model warm-up.
    """
    logger.info("Gradio: waiting 15 seconds for FastAPI warm-up...")
    time.sleep(15)

    try:
        import requests
        import gradio as gr

        FASTAPI_URL = f"http://localhost:{API_PORT}/predict"
        LABEL_EMOJI = {"CLEAN": "✅ CLEAN", "OFFENSIVE": "⚠️ OFFENSIVE", "HATE": "🚫 HATE"}

        def classify(text: str):
            """Send text to FastAPI /predict and format the response."""
            if not text or not text.strip():
                return "Please enter some text.", "", ""
            try:
                resp = requests.post(
                    FASTAPI_URL,
                    json={"text": text},
                    timeout=30,
                )
                resp.raise_for_status()
                data = resp.json()
            except requests.exceptions.ConnectionError:
                return "❌ FastAPI not ready. Wait a moment and retry.", "", ""
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
                label_display += "  ⟵ borderline (low confidence)"

            return label_display, conf_display, scores_text

        demo = gr.Interface(
            fn=classify,
            inputs=gr.Textbox(
                label="Input text",
                placeholder="Nhập văn bản tiếng Việt tại đây…",
                lines=4,
            ),
            outputs=[
                gr.Textbox(label="Prediction"),
                gr.Textbox(label="Confidence"),
                gr.Textbox(label="Score breakdown"),
            ],
            title="🛡️ Vietnamese Hate Speech Detector",
            description=(
                "Powered by **XLM-RoBERTa** fine-tuned on the ViHSD dataset (~33 K comments).<br>"
                "Labels: **CLEAN** · **OFFENSIVE** · **HATE**<br>"
                "Model: `thong7d/vihsd-xlmr-hate-speech` on HuggingFace Hub."
            ),
            examples=[
                ["Bài viết rất hay, cảm ơn tác giả!"],
                ["Mày ngu quá, biết gì mà nói chuyện"],
                ["Loại người như mày không xứng đáng sống trên đời này"],
            ],
            theme=gr.themes.Soft(),
            allow_flagging="never",
        )

        logger.info("Starting Gradio demo on port %d", GRADIO_PORT)
        demo.launch(
            server_name="0.0.0.0",
            server_port=GRADIO_PORT,
            share=False,          # No ngrok tunnel; Docker maps the port
            show_error=True,
        )

    except Exception as exc:
        logger.error("Gradio failed to start: %s", exc)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    if ENABLE_GRADIO:
        gradio_thread = threading.Thread(target=run_gradio, daemon=True, name="gradio")
        gradio_thread.start()
        logger.info("Gradio thread started (daemon).")

    # FastAPI blocks the main thread — Gradio thread dies when this exits
    run_fastapi()
