# ViHSD - Vietnamese Hate Speech Detection

This project provides a complete pipeline for detecting hate speech in Vietnamese text using fine-tuned **XLM-RoBERTa** models. It features a FastAPI backend, a Gradio web interface, and an intelligent moderation agent powered by **Google Gemini**.

## Project Overview

- **Core Technologies:** Python, PyTorch, Transformers (XLM-RoBERTa), FastAPI, Gradio, Pydantic, Google Gemini API.
- **Key Features:**
  - High-performance inference via FastAPI.
  - Interactive web UI via Gradio.
  - Multi-step moderation reasoning using Gemini with tool usage (classification, language detection, logging).
  - Fine-tuning and evaluation utilities for the ViHSD dataset.

## Architecture

- **`src/api/`**: FastAPI implementation.
  - `app.py`: Main API application with lifespan management and inference logic.
  - `schemas.py`: Pydantic models for request/response validation.
- **`src/agent/`**: Gemini-powered moderation agent.
  - `moderator.py`: `ContentModerator` and `ModerationTools` for advanced moderation workflows.
- **`src/models/`**: Model definitions and training utilities.
  - `classifier.py`: Metrics computation for fine-tuning.
- **`src/data/`**: Data loading and processing.
  - `dataset.py`: PyTorch `Dataset` implementation for ViHSD.
- **`main.py`**: Integrated entry point serving both FastAPI and Gradio UI on a single port.

## Building and Running

### Prerequisites
- Python 3.9+
- GPU (CUDA) recommended for inference, but CPU is supported.

### Installation
```bash
pip install -r requirements.txt
pip install -e .
```

### Running the Integrated Server (FastAPI + Gradio)
```bash
python main.py
```
- **Swagger UI:** [http://localhost:8000/docs](http://localhost:8000/docs)
- **Gradio UI:** [http://localhost:8000/ui](http://localhost:8000/ui)

### Environment Variables
| Variable | Description | Default |
| :--- | :--- | :--- |
| `HF_MODEL_ID` | HuggingFace Hub model ID | `thong7d/vihsd-xlmr-hate-speech` |
| `API_PORT` | Port for the integrated server | `8000` |
| `API_HOST` | Host for the integrated server | `0.0.0.0` |
| `BORDERLINE_LOW` | Low confidence threshold for agent intervention | `0.35` |
| `BORDERLINE_HIGH` | High confidence threshold for agent intervention | `0.65` |

## Development Conventions

- **Language:** All source code comments and documentation are in English, except for user-facing strings in the UI/Agent responses which may be in Vietnamese.
- **Typing:** Use Python type hints (Pydantic/PEP 484) throughout the codebase.
- **Logging:** Use the standard `logging` module. APIs should include latency monitoring.
- **Inference:** Prefer `torch.no_grad()` and explicit device placement (`cuda` if available).
- **Safety:** The moderation agent uses SHA-256 hashing for logging user text to ensure privacy.

## Testing and Evaluation
- Use `src/evaluation/evaluate.py` for computing metrics (F1-Macro, Confusion Matrix).
- Evaluation results should be saved atomically to avoid corruption on shared filesystems.
