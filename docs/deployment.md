# System Deployment Document

## 1. Technical Stack
The system is built on a fully open-source, offline-ready Python stack:
- **Core Web Framework**: FastAPI (high-performance async REST endpoints).
- **Offline GUI**: Streamlit (local operations panel) & Gradio (interactive demo).
- **Core ML Model**: standard XLM-RoBERTa Base (1.1GB weight file loaded via PyTorch).
- **Agent Backend**: Ollama serving `qwen2.5:7b-instruct` on local ports or Ngrok tunnels.
- **Containerization**: Docker & Docker Compose.

---

## 2. API Architecture
The REST API is implemented in `src/api/` and runs on port `8000`.

### Key Endpoints:
- `GET /health`: Checks system and device status (e.g. CUDA vs CPU).
- `GET /metadata`: Returns model properties, versions, and classification thresholds.
- `POST /predict`: Performs real-time inference on a single text string.
- `POST /predict-batch`: Processes a batch list of texts asynchronously.
- `POST /predict-hybrid`: Integrates the baseline model and Ollama agent routing.

---

## 3. Docker Containerization
We provide container definitions to ensure reproducibility:
- **`Dockerfile`**: Sets up Python 3.10-slim, installs dependencies via `requirements.api.txt`, downloads model weights, and exposes port `8000`.
- **`docker-compose.yml`**: Configures environment variables, maps the logging and results directories, and starts the FastAPI server using `uvicorn`.

---

## 4. Local GUI Applications
We expose two interactive offline dashboards:
1. **Streamlit GUI (`streamlit_app.py`)**: Runs on port `8501`. Offers single verification, thread-safe background batch processing (preventing UI freeze), and real-time model statistics tracking.
2. **Gradio GUI (`app.py`)**: Runs on port `7860`. Offers simple validation tabs for text input and batch CSV files.

---

## 5. Deployment Challenges & Mitigations

### A. High Latency of Local LLM
- **Challenge**: Invoking a 7B parameter LLM sequentially for every single row in a batch causes latency spikes (e.g., several minutes for a CSV of 1,000 items).
- **Mitigation**:
  - We use a **threshold cascade**. Only comments with confidence $< 0.65$ trigger the LLM.
  - Borderline cases in batch files are grouped together and sent as a single JSON array payload, reducing network request round-trips from $N$ to 1.

### B. GPU/RAM Resource Constraints
- **Challenge**: Running a 1.1GB XLM-R model alongside a 7B LLM requires significant VRAM.
- **Mitigation**:
  - The XLM-R model uses `device="auto"`, utilizing CUDA if available, but falling back cleanly to CPU.
  - The LLM can be offloaded to an external local/colab GPU server served via an Ollama endpoint, isolating the resource demands.
