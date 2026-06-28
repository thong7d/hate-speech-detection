# ViHSD: Vietnamese Hate Speech & Toxicity Detection System

An end-to-end, production-ready Vietnamese hate speech and toxicity detection system built on a fine-tuned **XLM-RoBERTa Base** backbone and integrated with a local **Ollama Qwen2.5 Agent** for advanced reasoning on borderline cases.

This repository serves as a university final project submission for NLP and MLOps.

---

## 📖 Project Documentation Index
All project documentation is available in the `docs/` folder:

1. **[Business Problem Definition](docs/problem_definition.md)**: Defines business context, stakeholders, system scope, and success metrics.
2. **[Data Description Document](docs/data_description.md)**: Details dataset splits, data schema, preprocessing steps, and class imbalances.
3. **[Agentic AI Architecture](docs/agent_architecture.md)**: Outlines the multi-step reasoning flow, tool configurations, system prompts, and contains a Mermaid architecture diagram.
4. **[Model Evaluation and Analysis](docs/model_evaluation.md)**: Compares the TF-IDF+LR baseline against the XLM-R classifier, includes the classification report, confusion matrix, and system trade-offs.
5. **[System Deployment Document](docs/deployment.md)**: Details FastAPI endpoints, local GUI serving (Streamlit + Gradio), and Docker container configuration.
6. **[Continual Learning & Monitoring Strategy](docs/continual_learning.md)**: Describes the sequential CL pipeline over VLSP-2019, rehearsal buffer strategy, Gatekeeper validation, and drift monitoring metrics.
7. **[Data Privacy & Model Robustness](docs/privacy_analysis.md)**: Analyzes PII handling, cryptographic logging (SHA-256), and adversarial attack robustness.
8. **[Ethics & Responsible AI Statement](docs/ethics_statement.md)**: Evaluates algorithmic fairness, explainability, regional biases, and mitigation of potential misuses.
9. **[Project Plan and Management](docs/project_plan.md)**: Documents project schedule, timeline milestones, and reflections on team scaling.

---

## 🛠️ System Architecture

```text
Incoming Vietnamese Text
  │
  ▼
XLM-RoBERTa Base Classifier ─────► [Confidence >= 0.65] ──► Output (Fast Classify)
  │
  ▼ [Confidence < 0.65] (Grey Area)
ContentModerator Agent
  ├─► Tool 1: detect_language (langdetect)
  ├─► Tool 2: classify_text (XLM-R probabilities)
  ├─► LLM Reasoning: Ollama Qwen2.5-7B (local or Ngrok Tunnel)
  └─► Tool 3: log_event (Privacy-safe SHA-256 logging)
  │
  ▼
Final Decision Label & Vietnamese Explanation
```

---

## 🚀 Getting Started

### 1. Clone and Set Up Environment
```bash
git clone https://github.com/<your-username>/hate-speech-detection.git
cd hate-speech-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate      # Linux/Mac
.venv\Scripts\activate          # Windows
```

### 2. Install Dependencies

The project uses **separate requirements files** for each use case:

| File | Purpose | Install command |
|---|---|---|
| `requirements.txt` | Core API/inference + Docker base | `pip install -r requirements.txt` |
| `requirements-local.txt` | Local GUI (Streamlit + Gradio + Ollama agent) | `pip install -r requirements-local.txt` |
| `requirements-train.txt` | Model training on Google Colab (GPU) | `pip install -r requirements-train.txt` |
| `requirements-dev.txt` | Full development (extends local + pytest, ruff, captum) | `pip install -r requirements-dev.txt` |

**For local serving (most common):**
```bash
pip install -r requirements-local.txt
```

### 3. Download the Pre-trained Model
The fine-tuned model weights are hosted on HuggingFace and are **not committed** to this repository.
Run the download script to fetch the model:
```bash
python src/data/download.py
```
This will download and cache the model from `thong7d/vihsd-xlmr-base-hate-speech`.

> Alternatively, the model is loaded automatically from HuggingFace at runtime when `MODEL_SOURCE=huggingface` (default).

### 4. Environment Variables Configuration
```bash
cp .env.example .env
# Edit .env and set your Ollama service URL (e.g., local server or Colab Ngrok tunnel)
```

---

## 🖥️ Running the Services

### 1. FastAPI Web Server (REST API)
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
API docs available at `http://localhost:8000/docs`.

### 2. Local Streamlit GUI (Admin Dashboard)
```bash
streamlit run streamlit_app.py
```
- Single-text tester and thread-safe batch CSV processor.
- Dynamic charts and live Routing Ratio metrics.

### 3. Gradio Demo Interface
```bash
python app.py
```
Serves the web page on `http://localhost:7860`.

### 4. Docker Deployment (API only)
```bash
docker compose up --build
```
The container uses `requirements.txt` (core dependencies only — no GUI/agent).

---

## 🤖 Model Training

### Fine-tuning on ViHSD (Google Colab)
Use the provided Colab notebook:
- **[`notebooks/train_colab.ipynb`](notebooks/train_colab.ipynb)**: Runs the full fine-tuning pipeline on ViHSD dataset using XLM-R Base with focal loss, LLRD, and temperature calibration. Outputs a model artifact uploaded to HuggingFace Hub.

To run locally (requires GPU):
```bash
python src/training/train.py --config configs/train.yaml
```

### Continual Learning on VLSP-2019 (Google Colab)
Use the dedicated CL notebook:
- **[`notebooks/cl_pipeline_colab.ipynb`](notebooks/cl_pipeline_colab.ipynb)**: Runs sequential CL in two steps over VLSP-2019 data using a rehearsal buffer (ViHSD samples) and Gatekeeper validation. CL model outputs are stored in `models/cl/`.

Install Colab dependencies before running:
```bash
pip install -r requirements-train.txt
```

---

## 🧪 Testing and Validation
```bash
# Run the full unit test suite
python -m pytest tests/

# Run manual robustness tests against the model
python -m src.evaluation.manual_tests

# Run data quality report
python -m src.data.quality
```

---

## 📊 Evaluation Results

Evaluation metrics are stored in `results/`:
- `results/finetune_report.json` — Baseline fine-tuning metrics (XLM-R on ViHSD test set)
- `results/reliability_diagram.png` — Calibration reliability diagram
- `results/cl_step1_metrics.json` — CL Round 1 metrics (VLSP Part 1 + ViHSD rehearsal)
- `results/cl_step2_metrics.json` — CL Round 2 metrics (VLSP Part 2 + ViHSD rehearsal)

---

## 📁 Repository Structure

```
hate-speech-detection/
├── src/                    # Core source code
│   ├── api/                # FastAPI REST server
│   ├── agent/              # Ollama Qwen2.5 moderation agent
│   ├── models/             # HateSpeechClassifier + XLMRobertaTextCNN
│   ├── data/               # Preprocessing, augmentation, download scripts
│   ├── training/           # Trainer, CL orchestrator, robustness cases
│   ├── evaluation/         # Metrics, calibration, manual tests
│   ├── export/             # Model export and atomic CL deployment
│   ├── features/           # Grad-CAM toxic span extraction
│   ├── monitoring/         # Logging configuration
│   └── utils/              # Config loader, seeding
├── configs/                # YAML configuration files
├── data/                   # Data loading scripts (raw data gitignored)
├── models/                 # Model checkpoints (gitignored, use download.py)
│   └── cl/                 # CL output models (CL_output_step1, CL_output_step2)
├── tests/                  # pytest unit tests
├── notebooks/              # Colab training and evaluation notebooks
├── docs/                   # Project documentation (9 markdown documents)
├── results/                # Evaluation results and charts
├── streamlit_app.py        # Streamlit admin GUI
├── app.py                  # Gradio demo interface
├── main.py                 # FastAPI entry point
├── Dockerfile              # Docker container for API serving
├── docker-compose.yml      # Docker Compose configuration
├── requirements.txt        # Core dependencies (API/Docker)
├── requirements-local.txt  # Local GUI dependencies
├── requirements-train.txt  # Training dependencies (Colab)
└── requirements-dev.txt    # Development dependencies
```
