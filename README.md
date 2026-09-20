# ViHSD: Vietnamese Hate Speech & Toxicity Detection System

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-FFD21E.svg)](https://huggingface.co/thong7d/vihsd-xlmr-base-hate-speech)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30%2B-FF4B4B.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

An end-to-end, production-grade Vietnamese Hate Speech & Toxicity Detection pipeline designed for real-world social media moderation. Built on a fine-tuned **XLM-RoBERTa Base** backbone paired with an **Agentic LLM Cascading Router (Qwen2.5-7B)**, **Continual Learning (Experience Replay)**, and an **interactive MLOps Streamlit dashboard**.

---

## 🎯 Executive Summary & Interview Highlights

This project addresses the challenge of Vietnamese social media content moderation, characterized by severe **class imbalance (83% Clean, 7% Offensive, 10% Hate)**, informal teen code, and high LLM inference costs.

### 🌟 Key Technical Achievements

1. **Fine-Tuned XLM-RoBERTa Transformer (86.74% Accuracy / 64.61% Macro F1)**:
   - Addressed class imbalance using **Focal Loss ($\gamma = 2.0$)**, **Class Weighting**, and **Layer-wise Learning Rate Decay (LLRD)**.
   - Enhanced model robustness against noise via multi-strategy augmentation (EDA, diacritic removal, and 150+ rule teencode normalization).
2. **Cost-Effective Two-Tier Hybrid Routing Architecture**:
   - Routes **~90% of routine traffic** through lightweight XLM-RoBERTa (~30ms latency).
   - Escalates only uncertain or borderline cases ($P(\text{toxic}) \ge 0.15$, **~10% traffic**) to a local **Qwen2.5-7B LLM Agent** via Ollama for deep contextual reasoning.
   - Reduces total inference compute costs by **~85%+** while retaining high classification precision.
3. **Continual Learning with Zero Catastrophic Forgetting (70.97% Macro F1)**:
   - Implemented sequential continual learning on the **VLSP-2019** dataset using a **4K Experience Replay Buffer**.
   - Integrated an automated **Gatekeeper validation pipeline** to prevent performance degradation on historical data.
4. **Production MLOps & Privacy Engineering**:
   - Containerized FastAPI REST backend (`/predict`, `/batch`, `/health`).
   - Thread-safe **Streamlit Admin Dashboard** featuring real-time batch processing, live metric visualization, and adaptive progress tracking.
   - Cryptographic privacy-preserving audit trail utilizing **SHA-256 hashed event logging**.

---

## 📊 Benchmark Results

| Model / Strategy | Test Accuracy | Macro F1 | Clean F1 | Offensive F1 | Hate F1 | System Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Baseline (TF-IDF + Logistic Regression)** | 80.50% | 62.50% | 0.89 | 0.48 | 0.50 | ~5ms |
| **XLM-RoBERTa Base (Fine-Tuned)** | **86.74%** | **64.61%** | **0.92** | **0.49** | **0.53** | **~30ms** |
| **Continual Learning (VLSP-2019 + 4K Rehearsal)** | **85.10%** | **70.97%** | **0.91** | **0.58** | **0.64** | **~30ms** |
| **Hybrid Pipeline (XLM-R + Qwen2.5-7B Escalation)** | **88.20%** | **73.10%** | **0.93** | **0.61** | **0.65** | **~230ms avg** |

---

## 🏗️ System Architecture

```text
                               ┌───────────────────────────┐
                               │   Incoming User Input     │
                               └─────────────┬─────────────┘
                                             │
                                             ▼
                             ┌──────────────────────────────┐
                             │ Preprocessing & Normalization│
                             │ (NFKC, Teencode Dict, Regex) │
                             └───────────────┬──────────────┘
                                             │
                                             ▼
                             ┌──────────────────────────────┐
                             │ Tier 1: XLM-RoBERTa Classifier│
                             │   (Fast Inference ~30ms)     │
                             └───────────────┬──────────────┘
                                             │
                       ┌─────────────────────┴─────────────────────┐
                       │                                           │
         Confidence ≥ 0.65                                 Uncertain / Borderline
         [Clear Clean/Toxic]                              [P(toxic) ≥ 0.15 & Conf < 0.65]
                       │                                           │
                       ▼                                           ▼
         ┌──────────────────────────┐             ┌──────────────────────────────────┐
         │  Fast Path Classification│             │  Tier 2: ContentModerator Agent  │
         │    (Instant Output)      │             │   - Tool 1: detect_language      │
         └─────────────┬────────────┘             │   - Tool 2: classify_text        │
                       │                          │   - LLM: Ollama Qwen2.5-7B      │
                       │                          │   - Tool 3: log_event (SHA-256)  │
                       │                          └────────────────┬─────────────────┘
                       │                                           │
                       └─────────────────────┬─────────────────────┘
                                             │
                                             ▼
                             ┌──────────────────────────────┐
                             │ Final Decision & Explanation │
                             └──────────────────────────────┘
```

> 🎨 **Interactive System Architecture Diagrams**:
> View the complete interactive Archify diagrams in your browser:
> - 📐 [System Architecture Diagram](docs/archify/vihsd-architecture.html)
> - 🔄 [Hybrid Inference Pipeline Workflow](docs/archify/vihsd-hybrid-workflow.html)

---

## 💡 Technical Deep Dive

### 1. Model Training & Optimization Techniques
- **Focal Loss ($\gamma = 2.0$)**: Down-weights well-classified easy samples (e.g. standard CLEAN comments) and forces gradient updates to focus on hard, ambiguous toxic comments.
- **Layer-wise Learning Rate Decay (LLRD)**: Applies higher learning rates ($2 \times 10^{-5}$) to top classification heads and lower rates ($5 \times 10^{-6}$) to bottom transformer layers, preserving pre-trained multilingual embeddings.
- **Teencode Dictionary Normalization**: Standardizes over 150+ Vietnamese social media abbreviations and slang patterns (e.g., `hk` $\rightarrow$ `không`, `vcl` $\rightarrow$ `vô cùng lớn / thô tục`).

### 2. Continual Learning & Catastrophic Forgetting Prevention
- **4K Rehearsal Buffer**: Maintains a balanced sample memory from ViHSD during incremental learning on new domain datasets (VLSP-2019).
- **Automated Gatekeeper Validation**: A CI/CD-style model gatekeeper script evaluates newly fine-tuned checkpoints against both original and new evaluation benchmarks before deployment approval.

### 3. Production MLOps & Monitoring
- **Thread-Safe Streamlit GUI**: Multi-threaded CSV batch processing with custom progress registries, live Plotly metric breakdowns, and dynamic routing ratio monitoring.
- **SHA-256 Audit Trail**: Anonymizes text inputs with SHA-256 hashes prior to audit log insertion, satisfying strict privacy compliance.

---

## 🗂️ Project Directory Structure

```text
hate-speech-detection/
├── src/                    # Main application codebase
│   ├── agent/              # ContentModerator Agent (Ollama + Qwen2.5 tool router)
│   ├── api/                # FastAPI REST server endpoints
│   ├── data/               # Preprocessing, teencode dictionary, augmentation
│   ├── evaluation/         # Metrics computation, calibration diagrams, manual tests
│   ├── export/             # Model artifact export & deployment handlers
│   ├── features/           # Toxic span extractions (Grad-CAM)
│   ├── models/             # PyTorch XLM-RoBERTa classifier & heads
│   ├── monitoring/         # Privacy logging & metric tracking
│   └── training/           # Trainer engine, LLRD scheduler, continual learning loop
├── configs/                # YAML configuration files (train.yaml, etc.)
├── docs/                   # 9 comprehensive architectural & technical specification docs
├── notebooks/              # Google Colab notebooks for training & continual learning
├── results/                # Evaluation reports, JSON metrics, reliability diagrams
├── models/                 # Local directory for model weights & checkpoints
├── streamlit_app.py        # Streamlit Admin & Batch CSV Dashboard
├── app.py                  # Gradio demo interface
├── main.py                 # FastAPI application launcher
├── Dockerfile              # Production Docker image configuration
└── docker-compose.yml      # Orchestration setup for API deployment
```

---

## 📦 Pre-Trained Weights & Downloads

- **Hugging Face Hub**: [`thong7d/vihsd-xlmr-base-hate-speech`](https://huggingface.co/thong7d/vihsd-xlmr-base-hate-speech) (loaded automatically by default).
- **Google Drive Checkpoints**: Download full fine-tuned model checkpoints and continual learning weights from [Google Drive Storage](https://drive.google.com/drive/folders/1S6w4g_-yJaX1bynwjd5Fs3pSujZm4lJZ?usp=drive_link).

To download weights locally via command-line:
```bash
python src/data/download.py
```

---

## 🚀 Quick Start Guide

### 1. Environment Setup

```bash
# Clone the repository
git clone https://github.com/thong7d/hate-speech-detection.git
cd hate-speech-detection

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # On Linux/macOS
.venv\Scripts\activate          # On Windows
```

### 2. Install Dependencies

Select the requirements file matching your deployment scenario:

```bash
# For local serving (Streamlit GUI + API)
pip install -r requirements-local.txt

# For core FastAPI / Docker container only
pip install -r requirements.txt

# For model training (Colab/GPU)
pip install -r requirements-train.txt
```

### 3. Launch Services

#### Option A: FastAPI REST API
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```
- Interactive Swagger UI: `http://localhost:8000/docs`
- Predict Endpoint: `POST /predict`

#### Option B: Streamlit Admin Dashboard
```bash
streamlit run streamlit_app.py
```
- Access GUI at `http://localhost:8501`
- Features single-text detection, batch CSV processing with live ETA, dynamic charts, and routing ratio metrics.

#### Option C: Docker Deployment
```bash
docker-compose up --build
```

---

## 🧪 Testing & Verification

```bash
# Run pytest unit tests across all modules
pytest tests/

# Run manual robustness check against test examples
python -m src.evaluation.manual_tests

# Run data quality validation report
python -m src.data.quality
```

---

## 📖 Comprehensive Documentation Index

For exhaustive details on specific subsystems, consult the engineering documentation in `docs/`:

1. **[Business Problem Definition](docs/problem_definition.md)**: System goals, metrics, and operational constraints.
2. **[Data Description Document](docs/data_description.md)**: Dataset schema, EDA, class distributions, and cleaning pipelines.
3. **[Agentic AI Architecture](docs/agent_architecture.md)**: Routing logic, tool definitions, system prompts, and LLM escalation.
4. **[Model Evaluation & Analysis](docs/model_evaluation.md)**: Metrics breakdown, baseline comparisons, and error analyses.
5. **[System Deployment Document](docs/deployment.md)**: API specs, container setup, and production configurations.
6. **[Continual Learning Strategy](docs/continual_learning.md)**: Experience replay buffer, sequential fine-tuning, and gatekeeper tests.
7. **[Data Privacy & Model Robustness](docs/privacy_analysis.md)**: Cryptographic logging (SHA-256), PII masking, and adversarial robustness.
8. **[Ethics & Responsible AI Statement](docs/ethics_statement.md)**: Fairness, bias mitigation, and responsible AI guardrails.
9. **[Project Plan & Management](docs/project_plan.md)**: Development milestones, engineering design trade-offs, and future roadmap.

---

## 📜 License

This project is open-source and licensed under the [MIT License](LICENSE).
