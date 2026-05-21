# Hate Speech Detection AI Agent

## 1. Overview

An end-to-end Vietnamese hate speech detection system built on **PhoBERT (vinai/phobert-base-v2)** fine-tuned on the **ViHSD** dataset (~33,400 comments, 3 classes: CLEAN / OFFENSIVE / HATE).

The system includes:
- **FastAPI REST API** for real-time inference
- **Gemini-powered content moderation agent** with multi-step reasoning
- **Gradio web demo** for interactive testing
- **Docker** support for containerized deployment

### Architecture

```text
Raw Vietnamese Text
  → Preprocessing (Unicode norm, teencode, word segmentation)
  → PhoBERT Fine-tuned Classifier
  → FastAPI REST API
  → Gemini Agent (multi-step reasoning, tool usage)
  → Gradio / HF Spaces Demo
```

## 2. Project Structure

```text
configs/                 YAML config for training, model loading, and API
notebooks/train_colab.ipynb
notebooks/archive/       legacy notebooks kept for reference
src/api/                 FastAPI app, schemas, dependencies
src/data/                dataset, preprocessing, augmentation
src/training/            Trainer entry point + custom loss/LLRD
src/evaluation/          metrics, error analysis, manual robustness tests
src/export/              artifact export and Hugging Face upload
src/models/              inference class and label mapping registry
src/agent/               Gemini-based content moderator
tests/                   pytest coverage for API/config/inference/preprocessing
.github/workflows/       CI and Docker workflows
```

## 3. Key Improvements (v2.0)

| Feature | v1 (XLM-R) | v2 (PhoBERT) |
|---|---|---|
| Backbone | xlm-roberta-base | **vinai/phobert-base-v2** |
| Preprocessing | Basic (URL, mention) | **+Unicode NFKC, +teencode norm, +repeated chars, +word segmentation** |
| Augmentation | Robustness cases | **+EDA, +diacritic removal, +teencode variants** |
| Training | Standard fine-tune | **+Layer-wise LR decay, +label smoothing, +focal loss γ=2.0** |
| Evaluation | Basic metrics | **+AUC-ROC per class, +error_analysis.csv** |
| Target F1 | ~0.60–0.68 | **~0.72–0.80** |

## 4. Local Setup

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
pip install -r requirements.txt
```

Do not commit `.env`, `.venv`, private datasets, local caches, or tokens.

## 5. Train on Google Colab

Open `notebooks/train_colab.ipynb`. The notebook calls project modules:

```bash
python -m src.training.train --config configs/train.yaml
python -m src.evaluation.evaluate --config configs/train.yaml
python -m src.export.export_model --config configs/train.yaml --push-to-hub
python -m src.evaluation.manual_tests --config configs/model.yaml
```

Configure `HF_TOKEN` in Colab secrets for Hugging Face Hub access.

## 6. Model Artifacts

Local artifact root: `artifacts/hate_speech_model/`

Expected files:
```text
checkpoint/
model/
label_mapping.json
metadata.json
metrics.json
model_card.md
```

## 7. Push Model to Hugging Face Hub

```bash
set HF_TOKEN=your_token
python -m src.export.export_model --config configs/train.yaml --push-to-hub
```

## 8. Run API Locally

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The API loads from Hugging Face first when configured, then falls back to a local artifact.

## 9. Run with Docker

```bash
docker build -t hate-speech-api .
docker run -p 8000:8000 hate-speech-api
```

With Compose:
```bash
docker compose up --build
```

## 10. API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process health check |
| `GET /ready` | Model readiness and version |
| `GET /metadata` | Returns `metadata.json` content |
| `POST /predict` | Predict one text (with language detection) |
| `POST /predict-batch` | Predict multiple texts |

### Predict Request/Response

```json
// POST /predict
{
  "text": "Bài viết này thật tuyệt vời!",
  "language": "vi"
}

// Response
{
  "text": "Bài viết này thật tuyệt vời!",
  "label": "CLEAN",
  "confidence": 0.95,
  "probabilities": {"CLEAN": 0.95, "OFFENSIVE": 0.03, "HATE": 0.02},
  "model_version": "v2.0.0",
  "language": "vi"
}
```

## 11. Preprocessing Pipeline

```text
Input: raw Vietnamese text
  → Unicode NFKC normalization
  → Remove URLs, HTML entities, @mentions
  → Normalize repeated characters ("đẹpppp" → "đẹp")
  → Normalize teencode ("ko" → "không", "dc" → "được")
  → Normalize whitespace
  → Word segmentation via underthesea ("sinh viên" → "sinh_viên")
  → Do NOT lowercase or strip diacritics
Output: cleaned text ready for PhoBERT tokenizer
```

## 12. Evaluation

```bash
python -m src.evaluation.evaluate --config configs/train.yaml
```

Outputs:
- `artifacts/hate_speech_model/metrics.json` — all metrics including AUC-ROC
- `results/evaluation_report.md` — human-readable report
- `results/confusion_matrix.png` — visualization
- `results/error_analysis.csv` — misclassified samples with error types

## 13. Manual Robustness Tests

```bash
python -m src.evaluation.manual_tests --config configs/model.yaml
```

Report written to `results/manual_test_report.md`.

## 14. Agentic AI Component

The system includes a Gemini-powered content moderation agent with:
- **Multi-step reasoning**: Language detection → Classification → Decision
- **Tool usage**: classify_text(), detect_language(), log_event()
- **Dynamic decisions**: Borderline texts trigger clarifying questions

## 15. CI/CD

GitHub Actions includes:
- `.github/workflows/ci.yml`: lint, test, API import check
- `.github/workflows/docker.yml`: Docker build and health check

## 16. Limitations

- Model quality depends on the ViHSD dataset (Vietnamese social media)
- PhoBERT requires Vietnamese word segmentation (underthesea dependency)
- Automated moderation should be paired with human review for high-impact decisions
- Private Hugging Face repos require `HF_TOKEN`
