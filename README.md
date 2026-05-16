# Hate Speech Detection AI Agent

## 1. Overview

This project trains and serves a hate speech detection model. Training is designed for Google Colab, while local and production environments focus on API serving, tests, Docker, and CI/CD.

The production flow is:

```text
notebooks/train_colab.ipynb
  -> src.training.train
  -> src.evaluation.evaluate
  -> src.export.export_model
  -> Hugging Face Hub
  -> FastAPI production
  -> Docker + CI/CD
```

## 2. Project Structure

```text
configs/                 YAML config for training, model loading, and API
notebooks/train_colab.ipynb
notebooks/archive/       legacy notebooks kept for reference
src/api/                 FastAPI app, schemas, dependencies
src/data/                dataset and shared preprocessing
src/training/            Trainer entry point
src/evaluation/          metrics and manual robustness tests
src/export/              artifact export and Hugging Face upload
src/models/              inference class and label mapping registry
tests/                   pytest coverage for API/config/inference/preprocessing
.github/workflows/       CI and Docker workflows
```

## 3. Local Setup

```bash
python -m venv .venv
pip install -r requirements-dev.txt
pytest
```

Do not commit `.env`, `.venv`, private datasets, local caches, or tokens.

## 4. Train on Google Colab

Open `notebooks/train_colab.ipynb`. The notebook only calls project modules:

```bash
python -m src.training.train --config configs/train.yaml
python -m src.evaluation.evaluate --config configs/train.yaml
python -m src.export.export_model --config configs/train.yaml --push-to-hub
python -m src.evaluation.manual_tests --config configs/model.yaml
```

Configure `HF_TOKEN` in Colab secrets. If no token is available, training/export still runs locally and the push step is skipped.

## 5. Model Artifacts

The local artifact root is:

```text
artifacts/hate_speech_model/
```

Expected files:

```text
checkpoint/
model/
label_mapping.json
metadata.json
metrics.json
model_card.md
```

Metrics are only written from a real evaluation run. If evaluation cannot run, the artifact states `Khong du du lieu de xac minh`.

## 6. Push Model to Hugging Face Hub

Set the repo in `configs/train.yaml` or `configs/model.yaml`:

```yaml
hf_repo_id: "quanghs1020/hate-speech-detection"
```

Then run:

```bash
set HF_TOKEN=your_token
python -m src.export.export_model --config configs/train.yaml --push-to-hub
```

Only the final model, label mapping, metadata, metrics, and model card are uploaded by default. Checkpoints are skipped unless `--push-checkpoints` is passed.

## 7. Run API Locally

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

The API loads from Hugging Face first when configured, then falls back to a local artifact if available.

## 8. Run with Docker

```bash
docker build -t hate-speech-api .
docker run -p 8000:8000 hate-speech-api
```

With Compose:

```bash
docker compose up --build
```

## 9. API Endpoints

| Endpoint | Purpose |
|---|---|
| `GET /health` | Process health check |
| `GET /ready` | Model readiness and version |
| `GET /metadata` | Returns `metadata.json` content |
| `POST /predict` | Predict one text |
| `POST /predict-batch` | Predict multiple texts |

## 10. Evaluation

```bash
python -m src.evaluation.evaluate --config configs/train.yaml
```

Outputs are written to `artifacts/hate_speech_model/metrics.json` and `results/evaluation_report.md`.

## 11. Manual Robustness Tests

```bash
python -m src.evaluation.manual_tests --config configs/model.yaml
```

The report is written to `results/manual_test_report.md`.

## 12. CI/CD

GitHub Actions includes:

- `.github/workflows/ci.yml`: install dev dependencies, run `ruff`, run `pytest`, import the API.
- `.github/workflows/docker.yml`: build the Docker image and call `/health`.

CI does not download large datasets or train a model.

## 13. Limitations

- Model quality depends on the dataset and evaluation run used to create the artifact.
- Automated moderation should be paired with human review for high-impact decisions.
- Private Hugging Face repos require `HF_TOKEN`; never hard-code it in notebooks, source, Dockerfile, or Compose files.
