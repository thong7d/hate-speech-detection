# Bao cao Refactor Production Hate Speech Detection

## 1. Tom tat thay doi

Da refactor project theo huong production:

- Chi giu mot notebook train chinh: `notebooks/train_colab.ipynb`.
- Chuyen cac notebook cu vao `notebooks/archive/`.
- Dua logic train, evaluation, export, inference vao `src/`.
- Them config rieng cho train, model va API.
- Them model artifact contract tai `artifacts/hate_speech_model/`.
- Them ho tro export va upload model len Hugging Face Hub.
- Refactor FastAPI thanh production API.
- Them Dockerfile, docker-compose, Makefile, GitHub Actions va pytest tests.
- Cap nhat README huong dan train Colab, chay local, Docker, Hugging Face va CI/CD.

## 2. Cau truc project sau refactor

```text
project_root/
├── configs/
│   ├── paths.yaml
│   ├── train.yaml
│   ├── model.yaml
│   └── api.yaml
├── notebooks/
│   ├── train_colab.ipynb
│   └── archive/
│       ├── 01_data.ipynb
│       ├── 02_eda.ipynb
│       ├── 03_preprocessing.ipynb
│       ├── 04_baseline.ipynb
│       ├── 05_finetune.ipynb
│       ├── 06_evaluation.ipynb
│       └── 07_deployment.ipynb
├── src/
│   ├── api/
│   ├── data/
│   ├── evaluation/
│   ├── export/
│   ├── models/
│   ├── monitoring/
│   ├── training/
│   └── utils/
├── tests/
├── .github/workflows/
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── requirements-dev.txt
├── requirements-deploy.txt
├── README.md
└── main.py
```

## 3. File da them

| File | Muc dich |
|---|---|
| `configs/train.yaml` | Config train, evaluation va export artifact |
| `configs/model.yaml` | Config load model tu Hugging Face hoac local artifact |
| `configs/api.yaml` | Config API production |
| `notebooks/train_colab.ipynb` | Notebook Colab duy nhat, chi goi command trong `src/` |
| `src/utils/config.py` | Load YAML va resolve path |
| `src/utils/seed.py` | Set seed cho reproducibility |
| `src/training/train.py` | CLI train |
| `src/training/trainer.py` | Logic train bang Hugging Face Trainer |
| `src/evaluation/metrics.py` | Metric helper dung cho evaluation |
| `src/export/export_model.py` | Export artifact va push Hugging Face Hub |
| `src/models/registry.py` | Label mapping registry |
| `src/api/schema.py` | Pydantic schemas production |
| `src/api/dependencies.py` | Load classifier dependency |
| `src/monitoring/logging_config.py` | Logging config |
| `tests/test_api.py` | API schema tests bang mock classifier |
| `tests/test_config.py` | Config tests |
| `tests/test_inference.py` | Inference contract tests |
| `tests/test_label_mapping.py` | Label mapping consistency tests |
| `tests/test_preprocessing.py` | Shared preprocessing tests |
| `.github/workflows/ci.yml` | CI: install, ruff, pytest, import API |
| `.github/workflows/docker.yml` | Docker build va `/health` smoke test |
| `requirements-dev.txt` | Dev/test dependencies |
| `Makefile` | Lenh tien ich train, test, lint, Docker, API |

## 4. File da sua

| File | Thay doi |
|---|---|
| `.gitignore` | Them cache, wandb, mlruns, checkpoint artifact |
| `Dockerfile` | Refactor thanh Dockerfile production cho FastAPI |
| `docker-compose.yml` | Service API production, env config khong hard-code token |
| `main.py` | Don gian hoa entrypoint uvicorn |
| `README.md` | Viet lai huong dan production |
| `requirements-deploy.txt` | Rut gon deploy dependencies, them PyYAML/HF hub |
| `src/api/app.py` | Them `/health`, `/ready`, `/metadata`, `/predict`, `/predict-batch` |
| `src/api/schemas.py` | Giu compatibility, re-export tu `schema.py` |
| `src/data/preprocessing.py` | Them `preprocess_text()` dung chung training/inference |
| `src/evaluation/evaluate.py` | Them evaluation CLI config-driven |
| `src/evaluation/manual_tests.py` | Manual robustness test config-driven |
| `src/models/classifier.py` | Them `HateSpeechClassifier`, fallback HF/local, predict batch |

## 5. Notebook Colab moi

- Duong dan: `notebooks/train_colab.ipynb`
- Vai tro: wrapper, khong chua logic train/evaluate/export chinh.
- Cac command chinh:

```bash
python -m src.training.train --config configs/train.yaml
python -m src.evaluation.evaluate --config configs/train.yaml
python -m src.export.export_model --config configs/train.yaml --push-to-hub
python -m src.evaluation.manual_tests --config configs/model.yaml
```

- Output artifact: `artifacts/hate_speech_model/`
- Neu khong co `HF_TOKEN`, notebook bo qua push Hub va in thong bao:

```text
Khong du du lieu de xac minh HF_TOKEN hoac HF_TOKEN chua duoc cau hinh
```

## 6. Model artifact

- Local artifact path: `artifacts/hate_speech_model/`
- Final model path: `artifacts/hate_speech_model/model/`
- Hugging Face repo mac dinh: `quanghs1020/hate-speech-detection`
- File upload mac dinh:
  - `model/`
  - `label_mapping.json`
  - `metadata.json`
  - `metrics.json`
  - `model_card.md`
- File khong upload mac dinh:
  - checkpoint
  - `.env`
  - `HF_TOKEN`
  - private dataset
  - cache
  - `.venv`
  - `__pycache__`

## 7. API production

| Endpoint | Muc dich |
|---|---|
| `GET /health` | Healthcheck process, tra `{"status": "ok"}` |
| `GET /ready` | Kiem tra model da load chua |
| `GET /metadata` | Tra metadata tu artifact |
| `POST /predict` | Predict mot text |
| `POST /predict-batch` | Predict nhieu text |

Output prediction co contract:

```json
{
  "text": "sample text",
  "label": "CLEAN",
  "confidence": 0.99,
  "probabilities": {
    "CLEAN": 0.99,
    "OFFENSIVE": 0.01,
    "HATE": 0.0
  },
  "model_version": "v1.0.0"
}
```

## 8. Docker

Build:

```bash
docker build -t hate-speech-api .
```

Run:

```bash
docker run -p 8000:8000 hate-speech-api
```

Compose:

```bash
docker compose up --build
```

Healthcheck:

```text
GET http://localhost:8000/health
```

## 9. CI/CD

Workflow da them:

- `.github/workflows/ci.yml`
  - checkout
  - setup Python 3.10
  - install `requirements-dev.txt`
  - `ruff check .`
  - `pytest`
  - import API

- `.github/workflows/docker.yml`
  - build Docker image
  - run container
  - call `/health`

CI khong train model nang va khong download dataset lon.

## 10. Test

Lenh chay:

```bash
pytest
```

Trang thai verification da chay trong moi truong hien tai:

```text
python -m compileall -q src tests
```

Ket qua:

```text
passed
```

Smoke checks da chay:

- load YAML config: passed
- preprocessing helper: passed
- label mapping helper: passed
- compute metrics helper: passed

Chua chay duoc `pytest` vi moi truong hien tai chua co `pytest`.

Chua import duoc FastAPI app bang moi truong hien tai vi chua co `fastapi`.

```text
Khong du du lieu de xac minh
```

## 11. Cach train tren Colab

Mo:

```text
notebooks/train_colab.ipynb
```

Thiet lap Colab secret:

```text
HF_TOKEN
```

Pipeline:

```bash
python -m src.training.train --config configs/train.yaml
python -m src.evaluation.evaluate --config configs/train.yaml
python -m src.export.export_model --config configs/train.yaml --push-to-hub
python -m src.evaluation.manual_tests --config configs/model.yaml
```

## 12. Cach chay local API

```bash
pip install -r requirements-deploy.txt
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

Hoac:

```bash
make run-api
```

## 13. Ket luan ky thuat

- Project da tach train/deploy ro rang.
- Notebook Colab chi con la wrapper, khong con la noi chua logic chinh.
- API production khong phu thuoc Colab.
- Model co the load tu Hugging Face Hub hoac fallback tu local artifact.
- Label mapping duoc gom ve `src/models/registry.py`, khong hard-code rai rac.
- Training va inference dung chung preprocessing qua `preprocess_text()`.
- CI/CD kiem tra lint, tests, API import va Docker healthcheck.
- Can lam tiep:
  - Cai dev dependencies de chay `pytest` that.
  - Train model tren Colab de tao artifact that.
  - Chay evaluation that de sinh `metrics.json`.
  - Push artifact len Hugging Face Hub voi `HF_TOKEN`.
