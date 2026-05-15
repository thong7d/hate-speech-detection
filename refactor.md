# Bao cao refactor va debug project Hate Speech Detection

## 1. Muc tieu

File nay ghi lai cu the nhung thay doi da thuc hien de project co pipeline ro rang hon cho:

- preprocessing;
- train/evaluation;
- inference trong API va agent;
- manual robustness test;
- data quality report;
- bao cao ky thuat.

Nguyen tac khi sua:

- Khong tu tao metric accuracy/F1 neu chua chay duoc voi du lieu that.
- Khong refactor lon toan bo cau truc project.
- Uu tien tao ham dung chung trong `src/` de notebook, API va agent khong bi lech logic.

## 2. Tom tat thay doi

| Nhom | Noi dung da sua |
|---|---|
| Preprocessing | Them module dung chung `src/data/preprocessing.py` |
| Data quality | Them script `src/data/quality.py` va report `results/data_quality_report.md` |
| Inference | Them shared predictor trong `src/evaluation/classifier.py` |
| API | Sua `src/api/app.py` de dung shared predictor |
| API schema | Cap nhat `src/api/schemas.py`, them `processed_text` va `probabilities` |
| Agent | Sua `src/agent/moderator.py` de dung shared predictor |
| Evaluation | Cap nhat `src/evaluation/evaluate.py` va `src/models/classifier.py` voi metric day du hon |
| Manual tests | Them `src/evaluation/manual_tests.py` va report `results/manual_test_report.md` |
| Notebook | Sua `notebooks/03_preprocessing.ipynb`, `notebooks/04_baseline.ipynb`, `notebooks/05_finetune.ipynb` |
| Report | Them `results/evaluation_report.md` |

## 3. Chi tiet cac file da sua

### 3.1. `src/data/preprocessing.py`

File moi, dung lam single source of truth cho preprocessing.

Da them:

- `clean_text(text)`: xoa URL, HTML entity, mention, normalize whitespace.
- `normalize_text_label_frame(df)`: chuan hoa DataFrame ve 2 cot bat buoc `text`, `label`.
- `process_split(input_path, output_path)`: doc raw CSV, clean text, ghi parquet.
- `compute_balanced_class_weights(labels, num_labels)`: tinh class weights tu train split.
- `save_class_weights(weights, output_path)`: luu class weights dung path config.

Ly do sua:

- Notebook cu ghi parquet voi cot `label_id`, trong khi fine-tuning notebook lai yeu cau cot `label`.
- De tranh train/evaluation/API dung cac cach clean text khac nhau.

### 3.2. `src/data/quality.py`

File moi de tao data quality report.

Da them kha nang kiem tra:

- so dong tung split;
- phan bo class;
- duplicate text trong tung split;
- overlap text giua train/val/test;
- report markdown tai `results/data_quality_report.md`.

Trang thai hien tai:

- Moi truong local chua co `pandas`, `omegaconf`, va chua co data processed nen report hien tai ghi ro `Khong du du lieu de xac minh`.

Lenh chay lai:

```bash
python -m src.data.quality
```

### 3.3. `src/evaluation/classifier.py`

Truoc do file nay trong. Da them shared inference layer.

Da them:

- `ModelArtifacts`: gom model, tokenizer, device, label map, max length.
- `load_hf_artifacts(...)`: load model/tokenizer tu Hugging Face hoac local path.
- `predict_text(...)`: ham predict dung chung.
- `predict_with_artifacts(...)`: predict bang artifact da load.
- `predict_many(...)`: predict nhieu cau.

Ham predict tra ve:

```python
{
    "text": "...",
    "processed_text": "...",
    "label": "...",
    "label_id": 0,
    "confidence": 0.0,
    "probabilities": {"CLEAN": 0.0, "OFFENSIVE": 0.0, "HATE": 0.0},
    "scores": {"CLEAN": 0.0, "OFFENSIVE": 0.0, "HATE": 0.0},
    "is_borderline": false
}
```

Ly do sua:

- API va agent truoc do moi noi tu tokenize, softmax, map label rieng.
- Neu label map, max length, preprocessing hoac threshold thay doi, inference rat de bi lech.

### 3.4. `src/api/app.py`

Da refactor API de dung `load_hf_artifacts()` va `predict_with_artifacts()`.

Thay doi cu the:

- Bo logic load `AutoTokenizer`/`AutoModelForSequenceClassification` truc tiep trong API.
- Bo logic tokenization/softmax truc tiep trong endpoint `/predict`.
- API dung shared preprocessing va shared inference.
- Health check lay device tu artifact chung.

Ly do sua:

- Dam bao API prediction giong evaluation/manual test/agent.

### 3.5. `src/api/schemas.py`

Da cap nhat response schema.

Them:

- `processed_text`: text sau preprocessing chung.
- `probabilities`: xac suat tung class.

Giu lai:

- `scores`: alias backward-compatible de `main.py` va UI cu van dung duoc.

### 3.6. `src/agent/moderator.py`

Da sua `ModerationTools` de dung shared predictor.

Thay doi cu the:

- Bo load tokenizer/model rieng trong agent.
- Bo hard-code inference rieng trong `classify_text`.
- Dung `load_hf_artifacts()` khi khoi tao.
- Dung `predict_with_artifacts()` khi classify.
- Threshold borderline doc tu env `BORDERLINE_LOW`, `BORDERLINE_HIGH`.

Ly do sua:

- Agent va API truoc do co the dua ra ket qua khac nhau cho cung mot text neu logic drift.

### 3.7. `src/evaluation/evaluate.py`

Da bo sung metric day du hon.

Truoc do co:

- macro F1;
- per-class F1;
- classification report;
- confusion matrix;
- optional AUC.

Da them:

- accuracy;
- precision macro;
- recall macro;
- weighted F1;
- recall hate speech;
- `zero_division=0`;
- labels co dinh `[0, 1, 2]`.

Ly do sua:

- Accuracy cao khong du de ket luan voi dataset mat can bang.
- Can theo doi macro F1 va recall cua class HATE.

### 3.8. `src/models/classifier.py`

Da cap nhat `compute_metrics()` cho Hugging Face Trainer.

Da them:

- accuracy;
- precision macro;
- recall macro;
- weighted F1;
- zero division handling.

Ly do sua:

- Training/evaluation trong Trainer can metric phu hop hon chi macro F1/per-class F1.

### 3.9. `src/evaluation/manual_tests.py`

File moi de chay manual robustness tests.

Bao gom 30 test cases cho:

- doi chu ngu;
- keyword bias;
- hate speech ngam;
- cau trung lap;
- phu dinh hate speech.

Output markdown co bang:

```markdown
| text | expected | predicted | confidence | probabilities | pass | category |
```

Lenh chay lai:

```bash
python -m src.evaluation.manual_tests --model-source thong7d/vihsd-xlmr-hate-speech
```

Trang thai hien tai:

- Da tao `results/manual_test_report.md`.
- Chua co prediction that vi moi truong `.venv` hien tai thieu `torch`.

### 3.10. `notebooks/03_preprocessing.ipynb`

Da sua Cell 1.

Truoc do:

- Dinh nghia `clean_text` truc tiep trong notebook.
- Ghi parquet voi cot `text`, `label_id`.
- Tinh class weights trong notebook.

Sau khi sua:

- Import tu `data.preprocessing`.
- Ghi parquet canonical voi cot `text`, `label`.
- Tinh class weights tu train split only.
- Luu class weights theo path `cfg.results.class_weights`.

Loi duoc fix:

- Schema mismatch `label_id` vs `label`.

### 3.11. `notebooks/04_baseline.ipynb`

Da sua Cell 1.

Truoc do:

- Doc config sai bang cac key khong ton tai nhu `raw_data`, `processed_data`, `results`.
- Tao path sai, dan den file la:

```text
{project_root}/results/class_weights.json/baseline_report.json
```

Sau khi sua:

- Dung `OmegaConf`.
- Dung `cfg.data.train_processed`, `cfg.data.val_processed`.
- Dung `cfg.models.baseline`.
- Dung `cfg.results.baseline_report`.
- Luu baseline report day du hon gom:
  - evaluated split;
  - macro F1;
  - classification report;
  - confusion matrix.

### 3.12. `notebooks/05_finetune.ipynb`

Da sua resolve path.

Truoc do:

- `resolve_path()` chi thay `{drive_root}`.
- Trong `configs/paths.yaml` hien tai lai dung `{project_root}`.

Sau khi sua:

```python
return template_path.replace("{project_root}", PROJECT_ROOT).replace("{drive_root}", DRIVE_ROOT)
```

Ly do sua:

- Fine-tuning notebook co the load sai path hoac giu nguyen literal `{project_root}`.

## 4. File report da tao

### 4.1. `results/evaluation_report.md`

Noi dung:

- tom tat thay doi;
- loi phat hien;
- file da chinh sua;
- tinh trang data quality;
- tinh trang evaluation;
- ket luan ky thuat;
- viec can lam tiep theo.

Luu y:

- Khong tu bia metric.
- Vi local environment thieu data/model/dependency, report ghi ro `Khong du du lieu de xac minh`.

### 4.2. `results/manual_test_report.md`

Noi dung:

- 30 manual test cases;
- expected label;
- predicted label;
- confidence;
- probabilities;
- pass/fail;
- category.

Trang thai hien tai:

- Prediction la `N/A` do chua load duoc model trong local environment vi thieu `torch`.

### 4.3. `results/data_quality_report.md`

Noi dung:

- report trang thai moi truong;
- thong bao thieu dependency.

Trang thai hien tai:

- Chua co du lieu local va dependency can thiet nen chua xac minh duoc duplicate/overlap/class distribution.

## 5. Kiem tra da chay

Da chay compile check:

```bash
.venv/Scripts/python.exe -m py_compile src/data/preprocessing.py src/data/quality.py src/evaluation/classifier.py src/evaluation/evaluate.py src/evaluation/manual_tests.py src/api/app.py src/api/schemas.py src/agent/moderator.py src/models/classifier.py
```

Ket qua:

```text
OK
```

Da kiem tra notebook JSON parse:

```bash
python - <<'PY'
import json
from pathlib import Path
for path in Path("notebooks").glob("*.ipynb"):
    json.loads(path.read_text(encoding="utf-8"))
print("notebook json ok")
PY
```

Ket qua:

```text
notebook json ok
```

Da chay manual test script:

```bash
.venv/Scripts/python.exe -m src.evaluation.manual_tests --output results/manual_test_report.md
```

Ket qua:

```text
Manual test rows: 30
Report written to results/manual_test_report.md
```

Nhung prediction la `N/A` vi thieu `torch`.

## 6. Nhung dieu chua the ket luan

Chua the ket luan bang so lieu that:

- accuracy cu co dang tin khong;
- model co keyword bias khong;
- model co loi doi chu ngu khong;
- model co phan biet duoc phu dinh nhu `I do not hate anyone` khong;
- model co overfit/underfit khong;
- confusion matrix that;
- macro F1/weighted F1/recall HATE that.

Ly do:

- Repo local khong co `data/processed/train.parquet`, `dev.parquet`, `test.parquet`.
- Repo local khong co model checkpoint/final model.
- `.venv` hien tai thieu dependency quan trong nhu `pandas`, `omegaconf`, `torch`.

## 7. Cach chay tiep de co bao cao that

Sau khi cai dependency:

```bash
pip install -r requirements.txt
```

Chay lai pipeline:

```bash
python -m src.data.quality
python -m src.evaluation.manual_tests --model-source thong7d/vihsd-xlmr-hate-speech
```

Neu dung notebook:

1. Chay `notebooks/03_preprocessing.ipynb`.
2. Chay `notebooks/04_baseline.ipynb`.
3. Chay `notebooks/05_finetune.ipynb`.
4. Chay `notebooks/06_evaluation.ipynb`.
5. Chay manual tests.

## 8. Ket luan

Da refactor project theo huong:

- preprocessing co ham dung chung;
- inference co ham dung chung;
- API va agent thong nhat prediction;
- evaluation khong dua vao accuracy don le;
- manual robustness tests san sang de bat loi doi chu ngu, negation va keyword bias;
- report khong tu tao so lieu khi moi truong chua du de xac minh.
