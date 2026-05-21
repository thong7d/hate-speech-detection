# Xây Dựng Lại Hate Speech Detection System — Cải Thiện F1-Score

Rebuild toàn bộ pipeline từ đầu, giữ kiến trúc project hiện tại nhưng cải thiện mạnh preprocessing, data augmentation, training strategy để nâng Macro F1 lên mức **0.72–0.80+** (so với baseline ~0.60–0.68).

## User Review Required

> [!IMPORTANT]
> **Chọn backbone model:** Nghiên cứu cho thấy **PhoBERT** (monolingual Vietnamese) thường đạt F1 cao hơn XLM-RoBERTa trên ViHSD (~0.72–0.80 vs slightly lower). Tuy nhiên, pipeline hiện tại dùng `xlm-roberta-base`.
> 
> **Đề xuất:** Chuyển sang `vinai/phobert-base-v2` làm backbone chính, giữ `xlm-roberta-base` làm baseline comparison. Nếu bạn muốn giữ XLM-R, plan vẫn áp dụng tốt.

> [!IMPORTANT]
> **Scope rebuild:** Plan này rebuild **toàn bộ pipeline từ Phase 0→Phase 8**, tạo mới tất cả source files trong `src/`. Các file config, test, CI/CD, Docker được giữ lại và cập nhật. Bạn có đồng ý?

## Open Questions

> [!IMPORTANT]
> 1. **Backbone model cuối cùng?** `vinai/phobert-base-v2` (recommended) hay `xlm-roberta-base` (hiện tại)?
> 2. **HuggingFace repo ID** để push model mới? Vẫn dùng `quanghs1020/hate-speech-detection` hay tạo repo mới?
> 3. **Gemini API key** có sẵn cho Phase 8 (Agentic Component) không? Hay chỉ build code thôi?
> 4. **Train trên Colab hay local?** Plan hướng Colab-first, nhưng cần confirm.

---

## Proposed Changes

Tổ chức theo **10 component**, mỗi component là một nhóm file liên quan.

---

### Component 1: Infrastructure & Config

Cập nhật configs để phản ánh model mới, thêm các hyperparameter đã chứng minh hiệu quả.

#### [MODIFY] [paths.yaml](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/configs/paths.yaml)
- Cập nhật `model.backbone` sang `vinai/phobert-base-v2`
- Thêm config cho label smoothing, MLM adaptation
- Cập nhật `hf_model_id`

#### [MODIFY] [train.yaml](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/configs/train.yaml)
Thêm các cải tiến training:
```yaml
training:
  # Mới: label smoothing
  label_smoothing: 0.1
  
  # Mới: layer-wise learning rate decay
  layerwise_lr_decay: 0.95
  
  # Mới: warmup steps thay vì ratio
  warmup_ratio: 0.1
  
  # Giữ focal loss nhưng tune gamma
  loss: "focal"
  focal_gamma: 2.0
  
  # Tăng epochs, dùng early stopping
  num_epochs: 10
  early_stopping_patience: 3
  
  # Word segmentation flag
  use_word_segmentation: true
```

#### [MODIFY] [api.yaml](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/configs/api.yaml)
- Cập nhật `hf_repo_id` mới

#### [MODIFY] [model.yaml](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/configs/model.yaml)
- Cập nhật `hf_repo_id` mới

---

### Component 2: Data Acquisition & Download

#### [MODIFY] [download.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/data/download.py)
- Đảm bảo download ViHSD raw CSV từ GitHub
- Thêm validation steps (check column names, unique labels)
- Thêm basic stats logging

---

### Component 3: Preprocessing (CẢI TIẾN LỚN)

Đây là component có nhiều cải tiến nhất so với pipeline cũ.

#### [MODIFY] [preprocessing.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/data/preprocessing.py)

**Cải tiến preprocessing pipeline mới:**

```
Input: raw text
  ↓
Step 1: Unicode NFKC normalization (chuẩn hóa ký tự Unicode)
Step 2: Xoá URL (regex improved)
Step 3: Xoá HTML entities
Step 4: Xoá @mentions  
Step 5: Normalize emoji → text description (tùy chọn)
Step 6: Normalize repeated characters ("đẹpppp" → "đẹp")
Step 7: Normalize Vietnamese teencode (common patterns)
        "ko" → "không", "dc" → "được", "mk" → "mình", etc.
Step 8: Chuẩn hoá khoảng trắng thừa
Step 9: Loại bỏ text quá ngắn (< 3 ký tự)
Step 10: Vietnamese word segmentation (underthesea)
         "tôi đi học" → "tôi đi học" (already correct)
         "sinh viên" → "sinh_viên" (compound word)
         ONLY for PhoBERT (PhoBERT requires word-segmented input)
Step 11: KHÔNG lowercase (model handles casing)
Step 12: KHÔNG strip dấu tiếng Việt
```

**Thêm mới:**
- `normalize_teencode(text)` — Dictionary-based Vietnamese teencode normalization
- `normalize_repeated_chars(text)` — Reduce repeated chars
- `word_segment(text)` — Vietnamese word segmentation via underthesea
- `preprocess_text(text, use_word_segmentation=True)` — Main entry point

---

### Component 4: Data Augmentation (MỚI)

#### [NEW] [augmentation.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/data/augmentation.py)

**Kỹ thuật augmentation:**

1. **EDA (Easy Data Augmentation)** cho Vietnamese:
   - Random Deletion: xóa ngẫu nhiên 10-20% từ
   - Random Swap: hoán đổi vị trí 2 từ ngẫu nhiên
   - Random Insertion: chèn từ ngẫu nhiên từ cùng câu

2. **Back-translation augmentation** (offline, pre-computed):
   - Nếu có sẵn, dùng data đã back-translate
   
3. **Diacritic removal augmentation**:
   - Tạo bản copy không dấu của OFFENSIVE/HATE texts
   - Model học rằng nội dung không dấu vẫn toxic

4. **Teencode augmentation**:
   - Biến text chuẩn thành teencode variant
   - Giúp model robust với ngôn ngữ mạng

5. **Oversampling minority classes**:
   - SMOTE-like approach: oversample HATE và OFFENSIVE
   - Multiplier configurable qua `train.yaml`

#### [MODIFY] [robustness_cases.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/training/robustness_cases.py)
- Giữ nguyên các cases hiện tại (đã tốt)
- Thêm thêm cases cho teencode và emoji patterns

---

### Component 5: EDA Notebook

#### [NEW] [02_eda.ipynb](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/notebooks/02_eda.ipynb)
Hoặc tích hợp vào `train_colab.ipynb`:
- Phân bố nhãn (bar chart)
- Phân bố độ dài text (histogram)
- Top từ/bigram theo nhãn
- Tỉ lệ URL, mention, emoji
- Output: quyết định max_length, class_weights

---

### Component 6: Training (CẢI TIẾN LỚN)

#### [MODIFY] [trainer.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/training/trainer.py)

**Cải tiến training strategy:**

1. **Label Smoothing Loss**:
   ```python
   # Thay vì hard labels [0, 1, 0]
   # Dùng soft labels [0.05, 0.9, 0.05]
   label_smoothing_factor=0.1  # trong TrainingArguments
   ```

2. **Layer-wise Learning Rate Decay (LLRD)**:
   ```python
   # Layer gần input: LR thấp (giữ pretrained knowledge)
   # Layer gần output: LR cao (learn task-specific features)
   # Decay factor: 0.95 per layer
   ```

3. **Focal Loss** (đã có, tune gamma):
   ```python
   focal_gamma: 2.0  # Tăng từ 1.5, focus hơn vào hard examples
   ```

4. **Gradient Accumulation** (nếu cần batch size lớn hơn):
   ```python
   gradient_accumulation_steps: 2  # effective batch = 32
   ```

5. **Mixed Precision (FP16)**:
   - Đã có, giữ nguyên

6. **MLM Domain Adaptation** (Optional, high-impact):
   ```python
   # Trước khi fine-tune classification:
   # 1. Continue pre-training backbone trên ViHSD data với MLM objective
   # 2. Sau đó mới fine-tune classification head
   # → Giúp model học vocab/style của social media Vietnamese
   ```

#### [MODIFY] [train.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/training/train.py)
- Cập nhật CLI để support các config mới

---

### Component 7: Evaluation (CẢI TIẾN)

#### [MODIFY] [evaluate.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/evaluation/evaluate.py)
- Thêm AUC-ROC per class
- Thêm error analysis tự động (4 nhóm lỗi)
- Output `error_analysis.csv`

#### [MODIFY] [metrics.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/evaluation/metrics.py)
- Thêm AUC-ROC computation
- Thêm error type classification

---

### Component 8: Inference & Classifier

#### [MODIFY] [classifier.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/models/classifier.py)
- Cập nhật `preprocess_text()` call để include word segmentation
- Thêm `language` field vào prediction output

#### [MODIFY] [classifier.py (evaluation)](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/evaluation/classifier.py)
- Sync preprocessing với models/classifier.py

---

### Component 9: API & Deployment

#### [MODIFY] [app.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/api/app.py)
- Thêm `/predict` response field: `language`
- Thêm CORS middleware

#### [MODIFY] [schema.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/api/schema.py)
- Thêm `language` field vào PredictionResponse

#### [MODIFY] [Dockerfile](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/Dockerfile)
- Cập nhật model repo ID

---

### Component 10: Agent (Phase 8)

#### [MODIFY] [moderator.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/agent/moderator.py)
- Cập nhật model loading để dùng `HateSpeechClassifier` thống nhất (thay vì `load_hf_artifacts`)
- Tối ưu system prompt
- Thêm Gradio chatbot interface code

---

### Component 11: Notebooks (Colab)

#### [MODIFY] [train_colab.ipynb](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/notebooks/train_colab.ipynb)
- Cập nhật notebook để reflect tất cả thay đổi
- Thêm cells cho: download → EDA → preprocessing → train → evaluate → export → push

---

### Component 12: Documentation

#### [MODIFY] [README.md](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/README.md)
- Cập nhật với model mới, preprocessing steps, evaluation results

#### [MODIFY] [requirements.txt](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/requirements.txt)
- Thêm `underthesea` (nếu chưa có — đã có)
- Kiểm tra version compatibility

---

## Tóm Tắt Cải Tiến vs Pipeline Cũ

| Aspect | Pipeline Cũ | Pipeline Mới |
|---|---|---|
| **Backbone** | xlm-roberta-base | **vinai/phobert-base-v2** (recommended) |
| **Preprocessing** | Basic (URL, mention, whitespace) | **+Unicode norm, +teencode, +repeated chars, +word segmentation** |
| **Augmentation** | Robustness cases + oversampling | **+EDA, +diacritic removal, +teencode variants** |
| **Loss** | Focal (γ=1.5) + manual weights | **Focal (γ=2.0) + label smoothing + LLRD** |
| **Training** | Standard fine-tune | **+Layer-wise LR decay, +gradient accumulation** |
| **Evaluation** | Basic metrics | **+AUC-ROC, +error analysis (4 types), +error_analysis.csv** |
| **Expected F1** | ~0.60–0.68 | **~0.72–0.80** |

---

## Verification Plan

### Automated Tests
1. `python -m compileall -q src tests` — syntax check
2. `pytest tests/` — existing test suite
3. Training script dry-run with 1 epoch on small subset
4. API smoke test: `curl localhost:8000/health`

### Colab Training
1. Chạy `notebooks/train_colab.ipynb` trên Colab (T4 GPU)
2. Monitor training loss & val F1 per epoch
3. So sánh Baseline (TF-IDF + LR) vs Fine-tuned model
4. Export và push to HuggingFace Hub
5. Test inference từ HF Hub

### Manual Verification
1. Chạy `python -m src.evaluation.manual_tests` trên model mới
2. Demo trên Gradio/HF Spaces
3. So sánh F1 mới vs F1 cũ

---

## Execution Order

```
1. Cập nhật configs (paths.yaml, train.yaml, api.yaml, model.yaml)
2. Cải tiến preprocessing.py (teencode, word segmentation, etc.)
3. Tạo augmentation.py mới
4. Cải tiến trainer.py (LLRD, label smoothing)
5. Cải tiến evaluate.py + metrics.py
6. Cập nhật classifier.py (models + evaluation)
7. Cập nhật API (app.py, schema.py)
8. Cập nhật moderator.py (agent)
9. Cập nhật train_colab.ipynb
10. Cập nhật README.md + requirements.txt
11. Chạy tests locally
12. Train trên Colab → evaluate → push model
```
