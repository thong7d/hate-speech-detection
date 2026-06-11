# Model Evaluation and Analysis

## 1. Baseline vs. Deep Learning Model Comparison
We evaluate model performance on the ViHSD test set (6,650 samples). The deep learning model (fine-tuned XLM-RoBERTa Base) is benchmarked against a classic machine learning baseline (TF-IDF vectorizer + Logistic Regression).

| Model | Classification Accuracy | Macro F1-Score | AUC-ROC |
| :--- | :---: | :---: | :---: |
| **Baseline (TF-IDF + LR)** | 80.53% | 62.50% | N/A |
| **Main Model (XLM-R Base)** | **86.74%** | **64.61%** | **87.88%** |

### Key Improvements:
- XLM-R yields an **accuracy increase of 6.21%** and a **macro F1 improvement of 2.11%**.
- While TF-IDF fails to catch contextual meaning, the transformer-based XLM-R captures word order, spelling noises, and semantic dependencies.

---

## 2. Detailed Performance Metrics (XLM-R)
The classification report for the fine-tuned XLM-R model is analyzed below:

| Class | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **CLEAN** | 93.16% | 94.07% | **93.62%** | 5,518 |
| **OFFENSIVE** | 43.28% | 39.19% | **41.13%** | 444 |
| **HATE** | 59.62% | 58.58% | **59.09%** | 688 |
| **Macro Average** | 65.35% | 63.95% | **64.61%** | 6,650 |

### Insights:
- The **CLEAN** class performs exceptionally well (F1 = 93.62%), reflecting the high frequency of clean samples in the training set.
- The **OFFENSIVE** class exhibits the lowest performance (F1 = 41.13%). This is due to class imbalance and structural overlap with both HATE and CLEAN categories (e.g. general swearing without targeted hostility).
- The **HATE** class reaches a balanced F1 of 59.09%, showing solid recall (58.58%) for harmful comments.

---

## 3. Confusion Matrix Breakdown
The test set confusion matrix represents the following distributions:

```
                  Predicted CLEAN  Predicted OFFENSIVE  Predicted HATE
Actual CLEAN           5,191               147               180
Actual OFFENSIVE         177               174                93
Actual HATE              204                81               403
```

- **False Positives (CLEAN predicted as HATE)**: 180 cases. This can harm user experience by blocking safe speech.
- **False Negatives (HATE predicted as CLEAN)**: 204 cases. These pose safety risks by letting hate speech pass through.
- **Mitigation**: Borderline predictions (where predictions are uncertain) are intercepted by the **Agentic AI** router step. The LLM acts as a high-precision filter to reduce both false positives and false negatives.

---

## 4. System Trade-offs

### A. Latency vs. Accuracy
- **TF-IDF + LR**: Extremely fast (<1ms), low footprint, but poor accuracy on complex sentences.
- **XLM-R**: Slower (~20-50ms), demands 1.1GB memory, but highly accurate.
- **Decision**: The project utilizes XLM-R as the primary classifier to prioritize moderation safety.

### B. Computational Cost vs. Reasoning Depth
- Calling a 7B LLM (Ollama Qwen2.5) on every query is computationally expensive and slow (~500ms-1s).
- **Decision**: The hybrid architecture uses XLM-R as the gatekeeper. Only predictions with confidence $< 0.65$ (borderline) are routed to Qwen2.5. This limits LLM invocation to roughly **5-10% of total traffic**, balancing compute costs and response speed.
