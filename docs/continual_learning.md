# Continual Learning & Monitoring Strategy

This document describes the sequential Continual Learning (CL) pipeline and online monitoring system designed to maintain the stable performance of the XLM-R Base hate speech classifier in real-world production environments. The design follows MLOps production standards and addresses system risks at the concurrency, Windows OS, and memory management levels.

---

## 1. Continual Learning Pipeline via Google Colab (MLOps Cycle)

To ensure reproducibility and high stability, the CL pipeline follows a **Hybrid MLOps Architecture (Colab GPU + Local CPU Serving)**:

```
[Git Branch: cl-pipeline] ────────────► Git clone directly on Colab GPU
[Google Drive: /MyDrive/CL_data/] ───► Mounted directly as raw data source
                                               │
                                               ▼
                              [Google Colab (GPU Environment)]
                              - Runs CL (WeightedSampler, Soft targets, device="cuda")
                              - Packages output into CL_output.zip and uploads to Drive
                              - Creates an empty CL_output.zip.success marker file
                                               │
                                               ▼
[Local CPU Machine] ◄──────────────── Detects the .success marker file
- Runs deploy_cl_model.py
- Extracts into model_staging_<timestamp>/ (avoids static I/O buffer contention)
- Runs load test to verify model integrity (guards against environment mismatch)
- Windows Atomic Operation: Rename directory using os.replace() + 3-retry loop
  - On complete failure → dynamic staging dir is isolated, does not block next cycle
- Sets is_reloading = True flag before the reload_model lock (prevents predict deadlock)
- Actively frees old RAM/VRAM (self.model = None, gc.collect())
- Thread-safe via threading.Lock
- Loads new weights & updates active_version.json (safe Hot-Reloading)
```

### Step 1: Data Merging & CL Dataset Creation (Mixed Validation)
1. **Inner Merge**: Join `02_train_text.csv` and `03_train_label.csv` from VLSP-2019 by the `id` column. Map labels `0` → `CLEAN`, `1` → `OFFENSIVE`, `2` → `HATE`. Add `source` field = `"vlsp"`.
2. **Rehearsal Buffer**: Stratified sample of 2,500 rows from the original ViHSD training split (`data/processed/train.parquet`). Add `source` field = `"vihsd"`.
3. **Training Set Assembly**: Concatenate VLSP data and the Rehearsal Buffer into `data/processed/continual_train.parquet`.
4. **Resolving Validation Bias (Mixed Validation Set)**: Create a 50/50 stratified mix of `vlsp_dev.parquet` (500 rows) and the original ViHSD dev split `data/processed/dev.parquet` (500 rows) → `continual_dev.parquet`.

### Step 2: DataLoader with `WeightedRandomSampler` to Counter Rehearsal Imbalance
Uses PyTorch `WeightedRandomSampler` to control the mixing ratio at the **per-batch training level**:
- ViHSD sample weight = $\frac{0.25}{N_{\text{vihsd\_train}}}$
- VLSP sample weight = $\frac{0.75}{N_{\text{vlsp\_train}}}$

Each batch of 32 samples will contain on average **8 ViHSD samples (25%) and 24 VLSP samples (75%)**, ensuring memory gradients are continuously and uniformly updated at each optimization step.

### Step 3: Resolving Focal Loss / Label Smoothing Mathematical Conflict
When a CL cycle activates Label Smoothing ($\alpha=0.15$), the system automatically switches the loss function from Focal Loss to **standard Cross-Entropy Loss with soft target support** to ensure stable and accurate gradient convergence.

### Step 4: Training and Model Packaging on Colab
1. Train incrementally on Colab GPU for **3 epochs** with a low learning rate $LR = 1.0 \times 10^{-5}$.
2. Run the **Gatekeeper** evaluation on the original ViHSD test set. Requires Macro F1 to not degrade by more than 1% compared to the previous baseline (F1 $\ge 64.61\%$).
3. Run temperature ECE recalibration on the mixed validation set to update the temperature $T$ in `metadata.json`.
4. Compress the new version directory into `CL_output.zip` and copy to Google Drive.
5. **`.success` file mechanism (prevents reading an incomplete ZIP)**: After 100% upload, Colab creates an empty file `CL_output.zip.success` in the same Drive directory. The local process only activates upon detecting this file.

### Step 5: Atomic Deployment and Hot-Reloading on Local CPU
Extraction and loading are automated via [`src/export/deploy_cl_model.py`](../src/export/deploy_cl_model.py):
1. **Dynamic Staging Directory**: Extract `CL_output.zip` into a timestamp-named staging directory: `artifacts/hate_speech_model/model_staging_<timestamp>/`.
2. **Load Integrity Test**: Load the new model into RAM and run a dry-run inference on a sample sentence. Halt on any error.
3. **Windows Atomic Replacement via `os.replace`**: Use `os.replace()` for a strong forced overwrite, wrapped in a 3-retry loop with 1-second delays to handle temporary Windows file locks.
4. **RAM-safe Hot-Reloading & Predict Deadlock Prevention**:
   - **Predict Deadlock Prevention**: Set `self.is_reloading = True` **before** the `with self.lock:` block, so concurrent `predict()` calls immediately fallback to a safe result (`"CLEAN"`, confidence `1.0`, reload note) without blocking.
   - **RAM Cleanup**: Inside the lock block, unlink the old model (`self.model = None`) and call Python's garbage collector (`gc.collect()`) before loading new weights.

---

## 2. Real-World Data Collection & Feedback Loop

New data for subsequent CL cycles is accumulated continuously via:
1. **Borderline Audit Logs**: Grey-area samples routed to the LLM Agent are saved in `scratch/system_audit_log.jsonl` as automatic training material.
2. **HITL (Human-in-the-Loop)**: The Streamlit interface allows administrators to review and correct mislabeled samples before exporting rehearsal data.
3. **Active Learning**: Prioritize high-uncertainty samples for additional manual annotation.

---

## 3. Monitoring Metrics & Drift Detection

| Monitoring Metric | Formula / Definition | Alert Threshold & Behavior |
| :--- | :--- | :--- |
| **Routing Ratio (RR)** | Fraction of inputs forwarded to the LLM Agent. | If RR exceeds **30%** continuously for 7 days → Covariate Shift warning. |
| **Prediction Label Drift** | Daily output label distribution shift. | If any label (e.g. HATE) fluctuates by more than **15%** → User behavior change or label drift warning. |
| **Mean Confidence Score** | Average of all prediction confidence values. | If consistently below **0.75** → Model capability degradation warning on new data domain. |

---

## 4. Drift Risks & Mitigation Strategies

| Drift Type | Risk | Mitigation |
| :--- | :--- | :--- |
| **Concept Drift** | Slang semantics evolve over time | Periodically update the teencode normalization dictionary and harvest LLM Agent analysis samples |
| **Covariate Shift** | New topics emerge suddenly | Incremental training with a Rehearsal Buffer mixing original data to retain prior knowledge |
| **Label Noise** | Inconsistent labeling standards across datasets | Apply **Label Smoothing ($\alpha=0.15$)** and enforce strict Gatekeeper evaluation before deployment |

---

## 5. Empirical CL Results

| Round | Training Data | Validation Set | Macro F1 | OFFENSIVE F1 | HATE F1 |
| :--- | :--- | :--- | :---: | :---: | :---: |
| **Baseline (ViHSD only)** | ViHSD train (18,638) | ViHSD dev | **64.61%** | 41.13% | 59.09% |
| **CL Round 1** | VLSP Part 1 + ViHSD rehearsal (4,000) | Mixed (ViHSD 500 + VLSP 500) | See `results/cl_step1_metrics.json` | — | — |
| **CL Round 2** | VLSP Part 2 + ViHSD rehearsal (4,000) | Mixed (ViHSD 500 + VLSP 500) | See `results/cl_step2_metrics.json` | — | — |

> **Note on performance drop**: A Macro F1 decline after CL is expected and acceptable. The VLSP-2019 dataset uses a different annotation scheme and domain (political commentary vs. ViHSD's social media comments), causing distributional shift. The rehearsal buffer and Gatekeeper threshold mitigate catastrophic forgetting while allowing the model to adapt to the new domain.
