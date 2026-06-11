# Data Description Document

## 1. Data Sourcing and Overview
The project utilizes the **ViHSD (Vietnamese Hate Speech Detection)** dataset, a standard academic dataset for Vietnamese social media text classification.
- **Classification Categories**: 
  - `CLEAN` (0): Text contains no offensive or hateful language.
  - `OFFENSIVE` (1): Text contains rude, uncivil, or vulgar language but does not target groups.
  - `HATE` (2): Text contains hate speech targeting specific groups (ethnicity, religion, gender, etc.).

---

## 2. Dataset Split Statistics
The dataset is split into three subsets: Train, Dev (Validation), and Test.

| Dataset Split | File Path | Format | Record Count | File Size | Label Distribution (CLEAN / OFFENSIVE / HATE) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Train** | `data/raw/vihsd/train.csv` | CSV | 24,118 | 1.6 MB | 83% / 7% / 10% |
| **Dev** | `data/raw/vihsd/dev.csv` | CSV | 2,680 | 177 KB | 83% / 7% / 10% |
| **Test** | `data/raw/vihsd/test.csv` | CSV | 6,650 | 442 KB | 83% / 7% / 10% |

- **Total records**: 33,448 rows.
- **Parquet conversions**: Data is optimized and stored in `.parquet` format under `data/processed/` for faster loading in the training pipeline.

---

## 3. Data Schema
All splits share the same schema:
- `text` (string): The raw comments collected from social media.
- `label` (integer): The annotated ground-truth (0 = CLEAN, 1 = OFFENSIVE, 2 = HATE).

---

## 4. Preprocessing and Cleaning Pipeline
Implemented in `src/data/preprocessing.py`, the cleaning process prepares raw Vietnamese social text:
1. **Lowercasing & Whitespace Stripping**: Standardizes text representation.
2. **Special Character and Link Removal**: Removes URLs, email addresses, and excessive punctuation that do not contribute to classification.
3. **Emoji Normalization**: Emojis are parsed and translated or stripped to prevent tokenization issues.
4. **VnCoreNLP Word Segmentation (Optional)**: Employs word segmentation to group syllables into compound words (e.g., `học_sinh` instead of `học` `sinh`). In the current production deployment, word segmentation is disabled on the XLM-R classifier to reduce inference latency, which remains competitive in accuracy.

---

## 5. Dataset Biases and Limitations
- **Class Imbalance**: Over 83% of the dataset consists of CLEAN samples. The minority classes (OFFENSIVE at 7% and HATE at 10%) are underrepresented. This causes classifiers to be biased towards predicting CLEAN. Mitigation includes applying custom threshold cascades during inference and using weighted loss functions during training.
- **Dialect Bias**: The majority of annotations and social media slang reflect Northern dialects. Sarcasm or regional slang from Southern/Central dialects may exhibit higher error rates.
- **Annotation Subjectivity**: Distinguishing between OFFENSIVE and HATE contains human bias. A comment might be annotated as HATE by one annotator and OFFENSIVE by another, introducing noise.
