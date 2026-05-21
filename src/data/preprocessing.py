"""
Shared preprocessing utilities for ViHSD data and inference.

The training notebooks, API, agent, and manual tests should all call these
helpers instead of carrying separate text-cleaning snippets.

Pipeline order (invariant):
  1. Unicode NFKC normalization
  2. Remove URLs
  3. Remove HTML entities
  4. Remove @mentions
  5. Normalize repeated characters ("đẹpppp" -> "đẹp")
  6. Normalize Vietnamese teencode ("ko" -> "không", etc.)
  7. Normalize whitespace
  8. Filter short texts (< min_length characters)
  9. Vietnamese word segmentation (for PhoBERT)
  10. Do NOT lowercase
  11. Do NOT strip Vietnamese diacritics
"""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


TEXT_COLUMN_CANDIDATES = ("text", "free_text", "comment", "content")
LABEL_COLUMN_CANDIDATES = ("label", "label_id", "class", "target")


# ============================================================
# Vietnamese teencode dictionary
# ============================================================
TEENCODE_MAP: dict[str, str] = {
    # Common abbreviations
    "ko": "không",
    "k": "không",
    "kh": "không",
    "khg": "không",
    "hk": "không",
    "hem": "không",
    "dc": "được",
    "dk": "được",
    "đc": "được",
    "mk": "mình",
    "mik": "mình",
    "m": "mình",
    "b": "bạn",
    "bn": "bạn",
    "bjo": "bây giờ",
    "bh": "bây giờ",
    "bik": "biết",
    "bt": "bình thường",
    "bth": "bình thường",
    "cx": "cũng",
    "cg": "cũng",
    "ck": "chồng",
    "vk": "vợ",
    "nc": "nói chuyện",
    "nch": "nói chuyện",
    "ns": "nói",
    "ntn": "như thế nào",
    "nv": "nhiệm vụ",
    "nt": "nhắn tin",
    "r": "rồi",
    "rr": "rồi",
    "vs": "với",
    "ms": "mới",
    "mn": "mọi người",
    "mng": "mọi người",
    "ng": "người",
    "ngta": "người ta",
    "ngt": "người ta",
    "j": "gì",
    "z": "gì",
    "g": "gì",
    "gì": "gì",
    "trc": "trước",
    "trl": "trả lời",
    "ib": "inbox",
    "đt": "điện thoại",
    "dt": "điện thoại",
    "ae": "anh em",
    "a": "anh",
    "e": "em",
    "vc": "vợ chồng",
    "gđ": "gia đình",
    "gd": "gia đình",
    "th": "thầy",
    "tks": "thanks",
    "thanks": "cảm ơn",
    "tks": "cảm ơn",
    "ok": "được",
    "okie": "được",
    "oke": "được",
    "nma": "nhưng mà",
    "nhma": "nhưng mà",
    "lm": "làm",
    "dn": "doanh nghiệp",
    "sg": "sài gòn",
    "hn": "hà nội",
    "hcm": "hồ chí minh",
    "đhs": "đại học sư",
    "sv": "sinh viên",
    "hs": "học sinh",
    "gv": "giáo viên",
    "cty": "công ty",
    "cs": "cơ sở",
    "tk": "tài khoản",
    "fb": "facebook",
    "acc": "tài khoản",
    "ad": "admin",
    "qc": "quảng cáo",
    "rep": "trả lời",
    "cmt": "bình luận",
    "cmn": "con mẹ nó",
    "vcl": "vãi cả lồn",
    "vl": "vãi lồn",
    "vkl": "vãi cả lồn",
    "dm": "đụ má",
    "đm": "đụ má",
    "dmm": "đụ má mày",
    "clm": "cái lồn mày",
    "cl": "cái lồn",
    "đkm": "đụ kỳ má",
    "cc": "cặc",
    "wtf": "cái quái gì",
    "omg": "trời ơi",
    "lol": "haha",
    "bruh": "ôi trời",
    "ngu": "ngu",
    "nguu": "ngu",
    "nguuu": "ngu",
    "qá": "quá",
    "wa": "quá",
    "wá": "quá",
    "trl": "trả lời",
    "đag": "đang",
    "dag": "đang",
    "dg": "đang",
    "hiu": "hiểu",
    "hỉu": "hiểu",
    "hjhj": "hehe",
    "hj": "hehe",
    "lun": "luôn",
    "lun": "luôn",
    "lg": "lại",
    "lg": "lại",
    "dv": "diễn viên",
    "kq": "kết quả",
}

# Compile teencode pattern once (match whole words only, case-insensitive)
_TEENCODE_PATTERN = re.compile(
    r"\b(" + "|".join(re.escape(k) for k in sorted(TEENCODE_MAP.keys(), key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


# ============================================================
# Core preprocessing functions
# ============================================================

def unicode_normalize(text: str) -> str:
    """Apply Unicode NFKC normalization to standardize characters."""
    return unicodedata.normalize("NFKC", text)


def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"https?://\S+|www\.\S+", "", text)


def remove_html_entities(text: str) -> str:
    """Remove HTML entities like &amp; &lt; etc."""
    return re.sub(r"&[a-zA-Z]+;", "", text)


def remove_mentions(text: str) -> str:
    """Remove @mentions from text."""
    return re.sub(r"@\w+", "", text)


def normalize_repeated_chars(text: str) -> str:
    """Reduce repeated characters to at most 2.
    e.g. "đẹpppp" -> "đẹpp", "hahahaha" -> "haha"
    """
    return re.sub(r"(.)\1{2,}", r"\1\1", text)


def normalize_teencode(text: str) -> str:
    """Normalize Vietnamese teencode/internet slang to standard Vietnamese."""

    def _replace(match: re.Match) -> str:
        word = match.group(0).lower()
        return TEENCODE_MAP.get(word, word)

    return _TEENCODE_PATTERN.sub(_replace, text)


def normalize_whitespace(text: str) -> str:
    """Normalize excessive whitespace to single spaces and strip."""
    return " ".join(text.split())


def word_segment(text: str) -> str:
    """Apply Vietnamese word segmentation using underthesea.

    PhoBERT requires word-segmented input where compound words
    are joined with underscores: "sinh viên" -> "sinh_viên".
    """
    try:
        from underthesea import word_tokenize
        return word_tokenize(text, format="text")
    except ImportError:
        # Fallback: return text as-is if underthesea not installed
        return text


def clean_text(
    text: object,
    *,
    use_word_segmentation: bool = False,
    use_teencode_normalization: bool = True,
    use_repeated_char_normalization: bool = True,
    use_unicode_normalization: bool = True,
) -> str:
    """Full preprocessing pipeline.

    Args:
        text: Raw input text.
        use_word_segmentation: Apply Vietnamese word segmentation (for PhoBERT).
        use_teencode_normalization: Normalize Vietnamese internet slang.
        use_repeated_char_normalization: Reduce repeated characters.
        use_unicode_normalization: Apply Unicode NFKC normalization.

    Returns:
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Unicode normalization
    if use_unicode_normalization:
        text = unicode_normalize(text)

    # Step 2: Remove URLs
    text = remove_urls(text)

    # Step 3: Remove HTML entities
    text = remove_html_entities(text)

    # Step 4: Remove @mentions
    text = remove_mentions(text)

    # Step 5: Normalize repeated characters
    if use_repeated_char_normalization:
        text = normalize_repeated_chars(text)

    # Step 6: Normalize teencode
    if use_teencode_normalization:
        text = normalize_teencode(text)

    # Step 7: Normalize whitespace
    text = normalize_whitespace(text)

    # Step 8: Word segmentation (for PhoBERT)
    if use_word_segmentation:
        text = word_segment(text)

    return text


def preprocess_text(text: object, *, use_word_segmentation: bool = True) -> str:
    """Public alias used by both training and production inference.

    Defaults to word segmentation ON for PhoBERT compatibility.
    """
    return clean_text(
        text,
        use_word_segmentation=use_word_segmentation,
        use_teencode_normalization=True,
        use_repeated_char_normalization=True,
        use_unicode_normalization=True,
    )


# ============================================================
# DataFrame-level utilities
# ============================================================

def resolve_project_path(template_path: str, project_root: str | Path = ".") -> Path:
    """Resolve paths that contain the literal {project_root} placeholder."""
    return Path(str(template_path).replace("{project_root}", str(project_root))).resolve()


def _first_existing_column(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    existing = set(columns)
    for candidate in candidates:
        if candidate in existing:
            return candidate
    return None


def normalize_text_label_frame(
    df: pd.DataFrame,
    *,
    min_text_length: int = 3,
    text_column: str | None = None,
    label_column: str | None = None,
    use_word_segmentation: bool = True,
) -> pd.DataFrame:
    """Return a canonical DataFrame with exactly text and label columns.

    Applies the full preprocessing pipeline to the text column.
    """
    text_col = text_column or _first_existing_column(df.columns, TEXT_COLUMN_CANDIDATES)
    label_col = label_column or _first_existing_column(df.columns, LABEL_COLUMN_CANDIDATES)

    if text_col is None:
        raise ValueError(f"Could not find a text column. Columns: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Could not find a label column. Columns: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "text": df[text_col].map(
                lambda t: clean_text(t, use_word_segmentation=use_word_segmentation)
            ),
            "label": pd.to_numeric(df[label_col], errors="raise").astype(int),
        }
    )
    out = out[out["text"].str.len() >= min_text_length].reset_index(drop=True)
    return out


def process_split(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_text_length: int = 3,
    use_word_segmentation: bool = True,
) -> pd.DataFrame:
    """Load a raw CSV split, normalize it, and save canonical parquet output."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Raw split not found: {input_path}")

    df = pd.read_csv(input_path)
    processed = normalize_text_label_frame(
        df,
        min_text_length=min_text_length,
        use_word_segmentation=use_word_segmentation,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    processed.to_parquet(output_path, index=False)
    return processed


def compute_balanced_class_weights(labels: Iterable[int], num_labels: int) -> dict[int, float]:
    """Compute balanced class weights in label-index order from training labels only."""
    from sklearn.utils.class_weight import compute_class_weight

    y = np.asarray(list(labels), dtype=int)
    classes = np.arange(num_labels, dtype=int)
    missing = sorted(set(classes) - set(np.unique(y)))
    if missing:
        raise ValueError(f"Cannot compute class weights; missing labels in train split: {missing}")
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(label): float(weight) for label, weight in zip(classes, weights)}


def save_class_weights(weights: Mapping[int, float], output_path: str | Path) -> None:
    """Persist class weights as a JSON object keyed by label index."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump({str(int(k)): float(v) for k, v in weights.items()}, f, indent=2)
