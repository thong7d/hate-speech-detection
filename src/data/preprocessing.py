"""
Shared preprocessing utilities for ViHSD data and inference.

The training notebooks, API, agent, and manual tests should all call these
helpers instead of carrying separate text-cleaning snippets.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np
import pandas as pd


TEXT_COLUMN_CANDIDATES = ("text", "free_text", "comment", "content")
LABEL_COLUMN_CANDIDATES = ("label", "label_id", "class", "target")


def clean_text(text: object) -> str:
    """Normalize text without changing case or stripping Vietnamese accents."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r"http\S+|www\S+|&[a-z]+;", "", text)
    text = re.sub(r"@\w+", "", text)
    return " ".join(text.split())


def preprocess_text(text: object) -> str:
    """Public alias used by both training and production inference."""
    return clean_text(text)


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
    min_text_length: int = 2,
    text_column: str | None = None,
    label_column: str | None = None,
) -> pd.DataFrame:
    """
    Return a canonical DataFrame with exactly text and label columns.

    This bridges the current notebook drift where preprocessing writes
    label_id but later fine-tuning expects label.
    """
    text_col = text_column or _first_existing_column(df.columns, TEXT_COLUMN_CANDIDATES)
    label_col = label_column or _first_existing_column(df.columns, LABEL_COLUMN_CANDIDATES)

    if text_col is None:
        raise ValueError(f"Could not find a text column. Columns: {list(df.columns)}")
    if label_col is None:
        raise ValueError(f"Could not find a label column. Columns: {list(df.columns)}")

    out = pd.DataFrame(
        {
            "text": df[text_col].map(clean_text),
            "label": pd.to_numeric(df[label_col], errors="raise").astype(int),
        }
    )
    out = out[out["text"].str.len() >= min_text_length].reset_index(drop=True)
    return out


def process_split(
    input_path: str | Path,
    output_path: str | Path,
    *,
    min_text_length: int = 2,
) -> pd.DataFrame:
    """Load a raw CSV split, normalize it, and save canonical parquet output."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Raw split not found: {input_path}")

    df = pd.read_csv(input_path)
    processed = normalize_text_label_frame(df, min_text_length=min_text_length)
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
