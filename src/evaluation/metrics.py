from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def compute_classification_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    *,
    labels: list[int],
    label_names: list[str],
) -> dict:
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    per_class_f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    offensive_idx = label_names.index("OFFENSIVE") if "OFFENSIVE" in label_names else 1
    hate_idx = label_names.index("HATE") if "HATE" in label_names else 2
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "recall_macro": round(recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "macro_f1": round(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0), 4),
        "weighted_f1": round(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0), 4),
        "critical_f1": round(float((per_class_f1[offensive_idx] + per_class_f1[hate_idx]) / 2), 4),
        "critical_recall": round(float((per_class_recall[offensive_idx] + per_class_recall[hate_idx]) / 2), 4),
        "offensive_priority_f1": round(float((0.7 * per_class_f1[offensive_idx]) + (0.3 * per_class_f1[hate_idx])), 4),
        "offensive_priority_recall": round(
            float((0.7 * per_class_recall[offensive_idx]) + (0.3 * per_class_recall[hate_idx])), 4
        ),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def write_json(path: str | Path, payload: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
