"""
Classification metrics helper for evaluation.

Includes per-class metrics, AUC-ROC computation, and error analysis.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    *,
    labels: list[int] | None = None,
    label_names: list[str] | None = None,
    y_proba: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute comprehensive classification metrics.

    Args:
        y_true: True label indices.
        y_pred: Predicted label indices.
        labels: List of label indices (e.g. [0, 1, 2]).
        label_names: Human-readable label names (e.g. ["CLEAN", "OFFENSIVE", "HATE"]).
        y_proba: Predicted probabilities array (n_samples, n_classes) for AUC-ROC.

    Returns:
        Dictionary with all metrics.
    """
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    if label_names is None:
        label_names = [str(i) for i in labels]

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    metrics: dict[str, Any] = {
        "accuracy": round(float(accuracy_score(y_true_arr, y_pred_arr)), 4),
        "macro_f1": round(float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(y_true_arr, y_pred_arr, labels=labels, average="weighted", zero_division=0)), 4),
    }

    # Per-class metrics
    precision_per = precision_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    recall_per = recall_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)
    f1_per = f1_score(y_true_arr, y_pred_arr, labels=labels, average=None, zero_division=0)

    for idx, label in enumerate(label_names):
        metrics[f"precision_{label}"] = round(float(precision_per[idx]), 4)
        metrics[f"recall_{label}"] = round(float(recall_per[idx]), 4)
        metrics[f"f1_{label}"] = round(float(f1_per[idx]), 4)

    # Composite metrics for toxic classes
    if len(labels) >= 3:
        metrics["critical_f1"] = round(float((f1_per[1] + f1_per[2]) / 2), 4)
        metrics["critical_recall"] = round(float((recall_per[1] + recall_per[2]) / 2), 4)
        metrics["offensive_priority_f1"] = round(float((0.7 * f1_per[1]) + (0.3 * f1_per[2])), 4)
        metrics["balanced_critical_f1"] = round(
            float((0.4 * f1_per[1]) + (0.4 * f1_per[2]) + (0.2 * metrics["macro_f1"])), 4
        )

    # Confusion matrix
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=labels)
    metrics["confusion_matrix"] = cm.tolist()

    # Classification report
    report = classification_report(
        y_true_arr, y_pred_arr,
        labels=labels,
        target_names=label_names,
        output_dict=True,
        zero_division=0,
    )
    metrics["classification_report"] = report

    # AUC-ROC (if probabilities available)
    if y_proba is not None and len(labels) > 2:
        try:
            auc_ovr = roc_auc_score(
                y_true_arr, y_proba,
                multi_class="ovr",
                average="macro",
                labels=labels,
            )
            metrics["auc_roc_macro"] = round(float(auc_ovr), 4)

            # Per-class AUC-ROC (one-vs-rest)
            for idx, label in enumerate(label_names):
                try:
                    binary_true = (y_true_arr == labels[idx]).astype(int)
                    auc_class = roc_auc_score(binary_true, y_proba[:, idx])
                    metrics[f"auc_roc_{label}"] = round(float(auc_class), 4)
                except (ValueError, IndexError):
                    metrics[f"auc_roc_{label}"] = None
        except (ValueError, IndexError):
            metrics["auc_roc_macro"] = None

    # Error analysis summary
    metrics["error_analysis"] = _compute_error_analysis(
        y_true_arr, y_pred_arr, labels, label_names
    )

    return metrics


def _compute_error_analysis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[int],
    label_names: list[str],
) -> dict[str, Any]:
    """Analyze prediction errors into 4 categories.

    1. False Positive: CLEAN predicted as OFFENSIVE/HATE
    2. False Negative: OFFENSIVE/HATE predicted as CLEAN
    3. OFFENSIVE vs HATE confusion
    4. Overall error rate by text characteristics
    """
    if len(labels) < 3:
        return {"note": "Error analysis requires 3+ classes"}

    clean_idx, off_idx, hate_idx = labels[0], labels[1], labels[2]

    false_positive_count = int(np.sum((y_true == clean_idx) & (y_pred != clean_idx)))
    false_negative_off = int(np.sum((y_true == off_idx) & (y_pred == clean_idx)))
    false_negative_hate = int(np.sum((y_true == hate_idx) & (y_pred == clean_idx)))
    off_as_hate = int(np.sum((y_true == off_idx) & (y_pred == hate_idx)))
    hate_as_off = int(np.sum((y_true == hate_idx) & (y_pred == off_idx)))

    total_errors = int(np.sum(y_true != y_pred))
    total_samples = len(y_true)

    return {
        "total_errors": total_errors,
        "total_samples": total_samples,
        "error_rate": round(total_errors / total_samples, 4) if total_samples > 0 else 0,
        "false_positive_clean_as_toxic": false_positive_count,
        "false_negative_offensive_as_clean": false_negative_off,
        "false_negative_hate_as_clean": false_negative_hate,
        "offensive_predicted_as_hate": off_as_hate,
        "hate_predicted_as_offensive": hate_as_off,
    }


def write_json(path: str | Path, data: dict[str, Any]) -> None:
    """Write metrics dictionary to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
