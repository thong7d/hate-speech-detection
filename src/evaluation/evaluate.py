"""
Standardized evaluation utilities for ViHSD models.
"""
import json
import os
import shutil
import tempfile
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

LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]

def compute_full_metrics(y_true, y_pred, y_prob=None):
    labels = list(range(len(LABEL_NAMES)))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=LABEL_NAMES,
        output_dict=True,
        zero_division=0,
    )
    cm = confusion_matrix(y_true, y_pred, labels=labels).tolist()
    f1_macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    precision_macro = precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    recall_macro = recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0).tolist()

    result = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "macro_f1": round(f1_macro, 4),
        "weighted_f1": round(f1_weighted, 4),
        "recall_hate_speech": round(report["HATE"]["recall"], 4),
        "f1_per_class": {name: round(f1, 4) for name, f1 in zip(LABEL_NAMES, f1_per_class)},
        "classification_report": report,
        "confusion_matrix": cm,
    }

    if y_prob is not None:
        try:
            auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
            result["auc_roc_macro"] = round(auc, 4)
        except ValueError:
            result["auc_roc_macro"] = None

    return result

def save_metrics_atomic(metrics: dict, filepath: str):
    dir_path = os.path.dirname(filepath)
    os.makedirs(dir_path, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix=".json")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        # BẢN VÁ: Dùng shutil.move thay cho os.replace để tránh lỗi Cross-device link trên Drive
        shutil.move(tmp_path, filepath, copy_function=shutil.copy2)
    except Exception:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        raise
