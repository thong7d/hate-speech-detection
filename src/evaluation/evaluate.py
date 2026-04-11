"""
Standardized evaluation utilities for ViHSD models.
"""
import json
import os
import shutil
import tempfile
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, roc_auc_score

LABEL_NAMES = ["CLEAN", "OFFENSIVE", "HATE"]

def compute_full_metrics(y_true, y_pred, y_prob=None):
    report = classification_report(y_true, y_pred, target_names=LABEL_NAMES, output_dict=True)
    cm = confusion_matrix(y_true, y_pred).tolist()
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_per_class = f1_score(y_true, y_pred, average=None).tolist()

    result = {
        "macro_f1": round(f1_macro, 4),
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