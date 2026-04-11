"""
Fine-tuning utilities for XLM-RoBERTa on ViHSD.
Provides custom metric computation and Trainer factory.
"""
import numpy as np
from sklearn.metrics import f1_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    f1_macro = f1_score(labels, preds, average="macro")
    f1_per_class = f1_score(labels, preds, average=None).tolist()

    return {
        "f1_macro": round(f1_macro, 4),
        "f1_clean": round(f1_per_class[0], 4),
        "f1_offensive": round(f1_per_class[1], 4),
        "f1_hate": round(f1_per_class[2], 4),
    }