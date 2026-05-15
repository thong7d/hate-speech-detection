"""
Fine-tuning utilities for XLM-RoBERTa on ViHSD.
Provides custom metric computation and Trainer factory.
"""
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)

    label_ids = [0, 1, 2]
    f1_macro = f1_score(labels, preds, labels=label_ids, average="macro", zero_division=0)
    f1_weighted = f1_score(labels, preds, labels=label_ids, average="weighted", zero_division=0)
    precision_macro = precision_score(labels, preds, labels=label_ids, average="macro", zero_division=0)
    recall_macro = recall_score(labels, preds, labels=label_ids, average="macro", zero_division=0)
    f1_per_class = f1_score(labels, preds, labels=label_ids, average=None, zero_division=0).tolist()

    return {
        "accuracy": round(accuracy_score(labels, preds), 4),
        "precision_macro": round(precision_macro, 4),
        "recall_macro": round(recall_macro, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "f1_clean": round(f1_per_class[0], 4),
        "f1_offensive": round(f1_per_class[1], 4),
        "f1_hate": round(f1_per_class[2], 4),
    }
