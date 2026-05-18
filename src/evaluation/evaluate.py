from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

try:
    from src.data.preprocessing import normalize_text_label_frame
    from src.evaluation.metrics import compute_classification_metrics, write_json
    from src.models.classifier import HateSpeechClassifier
    from src.models.registry import label2id_from_mapping
    from src.utils.config import load_yaml_config, resolve_path
except ImportError:
    from data.preprocessing import normalize_text_label_frame
    from evaluation.metrics import compute_classification_metrics, write_json
    from models.classifier import HateSpeechClassifier
    from models.registry import label2id_from_mapping
    from utils.config import load_yaml_config, resolve_path


INSUFFICIENT = "Khong du du lieu de xac minh"


def evaluate_from_config(config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    evaluation_cfg = config["evaluation"]
    export_cfg = config["export"]
    model_cfg = config["model"]

    metrics_path = resolve_path(evaluation_cfg["metrics_output_path"])
    report_path = resolve_path(evaluation_cfg["report_output_path"])
    test_path = resolve_path(data_cfg["test_path"])
    final_model_dir = resolve_path(export_cfg["final_model_dir"])

    if not test_path.exists() or not final_model_dir.exists():
        payload = {"status": INSUFFICIENT, "reason": "missing test split or model artifact"}
        write_json(metrics_path, payload)
        _write_report(report_path, payload)
        return payload

    import pandas as pd

    df = pd.read_parquet(test_path) if test_path.suffix.lower() == ".parquet" else pd.read_csv(test_path)
    canonical = normalize_text_label_frame(
        df,
        text_column=data_cfg.get("text_column"),
        label_column=data_cfg.get("label_column"),
    )
    classifier = HateSpeechClassifier(
        model_source="local",
        model_path=str(final_model_dir),
        artifact_dir=export_cfg["artifact_dir"],
        max_length=int(model_cfg.get("max_length", 128)),
    )
    label2id = label2id_from_mapping(classifier.label_mapping)
    predictions = classifier.predict_batch(canonical["text"].tolist())
    y_true = canonical["label"].astype(int).tolist()
    y_pred = [label2id[pred["label"]] for pred in predictions]
    label_names = [classifier.id2label[idx] for idx in sorted(classifier.id2label)]
    labels = list(range(len(label_names)))

    metrics = compute_classification_metrics(y_true, y_pred, labels=labels, label_names=label_names)
    write_json(metrics_path, metrics)
    _write_report(report_path, metrics)
    _write_confusion_matrix(resolve_path(evaluation_cfg["confusion_matrix_path"]), metrics, label_names)
    return metrics


def _write_report(path: Path, metrics: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["# Evaluation Report", ""]
    if metrics.get("status") == INSUFFICIENT:
        lines.extend([INSUFFICIENT, "", f"Reason: {metrics.get('reason', INSUFFICIENT)}"])
    else:
        lines.extend(
            [
                f"- Accuracy: {metrics['accuracy']}",
                f"- Macro F1: {metrics['macro_f1']}",
                f"- Weighted F1: {metrics['weighted_f1']}",
                "",
                "## Per-Class Metrics",
                "",
                "| Class | Precision | Recall | F1 | Support |",
                "|---|---:|---:|---:|---:|",
                *_classification_report_lines(metrics),
                "",
                "## Confusion Matrix",
                "",
                "```text",
                str(metrics["confusion_matrix"]),
                "```",
                "",
                "## Error Counts",
                "",
                *_error_count_lines(metrics),
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _classification_report_lines(metrics: dict[str, Any]) -> list[str]:
    report = metrics.get("classification_report") or {}
    rows = []
    for label, values in report.items():
        if not isinstance(values, dict) or label in {"macro avg", "weighted avg"}:
            continue
        rows.append(
            "| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {support:.0f} |".format(
                label=label,
                precision=float(values.get("precision", 0)),
                recall=float(values.get("recall", 0)),
                f1=float(values.get("f1-score", 0)),
                support=float(values.get("support", 0)),
            )
        )
    return rows


def _error_count_lines(metrics: dict[str, Any]) -> list[str]:
    report = metrics.get("classification_report") or {}
    labels = [
        label
        for label, values in report.items()
        if isinstance(values, dict) and label not in {"macro avg", "weighted avg"}
    ]
    matrix = metrics.get("confusion_matrix") or []
    if not labels or not matrix:
        return ["Khong du du lieu de xac minh"]

    lines = ["| Expected | Missed as CLEAN | Predicted as toxic from CLEAN |", "|---|---:|---:|"]
    clean_idx = labels.index("CLEAN") if "CLEAN" in labels else 0
    for idx, label in enumerate(labels):
        missed_as_clean = int(matrix[idx][clean_idx]) if idx != clean_idx else 0
        clean_to_label = int(matrix[clean_idx][idx]) if idx != clean_idx else 0
        lines.append(f"| {label} | {missed_as_clean} | {clean_to_label} |")
    return lines


def _write_confusion_matrix(path: Path, metrics: dict[str, Any], label_names: list[str]) -> None:
    if "confusion_matrix" not in metrics:
        return
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        path.parent.mkdir(parents=True, exist_ok=True)
        plt.figure(figsize=(6, 5))
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names)
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
    except Exception:
        pass


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the exported hate speech model.")
    parser.add_argument("--config", default="configs/train.yaml")
    args = parser.parse_args()
    metrics = evaluate_from_config(load_yaml_config(args.config))
    print(metrics)


if __name__ == "__main__":
    main()
