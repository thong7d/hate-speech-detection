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


def evaluate_from_config(config: dict[str, Any], hf_repo_id: str | None = None) -> dict[str, Any]:
    data_cfg = config["data"]
    evaluation_cfg = config["evaluation"]
    export_cfg = config["export"]
    model_cfg = config["model"]
    preprocess_cfg = config.get("preprocessing", {})

    metrics_path = resolve_path(evaluation_cfg["metrics_output_path"])
    report_path = resolve_path(evaluation_cfg["report_output_path"])
    test_path = resolve_path(data_cfg["test_path"])
    final_model_dir = resolve_path(export_cfg["final_model_dir"])
    error_analysis_path = resolve_path(evaluation_cfg.get("error_analysis_path", "results/error_analysis.csv"))

    if hf_repo_id is None:
        hf_repo_id = (
            export_cfg.get("hf_repo_id")
            or config.get("api", {}).get("hf_repo_id")
            or __import__("os").environ.get("HF_REPO_ID")
            or __import__("os").environ.get("HF_MODEL_ID")
        )

    has_local_model = final_model_dir.exists()
    has_remote_model = bool(hf_repo_id)

    if not test_path.exists() or (not has_local_model and not has_remote_model):
        payload = {
            "status": INSUFFICIENT,
            "reason": f"missing test split (exists: {test_path.exists()}) or model (local exists: {has_local_model}, remote repo: {hf_repo_id})"
        }
        write_json(metrics_path, payload)
        _write_report(report_path, payload)
        return payload

    import numpy as np
    import pandas as pd

    df = pd.read_parquet(test_path) if test_path.suffix.lower() == ".parquet" else pd.read_csv(test_path)
    use_word_seg = bool(preprocess_cfg.get("use_word_segmentation", True))
    canonical = normalize_text_label_frame(
        df,
        text_column=data_cfg.get("text_column"),
        label_column=data_cfg.get("label_column"),
        use_word_segmentation=use_word_seg,
    )
    
    model_source = "local" if has_local_model else "huggingface"
    classifier = HateSpeechClassifier(
        model_source=model_source,
        model_path=str(final_model_dir),
        hf_repo_id=hf_repo_id,
        artifact_dir=export_cfg["artifact_dir"],
        max_length=int(model_cfg.get("max_length", 128)),
        use_word_segmentation=use_word_seg,
    )
    label2id = label2id_from_mapping(classifier.label_mapping)
    predictions = classifier.predict_batch(canonical["text"].tolist())
    y_true = canonical["label"].astype(int).tolist()
    y_pred = [label2id[pred["label"]] for pred in predictions]
    label_names = [classifier.id2label[idx] for idx in sorted(classifier.id2label)]
    labels = list(range(len(label_names)))

    # Collect probabilities for AUC-ROC
    y_proba = np.array([
        [pred["probabilities"].get(label_names[i], 0.0) for i in range(len(label_names))]
        for pred in predictions
    ])

    metrics = compute_classification_metrics(
        y_true, y_pred,
        labels=labels,
        label_names=label_names,
        y_proba=y_proba,
    )
    write_json(metrics_path, metrics)
    _write_report(report_path, metrics)
    _write_confusion_matrix(resolve_path(evaluation_cfg["confusion_matrix_path"]), metrics, label_names)

    # Write error analysis CSV
    _write_error_analysis_csv(
        error_analysis_path,
        canonical["text"].tolist(),
        y_true,
        y_pred,
        label_names,
        predictions,
    )

    return metrics


def _write_error_analysis_csv(
    path: Path,
    texts: list[str],
    y_true: list[int],
    y_pred: list[int],
    label_names: list[str],
    predictions: list[dict],
) -> None:
    """Write misclassified samples to a CSV file with error type categorization."""
    import pandas as pd

    rows = []
    for i in range(len(texts)):
        if y_true[i] == y_pred[i]:
            continue

        true_label = label_names[y_true[i]] if y_true[i] < len(label_names) else str(y_true[i])
        pred_label = label_names[y_pred[i]] if y_pred[i] < len(label_names) else str(y_pred[i])

        # Categorize error type
        if true_label == "CLEAN" and pred_label in ("OFFENSIVE", "HATE"):
            error_type = "FALSE_POSITIVE"
        elif true_label in ("OFFENSIVE", "HATE") and pred_label == "CLEAN":
            error_type = "FALSE_NEGATIVE"
        elif true_label == "OFFENSIVE" and pred_label == "HATE":
            error_type = "OFFENSIVE_AS_HATE"
        elif true_label == "HATE" and pred_label == "OFFENSIVE":
            error_type = "HATE_AS_OFFENSIVE"
        else:
            error_type = "OTHER"

        rows.append({
            "text": texts[i],
            "true_label": true_label,
            "pred_label": pred_label,
            "confidence": predictions[i].get("confidence", 0.0),
            "error_type": error_type,
            "text_length": len(texts[i]),
        })

    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"[EVAL] Error analysis written: {len(rows)} errors -> {path}")


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
                f"- Critical F1: {metrics.get('critical_f1', 'N/A')}",
                f"- Offensive Priority F1: {metrics.get('offensive_priority_f1', 'N/A')}",
                f"- Balanced Critical F1: {metrics.get('balanced_critical_f1', 'N/A')}",
                f"- AUC-ROC Macro: {metrics.get('auc_roc_macro', 'N/A')}",
                "",
                "## Per-Class Metrics",
                "",
                "| Class | Precision | Recall | F1 | AUC-ROC |",
                "|---|---:|---:|---:|---:|",
                *_classification_report_lines(metrics),
                "",
                "## Confusion Matrix",
                "",
                "```text",
                str(metrics["confusion_matrix"]),
                "```",
                "",
                "## Error Analysis Summary",
                "",
                *_error_analysis_lines(metrics),
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
        if not isinstance(values, dict) or label in {"macro avg", "weighted avg", "accuracy"}:
            continue
        auc = metrics.get(f"auc_roc_{label}", "N/A")
        rows.append(
            "| {label} | {precision:.4f} | {recall:.4f} | {f1:.4f} | {auc} |".format(
                label=label,
                precision=float(values.get("precision", 0)),
                recall=float(values.get("recall", 0)),
                f1=float(values.get("f1-score", 0)),
                auc=auc if auc is not None else "N/A",
            )
        )
    return rows


def _error_analysis_lines(metrics: dict[str, Any]) -> list[str]:
    ea = metrics.get("error_analysis") or {}
    if not ea or ea.get("note"):
        return [ea.get("note", INSUFFICIENT)]
    return [
        f"- Total errors: {ea.get('total_errors', 'N/A')} / {ea.get('total_samples', 'N/A')}",
        f"- Error rate: {ea.get('error_rate', 'N/A')}",
        f"- False positive (CLEAN → toxic): {ea.get('false_positive_clean_as_toxic', 'N/A')}",
        f"- False negative (OFFENSIVE → CLEAN): {ea.get('false_negative_offensive_as_clean', 'N/A')}",
        f"- False negative (HATE → CLEAN): {ea.get('false_negative_hate_as_clean', 'N/A')}",
        f"- OFFENSIVE ↔ HATE confusion: {ea.get('offensive_predicted_as_hate', 0) + ea.get('hate_predicted_as_offensive', 0)}",
    ]


def _error_count_lines(metrics: dict[str, Any]) -> list[str]:
    report = metrics.get("classification_report") or {}
    labels = [
        label
        for label, values in report.items()
        if isinstance(values, dict) and label not in {"macro avg", "weighted avg", "accuracy"}
    ]
    matrix = metrics.get("confusion_matrix") or []
    if not labels or not matrix:
        return [INSUFFICIENT]

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
    parser.add_argument("--hf_repo_id", default=None, help="Hugging Face repo ID to load model from.")
    parser.add_argument("--use_word_segmentation", default=None, help="Override use_word_segmentation (True/False)")
    args = parser.parse_args()

    config = load_yaml_config(args.config)
    if args.use_word_segmentation is not None:
        val = str(args.use_word_segmentation).lower() in ("true", "1", "yes")
        if "preprocessing" not in config:
            config["preprocessing"] = {}
        config["preprocessing"]["use_word_segmentation"] = val

    metrics = evaluate_from_config(config, hf_repo_id=args.hf_repo_id)
    print(metrics)


if __name__ == "__main__":
    main()
