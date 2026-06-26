"""
Temperature Scaling Calibration for Vietnamese Hate Speech Detection Models.

Implements:
  - Weighted Scalar Temperature Scaling (single T, preserves argmax)
  - Weighted NLL loss with class weights for imbalanced ViHSD dataset (~83% CLEAN)
  - L2 Regularization on temperature parameters (prevents divergence)
  - ECE (Expected Calibration Error) computation before/after calibration
  - Reliability diagram generation for visual inspection

Mathematical Foundation:
  - Scalar: p_calibrated = softmax(z / T), T > 0
  - Loss = WeightedNLL(p_calibrated, y) + lambda * (T - 1)^2
  - ECE = sum_{m=1}^{M} (|B_m|/N) * |acc(B_m) - conf(B_m)|
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize

try:
    from src.models.registry import DEFAULT_LABELS
except ImportError:
    from models.registry import DEFAULT_LABELS


# ============================================================
# ECE Computation
# ============================================================

def compute_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).

    ECE = sum_{m=1}^{M} (|B_m| / N) * |acc(B_m) - conf(B_m)|

    Args:
        y_true: Ground truth labels, shape (N,).
        y_proba: Predicted probabilities, shape (N, C).
        n_bins: Number of calibration bins.

    Returns:
        ECE value (float, lower is better).
    """
    y_pred = np.argmax(y_proba, axis=1)
    confidences = np.max(y_proba, axis=1)
    accuracies = (y_pred == y_true).astype(float)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_samples = len(y_true)

    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
        prop_in_bin = in_bin.sum() / n_samples
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += prop_in_bin * abs(avg_accuracy - avg_confidence)

    return float(ece)


def compute_classwise_ece(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    n_bins: int = 15,
    label_names: list[str] | None = None,
) -> dict[str, float]:
    """Compute per-class ECE for each label."""
    if label_names is None:
        label_names = DEFAULT_LABELS
    n_classes = y_proba.shape[1]
    classwise_ece = {}

    for c in range(n_classes):
        # Binary: true label == c vs not
        binary_true = (y_true == c).astype(float)
        binary_conf = y_proba[:, c]

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        ece_c = 0.0
        n_samples = len(y_true)

        for i in range(n_bins):
            in_bin = (binary_conf > bin_boundaries[i]) & (binary_conf <= bin_boundaries[i + 1])
            prop_in_bin = in_bin.sum() / n_samples
            if prop_in_bin > 0:
                avg_conf = binary_conf[in_bin].mean()
                avg_acc = binary_true[in_bin].mean()
                ece_c += prop_in_bin * abs(avg_acc - avg_conf)

        class_name = label_names[c] if c < len(label_names) else str(c)
        classwise_ece[f"ece_{class_name}"] = round(float(ece_c), 6)

    return classwise_ece


# ============================================================
# Weighted NLL Loss + L2 Regularization
# ============================================================

def weighted_nll_with_l2(
    temperature: float,
    logits: np.ndarray,
    labels: np.ndarray,
    class_weights: np.ndarray,
    l2_lambda: float = 0.01,
) -> float:
    """
    Compute Weighted Negative Log-Likelihood with L2 regularization.

    Loss = -sum_i w_{y_i} * log(softmax(z_i / T)_{y_i}) + lambda * (T - 1)^2

    The L2 term (T-1)^2 regularizes T towards 1.0 (the uncalibrated baseline),
    preventing the optimization from finding extreme temperature values.

    Args:
        temperature: Scalar temperature T > 0.
        logits: Raw logits, shape (N, C).
        labels: Ground truth label indices, shape (N,).
        class_weights: Per-class weights, shape (C,).
        l2_lambda: L2 regularization strength.

    Returns:
        Loss value (float).
    """
    T = max(temperature, 1e-6)  # Prevent division by zero
    scaled_logits = logits / T

    # Numerically stable log-softmax
    log_probs = scaled_logits - np.log(np.sum(np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True)), axis=1, keepdims=True)) - scaled_logits.max(axis=1, keepdims=True)

    # Weighted NLL
    n_samples = len(labels)
    sample_weights = class_weights[labels]
    nll = -np.sum(sample_weights * log_probs[np.arange(n_samples), labels]) / n_samples

    # L2 regularization: penalize deviation from T=1
    l2_penalty = l2_lambda * (T - 1.0) ** 2

    return float(nll + l2_penalty)


def weighted_nll_gradient(
    temperature: float,
    logits: np.ndarray,
    labels: np.ndarray,
    class_weights: np.ndarray,
    l2_lambda: float = 0.01,
) -> np.ndarray:
    """
    Compute gradient of Weighted NLL + L2 w.r.t. temperature T.

    d(Loss)/dT = (1/N) * sum_i w_{y_i} * sum_c (p_c * z_c - z_{y_i}) / T^2 + 2*lambda*(T-1)
    """
    T = max(temperature, 1e-6)
    scaled_logits = logits / T

    # Stable softmax
    exp_logits = np.exp(scaled_logits - scaled_logits.max(axis=1, keepdims=True))
    probs = exp_logits / exp_logits.sum(axis=1, keepdims=True)

    n_samples = len(labels)
    sample_weights = class_weights[labels]

    # d(NLL)/dT for each sample
    # = w_i * (sum_c p_c * z_c - z_{y_i}) / T^2
    expected_logit = np.sum(probs * logits, axis=1)  # sum_c p_c * z_c
    true_logit = logits[np.arange(n_samples), labels]  # z_{y_i}
    grad_per_sample = sample_weights * (expected_logit - true_logit) / (T ** 2)
    grad_nll = -np.mean(grad_per_sample)

    # d(L2)/dT = 2 * lambda * (T - 1)
    grad_l2 = 2.0 * l2_lambda * (T - 1.0)

    return np.array([grad_nll + grad_l2])


# ============================================================
# Calibration Pipeline
# ============================================================

def calibrate_temperature(
    logits: np.ndarray,
    labels: np.ndarray,
    *,
    class_weights: np.ndarray | None = None,
    l2_lambda: float = 0.01,
    init_temperature: float = 1.5,
    method: str = "L-BFGS-B",
) -> dict[str, Any]:
    """
    Learn optimal scalar temperature T on a validation set.

    Uses L-BFGS-B optimization to minimize Weighted NLL + L2 regularization.

    Args:
        logits: Raw model logits on validation set, shape (N, C).
        labels: Ground truth labels, shape (N,).
        class_weights: Per-class weights, shape (C,). If None, computed as balanced.
        l2_lambda: L2 regularization strength.
        init_temperature: Starting temperature value.
        method: Scipy optimization method.

    Returns:
        Dict with optimal_temperature, ece_before, ece_after, and optimization details.
    """
    n_classes = logits.shape[1]

    # Compute balanced class weights if not provided
    if class_weights is None:
        from sklearn.utils.class_weight import compute_class_weight
        classes = np.arange(n_classes)
        class_weights = compute_class_weight("balanced", classes=classes, y=labels)

    class_weights = np.asarray(class_weights, dtype=np.float64)

    # ECE before calibration
    probs_before = _softmax_np(logits)
    ece_before = compute_ece(labels, probs_before)
    classwise_ece_before = compute_classwise_ece(labels, probs_before)

    # Optimize temperature
    start_time = time.time()
    result = minimize(
        fun=weighted_nll_with_l2,
        x0=[init_temperature],
        args=(logits, labels, class_weights, l2_lambda),
        jac=weighted_nll_gradient,
        method=method,
        bounds=[(0.01, 20.0)],
        options={"maxiter": 200, "ftol": 1e-12},
    )
    elapsed = time.time() - start_time

    optimal_T = float(result.x[0])

    # ECE after calibration
    probs_after = _softmax_np(logits / optimal_T)
    ece_after = compute_ece(labels, probs_after)
    classwise_ece_after = compute_classwise_ece(labels, probs_after)

    # Verify argmax preservation (mandatory for scalar T)
    preds_before = np.argmax(probs_before, axis=1)
    preds_after = np.argmax(probs_after, axis=1)
    argmax_preserved = bool(np.all(preds_before == preds_after))

    return {
        "optimal_temperature": round(optimal_T, 6),
        "ece_before": round(ece_before, 6),
        "ece_after": round(ece_after, 6),
        "ece_reduction_pct": round((ece_before - ece_after) / max(ece_before, 1e-10) * 100, 2),
        "classwise_ece_before": classwise_ece_before,
        "classwise_ece_after": classwise_ece_after,
        "argmax_preserved": argmax_preserved,
        "optimization_success": bool(result.success),
        "optimization_message": str(result.message),
        "optimization_iterations": int(result.nit),
        "optimization_time_seconds": round(elapsed, 3),
        "l2_lambda": l2_lambda,
        "method": "weighted_scalar_temperature_scaling",
    }


def _softmax_np(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / exp_logits.sum(axis=1, keepdims=True)


# ============================================================
# Reliability Diagram
# ============================================================

def plot_reliability_diagram(
    y_true: np.ndarray,
    y_proba_before: np.ndarray,
    y_proba_after: np.ndarray,
    output_path: str | Path,
    n_bins: int = 15,
    title: str = "Reliability Diagram: Before vs After Calibration",
) -> None:
    """
    Generate a reliability diagram comparing calibrated vs uncalibrated predictions.

    Args:
        y_true: Ground truth labels, shape (N,).
        y_proba_before: Uncalibrated probabilities, shape (N, C).
        y_proba_after: Calibrated probabilities, shape (N, C).
        output_path: Path to save the diagram image.
        n_bins: Number of calibration bins.
        title: Diagram title.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[WARN] matplotlib not available, skipping reliability diagram.")
        return

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, y_proba, subtitle in [
        (axes[0], y_proba_before, "Before Calibration"),
        (axes[1], y_proba_after, "After Calibration"),
    ]:
        y_pred = np.argmax(y_proba, axis=1)
        confidences = np.max(y_proba, axis=1)
        accuracies = (y_pred == y_true).astype(float)

        bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
        bin_centers = []
        bin_accs = []
        bin_confs = []
        bin_counts = []

        for i in range(n_bins):
            in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i + 1])
            if in_bin.sum() > 0:
                bin_centers.append((bin_boundaries[i] + bin_boundaries[i + 1]) / 2)
                bin_accs.append(accuracies[in_bin].mean())
                bin_confs.append(confidences[in_bin].mean())
                bin_counts.append(in_bin.sum())

        ax.bar(bin_centers, bin_accs, width=1.0 / n_bins, alpha=0.6,
               edgecolor="black", label="Accuracy")
        ax.plot([0, 1], [0, 1], "r--", label="Perfect calibration")
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Accuracy")
        ax.set_title(subtitle)
        ax.legend(loc="upper left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ece = compute_ece(y_true, y_proba, n_bins=n_bins)
        ax.text(0.05, 0.92, f"ECE = {ece:.4f}", transform=ax.transAxes,
                fontsize=12, verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"[CALIBRATE] Reliability diagram saved to {output_path}")


# ============================================================
# Save / Load Calibration
# ============================================================

def save_calibration_results(
    results: dict[str, Any],
    metadata_path: str | Path,
) -> None:
    """
    Save calibration temperature and metrics into the model's metadata.json.

    If the file exists, merge calibration results into it. Otherwise, create new.
    """
    metadata_path = Path(metadata_path)
    metadata = {}
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)

    metadata["calibration"] = {
        "temperature": results["optimal_temperature"],
        "method": results["method"],
        "ece_before": results["ece_before"],
        "ece_after": results["ece_after"],
        "ece_reduction_pct": results["ece_reduction_pct"],
        "argmax_preserved": results["argmax_preserved"],
        "classwise_ece_before": results.get("classwise_ece_before", {}),
        "classwise_ece_after": results.get("classwise_ece_after", {}),
    }

    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"[CALIBRATE] Calibration saved to {metadata_path}")


def load_calibration_temperature(metadata_path: str | Path) -> float | None:
    """Load the calibration temperature from metadata.json. Returns None if not found."""
    metadata_path = Path(metadata_path)
    if not metadata_path.exists():
        return None
    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    calib = metadata.get("calibration", {})
    return calib.get("temperature", None)


# ============================================================
# CLI Entry Point
# ============================================================

def main() -> None:
    """Run calibration from CLI using the validation split."""
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description="Temperature Scaling Calibration")
    parser.add_argument("--val-path", default="data/processed/dev.parquet",
                        help="Path to validation split (parquet).")
    parser.add_argument("--metadata-path", default="artifacts/hate_speech_model/metadata.json",
                        help="Path to save calibration results.")
    parser.add_argument("--diagram-path", default="results/reliability_diagram.png",
                        help="Path to save reliability diagram.")
    parser.add_argument("--l2-lambda", type=float, default=0.01,
                        help="L2 regularization strength.")
    parser.add_argument("--init-temp", type=float, default=1.5,
                        help="Initial temperature for optimization.")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for logit extraction.")
    parser.add_argument("--model-type", default="xlm-roberta",
                        help="Model type for labeling the outputs.")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to use for calibration.")
    args = parser.parse_args()

    # Load validation data
    val_path = Path(args.val_path)
    if not val_path.exists():
        print(f"[ERROR] Validation file not found: {val_path}")
        return

    df = pd.read_parquet(val_path) if val_path.suffix == ".parquet" else pd.read_csv(val_path)
    
    if args.max_samples is not None and len(df) > args.max_samples:
        from sklearn.model_selection import train_test_split
        df, _ = train_test_split(
            df,
            train_size=args.max_samples,
            stratify=df["label"],
            random_state=42
        )
        
    texts = df["text"].tolist()
    labels = df["label"].astype(int).values

    print(f"[CALIBRATE] Loaded {len(texts)} validation samples.")
    print(f"[CALIBRATE] Label distribution: {np.bincount(labels)}")

    # Extract raw logits (XLM-R)
    from src.models.classifier import HateSpeechClassifier
    classifier = HateSpeechClassifier(
        model_source="huggingface",
        hf_repo_id="thong7d/vihsd-xlmr-base-hate-speech",
        device="auto",
        use_word_segmentation=False,
    )
    all_logits = []
    n_batches = (len(texts) + args.batch_size - 1) // args.batch_size
    print(f"[CALIBRATE] Extracting logits in {n_batches} batches (use_word_segmentation=False)...")
    for i in range(0, len(texts), args.batch_size):
        batch = texts[i:i + args.batch_size]
        batch_logits = classifier.get_raw_logits_batch(batch)
        all_logits.append(batch_logits)
        batch_idx = i // args.batch_size
        if batch_idx % 10 == 0 or batch_idx == n_batches - 1:
            print(f"[CALIBRATE] Progress: batch {batch_idx + 1}/{n_batches} processed.")
    logits = np.vstack(all_logits)

    print(f"[CALIBRATE] Logits extracted: shape={logits.shape}")

    # Run calibration
    results = calibrate_temperature(
        logits=logits,
        labels=labels,
        l2_lambda=args.l2_lambda,
        init_temperature=args.init_temp,
    )

    print(f"\n{'='*60}")
    print(f"  CALIBRATION RESULTS ({args.model_type.upper()})")
    print(f"{'='*60}")
    print(f"  Optimal Temperature:  {results['optimal_temperature']}")
    print(f"  ECE Before:           {results['ece_before']}")
    print(f"  ECE After:            {results['ece_after']}")
    print(f"  ECE Reduction:        {results['ece_reduction_pct']}%")
    print(f"  Argmax Preserved:     {results['argmax_preserved']}")
    print(f"  Optimization Time:    {results['optimization_time_seconds']}s")
    print(f"{'='*60}\n")

    # Save results
    save_calibration_results(results, args.metadata_path)

    # Generate reliability diagram
    probs_before = _softmax_np(logits)
    probs_after = _softmax_np(logits / results["optimal_temperature"])
    plot_reliability_diagram(
        labels, probs_before, probs_after,
        output_path=args.diagram_path,
        title=f"Reliability Diagram: {args.model_type.upper()} Calibration",
    )


if __name__ == "__main__":
    main()
