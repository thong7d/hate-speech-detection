"""
Unit tests for Temperature Scaling Calibration.

Tests:
  - ECE computation correctness
  - Weighted NLL loss computation
  - Scalar Temperature Scaling argmax preservation
  - Calibration pipeline integration
  - Reliability diagram generation (no crash)
  - Save/load calibration results
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

try:
    from src.evaluation.calibrate import (
        calibrate_temperature,
        compute_ece,
        compute_classwise_ece,
        load_calibration_temperature,
        plot_reliability_diagram,
        save_calibration_results,
        weighted_nll_with_l2,
        _softmax_np,
    )
except ImportError:
    from evaluation.calibrate import (
        calibrate_temperature,
        compute_ece,
        compute_classwise_ece,
        load_calibration_temperature,
        plot_reliability_diagram,
        save_calibration_results,
        weighted_nll_with_l2,
        _softmax_np,
    )


# ============================================================
# Fixtures
# ============================================================

@pytest.fixture
def synthetic_data():
    """Create synthetic logits and labels for testing."""
    np.random.seed(42)
    n_samples = 200
    n_classes = 3

    # Create logits with known overconfidence pattern
    # Class 0 (CLEAN) dominates, simulating the ViHSD class imbalance
    logits = np.random.randn(n_samples, n_classes)
    labels = np.random.choice(n_classes, n_samples, p=[0.83, 0.10, 0.07])

    # Make logits somewhat correlated with labels (better than random)
    for i in range(n_samples):
        logits[i, labels[i]] += 2.0  # Boost correct class logit

    return logits, labels


@pytest.fixture
def perfectly_calibrated_data():
    """Create data where predicted confidence matches accuracy."""
    np.random.seed(123)
    n_samples = 1000
    # Create logits with larger spread to cover a wider probability range
    logits = np.random.randn(n_samples, 3) * 2.0
    probs = _softmax_np(logits)
    labels = np.array([np.random.choice(3, p=probs[i]) for i in range(n_samples)])
    return logits, labels


# ============================================================
# Test: ECE Computation
# ============================================================

def test_ece_perfect_calibration(perfectly_calibrated_data):
    """ECE should be near 0 for perfectly calibrated predictions."""
    logits, labels = perfectly_calibrated_data
    probs = _softmax_np(logits)
    ece = compute_ece(labels, probs, n_bins=10)
    # Not exactly 0 due to binning, but should be small
    assert ece < 0.15, f"ECE for perfectly calibrated data should be small, got {ece}"


def test_ece_range(synthetic_data):
    """ECE should be between 0 and 1."""
    logits, labels = synthetic_data
    probs = _softmax_np(logits)
    ece = compute_ece(labels, probs)
    assert 0.0 <= ece <= 1.0, f"ECE out of range: {ece}"


def test_ece_overconfident_is_high():
    """ECE should be high when predictions are very overconfident."""
    n = 100
    # Create extremely overconfident predictions (99.9% on wrong class)
    probs = np.full((n, 3), 0.0005)
    probs[:, 0] = 0.999  # All predict class 0 with 99.9%
    labels = np.ones(n, dtype=int)  # But true labels are class 1

    ece = compute_ece(labels, probs)
    assert ece > 0.5, f"ECE should be high for overconfident wrong predictions, got {ece}"


def test_classwise_ece_returns_all_classes(synthetic_data):
    """Per-class ECE should have entries for each class."""
    logits, labels = synthetic_data
    probs = _softmax_np(logits)
    classwise = compute_classwise_ece(labels, probs, label_names=["CLEAN", "OFFENSIVE", "HATE"])
    assert "ece_CLEAN" in classwise
    assert "ece_OFFENSIVE" in classwise
    assert "ece_HATE" in classwise


# ============================================================
# Test: Weighted NLL Loss
# ============================================================

def test_weighted_nll_positive(synthetic_data):
    """Weighted NLL should return a positive loss value."""
    logits, labels = synthetic_data
    class_weights = np.array([1.0, 1.9, 1.5])
    loss = weighted_nll_with_l2(1.5, logits, labels, class_weights, l2_lambda=0.01)
    assert loss > 0, f"Loss should be positive, got {loss}"


def test_weighted_nll_temperature_effect(synthetic_data):
    """Loss should change with different temperatures."""
    logits, labels = synthetic_data
    class_weights = np.array([1.0, 1.9, 1.5])

    loss_t1 = weighted_nll_with_l2(1.0, logits, labels, class_weights, l2_lambda=0.0)
    loss_t2 = weighted_nll_with_l2(2.0, logits, labels, class_weights, l2_lambda=0.0)

    assert loss_t1 != loss_t2, "Loss should differ for different temperatures"


def test_weighted_nll_l2_regularization(synthetic_data):
    """L2 penalty should increase loss when T deviates from 1."""
    logits, labels = synthetic_data
    class_weights = np.array([1.0, 1.0, 1.0])

    loss_no_l2 = weighted_nll_with_l2(2.0, logits, labels, class_weights, l2_lambda=0.0)
    loss_with_l2 = weighted_nll_with_l2(2.0, logits, labels, class_weights, l2_lambda=1.0)

    assert loss_with_l2 > loss_no_l2, \
        f"L2 should increase loss: without={loss_no_l2}, with={loss_with_l2}"


# ============================================================
# Test: Calibration Pipeline
# ============================================================

def test_calibrate_returns_valid_structure(synthetic_data):
    """Calibration should return a dict with all required keys."""
    logits, labels = synthetic_data
    result = calibrate_temperature(logits, labels)

    required_keys = {
        "optimal_temperature", "ece_before", "ece_after",
        "ece_reduction_pct", "argmax_preserved", "optimization_success",
        "method",
    }
    assert required_keys.issubset(result.keys()), \
        f"Missing keys: {required_keys - set(result.keys())}"


def test_calibrate_temperature_positive(synthetic_data):
    """Optimal temperature should be positive."""
    logits, labels = synthetic_data
    result = calibrate_temperature(logits, labels)
    assert result["optimal_temperature"] > 0, \
        f"Temperature should be positive, got {result['optimal_temperature']}"


def test_calibrate_reduces_ece(synthetic_data):
    """Calibration should reduce or maintain ECE."""
    logits, labels = synthetic_data
    result = calibrate_temperature(logits, labels)
    assert result["ece_after"] <= result["ece_before"] + 0.01, \
        f"ECE should not significantly increase: before={result['ece_before']}, after={result['ece_after']}"


def test_scalar_calibration_preserves_argmax(synthetic_data):
    """Scalar Temperature Scaling must preserve argmax predictions."""
    logits, labels = synthetic_data
    result = calibrate_temperature(logits, labels)
    assert result["argmax_preserved"] is True, \
        "Scalar Temperature Scaling must preserve argmax predictions"


def test_calibrate_with_custom_weights(synthetic_data):
    """Calibration should work with custom class weights."""
    logits, labels = synthetic_data
    custom_weights = np.array([0.5, 3.0, 2.5])
    result = calibrate_temperature(logits, labels, class_weights=custom_weights)
    assert result["optimization_success"]


# ============================================================
# Test: Save/Load Calibration
# ============================================================

def test_save_and_load_calibration(synthetic_data, tmp_path):
    """Save and load calibration temperature from metadata.json."""
    logits, labels = synthetic_data
    result = calibrate_temperature(logits, labels)

    metadata_path = tmp_path / "metadata.json"
    save_calibration_results(result, metadata_path)

    # Verify file exists
    assert metadata_path.exists()

    # Load and verify
    loaded_T = load_calibration_temperature(metadata_path)
    assert loaded_T is not None
    assert abs(loaded_T - result["optimal_temperature"]) < 1e-6


def test_save_calibration_merges_existing(tmp_path):
    """Save should merge into existing metadata without overwriting other fields."""
    metadata_path = tmp_path / "metadata.json"

    # Create existing metadata
    existing = {"model_version": "v2.0.0", "note": "existing data"}
    with metadata_path.open("w") as f:
        json.dump(existing, f)

    # Save calibration
    result = {"optimal_temperature": 1.5, "method": "weighted_scalar",
              "ece_before": 0.1, "ece_after": 0.05, "ece_reduction_pct": 50.0,
              "argmax_preserved": True}
    save_calibration_results(result, metadata_path)

    # Verify merge
    with metadata_path.open("r") as f:
        merged = json.load(f)
    assert merged["model_version"] == "v2.0.0"
    assert merged["note"] == "existing data"
    assert "calibration" in merged
    assert merged["calibration"]["temperature"] == 1.5


def test_load_nonexistent_returns_none(tmp_path):
    """Load from nonexistent file should return None."""
    loaded = load_calibration_temperature(tmp_path / "nonexistent.json")
    assert loaded is None


# ============================================================
# Test: Reliability Diagram
# ============================================================

def test_reliability_diagram_no_crash(synthetic_data, tmp_path):
    """Reliability diagram generation should not crash."""
    logits, labels = synthetic_data
    probs_before = _softmax_np(logits)
    probs_after = _softmax_np(logits / 1.5)
    output_path = tmp_path / "reliability.png"

    # This should not raise (even if matplotlib is missing, it just warns)
    try:
        plot_reliability_diagram(labels, probs_before, probs_after, output_path)
    except ImportError:
        pytest.skip("matplotlib not available")


# ============================================================
# Test: Softmax Utility
# ============================================================

def test_softmax_sums_to_one():
    """Softmax output should sum to 1 for each sample."""
    logits = np.array([[1.0, 2.0, 3.0], [10.0, -5.0, 0.0]])
    probs = _softmax_np(logits)
    for i in range(len(logits)):
        assert abs(probs[i].sum() - 1.0) < 1e-6, f"Row {i} sums to {probs[i].sum()}"


def test_softmax_numerically_stable():
    """Softmax should handle very large logits without overflow."""
    logits = np.array([[1000.0, 999.0, 998.0]])
    probs = _softmax_np(logits)
    assert not np.any(np.isnan(probs)), "NaN in softmax output"
    assert not np.any(np.isinf(probs)), "Inf in softmax output"
    assert abs(probs.sum() - 1.0) < 1e-6


def test_classifier_temperature_scaling():
    """Verify that HateSpeechClassifier loads temperature and applies scaling."""
    from src.models.classifier import HateSpeechClassifier

    # Instantiate classifier with custom temperature
    classifier = HateSpeechClassifier(
        model_source="huggingface",
        hf_repo_id="thong7d/vihsd-xlmr-base-hate-speech",
        device="cpu",
        temperature=2.0
    )

    # Run prediction and verify that temperature is applied
    assert classifier.temperature == 2.0
    result1 = classifier.predict("xin chào")

    # Setting temperature to None (T=1.0) should produce different probabilities
    classifier.temperature = None
    result2 = classifier.predict("xin chào")

    # Verify that they differ due to scaling
    any_diff = False
    for label in ["CLEAN", "OFFENSIVE", "HATE"]:
        if abs(result1["probabilities"][label] - result2["probabilities"][label]) > 1e-4:
            any_diff = True
            break
    assert any_diff, "Temperature scaling should modify prediction probabilities"
