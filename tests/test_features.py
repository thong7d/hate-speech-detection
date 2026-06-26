import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import torch
from transformers import XLMRobertaConfig, AutoTokenizer
from src.models.classifier import XLMRobertaTextCNN
from src.features.toxicity_score import compute_toxicity_score
from src.features.toxic_spans import extract_toxic_spans_gradcam, extract_toxic_spans_ig

def test_compute_toxicity_score():
    # Test case 1: Standard weights
    probs1 = {"CLEAN": 0.2, "OFFENSIVE": 0.5, "HATE": 0.3}
    # Expected: 0.5 * 0.4 + 0.3 * 1.0 = 0.2 + 0.3 = 0.5
    assert compute_toxicity_score(probs1) == pytest.approx(0.5)

    # Test case 2: Clamp upper bound
    probs2 = {"CLEAN": 0.0, "OFFENSIVE": 0.5, "HATE": 0.9}
    # Expected: 0.5 * 0.4 + 0.9 * 1.0 = 0.2 + 0.9 = 1.1 -> clamped to 1.0
    assert compute_toxicity_score(probs2) == pytest.approx(1.0)

    # Test case 3: Custom weights
    weights = {"OFFENSIVE": 0.2, "HATE": 0.8}
    probs3 = {"CLEAN": 0.1, "OFFENSIVE": 0.4, "HATE": 0.5}
    # Expected: 0.4 * 0.2 + 0.5 * 0.8 = 0.08 + 0.4 = 0.48
    assert compute_toxicity_score(probs3, weights=weights) == pytest.approx(0.48)


def test_toxic_spans_extraction():
    # Use a tiny config to initialize a lightweight XLMRobertaTextCNN
    config = XLMRobertaConfig(
        vocab_size=250002,  # standard XLM-R vocab size to match tokenizer
        hidden_size=16,
        num_attention_heads=2,
        num_hidden_layers=1,
        num_labels=3,
        initializer_range=0.02
    )
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")
    model = XLMRobertaTextCNN(config)
    
    text = "mày ngu vl"
    
    # Test Grad-CAM (inference pipeline)
    spans_gradcam = extract_toxic_spans_gradcam(
        model=model,
        tokenizer=tokenizer,
        text=text,
        target_class=2,
        top_k=2
    )
    
    assert isinstance(spans_gradcam, list)
    assert len(spans_gradcam) <= 2
    for span in spans_gradcam:
        assert "token" in span
        assert "score" in span
        assert "position" in span
        assert isinstance(span["score"], float)
        assert isinstance(span["position"], int)

    # Test Integrated Gradients (analysis pipeline)
    spans_ig = extract_toxic_spans_ig(
        model=model,
        tokenizer=tokenizer,
        text=text,
        target_class=2,
        top_k=2
    )
    
    assert isinstance(spans_ig, list)
    assert len(spans_ig) <= 2
    for span in spans_ig:
        assert "token" in span
        assert "score" in span
        assert "position" in span
        assert isinstance(span["score"], float)
        assert isinstance(span["position"], int)
