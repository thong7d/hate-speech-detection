"""
Shared Hugging Face inference helpers.

The API, agent, evaluation scripts, and manual robustness tests should use
this module so tokenization, preprocessing, label mapping, and confidence
calculation stay consistent.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping

import torch

try:
    from data.preprocessing import clean_text
except ImportError:  # package imported as src.evaluation.*
    from src.data.preprocessing import clean_text


DEFAULT_LABEL_MAP = {0: "CLEAN", 1: "OFFENSIVE", 2: "HATE"}


@dataclass
class ModelArtifacts:
    model: torch.nn.Module
    tokenizer: object
    device: torch.device
    label_map: dict[int, str]
    max_length: int = 128
    use_word_segmentation: bool = True


def normalize_label_map(label_map: Mapping | None, num_labels: int = 3) -> dict[int, str]:
    """Return an int->label mapping, falling back from LABEL_0 style names."""
    if not label_map:
        return dict(DEFAULT_LABEL_MAP)

    normalized: dict[int, str] = {}
    for key, value in dict(label_map).items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        normalized[idx] = str(value)

    if len(normalized) != num_labels:
        return dict(DEFAULT_LABEL_MAP)
    if all(value.upper().startswith("LABEL_") for value in normalized.values()):
        return dict(DEFAULT_LABEL_MAP)
    return {idx: normalized[idx] for idx in sorted(normalized)}


def label_map_from_model(model: torch.nn.Module, num_labels: int = 3) -> dict[int, str]:
    config = getattr(model, "config", None)
    id2label = getattr(config, "id2label", None)
    return normalize_label_map(id2label, num_labels=num_labels)


def load_hf_artifacts(
    model_source: str,
    *,
    token: str | None = None,
    device: str | torch.device = "auto",
    max_length: int = 128,
    label_map: Mapping | None = None,
    use_word_segmentation: bool = True,
) -> ModelArtifacts:
    """Load tokenizer/model and return a ready-to-use inference bundle."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_source, token=token, trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_source, token=token, trust_remote_code=False
    )

    if device == "auto":
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)

    model = model.to(resolved_device)
    model.eval()

    num_labels = int(getattr(getattr(model, "config", None), "num_labels", 3))
    labels = normalize_label_map(label_map, num_labels=num_labels) if label_map else label_map_from_model(model, num_labels)

    return ModelArtifacts(
        model=model,
        tokenizer=tokenizer,
        device=resolved_device,
        label_map=labels,
        max_length=max_length,
        use_word_segmentation=use_word_segmentation,
    )


def predict_text(
    model: torch.nn.Module,
    tokenizer: object,
    text: str,
    *,
    device: str | torch.device,
    max_length: int = 128,
    label_map: Mapping[int, str] | None = None,
    borderline_low: float = 0.35,
    borderline_high: float = 0.65,
    preprocess: bool = True,
    use_word_segmentation: bool = True,
) -> dict:
    """Classify one text sample and return label, confidence, and probabilities."""
    labels = normalize_label_map(label_map, num_labels=len(label_map or DEFAULT_LABEL_MAP))
    resolved_device = torch.device(device)
    original_text = text
    model_text = clean_text(text, use_word_segmentation=use_word_segmentation) if preprocess else str(text)

    model.eval()
    encoding = tokenizer(
        model_text,
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    encoding = {key: value.to(resolved_device) for key, value in encoding.items()}

    with torch.no_grad():
        outputs = model(**encoding)
        probs = torch.softmax(outputs.logits, dim=-1).detach().cpu()[0]

    pred_id = int(torch.argmax(probs).item())
    confidence = float(probs[pred_id].item())
    probabilities = {
        labels.get(i, str(i)): round(float(probs[i].item()), 4)
        for i in range(len(probs))
    }

    return {
        "text": original_text,
        "processed_text": model_text,
        "label": labels.get(pred_id, str(pred_id)),
        "label_id": pred_id,
        "confidence": round(confidence, 4),
        "probabilities": probabilities,
        "scores": probabilities,
        "is_borderline": borderline_low <= confidence <= borderline_high,
    }


def predict_with_artifacts(
    artifacts: ModelArtifacts,
    text: str,
    *,
    borderline_low: float = 0.35,
    borderline_high: float = 0.65,
    preprocess: bool = True,
) -> dict:
    return predict_text(
        artifacts.model,
        artifacts.tokenizer,
        text,
        device=artifacts.device,
        max_length=artifacts.max_length,
        label_map=artifacts.label_map,
        borderline_low=borderline_low,
        borderline_high=borderline_high,
        preprocess=preprocess,
        use_word_segmentation=artifacts.use_word_segmentation,
    )


def predict_many(
    artifacts: ModelArtifacts,
    texts: Iterable[str],
    *,
    borderline_low: float = 0.35,
    borderline_high: float = 0.65,
    preprocess: bool = True,
) -> list[dict]:
    return [
        predict_with_artifacts(
            artifacts,
            text,
            borderline_low=borderline_low,
            borderline_high=borderline_high,
            preprocess=preprocess,
        )
        for text in texts
    ]
