from __future__ import annotations

import logging

try:
    from src.models.classifier import HateSpeechClassifier
    from src.utils.config import load_yaml_config
except ImportError:
    from models.classifier import HateSpeechClassifier
    from utils.config import load_yaml_config


logger = logging.getLogger(__name__)
classifier: HateSpeechClassifier | None = None
load_error: str | None = None


def load_classifier(config_path: str = "configs/api.yaml") -> HateSpeechClassifier | None:
    global classifier, load_error
    if classifier is not None:
        return classifier
    try:
        config = load_yaml_config(config_path)
        classifier = HateSpeechClassifier.from_config(config)
        load_error = None
        return classifier
    except Exception as exc:
        load_error = str(exc)
        logger.warning("Model is not ready: %s", load_error)
        return None


def get_classifier() -> HateSpeechClassifier | None:
    return classifier or load_classifier()


def readiness() -> dict:
    model = classifier
    return {
        "ready": model is not None,
        "model_source": getattr(model, "loaded_from", None) if model else None,
        "model_version": getattr(model, "model_version", None) if model else None,
        "error": load_error if model is None else None,
    }
