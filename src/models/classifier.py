from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from src.data.preprocessing import preprocess_text
    from src.models.registry import (
        DEFAULT_LABELS,
        build_label_mapping,
        id2label_from_mapping,
        load_label_mapping,
    )
    from src.utils.config import resolve_path
except ImportError:
    from data.preprocessing import preprocess_text
    from models.registry import DEFAULT_LABELS, build_label_mapping, id2label_from_mapping, load_label_mapping
    from utils.config import resolve_path


def compute_metrics(eval_pred: tuple[np.ndarray, np.ndarray]) -> dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    labels = np.asarray(labels).reshape(-1)
    preds = np.asarray(preds).reshape(-1)
    label_ids = list(range(len(DEFAULT_LABELS)))
    metrics = {
        "accuracy": float(round(accuracy_score(labels, preds), 4)),
        "precision_macro": round(
            float(precision_score(labels, preds, labels=label_ids, average="macro", zero_division=0)), 4
        ),
        "recall_macro": round(
            float(recall_score(labels, preds, labels=label_ids, average="macro", zero_division=0)), 4
        ),
        "macro_f1": round(float(f1_score(labels, preds, labels=label_ids, average="macro", zero_division=0)), 4),
        "weighted_f1": round(
            float(f1_score(labels, preds, labels=label_ids, average="weighted", zero_division=0)), 4
        ),
    }
    precision = precision_score(labels, preds, labels=label_ids, average=None, zero_division=0)
    recall = recall_score(labels, preds, labels=label_ids, average=None, zero_division=0)
    f1 = f1_score(labels, preds, labels=label_ids, average=None, zero_division=0)
    for idx, label in enumerate(DEFAULT_LABELS):
        metrics[f"precision_{label}"] = round(float(precision[idx]), 4)
        metrics[f"recall_{label}"] = round(float(recall[idx]), 4)
        metrics[f"f1_{label}"] = round(float(f1[idx]), 4)
    return metrics


class HateSpeechClassifier:
    def __init__(
        self,
        model_source: str = "huggingface",
        model_path: str = "artifacts/hate_speech_model/model",
        hf_repo_id: str | None = None,
        *,
        artifact_dir: str = "artifacts/hate_speech_model",
        label_mapping_path: str | None = None,
        metadata_path: str | None = None,
        threshold: float = 0.5,
        max_length: int = 128,
        device: str = "auto",
    ) -> None:
        self.model_source = model_source
        self.hf_repo_id = hf_repo_id
        self.model_path = str(resolve_path(model_path))
        self.artifact_dir = resolve_path(artifact_dir)
        self.label_mapping_path = resolve_path(label_mapping_path or self.artifact_dir / "label_mapping.json")
        self.metadata_path = resolve_path(metadata_path or self.artifact_dir / "metadata.json")
        self.threshold = threshold
        self.max_length = max_length
        self.device_name = device
        self.model = None
        self.tokenizer = None
        self.device = None
        self.label_mapping = load_label_mapping(self.label_mapping_path, DEFAULT_LABELS)
        self.id2label = id2label_from_mapping(self.label_mapping)
        self.metadata = self._load_metadata()
        self.loaded_from = None
        self._load()

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "HateSpeechClassifier":
        model_cfg = dict(config.get("model", config))
        source = str(model_cfg.get("source", "huggingface"))
        env_source = _env("MODEL_SOURCE")
        env_repo = _env("HF_REPO_ID") or _env("HF_MODEL_ID")
        env_local = _env("MODEL_LOCAL_PATH")
        return cls(
            model_source=env_source or source,
            model_path=env_local or model_cfg.get("local_path", "artifacts/hate_speech_model/model"),
            hf_repo_id=env_repo or model_cfg.get("hf_repo_id"),
            artifact_dir=model_cfg.get("artifact_dir", "artifacts/hate_speech_model"),
            label_mapping_path=model_cfg.get("label_mapping_path"),
            metadata_path=model_cfg.get("metadata_path"),
            threshold=float(model_cfg.get("threshold", 0.5)),
            max_length=int(model_cfg.get("max_length", 128)),
        )

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version") or self.metadata.get("version") or "v1.0.0")

    def predict(self, text: str) -> dict:
        if self.model is None or self.tokenizer is None or self.device is None:
            raise RuntimeError("Model is not loaded.")

        import torch

        original_text = "" if text is None else str(text)
        model_text = preprocess_text(original_text)
        encoding = self.tokenizer(
            model_text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        encoding = {key: value.to(self.device) for key, value in encoding.items()}

        self.model.eval()
        with torch.no_grad():
            logits = self.model(**encoding).logits
            probabilities_tensor = torch.softmax(logits, dim=-1).detach().cpu()[0]

        pred_id = int(torch.argmax(probabilities_tensor).item())
        probabilities = {
            self.id2label.get(idx, str(idx)): round(float(probabilities_tensor[idx].item()), 4)
            for idx in range(len(probabilities_tensor))
        }
        label = self.id2label.get(pred_id, str(pred_id))
        confidence = round(float(probabilities_tensor[pred_id].item()), 4)
        return {
            "text": original_text,
            "label": label,
            "confidence": confidence,
            "probabilities": probabilities,
            "model_version": self.model_version,
        }

    def predict_batch(self, texts: list[str]) -> list[dict]:
        return [self.predict(text) for text in texts]

    def _load(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if self.device_name == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(self.device_name)

        sources: list[tuple[str, str]] = []
        if self.model_source == "huggingface" and self.hf_repo_id:
            sources.append(("huggingface", self.hf_repo_id))
        if Path(self.model_path).exists():
            sources.append(("local", self.model_path))
        if self.model_source == "local" and not Path(self.model_path).exists():
            raise FileNotFoundError(f"Local model artifact not found: {self.model_path}")
        if not sources:
            raise FileNotFoundError(
                "No model source is available. Configure model.hf_repo_id or create "
                f"a local artifact at {self.model_path}."
            )

        last_error: Exception | None = None
        for source_name, source_value in sources:
            try:
                token = _env("HF_TOKEN") if source_name == "huggingface" else None
                self.tokenizer = AutoTokenizer.from_pretrained(source_value, token=token, trust_remote_code=False)
                self.model = AutoModelForSequenceClassification.from_pretrained(
                    source_value,
                    token=token,
                    trust_remote_code=False,
                )
                self.model.to(self.device)
                self.model.eval()
                self.loaded_from = source_name
                self._sync_mapping_from_model_config()
                return
            except Exception as exc:
                last_error = exc
                self.model = None
                self.tokenizer = None

        raise RuntimeError(f"Unable to load model from Hugging Face or local artifact: {last_error}")

    def _sync_mapping_from_model_config(self) -> None:
        config = getattr(self.model, "config", None)
        id2label = getattr(config, "id2label", None)
        if id2label:
            normalized = {str(int(k)): str(v) for k, v in dict(id2label).items()}
            if not all(value.startswith("LABEL_") for value in normalized.values()):
                self.label_mapping = build_label_mapping(normalized[str(idx)] for idx in sorted(map(int, normalized)))
                self.id2label = id2label_from_mapping(self.label_mapping)

    def _load_metadata(self) -> dict[str, Any]:
        if self.metadata_path.exists():
            with self.metadata_path.open("r", encoding="utf-8") as f:
                return json.load(f)
        return {
            "model_version": "v1.0.0",
            "note": "Khong du du lieu de xac minh metadata artifact.",
        }


def _env(name: str) -> str | None:
    value = __import__("os").environ.get(name)
    return value or None
