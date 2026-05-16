from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping


DEFAULT_LABELS = ["CLEAN", "OFFENSIVE", "HATE"]


def build_label_mapping(labels: Iterable[str] | None = None) -> dict:
    names = list(labels or DEFAULT_LABELS)
    return {
        "id2label": {str(idx): label for idx, label in enumerate(names)},
        "label2id": {label: idx for idx, label in enumerate(names)},
    }


def load_label_mapping(path: str | Path | None, labels: Iterable[str] | None = None) -> dict:
    if path is None or not Path(path).exists():
        return build_label_mapping(labels)
    with Path(path).open("r", encoding="utf-8") as f:
        mapping = json.load(f)
    validate_label_mapping(mapping)
    return mapping


def save_label_mapping(mapping: Mapping, path: str | Path) -> None:
    validate_label_mapping(mapping)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)


def id2label_from_mapping(mapping: Mapping) -> dict[int, str]:
    validate_label_mapping(mapping)
    return {int(k): str(v) for k, v in mapping["id2label"].items()}


def label2id_from_mapping(mapping: Mapping) -> dict[str, int]:
    validate_label_mapping(mapping)
    return {str(k): int(v) for k, v in mapping["label2id"].items()}


def validate_label_mapping(mapping: Mapping) -> None:
    if "id2label" not in mapping or "label2id" not in mapping:
        raise ValueError("label mapping must contain id2label and label2id")
    id2label = {int(k): str(v) for k, v in mapping["id2label"].items()}
    label2id = {str(k): int(v) for k, v in mapping["label2id"].items()}
    for idx, label in id2label.items():
        if label2id.get(label) != idx:
            raise ValueError(f"inconsistent mapping for label {label!r}")
