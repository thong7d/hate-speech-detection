from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    config_path = resolve_path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def resolve_path(path: str | Path, *, root: str | Path | None = None) -> Path:
    raw = str(path).replace("{project_root}", str(root or project_root()))
    candidate = Path(os.path.expandvars(raw)).expanduser()
    if candidate.is_absolute():
        return candidate
    return (Path(root or project_root()) / candidate).resolve()


def deep_get(config: dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict) or part not in current:
            return default
        current = current[part]
    return current
