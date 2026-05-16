from __future__ import annotations

import argparse
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from src.models.registry import build_label_mapping, save_label_mapping
    from src.utils.config import load_yaml_config, resolve_path
except ImportError:
    from models.registry import build_label_mapping, save_label_mapping
    from utils.config import load_yaml_config, resolve_path


INSUFFICIENT_TOKEN = "Khong du du lieu de xac minh HF_TOKEN hoac HF_TOKEN chua duoc cau hinh"


def export_from_config(
    config: dict[str, Any],
    *,
    push_to_hub: bool = False,
    push_checkpoints: bool = False,
) -> dict[str, Any]:
    export_cfg = config.get("export", config.get("model", {}))
    model_cfg = config.get("model", {})
    artifact_dir = resolve_path(export_cfg.get("artifact_dir", "artifacts/hate_speech_model"))
    final_model_dir = resolve_path(export_cfg.get("final_model_dir") or model_cfg.get("local_path", artifact_dir / "model"))
    artifact_dir.mkdir(parents=True, exist_ok=True)
    final_model_dir.mkdir(parents=True, exist_ok=True)

    label_mapping_path = artifact_dir / "label_mapping.json"
    if not label_mapping_path.exists():
        labels = model_cfg.get("labels") or ["CLEAN", "OFFENSIVE", "HATE"]
        save_label_mapping(build_label_mapping(labels), label_mapping_path)

    metadata_path = artifact_dir / "metadata.json"
    if not metadata_path.exists():
        _write_json(
            metadata_path,
            {
                "model_version": "v1.0.0",
                "base_model": model_cfg.get("base_model", "Khong du du lieu de xac minh"),
                "trained_at": "Khong du du lieu de xac minh",
                "exported_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    metrics_path = artifact_dir / "metrics.json"
    configured_metrics = resolve_path(config.get("evaluation", {}).get("metrics_output_path", metrics_path))
    if configured_metrics.exists() and configured_metrics != metrics_path:
        shutil.copy2(configured_metrics, metrics_path)
    elif not metrics_path.exists():
        _write_json(metrics_path, {"status": "Khong du du lieu de xac minh"})

    model_card = artifact_dir / "model_card.md"
    if not model_card.exists():
        model_card.write_text(_model_card(config, artifact_dir), encoding="utf-8")

    result = {
        "artifact_dir": str(artifact_dir),
        "final_model_dir": str(final_model_dir),
        "files": ["model/", "label_mapping.json", "metadata.json", "metrics.json", "model_card.md"],
        "pushed_to_hub": False,
    }
    if push_to_hub:
        result.update(push_artifact_to_hub(artifact_dir, export_cfg["hf_repo_id"], push_checkpoints=push_checkpoints))
    return result


def push_artifact_to_hub(artifact_dir: Path, repo_id: str, *, push_checkpoints: bool = False) -> dict[str, Any]:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print(INSUFFICIENT_TOKEN)
        return {"pushed_to_hub": False, "reason": INSUFFICIENT_TOKEN}

    from huggingface_hub import HfApi, upload_folder

    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, exist_ok=True, private=False)

    ignore_patterns = [
        ".env",
        "*HF_TOKEN*",
        "__pycache__/*",
        "*.pyc",
        ".cache/*",
        "wandb/*",
        "mlruns/*",
    ]
    if not push_checkpoints:
        ignore_patterns.append("checkpoint/*")

    upload_folder(
        repo_id=repo_id,
        folder_path=str(artifact_dir),
        token=token,
        ignore_patterns=ignore_patterns,
    )
    return {"pushed_to_hub": True, "hf_repo_id": repo_id, "push_checkpoints": push_checkpoints}


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def _model_card(config: dict[str, Any], artifact_dir: Path) -> str:
    repo_id = config.get("export", {}).get("hf_repo_id") or config.get("model", {}).get("hf_repo_id", "")
    return (
        "# Hate Speech Detection Model\n\n"
        "This repository contains the exported inference artifact for the hate speech detection project.\n\n"
        "## Intended Use\n\n"
        "Content moderation assistance for detecting hate or offensive speech. Human review is recommended for high-impact decisions.\n\n"
        "## Metrics\n\n"
        "See `metrics.json`. If evaluation has not been run, the file states `Khong du du lieu de xac minh`.\n\n"
        "## Local Artifact\n\n"
        f"- Artifact directory: `{artifact_dir.as_posix()}`\n"
        f"- Hugging Face repo: `{repo_id}`\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Export and optionally push the final model artifact.")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--push-checkpoints", action="store_true")
    args = parser.parse_args()
    result = export_from_config(
        load_yaml_config(args.config),
        push_to_hub=args.push_to_hub,
        push_checkpoints=args.push_checkpoints,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
