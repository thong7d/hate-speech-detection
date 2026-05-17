from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.data.dataset import ViHSDDataset
    from src.data.preprocessing import compute_balanced_class_weights, normalize_text_label_frame
    from src.models.classifier import compute_metrics
    from src.models.registry import build_label_mapping, label2id_from_mapping, save_label_mapping
    from src.utils.config import resolve_path
    from src.utils.seed import set_seed
except ImportError:
    from data.dataset import ViHSDDataset
    from data.preprocessing import compute_balanced_class_weights, normalize_text_label_frame
    from models.classifier import compute_metrics
    from models.registry import build_label_mapping, label2id_from_mapping, save_label_mapping
    from utils.config import resolve_path
    from utils.seed import set_seed


def train_from_config(config: dict[str, Any]) -> dict[str, Any]:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        EarlyStoppingCallback,
        Trainer,
        TrainingArguments,
    )

    seed = int(config.get("seed", 42))
    set_seed(seed)

    data_cfg = config["data"]
    model_cfg = config["model"]
    training_cfg = config["training"]
    export_cfg = config["export"]

    labels = list(model_cfg.get("labels") or [])
    if not labels:
        labels = [str(i) for i in range(int(model_cfg.get("num_labels", 2)))]
    label_mapping = build_label_mapping(labels)
    label2id = label2id_from_mapping(label_mapping)
    id2label = {idx: label for label, idx in label2id.items()}

    train_df = _load_frame(data_cfg["train_path"], data_cfg)
    valid_df = _load_frame(data_cfg["valid_path"], data_cfg)
    num_labels = int(model_cfg.get("num_labels") or len(label_mapping["id2label"]))

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], trust_remote_code=False)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_cfg["base_model"],
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        trust_remote_code=False,
    )

    max_length = int(model_cfg.get("max_length", 128))
    train_dataset = ViHSDDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    valid_dataset = ViHSDDataset(valid_df["text"].tolist(), valid_df["label"].tolist(), tokenizer, max_length)
    class_weights = _class_weights_from_training_config(training_cfg, train_df["label"].tolist(), num_labels)

    output_dir = resolve_path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_for_best_model = str(training_cfg.get("metric_for_best_model", "macro_f1"))
    args = TrainingArguments(
        output_dir=str(output_dir),
        learning_rate=float(training_cfg.get("learning_rate", 2e-5)),
        per_device_train_batch_size=int(training_cfg.get("batch_size", 16)),
        per_device_eval_batch_size=int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 16))),
        num_train_epochs=float(training_cfg.get("num_epochs", 3)),
        weight_decay=float(training_cfg.get("weight_decay", 0.01)),
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model=metric_for_best_model,
        greater_is_better=bool(training_cfg.get("greater_is_better", True)),
        save_total_limit=int(training_cfg.get("save_total_limit", 2)),
        fp16=bool(training_cfg.get("fp16", False)),
        report_to=[],
        seed=seed,
    )
    callbacks = []
    if int(training_cfg.get("early_stopping_patience", 0)) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(training_cfg.get("early_stopping_patience", 2))
            )
        )

    trainer_cls = _weighted_trainer_class(Trainer) if class_weights else Trainer
    trainer_kwargs = {}
    if class_weights:
        trainer_kwargs["class_weights"] = class_weights

    trainer = trainer_cls(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
        **trainer_kwargs,
    )

    trainer.train()
    metrics = trainer.evaluate()

    artifact_dir = resolve_path(export_cfg["artifact_dir"])
    final_model_dir = resolve_path(export_cfg["final_model_dir"])
    final_model_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    save_label_mapping(label_mapping, artifact_dir / "label_mapping.json")
    metadata = {
        "model_version": "v1.0.0",
        "base_model": model_cfg["base_model"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "train_path": data_cfg["train_path"],
            "valid_path": data_cfg["valid_path"],
            "test_path": data_cfg.get("test_path"),
            "text_column": data_cfg.get("text_column", "text"),
            "label_column": data_cfg.get("label_column", "label"),
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "class_weighting": training_cfg.get("class_weighting", "none"),
        "class_weights": class_weights,
    }
    _write_json(artifact_dir / "metadata.json", metadata)
    _write_json(resolve_path(config["evaluation"]["metrics_output_path"]), metrics)
    _copy_checkpoint(trainer.state.best_model_checkpoint, artifact_dir / "checkpoint" / "checkpoint-best")
    _copy_checkpoint(_last_checkpoint(output_dir), artifact_dir / "checkpoint" / "checkpoint-last")

    return {
        "artifact_dir": str(artifact_dir),
        "final_model_dir": str(final_model_dir),
        "metrics": metrics,
        "metadata": metadata,
    }


def _class_weights_from_training_config(
    training_cfg: dict[str, Any],
    labels: list[int],
    num_labels: int,
) -> list[float] | None:
    mode = str(training_cfg.get("class_weighting", "none")).lower()
    if mode in {"", "none", "false", "off"}:
        return None

    weights_by_id = compute_balanced_class_weights(labels, num_labels)
    weights = [float(weights_by_id[idx]) for idx in range(num_labels)]
    if mode == "balanced":
        return weights
    if mode == "sqrt_balanced":
        return [weight**0.5 for weight in weights]
    raise ValueError("training.class_weighting must be one of: none, balanced, sqrt_balanced")


def _weighted_trainer_class(base_trainer):
    class WeightedTrainer(base_trainer):
        def __init__(self, *args, class_weights: list[float], **kwargs) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            weights = torch.tensor(self.class_weights, dtype=logits.dtype, device=logits.device)
            loss_fct = torch.nn.CrossEntropyLoss(weight=weights)
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
            return (loss, outputs) if return_outputs else loss

    return WeightedTrainer


def _load_frame(path: str, data_cfg: dict[str, Any]) -> pd.DataFrame:
    input_path = resolve_path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Dataset split not found: {input_path}")
    if input_path.suffix.lower() == ".parquet":
        df = pd.read_parquet(input_path)
    else:
        df = pd.read_csv(input_path)
    return normalize_text_label_frame(
        df,
        text_column=data_cfg.get("text_column"),
        label_column=data_cfg.get("label_column"),
    )


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def _copy_checkpoint(source: str | Path | None, target: Path) -> None:
    if source is None:
        return
    source_path = Path(source)
    if not source_path.exists():
        return
    if target.exists():
        shutil.rmtree(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_path, target)


def _last_checkpoint(output_dir: Path) -> Path | None:
    checkpoints = [p for p in output_dir.glob("checkpoint-*") if p.is_dir()]
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)
