from __future__ import annotations

import torch

# PyTorch 2.6+ weights_only=True compatibility guard for NumPy 1.x/2.x RNG states
try:
    import numpy as np
    safe_globals = []
    
    # 1. Resolve core array reconstruction functions
    if hasattr(np, "_core") and hasattr(np._core, "multiarray") and hasattr(np._core.multiarray, "_reconstruct"):
        safe_globals.append(np._core.multiarray._reconstruct)
    if hasattr(np, "core") and hasattr(np.core, "multiarray") and hasattr(np.core.multiarray, "_reconstruct"):
        safe_globals.append(np.core.multiarray._reconstruct)
        
    # 2. Resolve standard NumPy data types
    if hasattr(np, "dtype"):
        safe_globals.append(np.dtype)
    if hasattr(np, "ndarray"):
        safe_globals.append(np.ndarray)
        
    # 3. Resolve NumPy 2.0+ specific strict DTypes required for RNG unpickling
    try:
        import numpy.dtypes
        if hasattr(numpy.dtypes, "UInt32DType"):
            safe_globals.append(numpy.dtypes.UInt32DType)
    except ImportError:
        pass

    # Register all resolved globals into PyTorch trusted serialization sandbox
    if safe_globals and hasattr(torch.serialization, "add_safe_globals"):
        torch.serialization.add_safe_globals(safe_globals)
except (ImportError, AttributeError):
    pass

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from src.data.augmentation import (
        augment_with_diacritic_removal,
        augment_with_eda,
        augment_with_teencode,
    )
    from src.data.dataset import ViHSDDataset
    from src.data.preprocessing import compute_balanced_class_weights, normalize_text_label_frame
    from src.models.classifier import compute_metrics
    from src.models.registry import build_label_mapping, label2id_from_mapping, save_label_mapping
    from src.training.robustness_cases import build_contrastive_frame, build_diacritic_frame, build_robustness_frame
    from src.utils.config import resolve_path
    from src.utils.seed import set_seed
except ImportError:
    from data.augmentation import (
        augment_with_diacritic_removal,
        augment_with_eda,
        augment_with_teencode,
    )
    from data.dataset import ViHSDDataset
    from data.preprocessing import compute_balanced_class_weights, normalize_text_label_frame
    from models.classifier import compute_metrics
    from models.registry import build_label_mapping, label2id_from_mapping, save_label_mapping
    from training.robustness_cases import build_contrastive_frame, build_diacritic_frame, build_robustness_frame
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
    preprocess_cfg = config.get("preprocessing", {})

    labels = list(model_cfg.get("labels") or [])
    if not labels:
        labels = [str(i) for i in range(int(model_cfg.get("num_labels", 2)))]
    label_mapping = build_label_mapping(labels)
    label2id = label2id_from_mapping(label_mapping)
    id2label = {idx: label for label, idx in label2id.items()}

    use_word_seg = bool(preprocess_cfg.get("use_word_segmentation", True))
    train_df = _load_frame(data_cfg["train_path"], data_cfg, use_word_segmentation=use_word_seg)
    valid_df = _load_frame(data_cfg["valid_path"], data_cfg, use_word_segmentation=use_word_seg)
    num_labels = int(model_cfg.get("num_labels") or len(label_mapping["id2label"]))

    # --- Class oversampling ---
    train_df, oversampling_summary = _oversample_training_frame(
        train_df, training_cfg, label2id, seed=seed,
    )

    # --- Robustness augmentation (hard-coded cases) ---
    train_df, augmentation_summary = _augment_training_frame(
        train_df, training_cfg, label2id, seed=seed,
    )
    train_df, contrastive_summary = _augment_contrastive_frame(
        train_df, training_cfg, label2id, seed=seed,
    )
    train_df, diacritic_summary = _augment_diacritic_frame(
        train_df, training_cfg, label2id, seed=seed,
    )

    # --- NEW: EDA augmentation ---
    eda_summary: dict[str, Any] = {"enabled": False}
    eda_cfg = training_cfg.get("eda_augmentation") or {}
    if eda_cfg and bool(eda_cfg.get("enabled", False)):
        target_labels = eda_cfg.get("target_labels", ["OFFENSIVE", "HATE"])
        train_df, eda_summary = augment_with_eda(
            train_df,
            label2id,
            target_labels=target_labels,
            alpha_rd=float(eda_cfg.get("alpha_rd", 0.15)),
            alpha_rs=float(eda_cfg.get("alpha_rs", 0.1)),
            num_aug_per_sample=int(eda_cfg.get("num_aug_per_sample", 2)),
            seed=seed,
        )

    # --- NEW: Diacritic removal augmentation ---
    diacritic_removal_summary: dict[str, Any] = {"enabled": False}
    if eda_cfg and bool(eda_cfg.get("enabled", False)):
        train_df, diacritic_removal_summary = augment_with_diacritic_removal(
            train_df, label2id, seed=seed,
        )

    # --- NEW: Teencode variant augmentation ---
    teencode_summary: dict[str, Any] = {"enabled": False}
    if eda_cfg and bool(eda_cfg.get("enabled", False)):
        train_df, teencode_summary = augment_with_teencode(
            train_df, label2id, seed=seed,
        )

    print(f"[TRAIN] Final training set size: {len(train_df)} samples")
    print(f"[TRAIN] Validation set size: {len(valid_df)} samples")
    print(f"[TRAIN] Label distribution: {train_df['label'].value_counts().to_dict()}")

    tokenizer = AutoTokenizer.from_pretrained(model_cfg["base_model"], trust_remote_code=True)
    from transformers import AutoConfig
    try:
        auto_config = AutoConfig.from_pretrained(model_cfg["base_model"], trust_remote_code=True)
        archs = getattr(auto_config, "architectures", []) or []
        model_type = getattr(auto_config, "model_type", "")
        
        if any("TextCNN" in a for a in archs):
            is_xlm_roberta = True
        elif any("SequenceClassification" in a for a in archs):
            is_xlm_roberta = False
        elif any("XLMRoberta" in a or "Roberta" in a for a in archs) or model_type == "xlm-roberta":
            is_xlm_roberta = True
        else:
            is_xlm_roberta = "xlm-roberta" in model_cfg["base_model"].lower()
    except Exception:
        is_xlm_roberta = "xlm-roberta" in model_cfg["base_model"].lower()

    if is_xlm_roberta:
        from src.models.classifier import XLMRobertaTextCNN
        model = XLMRobertaTextCNN.from_pretrained(
            model_cfg["base_model"],
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=True,
        )
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_cfg["base_model"],
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            trust_remote_code=False,
        )

    # Patch load_state_dict to dynamically map beta -> bias and gamma -> weight for LayerNorm
    original_load_state_dict = model.load_state_dict
    def patched_load_state_dict(state_dict, strict=True):
        new_state_dict = {}
        for k, v in state_dict.items():
            new_key = k
            if ".LayerNorm." in new_key or ".layer_norm." in new_key:
                if new_key.endswith(".beta"):
                    new_key = new_key[:-4] + "bias"
                elif new_key.endswith(".gamma"):
                    new_key = new_key[:-5] + "weight"
            new_state_dict[new_key] = v
        return original_load_state_dict(new_state_dict, strict=strict)
    model.load_state_dict = patched_load_state_dict

    max_length = int(model_cfg.get("max_length", 128))
    train_dataset = ViHSDDataset(train_df["text"].tolist(), train_df["label"].tolist(), tokenizer, max_length)
    valid_dataset = ViHSDDataset(valid_df["text"].tolist(), valid_df["label"].tolist(), tokenizer, max_length)
    class_weights = _class_weights_from_training_config(training_cfg, train_df["label"].tolist(), num_labels)

    output_dir = resolve_path(training_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    metric_for_best_model = str(training_cfg.get("metric_for_best_model", "macro_f1"))
    label_smoothing = float(training_cfg.get("label_smoothing", 0.0))
    gradient_accumulation_steps = int(training_cfg.get("gradient_accumulation_steps", 1))

    hf_token = __import__("os").environ.get("HF_TOKEN")
    push_to_hub = bool(hf_token and export_cfg.get("hf_repo_id"))
    import inspect
    training_args_kwargs = {
        "output_dir": str(output_dir),
        "learning_rate": float(training_cfg.get("learning_rate", 2e-5)),
        "per_device_train_batch_size": int(training_cfg.get("batch_size", 16)),
        "per_device_eval_batch_size": int(training_cfg.get("eval_batch_size", training_cfg.get("batch_size", 16))),
        "num_train_epochs": float(training_cfg.get("num_epochs", 3)),
        "weight_decay": float(training_cfg.get("weight_decay", 0.01)),
        "warmup_ratio": float(training_cfg.get("warmup_ratio", 0.1)),
        "save_strategy": "epoch",
        "load_best_model_at_end": True,
        "metric_for_best_model": metric_for_best_model,
        "greater_is_better": bool(training_cfg.get("greater_is_better", True)),
        "save_total_limit": int(training_cfg.get("save_total_limit", 2)),
        "fp16": bool(training_cfg.get("fp16", False)),
        "max_grad_norm": float(training_cfg.get("max_grad_norm", 1.0)),
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "label_smoothing_factor": label_smoothing,
        "report_to": [],
        "seed": seed,
        "push_to_hub": push_to_hub,
        "hub_model_id": export_cfg.get("hf_repo_id") if push_to_hub else None,
        "hub_strategy": "checkpoint",  # Chỉ đồng bộ checkpoint mới nhất lên Hub
        "hub_token": hf_token if push_to_hub else None,
    }
    
    # Resolve version compatibility for evaluation_strategy vs eval_strategy
    sig = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in sig.parameters:
        training_args_kwargs["eval_strategy"] = "epoch"
    else:
        training_args_kwargs["evaluation_strategy"] = "epoch"
        
    args = TrainingArguments(**training_args_kwargs)
    callbacks = []
    if int(training_cfg.get("early_stopping_patience", 0)) > 0:
        callbacks.append(
            EarlyStoppingCallback(
                early_stopping_patience=int(training_cfg.get("early_stopping_patience", 2))
            )
        )

    if training_cfg.get("freeze_backbone_first_epoch", False):
        from transformers import TrainerCallback
        class FreezeBackboneCallback(TrainerCallback):
            def __init__(self, model):
                self.model = model
            def on_epoch_begin(self, args, state, control, **kwargs):
                epoch = int(round(state.epoch))
                if epoch == 0:
                    print("[CALLBACK] Freezing roberta.encoder for Epoch 1...")
                    if hasattr(self.model, "roberta") and hasattr(self.model.roberta, "encoder"):
                        for param in self.model.roberta.encoder.parameters():
                            param.requires_grad = False
                else:
                    print(f"[CALLBACK] Unfreezing roberta.encoder for Epoch {epoch + 1}...")
                    if hasattr(self.model, "roberta") and hasattr(self.model.roberta, "encoder"):
                        for param in self.model.roberta.encoder.parameters():
                            param.requires_grad = True
        callbacks.append(FreezeBackboneCallback(model))

    loss_name = str(training_cfg.get("loss", "cross_entropy")).lower()
    focal_gamma = float(training_cfg.get("focal_gamma", 2.0))
    layerwise_lr_decay = float(training_cfg.get("layerwise_lr_decay", 0.0))

    if loss_name == "focal" and label_smoothing > 0.0:
        print("[WARNING] Focal Loss is incompatible with label smoothing > 0.0. Falling back to cross_entropy.")
        loss_name = "cross_entropy"

    if loss_name == "focal" and not class_weights:
        weights_by_id = compute_balanced_class_weights(train_df["label"].tolist(), num_labels)
        class_weights = [float(weights_by_id[idx]) for idx in range(num_labels)]

    # Extract sample weight if present in dataframe (for continual learning sampler)
    sample_weights = None
    if "sample_weight" in train_df.columns:
        sample_weights = train_df["sample_weight"].tolist()

    # Determine trainer class
    use_custom_trainer = bool(class_weights or loss_name == "focal" or layerwise_lr_decay > 0 or sample_weights is not None)
    trainer_cls = _custom_loss_trainer_class(Trainer) if use_custom_trainer else Trainer
    trainer_kwargs: dict[str, Any] = {}
    if use_custom_trainer:
        trainer_kwargs.update(
            {
                "class_weights": class_weights,
                "loss_name": loss_name,
                "focal_gamma": focal_gamma,
                "layerwise_lr_decay": layerwise_lr_decay,
                "sample_weights": sample_weights,
            }
        )

    # Resolve version compatibility for tokenizer vs processing_class in Trainer
    import inspect
    trainer_kwargs_main = {
        "model": model,
        "args": args,
        "train_dataset": train_dataset,
        "eval_dataset": valid_dataset,
        "compute_metrics": compute_metrics,
        "callbacks": callbacks,
    }
    
    trainer_sig = inspect.signature(Trainer.__init__)
    if "processing_class" in trainer_sig.parameters:
        trainer_kwargs_main["processing_class"] = tokenizer
    else:
        trainer_kwargs_main["tokenizer"] = tokenizer
        
    trainer_kwargs_main.update(trainer_kwargs)
    
    trainer = trainer_cls(**trainer_kwargs_main)

    output_dir = __import__("pathlib").Path(output_dir)
    resume_from_checkpoint = None
    if output_dir.exists():
        # Quét tìm tất cả các thư mục dạng checkpoint-* thực tế trong thư mục đầu ra (bao gồm cả đệ quy)
        checkpoints = [p for p in output_dir.glob("**/checkpoint-*") if p.name.split("-")[-1].isdigit()]
        if checkpoints:
            # Sắp xếp danh sách dựa trên chỉ số bước (step) huấn luyện tăng dần
            checkpoints.sort(key=lambda x: int(x.name.split("-")[-1]))
            resume_from_checkpoint = str(checkpoints[-1])

    if resume_from_checkpoint:
        print(f"[TRAIN] Phát hiện checkpoint hợp lệ. Khôi phục tiến trình từ: {resume_from_checkpoint}")
    else:
        print("[TRAIN] Không tìm thấy thư mục checkpoint-* nào. Tiến hành huấn luyện phiên mới từ Epoch 0.")
        resume_from_checkpoint = None
    
    # Bắt đầu huấn luyện và truyền cờ khôi phục
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    metrics = trainer.evaluate()

    artifact_dir = resolve_path(export_cfg["artifact_dir"])
    final_model_dir = resolve_path(export_cfg["final_model_dir"])
    final_model_dir.mkdir(parents=True, exist_ok=True)

    if is_xlm_roberta:
        from src.models.classifier import XLMRobertaTextCNN
        # Register ONLY the custom architecture, DO NOT register built-in XLMRobertaConfig
        XLMRobertaTextCNN.register_for_auto_class("AutoModelForSequenceClassification")

    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    save_label_mapping(label_mapping, artifact_dir / "label_mapping.json")
    metadata = {
        "model_version": "v2.0.0",
        "base_model": model_cfg["base_model"],
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "dataset": {
            "train_path": data_cfg["train_path"],
            "valid_path": data_cfg["valid_path"],
            "test_path": data_cfg.get("test_path"),
            "text_column": data_cfg.get("text_column", "text"),
            "label_column": data_cfg.get("label_column", "label"),
        },
        "preprocessing": {
            "use_word_segmentation": use_word_seg,
            "normalize_teencode": bool(preprocess_cfg.get("normalize_teencode", True)),
            "normalize_repeated_chars": bool(preprocess_cfg.get("normalize_repeated_chars", True)),
            "unicode_normalize": bool(preprocess_cfg.get("unicode_normalize", True)),
        },
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "loss": loss_name,
        "focal_gamma": focal_gamma if loss_name == "focal" else None,
        "label_smoothing": label_smoothing,
        "layerwise_lr_decay": layerwise_lr_decay if layerwise_lr_decay > 0 else None,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "class_weighting": training_cfg.get("class_weighting", "none"),
        "class_weights": class_weights,
        "class_oversampling": oversampling_summary,
        "robustness_augmentation": augmentation_summary,
        "contrastive_augmentation": contrastive_summary,
        "diacritic_augmentation": diacritic_summary,
        "eda_augmentation": eda_summary,
        "diacritic_removal_augmentation": diacritic_removal_summary,
        "teencode_augmentation": teencode_summary,
        "training_set_size": len(train_df),
        "validation_set_size": len(valid_df),
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


def _oversample_training_frame(
    train_df: pd.DataFrame,
    training_cfg: dict[str, Any],
    label2id: dict[str, int],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    oversampling_cfg = training_cfg.get("class_oversampling") or {}
    if not oversampling_cfg or not bool(oversampling_cfg.get("enabled", False)):
        return train_df, {"enabled": False}

    multipliers = oversampling_cfg.get("multipliers") or {}
    if not isinstance(multipliers, dict):
        raise ValueError("training.class_oversampling.multipliers must be a mapping")

    extra_frames = []
    added_by_label: dict[str, int] = {}
    for label, label_id in label2id.items():
        multiplier = float(multipliers.get(label, multipliers.get(str(label_id), 1.0)))
        if multiplier <= 0:
            raise ValueError(f"class oversampling multiplier for {label} must be positive")
        if multiplier <= 1.0:
            continue

        label_rows = train_df[train_df["label"] == int(label_id)]
        if label_rows.empty:
            continue
        extra_count = int(round((multiplier - 1.0) * len(label_rows)))
        if extra_count <= 0:
            continue
        extra = label_rows.sample(n=extra_count, replace=True, random_state=seed + int(label_id) + 1009)
        extra_frames.append(extra)
        added_by_label[label] = int(extra_count)

    if not extra_frames:
        return train_df, {"enabled": True, "added_examples": 0, "added_by_label": {}}

    augmented = pd.concat([train_df, *extra_frames], ignore_index=True)
    augmented = augmented.sample(frac=1, random_state=seed).reset_index(drop=True)
    return augmented, {
        "enabled": True,
        "added_examples": int(sum(added_by_label.values())),
        "added_by_label": added_by_label,
        "multipliers": multipliers,
    }


def _augment_training_frame(
    train_df: pd.DataFrame,
    training_cfg: dict[str, Any],
    label2id: dict[str, int],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    augmentation_cfg = training_cfg.get("robustness_augmentation") or {}
    if not augmentation_cfg or not bool(augmentation_cfg.get("enabled", False)):
        return train_df, {"enabled": False}

    repeats = int(augmentation_cfg.get("repeats", 1))
    if repeats <= 0:
        return train_df, {"enabled": False, "reason": "repeats <= 0"}

    robustness_df, summary = build_robustness_frame(label2id)
    repeated_frames = [robustness_df.copy() for _ in range(repeats)]
    augmented = pd.concat([train_df, *repeated_frames], ignore_index=True)
    augmented = augmented.sample(frac=1, random_state=seed).reset_index(drop=True)
    return augmented, {
        "enabled": True,
        "repeats": repeats,
        "added_examples": int(len(robustness_df) * repeats),
        **summary,
    }


def _augment_contrastive_frame(
    train_df: pd.DataFrame,
    training_cfg: dict[str, Any],
    label2id: dict[str, int],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    augmentation_cfg = training_cfg.get("contrastive_augmentation") or {}
    if not augmentation_cfg or not bool(augmentation_cfg.get("enabled", False)):
        return train_df, {"enabled": False}

    repeats = int(augmentation_cfg.get("repeats", 1))
    if repeats <= 0:
        return train_df, {"enabled": False, "reason": "repeats <= 0"}

    contrastive_df, summary = build_contrastive_frame(label2id)
    repeated_frames = [contrastive_df.copy() for _ in range(repeats)]
    augmented = pd.concat([train_df, *repeated_frames], ignore_index=True)
    augmented = augmented.sample(frac=1, random_state=seed + 1701).reset_index(drop=True)
    return augmented, {
        "enabled": True,
        "repeats": repeats,
        "added_examples": int(len(contrastive_df) * repeats),
        **summary,
    }


def _augment_diacritic_frame(
    train_df: pd.DataFrame,
    training_cfg: dict[str, Any],
    label2id: dict[str, int],
    *,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    augmentation_cfg = training_cfg.get("diacritic_augmentation") or {}
    if not augmentation_cfg or not bool(augmentation_cfg.get("enabled", False)):
        return train_df, {"enabled": False}

    repeats = int(augmentation_cfg.get("repeats", 1))
    if repeats <= 0:
        return train_df, {"enabled": False, "reason": "repeats <= 0"}

    diacritic_df, summary = build_diacritic_frame(label2id)
    repeated_frames = [diacritic_df.copy() for _ in range(repeats)]
    augmented = pd.concat([train_df, *repeated_frames], ignore_index=True)
    augmented = augmented.sample(frac=1, random_state=seed + 2401).reset_index(drop=True)
    return augmented, {
        "enabled": True,
        "repeats": repeats,
        "added_examples": int(len(diacritic_df) * repeats),
        **summary,
    }


def _class_weights_from_training_config(
    training_cfg: dict[str, Any],
    labels: list[int],
    num_labels: int,
) -> list[float] | None:
    mode = str(training_cfg.get("class_weighting", "none")).lower()
    if mode in {"", "none", "false", "off"}:
        return None

    manual_weights = training_cfg.get("class_weights") or {}
    if mode == "manual":
        if not isinstance(manual_weights, dict):
            raise ValueError("training.class_weights must be a mapping when class_weighting is manual")
        return _manual_class_weights(manual_weights, num_labels)

    weights_by_id = compute_balanced_class_weights(labels, num_labels)
    weights = [float(weights_by_id[idx]) for idx in range(num_labels)]
    if mode == "balanced":
        return weights
    if mode == "sqrt_balanced":
        return [weight**0.5 for weight in weights]
    raise ValueError("training.class_weighting must be one of: none, balanced, sqrt_balanced, manual")


def _manual_class_weights(raw_weights: dict[str, Any], num_labels: int) -> list[float]:
    label_names = ["CLEAN", "OFFENSIVE", "HATE"]
    weights: list[float] = []
    for idx in range(num_labels):
        label = label_names[idx] if idx < len(label_names) else str(idx)
        raw_value = raw_weights.get(label, raw_weights.get(str(idx), 1.0))
        weight = float(raw_value)
        if weight <= 0:
            raise ValueError(f"class weight for {label} must be positive")
        weights.append(weight)
    return weights


def _custom_loss_trainer_class(base_trainer):
    """Create a custom Trainer subclass with:
    - Focal loss support
    - Class weighting
    - Layer-wise learning rate decay (LLRD)
    """

    class CustomLossTrainer(base_trainer):
        def __init__(
            self,
            *args,
            class_weights: list[float] | None,
            loss_name: str,
            focal_gamma: float,
            layerwise_lr_decay: float = 0.0,
            sample_weights: list[float] | None = None,
            **kwargs,
        ) -> None:
            super().__init__(*args, **kwargs)
            self.class_weights = class_weights
            self.loss_name = loss_name
            self.focal_gamma = focal_gamma
            self.layerwise_lr_decay = layerwise_lr_decay
            self.sample_weights = sample_weights

        def get_train_dataloader(self):
            if self.train_dataset is None:
                raise ValueError("Trainer: training requires a train_dataset.")
                
            if self.sample_weights is not None:
                import torch
                from torch.utils.data import DataLoader, WeightedRandomSampler
                
                sampler = WeightedRandomSampler(
                    weights=self.sample_weights,
                    num_samples=len(self.sample_weights),
                    replacement=True
                )
                
                return DataLoader(
                    self.train_dataset,
                    batch_size=self.args.train_batch_size,
                    sampler=sampler,
                    collate_fn=self.data_collator,
                    num_workers=self.args.dataloader_num_workers,
                    pin_memory=getattr(self.args, "dataloader_pin_memory", getattr(self.args, "pin_memory", True)),
                )
            
            return super().get_train_dataloader()

        def create_optimizer(self):
            """Override to implement layer-wise learning rate decay (LLRD).

            Earlier layers get lower learning rates, later layers get higher.
            This preserves pre-trained knowledge while allowing task-specific
            adaptation in upper layers.
            """
            if self.layerwise_lr_decay <= 0 or self.layerwise_lr_decay >= 1.0:
                return super().create_optimizer()

            import torch

            model = self.model
            base_lr = self.args.learning_rate
            decay = self.layerwise_lr_decay
            weight_decay = self.args.weight_decay
            no_decay = {"bias", "LayerNorm.weight", "LayerNorm.bias", "layer_norm.weight", "layer_norm.bias"}

            optimizer_grouped_parameters = []
            num_layers = _get_num_layers(model)

            classifier_params = []
            layer_params = {i: [] for i in range(num_layers)}
            emb_params = []
            remaining_params = []

            # Sequentially categorize each parameter to guarantee zero overlaps/omissions
            for name, param in model.named_parameters():
                if not param.requires_grad:
                    continue
                
                # 1. Classifier head layers (full base_lr)
                if any(kw in name for kw in ["classifier", "pooler", "convs", "fc"]):
                    classifier_params.append((name, param))
                # 2. Encoder layers (decayed lr)
                elif any(f".layer.{i}." in name or f".layers.{i}." in name for i in range(num_layers)):
                    matched_idx = None
                    for i in range(num_layers):
                        if f".layer.{i}." in name or f".layers.{i}." in name:
                            matched_idx = i
                            break
                    if matched_idx is not None:
                        layer_params[matched_idx].append((name, param))
                    else:
                        remaining_params.append((name, param))
                # 3. Embeddings (lowest lr)
                elif "embedding" in name.lower():
                    emb_params.append((name, param))
                # 4. Remaining parameters fallback
                else:
                    remaining_params.append((name, param))

            # Helper function to split a group into decay and no_decay subgroups
            def add_group(params_list, lr):
                if not params_list:
                    return
                decay_p = []
                nodecay_p = []
                for n, p in params_list:
                    if any(nd in n for nd in no_decay):
                        nodecay_p.append(p)
                    else:
                        decay_p.append(p)
                
                if decay_p:
                    optimizer_grouped_parameters.append({
                        "params": decay_p,
                        "weight_decay": weight_decay,
                        "lr": lr,
                    })
                if nodecay_p:
                    optimizer_grouped_parameters.append({
                        "params": nodecay_p,
                        "weight_decay": 0.0,
                        "lr": lr,
                    })

            # Add classifier head parameters
            add_group(classifier_params, base_lr)

            # Add encoder layer parameters (from top layers to bottom layers)
            for layer_idx in range(num_layers - 1, -1, -1):
                layer_lr = base_lr * (decay ** (num_layers - 1 - layer_idx))
                add_group(layer_params[layer_idx], layer_lr)

            # Add embedding parameters
            emb_lr = base_lr * (decay ** num_layers)
            add_group(emb_params, emb_lr)

            # Add remaining parameters
            remaining_lr = base_lr * (decay ** (num_layers // 2))
            add_group(remaining_params, remaining_lr)

            self.optimizer = torch.optim.AdamW(optimizer_grouped_parameters)
            return self.optimizer

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            import torch
            import torch.nn.functional as F

            labels = inputs.pop("labels")
            outputs = model(**inputs)
            logits = outputs.logits
            logits = logits.view(-1, model.config.num_labels)
            labels = labels.view(-1)
            weights = None
            if self.class_weights:
                weights = torch.tensor(self.class_weights, dtype=logits.dtype, device=logits.device)

            if self.loss_name == "focal":
                log_probs = F.log_softmax(logits, dim=-1)
                nll = -log_probs.gather(dim=-1, index=labels.unsqueeze(1)).squeeze(1)
                pt = torch.exp(-nll)
                # Numerical stability guard: clamp pt to prevent negative bases or zero-gradient anomalies in pow()
                pt = torch.clamp(pt, min=1e-7, max=1.0 - 1e-7)
                if weights is not None:
                    nll = nll * weights.gather(0, labels)
                loss = ((1 - pt) ** self.focal_gamma * nll).mean()
            elif self.loss_name in {"cross_entropy", "ce"}:
                loss = F.cross_entropy(logits, labels, weight=weights, label_smoothing=self.args.label_smoothing_factor)
            else:
                raise ValueError("training.loss must be one of: cross_entropy, focal")
            return (loss, outputs) if return_outputs else loss

    return CustomLossTrainer


def _get_num_layers(model) -> int:
    """Detect number of transformer layers in the model."""
    config = getattr(model, "config", None)
    if config is None:
        return 12
    return int(getattr(config, "num_hidden_layers", 12))


def _load_frame(
    path: str,
    data_cfg: dict[str, Any],
    *,
    use_word_segmentation: bool = True,
) -> pd.DataFrame:
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
        use_word_segmentation=use_word_segmentation,
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
