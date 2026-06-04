from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from transformers import XLMRobertaPreTrainedModel, XLMRobertaModel, XLMRobertaConfig
from transformers.modeling_outputs import SequenceClassifierOutput

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


class XLMRobertaTextCNN(XLMRobertaPreTrainedModel):
    config_class = XLMRobertaConfig

    def __init__(self, config: XLMRobertaConfig):
        super().__init__(config)
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        
        hidden_size = config.hidden_size
        num_labels = config.num_labels
        num_filters = 128
        kernel_sizes = [2, 3, 4, 5]
        
        self.convs = nn.ModuleList([
            nn.Conv1d(
                in_channels=hidden_size,
                out_channels=num_filters,
                kernel_size=k,
                padding=0
            ) for k in kernel_sizes
        ])
        
        self.fc = nn.Linear(len(kernel_sizes) * num_filters, num_labels)
        self.post_init()

    def _init_weights(self, module):
        """Initialize custom Conv1d and Linear layers using the same pre-trained distribution range."""
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        output_attentions = (
            output_attentions if output_attentions is not None else self.config.output_attentions
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Trích xuất hidden states từ XLM-R: (batch_size, sequence_length, hidden_size)
        sequence_output = outputs[0]

        # Sequence Length Guard: Bảo vệ độ rộng tối thiểu của chiều chuỗi là 5
        seq_len = sequence_output.size(1)
        max_kernel = 5
        if seq_len < max_kernel:
            padding_len = max_kernel - seq_len
            pad_tensor = torch.zeros(
                sequence_output.size(0), 
                padding_len, 
                sequence_output.size(2), 
                device=sequence_output.device, 
                dtype=sequence_output.dtype
            )
            sequence_output = torch.cat([sequence_output, pad_tensor], dim=1)

        # Chuyển vị trục để phù hợp với Conv1d: (batch_size, hidden_size, sequence_length)
        x = sequence_output.transpose(1, 2)

        # Đẩy qua các tầng Conv1d song song + Max pooling qua chiều dài chuỗi
        pooled_outputs = []
        for conv in self.convs:
            c = torch.relu(conv(x))
            p = torch.max(c, dim=2)[0]
            pooled_outputs.append(p)

        # Nối đặc trưng: (batch_size, num_filters * 4)
        cat = torch.cat(pooled_outputs, dim=1)

        # Đưa qua tầng tuyến tính đầu ra: (batch_size, num_labels)
        logits = self.fc(cat)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states if getattr(outputs, "hidden_states", None) is not None else None,
            attentions=outputs.attentions if getattr(outputs, "attentions", None) is not None else None,
        )


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
    metrics["critical_f1"] = round(float((f1[1] + f1[2]) / 2), 4)
    metrics["critical_recall"] = round(float((recall[1] + recall[2]) / 2), 4)
    metrics["offensive_priority_f1"] = round(float((0.7 * f1[1]) + (0.3 * f1[2])), 4)
    metrics["offensive_priority_recall"] = round(float((0.7 * recall[1]) + (0.3 * recall[2])), 4)
    metrics["balanced_critical_f1"] = round(float((0.4 * f1[1]) + (0.4 * f1[2]) + (0.2 * metrics["macro_f1"])), 4)
    metrics["balanced_critical_recall"] = round(
        float((0.4 * recall[1]) + (0.4 * recall[2]) + (0.2 * metrics["recall_macro"])), 4
    )
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
        thresholds: dict[str, float] | None = None,
        max_length: int = 128,
        device: str = "auto",
        use_word_segmentation: bool = True,
    ) -> None:
        self.model_source = model_source
        self.hf_repo_id = hf_repo_id
        self.model_path = str(resolve_path(model_path))
        self.artifact_dir = resolve_path(artifact_dir)
        self.label_mapping_path = resolve_path(label_mapping_path or self.artifact_dir / "label_mapping.json")
        self.metadata_path = resolve_path(metadata_path or self.artifact_dir / "metadata.json")
        self.threshold = threshold
        self.thresholds = thresholds
        self.max_length = max_length
        self.device_name = device
        self.use_word_segmentation = use_word_segmentation
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
        thresholds = model_cfg.get("thresholds")
        return cls(
            model_source=env_source or source,
            model_path=env_local or model_cfg.get("local_path", "artifacts/hate_speech_model/model"),
            hf_repo_id=env_repo or model_cfg.get("hf_repo_id"),
            artifact_dir=model_cfg.get("artifact_dir", "artifacts/hate_speech_model"),
            label_mapping_path=model_cfg.get("label_mapping_path"),
            metadata_path=model_cfg.get("metadata_path"),
            threshold=float(model_cfg.get("threshold", 0.5)),
            thresholds=thresholds,
            max_length=int(model_cfg.get("max_length", 128)),
            use_word_segmentation=bool(model_cfg.get("use_word_segmentation", True)),
        )

    @property
    def model_version(self) -> str:
        return str(self.metadata.get("model_version") or self.metadata.get("version") or "v2.0.0")

    def predict(self, text: str) -> dict:
        if self.model is None or self.tokenizer is None or self.device is None:
            raise RuntimeError("Model is not loaded.")

        original_text = "" if text is None else str(text)
        model_text = preprocess_text(original_text, use_word_segmentation=self.use_word_segmentation)
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
            probabilities_tensor = torch.nn.functional.softmax(logits, dim=-1).detach().cpu()[0]

        probabilities = {
            self.id2label.get(idx, str(idx)): round(float(probabilities_tensor[idx].item()), 4)
            for idx in range(len(probabilities_tensor))
        }

        # Check if thresholds are valid for cascade matching, otherwise fallback to standard argmax
        if (
            self.thresholds is not None 
            and "HATE" in self.thresholds 
            and "OFFENSIVE" in self.thresholds
        ):
            prob_offensive = probabilities.get("OFFENSIVE", 0.0)
            prob_hate = probabilities.get("HATE", 0.0)

            thresh_hate = self.thresholds["HATE"]
            thresh_offensive = self.thresholds["OFFENSIVE"]

            if prob_hate >= thresh_hate:
                pred_id = 2
            elif prob_offensive >= thresh_offensive:
                pred_id = 1
            else:
                pred_id = 0
        else:
            # Standard fallback using argmax
            pred_id = int(torch.argmax(probabilities_tensor).item())

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
        from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

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
                subfolder = "" if source_name == "huggingface" else None

                # Đọc cấu hình AutoConfig trước
                config = AutoConfig.from_pretrained(
                    source_value,
                    subfolder=subfolder,
                    token=token,
                    trust_remote_code=True
                )

                self.tokenizer = AutoTokenizer.from_pretrained(
                    source_value,
                    subfolder=subfolder,
                    token=token,
                    trust_remote_code=True
                )

                # Nếu là mô hình xlm-roberta, tự động khởi tạo bằng XLMRobertaTextCNN với trust_remote_code=True
                if (
                    getattr(config, "model_type", None) == "xlm-roberta"
                    and getattr(config, "architectures", [None])[0] == "XLMRobertaTextCNN"
                ):
                    self.model, loading_info = XLMRobertaTextCNN.from_pretrained(
                        source_value,
                        subfolder=subfolder,
                        token=token,
                        config=config,
                        trust_remote_code=True,
                        output_loading_info=True,
                    )
                    
                    # Kiểm tra xem có thiếu trọng số head tùy chỉnh không
                    missing_keys = loading_info.get("missing_keys", [])
                    custom_head_missing = [k for k in missing_keys if any(hk in k for hk in ["convs", "fc"])]
                    if custom_head_missing:
                        import warnings
                        warnings.warn(
                            f"\n[CRITICAL WARNING] Trọng số của tầng head tùy chỉnh TextCNN {custom_head_missing} "
                            f"KHÔNG có trong checkpoint '{source_value}' và đã bị khởi tạo ngẫu nhiên! "
                            f"Hiệu suất mô hình sẽ bị sụp đổ nghiêm trọng (Weight Mismatch). "
                            f"Vui lòng đảm bảo bạn đã đẩy đầy đủ checkpoint đã fine-tune của lớp custom "
                            f"XLMRobertaTextCNN lên Hugging Face Hub hoặc thư mục cục bộ.\n",
                            RuntimeWarning
                        )
                else:
                    self.model = AutoModelForSequenceClassification.from_pretrained(
                        source_value,
                        subfolder=subfolder,
                        token=token,
                        config=config,
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
            "model_version": "v2.0.0",
            "note": "Khong du du lieu de xac minh metadata artifact.",
        }


def _env(name: str) -> str | None:
    value = __import__("os").environ.get(name)
    return value or None
