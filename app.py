import os
import gradio as gr
import torch
import pandas as pd
import time
from pathlib import Path

# Thiết lập biến môi trường để ép mô hình chạy ở chế độ huggingface
os.environ["MODEL_SOURCE"] = "huggingface"
os.environ["HF_REPO_ID"] = "thong7d/vihsd-xlmr-base-hate-speech"

_classifier_instance = None

def get_classifier():
    global _classifier_instance
    if _classifier_instance is None:
        # lazy import để tối ưu hóa cold start UI render
        from src.models.classifier import HateSpeechClassifier
        _classifier_instance = HateSpeechClassifier(
            model_source="huggingface",
            hf_repo_id="thong7d/vihsd-xlmr-base-hate-speech",
            thresholds={"CLEAN": 0.5, "OFFENSIVE": 0.38, "HATE": 0.32},
            device="cpu",
            use_word_segmentation=False
        )
    return _classifier_instance

def process_batch(file_input, progress=gr.Progress()):
    if file_input is None:
        return None, "Vui lòng tải lên tệp dữ liệu."
    
    # Đọc và giới hạn tối đa 1.000 dòng để tránh lạm dụng tài nguyên
    df_raw = pd.read_csv(file_input.name, nrows=1001)
    if len(df_raw) > 1000:
        return None, "Lỗi: Hệ thống demo giới hạn tối đa 1.000 dòng để tránh quá tải bộ nhớ."
    
    classifier = get_classifier()
    
    output_path = Path("scratch/batch_result.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Đọc chunking và suy luận ghi trực tiếp xuống file tạm trên đĩa
    chunk_size = 100
    first_chunk = True
    
    for i in range(0, len(df_raw), chunk_size):
        chunk = df_raw.iloc[i:i+chunk_size].copy()
        # Chạy suy luận cho từng mẫu trong chunk
        results = []
        for idx, row in chunk.iterrows():
            text = str(row.get("text", ""))
            pred = classifier.predict(text)
            results.append({
                "text": text,
                "label": pred["label"],
                "confidence": pred["confidence"],
                "CLEAN_prob": pred["probabilities"].get("CLEAN", 0.0),
                "OFFENSIVE_prob": pred["probabilities"].get("OFFENSIVE", 0.0),
                "HATE_prob": pred["probabilities"].get("HATE", 0.0),
            })
        
        df_chunk_res = pd.DataFrame(results)
        # Ghi nối tiếp vào file tạm (Append mode)
        df_chunk_res.to_csv(output_path, mode='a', index=False, header=first_chunk)
        first_chunk = False
        progress((i + len(chunk)) / len(df_raw), desc="Đang kiểm duyệt dữ liệu...")
    
    # Đọc lại 5 dòng đầu làm Preview
    df_preview = pd.read_csv(output_path, nrows=5)
    return str(output_path), df_preview
