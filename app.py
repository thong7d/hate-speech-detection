import os
import gradio as gr
import torch
import pandas as pd
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

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

def predict_single(text: str):
    if not text.strip():
        return "Vui lòng nhập văn bản."
    
    classifier = get_classifier()
    from src.agent.moderator import ContentModerator, ModerationTools
    tools = ModerationTools(
        classifier=classifier,
        log_path="scratch/gradio_system_log.jsonl"
    )
    moderator = ContentModerator(
        tools=tools,
        ollama_endpoint=os.getenv("OLLAMA_ENDPOINT")
    )
    
    res = moderator.moderate(text)
    
    label = res["label"]
    confidence = res["confidence"]
    explanation = res["explanation"]
    
    result_str = f"Nhãn dự đoán: {label}\nĐộ tin cậy: {confidence:.2%}\n"
    if res["agent_triggered"]:
        result_str += f"\n🤖 [Agentic Optimization]:\n{explanation}"
    elif explanation:
        result_str += f"\n⚠️ [Borderline Fallback]:\n{explanation}"
        
    return result_str

def process_batch(file_input, progress=gr.Progress()):
    if file_input is None:
        return None, None
    
    # Đọc và giới hạn tối đa 1.000 dòng để tránh lạm dụng tài nguyên
    df_raw = pd.read_csv(file_input.name, nrows=1001)
    if len(df_raw) > 1000:
        # Trả về một dataframe lỗi
        err_df = pd.DataFrame([{"error": "Lỗi: Hệ thống giới hạn tối đa 1.000 dòng để tránh quá tải bộ nhớ."}])
        return None, err_df
    
    classifier = get_classifier()
    from src.agent.moderator import ContentModerator, ModerationTools
    tools = ModerationTools(
        classifier=classifier,
        log_path="scratch/gradio_system_log.jsonl"
    )
    moderator = ContentModerator(
        tools=tools,
        ollama_endpoint=os.getenv("OLLAMA_ENDPOINT")
    )
    
    output_path = Path("scratch/gradio_batch_result.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.exists():
        try:
            output_path.unlink()
        except Exception:
            pass
            
    text_col = "text" if "text" in df_raw.columns else df_raw.columns[0]
    
    chunk_size = 10
    first_chunk = True
    total_rows = len(df_raw)
    
    for i in range(0, total_rows, chunk_size):
        chunk = df_raw.iloc[i:i+chunk_size].copy()
        chunk_preds = []
        for idx, row in chunk.iterrows():
            text = str(row.get(text_col, ""))
            pred = classifier.predict(text)
            chunk_preds.append({
                "text": text,
                "pred": pred,
                "agent_triggered": False,
                "explanation": ""
            })
            
        borderline_items = []
        for idx_c, item in enumerate(chunk_preds):
            if item["pred"]["confidence"] < 0.65:
                borderline_items.append({
                    "id": idx_c,
                    "text": item["text"]
                })
                item["explanation"] = "LLM Refusal - Fallback to XLM-R"
                
        if borderline_items:
            chunk_preds = moderator.moderate_batch(chunk_preds, borderline_items)
            
        results = []
        for item in chunk_preds:
            res = item["pred"]
            results.append({
                "text": item["text"],
                "label": res["label"],
                "confidence": res["confidence"],
                "agent_processed": "YES" if item["agent_triggered"] else "NO",
                "explanation": item["explanation"],
                "CLEAN_prob": res["probabilities"].get("CLEAN", 0.0),
                "OFFENSIVE_prob": res["probabilities"].get("OFFENSIVE", 0.0),
                "HATE_prob": res["probabilities"].get("HATE", 0.0),
            })
            
        df_chunk_res = pd.DataFrame(results)
        df_chunk_res.to_csv(output_path, mode='a', index=False, header=first_chunk, encoding="utf-8")
        first_chunk = False
        progress((i + len(chunk)) / total_rows, desc="Đang kiểm duyệt dữ liệu...")
        
    df_preview = pd.read_csv(output_path, nrows=5)
    return str(output_path), df_preview

# Khởi dựng giao diện blocks
with gr.Blocks(title="ViHSD Content Moderation Platform") as demo:
    gr.Markdown("# 🛡️ ViHSD Content Moderation Platform (Gradio GUI)")
    gr.Markdown("Hệ thống Offline tương tác đa chiều phát hiện Hate Speech tiếng Việt, tích hợp mô hình XLM-R Base kết hợp Qwen2.5 Agentic.")
    
    with gr.Tab("💬 Kiểm duyệt đơn lẻ"):
        text_input = gr.Textbox(label="Nội dung bình luận cần kiểm tra", placeholder="Nhập bình luận tiếng Việt tại đây...", lines=4)
        single_btn = gr.Button("Bắt đầu kiểm duyệt", variant="primary")
        text_output = gr.Textbox(label="Kết quả phân tích", interactive=False)
        
        single_btn.click(fn=predict_single, inputs=text_input, outputs=text_output)
        
    with gr.Tab("📁 Kiểm duyệt hàng loạt (Batch CSV)"):
        gr.Markdown("Tải lên tệp CSV chứa cột dữ liệu tiếng Việt (sử dụng cột 'text' hoặc cột đầu tiên) để thực thi phân loại hàng loạt.")
        file_input = gr.File(label="Tải lên tệp CSV dữ liệu", file_types=[".csv"])
        batch_btn = gr.Button("Bắt đầu xử lý hàng loạt", variant="primary")
        
        with gr.Row():
            file_output = gr.File(label="Tải xuống tệp kết quả")
            preview_output = gr.DataFrame(label="Xem trước 5 dòng kết quả đầu tiên")
            
        batch_btn.click(
            fn=process_batch, 
            inputs=file_input, 
            outputs=[file_output, preview_output]
        )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
