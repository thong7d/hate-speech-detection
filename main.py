# main.py
import os
import uvicorn
import logging
import gradio as gr
from fastapi.testclient import TestClient

# Cố gắng import app từ src/api/app.py
try:
    from api.app import app
except ImportError:
    from src.api.app import app

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vihsd.main")

API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))

# Khởi tạo TestClient để gọi API trực tiếp trong RAM (độ trễ bằng 0, không lỗi mạng)
client = TestClient(app)

LABEL_EMOJI = {"CLEAN": "✅ CLEAN", "OFFENSIVE": "⚠️ OFFENSIVE", "HATE": "🚫 HATE"}

def classify(text: str):
    if not text or not text.strip():
        return "Vui lòng nhập văn bản.", "", ""
    try:
        # Gọi trực tiếp bộ não FastAPI nội bộ
        resp = client.post("/predict", json={"text": text})
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        return f"❌ Lỗi xử lý: {exc}", "", ""

    label_display = LABEL_EMOJI.get(data["label"], data["label"])
    conf_display  = f"{data['confidence']:.2%}"
    scores_text   = (
        f"CLEAN:     {data['scores'].get('CLEAN', 0):.4f}\n"
        f"OFFENSIVE: {data['scores'].get('OFFENSIVE', 0):.4f}\n"
        f"HATE:      {data['scores'].get('HATE', 0):.4f}\n"
        f"\nĐộ trễ: {data['latency_ms']:.1f} ms"
    )
    if data.get("is_borderline"):
        label_display += "  ⟵ Vùng xám (Độ tự tin thấp)"

    return label_display, conf_display, scores_text

# 1. Tạo giao diện Gradio
demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(label="Văn bản đầu vào", lines=4, placeholder="Nhập câu tiếng Việt..."),
    outputs=[
        gr.Textbox(label="Dự đoán"),
        gr.Textbox(label="Độ tự tin"),
        gr.Textbox(label="Chi tiết điểm số"),
    ],
    title="🛡️ Hệ thống kiểm duyệt ngôn từ thù địch",
    description="Chạy trên một Single-Process kết hợp FastAPI và Gradio.",
    examples=[
        ["Bài viết rất hay, cảm ơn tác giả!"],
        ["Thằng ngu này, cút đi cho khuất mắt"],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never",
)

# 2. BƯỚC QUAN TRỌNG: Gắn (Mount) Gradio thẳng vào FastAPI
# Giao diện Web sẽ nằm ở trang chủ (/), API Swagger nằm ở /docs
app = gr.mount_gradio_app(app, demo, path="/ui")

if __name__ == "__main__":
    logger.info("🚀 Khởi chạy hệ thống tích hợp (FastAPI + Gradio) trên cổng %d", API_PORT)
    # Tắt tính năng reload để chống lỗi khi chạy trong Docker
    uvicorn.run(app, host=API_HOST, port=API_PORT, reload=False)