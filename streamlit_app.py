import os
import sys
import time
import uuid
import threading
from pathlib import Path
import pandas as pd
import streamlit as st
import plotly.express as px
from dotenv import load_dotenv
# Bổ sung hàm trích xuất và gắn ngữ cảnh luồng chạy
from streamlit.runtime.scriptrunner import add_script_run_ctx, get_script_run_ctx

import warnings
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Nạp các biến môi trường từ file .env cục bộ (Lấy GEMINI_API_KEY)
load_dotenv()

# Ensure the src/ package structure is in the python path
sys.path.insert(0, str(Path(__file__).resolve().parent))

os.environ["MODEL_SOURCE"] = "huggingface"
os.environ["HF_REPO_ID"] = "thong7d/vihsd-xlmr-base-hate-speech"

os.environ["TOKENIZERS_PARALLELISM"] = "false"

st.set_page_config(
    page_title="ViHSD Local GUI - Offline Serving",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==============================================================================
# 1. THREAD-SAFE PROGRESS REGISTRY (PLAIN PYTHON OBJECT)
# ==============================================================================
class InferenceProgressRegistry:
    def __init__(self):
        self._lock = threading.Lock()
        self._data = {}

    def init_session(self, session_id: str):
        with self._lock:
            if session_id not in self._data:
                self._data[session_id] = {
                    "in_progress": False,
                    "status": "IDLE",
                    "progress_pct": 0.0,
                    "processed_count": 0,
                    "total_count": 0,
                    "eta_str": "Đang khởi động...",
                    "preview_df": None,
                    "output_path": None,
                    "error_msg": None,
                    "agent_routed_count": 0,
                    "routing_ratio": 0.0
                }

    def update(self, session_id: str, **kwargs):
        with self._lock:
            if session_id not in self._data:
                self._data[session_id] = {}
            self._data[session_id].update(kwargs)

    def get(self, session_id: str) -> dict:
        with self._lock:
            return dict(self._data.get(session_id, {}))

    def reset(self, session_id: str):
        with self._lock:
            if session_id in self._data:
                self._data[session_id] = {
                    "in_progress": False,
                    "status": "IDLE",
                    "progress_pct": 0.0,
                    "processed_count": 0,
                    "total_count": 0,
                    "eta_str": "Đang khởi động...",
                    "preview_df": None,
                    "output_path": None,
                    "error_msg": None,
                    "agent_routed_count": 0,
                    "routing_ratio": 0.0
                }

progress_registry = InferenceProgressRegistry()

# ==============================================================================
# 2. SINGLETON MODEL LAZY LOADING
# ==============================================================================
@st.cache_resource(show_spinner="Đang nạp mô hình XLM-R Base (1.1GB) vào RAM/VRAM... Vui lòng đợi trong giây lát.")
def get_classifier():
    from src.models.classifier import HateSpeechClassifier
    classifier = HateSpeechClassifier(
        model_source="huggingface",
        hf_repo_id="thong7d/vihsd-xlmr-base-hate-speech",
        thresholds={"CLEAN": 0.5, "OFFENSIVE": 0.38, "HATE": 0.32},
        device="auto",
        use_word_segmentation=False
    )
    return classifier

# ==============================================================================
# INTEGRATED INFRASTRUCTURE: CORE AGENTIC PIPELINE INTERFACE
# ==============================================================================
def execute_hybrid_inference(text: str, classifier):
    pred = classifier.predict(text)
    
    label = pred["label"]
    probs = pred["probabilities"]
    confidence = pred["confidence"]
    agent_triggered = False
    explanation = ""

    # Kích hoạt luồng xử lý Agentic khi mô hình rơi vào vùng xám có độ tự tin thấp < 65%
    if confidence < 0.65:
        explanation = "LLM Refusal - Fallback to XLM-R"
        ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
        if ollama_endpoint:
            try:
                import requests
                import json
                system_prompt = 'Bạn là chuyên gia kiểm duyệt nội dung của hệ thống ViHSD. Hãy phân tích sắc thái ngữ nghĩa (mỉa mai, châm biếm, từ lóng) của văn bản đầu vào. Chỉ trả về chuỗi JSON duy nhất theo định dạng bắt buộc, không kèm lời dẫn: {"final_label": "CLEAN"|"OFFENSIVE"|"HATE", "explanation": "Lý do ngắn gọn"}'
                payload = {
                    "model": "qwen2.5:7b-instruct",
                    "system": system_prompt,
                    "prompt": text,
                    "format": "json",
                    "stream": False
                }
                url = f"{ollama_endpoint.rstrip('/')}/api/generate"
                response = requests.post(url, json=payload, timeout=15)
                if response.status_code == 200:
                    res_json = response.json()
                    response_text = res_json.get("response", "").strip()
                    try:
                        agent_res = json.loads(response_text)
                        if isinstance(agent_res, dict) and "final_label" in agent_res:
                            final_lbl = agent_res["final_label"]
                            if final_lbl in ["CLEAN", "OFFENSIVE", "HATE"]:
                                label = final_lbl
                                explanation = agent_res.get("explanation", "Agent đã tối ưu nhãn dựa trên ngữ cảnh sâu.")
                                agent_triggered = True
                    except json.JSONDecodeError:
                        pass
            except Exception:
                pass # Fallback giữ nguyên nhãn thô của XLM-R nếu API Agent gặp lỗi ngoại lệ

    return {
        "text": text,
        "label": label,
        "confidence": confidence,
        "probabilities": probs,
        "agent_triggered": agent_triggered,
        "explanation": explanation
    }

# ==============================================================================
# 3. TRY-EXCEPT-FINALLY HEARTBEAT BACKGROUND BATCH WORKER
# ==============================================================================
# Bổ sung tham số classifier vào định nghĩa hàm
def background_batch_inference_worker(session_id: str, df_raw: pd.DataFrame, output_path: str, classifier):
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    
    progress_registry.update(
        session_id,
        in_progress=True,
        status="RUNNING",
        processed_count=0,
        total_count=len(df_raw),
        progress_pct=0.0,
        eta_str="Đang xử lý dữ liệu...",
        preview_df=None,
        error_msg=None,
        agent_routed_count=0,
        routing_ratio=0.0
    )
    
    try:
        text_col = "text" if "text" in df_raw.columns else df_raw.columns[0]
        
        chunk_size = 10
        total_rows = len(df_raw)
        processed = 0
        first_chunk = True
        
        alpha = 0.15
        ema_latency = 0.20
        agent_routed_count = 0
        
        for start_idx in range(0, total_rows, chunk_size):
            chunk = df_raw.iloc[start_idx:start_idx+chunk_size].copy()
            chunk_start_time = time.time()
            
            # Step 1: Run predictions through XLM-R Base for all rows in the chunk
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
            
            # Step 2: Filter borderline texts (confidence < 0.65)
            borderline_items = []
            for i, item in enumerate(chunk_preds):
                if item["pred"]["confidence"] < 0.65:
                    borderline_items.append({
                        "id": i,
                        "text": item["text"]
                    })
                    item["explanation"] = "LLM Refusal - Fallback to XLM-R"
            
            # Step 3: Call Ollama API in batch (JSON Array) for borderline texts
            if borderline_items:
                ollama_endpoint = os.getenv("OLLAMA_ENDPOINT")
                if ollama_endpoint:
                    try:
                        import requests
                        import json
                        system_prompt = (
                            "Bạn là chuyên gia của hệ thống ViHSD. Hãy phân tích mảng dữ liệu đầu vào và trả về mảng kết quả JSON tương ứng. "
                            "Định dạng bắt buộc: [{\"id\": int, \"final_label\": \"CLEAN\"|\"OFFENSIVE\"|\"HATE\", \"explanation\": \"string\"}]"
                        )
                        prompt_str = json.dumps(borderline_items, ensure_ascii=False)
                        payload = {
                            "model": "qwen2.5:7b-instruct",
                            "system": system_prompt,
                            "prompt": prompt_str,
                            "format": "json",
                            "stream": False
                        }
                        url = f"{ollama_endpoint.rstrip('/')}/api/generate"
                        response = requests.post(url, json=payload, timeout=15)
                        if response.status_code == 200:
                            res_json = response.json()
                            response_text = res_json.get("response", "").strip()
                            try:
                                agent_results = json.loads(response_text)
                                if isinstance(agent_results, list):
                                    for res_item in agent_results:
                                        if isinstance(res_item, dict) and "id" in res_item and "final_label" in res_item:
                                            orig_id = res_item["id"]
                                            final_lbl = res_item["final_label"]
                                            if 0 <= orig_id < len(chunk_preds) and final_lbl in ["CLEAN", "OFFENSIVE", "HATE"]:
                                                chunk_preds[orig_id]["pred"]["label"] = final_lbl
                                                chunk_preds[orig_id]["explanation"] = res_item.get("explanation", "Agent đã tối ưu nhãn dựa trên ngữ cảnh sâu.")
                                                chunk_preds[orig_id]["agent_triggered"] = True
                                                agent_routed_count += 1
                            except json.JSONDecodeError:
                                pass
                    except Exception:
                        pass # Fallback giữ nguyên nhãn thô của XLM-R
            
            # Step 4: Map predictions to results list
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
            
            processed += len(chunk)
            progress_pct = min(float(processed) / total_rows, 1.0)
            
            # Calculate EWMA latency per comment (chunk duration divided by chunk size)
            chunk_duration = time.time() - chunk_start_time
            c_latency = chunk_duration / len(chunk)
            ema_latency = alpha * c_latency + (1.0 - alpha) * ema_latency
            
            remaining_rows = total_rows - processed
            remaining_time_secs = remaining_rows * ema_latency
            
            if remaining_time_secs > 60:
                eta_str = f"{int(remaining_time_secs // 60)} phút {int(remaining_time_secs % 60)} giây"
            else:
                eta_str = f"{int(remaining_time_secs)} giây"
                
            # Kiểm tra an toàn sự tồn tại và dung lượng tệp tin trước khi đọc preview
            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                df_preview = pd.read_csv(output_path, nrows=5)
            else:
                df_preview = None
            
            routing_ratio = float(agent_routed_count) / processed if processed > 0 else 0.0
            
            progress_registry.update(
                session_id,
                progress_pct=progress_pct,
                processed_count=processed,
                eta_str=eta_str,
                preview_df=df_preview,
                agent_routed_count=agent_routed_count,
                routing_ratio=routing_ratio
            )
            
        progress_registry.update(
            session_id,
            in_progress=False,
            status="SUCCESS",
            output_path=output_path
        )
        
    except Exception as exc:
        progress_registry.update(
            session_id,
            in_progress=False,
            status="FAILED",
            error_msg=str(exc)
        )

# ==============================================================================
# 4. STREAMLIT RENDERING & UI CONTROL LAYOUT
# ==============================================================================

st.markdown("""
<style>
    .badge {
        font-family: 'Inter', sans-serif;
        font-size: 14px;
        font-weight: bold;
        padding: 6px 12px;
        border-radius: 6px;
        display: inline-block;
        text-align: center;
        margin: 4px 0px;
    }
    .badge-clean {
        background-color: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    .badge-offensive {
        background-color: rgba(245, 158, 11, 0.15);
        color: #f59e0b;
        border: 1px solid rgba(245, 158, 11, 0.3);
    }
    .badge-hate {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

def render_badge(label: str) -> str:
    if label == "HATE":
        return f'<span class="badge badge-hate">🛡️ HATE (Thù ghét)</span>'
    elif label == "OFFENSIVE":
        return f'<span class="badge badge-offensive">⚠️ OFFENSIVE (Xúc phạm)</span>'
    else:
        return f'<span class="badge badge-clean">✅ CLEAN (An toàn)</span>'

if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
progress_registry.init_session(st.session_state.session_id)

session_id = st.session_state.session_id

with st.sidebar:
    st.image("https://huggingface.co/front/assets/huggingface_logo-noborder.svg", width=60)
    st.title("ViHSD Offline GUI")
    st.markdown("---")
    st.markdown("**Kiến trúc mạng:** standard XLM-R Base")
    st.markdown("**Phiên bản:** `v2.0.0` (Production)")
    st.markdown("**Bộ lọc:** Cascade Thresholds + Agent")
    st.markdown("---")
    st.caption(f"Session ID: `{session_id[:8]}...`")

st.title("🛡️ Hệ Thống Kiểm Duyệt Phát Ngôn Thù Ghét Tiếng Việt (ViHSD)")
st.caption("Giao diện tương tác cục bộ tích hợp bộ đăng ký luồng đa nhiệm an toàn và xử lý Agentic vùng xám.")

tab1, tab2, tab3 = st.tabs([
    "💬 Kiểm duyệt đơn lẻ", 
    "📁 Kiểm duyệt hàng loạt (Batch)", 
    "📈 Báo cáo & Insights mô hình"
])

# TAB 1: SINGLE COMMENT INFERENCE
with tab1:
    st.header("Kiểm duyệt bình luận đơn lẻ")
    st.markdown("Nhập văn bản cần kiểm duyệt để phân tích mức độ thù ghét/xúc phạm của nội dung.")
    
    examples = [
        "Học máy là một lĩnh vực cực kỳ thú vị và nhiều tiềm năng phát triển.",
        "Đồ óc heo, có mỗi việc đơn giản thế này mà cũng làm không xong!",
        "Lũ mọi rợ này đáng bị trục xuất ra khỏi đất nước ngay lập tức!"
    ]
    
    selected_example = st.selectbox("Chọn câu ví dụ mẫu nhanh:", ["-- Tự nhập --"] + examples)
    default_text = "" if selected_example == "-- Tự nhập --" else selected_example
    
    with st.form("single_predict_form"):
        input_text = st.text_area("Nội dung bình luận:", value=default_text, height=120, placeholder="Nhập bình luận tiếng Việt tại đây...")
        submit_btn = st.form_submit_button("Bắt đầu kiểm duyệt")
        
    if submit_btn and input_text.strip():
        classifier = get_classifier()
        
        start_t = time.time()
        # Gọi luồng xử lý hỗn hợp (Core + Agentic)
        prediction = execute_hybrid_inference(input_text, classifier)
        latency_ms = (time.time() - start_t) * 1000
        
        st.markdown("### Kết quả phân tích:")
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown(f"**Văn bản gốc:** *\"{prediction['text']}\"*")
            badge_html = render_badge(prediction['label'])
            st.markdown(f"**Nhãn dự đoán:** {badge_html}", unsafe_allow_html=True)
            st.markdown(f"**Độ tin cậy (Confidence):** `{prediction['confidence']:.2%}`")
            st.markdown(f"**Thời gian xử lý:** `{latency_ms:.2f} ms`")
            
            if prediction["agent_triggered"]:
                st.info(f"🤖 **Luồng phân tích Agentic phản biện vùng xám:**\n{prediction['explanation']}")
            elif prediction["confidence"] < 0.65:
                st.warning(f"⚠️ **Vùng xám phát hiện:** Không thể kích hoạt LLM ({prediction['explanation']}). Đang sử dụng nhãn mặc định từ XLM-R.")
            
        with col2:
            st.markdown("**Phân phối xác suất mềm từ XLM-R Base:**")
            probs = prediction["probabilities"]
            df_probs = pd.DataFrame([
                {"Lớp nhãn": k, "Xác suất": v} for k, v in probs.items()
            ])
            
            fig = px.bar(
                df_probs,
                x="Xác suất",
                y="Lớp nhãn",
                orientation="h",
                color="Lớp nhãn",
                color_discrete_map={"CLEAN": "#10b981", "OFFENSIVE": "#f59e0b", "HATE": "#ef4444"},
                range_x=[0.0, 1.0],
                text_auto=".2%",
                height=180
            )
            fig.update_layout(
                showlegend=False,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis=dict(tickformat=".0%"),
                yaxis=dict(autorange="reversed")
            )
            st.plotly_chart(fig, use_container_width=True)

# TAB 2: BATCH COMMENT INFERENCE
with tab2:
    st.header("Kiểm duyệt hàng loạt (Batch CSV)")
    st.markdown("Tải lên tệp tin CSV để phân loại hàng loạt.")
    
    # Sửa lỗi AttributeError: Thay st.fragment bằng st.experimental_fragment phục vụ bản 1.35.0
    @st.experimental_fragment(run_every=1.0)
    def render_batch_progress_fragment(session_id: str):
        progress_data = progress_registry.get(session_id)
        
        if progress_data.get("status") == "RUNNING":
            st.warning("🔄 Hệ thống đang xử lý dữ liệu ngầm dưới nền...")
            progress_pct = progress_data.get("progress_pct", 0.0)
            st.progress(progress_pct)
            
            st.write(
                f"**Đã xử lý:** `{progress_data.get('processed_count', 0)}` / `{progress_data.get('total_count', 0)}` dòng. "
                f"**Thời gian dự kiến còn lại (Adaptive ETA):** `{progress_data.get('eta_str', '')}`"
            )
            
            # Hiển thị tỷ lệ định tuyến thời gian thực (Routing Ratio)
            routing_ratio = progress_data.get("routing_ratio", 0.0)
            routed_count = progress_data.get("agent_routed_count", 0)
            
            col_metric1, col_metric2 = st.columns(2)
            with col_metric1:
                st.metric(
                    label="Tỷ lệ rẽ nhánh sang LLM (Routing Ratio)",
                    value=f"{routing_ratio:.2%}",
                    help="Tỷ lệ số lượng bình luận rơi vào vùng tự tin thấp < 65% được chuyển tiếp sang Qwen2.5."
                )
            with col_metric2:
                st.metric(
                    label="Số câu gửi sang LLM",
                    value=f"{routed_count} câu"
                )
            
            if routing_ratio > 0.35:
                st.warning(
                    f"⚠️ Cảnh báo: Tỷ lệ định tuyến sang LLM ({routing_ratio:.2%}) đã vượt ngưỡng an toàn (35%). "
                    "Hạ tầng API Qwen2.5/Colab có nguy cơ bị chậm do lượng câu vùng xám tăng cao."
                )
            
            df_preview = progress_data.get("preview_df")
            if df_preview is not None:
                st.markdown("**Xem trước kết quả tạm thời (5 dòng đầu tiên):**")
                st.dataframe(df_preview, use_container_width=True)
                
        elif progress_data.get("status") == "SUCCESS":
            st.success("✅ Quá trình kiểm duyệt hàng loạt hoàn tất thành công!")
            
            # Hiển thị tỷ lệ định tuyến cuối cùng
            routing_ratio = progress_data.get("routing_ratio", 0.0)
            routed_count = progress_data.get("agent_routed_count", 0)
            st.info(
                f"📊 **Thống kê phiên làm việc:** Đã định tuyến tổng cộng **{routed_count}** bình luận sang Qwen2.5 "
                f"(chiếm **{routing_ratio:.2%}** tổng số dữ liệu)."
            )
            
            output_path = progress_data.get("output_path")
            if output_path and Path(output_path).exists():
                df_res = pd.read_csv(output_path)
                st.markdown("**Xem trước 5 dòng đầu tiên của kết quả:**")
                st.dataframe(df_res.head(5), use_container_width=True)
                
                with open(output_path, "r", encoding="utf-8") as f:
                    st.download_button(
                        label="📥 Tải xuống tệp báo cáo phân loại CSV hoàn chỉnh",
                        data=f.read(),
                        file_name=Path(output_path).name,
                        mime="text/csv"
                    )
            
            if st.button("🔄 Thực hiện phiên làm việc mới"):
                if output_path and Path(output_path).exists():
                    try:
                        os.remove(output_path)
                    except Exception:
                        pass
                progress_registry.reset(session_id)
                st.rerun()
                
        elif progress_data.get("status") == "FAILED":
            st.error(f"❌ Tiến trình suy luận gặp lỗi: {progress_data.get('error_msg')}")
            if st.button("🔄 Thử lại phiên làm việc mới"):
                progress_registry.reset(session_id)
                st.rerun()
                
        else:
            st.info("Hệ thống chấp nhận file CSV có trường văn bản tại cột tên là `text`. Giới hạn cứng tối đa 1.000 dòng.")
            uploaded_file = st.file_uploader("Chọn tệp CSV dữ liệu:", type=["csv"], key="batch_uploader")
            
            if uploaded_file is not None:
                try:
                    df_check = pd.read_csv(uploaded_file, nrows=1002)
                    text_col = "text" if "text" in df_check.columns else df_check.columns[0]
                    
                    if "text" not in df_check.columns:
                        st.warning(f"⚠️ Không tìm thấy cột `text`. Hệ thống sẽ mặc định nhận diện trường văn bản tại cột đầu tiên: `{text_col}`")
                    
                    if len(df_check) > 1000:
                        st.error("❌ Lỗi: Bản demo local giới hạn tối đa 1.000 dòng dữ liệu để tránh quá tải bộ nhớ RAM.")
                    else:
                        st.success(f"Tệp tải lên hợp lệ. Phát hiện `{len(df_check)}` dòng cần xử lý.")
                        
                        if st.button("🚀 Bắt đầu xử lý hàng loạt"):
                            # Nạp sẵn mô hình tại luồng chính để ngăn chặn deadlock
                            classifier = get_classifier()
                            
                            uploaded_file.seek(0)
                            df_raw = pd.read_csv(uploaded_file)
                            
                            Path("scratch").mkdir(exist_ok=True)
                            timestamp = int(time.time())
                            output_filename = f"scratch/streamlit_batch_results_{session_id}_{timestamp}.csv"
                            output_path = str(Path(output_filename).resolve())
                            
                            # 1. Trích xuất ngữ cảnh chạy của luồng chính (Main UI Thread)
                            # 1. Trích xuất ngữ cảnh chạy của luồng chính (Main UI Thread)
                            ctx = get_script_run_ctx()
                            
                            t = threading.Thread(
                                target=background_batch_inference_worker,
                                args=(session_id, df_raw, output_path, classifier)
                            )
                            # [QUAN TRỌNG] Tắt chế độ Daemon. Chế độ này khiến luồng bị hệ thống giết ngầm khi UI thay đổi.
                            t.daemon = False 
                            
                            # 2. Gắn ngữ cảnh luồng chính vào luồng phụ trước khi kích hoạt .start()
                            add_script_run_ctx(t, ctx)
                            
                            t.start()
                            # KHÔNG DÙNG st.rerun() Ở ĐÂY. Thay bằng Toast. Fragment sẽ tự động cập nhật tiến trình.
                            st.toast("Khởi động luồng xử lý nền thành công! Vui lòng chờ...", icon="🚀")
                            
                except Exception as e:
                    st.error(f"Định dạng tệp tải lên không hợp lệ: {e}")
                    
    render_batch_progress_fragment(session_id)

# TAB 3: PROJECT INFORMATION & INSIGHTS
with tab3:
    st.header("Báo cáo và Thông tin Mô hình")
    col_inf1, col_inf2 = st.columns([1, 1])
    
    with col_inf1:
        st.markdown("### Chỉ số đánh giá XLM-R Base chuẩn (Phân loại tự nhiên)")
        st.markdown("""
        | Chỉ số | Baseline (TF-IDF + LR) | XLM-R Base Tiêu chuẩn (Thực tế) |
        | :--- | :---: | :---: |
        | **Accuracy** | `80.50%` | **`86.74%`** |
        | **Macro F1** | `62.50%` | **`64.61%`** |
        | **Weighted F1** | `82.40%` | **`86.54%`** |
        | **F1 CLEAN** | `89.70%` | **`93.62%`** |
        | **F1 OFFENSIVE** | **`41.20%`** | `41.13%` |
        | **F1 HATE** | `56.60%` | **`59.09%`** |
        | **AUC-ROC Macro** | `N/A` | **`87.88%`** |
        """)
        
    with col_inf2:
        st.markdown("### Cơ chế ra quyết định: Cascade Thresholds")
        st.markdown("""
        Để tối ưu hóa độ nhạy đối với các bình luận xúc phạm và thù ghét, hệ thống tích hợp bộ **Ngưỡng thác đa tầng tĩnh (Cascade Thresholds Layer)** kết hợp rẽ nhánh Agent bảo vệ:
        
        1. **HATE Ngưỡng $\ge 0.32$**: Nếu xác suất lớp HATE đạt trên 32%, gán nhãn thù ghét.
        2. **OFFENSIVE Ngưỡng $\ge 0.38$**: Kiểm tra nếu xác suất lớp OFFENSIVE đạt trên 38%, gán nhãn xúc phạm.
        3. **Vùng xám bảo vệ**: Nếu xác suất rơi vào miền tranh chấp có độ tự tin thấp, cơ chế **Agentic** rẽ nhánh sử dụng mô hình ngôn ngữ lớn hậu xử lý sâu.
        """)