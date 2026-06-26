import pandas as pd
from tqdm import tqdm
from src.models.classifier import HateSpeechClassifier

# 1. Khởi tạo mô hình
classifier = HateSpeechClassifier(
    model_source="huggingface",
    hf_repo_id="thong7d/vihsd-xlmr-base-hate-speech",
    device="auto",
    use_word_segmentation=False
)

# 2. Đọc file CSV đầu vào (ví dụ input.csv có cột 'text')
input_csv = "testdemo.csv"  # Thay bằng đường dẫn file của bạn
df = pd.read_csv(input_csv)

# 3. Tạo danh sách chứa kết quả
output_rows = []

print("🚀 Đang chạy phân tích hàng loạt với Integrated Gradients...")
for idx, row in tqdm(df.iterrows(), total=len(df)):
    text = str(row["text"])
    
    # Chạy phân tích sâu (sử dụng Integrated Gradients để trích xuất từ khóa chính xác nhất)
    res = classifier.predict(text, mode="analysis")
    
    # Định dạng chuỗi spans hiển thị
    spans = res.get("toxic_spans", [])
    spans_str = ", ".join([f"{item['token']}({item['score']:.4f})" for item in spans])
    
    output_rows.append({
        "text": text,
        "label": res["label"],
        "confidence": res["confidence"],
        "toxicity_score": res.get("toxicity_score", 0.0),
        "toxic_spans_highlight": spans_str
    })

# 4. Lưu ra file CSV mới
df_out = pd.DataFrame(output_rows)
df_out.to_csv("results/batch_analysis_results.csv", index=False, encoding="utf-8-sig")
print("✅ Hoàn tất! Kết quả đã được lưu tại: results/batch_analysis_results.csv")
