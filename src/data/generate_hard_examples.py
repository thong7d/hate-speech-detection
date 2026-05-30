import os
import json
import time
from pathlib import Path
import pandas as pd
import google.generativeai as genai

def generate_hard_examples():
    # Paths (relative to repo root, resolved dynamically)
    project_root = Path(__file__).resolve().parents[2]
    error_analysis_path = project_root / "results" / "error_analysis.csv"
    train_raw_path = project_root / "data" / "raw" / "vihsd" / "train.csv"
    processed_train_path = project_root / "data" / "processed" / "train.parquet"

    print("🔍 Reading error analysis report...")
    if not error_analysis_path.exists():
        print(f"❌ Error analysis file not found at: {error_analysis_path}")
        return

    df = pd.read_csv(error_analysis_path)
    
    # Filter FALSE_NEGATIVE rows (toxic sentences predicted as CLEAN)
    fn_df = df[df["error_type"] == "FALSE_NEGATIVE"]
    if fn_df.empty:
        print("✅ No FALSE_NEGATIVE examples found to augment.")
        return

    print(f"🎯 Found {len(fn_df)} FALSE_NEGATIVE examples. Starting targeted augmentation...")

    # Load Gemini API Key from environment or google.colab.userdata
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        try:
            from google.colab import userdata
            api_key = userdata.get("GEMINI_API_KEY")
        except ImportError:
            pass

    if not api_key:
        print("⚠️ GEMINI_API_KEY not found in environment or Colab secrets. Skipping augmentation script.")
        return

    # Configure Gemini API
    genai.configure(api_key=api_key)
    # Using gemini-1.5-flash as it is fast, free/low-cost, and robust
    model = genai.GenerativeModel("gemini-1.5-flash")

    # Extract sentences and true labels
    # We will map true labels to label_id (OFFENSIVE -> 1, HATE -> 2)
    # In train.csv, columns are: free_text, label_id
    label_map = {"CLEAN": 0, "OFFENSIVE": 1, "HATE": 2}
    
    new_records = []
    
    # Process in batches of 10 to fit cleanly into prompt and respect API limits
    batch_size = 10
    fn_records = fn_df.to_dict("records")
    
    print(f"🚀 Processing in batches of {batch_size}...")
    for i in range(0, len(fn_records), batch_size):
        batch = fn_records[i : i + batch_size]
        
        batch_text_numbered = ""
        for idx, item in enumerate(batch):
            batch_text_numbered += f"{idx+1}. [{item['true_label']}] {item['text']}\n"

        prompt = f"""Hãy đóng vai trò là chuyên gia ngôn ngữ mạng xã hội tiếng Việt.
Dựa trên danh sách các câu độc hại mập mờ dưới đây, hãy sinh ra thêm 3 biến thể mang phong cách mỉa mai, nói kháy, nói ẩn dụ tương đương nhưng thay đổi từ vựng nhằm tạo mẫu thử khó (hard examples) cho nhãn OFFENSIVE hoặc HATE.

Danh sách câu gốc:
{batch_text_numbered}

Yêu cầu đầu ra:
- Trả về CHỈ một chuỗi JSON chứa danh sách các đối tượng. Không thêm markdown code blocks (như ```json) hay bất kỳ văn bản giải thích nào khác ngoài JSON.
- Mỗi đối tượng trong danh sách JSON có cấu trúc chính xác như sau:
[
  {{
    "original_text": "câu gốc",
    "variants": ["biến thể 1", "biến thể 2", "biến thể 3"],
    "label": "OFFENSIVE" hoặc "HATE"
  }}
]
"""
        print(f"   Sending batch {i//batch_size + 1}/{len(fn_records)//batch_size + 1} to Gemini API...")
        try:
            response = model.generate_content(prompt)
            response_text = response.text.strip()
            
            # Remove potential markdown block wrap
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            response_text = response_text.strip()

            batch_results = json.loads(response_text)
            for res in batch_results:
                orig = res.get("original_text")
                variants = res.get("variants", [])
                label_str = res.get("label", "OFFENSIVE")
                
                label_id = label_map.get(label_str, 1) # Default to 1 (OFFENSIVE) if mismatch
                
                for var in variants:
                    if var and len(var.strip()) >= 3:
                        new_records.append({
                            "free_text": var.strip(),
                            "label_id": label_id
                        })
        except Exception as e:
            print(f"   ⚠️ Error processing batch: {e}")
            time.sleep(2)  # Wait a bit on error
            continue
            
        time.sleep(1) # Rate limit guard

    if not new_records:
        print("❌ No augmented examples were generated successfully.")
        return

    new_df = pd.DataFrame(new_records)
    print(f"✨ Successfully generated {len(new_df)} new hard examples!")

    # Read original train.csv
    print(f"📖 Reading original training data from: {train_raw_path}")
    if not train_raw_path.exists():
        print(f"❌ Original train.csv not found at: {train_raw_path}")
        return

    train_df = pd.read_csv(train_raw_path)
    
    # Concat and save directly back
    updated_train_df = pd.concat([train_df, new_df], ignore_index=True)
    updated_train_df.to_csv(train_raw_path, index=False)
    print(f"💾 Successfully appended hard examples! New train size: {len(updated_train_df)} rows.")

    # Cache Invalidation: Delete old Parquet cache to force reprocessing
    if processed_train_path.exists():
        processed_train_path.unlink()
        print("🗑️ Đã xóa bộ nhớ đệm Parquet cũ để kích hoạt tiền xử lý lại dữ liệu mới!")

if __name__ == "__main__":
    generate_hard_examples()
