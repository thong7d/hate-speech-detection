# Chiến lược Giám sát và Học liên tục (Continual Learning & Monitoring Strategy)

Tài liệu này mô tả chi tiết chiến lược học liên tục (Continual Learning - CL) và hệ thống giám sát trực tuyến để duy trì hiệu năng ổn định của mô hình XLM-R Base phân loại ngôn từ thù ghét trên môi trường sản phẩm thực tế. Tài liệu được thiết kế theo các tiêu chuẩn MLOps sản xuất, giải quyết triệt để các rủi ro hệ thống ở cấp độ concurrency (đồng thời), hệ điều hành Windows và kiến trúc quản lý bộ nhớ RAM.

---

## 1. Pipeline Thực hiện Học liên tục qua Google Colab (Chu trình MLOps)

Để đảm bảo tính tái lập (reproducibility) và tính ổn định cao, hệ thống thiết kế pipeline học liên tục theo kiến trúc **MLOps Hybrid (Colab GPU + Local CPU Serving)**:

```
[Nhánh Git: cl-pipeline] ------------> Git clone trực tiếp trên Colab GPU
[Drive: /MyDrive/CL_data/] ----------> Mount trực tiếp làm nguồn dữ liệu thô
                                               |
                                               v
                                [Google Colab (Môi trường GPU)]
                                - Chạy CL (WeightedSampler, Soft targets, device="cuda")
                                - Đóng gói kết quả thành CL_output.zip đẩy lên Drive
                                - Tạo tệp trống CL_output.zip.success báo hiệu hoàn tất 100%
                                               |
                                               v
[Máy Local CPU] <------------------- Phát hiện file .success xuất hiện
- Chạy deploy_cl_model.py
- Giải nén vào model_staging_<timestamp>/ (Tránh tranh chấp I/O đệm tĩnh)
- Khởi chạy Load Test kiểm thử tính toàn vẹn (Chống lệch môi trường)
- Thao tác Nguyên tử Windows: Đổi tên thư mục bằng os.replace() + 3 lần Retry
  - Nếu thất bại hoàn toàn -> Thư mục đệm động bị cô lập, không gây nghẽn chu kỳ sau
- Bật cờ is_reloading = True trước khối lock của reload_model (Tránh nghẽn predict song song)
- Giải phóng RAM/VRAM cũ chủ động (self.model = None, gc.collect())
- Bảo vệ đa luồng bằng threading.Lock 
- Nạp trọng số mới & Cập nhật active_version.json (Hot-Reloading an toàn)
```

### Bước 1: Khớp nối dữ liệu & Tạo tập dữ liệu CL (Validation hỗn hợp)
1. **Khớp nối nội bộ (Inner Merge)**: Hợp nhất `02_train_text.csv` và `03_train_label.csv` của VLSP-2019 theo cột `id`. Gán nhãn `0` -> `CLEAN`, `1` -> `OFFENSIVE`, `2` -> `HATE`. Thêm trường `source` = `"vlsp"`.
2. **Rehearsal Buffer**: Lấy mẫu phân tầng 2,500 dòng từ tập huấn luyện ViHSD gốc (`data/processed/train.parquet`). Thêm trường `source` = `"vihsd"`.
3. **Gộp tập Huấn luyện**: Ghép (Concatenate) dữ liệu VLSP và Rehearsal Buffer thành tập huấn luyện liên tục `data/processed/continual_train.parquet`.
4. **Hóa giải Validation Strategy Bias (Tập xác thực hỗn hợp)**: Trộn phân tầng tỷ lệ 50/50 giữa `vlsp_dev.parquet` (500 dòng) và tập dev gốc ViHSD `data/processed/dev.parquet` (500 dòng) để tạo thành `continual_dev.parquet`.

### Bước 2: DataLoader với `WeightedRandomSampler` chống Lệch phân phối Rehearsal
Sử dụng PyTorch `WeightedRandomSampler` để kiểm soát tỷ lệ trộn ở mức **từng Batch huấn luyện**:
* Trọng số mẫu ViHSD = $\frac{0.25}{N_{\text{vihsd\_train}}}$
* Trọng số mẫu VLSP = $\frac{0.75}{N_{\text{vlsp\_train}}}$
Mỗi Batch có kích thước 32 mẫu được bốc lên sẽ chứa trung bình **8 mẫu ViHSD (25%) và 24 mẫu VLSP (75%)**, đảm bảo gradient bộ nhớ cũ được cập nhật liên tục và đồng đều trong mỗi bước tối ưu.

### Bước 3: Hóa giải xung đột toán học Focal Loss và Label Smoothing
Khi tiến hành chu kỳ học liên tục có kích hoạt Label Smoothing ($\alpha=0.15$), hệ thống tự động tạm thời chuyển đổi hàm mất mát từ Focal Loss về **Cross-Entropy Loss tiêu chuẩn** hỗ trợ nhãn mềm (Soft Targets) để đảm bảo gradient hội tụ ổn định và chính xác.

### Bước 4: Chạy huấn luyện và Đóng gói mô hình trên Colab
1. Huấn luyện mô hình gia tăng trên Colab GPU trong **3 epochs** với tốc độ học thấp $LR = 1.0 \times 10^{-5}$.
2. Chạy đánh giá Gatekeeper trên tập test ViHSD gốc, yêu cầu Macro F1 không suy giảm quá 1% so với baseline cũ (F1 $\ge 64.61\%$).
3. Chạy recalibrate nhiệt độ ECE trên tập xác thực hỗn hợp để cập nhật nhiệt độ $T$ mới vào `metadata.json`.
4. Nén thư mục phiên bản mới thành `CL_output.zip` và copy lên Google Drive.
5. **Cơ chế tệp đánh dấu `.success` (Tránh đọc tệp ZIP chưa tải xong)**: Colab sau khi upload xong 100% tệp `CL_output.zip` sẽ tạo một tệp trống tên là `CL_output.zip.success` và lưu cùng thư mục trên Google Drive. Tiến trình local chỉ kích hoạt khi thấy file `.success` này.

### Bước 5: Triển khai nguyên tử và Hot-Reloading chống sụt RAM trên Local CPU
Tiến trình giải nén và nạp được tự động hóa qua script [deploy_cl_model.py](file:///d:/000MINHTHONG/Junior%20-%20Semester%20II/ANLPB/FinalProject/hate-speech-detection/src/export/deploy_cl_model.py):
1. **Giải nén vào thư mục Đệm Động (Dynamic Staging)**: 
   * Giải nén file `CL_output.zip` vào thư mục đệm có tên động chứa dấu thời gian: `artifacts/hate_speech_model/model_staging_<timestamp>/`.
   * **Giải phóng hoàn toàn Deadlock Windows**: Trên Windows, khi một file bên trong thư mục đang bị hệ thống hoặc chương trình diệt virus khóa, việc di chuyển hay đổi tên thư mục đệm tĩnh là bất khả thi. Nhờ cơ chế thư mục đệm mang tên động, nếu chu kỳ CL trước bị lỗi khóa file, chu kỳ CL tiếp theo sẽ tạo một thư mục đệm động mới với timestamp khác để chạy bình thường mà không bị chặn bởi PermissionError.
2. **Kiểm thử nạp (Load Test Integrity)**: Chạy test nạp thử mô hình mới vào RAM, thực hiện suy luận nháp trên 1 câu ví dụ mẫu. Nếu có lỗi, chặn đứng tiến trình.
3. **Thao tác Nguyên tử Windows bằng `os.replace`**: Sử dụng `os.replace()` để ép buộc ghi đè mạnh mẽ và bọc trong vòng lặp thử lại tối đa 3 lần, giãn cách 1 giây để xử lý các khóa file tạm thời của Windows.
4. **Hot-Reloading chống sụt RAM & Chống nghẽn luồng predict**:
   * **Chống nghẽn luồng predict**: Khi bắt đầu chạy lệnh nạp mô hình mới `reload_model()`, cờ `self.is_reloading = True` phải được thiết lập **phía trước** khối khóa luồng `with self.lock:`. Điều này giúp các request suy luận song song gọi vào `predict()` phát hiện ngay trạng thái reload và lập tức rẽ nhánh sang kết quả fallback an toàn (`"CLEAN"`, confidence `1.0`, note reload) mà không bị Block đứng chờ luồng nạp giải phóng khóa (mất 1-2 giây).
   * **RAM Cleanup**: Trong khối `with self.lock:`, hệ thống ngắt liên kết mô hình cũ (`self.model = None`), gọi trình dọn rác Python (`import gc; gc.collect()`) để thu hồi toàn bộ RAM cũ trước khi tiến hành nạp trọng số mới.

---

## 2. Thu thập dữ liệu thực tế & Vòng phản hồi (Feedback Loop)

Dữ liệu mới phục vụ cho các chu kỳ học liên tục tiếp theo được tích lũy liên tục qua:
1. **Borderline Audit Logs**: Các câu thuộc vùng xám được đưa sang LLM Agent sẽ được lưu lại trong `scratch/system_audit_log.jsonl` làm dữ liệu mẫu tự động.
2. **Cơ chế HITL (Human-in-the-Loop)**: Giao diện Streamlit cho phép quản trị viên duyệt và hiệu chỉnh nhãn lỗi trước khi xuất tập rehearsal.
3. **Active Learning**: Ưu tiên chọn các câu có độ bất định cao để gán nhãn thủ công bổ sung.

---

## 3. Chỉ số Giám sát & Phát hiện Độ lệch (Drift Detection)

| Chỉ số Giám sát | Công thức / Định nghĩa | Ngưỡng Cảnh báo & Hành vi |
| :--- | :--- | :--- |
| **Routing Ratio (RR)** | Tỷ lệ số câu chuyển tiếp sang LLM Agent. | Nếu RR vượt quá **30%** liên tục trong 7 ngày $\to$ Cảnh báo phân phối đầu vào bị lệch (Covariate Shift). |
| **Prediction Label Drift** | Lệch phân phối nhãn đầu ra hàng ngày. | Nếu tỷ lệ một nhãn (ví dụ HATE) biến động quá **15%** $\to$ Cảnh báo có sự thay đổi hành vi người dùng hoặc trôi nhãn. |
| **Mean Confidence Score** | Giá trị trung bình của độ tin cậy dự đoán. | Nếu giảm liên tục dưới **0.75** $\to$ Cảnh báo mô hình mất năng lực phân loại trên miền dữ liệu thực tế mới. |

---

## 4. Rủi ro về Drift & Chiến lược Giảm thiểu

* **Concept Drift**: Ngữ nghĩa từ lóng thay đổi theo thời gian.
  - *Giảm thiểu*: Cập nhật định kỳ từ điển chuẩn hóa teencode trong tiền xử lý và thu thập các câu phân tích từ LLM Agent.
* **Covariate Shift**: Sự xuất hiện của các chủ đề mới đột ngột.
  - *Giảm thiểu*: Huấn luyện gia tăng với tập Rehearsal Buffer trộn dữ liệu gốc để giữ vững nền tảng kiến thức cũ.
* **Label Noise**: Bất đồng bộ tiêu chuẩn gán nhãn giữa các dự án.
  - *Giảm thiểu*: Áp dụng **Label Smoothing ($\alpha=0.15$)** và đánh giá nghiêm ngặt qua Gatekeeper.
