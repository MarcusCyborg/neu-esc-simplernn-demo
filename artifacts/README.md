
# Kết quả NEU-ESC SimpleRNN

- model.keras: vectorizer và SimpleRNN đã chọn bằng validation loss.
- preprocessing.py + predict.py: suy luận từ văn bản gốc, từ chối câu rỗng.
- metrics.json và các CSV/PNG: số liệu, biểu đồ và thời gian chạy thực tế.
- sample_manifest.csv: chỉ số dòng (bắt đầu từ 0, không tính header) trong từng CSV gốc.
- config.json: cấu hình, SHA-256 ZIP nguồn, tỷ lệ lấy mẫu và quy tắc tiền xử lý.
- requirements-inference.txt: phiên bản thư viện cho suy luận ở môi trường hiện tại.

Chạy lại suy luận trong môi trường Python tương thích:
    pip install -r requirements-inference.txt
    python predict.py

Đầu vào gọi trực tiếp model.keras phải được normalize_text() trước và không được rỗng.
Model giữ các token đầu câu theo MAX_LENGTH; các giá trị Softmax không phải độ chính xác
đã hiệu chuẩn của từng câu. Không dùng model để tự động quyết định xử lý một người.

Điểm đánh giá là trên subset của test NEU-ESC. Lớp Toxic có ít mẫu; xem support trong report.
Class weight chỉ dùng khi train. Không diễn giải clipping là cách chữa vanishing gradient.
Không so sánh điểm trên subset này như benchmark toàn bộ test gốc.

Nguồn: https://huggingface.co/datasets/hung20gg/NEU-ESC
File test_predictions.csv chứa văn bản nguồn; kiểm tra quyền chia sẻ trước khi công khai.
