# Giao diện tích hợp đã chốt

Ứng dụng dùng artifact nguyên bản từ một lần chạy notebook chính thức. Không huấn luyện,
không tạo model giả làm kết quả, không lấy artifact smoke test.

## Cấu trúc

`app.py`, `inference.py`, `preprocessing.py`, `requirements.txt`, `README.md`,
`check_demo.py`, `.streamlit/config.toml`, `artifacts/`.

Trong `artifacts/`: bắt buộc có `model.keras`, `preprocessing.py`, `label_map.json`,
`config.json`; tùy chọn `environment.json`, `requirements-inference.txt`, `metrics.json`,
`confusion_matrix.png`, `training_curves.png`, `predict.py` và các file xuất khác.
Đặt toàn bộ nội dung ZIP của cùng một lần train vào đây. Bản preprocessing ở gốc chỉ
là bản tham chiếu trích nguyên từ notebook; suy luận luôn dùng bản trong artifact.

## API inference.py

- `DEFAULT_ARTIFACT_DIR: Path`: thư mục artifacts cạnh inference.py.
- `LABEL_NAMES = ("Neutral", "Positive", "Negative", "Toxic")`.
- `LABEL_VI = ("Trung tính", "Tích cực", "Tiêu cực", "Độc hại")`.
- `ArtifactError(RuntimeError)`: lỗi có thông báo tiếng Việt, giữ nguyên cause kỹ thuật.
- `missing_artifacts(artifact_dir=DEFAULT_ARTIFACT_DIR) -> list[str]`.
- `load_artifacts(artifact_dir=DEFAULT_ARTIFACT_DIR) -> dict`: các khóa `model`,
  `normalize_text` (callable nạp từ artifact), `config`, `label_map`, `max_length` (int),
  `artifact_dir` (Path), `environment` (dict, {} nếu thiếu/hỏng).
  Nạp TensorFlow trễ, chạy CPU, load model compile=False. Xác minh bốn ID và tên gốc,
  MAX_LENGTH nguyên dương; nếu kiểm tra vectorizer được thì đối chiếu output_sequence_length.
- `predict_text(text: str, artifacts: dict) -> dict`: khóa `text`, `normalized_text`,
  `label_id` (int), `label` (tiếng Việt), `scores` (list[float] đúng thứ tự 0..3),
  `token_count`, `max_length`, `truncated` (bool). Từ chối chuỗi rỗng bằng ValueError
  trước khi gọi model. Gọi tf.constant([normalized_text]), training=False.
  Xác minh đầu ra shape (1, 4), hữu hạn, trong [0,1], tổng gần 1.
- `load_metrics(artifact_dir=DEFAULT_ARTIFACT_DIR) -> dict | None`: trả JSON gốc đã
  kiểm tra `test.accuracy`, `test.macro_f1`, `test.f1_toxic` hữu hạn thuộc [0,1];
  thiếu/hỏng trả None, không cản dự đoán.

UI tự bọc load_artifacts bằng st.cache_resource. Chỉ suy luận khi bấm nút.
Xóa kết quả khi sửa văn bản/chọn mẫu. Không tự điền điểm hay metric.

## Phân quyền file

- Agent inference: chỉ `inference.py`, `preprocessing.py`.
- Agent UI: chỉ `app.py`, `.streamlit/config.toml`.
- Agent chính: tài liệu, requirements, artifacts/.gitkeep, check_demo.py,
  môi trường, kiểm tra và sửa tích hợp sau khi hai agent bàn giao.
