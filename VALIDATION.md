# Kết quả kiểm tra ngày 23/10/2026

## Artifact đã tích hợp

Nguồn: `../run_20261023_065529_443169.zip`, do người dùng xác nhận là model chính thức.
21 file giải nén khớp từng byte với ZIP. SHA-256 của model:

`c4a9b1240c17c329269ec1a79ae84a53e8ac422b6e411240053b77f86d8e171a`

Toàn bộ checksum ở `artifact_manifest.json`. Không tạo model mới, sửa trọng số,
adapt vectorizer hoặc huấn luyện lại. Metric và hai biểu đồ lấy từ chính ZIP này.

## Môi trường chạy thành công

Windows, Python 3.13.5, TensorFlow 2.21.0, Keras 3.13.2, NumPy 2.1.3,
Streamlit 1.66.1, Altair 5.5.0. Chi tiết ở `environment-local.json` và
`requirements-lock.txt`. `pip check`: không có dependency lỗi.
Môi trường thử Python 3.10 `.venv-ui` còn lại do công cụ chặn lệnh dọn thư mục;
không được ứng dụng sử dụng. Môi trường đang chạy là `.venv` Python 3.13.

**Lưu ý Windows:** bật `-X utf8` hoặc `PYTHONUTF8=1` khi khởi chạy Python.
Keras đọc từ vựng tiếng Việt bằng bảng mã mặc định; thiếu tùy chọn này gây lỗi
`charmap` trên máy hiện tại. `run_demo.cmd` đã bật UTF-8 sẵn.

## Suy luận

Lệnh đã chạy thành công:

```powershell
.\.venv\Scripts\python.exe -X utf8 check_demo.py --require-model
```

- Đối chiếu 8 câu với `artifacts/predict.py` nguyên bản: bốn loại bình luận,
  dấu tiếng Việt tổ hợp, emoji, từ phủ định, nhiều khoảng trắng và mốc 64/65 token.
- Sai lệch lớn nhất của điểm so với predict.py: **0.0** trên máy này.
  Ngưỡng kiểm tra `rtol=1e-5, atol=1e-6`; nhãn và cờ cắt token cũng khớp.
- Đầu ra đúng bốn lớp, hữu hạn, trong [0,1], tổng gần 1.
- Câu rỗng/toàn khoảng trắng bị từ chối trước khi gọi model.
- Nạp model khoảng 2,39 giây; dự đoán đầu khoảng 0,18 giây;
  bảy câu sau khoảng 0,10–0,12 giây/câu. Đây là phép đo local, không phải benchmark.

## Tích hợp giao diện

Streamlit AppTest đã chạy model thật: nhập câu, bấm phân tích, đối chiếu điểm
trong session với hàm suy luận, đổi câu xóa kết quả cũ, chọn mẫu điền văn bản,
câu dài báo giới hạn token, thiếu artifact không làm crash.
Kiểm tra lần bấm tiếp theo không gọi lại load_artifacts, xác nhận cache hoạt động.

Trình duyệt tại `http://127.0.0.1:8501`: đã bấm dự đoán và nhìn thấy nhãn Tích cực,
đúng bốn thanh điểm; mở phần đánh giá thấy metric, baseline, confusion matrix và
training curves. Đã xem bố cục desktop 1280 px và màn hình nhỏ 390 px.
Ảnh kết quả thật: `screenshots/demo-real.jpg`.

## Số liệu được hiển thị

Đánh giá trên subset test NEU-ESC gồm 1.984 câu:

| Chỉ số | SimpleRNN | Majority baseline |
|---|---:|---:|
| Accuracy | 0.637601 | 0.689012 |
| Macro-F1 | 0.546573 | 0.203969 |
| F1 Toxic | 0.485294 | 0.000000 |

Accuracy của model thấp hơn baseline lớp đa số, trong khi Macro-F1 cao hơn.
Không diễn giải đây là kết quả trên toàn bộ test gốc. Lớp Toxic trong subset test
chỉ có 53 mẫu. Không chạy lại đánh giá toàn bộ dataset; các chỉ số trên lấy nguyên
từ artifact, đúng phạm vi demo được yêu cầu.

## Sửa lỗi biểu đồ giữ điểm cũ khi phân tích câu tiếp theo

Đã tái hiện trên trình duyệt: đổi từ câu khen sang câu phàn nàn làm nhãn đổi sang
Tiêu cực, nhưng biểu đồ vẫn giữ điểm Tích cực 94,3% của câu trước. Dữ liệu inline
trong biểu đồ Altair không được cập nhật đúng trong phiên Streamlit đang mở.

Chuyển nguồn dữ liệu biểu đồ sang DataFrame để Streamlit tạo dataset có tên theo
nội dung. Khi điểm đổi, cả thanh và nhãn phần trăm nhận dataset mới.

Đã kiểm tra trong cùng một phiên trình duyệt, chỉ sửa câu và bấm Phân tích:
Tiêu cực 74,8% -> Tích cực 94,3% -> Độc hại 49,6% -> Tiêu cực 74,8%.
Không tải lại trang giữa các câu. Ảnh: `screenshots/repeated-analysis.jpg`.

`check_demo.py --require-model` đã chạy thành công sau sửa. Kiểm tra bổ sung gửi
ba câu liên tiếp trong cùng session, đối chiếu điểm trong session và dataset của
biểu đồ với suy luận thật, kiểm tra thứ tự nhãn, chữ phần trăm và model cache.
