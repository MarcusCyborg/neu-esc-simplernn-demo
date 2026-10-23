# Demo phân tích bình luận tiếng Việt

Ứng dụng Streamlit chạy local bằng CPU, sử dụng model SimpleRNN được xuất từ
notebook `NEU_ESC_SimpleRNN_A100.ipynb` của assignment. Không có bước huấn luyện
trong ứng dụng. Repo chứa mã demo và artifact suy luận đã train sẵn.

## Cài đặt và chạy trên Windows

Yêu cầu Python 3.13 và Git. Mở PowerShell:

```powershell
git clone https://github.com/MarcusCyborg/neu-esc-simplernn-demo.git
cd neu-esc-simplernn-demo
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -X utf8 -m streamlit run app.py
```

Các file cần cho suy luận từ bộ artifact chính thức
`run_20261023_065529_443169.zip` đã có trong `artifacts/` của repo.
Môi trường `.venv` phải được tạo trên từng máy bằng lệnh ở trên.
Có thể nhấp đúp `run_demo.cmd` để chạy. Không cần kích hoạt môi trường bằng Activate.ps1.
Bắt buộc bật `-X utf8` trên Windows để Keras đọc từ vựng tiếng Việt trong model.
Nếu dùng lệnh `streamlit run app.py` sau khi kích hoạt venv, đặt `$env:PYTHONUTF8 = "1"` trước.
Sau khi cài đủ dependency, chạy demo không cần Colab hoặc Internet.

## Đặt artifact chính thức

Nếu muốn thay bộ model, giải nén ZIP từ lần train chính thức. Chép **nội dung thư mục lần chạy** vào
`artifacts/`, không lồng thêm một cấp thư mục và không trộn nhiều lần train.

```text
streamlit_demo/
  app.py
  inference.py
  preprocessing.py          # Bản tham chiếu nguyên từ notebook
  artifacts/
    model.keras             # Bắt buộc: đã chứa TextVectorization
    preprocessing.py        # Bắt buộc: hàm chuẩn hóa của chính lần train
    label_map.json          # Bắt buộc
    config.json             # Bắt buộc, có MAX_LENGTH
    environment.json
    requirements-inference.txt
    predict.py              # Dùng đối chiếu pipeline gốc
    metrics.json
    confusion_matrix.png
    training_curves.png
```

Suy luận nạp `normalize_text()` từ `artifacts/preprocessing.py`. Bản ở gốc để đọc,
không dùng thay thế khi thiếu file trong artifact. Chỉ dùng mã Python xuất từ
notebook của nhóm vì việc import sẽ chạy mã trong file đó.

Repo giữ model, preprocessing, nhãn, cấu hình, môi trường, requirements, `predict.py`,
metric và hai biểu đồ. Các file chứa bình luận test, manifest lấy mẫu và checkpoint
trọng số riêng vẫn nằm trong ZIP gốc, được bỏ qua khi commit. `artifact_manifest.json`
lưu checksum của toàn bộ 21 file trong ZIP; một số file trong danh sách này không
được phân phối trong repo. Các file artifact có trong repo giữ nguyên từng byte.

Xem `environment.json` để chọn phiên bản Python được TensorFlow của lần train hỗ
trợ. Môi trường local dùng Python 3.13.5; notebook dùng Python 3.13.15.
TensorFlow 2.21.0, Keras 3.13.2 và NumPy 2.1.3 giữ đúng phiên bản trong ZIP.
Cài các phiên bản xuất từ notebook vào cùng môi trường:

```powershell
.\.venv\Scripts\python.exe -m pip install -r artifacts\requirements-inference.txt -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
.\.venv\Scripts\python.exe -X utf8 check_demo.py --require-model
.\.venv\Scripts\python.exe -X utf8 -m streamlit run app.py
```

Nếu bản TensorFlow không hỗ trợ Python hiện tại, tạo môi trường khác với phiên bản
Python phù hợp. Không tự đổi phiên bản model hay train lại để sửa lỗi cài đặt.
Nếu ứng dụng đang chạy lúc thay artifact, dừng bằng Ctrl+C rồi khởi động lại để
xóa model trong cache. Không cần GPU, CUDA hay A100 cho demo.

## Kiểm tra

```powershell
.\.venv\Scripts\python.exe -X utf8 check_demo.py
.\.venv\Scripts\python.exe -X utf8 check_demo.py --require-model
```

Lệnh đầu kiểm tra chuẩn hóa, lỗi đầu vào và trạng thái ứng dụng bằng Streamlit
AppTest; nếu thiếu model sẽ báo SKIP rõ ràng. Lệnh thứ hai yêu cầu đủ model và
`predict.py`, kiểm tra điểm của bốn lớp rồi đối chiếu pipeline gốc với
`rtol=1e-5, atol=1e-6`. Kiểm tra UI cũng gọi model thật và đối chiếu điểm đã hiển thị.

Xem `VALIDATION.md` để biết phần nào đã kiểm chứng thực tế và bộ phiên bản đã dùng.

## Cách hiểu kết quả

- Thứ tự cố định: 0 Neutral / Trung tính, 1 Positive / Tích cực,
  2 Negative / Tiêu cực, 3 Toxic / Độc hại.
- Điểm softmax không phải độ chính xác được bảo đảm cho từng câu.
- Giới hạn token lấy từ `config.json`, tách token theo khoảng trắng sau chuẩn hóa.
  Vectorizer giữ các token đầu; ứng dụng không tự tokenize lại, adapt hoặc bỏ dấu.
- Accuracy, Macro-F1 và F1 Toxic lấy từ `metrics.json` mục `test`, trên **subset
  test NEU-ESC**, không đại diện benchmark toàn bộ test gốc.
- Không dùng model/metric smoke test làm kết quả chính thức.

## Lỗi thường gặp

| Hiện tượng | Cách xử lý |
|---|---|
| Thiếu artifact | Đặt các file ứng dụng liệt kê trực tiếp trong `artifacts/`. |
| Không nạp được model | Mở chi tiết kỹ thuật; kiểm tra phiên bản trong `environment.json` và cài requirements của lần train. |
| Không tìm thấy TensorFlow | Cài `artifacts/requirements-inference.txt` bằng Python của chính `.venv`. |
| Chưa có kết quả đánh giá | Bổ sung `metrics.json` cùng lần train; dự đoán vẫn dùng được khi đủ file bắt buộc. |
| Nhãn/config không hợp lệ | Lấy lại bản xuất đúng từ notebook, giữ thứ tự 0..3 và MAX_LENGTH nguyên dương. |
| Điểm hoặc model không thay đổi sau thay ZIP | Khởi động lại ứng dụng để xóa cache. |

## Kịch bản trình bày hai phút

1. 20 giây: giới thiệu SimpleRNN, NEU-ESC và bốn nhãn.
2. 50 giây: phân tích hai câu đã kiểm tra trước, đọc nhãn và phân bố điểm.
3. 30 giây: sửa câu, ví dụ thêm từ “không”, bấm lại và quan sát phản ứng thực tế.
4. 20 giây: mở đánh giá, nêu giới hạn mỉa mai, tiếng lóng hoặc lớp Toxic ít mẫu.

Không gán sẵn nhãn đúng cho các câu mẫu. Model chính thức đã được kiểm tra trên CPU;
xem `VALIDATION.md` và ảnh thật tại `screenshots/demo-real.jpg`. Nên đo lại tốc độ
trên laptop dùng thuyết trình và chuẩn bị ảnh/video dự phòng.
