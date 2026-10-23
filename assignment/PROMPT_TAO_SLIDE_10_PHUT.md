# Prompt tạo slide thuyết trình NEU-ESC SimpleRNN

Hãy thực thi tạo bộ slide PowerPoint hoàn chỉnh cho assignment Applied Deep Learning dưới đây. Không chỉ trả về outline. Nội dung và lời thuyết trình bằng tiếng Việt, thuật ngữ kỹ thuật có thể giữ tiếng Anh. Đọc nguồn có sẵn, sử dụng skill tạo presentation phù hợp, thiết kế slide, kiểm tra bản render và bàn giao file chỉnh sửa được.

## 1. Mục tiêu, thời lượng và nhóm

- Chủ đề: Phân loại bình luận tiếng Việt bằng RNN truyền thống trên NEU-ESC, có demo Streamlit dùng model đã train và lưu từ notebook.
- Người nghe: giảng viên và sinh viên môn Applied Deep Learning; cần thấy nhóm hiểu RNN, quy trình thực nghiệm và sản phẩm hoạt động.
- Tổng phần trình bày: đúng khoảng 10 phút, BAO GỒM chuyển người nói và live demo. Đề bài cho tối đa 12 phút kể cả Q&A, nên chừa khoảng 2 phút hỏi đáp.
- Nhóm gồm: Nhân, Huy, Đạt, Khang. Không tự thêm họ tên, MSSV, lớp hoặc tên giảng viên.
- Phương án mặc định: Huy, Đạt, Khang thuyết trình; Nhân vận hành demo và chuyển slide, không bắt buộc nói.
- Phương án thay thế: Nhân chỉ nói mở đầu slide 1 trong 30 giây; Huy tiếp tục từ slide 2. Không thay đổi tổng thời lượng hoặc nội dung bộ slide.
- Chỉ cần MỘT bộ slide dùng được cho cả hai phương án. Có lời mở đầu ngắn, dễ học cho Nhân và phương án để Huy đọc thay.
- Giữ nội dung vừa sức assignment, giải thích dễ hiểu và có căn cứ; không biến thành bài tổng quan quá rộng về deep learning.

## 2. Nguồn cần đọc

Thư mục làm việc: `F:\DPL project\Assignemtn 23.10`.

Các nguồn chính:

1. `F:\DPL project\Assignemtn 23.10\run_20261023_065529_443169.zip`: artifact thật từ lần train trên Colab; đây là nguồn chính cho số liệu.
2. `F:\DPL project\Assignemtn 23.10\NEU_ESC_SimpleRNN_A100.ipynb`: kiến trúc và pipeline thực tế.
3. `F:\DPL project\Assignemtn 23.10\HUONG_DAN_NOTEBOOK.md`: hướng dẫn, cách dùng artifact và giới hạn.
4. `F:\DPL project\Assignemtn 23.10\PROPOSAL_RNN_TRUYEN_THONG.md`: bối cảnh lựa chọn phương án. Nếu proposal khác kết quả thực tế, ưu tiên notebook và artifact.
5. `F:\1. FPT Material\Sem5\DPL302m\5.1 Recurrent Neural Networks.pptx`: tham khảo phần kiến thức RNN nếu truy cập được; không cần sao chép thiết kế giáo trình.
6. Tìm source/demo Streamlit hiện có trong thư mục dự án nếu đã triển khai. Kiểm tra trước khi mô tả trạng thái sản phẩm hoặc chụp ảnh demo.

ZIP có `model.keras`, `preprocessing.py`, `predict.py`, `label_map.json`, `config.json`, `environment.json`, `metrics.json`, `classification_report.csv`, `model_comparison.csv`, `training_history.csv`, `subset_counts.csv`, `confusion_matrix.png`, `training_curves.png` và các file bổ trợ.

Không train lại model để tạo slide. Không sửa notebook hay ứng dụng. Không ghi đè file có sẵn. Chỉ đọc dữ liệu cần thiết và tạo tài liệu trình bày trong thư mục đầu ra riêng.

## 3. Dữ kiện kỹ thuật đã xác minh

### Dataset và tiền xử lý

- Dataset: NEU-ESC, nguồn https://huggingface.co/datasets/hung20gg/NEU-ESC.
- Dùng trường `text` làm đầu vào, `sentiment` làm nhãn; không dùng `classification` làm feature.
- Giữ nguyên bốn nhãn: 0 Neutral/Trung tính, 1 Positive/Tích cực, 2 Negative/Tiêu cực, 3 Toxic/Độc hại.
- Lấy 30% riêng trong từng split gốc bằng lấy mẫu phân tầng; giữ gần nguyên tỷ lệ nhãn và tỷ lệ các split, sai lệch nhỏ do làm tròn. Seed 42, không lấy mẫu lặp.
- Không cân bằng số lượng mẫu thành bốn lớp bằng nhau và không trộn lại train/validation/test.

| Split | Neutral | Positive | Negative | Toxic | Tổng |
|---|---:|---:|---:|---:|---:|
| Train | 4781 | 869 | 1089 | 175 | 6914 |
| Validation | 684 | 125 | 157 | 26 | 992 |
| Test | 1367 | 250 | 314 | 53 | 1984 |

- Chuẩn hóa Unicode NFC, lowercase, gộp khoảng trắng. Giữ dấu tiếng Việt, emoji và từ phủ định.
- Token hóa theo khoảng trắng; giới hạn 64 token, giữ phần đầu câu, padding bên phải.
- Vocabulary chỉ học từ subset train; giới hạn 10.000, kích thước thực tế ghi trong config là 6.630.
- Dùng class weight tính từ train để xử lý mất cân bằng; phân phối validation và test vẫn giữ như lấy mẫu từ dữ liệu gốc.

### Mô hình và huấn luyện

Kiến trúc:

`Embedding(64, mask_zero=True) → SimpleRNN(64, tanh, return_sequences=True, recurrent_initializer="orthogonal") → GlobalAveragePooling1D có mask → Dropout(0.2) → Dense(4, softmax)`.

- Đây là RNN truyền thống một chiều, không phải LSTM/GRU/BiLSTM/Transformer.
- Dùng công thức hidden state đơn giản khi giải thích: `h_t = tanh(W_x x_t + W_h h_(t-1) + b)`.
- Các vị trí chia sẻ cùng trọng số RNN. Pooling chỉ tổng hợp hidden state ở các vị trí có nội dung, bỏ qua padding.
- Giảm ảnh hưởng vanishing gradient bằng chuỗi ngắn, khởi tạo ma trận recurrent trực giao và tổng hợp hidden state ở nhiều vị trí. Đây là lựa chọn thiết kế, chưa có ablation hoặc đo gradient để chứng minh mức cải thiện riêng từng biện pháp; không khẳng định loại bỏ hoàn toàn vanishing gradient.
- Adam, learning rate 0,001, clipnorm 1, batch size 64, tối đa 6 epoch, sparse categorical cross-entropy.
- Gradient clipping hạn chế EXPLODING gradient, không phải biện pháp chữa vanishing gradient.
- Có EarlyStopping theo validation loss, patience 2; lần chạy thực tế hoàn thành đủ 6 epoch, checkpoint tốt nhất là epoch 5. Không nói rằng EarlyStopping đã dừng sớm lần chạy này.
- GPU thực tế: NVIDIA A100-SXM4-40GB.
- Thời gian train ghi nhận: khoảng 47,32 giây cho lần chạy này. Đây là thời gian huấn luyện được log, không phải toàn bộ thời gian upload, xử lý dữ liệu, export và chạy notebook; không cam kết mọi máy/lần chạy đều nhanh như vậy.
- Train loss có class weight, validation loss không có class weight; không diễn giải chênh lệch hai loss như thể chúng được tính giống nhau.

### Kết quả thật

| Mô hình | Test Accuracy | Test Macro-F1 | Test F1 Toxic |
|---|---:|---:|---:|
| Majority baseline: luôn dự đoán Neutral | 68,90% | 20,40% | 0,00% |
| SimpleRNN + masked average pooling | 63,76% | 54,66% | 48,53% |

- Validation Accuracy: 63,21%; Validation Macro-F1: 53,31%.
- F1 từng lớp trên test: Neutral 73,80%; Positive 47,66%; Negative 48,64%; Toxic 48,53%.
- Toxic trên test chỉ có 53 mẫu; precision khoảng 39,76%, recall khoảng 62,26%. Chỉ đưa chi tiết precision/recall vào phần dự phòng nếu slide chính bị dày.
- Baseline có Accuracy cao vì Neutral chiếm khoảng 68,90% subset test. SimpleRNN có Accuracy thấp hơn nhưng Macro-F1 cao hơn: khả năng phân loại bốn lớp tốt hơn theo chỉ số này. Không nói mô hình vượt baseline trên mọi chỉ số.
- Macro-F1 là trung bình F1 của bốn lớp, mỗi lớp được tính trọng số như nhau.
- Mọi số liệu là trên subset phân tầng của test NEU-ESC, không phải benchmark toàn bộ test gốc.
- Không đưa ra quan hệ nhân quả chưa kiểm chứng như “class weight làm Macro-F1 tăng X điểm” khi không có thực nghiệm đối chứng.

### Demo Streamlit

- Dùng trực tiếp `model.keras` từ ZIP trên, gồm vectorizer và trọng số mạng đã huấn luyện. Chuẩn hóa bằng cùng `preprocessing.py`.
- Luồng: nhập bình luận → chuẩn hóa → model đã lưu → điểm bốn lớp → nhãn có điểm cao nhất.
- Không train trong ứng dụng, không cần kết nối Colab để suy luận. Có thể chạy bằng CPU trên laptop sau khi kiểm chứng môi trường.
- Điểm softmax không phải “độ chính xác của từng câu”.
- Nếu app chưa có, không bịa screenshot hoặc tuyên bố demo đã chạy thành công. Dùng sơ đồ luồng và ghi việc cần bổ sung ở ghi chú bàn giao. Nếu có app, lấy ảnh từ ứng dụng thật.

## 4. Cấu trúc 10 slide chính và phân vai

Tổng thời lượng bảng dưới là 600 giây, đã gồm demo. Viết lời dẫn và chuyển người nói nằm trong ngân sách từng slide.

| Slide | Nội dung chính | Người nói mặc định | Thời lượng |
|---|---|---|---:|
| 1 | Tên đề tài, tên 4 thành viên, mục tiêu sản phẩm | Huy; Nhân có thể thay phần này | 30 giây |
| 2 | Bài toán: một bình luận đầu vào, bốn nhãn đầu ra; vì sao hữu ích | Huy | 45 giây |
| 3 | Dataset, subset 30%, mất cân bằng nhãn | Huy | 65 giây |
| 4 | Pipeline tiền xử lý và tránh leakage | Huy | 60 giây |
| 5 | RNN truyền thống hoạt động ra sao và kiến trúc thực tế | Đạt | 85 giây |
| 6 | Vanishing gradient, các biện pháp đã dùng và giới hạn | Đạt | 65 giây |
| 7 | Cấu hình train, thời gian thực tế và chọn checkpoint epoch 5 | Đạt | 60 giây |
| 8 | Kết quả thật so với baseline; giải thích Accuracy và Macro-F1 | Khang | 80 giây |
| 9 | Live demo Streamlit dùng model đã lưu | Khang nói, Nhân thao tác | 90 giây |
| 10 | Kết luận ngắn, giới hạn và hướng cải thiện | Khang | 20 giây |

Phân bổ mặc định: Huy 3 phút 20 giây, Đạt 3 phút 30 giây, Khang 3 phút 10 giây. Nếu Nhân nói slide 1: Nhân 30 giây, Huy còn 2 phút 50 giây; các phần còn lại giữ nguyên.

Slide 8 là trọng tâm phân tích kết quả. Dùng biểu đồ so sánh có cùng thang đo và số liệu rõ ràng, không che việc Accuracy của SimpleRNN thấp hơn baseline. Confusion matrix chi tiết có thể đặt vào phụ lục để slide chính không quá dày.

Slide 9: dành khoảng 15 giây giải thích model đã được nạp, 55 giây cho hai câu và một lần sửa câu nếu kịp, 20 giây giải thích kết quả và chuyển về slide. Không hứa câu có từ phủ định chắc chắn sẽ đổi nhãn. Câu minh họa phải được thử bằng model thật nếu môi trường cho phép; ghi rõ câu tự tạo để demo, không dùng để tuyên bố điểm đánh giá.

Slide 10 chỉ chốt hai ý đủ nói trong 20 giây: đã xây pipeline RNN và demo dùng model đã lưu; mô hình còn hạn chế với mỉa mai/câu dài/lớp ít mẫu, có thể nghiên cứu LSTM/GRU về sau. Hướng cải thiện chưa triển khai phải được ghi đúng trạng thái.

## 5. Yêu cầu thiết kế

- Khổ widescreen 16:9, ưu tiên dễ đọc khi trình chiếu trong lớp.
- Một thông điệp chính mỗi slide; tiêu đề nên diễn đạt kết luận hoặc câu hỏi cụ thể, không chỉ “Dataset”, “Model”, “Results”.
- Ít chữ trên slide, diễn giải đầy đủ trong speaker notes. Không dán nguyên đoạn report hoặc code dài.
- Dùng sơ đồ vector cho pipeline và kiến trúc; biểu đồ từ số liệu thật cho phân phối nhãn và so sánh metric.
- Chọn một hệ màu nhất quán, nền sáng, tương phản tốt. Màu của bốn nhãn giữ nhất quán trên các slide. Không dùng màu làm tín hiệu duy nhất.
- Font hỗ trợ tiếng Việt, cỡ chữ thân bài khoảng 24 pt trở lên; chú thích nguồn đủ đọc. Kiểm tra lỗi dấu, xuống dòng, tràn chữ và cắt biểu đồ.
- Không dùng ảnh stock robot, ảnh AI chung chung hoặc trang trí chiếm diện tích của dữ liệu.
- Giữ bảng nhỏ; bảng đầy đủ dữ liệu và thông số có thể chuyển sang phụ lục.
- Footer gọn: số slide và nguồn rút gọn khi cần. Có nguồn đầy đủ trong ghi chú hoặc phụ lục.

## 6. Speaker notes và chuẩn bị hỏi đáp

Mỗi slide chính cần có speaker notes gồm: người nói, thời lượng, lời thoại tiếng Việt tự nhiên, thao tác/chỉ điểm trên slide và câu chuyển ý. Lời thoại phải đủ ngắn để nói theo lịch trên, có khoảng dừng; kiểm tra tổng thời lượng bằng đọc thử hoặc ước lượng có nêu rõ phương pháp. Không khẳng định đã rehearsal nếu chưa thực hiện.

Viết riêng lời mở đầu 30 giây cho Nhân, dùng ngôn ngữ đơn giản, không cần giải thích công thức. Ghi Huy có thể đọc thay nguyên đoạn này. Nếu Nhân không nói, vẫn ghi tên Nhân trong nhóm và hướng dẫn thao tác demo cho Nhân; không bịa phân công đóng góp lịch sử của các thành viên.

Thêm khoảng 3 slide phụ lục, không tính trong 10 phút:

1. Confusion matrix và precision/recall/F1/support từng lớp.
2. Learning curves và thông số train bổ sung; nêu checkpoint tốt nhất epoch 5 và val loss tăng ở epoch 6.
3. Pipeline export/load model và nguồn tham khảo.

Trong tài liệu lời thoại, thêm câu trả lời ngắn, dễ hiểu cho các câu hỏi:

- Vì sao dùng SimpleRNN thay vì LSTM/GRU?
- RNN ghi nhớ thông tin trước đó như thế nào? BPTT là gì?
- Vanishing gradient là gì? Các lựa chọn hiện tại giúp gì và có giới hạn gì?
- Vì sao gradient clipping không chữa vanishing gradient?
- Vì sao Accuracy thấp hơn baseline nhưng vẫn có kết quả đáng phân tích?
- Class weight khác lấy mẫu cân bằng như thế nào?
- Vì sao lấy 30% và làm sao tránh leakage?
- Vì sao chọn epoch 5 dù train đủ 6 epoch?
- Streamlit có dùng model đã train không, có cần GPU không?
- Có thể tin chắc điểm softmax hoặc kết luận về lớp Toxic không?

Chỉ trả lời bằng điều đã làm hoặc kiến thức đúng; thừa nhận chưa có ablation, chưa đánh giá full test hoặc chưa hiệu chuẩn xác suất khi được hỏi.

## 7. Deliverable và kiểm tra cuối

Tạo thư mục đầu ra riêng, ví dụ `F:\DPL project\Assignemtn 23.10\presentation`, gồm:

1. `NEU_ESC_SimpleRNN_Presentation.pptx`: 10 slide chính + khoảng 3 phụ lục, chỉnh sửa được, có speaker notes.
2. Bản PDF để xem và dự phòng nếu công cụ xuất PDF khả dụng.
3. `KICH_BAN_THUYET_TRINH.md`: lời thoại theo slide, thời lượng, người nói, hai phương án Nhân nói/không nói, chuyển người và Q&A.
4. `CHECKLIST_DEMO.md`: cách mở app thật, đặt artifact, thử câu mẫu, phân vai Nhân thao tác/Khang nói, cách xử lý nếu app không chạy. Không dùng kết quả dự đoán giả để thay demo.

Trước bàn giao: đối chiếu số liệu với ZIP, kiểm tra thứ tự nhãn, render toàn bộ slide và xem lỗi layout. Nếu có phần chưa làm được, ghi rõ trong bàn giao thay vì nói đã hoàn thành. Nếu demo chưa có, hoàn thiện các slide còn lại và ghi chính xác việc cần bổ sung cho slide demo; không tự mở rộng sang xây ứng dụng trong nhiệm vụ này.

Không upload/publish bộ slide hoặc gửi cho thành viên nhóm. Bàn giao file local cùng đường dẫn mở được.
