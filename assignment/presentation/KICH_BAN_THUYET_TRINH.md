# Kịch bản thuyết trình NEU-ESC SimpleRNN

Bộ slide gồm 10 slide chính và 3 phụ lục. Lịch trình chính là 600 giây, đã gồm chuyển người, chuyển cửa sổ và demo 90 giây. Q&A dùng khoảng 2 phút còn lại, kết thúc trong giới hạn 12 phút.

## Hai phương án dùng cùng một bộ slide

- **Mặc định:** Huy nói slide 1–4 (3 phút 20 giây), Đạt nói slide 5–7 (3 phút 30 giây), Khang nói slide 8–10 (3 phút 10 giây). Nhân chuyển slide và thao tác app.
- **Nhân mở đầu:** Nhân nói slide 1 trong 30 giây, Huy nói slide 2–4 trong 2 phút 50 giây. Đạt và Khang giữ nguyên. Nhân tiếp tục vận hành slide/demo. Không thêm lời chào hoặc slide riêng.

Đây là phân công trình bày, không phải mô tả đóng góp lịch sử của thành viên.

## Lời mở đầu riêng cho Nhân (Huy đọc thay nguyên đoạn nếu cần)

> Em chào thầy cô và các bạn. Nhóm gồm Nhân, Huy, Đạt và Khang. Nhóm xây dựng mô hình RNN truyền thống để phân loại bình luận tiếng Việt thành bốn nhãn trên NEU-ESC. Phần trình bày gồm cách xử lý dữ liệu, cách RNN hoạt động, kết quả thực nghiệm và demo dùng model đã lưu. Sau đây là bài toán cụ thể.

Đọc chậm, có một nhịp dừng sau tên nhóm. Nhân không cần giải thích công thức ở phần này. Huy bắt đầu slide 2 ngay sau câu cuối.

## Đồng hồ cho người chuyển slide

| Slide | Mốc từ đầu buổi | Người nói | Ngân sách |
|---|---|---|---:|
| 1 | 0:00–0:30 | Huy (Nhân có thể đọc thay) | 30 giây |
| 2 | 0:30–1:15 | Huy | 45 giây |
| 3 | 1:15–2:20 | Huy | 65 giây |
| 4 | 2:20–3:20 | Huy | 60 giây |
| 5 | 3:20–4:45 | Đạt | 85 giây |
| 6 | 4:45–5:50 | Đạt | 65 giây |
| 7 | 5:50–6:50 | Đạt | 60 giây |
| 8 | 6:50–8:10 | Khang | 80 giây |
| 9 | 8:10–9:40 | Khang nói, Nhân thao tác | 90 giây |
| 10 | 9:40–10:00 | Khang | 20 giây |
| Q&A | 10:00–12:00 | Cả nhóm, người hiểu phần nào trả lời phần đó | khoảng 120 giây |

Nhân theo dõi đồng hồ độc lập và báo hiệu kín khi còn 10 giây mỗi slide. Mốc cần giữ: 3:20 chuyển Huy sang Đạt, 6:50 chuyển Đạt sang Khang, 8:10 vào demo, 9:40 trở về kết luận, 10:00 kết thúc. Nếu chậm, bỏ lần sửa câu C trong demo và rút phần diễn giải phụ, không tăng tổng 600 giây. Không tự động chuyển slide theo timer vì còn live demo.

## Ước lượng thời lượng (chưa rehearsal trực tiếp)

Đếm các đơn vị tách bằng khoảng trắng trong lời thoại, xấp xỉ âm tiết tiếng Việt, rồi chia cho **3 đơn vị/giây**. Tên tiếng Anh và cách đọc số có thể cần thêm thời gian, nên đây là ước lượng. Cột còn lại dành cho dừng, chỉ sơ đồ, đổi người/cửa sổ và thao tác. Không đọc phần hướng dẫn hay nguồn ra tiếng. Chưa có người thật đọc thử toàn bộ để xác nhận.

| Slide | Đơn vị lời thoại | Ước lượng nói | Thao tác, dừng và chuyển ý còn lại |
|---|---:|---:|---:|
| 1 | 64 | 22 giây | 8 giây |
| 2 | 109 | 37 giây | 8 giây |
| 3 | 137 | 46 giây | 19 giây |
| 4 | 130 | 44 giây | 16 giây |
| 5 | 177 | 59 giây | 26 giây |
| 6 | 150 | 50 giây | 15 giây |
| 7 | 132 | 44 giây | 16 giây |
| 8 | 176 | 59 giây | 21 giây |
| 9 | 127 | 43 giây | 47 giây |
| 10 | 50 | 17 giây | 3 giây |

Tổng ngân sách là **600 giây**. Người nói cần tập theo mốc, nhất là 85 giây sơ đồ RNN và 90 giây demo. Các khoảng dừng không phải lời thoại bổ sung.

## Slide 1: Phân loại bình luận bằng RNN truyền thống

**Người nói:** Huy (Nhân có thể đọc thay). **Thời lượng:** 30 giây.

### Lời thoại

Em chào thầy cô và các bạn. Nhóm gồm Nhân, Huy, Đạt và Khang. Nhóm xây dựng mô hình RNN truyền thống để phân loại bình luận tiếng Việt thành bốn nhãn trên NEU-ESC. Phần trình bày gồm cách xử lý dữ liệu, cách RNN hoạt động, kết quả thực nghiệm và demo dùng model đã lưu. Sau đây là bài toán cụ thể.

### Thao tác và chỉ điểm

Huy hoặc Nhân đọc nguyên đoạn mở đầu này. Nhân mở slide 2 sau 30 giây. Nếu Nhân mở đầu, Huy tiếp tục ngay ở slide 2, không giới thiệu lại.

### Chuyển ý

Huy tiếp tục slide 2. Không có lần chuyển người thứ hai trong slide 1.

### Nguồn

- Yêu cầu assignment do người dùng cung cấp
- Notebook NEU_ESC_SimpleRNN_A100.ipynb và artifact run_20261023_065529_443169.zip

## Slide 2: Một bình luận được gán một trong bốn nhãn

**Người nói:** Huy. **Thời lượng:** 45 giây.

### Lời thoại

Đầu vào là một bình luận tiếng Việt. Mô hình trả bốn điểm và chọn nhãn có điểm cao nhất. Bốn nhãn giữ nguyên theo dữ liệu: mã không là trung tính, một là tích cực, hai là tiêu cực và ba là độc hại. Tiêu cực có thể là một lời phàn nàn, còn độc hại thường mang tính công kích. Công cụ có thể hỗ trợ phân loại phản hồi để người dùng xem xét tiếp. Đây là bài toán gán một nhãn cho cả câu, nên vẫn cần con người kiểm tra các trường hợp khó. Trước hết, nhóm chuẩn bị dữ liệu như thế nào?

### Thao tác và chỉ điểm

Chỉ lần lượt đầu vào, model và các mã 0–3. Không gán nhãn sẵn cho ví dụ như thể đó là dự đoán đã chạy.

### Chuyển ý

Huy chuyển sang quy mô và tỷ lệ nhãn trên slide 3.

### Nguồn

- ZIP: label_map.json
- Notebook: text làm đầu vào, sentiment làm nhãn

## Slide 3: Lấy 30% từng split, vẫn giữ mất cân bằng

**Người nói:** Huy. **Thời lượng:** 65 giây.

### Lời thoại

NEU-ESC có sẵn train, validation và test. Nhóm lấy ba mươi phần trăm riêng trong từng split bằng lấy mẫu phân tầng với seed bốn mươi hai, không lấy mẫu lặp và không gộp rồi chia lại. Kết quả có sáu nghìn chín trăm mười bốn câu train, chín trăm chín mươi hai câu validation và một nghìn chín trăm tám mươi bốn câu test. Biểu đồ cho thấy test vẫn lệch về Neutral, khoảng sáu mươi chín phần trăm, còn Toxic chỉ có năm mươi ba câu. Nhóm giữ tỷ lệ gần dữ liệu gốc, có sai số làm tròn, chứ không chia đều bốn lớp. Class weight chỉ thay đổi trọng số loss khi train. Vì vậy, các kết quả sau đây chỉ đại diện cho subset test này. Tiếp theo là pipeline xử lý.

### Thao tác và chỉ điểm

Chỉ bảng tổng số mẫu, rồi cột Neutral và cột Toxic. Dừng ngắn ở con số 53 để người nghe ghi nhận giới hạn đánh giá.

### Chuyển ý

Huy chuyển slide 4.

### Nguồn

- ZIP: subset_counts.csv, config.json, sample_manifest.csv
- Dataset: https://huggingface.co/datasets/hung20gg/NEU-ESC

## Slide 4: Vocabulary chỉ học từ subset train

**Người nói:** Huy. **Thời lượng:** 60 giây.

### Lời thoại

Nhóm chỉ dùng trường text làm đầu vào và sentiment làm nhãn. Trường classification không đi vào feature. Văn bản được chuẩn hóa Unicode NFC, chuyển chữ thường và gộp khoảng trắng. Nhóm vẫn giữ dấu tiếng Việt, emoji và từ phủ định. Sau đó token hóa theo khoảng trắng, nên với tiếng Việt nhiều token là âm tiết. Vocabulary chỉ học từ subset train, thực tế có sáu nghìn sáu trăm ba mươi mục, trong giới hạn mười nghìn. Validation và test chỉ dùng vocabulary đó để biến đổi. Mỗi câu giữ tối đa sáu mươi tư token đầu và padding bên phải. Model dùng mask để bỏ qua padding. Demo sử dụng lại cùng hàm chuẩn hóa và vectorizer đã lưu. Em mời Đạt giải thích phần RNN.

### Thao tác và chỉ điểm

Đi theo sơ đồ từ trái sang phải. Nhấn ở dòng “chỉ train adapt”. Huy kết thúc và Đạt tiếp lời trong khoảng 3 giây, đã nằm trong 60 giây.

### Chuyển ý

Huy: “Em mời Đạt giải thích phần RNN.” Đạt bắt đầu slide 5.

### Nguồn

- Notebook: cell 5, 10, 14, 25–26 (chỉ số cell từ 0)
- ZIP: preprocessing.py, config.json, model.keras

## Slide 5: RNN cập nhật trạng thái qua từng token

**Người nói:** Đạt. **Thời lượng:** 85 giây.

### Lời thoại

RNN đọc chuỗi từ trái sang phải. Ở mỗi bước, nó nhận vector của token hiện tại và hidden state từ bước trước, rồi tạo hidden state mới bằng hàm tanh. Trong công thức, x là embedding hiện tại, h trước là thông tin đã tích lũy, còn các ma trận W là trọng số học được. Các bước dùng chung trọng số, không tạo một mạng riêng cho mỗi từ. Vì vậy, trạng thái sau có thể mang thông tin từ các token trước đó. Kiến trúc thực tế bắt đầu bằng embedding sáu mươi tư chiều, rồi một tầng SimpleRNN sáu mươi tư units. Đây là RNN truyền thống một chiều. Nhóm bật return sequences để giữ hidden state ở mọi vị trí. Masked average pooling lấy trung bình các trạng thái có nội dung và bỏ padding. Sau đó dropout mức không phẩy hai và Dense softmax tạo bốn điểm. Cả câu vẫn chỉ nhận một nhãn. Cấu trúc này đơn giản, nhưng còn khó khăn khi cần giữ thông tin từ rất xa.

### Thao tác và chỉ điểm

Chỉ một ô h_t, chỉ x_t và h_(t−1), rồi chỉ mũi nối giữa các bước. Khi nói pooling, chỉ sơ đồ kiến trúc phía dưới. Không đọc từng ký hiệu như một đoạn code.

### Chuyển ý

Đạt chuyển sang giới hạn gradient trên slide 6.

### Nguồn

- Notebook: cell 14
- Giáo trình 5.1 Recurrent Neural Networks.pptx: slide 15 và 26

## Slide 6: RNN vẫn khó học phụ thuộc xa

**Người nói:** Đạt. **Thời lượng:** 65 giây.

### Lời thoại

Khi huấn luyện, BPTT truyền gradient ngược qua các bước thời gian để cập nhật trọng số dùng chung. Qua nhiều phép nhân, gradient có thể nhỏ dần. Khi đó tín hiệu lỗi tác động yếu đến các bước đầu, khiến RNN khó học phụ thuộc xa. Nhóm giới hạn chuỗi ở sáu mươi tư token để giảm số bước. Ma trận recurrent khởi tạo trực giao giúp giữ chuẩn vector ở thời điểm khởi tạo, nhưng tanh vẫn có thể làm gradient giảm. Pooling tạo đường truyền trực tiếp từ loss tới hidden state ở nhiều vị trí, nhưng có thể làm loãng tín hiệu. Đây là các lựa chọn thiết kế, nhóm chưa đo gradient hay làm ablation để tách riêng tác dụng. Gradient clipping mức một chỉ hạn chế gradient quá lớn. Nó không làm gradient nhỏ lớn trở lại. Tiếp theo là lần train thực tế.

### Thao tác và chỉ điểm

Chỉ ba lựa chọn và giới hạn tương ứng. Dừng ở dòng “chưa có ablation”. Chỉ phần clipping riêng ở cuối slide.

### Chuyển ý

Đạt chuyển slide 7.

### Nguồn

- Notebook: cell 14 và 18
- Giáo trình: slide 20–24 về BPTT và slide 42 về vanishing gradient
- PROPOSAL_RNN_TRUYEN_THONG.md: mục 4 (lý do thiết kế)

## Slide 7: Train đủ 6 epoch, chọn checkpoint ở epoch 5

**Người nói:** Đạt. **Thời lượng:** 60 giây.

### Lời thoại

Nhóm dùng Adam với learning rate không phẩy không không một, batch sáu mươi tư và sparse categorical cross-entropy. Class weight tính từ train tăng ảnh hưởng của lớp ít mẫu trong loss, nhưng không thay đổi phân phối dữ liệu. Lần chạy trên A một trăm ghi nhận khoảng bốn mươi bảy phẩy ba hai giây cho phần huấn luyện. Con số này không gồm toàn bộ thời gian xử lý và export notebook. Nhóm cấu hình EarlyStopping theo validation loss với patience hai, nhưng lần này vẫn chạy đủ sáu epoch. Epoch năm có validation loss thấp nhất, khoảng không phẩy tám tám hai, nên nhóm nạp lại checkpoint đó để đánh giá và export. Epoch sáu có validation loss tăng. Em mời Khang phân tích kết quả trên test.

### Thao tác và chỉ điểm

Chỉ thời gian có nhãn “train được log”. Chỉ điểm thấp nhất epoch 5 rồi điểm epoch 6. Dành 3 giây chuyển từ Đạt sang Khang, trong ngân sách slide.

### Chuyển ý

Đạt: “Em mời Khang phân tích kết quả trên test.”

### Nguồn

- ZIP: metrics.json, environment.json, training_history.csv, config.json
- Notebook: cell 18 và 25

## Slide 8: Macro-F1 cao hơn, Accuracy thấp hơn baseline

**Người nói:** Khang. **Thời lượng:** 80 giây.

### Lời thoại

Baseline luôn chọn Neutral, là lớp phổ biến nhất trong train. Trên cùng một nghìn chín trăm tám mươi bốn câu test, baseline đạt Accuracy sáu mươi tám phẩy chín phần trăm. SimpleRNN chỉ đạt sáu mươi ba phẩy bảy sáu phần trăm, nên mô hình không vượt baseline về Accuracy. Nhưng baseline có Macro-F1 hai mươi phẩy bốn phần trăm, còn SimpleRNN đạt năm mươi tư phẩy sáu sáu phần trăm. Macro-F1 lấy trung bình F1 của bốn lớp với trọng số bằng nhau, nên phản ánh thêm khả năng phân loại các lớp ít mẫu. Mức chênh ở đây là ba mươi tư phẩy hai sáu điểm phần trăm. F1 Toxic của model đạt bốn mươi tám phẩy năm ba phần trăm, trong khi baseline bằng không. Vì Toxic chỉ có năm mươi ba mẫu test, nhóm chưa thể kết luận rộng về độ ổn định của lớp này. Kết quả có giá trị để phân tích, đồng thời vẫn cho thấy giới hạn. Sau đây là demo bằng đúng model đã đánh giá.

### Thao tác và chỉ điểm

So sánh lần lượt Accuracy, Macro-F1 và F1 Toxic trên cùng thang 0–100%. Không che việc Accuracy giảm. Dành một nhịp dừng sau định nghĩa Macro-F1.

### Chuyển ý

Khang giới thiệu slide 9. Nhân chuẩn bị chuyển cửa sổ sang app đã mở và đã warm-up.

### Nguồn

- ZIP: metrics.json, model_comparison.csv, classification_report.csv, subset_counts.csv
- Phạm vi: subset phân tầng của split test NEU-ESC, không phải full test

## Slide 9: Demo nạp trực tiếp model đã lưu

**Người nói:** Khang nói, Nhân thao tác. **Thời lượng:** 90 giây.

### Lời thoại

Ứng dụng đang nạp model đã lưu, gồm vectorizer và trọng số từ lần train này. Demo chuẩn hóa bằng cùng preprocessing, chạy suy luận trên CPU và không train lại. Nhân giúp mình thử câu đầu. Với câu lời khen này, model chọn Tích cực, điểm cao nhất khoảng chín mươi hai phần trăm. Tiếp theo là câu không hài lòng. Lần này model chọn Tiêu cực. Nếu kịp, mình sửa câu đầu để quan sát phản ứng thực tế. Chúng ta đọc đúng kết quả đang hiển thị, kể cả khi nhãn không đổi. Bốn điểm là softmax, không phải độ chính xác bảo đảm cho từng câu. Các câu tự tạo này chỉ phục vụ demo. Nhân chuyển lại slide kết luận giúp mình.

### Thao tác và chỉ điểm

00–15 giây: Khang giải thích model đã nạp, Nhân chuyển sang app đã warm-up. 15–40 giây: Nhân dán câu A rồi bấm Phân tích, Khang đọc kết quả thật. 40–60 giây: dán câu B rồi bấm. 60–70 giây: nếu kịp, sửa sang câu C và bấm. 70–90 giây: Khang giải thích softmax/giới hạn, Nhân quay lại slide 10. Không chờ hơn 10 giây để xử lý lỗi app. Câu A: “Bài viết rất hữu ích, cảm ơn bạn đã chia sẻ!” Câu B: “Mình không hài lòng với cách xử lý này.” Câu C tùy chọn: “Bài viết không hữu ích, tôi rất thất vọng.” Khi chuẩn bị: A = Tích cực 92,224%; B = Tiêu cực 74,803%; C vẫn Tích cực 48,304%, cho thấy hạn chế với câu phủ định. Đây là dự đoán thật đã thử, không phải nhãn đúng để chấm điểm. Khi live, kết quả trên màn hình luôn là căn cứ. Nếu C vẫn Tích cực: nói “Câu này có sắc thái phàn nàn nhưng model vẫn chọn Tích cực. Điểm chỉ thể hiện phản ứng của model; đây là một ví dụ giới hạn.”

### Chuyển ý

Nhân quay lại PowerPoint ở slide 10 trước mốc 9:40.

### Nguồn

- streamlit_demo/app.py và inference.py
- Model/preprocessing/config/label_map khớp byte với ZIP chính thức
- Ảnh chụp ứng dụng thật tại http://127.0.0.1:8501 ngày 23/10/2026
- Suy luận câu tự tạo đã kiểm tra trên CPU bằng model thật, xem CHECKLIST_DEMO.md

## Slide 10: Pipeline chạy được, chất lượng còn giới hạn

**Người nói:** Khang. **Thời lượng:** 20 giây.

### Lời thoại

Nhóm đã xây pipeline RNN truyền thống và demo bằng model đã lưu. Mô hình còn hạn chế với mỉa mai, câu dài và lớp ít mẫu. LSTM hoặc GRU là hướng nghiên cứu tiếp theo, chưa triển khai. Nhóm xin kết thúc và dành thời gian cho câu hỏi.

### Thao tác và chỉ điểm

Chỉ hai ý trên slide. Dừng ở mốc 10:00. Nhân giữ slide 10 trong Q&A, chỉ mở phụ lục khi có câu hỏi liên quan.

### Chuyển ý

Khoảng 2 phút Q&A, không tính trong 600 giây trình bày.

### Nguồn

- Notebook, artifact thật và source Streamlit hiện có
- LSTM/GRU là hướng cải thiện chưa triển khai và chưa so sánh thực nghiệm

## Phụ lục dùng khi Q&A, ngoài 10 phút

### A1 (slide 11): Toxic có 53 mẫu, cần đọc cả precision và recall

Ma trận dùng hàng là nhãn thật và cột là nhãn dự đoán. Có 33 trong 53 câu Toxic được nhận ra, tương ứng recall 62,26%. Model dự đoán Toxic 83 lần nhưng chỉ đúng 33 lần, tương ứng precision 39,76%. Vì support nhỏ và còn nhiều nhầm lẫn, nhóm không xem đây là kết luận chắc chắn về Toxic.

**Thao tác:** Chỉ hàng Toxic và cột Toxic, rồi bảng precision/recall/F1/support. Bốn lớp theo thứ tự 0 Neutral, 1 Positive, 2 Negative, 3 Toxic.

**Nguồn:**

- ZIP: test_predictions.csv (tái lập confusion matrix), classification_report.csv, confusion_matrix.png

### A2 (slide 12): Epoch 5 có validation loss thấp nhất

Train chạy đủ sáu epoch. Validation loss thấp nhất ở epoch năm, sau đó tăng từ 0,8820 lên 1,0813 ở epoch sáu. Nhóm nạp lại checkpoint epoch năm. Hai loss có cách tính khác nhau: train dùng class weight, validation không dùng class weight, vì vậy không diễn giải độ lệch giữa hai đường như hai loss cùng trọng số. Đường cong không đủ để chứng minh từng biện pháp đã chữa vanishing gradient.

**Thao tác:** Chỉ hai điểm epoch 5 và 6. Nếu hỏi cấu hình: Adam lr 0,001, clipnorm 1, batch 64, sparse categorical cross-entropy, float32, jit_compile=False, recurrent_dropout=0, seed 42, EarlyStopping patience 2.

**Nguồn:**

- ZIP: training_history.csv, config.json, environment.json, training_curves.png
- Notebook: cell 14 và 18

### A3 (slide 13): Một artifact dùng cho cả đánh giá và demo

Notebook nạp checkpoint tốt nhất, ghép vectorizer với mạng rồi export model.keras. Ứng dụng load model với compile=False, chuẩn hóa bằng preprocessing.py của cùng artifact và suy luận trên CPU. Không adapt vocabulary hay train lại trong app. Môi trường local đã kiểm chứng Python 3.13.5, TensorFlow 2.21.0, Keras 3.13.2. Khi khởi chạy trên Windows cần bật UTF-8 như run_demo.cmd. Nếu chuyển máy, vẫn cần kiểm chứng môi trường mới.

**Thao tác:** Chỉ đường export/load và các tệp đi kèm. Mở CHECKLIST_DEMO.md nếu cần lệnh chi tiết.

**Nguồn:**

- F:\DPL project\Assignemtn 23.10\run_20261023_065529_443169.zip
- F:\DPL project\Assignemtn 23.10\NEU_ESC_SimpleRNN_A100.ipynb
- F:\DPL project\Assignemtn 23.10\HUONG_DAN_NOTEBOOK.md (hướng dẫn cũ, trạng thái train ưu tiên artifact)
- F:\DPL project\Assignemtn 23.10\PROPOSAL_RNN_TRUYEN_THONG.md (bối cảnh, không dùng số dự kiến thay số thật)
- F:\DPL project\Assignemtn 23.10\streamlit_demo\app.py và inference.py
- https://huggingface.co/datasets/hung20gg/NEU-ESC
- F:\1. FPT Material\Sem5\DPL302m\5.1 Recurrent Neural Networks.pptx: slide 15, 20–24, 26, 42

## Q&A ngắn

Trả lời trong khoảng 15–25 giây/câu, chỉ mở phụ lục phù hợp. Nếu cần số liệu chưa có, nói rõ chưa thực nghiệm thay vì suy đoán.

### 1. Vì sao dùng SimpleRNN thay vì LSTM/GRU?

Phạm vi assignment là RNN truyền thống. SimpleRNN giúp nhóm giải thích trực tiếp hidden state và chia sẻ trọng số, đồng thời có pipeline và baseline để đánh giá. Nhóm chưa triển khai đối chứng LSTM/GRU nên không kết luận SimpleRNN tốt hơn chúng.

### 2. RNN nhớ thông tin trước đó như thế nào? BPTT là gì?

Hidden state h_t được tính từ token hiện tại và h_(t−1), nên có thể mang thông tin từ phần trước của câu. BPTT trải RNN theo thời gian rồi truyền gradient ngược qua các bước. Đóng góp gradient ở các vị trí được cộng để cập nhật cùng bộ trọng số. Trạng thái không bảo đảm giữ nguyên mọi thông tin xa.

### 3. Vanishing gradient là gì? Thiết kế hiện tại giúp gì?

Gradient có thể nhỏ dần khi truyền qua nhiều bước, làm các bước đầu nhận tín hiệu học yếu. Chuỗi 64 token giảm số bước, khởi tạo recurrent trực giao hỗ trợ ở lúc khởi tạo, pooling tạo đường trực tiếp đến nhiều hidden state. Tanh vẫn có thể làm gradient giảm và pooling có thể làm loãng tín hiệu. Chưa có phép đo gradient hoặc ablation để định lượng tác dụng riêng.

### 4. Vì sao gradient clipping không chữa vanishing gradient?

Clipping giảm độ lớn gradient khi nó vượt ngưỡng, nhằm hạn chế exploding gradient. Nó không khuếch đại gradient đang quá nhỏ nên không khắc phục vanishing gradient.

### 5. Vì sao Accuracy thấp hơn baseline nhưng vẫn đáng phân tích?

Neutral chiếm 68,90% test nên baseline luôn Neutral có Accuracy cao nhưng bỏ lỡ cả ba lớp còn lại. Macro-F1 cho bốn lớp trọng số bằng nhau. SimpleRNN có Macro-F1 54,66% so với 20,40%, còn Accuracy 63,76% thấp hơn 68,90%. Kết quả tốt hơn theo Macro-F1, không tốt hơn trên mọi chỉ số.

### 6. Class weight khác lấy mẫu cân bằng thế nào?

Class weight đổi trọng số mỗi lớp trong loss, không thêm hay bớt câu. Lấy mẫu cân bằng thay đổi số lượng hoặc tần suất mẫu của từng lớp. Nhóm lấy phân tầng gần tỷ lệ gốc và dùng class weight N/(4*n_lớp) tính từ train. Không có ablation để nói class weight làm điểm tăng bao nhiêu.

### 7. Vì sao lấy 30% và tránh leakage thế nào?

Đó là lựa chọn vừa sức assignment, còn 6.914 câu train và 175 câu Toxic. Chưa chứng minh đây là tỷ lệ tối ưu. Nhóm giữ ba split gốc, lấy 30% riêng từng split, seed 42, không lấy mẫu lặp. Vocabulary và class weight chỉ học từ subset train, chọn checkpoint trên validation rồi đánh giá test. Notebook kiểm tra trùng hoàn toàn sau chuẩn hóa, chưa bảo đảm loại hết câu gần trùng.

### 8. Vì sao chọn epoch 5 dù train đủ 6?

Tiêu chí chọn model là validation loss thấp nhất. Epoch 5 đạt 0,8820; epoch 6 tăng lên 1,0813. Notebook nạp lại checkpoint epoch 5. EarlyStopping patience 2 được cấu hình nhưng không dừng sớm lần này vì đã chạm mức tối đa 6 epoch trước khi đủ điều kiện dừng.

### 9. Streamlit có dùng model đã train không? Có cần GPU không?

Có. model.keras gồm vectorizer và trọng số đã train; model và preprocessing/config/label_map trong app khớp ZIP chính thức. App load compile=False, không train hoặc adapt. Nhóm đã kiểm tra suy luận và giao diện bằng CPU trên môi trường local hiện có. Không cần kết nối Colab. Máy khác vẫn cần cài đúng thư viện và chạy kiểm tra.

### 10. Có thể tin chắc điểm softmax hoặc kết luận về Toxic không?

Không. Softmax là phân bố điểm tương đối của model, chưa được hiệu chuẩn xác suất và không phải độ chính xác của một câu. Toxic chỉ có 53 mẫu test, precision 39,76%, recall 62,26%, F1 48,53%. Support nhỏ và còn nhầm lẫn nên chưa thể khái quát rộng. Chưa đánh giá full test gốc.

### 11. Model xử lý câu dài và phủ định thế nào?

Model giữ 64 token đầu theo khoảng trắng, có thể mất thông tin cuối câu. Nhóm giữ từ phủ định khi chuẩn hóa nhưng điều đó không bảo đảm model hiểu đúng phủ định. Câu C thử trước vẫn bị gán Tích cực dù mang sắc thái phàn nàn.

### 12. Tại sao loss train và validation khác nhau?

Train loss có class weight còn validation loss không có. Hai đường không cùng cách tính trọng số, nên không dùng chênh lệch trực tiếp để kết luận đơn giản về chất lượng hoặc overfitting. Có thể theo dõi validation loss qua các epoch để chọn checkpoint.

## Bảng dữ liệu đầy đủ để đối chiếu khi hỏi

| Split subset | Neutral | Positive | Negative | Toxic | Tổng |
|---|---:|---:|---:|---:|---:|
| Train | 4.781 | 869 | 1.089 | 175 | 6.914 |
| Validation | 684 | 125 | 157 | 26 | 992 |
| Test | 1.367 | 250 | 314 | 53 | 1.984 |

Validation ở checkpoint 5: Accuracy **63,21%**, Macro-F1 **53,31%**. Test chỉ trên subset phân tầng: Accuracy **63,76%**, Macro-F1 **54,66%**, F1 Toxic **48,53%**. F1 Neutral **73,80%**, Positive **47,66%**, Negative **48,64%**.

Cấu hình bổ sung: một SimpleRNN 64 units, tanh, recurrent_initializer=orthogonal, recurrent_dropout=0, return_sequences=True. Embedding 64 chiều, mask_zero=True. GlobalAveragePooling1D nhận mask, Dropout 0,2, Dense 4 softmax. Adam lr 0,001, clipnorm 1, sparse categorical cross-entropy, float32, jit_compile=False, batch 64, tối đa 6 epoch. EarlyStopping monitor val_loss, patience 2, restore_best_weights=True. Notebook luôn nạp checkpoint tốt nhất trước đánh giá/export. Class weight 0..3 lần lượt khoảng **0,3615 / 1,9891 / 1,5872 / 9,8771**.

## Ghi chú bàn giao

HUONG_DAN_NOTEBOOK.md và proposal là tài liệu chuẩn bị trước lần train chính thức nên có câu trạng thái đã cũ. Bộ slide ưu tiên notebook và ZIP kết quả thật. Không dùng số tham số dự kiến trong proposal như số đo thật. Không train lại, không sửa notebook hoặc ứng dụng để tạo bộ trình bày này.

PPTX có speaker notes cho cả 13 slide. Biểu đồ, bảng và sơ đồ là đối tượng native chỉnh sửa được. Ảnh demo là ảnh thật đã chụp trong lần kiểm tra local; ảnh không tương tác. PDF là bản xem dự phòng dựng từ ảnh render, không chứa speaker notes và không dùng thay bản chỉnh sửa. Chưa rehearsal bằng người thật hoặc kiểm tra trình chiếu tại lớp; cần dùng đúng máy demo để tập một lượt.
