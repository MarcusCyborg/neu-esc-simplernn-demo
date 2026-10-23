# Giao diện demo

Bố cục và màu sắc thực hiện theo plan người dùng: một trang Streamlit nền sáng,
nhập câu rồi chọn mẫu, bấm Phân tích, đọc kết quả ngay bên dưới. Thông tin mô hình
và đánh giá trong expander cuối trang. Không thêm màn hình hoặc điều hướng.

Mục tiêu: thao tác nhanh khi trình chiếu. Dùng component sẵn có, không font tải từ
mạng, không animation trang trí. Trạng thái rỗng, đang xử lý, lỗi và thiếu artifact
đều có thông báo tiếng Việt. Kết quả biến mất khi thay câu.

Bốn nhãn luôn có chữ bên cạnh màu: Trung tính xám, Tích cực xanh lá,
Tiêu cực cam, Độc hại đỏ. Biểu đồ ngang dùng cùng thứ tự ID 0, 1, 2, 3.
Số liệu lấy từ artifact, ghi rõ điểm softmax và phạm vi subset test.
