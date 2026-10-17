# AI Cố vấn nhập học

Ứng dụng Streamlit hỏi–đáp dựa trên file Excel câu hỏi thường gặp, sử dụng mô hình embedding đa ngôn ngữ để tìm câu hỏi gần nghĩa.

## Chạy thử trên máy tính
1. Cài Python 3.10 hoặc 3.11.
2. Mở terminal trong thư mục dự án.
3. Chạy `pip install -r requirements.txt`.
4. Chạy `streamlit run app.py`.
5. Mở địa chỉ hiện ra trong terminal.

## Triển khai trên Streamlit Community Cloud
1. Tạo một repository mới trên GitHub.
2. Tải `app.py`, `requirements.txt`, `README.md` và `knowledge_base.xlsx` lên repository.
3. Vào https://share.streamlit.io/ và đăng nhập bằng GitHub.
4. Chọn tạo ứng dụng, chọn repository, nhánh `main`, file chính `app.py`.
5. Nhấn Deploy và đợi ứng dụng khởi động.

## Lưu ý
- Lần đầu chạy sẽ tải mô hình `paraphrase-multilingual-MiniLM-L12-v2`, nên có thể mất vài phút.
- Tài nguyên miễn phí có giới hạn; ứng dụng có thể ngủ hoặc cần khởi động lại.
- Kiểm tra độ chính xác của câu trả lời và cập nhật nội dung học vụ trước khi chia sẻ rộng rãi.
- Đây là dự án độc lập, không phải website chính thức của Học viện Ngân hàng.
- Không đưa dữ liệu cá nhân hoặc thông tin nội bộ không được phép công khai vào repository.
