# Công bố sử dụng AI

OpenAI Codex hỗ trợ xây dựng mã nguồn, notebook, tài liệu và kiểm thử V1; kiểm tra repository, cài môi trường cục bộ, đối chiếu hash, chạy thí nghiệm và ghi lại giới hạn môi trường. Không tạo số liệu thực nghiệm giả. Nhóm chịu trách nhiệm kiểm tra mã, thực thi trên dữ liệu chính thức và diễn giải kết quả.

V2 sử dụng Codex để hỗ trợ thiết kế đặc trưng, tìm cấu hình trên validation, kiểm thử và viết tài liệu. Các điểm được báo cáo đến từ lần chạy thực tế; không có kết quả test V2.

V3 sử dụng Codex cho kiểm định chéo chỉ trên train, kiểm tra, notebook và tài liệu. Nhận xét về ảnh lấy từ ví dụ validation đã lưu; giải thích nguyên nhân chỉ là giả thuyết. Không chạy đánh giá test mới.

V4 sử dụng Codex cho kiểm toán dữ liệu, suy luận E6 đóng băng, demo, kiểm thử và tài liệu. Sáu cặp ứng viên pHash không được khẳng định là trùng byte. Không chạy đánh giá test mới. Sau đó Codex hỗ trợ dọn repository, bàn giao và bản địa hóa tiếng Việt. Thành viên nhóm phải rà soát nội dung, ghi tên tác giả và công bố việc dùng AI theo quy định môn học.

Theo yêu cầu của chủ dự án, báo cáo tiếng Việt sử dụng trực tiếp cấu trúc, cấu hình, bìa và logo của template GitHub được chỉ định; thông tin mẫu được thay bằng thông tin nhóm do người dùng cung cấp hoặc placeholder cho các mục chưa biết. Repository mẫu không có tệp giấy phép rõ ràng; nhóm cần xác nhận quyền sử dụng và thay logo nếu không đúng trường.

Theo yêu cầu của nhóm, Codex hỗ trợ tính đường học E1/E4/E6 từ các fold train, kiểm tra kết quả và viết lại phần giải thích theo giọng sinh viên. Nhóm cần tự rà soát nội dung và đóng góp; không có đánh giá test E6 mới.

Theo yêu cầu của nhóm, Codex cập nhật thông tin bìa, mở rộng phần giải thích, cấu hình notebook tự chạy và dọn output. Nội dung report dùng số liệu đã được chạy trước khi xóa output; nhóm cần kiểm tra kết quả mới sau khi chạy lại.

Codex hỗ trợ sửa hướng dẫn cài đặt, đăng ký kernel `.venv` trong dự án, chuẩn bị model E6 từ train/validation và kiểm thử lại notebook/UI. Không có đánh giá test mới trong bước này.
