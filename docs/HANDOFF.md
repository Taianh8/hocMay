# Bàn giao — Nhóm 20, Đề tài 7

Repository chứa mã phân loại ảnh hoa, cấu hình E1–E6, 11 notebook, kiểm thử và [báo cáo](reports/latex/main.pdf). Theo yêu cầu của nhóm, output thực nghiệm, model sinh ra, cache đặc trưng và bản notebook đã thực thi đã được xóa. Giữ lại dữ liệu Oxford gốc cục bộ, mã, cấu hình, nguồn/PDF và những hình cần thiết để biên dịch báo cáo.

Cách chạy đầy đủ từng bước ở [README.md](README.md): tạo môi trường Python 3.11, chuẩn bị dữ liệu chính thức, chạy notebook 00–10 hoặc các lệnh thay thế, rồi tạo manifest E6 mới nếu muốn dùng ứng dụng. Notebook 05–08 mặc định chỉ đánh giá và phân tích validation; không tự chạy test. Output mới nằm trong `results/` và `data/processed/`.

Báo cáo ghi kết quả từ các lần chạy trước khi dọn output: E6 có F1 vĩ mô validation 0,2127; E4 0,2003. Đường học được tính từ 60 lần fit trên các fold train; điểm F1 train của K-NN bị lạc quan vì mỗi mẫu có thể là láng giềng của chính nó. Không dùng các con số trong báo cáo làm output của lần chạy mới nếu chưa tự kiểm chứng.

Không chạy `python -m src.pipeline all` hoặc `python -m src.pipeline evaluate` khi chưa có sự cho phép riêng vì chúng đánh giá test V1. E6 chưa được đánh giá test cuối. Test V1 từng được quan sát và kiểm toán V4 phát hiện train 7307 trùng byte test 7309; không coi test là một holdout hoàn toàn độc lập. Xem `reports/V4_DATASET_AUDIT.md` về giới hạn.

Khi cần ứng dụng, chạy xong V2/V3 rồi tạo manifest mới theo README; không thay đổi manifest lịch sử để ép nhận một model khác. Chỉ nạp pickle và manifest từ nguồn tin cậy. Nhóm cần kiểm tra upload bằng trình duyệt thực, rà lại báo cáo và thông tin bìa trước khi nộp. Không commit hoặc push khi chưa được yêu cầu.
