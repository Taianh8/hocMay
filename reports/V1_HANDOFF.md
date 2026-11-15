# Bàn giao V1 — baseline đã kiểm chứng

**Cập nhật:** Các output thực nghiệm và cache đã được xóa theo yêu cầu. Tài liệu này ghi lại kết quả lịch sử; chạy lại theo README để tạo output mới.
Lần chạy lịch sử dùng Python 3.11.15 trên macOS arm64; phiên bản ở `requirements-lock.txt`. Import, `pip check` và 6 kiểm thử khi đó đều thành công. Chín notebook đã chạy theo thứ tự; output và thời gian ở `results/notebooks/`. Jupyter cần loopback; lần chạy trước cần quyền ngoài sandbox. Khi đó notebook có cảnh báo thiếu cell ID; cảnh báo đã được xử lý ở V2.

Archive ảnh lấy từ bản sao được liên kết trong README và khớp MD5 torchvision; hai tệp metadata Oxford cũng khớp. Đã giải mã đủ 8.189 ảnh, 102 lớp, split chính thức 1.020 train, 1.020 validation, 6.149 test. Mỗi lớp có 10 ảnh train và validation; test có 20–238 ảnh/lớp. Tất cả ảnh ở chế độ RGB; kích thước và ID ghi trong `results/metrics/dataset_metadata.csv`.

Cấu hình tốt nhất trên validation (Accuracy / Macro F1): E1 pixel + K-NN k=1 là 0,1088 / 0,0914; E2 HOG + K-NN k=1 là 0,0608 / 0,0584; E3 HOG + cây quyết định không giới hạn độ sâu, min leaf=1 là 0,0245 / 0,0230. Chỉ chọn bằng validation. E1 từng được đánh giá test; output test đã được xóa theo yêu cầu. Các cấu hình khác không được đánh giá test. HOG không vượt pixel trong lưới này; điểm thấp chỉ phản ánh baseline, không chứng minh sẵn sàng triển khai.

`results/metrics/validation_search.json` ghi các cấu hình và thời gian; `selection.json`, `test_metrics.json`, `test_predictions.csv`, `confusion_matrix.npy`, `confused_pairs.csv` ghi lựa chọn và đầu ra. `results/figures/` chứa EDA, tiền xử lý, ma trận và ảnh lỗi; `results/models/selected_model.pkl` chỉ dùng từ nguồn tin cậy. Cache có 8.189 × 12.288 pixel và 8.189 × 1.764 HOG float32, khoảng 439 MiB chưa nén; cache nén khoảng 172 MiB. Lần chạy lại đo RSS đỉnh 1.226.932.224 byte, khoảng 1,14 GiB; notebook khoảng 1,13 GiB. Đây là số đo riêng máy, không phải giới hạn bộ nhớ chung.

Split giữ nguyên; chọn theo Macro F1 validation, Accuracy phá hòa, tiếp theo theo thứ tự cấu hình. Model cuối chỉ fit train, không có biến đổi học tham số. ID xuất ra là 1–102. Resize vuông biến dạng tỉ lệ, HOG bỏ màu, chưa có bảng tên lớp; test mất cân bằng. Xem test lặp lại hoặc tinh chỉnh theo test làm sai lệch kết luận. Phiên bản được ghim theo nền tảng, không ghim hash wheel; MD5 chỉ kiểm tra nhất quán, không phải chữ ký.

Việc được đề xuất lúc bàn giao: quan sát lỗi dựa trên bằng chứng, bổ sung cell ID, nghiên cứu PCA/scaling chỉ trên train hoặc resize giữ tỉ lệ dùng validation. Những việc sau đó được ghi riêng trong V2–V4. Xem README để chạy đúng quy trình hiện tại; không chạy test V1 lại khi QA thường quy.
