# V4 — Kiểm toán toàn vẹn dữ liệu và ảnh trùng

**Cập nhật:** Output kiểm toán đã được xóa theo yêu cầu; các cặp dưới đây là ghi nhận lịch sử. Chạy lại lệnh cuối tài liệu để tạo output mới.

Lần kiểm toán đã xác thực đủ 8.189 tệp ảnh, giải mã RGB, 102 lớp và split chính thức đầy đủ, không giao nhau: 1.020 train, 1.020 validation, 6.149 test. `imagelabels.mat` ánh xạ ID `i` với phần tử `i-1`; nhãn nội bộ bằng mã lớp chính thức trừ một. SHA-256 archive và hai tệp metadata khớp `configs/e6_manifest_v1.json`; MD5 cũng khớp metadata torchvision đã đối chiếu. Không thay đổi mẫu hay split.

## Phương pháp và kết quả

Đã xét **13.584.360** cặp qua các split. Trùng chính xác được kiểm tra bằng SHA-256 của byte JPEG và riêng byte RGB sau giải mã kèm kích thước. Ứng viên gần trùng dùng pHash DCT 64 bit: ảnh xám 32×32, lấy 8×8 hệ số tần số thấp và ngưỡng median của 63 hệ số không phải DC; báo cáo khoảng cách Hamming ≤4. Ngưỡng dùng để sàng lọc, chưa hiệu chuẩn trên tập cặp có nhãn. Mọi cặp trùng hash chính xác đều được báo cáo dù pHash khác; pHash không chứng minh trùng byte.

| Split A | Ảnh A | Split B | Ảnh B | Lớp | Khoảng cách pHash | Trùng byte |
|---|---:|---|---:|---:|---:|---|
| validation | 1835 | test | 1858 | 78 | 0 | Không |
| validation | 7015 | test | 7040 | 39 | 0 | Không |
| validation | 7029 | test | 8080 | 39 | 0 | Không |
| train | 7307 | test | 7309 | 94 | 0 | Có |
| validation | 8077 | test | 8067 | 31 | 0 | Có |
| validation | 1929 | test | 1959 | 78 | 2 | Không |
| train | 1855 | test | 1844 | 78 | 4 | Không |
| train | 5746 | validation | 5768 | 13 | 4 | Không |

Tám cặp cùng nhãn; hai cặp trùng byte và RGB sau giải mã. Bảng máy đọc được ở `results/v4/duplicate_candidates.csv`, tóm tắt ở `results/v4/dataset_audit.json`, hình đối chiếu ở `results/v4/duplicate_candidate_pairs.png` nếu được giữ cục bộ. Ảnh ứng viên có bố cục gần nhau, nhưng chưa chứng minh cùng ảnh gốc. pHash có thể bỏ sót ảnh cắt, xoay, lật, thay màu hoặc ánh sáng và có thể cho dương tính giả. Chưa kiểm tra đầy đủ ảnh trùng trong cùng split hoặc quan hệ giữa các buổi chụp; chưa biết ảnh hưởng đến chỉ số.

Cặp train–test trùng byte là chồng lấn trực tiếp; ứng viên train–validation và các cặp validation–test cũng ảnh hưởng diễn giải lựa chọn và tổng quát hóa. Không tự ý xóa ảnh hoặc đổi split. Nên công bố giới hạn và tìm holdout độc lập đã kiểm tra trùng. Kiểm toán đã xem ảnh test nhưng không chạy đánh giá model test mới.

`python -m scripts.check_v4_audit` kiểm tra giải mã, metadata, tám cặp và checksum archive/metadata, không tìm lại toàn bộ cặp hoặc so từng JPEG giải nén với archive. Muốn tìm lại đầy đủ, chạy `python -m src.dataset_audit` rồi `python -m scripts.duplicate_contact_sheet`; cần sao lưu đầu ra V4 trước vì lệnh sẽ ghi đè.
