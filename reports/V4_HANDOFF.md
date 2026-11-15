# Bàn giao V4 — QA, demo và trạng thái test E6

**Cập nhật sau bàn giao:** Theo yêu cầu của nhóm, output thực nghiệm, cache và model sinh ra đã được xóa; cần chạy lại theo README. Những đoạn dưới ghi tình trạng tại thời điểm V4, không khẳng định các artifact vẫn còn trong workspace hiện tại.

V4 gồm kiểm toán dataset/split, manifest E6 đóng băng, suy luận, demo Streamlit, kiểm thử tích hợp và hướng dẫn tái lập. Cấu hình và điểm V1–V3 được giữ. Chưa chạy đánh giá test E6 trong V2–V4. Kiểm toán V4 đã xem ảnh test để tìm ảnh trùng; V1 cũng đã quan sát điểm test, nên test chính thức không còn là holdout hoàn toàn chưa được xem. Bản nháp LaTeX cũ từng được chủ dự án chọn để xóa; bản báo cáo tiếng Việt mới được tạo sau đó theo yêu cầu mới.

## Dữ liệu và rò rỉ

Lần kiểm toán đầy đủ giải mã 8.189 ảnh RGB, xác thực 102 lớp và split 1.020/1.020/6.149 không giao nhau, kiểm tra 13.584.360 cặp qua split. Hai cặp trùng byte/RGB: train 7307 ↔ test 7309, validation 8077 ↔ test 8067. Sáu ứng viên pHash khác có khoảng cách Hamming ≤4; gồm một cặp train–validation và một train–test. Tám cặp và giới hạn ở `reports/V4_DATASET_AUDIT.md` và `results/v4/`. Không xóa hay gán lại nhãn. Lần kiểm tra tiếp theo xác thực giải mã, metadata, checksum CSV, archive, metadata và tám cặp; không tìm lại toàn bộ pHash.

Mã tiền xử lý, HOG và chuẩn hóa khối HSV hoạt động độc lập theo ảnh; không có scaler học tham số. PCA V2 fit train chính thức; PCA V3 fit trong train từng fold. E6 được chọn bằng Macro F1 validation sau 56 cấu hình; khoảng cách so E4 trong CV nhỏ và có nguy cơ lạc quan. Phản hồi test V1 và ảnh trùng là giới hạn quan trọng.

## Model và demo

`configs/e6_manifest_v1.json` đóng băng E6: HOG 8 + histogram HSV, PCA randomized 128 thành phần seed 42, K-NN cosine k=3, trọng số theo khoảng cách; fit trên 1.020 ảnh train. Manifest ghi split, tiền xử lý, phiên bản và SHA-256 mã, dữ liệu, model và kết quả. `src/inference.py` đối chiếu hash model/mã, phiên bản thư viện chính, PCA, K-NN, nhãn và cấu hình tiền xử lý trước khi dự đoán; không fit và trả ID 1–102. Chỉ nạp pickle và manifest tin cậy: hash không xác thực khi kẻ tấn công thay cả hai.

`app.py` nhận JPEG/PNG tối đa 10 MiB và 20 MP, trả mã lớp, không gán phiếu K-NN thành confidence. `requirements-v4-lock.txt` ghi môi trường có Streamlit. Model, dataset, cache và hình lớn không có trong Git; cần giữ bản tin cậy khớp manifest. Lần trước server localhost trả health `ok`, AppTest kiểm tra trang đầu và nhánh upload với uploader mô phỏng dùng ảnh train. Chưa kiểm tra upload trong trình duyệt thực do browser host không khả dụng.

## Kiểm chứng và bước tiếp

Lần V4 trước có 18 kiểm thử pass, gồm khớp đặc trưng/dự đoán validation E6, dữ liệu train, từ chối hash/cấu hình bị sửa, ảnh lỗi và upload mô phỏng. `python -m scripts.check_v4_audit` pass; CLI trên ảnh train 6765 trả ID 1. Các script kiểm tra V2/V3 từng pass. `scripts.verify_v4_notebooks` chạy notebook báo cáo 09, 10 và kiểm tra bản thực thi lưu sẵn 00–08; không chạy notebook 05 đánh giá test. Thành viên nhóm cần rà rubric, tác giả và báo cáo, kiểm tra upload thực, sao lưu artifact bị ignore.

E6 được khóa cho phương án fit train-only seed 42. Chỉ chạy test sau khi được cho phép riêng và định trước artifact, chính sách train, chỉ số và báo cáo; không điều chỉnh theo test. Cặp trùng train–test và phản hồi V1 làm giảm tính độc lập của test; nên tìm holdout mới đã kiểm tra trùng. Train thêm validation là quy trình khác, không tự ý thay thế.
