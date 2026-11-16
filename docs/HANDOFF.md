# Bàn giao — Nhóm 20, Đề tài 7

V1–V5 gồm các cấu hình E1–E7, hai biến thể và kiểm định chéo train. Quy tắc Macro F1 validation rồi Accuracy trên toàn bộ bảng chọn `E6_bins32` (0,2151), nhưng test đã được chạy cho E1 và E6 16 bin. Không có điểm test `E6_bins32`; không chạy lại test. Các giới hạn chọn mô hình và manifest ghi trong [kiểm toán quy trình](kiem_toan_quy_trinh.md).

Dữ liệu gốc và pickle không được lưu trong Git. `results/v5/`, `results/final_test/`, `results/v4/` có các bảng JSON/CSV nhỏ; `results/notebooks/` đã theo dõi là log chạy lịch sử. Nguồn báo cáo/PDF ở `reports/latex/`; tên và phân công thành viên cần nhóm rà soát. Cài môi trường, chuẩn bị dữ liệu, chạy notebook và UI theo [README](../README.md). Chỉ nạp pickle từ nguồn tin cậy.

Trước khi kết luận về `E6_bins32`, cần holdout mới độc lập đã kiểm tra trùng; test Oxford đã từng được quan sát và có ảnh trùng qua split. Không tự xóa khóa/output để thử lại test. Xem [kiểm toán dữ liệu](V4_DATASET_AUDIT.md) và `docs/history/` cho lịch sử V1–V4.
