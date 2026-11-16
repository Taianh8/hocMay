# Đóng góp

Dùng Python 3.11 và `requirements-v4-lock.txt`; giữ split Oxford, không đưa ảnh, cache hoặc pickle vào Git. Mọi scaler/PCA phải fit trên train hoặc fold train; chỉ chọn cấu hình bằng validation. Test đã được thực hiện cho E1 và E6 16 bin; **không chạy lại**. Đọc [kiểm toán quy trình](docs/kiem_toan_quy_trinh.md) trước khi diễn giải kết quả.

Chạy `python -m pytest -q` và báo rõ các bước đã kiểm tra. Output CSV/JSON nhỏ trong `results/v5`, `results/final_test`, `results/v4` là bằng chứng; notebook đã thực thi trong `results/notebooks` là log lịch sử, không phải mã nguồn. Chỉ nạp pickle từ nguồn tin cậy. Ghi hỗ trợ AI trong `AI_USAGE.md`; không tự gán phần việc cho thành viên.
