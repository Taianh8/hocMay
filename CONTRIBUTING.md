# Hướng dẫn đóng góp

Làm việc trên nhánh riêng, dùng Python 3.11 và môi trường `.venv` từ `requirements-v4-lock.txt`. Lock đã kiểm chứng trên macOS arm64; nền tảng khác có thể cần wheel khác. Không đưa dữ liệu gốc, cache, model pickle hoặc nhị phân lớn vào Git. Bảng CSV/JSON kết quả nhỏ có thể được theo dõi sau khi rà soát. Giữ nguyên split chính thức, mã thí nghiệm E1–E6, kết quả và nguồn gốc thực nghiệm.

Mọi phép biến đổi có học tham số phải được fit trong fold huấn luyện. Chọn cấu hình bằng validation; không dùng test để chọn hay điều chỉnh. Notebook 05 và `python -m src.pipeline all` chạy lại đánh giá test V1, không dùng cho QA thường quy. `python -m src.pipeline_v2` ghi lại artifact V2; dùng bản sao cách ly khi cần tái lập. Không chạy test E6 nếu chưa được chủ dự án cho phép riêng.

Trước khi bàn giao, chạy lại 11 notebook theo README, tạo manifest mới nếu cần kiểm thử model, rồi chạy `python -m pytest -q` và `python scripts/verify_v4_notebooks.py`. Lệnh `python -m scripts.check_v4_audit` chỉ dùng sau khi đã tạo lại kiểm toán V4. Kiểm thử V4 cần dữ liệu, cache và model E6 tin cậy. Nêu rõ lệnh, kết quả đã quan sát và giới hạn; ghi nhận hỗ trợ AI trong `AI_USAGE.md`.

Pickle có thể thực thi mã. Chỉ nạp model và manifest tin cậy; hash chỉ phát hiện thay đổi ngoài ý muốn, không xác thực hai file bị thay thế cùng lúc. Thay đổi model hoặc preprocessing cần manifest phiên bản mới được rà soát, không ghi đè bản đã đóng băng.
