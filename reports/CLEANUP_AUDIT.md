# Kiểm toán dọn dẹp và danh sách xóa

**Quyết định mới hơn:** Sau đợt cleanup này, nhóm yêu cầu xóa toàn bộ output thực nghiệm và cache; danh sách bên dưới ghi quyết định ở đợt cleanup trước. Xem README để chạy lại.

Phạm vi: giữ working tree, lịch sử Git, cấu hình và điểm E1–E6, CV V3, kiểm toán V4, notebook nguồn, model E6 và manifest, hình, dữ liệu và tài liệu. Không được chạy test mới trong cleanup.

| Quyết định | Tệp hoặc nhóm | Lý do |
|---|---|---|
| XÓA | `.pytest_cache/`, `scripts/__pycache__/`, `src/__pycache__/`, `tests/__pycache__/` | Cache có thể tạo lại, không phải bằng chứng nghiên cứu. |
| XÓA | `results/v4/main.pdf`, `results/v4/main.log` | Lần thử TeX cũ có placeholder thiếu hình, không phải bản đã kiểm tra. |
| XÓA | `results/v4/qa/main.pdf`, `results/v4/qa/main.log` và thư mục `qa/` rỗng | Lần QA cũ bị thay thế bởi bản bốn trang ở `results/v4/qa2/` tại thời điểm cleanup. |
| GIỮ | `data/raw/`, `data/processed/`, `.venv/` | Dữ liệu chính thức, cache có checksum và môi trường cục bộ; bị Git bỏ qua. |
| GIỮ | `results/metrics/`, `results/v2/metrics/`, `results/v3/metrics/`, `results/v4/dataset_audit.json`, `results/v4/duplicate_candidates.csv` | Điểm, dự đoán, fold, nguồn gốc và bằng chứng kiểm toán. |
| GIỮ | `results/figures/`, `results/v2/figures/`, `results/v3/figures/`, `results/v4/duplicate_candidate_pairs.png` | Hình và ma trận đã kiểm tra; hình sinh ra vẫn bị Git bỏ qua. |
| GIỮ | `results/models/`, `results/v2/models/` | Model lịch sử, gồm E6; pickle chỉ nạp từ nguồn tin cậy và bị Git bỏ qua. |
| GIỮ | `results/notebooks/`, `results/v2/notebooks/`, `results/v2/notebook_v1_check/`, `results/v3/notebooks/`, `results/v4/notebooks/` | Bằng chứng thực thi và output cách ly; chưa giả định là dư thừa. |
| GIỮ | `results/v3/work/`, log và handoff V1–V4 | Chưa chắc không còn giá trị hoặc nguồn gốc; dữ liệu sinh ra vẫn bị ignore. |
| GIỮ | `requirements.txt`, `requirements-lock.txt`, `requirements-v4-lock.txt` | Lần lượt là khoảng phiên bản, lock lịch sử và lock V4 với Streamlit; không gộp. |
| CHỈNH | Output notebook và log thực thi cục bộ | Thay tiền tố đường dẫn máy bằng đường dẫn tương đối; giữ nội dung và điểm. |
| CHỈNH | `.gitignore`, README và handoff | Bỏ qua file lớn nhưng cho phép theo dõi bảng CSV/JSON nhỏ cần cho nguồn gốc. |
| GIỮ | Mã nguồn, script, cấu hình, notebook và handoff còn lại | Không có module/script nào được chứng minh là lỗi thời. |

Localhost trong lệnh Streamlit là loopback có chủ đích. Nguồn LaTeX V4 cũ biến mất trong cleanup và chủ dự án chọn không khôi phục; sau đó chủ dự án yêu cầu báo cáo tiếng Việt mới. Quét mẫu thông tin nhạy cảm không phải kiểm toán bảo mật và không chứng minh chắc chắn không có bí mật. Test cuối E6 vẫn chờ.
