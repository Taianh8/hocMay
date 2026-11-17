# Phân loại ảnh hoa — Nhóm 20

Bài tập lớn FIT4013 dùng Oxford 102 Flowers và các mô hình K-NN, cây quyết định. Các cấu hình được chọn bằng validation; PCA được fit trên train. Bộ dữ liệu và model nhị phân không nằm trong Git.

**Sai lệch cần đọc trước:** E6 16 bin đã được đưa đi test một lần, nhưng nếu áp dụng quy tắc Macro F1 validation rồi Accuracy cho cả biến thể đối chứng, `E6_bins32` mới đứng đầu. Không có điểm test cho `E6_bins32`; không chạy lại test. Xem [kiểm toán quy trình](docs/kiem_toan_quy_trinh.md).

## Kết quả đã lưu

| Cấu hình | F1 validation | F1 CV train | F1 test |
|---|---:|---:|---:|
| E1: pixel + K-NN | 0,0914 | 0,0877 | 0,1103 |
| E2: HOG + K-NN | 0,0584 | — | — |
| E3: HOG + cây quyết định | 0,0230 | — | — |
| E4: HOG + HSV + K-NN | 0,2003 | 0,1852 | — |
| E5: pixel + PCA + K-NN | 0,1334 | — | — |
| E6: HOG + HSV + PCA + K-NN | 0,2127 | 0,1932 | 0,2349 |
| E7: HOG + HSV + cây quyết định | 0,0891 | 0,0695 | — |
| E6_size128 | 0,2034 | 0,1953 | — |
| **E6_bins32** | **0,2151** | 0,1907 | **chưa có** |

CV lấy năm fold từ train; dấu `—` nghĩa là chưa báo cáo. Test E1/E6 dùng model fit lại trên train+validation, không phải model train-only trong UI. Accuracy test tương ứng 0,1345 và 0,2602. Phân tích độ nhạy chỉ loại hai ảnh test trùng byte đã biết; không loại mọi ảnh gần trùng. Nguồn số liệu: `results/v5/unified_comparison_v2.json`, `results/final_test/metrics.json`.

## Chạy dự án

### 1. Cài đặt môi trường và dữ liệu

Tải và chuẩn bị [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) bằng script (script tự kiểm tra SHA-256 và giải nén ảnh):

```bash
python scripts/download_dataset.py
```

Script có thể chạy lại an toàn; dùng `--force` để tải lại. Dữ liệu được đặt trong `data/raw/` với split chính thức 1.020 ảnh train, 1.020 ảnh validation và 6.149 ảnh test.

Cài Python 3.11, tạo và kích hoạt môi trường:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v4-lock.txt
```

Trên Windows dùng `py -3.11 -m venv .venv` rồi `.venv\Scripts\activate`. Lock đã kiểm chứng trên macOS arm64; máy khác có thể cần wheel phù hợp.

Nếu dùng VS Code/Jupyter, đăng ký kernel của môi trường:

```bash
python scripts/cai_dat_kernel.py
```

### 2. Chạy notebook

Trong VS Code mở `notebooks/00_data_source.ipynb`, chọn kernel **Hoa Oxford (.venv)** hoặc Python `.venv`, sau đó chạy theo thứ tự:

```text
00_data_source → 01_eda → 02_preprocessing → 03_features
→ 04_model → 05_evaluation → 06_error_analysis
→ 07_conclusion → 08_sanity_checks → 09_v2_experiments
→ 10_v3_robustness
```

Notebook 00–04 chuẩn bị dữ liệu, đặc trưng và mô hình nền; 05–08 đánh giá và phân tích lỗi; 09–10 tái lập các thí nghiệm V2/V3 khi cần. Lần đầu tạo feature/model có thể mất vài phút. Notebook không chạy lại đánh giá test cuối; không xóa khóa hoặc output trong `results/final_test/`.

Có thể kiểm tra nhanh các bất biến của mã bằng:

```bash
python -m pytest -q
```

### 3. Chuẩn bị và chạy UI/UX

Chuẩn bị các artifact UI từ model E6 train-only:

```bash
python scripts/chuan_bi_ui.py
```

Khởi động giao diện:

```bash
streamlit run app.py
```

Mở địa chỉ Streamlit được in trong terminal, thường là `http://localhost:8501`, rồi tải một ảnh hoa bằng nút chọn tệp. UI hiện dùng E6 fit trên train, hiển thị nhãn dự đoán, tối đa ba lớp có phiếu K-NN và các ảnh tham chiếu từ lớp dự đoán. Tỷ lệ phiếu **không phải xác suất đã hiệu chuẩn**. Model test fit train+validation là artifact khác và không được dùng làm model demo. Chỉ nạp pickle và manifest từ nguồn tin cậy.

Khi kiểm tra UI/UX, cần xem: ảnh xem trước có hiển thị đúng, nút tải ảnh hoạt động, nhãn và mã lớp nhất quán, các ảnh tham chiếu thuộc đúng lớp dự đoán, giao diện không báo lỗi khi tải tệp ảnh không hợp lệ, và phần giải thích không gọi tỷ lệ phiếu là xác suất. Ảnh chụp giao diện: **[CẦN NHÓM ĐIỀN]**.

### 4. Kiểm thử và bảo vệ kết quả

Chạy kiểm thử mã bằng `python -m pytest -q`. Tái tạo V5 trong bản sao cách ly vì output V5 hiện đã có. **Không chạy `scripts/run_final_test.py`, `python -m src.pipeline all` hoặc `python -m src.pipeline evaluate` để đánh giá test lại.**

## Cấu trúc và bàn giao

`src/`: xử lý, thí nghiệm và suy luận; `notebooks/`: quy trình; `configs/`: tham số/manifest; `results/`: bảng, dự đoán và hình; `reports/latex/`: báo cáo; `docs/`: kiểm toán và lịch sử; `app.py`: Streamlit. `results/notebooks/*.ipynb` là log chạy lịch sử, không phải mã nguồn. Xem [đóng góp](CONTRIBUTING.md), [công bố AI](AI_USAGE.md) và [kiểm toán dữ liệu](docs/V4_DATASET_AUDIT.md).

## Phân công và đóng góp

Hai thành viên cùng rà soát mã, số liệu và báo cáo. Khối lượng được chia theo hai nửa tương đương; mỗi người chịu trách nhiệm chính một nửa nhưng vẫn kiểm tra chéo phần của người còn lại.

| Thành viên | Phần việc chính (50%) | Phần rà soát chéo |
|---|---|---|
| Nguyễn Duy Tài Anh | Dữ liệu và EDA; tiền xử lý; đặc trưng; notebook 00–03; kiểm tra split và tính tái lập dữ liệu | Rà soát mô hình, đánh giá, UI/UX và các bảng kết quả |
| Phạm Đăng Thao | Mô hình và đánh giá; notebook 04–10; phân tích lỗi; pipeline V5; UI/UX và hoàn thiện báo cáo | Rà soát EDA, đặc trưng, kiểm toán dữ liệu và nội dung công bố |

Mỗi người đóng góp khoảng 50% tổng khối lượng. Các kết luận cuối cùng, số liệu trong báo cáo và việc tuân thủ quy trình test thuộc trách nhiệm chung của cả nhóm.
