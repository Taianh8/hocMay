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

1. Tải [Oxford 102 Flowers](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) và đặt `102flowers.tgz`, `imagelabels.mat`, `setid.mat` và thư mục `jpg/` vào `data/raw/`. Giữ split chính thức 1.020/1.020/6.149.
2. Cài Python 3.11, tạo và kích hoạt môi trường:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v4-lock.txt
python scripts/cai_dat_kernel.py
```

Trên Windows dùng `py -3.11 -m venv .venv` rồi `.venv\Scripts\activate`. Lock đã kiểm chứng trên macOS arm64; máy khác có thể cần wheel phù hợp.
3. Trong VS Code mở `notebooks/00_data_source.ipynb`, chọn kernel **Hoa Oxford (.venv)** hoặc Python `.venv`, sau đó chạy lần lượt 00–10. Notebook 05–08 chỉ đánh giá validation. Lần đầu tạo feature/model có thể mất vài phút.
4. Chuẩn bị và chạy UI:

```bash
python scripts/chuan_bi_ui.py
streamlit run app.py
```

UI hiện dùng E6 fit trên train, hiển thị tối đa ba lớp có phiếu K-NN; tỷ lệ phiếu **không phải xác suất đã hiệu chuẩn**. Model test fit train+validation là artifact khác. Chỉ nạp pickle và manifest tin cậy. Ảnh chụp giao diện: **[CẦN NHÓM BỔ SUNG]**.

5. Kiểm thử: `python -m pytest -q`. Tái tạo V5 trong bản sao cách ly vì output V5 hiện đã có. **Không chạy `scripts/run_final_test.py`, `python -m src.pipeline all` hoặc `python -m src.pipeline evaluate` để đánh giá test lại.**

## Cấu trúc và bàn giao

`src/`: xử lý, thí nghiệm và suy luận; `notebooks/`: quy trình; `configs/`: tham số/manifest; `results/`: bảng, dự đoán và hình; `reports/latex/`: báo cáo; `docs/`: kiểm toán và lịch sử; `app.py`: Streamlit. `results/notebooks/*.ipynb` là log chạy lịch sử, không phải mã nguồn. Xem [đóng góp](CONTRIBUTING.md), [công bố AI](AI_USAGE.md) và [kiểm toán dữ liệu](docs/V4_DATASET_AUDIT.md).

## Phân công và đóng góp

| Thành viên | Phần việc | Notebook/module |
|---|---|---|
| Nguyễn Duy Tài Anh | [CẦN NHÓM ĐIỀN] | [CẦN NHÓM ĐIỀN] |
| Phạm Đăng Thao | [CẦN NHÓM ĐIỀN] | [CẦN NHÓM ĐIỀN] |
