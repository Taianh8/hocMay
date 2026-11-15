# Hướng dẫn cài đặt và chạy dự án phân loại ảnh hoa

Dự án dùng bộ Oxford 102 Flowers. Hướng dẫn dưới đây giúp các thành viên tạo môi trường Python, chạy notebook và mở giao diện Streamlit. Các lệnh cần được chạy tại **thư mục gốc dự án** `hocMay`.

## 1. Cài Python và tạo môi trường

Cài **Python 3.11** và mở Terminal trong VS Code. Trên macOS/Linux:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-v4-lock.txt
python scripts/cai_dat_kernel.py
```

Trên Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-v4-lock.txt
python scripts/cai_dat_kernel.py
```

Nếu PowerShell không cho chạy script kích hoạt, có thể dùng Command Prompt và `.venv\Scripts\activate.bat`. Cần kết nối mạng để cài thư viện. Bản lock đã kiểm chứng trên macOS arm64 với Python 3.11.15; máy khác có thể cần wheel tương thích. Không cài vào Python hệ thống hay một môi trường Conda khác.

Sau khi kích hoạt, kiểm tra:

```bash
python -c "import sys, skimage, sklearn, streamlit; print(sys.executable); print('scikit-image:', skimage.__version__)"
```

Đường dẫn Python được in ra phải nằm trong thư mục `.venv` của dự án. Nếu lệnh này báo lỗi, chạy lại lệnh cài thư viện ở trên và đọc thông báo lỗi trước khi tiếp tục.

## 2. Chuẩn bị dữ liệu

Cần ba tệp `102flowers.tgz`, `imagelabels.mat`, `setid.mat` và thư mục `jpg/` trong `data/raw/`. Nếu máy đã có đủ, giữ nguyên và chuyển sang bước 3. Nếu chưa có, từ Terminal macOS/Linux:

```bash
mkdir -p data/raw
curl -fL https://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz -o data/raw/102flowers.tgz
curl -fL https://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat -o data/raw/imagelabels.mat
curl -fL https://www.robots.ox.ac.uk/~vgg/data/flowers/102/setid.mat -o data/raw/setid.mat
python - <<'PY'
import hashlib
from pathlib import Path
checks = {'102flowers.tgz': '52808999861908f626f3c1f4e79d11fa', 'imagelabels.mat': 'e0620be6f572b9609742df49c70aed4d', 'setid.mat': 'a5357ecc9cb78c4bef273ce3793fc85c'}
for name, expected in checks.items():
    with (Path('data/raw') / name).open('rb') as f:
        actual = hashlib.file_digest(f, 'md5').hexdigest()
    if actual != expected:
        raise SystemExit(f'Sai checksum: {name}: {actual}')
    print(name, 'OK')
PY
tar -xzf data/raw/102flowers.tgz -C data/raw
```

Trên Windows có thể tải ba tệp từ [trang Oxford](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/), đặt vào `data/raw/` và giải nén `102flowers.tgz` tại đó. Sau khi giải nén, ví dụ đường dẫn là `data/raw/jpg/image_00001.jpg`. Không đổi tên ảnh, nhãn hoặc split. Nếu máy chủ Oxford chậm, xem [bản sao archive](https://huggingface.co/datasets/XiN0919/FGVC/blob/main/102flowers.tgz) và đối chiếu checksum. MD5 chỉ kiểm tra tính nhất quán, không phải chữ ký xác thực.

## 3. Chọn đúng kernel và chạy notebook trong VS Code

1. Mở **thư mục gốc `hocMay`** bằng VS Code; cài extension **Python** và **Jupyter** nếu chưa có.
2. Mở `notebooks/00_data_source.ipynb`.
3. Nhấn tên kernel ở góc trên bên phải → chọn **Select Another Kernel** → **Python Environments** → chọn **`.venv` / Python 3.11**. Nếu danh sách hiện **Hoa Oxford (.venv)**, có thể chọn kernel đó.
4. Không chọn `machine_learning (Python 3.12)` hoặc Conda khác. Nhấn **Restart Kernel**, sau đó **Run All**.
5. Chạy lần lượt các notebook **00 → 10**. Notebook 05–08 dùng validation; không tự đánh giá tập test. Notebook 09/10 sẽ tạo kết quả cần thiết nếu chưa có.

Nếu VS Code đang mở notebook với thay đổi chưa lưu, hãy sao chép nội dung bạn muốn giữ trước; việc lưu một buffer cũ có thể ghi đè bản notebook mới trên đĩa. Sau khi xử lý thay đổi đó, đóng và mở lại notebook để nạp bản hiện tại. Trong một ô notebook, có thể kiểm tra kernel đang chạy bằng:

```python
import sys
print(sys.executable)
```

Đường dẫn phải trỏ tới `.venv/bin/python` trên macOS/Linux hoặc `.venv\\Scripts\\python.exe` trên Windows. Nếu hiện lỗi **`No module named 'skimage'`**, thường là bạn đang chọn sai kernel: chọn lại `.venv` theo bước 3 và restart. Nếu đã chọn `.venv` mà vẫn lỗi, chạy `python -m pip install -r requirements-v4-lock.txt` trong Terminal đã kích hoạt `.venv`, rồi restart kernel.

Lần đầu xử lý ảnh và huấn luyện có thể mất vài phút. Không chạy các notebook sau khi bước trước đang lỗi. Các file kết quả sẽ được sinh vào `data/processed/` và `results/`; chúng bị Git bỏ qua. Không chạy `python -m src.pipeline all` hoặc `python -m src.pipeline evaluate` khi chưa được cho phép, vì các lệnh đó đánh giá tập test V1.

Nếu muốn chạy tự động thay vì từng notebook:

```bash
python scripts/execute_notebooks.py
python scripts/verify_v4_notebooks.py
```

Lệnh đầu chạy đủ 00–10 bằng kernel `Hoa Oxford (.venv)` đã đăng ký ở bước 1; Jupyter cần mở cổng loopback. Các notebook mặc định không đánh giá test chính thức. Nếu kết quả đã có, notebook 09/10 dùng lại output; để chạy lại, sao lưu rồi xử lý output cũ trước.

## 4. Chuẩn bị model và mở giao diện

Vẫn trong Terminal đã kích hoạt `.venv` và sau khi có dữ liệu:

```bash
python scripts/chuan_bi_ui.py
streamlit run app.py
```

Lần đầu, lệnh chuẩn bị sẽ tạo cache, huấn luyện E1–E6 trên train/validation và tạo `configs/e6_manifest_moi.json`; việc này có thể mất vài phút. Không có dự đoán test mới. Nếu đã có model/manifest cục bộ hợp lệ, lệnh sẽ dùng lại. Nếu thấy thông báo kết quả không đầy đủ hoặc E6 không được chọn, hãy kiểm tra thông báo và `results/` trước khi chạy tiếp; không tự sửa manifest để ép model.

Sau lệnh Streamlit, mở địa chỉ được in trong Terminal, thường là `http://localhost:8501`. Tải JPEG/PNG tối đa 10 MiB và 20 megapixel; ứng dụng trả mã lớp Oxford 1–102, không hiển thị confidence chưa được hiệu chuẩn. Dừng ứng dụng bằng **Ctrl+C**.

Chỉ nạp model pickle và manifest tin cậy. Hash chỉ phát hiện thay đổi ngoài ý muốn, không bảo vệ nếu cả model và manifest đều bị thay. Model và dữ liệu không được lưu trong Git.

## 5. Kiểm thử và cấu trúc

Sau khi hoàn thành bước 4:

```bash
python -m pytest -q
```

Kiểm thử tích hợp cần model E6, cache và output validation. Nếu vừa xóa output, hãy chuẩn bị model trước; không bỏ qua lỗi kiểm thử. Riêng khi chưa tạo model, có thể kiểm tra các phần độc lập bằng:

```bash
python -m pytest -q tests/test_baseline.py tests/test_v2.py tests/test_v3.py tests/test_duong_hoc.py tests/test_validation.py
```

- `notebooks/`: các notebook theo thứ tự 00–10.
- `src/`, `configs/`, `scripts/`, `tests/`: mã, cấu hình và kiểm thử.
- `data/raw/`: dữ liệu chính thức; `data/processed/`: cache sinh ra khi chạy.
- `results/`: chỉ số, hình và model sinh ra khi chạy.
- `app.py`: giao diện Streamlit; `reports/latex/`: nguồn và PDF báo cáo.

E6 được chọn trước đây bằng validation, chưa có kết quả test E6. Kiểm toán đã phát hiện ảnh trùng giữa train và test; test V1 cũng từng được quan sát. Không đổi split hay tự xóa ảnh; xem [kiểm toán dữ liệu](reports/V4_DATASET_AUDIT.md), [bàn giao](HANDOFF.md) và [công bố AI](AI_USAGE.md).
