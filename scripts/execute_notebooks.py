"""Chạy lần lượt 11 notebook; mặc định chỉ đánh giá validation."""
from pathlib import Path
from time import perf_counter
import json
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
import os
import sys
if Path(sys.prefix).resolve() != (root / '.venv').resolve():
    raise SystemExit('Hãy kích hoạt .venv rồi chạy lại; xem README.md.')
out = root / 'results' / 'notebooks'
out.mkdir(parents=True, exist_ok=True)
os.environ['JUPYTER_PATH'] = str(root / '.venv/share/jupyter') + os.pathsep + os.environ.get('JUPYTER_PATH', '')
if not (root / '.venv/share/jupyter/kernels/hoa-oxford/kernel.json').exists():
    raise SystemExit('Chưa có kernel dự án. Chạy: python scripts/cai_dat_kernel.py')
paths = sorted((root / 'notebooks').glob('[0-9][0-9]_*.ipynb'))
if len(paths) != 11:
    raise RuntimeError(f'Cần đúng 11 notebook 00–10; tìm thấy {len(paths)}')
records = []
for path in paths:
    print(f'Đang chạy {path.name}', flush=True)
    start = perf_counter()
    notebook = nbformat.read(path, as_version=4)
    try:
        NotebookClient(notebook, timeout=1800, kernel_name='hoa-oxford', resources={'metadata': {'path': str(root)}}).execute()
    finally:
        nbformat.write(notebook, out / path.name)
    records.append({'notebook': path.name, 'seconds': round(perf_counter() - start, 2), 'status': 'passed'})
    (out / 'execution.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'Đã chạy xong {path.name}', flush=True)
print('Đã chạy 11 notebook; không đánh giá test chính thức.', flush=True)
