"""Kiểm tra 11 bản notebook được thực thi bởi execute_notebooks.py."""
import json
from pathlib import Path
import nbformat

ROOT = Path(__file__).resolve().parents[1]
out = ROOT / 'results/notebooks'
log = out / 'execution.json'
if not log.is_file():
    raise SystemExit('Chưa có output notebook. Hãy chạy: python scripts/execute_notebooks.py')
records = json.loads(log.read_text())
expected = sorted(p.name for p in (ROOT/'notebooks').glob('[0-9][0-9]_*.ipynb'))
if len(expected) != 11 or [r['notebook'] for r in records] != expected or not all(r['status']=='passed' for r in records):
    raise ValueError('Danh sách hoặc trạng thái thực thi notebook không đầy đủ')
for name in expected:
    path = out/name
    notebook = nbformat.read(path, as_version=4)
    codes = [c for c in notebook.cells if c.cell_type == 'code']
    if not all(c.execution_count is not None for c in codes) or any(o.output_type=='error' for c in codes for o in c.outputs):
        raise ValueError(f'Notebook có lỗi hoặc chưa chạy đầy đủ: {name}')
    print(f'{name}: đã thực thi')
print('Đã kiểm tra 11 notebook; output mặc định chỉ đánh giá validation.')
