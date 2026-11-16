"""Đăng ký kernel Jupyter trong chính .venv của dự án."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
expected = (ROOT / '.venv').resolve()
if Path(sys.prefix).resolve() != expected:
    raise SystemExit('Hãy kích hoạt .venv của dự án trước: source .venv/bin/activate (macOS/Linux) hoặc .venv\\Scripts\\activate (Windows).')
subprocess.run([sys.executable, '-m', 'ipykernel', 'install', '--sys-prefix', '--name', 'hoa-oxford', '--display-name', 'Hoa Oxford (.venv)'], check=True)
print('Đã tạo kernel Hoa Oxford (.venv). Trong VS Code, chọn kernel này hoặc Python .venv.')
