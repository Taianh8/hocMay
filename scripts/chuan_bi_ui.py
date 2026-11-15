"""Tạo model E6 và manifest cục bộ từ train/validation, không dùng test."""
from pathlib import Path
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if Path(sys.prefix).resolve() != (ROOT/'.venv').resolve():
    raise SystemExit('Hãy kích hoạt .venv của dự án trước. Xem README.md.')
sys.path.insert(0, str(ROOT))
from src import pipeline as p
from src.inference import load_frozen_model
from src.pipeline_v2 import run as run_v2
from src.utils import config


def main():
    manifest = ROOT/'configs/e6_manifest_moi.json'
    if manifest.exists():
        load_frozen_model(manifest)
        print('Model E6 cục bộ đã sẵn sàng; không huấn luyện lại.')
        return
    required = [ROOT/'data/raw/102flowers.tgz', ROOT/'data/raw/imagelabels.mat', ROOT/'data/raw/setid.mat', ROOT/'data/raw/jpg']
    missing = [str(path.relative_to(ROOT)) for path in required if not path.exists()]
    if missing:
        raise SystemExit('Thiếu dữ liệu: ' + ', '.join(missing) + '. Hãy tải dữ liệu theo README.md.')
    baseline = [p.MET/'selection.json', p.MET/'validation_search.json', p.MOD/'selected_model.pkl']
    if all(x.is_file() for x in baseline):
        p.selected(config())
        print('Dùng model V1 đã có.')
    elif any(x.exists() for x in baseline):
        raise SystemExit('Kết quả V1 chưa đầy đủ. Sao lưu hoặc kiểm tra results/metrics và results/models trước khi chạy lại.')
    else:
        print('Đang kiểm tra dữ liệu và huấn luyện E1–E3 trên train/validation...', flush=True)
        df = p.data(); cfg = config(); p.feature_cache(df, cfg); p.train(df, cfg)
    comparison = ROOT/'results/v2/metrics/comparison.json'
    model = ROOT/'results/v2/models/E6.pkl'
    if comparison.is_file() and model.is_file():
        print('Dùng kết quả V2 đã có.')
    elif comparison.exists() or model.exists():
        raise SystemExit('Kết quả V2 chưa đầy đủ. Sao lưu hoặc kiểm tra results/v2 trước khi chạy lại.')
    else:
        print('Đang huấn luyện E4–E6 trên train/validation...', flush=True)
        run_v2()
    data = json.loads(comparison.read_text())
    if data['selected_experiment'] != 'E6':
        raise SystemExit('Lần chạy này không chọn E6 trên validation; không tự ý tạo manifest E6. Kiểm tra results/v2/metrics/comparison.json.')
    subprocess.run([sys.executable, '-m', 'scripts.freeze_e6', '--output', str(manifest)], cwd=ROOT, check=True)
    load_frozen_model(manifest)
    print('Đã chuẩn bị model E6. Chạy: streamlit run app.py')
    print('Không có dự đoán mới trên tập test.')

if __name__ == '__main__': main()
