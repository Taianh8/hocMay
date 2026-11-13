"""Execute V1 notebooks in isolated output folders and V2 report notebook."""
import argparse
import json
from pathlib import Path
from time import perf_counter
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/v2'


def execute(path: Path, isolate_v1: bool) -> dict:
    """Run one source notebook; route V1 output away from original artifacts."""
    nb = nbformat.read(path, as_version=4)
    if isolate_v1:
        patch = "from pathlib import Path\nfrom src import pipeline as p\np.FIG = root / 'results/v2/notebook_v1_check/figures'\np.MET = root / 'results/v2/notebook_v1_check/metrics'\np.MOD = root / 'results/v2/notebook_v1_check/models'\n"
        nb.cells.insert(2, nbformat.v4.new_code_cell(patch))
    start = perf_counter()
    try:
        NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        dest = OUT / 'notebooks' / path.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        nbformat.write(nb, dest)
    return {'notebook': path.name, 'seconds': round(perf_counter()-start, 2), 'status': 'passed', 'isolated_v1_outputs': isolate_v1}


def main():
    parser = argparse.ArgumentParser(); parser.add_argument('group', choices=['v1', 'v2']); group = parser.parse_args().group
    paths = sorted((ROOT / 'notebooks').glob('0[0-8]_*.ipynb')) if group == 'v1' else [ROOT / 'notebooks/09_v2_experiments.ipynb']
    if len(paths) != (9 if group == 'v1' else 1): raise ValueError('Unexpected notebook count')
    records = []
    for path in paths:
        print('Executing', path.name, flush=True)
        records.append(execute(path, group == 'v1'))
        print(records[-1], flush=True)
        (OUT / 'notebooks' / f'{group}_execution.json').write_text(json.dumps(records, indent=2) + '\n')

if __name__ == '__main__': main()
