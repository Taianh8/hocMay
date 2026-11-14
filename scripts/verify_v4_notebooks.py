"""Execute read-only V2/V3 report notebooks; validate prior V1 execution copies.

V1 notebook 05 re-evaluates test, so it is intentionally not re-executed here.
"""
import json
from pathlib import Path
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
out = ROOT/'results/v4/notebooks'
out.mkdir(parents=True, exist_ok=True)
for path in sorted((ROOT/'notebooks').glob('0[0-8]_*.ipynb')):
    saved = ROOT/'results/v2/notebooks'/path.name
    notebook = nbformat.read(saved, as_version=4)
    codes = [c for c in notebook.cells if c.cell_type=='code']
    if not all(c.execution_count is not None for c in codes) or any(o.output_type=='error' for c in codes for o in c.outputs):
        raise ValueError(f'Prior V1 notebook execution is incomplete: {saved}')
    print(f'{path.name}: prior isolated V1 execution verified')
for name in ('09_v2_experiments.ipynb', '10_v3_robustness.ipynb'):
    notebook = nbformat.read(ROOT/'notebooks'/name, as_version=4)
    try:
        NotebookClient(notebook, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
    finally:
        nbformat.write(notebook, out/name)
    print(f'{name}: executed')
(out/'verification.json').write_text(json.dumps({'prior_v1_notebooks_checked':9, 'executed_read_only_notebooks':['09_v2_experiments.ipynb','10_v3_robustness.ipynb'], 'test_re_evaluated':False},indent=2)+'\n')
