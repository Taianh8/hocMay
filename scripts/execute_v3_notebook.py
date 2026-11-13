"""Execute the V3 results notebook without running any model or test evaluation."""
import json
from pathlib import Path
from time import perf_counter
import nbformat
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
source = ROOT/'notebooks/10_v3_robustness.ipynb'
out = ROOT/'results/v3/notebooks'
out.mkdir(parents=True, exist_ok=True)
nb = nbformat.read(source, as_version=4)
start = perf_counter()
try:
    NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(ROOT)}}).execute()
finally:
    nbformat.write(nb, out/source.name)
(out/'execution.json').write_text(json.dumps({'notebook': source.name, 'status': 'passed', 'seconds': perf_counter()-start}, indent=2)+'\n')
print('V3 notebook passed')
