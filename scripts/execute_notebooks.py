"""Execute the nine notebooks in order, retaining copies under ignored results/."""
from pathlib import Path
from time import perf_counter
import json
import nbformat
from nbclient import NotebookClient

root = Path(__file__).resolve().parents[1]
out = root / 'results' / 'notebooks'
out.mkdir(parents=True, exist_ok=True)
records = []
for path in sorted((root / 'notebooks').glob('[0-8][0-9]_*.ipynb')):
    print(f'Executing {path.name}', flush=True)
    start = perf_counter()
    notebook = nbformat.read(path, as_version=4)
    try:
        NotebookClient(notebook, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(root)}}).execute()
    except Exception:
        nbformat.write(notebook, out / path.name)
        raise
    nbformat.write(notebook, out / path.name)
    records.append({'notebook': path.name, 'seconds': round(perf_counter() - start, 2), 'status': 'passed'})
    (out / 'execution.json').write_text(json.dumps(records, indent=2) + '\n')
    print(records[-1], flush=True)
if len(records) != 9:
    raise RuntimeError(f'Expected nine notebooks, found {len(records)}')
