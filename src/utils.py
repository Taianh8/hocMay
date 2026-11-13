"""Shared paths, configuration and artifact helpers."""
from pathlib import Path
import hashlib
import json
import yaml

ROOT = Path(__file__).resolve().parents[1]

def config() -> dict:
    """Load experiment settings from the project root."""
    with (ROOT / 'configs/experiments.yaml').open() as f:
        return yaml.safe_load(f)

def digest(path: Path) -> str:
    """Compute a file's SHA-256 digest."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def save_json(path: Path, value: object) -> None:
    """Write a readable JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
