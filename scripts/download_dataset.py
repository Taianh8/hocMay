"""Download and prepare the official Oxford 102 Flowers dataset."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import ssl
import tarfile
from urllib.error import URLError
from urllib.request import Request, urlopen

import certifi


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_DIR = ROOT / "data" / "raw"
BASE_URL = "https://www.robots.ox.ac.uk/~vgg/data/flowers/102"

FILES = {
    "102flowers.tgz": "2d01ecc807db462958cfe3d92f57a8c252b4abd240eb955770201e45f783b246",
    "imagelabels.mat": "4903e94206bac23bf772aadf06451916df56b58fc483a62db32a97b82656651d",
    "setid.mat": "46b8678f91fd95d3c8f4feab80d271a6c834a1dd896fe29fd3e6ad9ce5c8dccd",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(name: str, destination: Path, force: bool) -> None:
    expected = FILES[name]
    if destination.exists() and not force:
        actual = sha256(destination)
        if actual == expected:
            print(f"Đã có và đúng checksum: {destination}")
            return
        print(f"Checksum không khớp, tải lại: {destination}")

    partial = destination.with_name(destination.name + ".part")
    partial.unlink(missing_ok=True)
    request = Request(f"{BASE_URL}/{name}", headers={"User-Agent": "hocMay-dataset-downloader/1.0"})
    try:
        context = ssl.create_default_context(cafile=certifi.where())
        with urlopen(request, timeout=60, context=context) as response, partial.open("wb") as stream:
            total = response.headers.get("Content-Length")
            expected_size = int(total) if total else None
            downloaded = 0
            while block := response.read(1024 * 1024):
                stream.write(block)
                downloaded += len(block)
                if expected_size:
                    print(f"\rĐang tải {name}: {downloaded / expected_size:.0%}", end="", flush=True)
        print()
    except (OSError, URLError) as exc:
        partial.unlink(missing_ok=True)
        raise RuntimeError(f"Không thể tải {name} từ {BASE_URL}: {exc}") from exc

    actual = sha256(partial)
    if actual != expected:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"Checksum SHA-256 của {name} không khớp: nhận {actual}, cần {expected}."
        )
    os.replace(partial, destination)
    print(f"Đã tải: {destination}")


def extract_archive(archive: Path, data_dir: Path) -> None:
    jpg_dir = data_dir / "jpg"
    if jpg_dir.is_dir() and len(list(jpg_dir.glob("image_*.jpg"))) == 8189:
        print(f"Đã giải nén đủ 8.189 ảnh: {jpg_dir}")
        return

    with tarfile.open(archive, "r:gz") as tar:
        root = data_dir.resolve()
        for member in tar.getmembers():
            target = (data_dir / member.name).resolve()
            if os.path.commonpath((root, target)) != str(root):
                raise RuntimeError(f"Archive chứa đường dẫn không an toàn: {member.name}")
        tar.extractall(data_dir)
    print(f"Đã giải nén: {jpg_dir}")


def prepare_dataset(data_dir: Path, force: bool = False) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        download_file(name, data_dir / name, force)
    extract_archive(data_dir / "102flowers.tgz", data_dir)
    print("Dataset Oxford 102 Flowers đã sẵn sàng.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help=f"Thư mục dữ liệu (mặc định: {DEFAULT_DATA_DIR})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Tải lại các tệp dù tệp hiện tại có checksum đúng.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare_dataset(args.data_dir.resolve(), force=args.force)
