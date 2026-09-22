#!/usr/bin/env python3
"""Tải bộ dữ liệu UA-DETRAC từ Kaggle về data/raw/ua-detrac.

Dùng (trong venv của repo):
    .venv/bin/python scripts/00_download_detrac.py
    .venv/bin/python scripts/00_download_detrac.py --force
    .venv/bin/python scripts/00_download_detrac.py --slug dtrnngc/ua-detrac-dataset

Credential (kaggle >= 2.2): một trong các cách sau
    .venv/bin/kaggle auth login          # OAuth, khuyến nghị
    KAGGLE_API_TOKEN=... (đặt trong .env ở gốc repo)
    ~/.kaggle/access_token  hoặc  ~/.kaggle/kaggle.json (legacy)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import zipfile
from pathlib import Path

from _common import REPO_ROOT, load_dotenv

DEFAULT_SLUG = "bratjay/ua-detrac-orig"
DEST_DIR = REPO_ROOT / "data" / "raw" / "ua-detrac"
ZIP_DIR = REPO_ROOT / "data" / "downloads"


def log(msg: str) -> None:
    print(f"[detrac] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[detrac] LỖI: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def has_credentials() -> bool:
    if os.environ.get("KAGGLE_API_TOKEN") or os.environ.get("KAGGLE_KEY"):
        return True
    cfg = Path(os.environ.get("KAGGLE_CONFIG_DIR", Path.home() / ".kaggle"))
    return (cfg / "access_token").is_file() or (cfg / "kaggle.json").is_file()


def authenticate():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError:
        die("Chưa cài kaggle. Chạy: .venv/bin/pip install kaggle")
    api = KaggleApi()
    try:
        api.authenticate()
    except Exception as exc:  # noqa: BLE001 - kaggle ném nhiều loại lỗi khác nhau
        die(
            f"Xác thực Kaggle thất bại: {exc}\n"
            "  Cách 1 (khuyến nghị): .venv/bin/kaggle auth login\n"
            "  Cách 2: lấy token ở https://www.kaggle.com/settings/api rồi thêm\n"
            "          KAGGLE_API_TOKEN=... vào file .env ở gốc repo"
        )
    return api


def summarize(dest: Path) -> None:
    total = sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())
    seqs = [d for d in dest.rglob("MVI_*") if d.is_dir()]
    xmls = list(dest.rglob("*.xml"))
    images = [f for f in dest.rglob("*") if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    log(f"Dung lượng: {total / 1024**3:.2f} GB")
    log(f"Sequence MVI_*: {len(seqs)} | file XML: {len(xmls)} | ảnh: {len(images)}")
    log("Cấu trúc 2 cấp đầu:")
    for p in sorted(dest.glob("*")):
        kind = "d" if p.is_dir() else "f"
        print(f"    {kind} {p.relative_to(dest)}")
        if p.is_dir():
            for child in sorted(p.glob("*"))[:5]:
                print(f"        {'d' if child.is_dir() else 'f'} {child.relative_to(dest)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--slug", default=os.environ.get("DETRAC_SLUG", DEFAULT_SLUG),
                        help=f"slug dataset trên Kaggle (mặc định: {DEFAULT_SLUG})")
    parser.add_argument("--dest", type=Path, default=DEST_DIR, help="thư mục giải nén")
    parser.add_argument("--force", action="store_true", help="tải lại dù đã có dữ liệu")
    parser.add_argument("--delete-zip", action="store_true",
                        help="xoá file zip sau khi giải nén (mặc định giữ lại để khỏi tải lại)")
    args = parser.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    log(f"Đã nạp biến môi trường từ .env")

    if args.dest.is_dir() and any(args.dest.iterdir()) and not args.force:
        log(f"Đã có dữ liệu tại {args.dest} — bỏ qua (dùng --force để tải lại).")
        summarize(args.dest)
        return

    if not has_credentials():
        # Không chặn: `kaggle auth login` (OAuth) lưu credential ở chỗ ta không dò được.
        log("Không thấy token trong .env hay ~/.kaggle — thử credential OAuth của kaggle CLI...")

    api = authenticate()

    ZIP_DIR.mkdir(parents=True, exist_ok=True)
    args.dest.mkdir(parents=True, exist_ok=True)

    free_gb = shutil.disk_usage(ZIP_DIR).free / 1024**3
    if free_gb < 20:
        log(f"CẢNH BÁO: chỉ còn {free_gb:.1f} GB trống, UA-DETRAC cần ~20 GB (zip + giải nén).")

    log(f"Tải '{args.slug}' về {ZIP_DIR} ...")
    try:
        api.dataset_download_files(args.slug, path=str(ZIP_DIR), force=args.force, quiet=False, unzip=False)
    except Exception as exc:  # noqa: BLE001
        die(f"Tải thất bại: {exc}")

    zips = sorted(ZIP_DIR.glob("*.zip"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not zips:
        die("Không tìm thấy file zip sau khi tải.")
    zip_path = zips[0]

    log(f"Giải nén {zip_path.name} -> {args.dest}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(args.dest)

    if args.delete_zip:
        zip_path.unlink()
    else:
        log(f"Giữ zip gốc tại: {zip_path}")

    summarize(args.dest)
    log("Xong.")


if __name__ == "__main__":
    main()
