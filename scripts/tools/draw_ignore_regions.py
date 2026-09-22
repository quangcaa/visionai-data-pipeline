#!/usr/bin/env python3
"""Vẽ vùng bỏ qua của từng camera lên một frame mẫu, để kiểm tra bằng mắt.

Vùng bỏ qua khai báo tay trong configs/ignore_regions.json (toạ độ chuẩn hoá 0..1).
Đánh giá viên không thấy các vùng này trong CVAT, nên guideline kèm ảnh minh hoạ.

Chạy sau mỗi lần sửa configs/ignore_regions.json để xem mình vẽ có đúng chỗ không.

Ra: docs/ignore_regions/<camera_id>.jpg

Dùng:
    .venv/bin/python scripts/tools/draw_ignore_regions.py
    .venv/bin/python scripts/tools/draw_ignore_regions.py --camera cam03
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageDraw

# script nằm trong scripts/tools/ nên phải tự thêm scripts/ vào đường dẫn import
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from _common import (REPO_ROOT, add_io_args, ignore_regions_xyxy,  # noqa: E402
                     load_ignore_regions, load_yaml, logger, read_manifest)

log, die = logger("regions")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(ap, "config")
    ap.add_argument("--camera", help="chỉ vẽ một camera")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "docs" / "ignore_regions")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    work = REPO_ROOT / cfg["paths"]["work_dir"]
    ignore = load_ignore_regions(cfg)

    try:
        records = read_manifest(work)
    except FileNotFoundError:
        die(f"chưa có manifest ở {work}. Chạy scripts/00_ingest.py trước.")

    # một ảnh mẫu cho mỗi camera
    sample: dict[str, dict] = {}
    for r in records:
        sample.setdefault(r["camera_id"], r)

    cameras = [args.camera] if args.camera else sorted(sample)
    args.out.mkdir(parents=True, exist_ok=True)

    for camera_id in cameras:
        rec = sample.get(camera_id)
        if rec is None:
            log(f"{camera_id}: không có ảnh nào trong lô hiện tại — bỏ qua")
            continue

        frame = work / "images" / rec["image_id"]
        im = Image.open(frame).convert("RGBA")
        boxes = ignore_regions_xyxy(ignore.get(camera_id), *im.size)

        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for x1, y1, x2, y2 in boxes:
            draw.rectangle([x1, y1, x2, y2], fill=(255, 0, 0, 90), outline=(255, 0, 0, 255), width=3)

        out_im = Image.alpha_composite(im, overlay).convert("RGB")
        caption = f"{camera_id} ({rec['sequence_id']}) - vung do: KHONG gan nhan"
        ImageDraw.Draw(out_im).text((10, 10), caption, fill=(255, 255, 0))

        path = args.out / f"{camera_id}.jpg"
        out_im.save(path, quality=85)
        log(f"{path.relative_to(REPO_ROOT)}  ({len(boxes)} vùng)")

    if not ignore:
        log("Chưa khai báo vùng bỏ qua nào trong configs/ignore_regions.json")


if __name__ == "__main__":
    main()
