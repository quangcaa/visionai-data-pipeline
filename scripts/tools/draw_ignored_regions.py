#!/usr/bin/env python3
"""Vẽ vùng bỏ qua (ignored_region) của từng camera lên một frame mẫu.

Đánh giá viên không thấy các vùng này trong CVAT, nên guideline kèm ảnh minh hoạ.
Ra: docs/ignored_regions/<camera_id>_<sequence>.jpg

Dùng:
    .venv/bin/python scripts/tools/draw_ignored_regions.py
"""

from __future__ import annotations

import json

from PIL import Image, ImageDraw

from _common import REPO_ROOT, load_yaml


def main() -> None:
    cfg = load_yaml(REPO_ROOT / "configs" / "pipeline.yaml")
    work = REPO_ROOT / cfg["paths"]["work_dir"]
    scenes = json.loads((work / "scenes.json").read_text(encoding="utf-8"))
    out_dir = REPO_ROOT / "docs" / "ignored_regions"
    out_dir.mkdir(parents=True, exist_ok=True)

    for seq, info in scenes.items():
        cam = info["camera_id"]
        frame = next(iter(sorted((work / "images").glob(f"{cam}_{seq}_*.jpg"))), None)
        if frame is None:
            continue
        im = Image.open(frame).convert("RGB")
        overlay = Image.new("RGBA", im.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for r in info["ignored_regions"]:
            box = [r["left"], r["top"], r["left"] + r["width"], r["top"] + r["height"]]
            draw.rectangle(box, fill=(255, 0, 0, 90), outline=(255, 0, 0, 255), width=3)
        out = Image.alpha_composite(im.convert("RGBA"), overlay).convert("RGB")
        ImageDraw.Draw(out).text((10, 10), f"{cam} {seq} ({info['weather']}) - vung do: KHONG gan nhan",
                                 fill=(255, 255, 0))
        path = out_dir / f"{cam}_{seq}.jpg"
        out.save(path, quality=85)
        print(f"{path.relative_to(REPO_ROOT)}  ({len(info['ignored_regions'])} vùng)")


if __name__ == "__main__":
    main()
