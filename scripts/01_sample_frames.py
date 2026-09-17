#!/usr/bin/env python3
"""Bước 1 (rút gọn, chế độ offline) — lấy mẫu frame từ UA-DETRAC.

Đọc tập con sequence trong configs/pipeline.yaml, lấy mẫu thưa theo
`sample_every_n`, copy sang data/prototype/images/ với tên chứa sẵn
camera_id + frame_idx, và ghi manifest.jsonl để các bước sau truy ngược nguồn gốc.

Nguyên tắc bám theo docx:
  - không ghi đè / không di chuyển dữ liệu gốc (chỉ copy)
  - giữ camera_id, sequence_id, frame_idx để MLOps chia train/val theo video
  - mọi tham số nằm trong file cấu hình

Dùng:
    .venv/bin/python scripts/01_sample_frames.py
    .venv/bin/python scripts/01_sample_frames.py --force
    .venv/bin/python scripts/01_sample_frames.py --upload      # đẩy lên MinIO
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

import yaml
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[sample] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[sample] LỖI: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_sequence_attrs(xml_path: Path) -> dict:
    """Lấy thuộc tính cảnh (thời tiết, trạng thái camera) + vùng bỏ qua từ XML DETRAC."""
    if not xml_path.is_file():
        return {"weather": None, "camera_state": None, "ignored_regions": []}
    root = ET.parse(xml_path).getroot()
    attr = root.find("sequence_attribute")
    ignored = [
        {
            "left": float(box.get("left")),
            "top": float(box.get("top")),
            "width": float(box.get("width")),
            "height": float(box.get("height")),
        }
        for region in root.iter("ignored_region")
        for box in region.findall("box")
    ]
    return {
        "weather": attr.get("sence_weather") if attr is not None else None,
        "camera_state": attr.get("camera_state") if attr is not None else None,
        "ignored_regions": ignored,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--force", action="store_true", help="xoá lô cũ và lấy mẫu lại")
    ap.add_argument("--upload", action="store_true", help="upload lên bucket MinIO sau khi lấy mẫu")
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    paths, sampling = cfg["paths"], cfg["sampling"]

    img_root = REPO_ROOT / paths["detrac_images"]
    ann_root = REPO_ROOT / paths["detrac_annotations"]
    work_dir = REPO_ROOT / paths["work_dir"]
    out_images = work_dir / "images"

    if not img_root.is_dir():
        die(f"Không thấy thư mục ảnh gốc: {img_root}")

    if out_images.exists() and any(out_images.iterdir()):
        if not args.force:
            die(f"Đã có dữ liệu ở {out_images}. Dùng --force để lấy mẫu lại.")
        shutil.rmtree(out_images)
    out_images.mkdir(parents=True, exist_ok=True)

    batch_id = datetime.now(timezone.utc).strftime("batch_%Y%m%d_%H%M%S")
    every = int(sampling["sample_every_n"])
    limit = int(sampling["max_per_camera"])
    src_fps = float(sampling["source_fps"])

    manifest_path = work_dir / "manifest.jsonl"
    scenes_path = work_dir / "scenes.json"
    records, scenes = [], {}

    for cam in cfg["cameras"]:
        camera_id, seq = cam["camera_id"], cam["sequence"]
        seq_dir = img_root / seq
        if not seq_dir.is_dir():
            die(f"Không thấy sequence {seq} tại {seq_dir}")

        frames = sorted(seq_dir.glob("img*.jpg"))
        if not frames:
            die(f"Sequence {seq} không có ảnh img*.jpg")

        attrs = read_sequence_attrs(ann_root / f"{seq}.xml")
        scenes[seq] = {"camera_id": camera_id, "total_frames": len(frames), **attrs}

        picked = frames[:: every][:limit]
        for src in picked:
            # img00123.jpg -> frame_idx 123 (DETRAC đánh số từ 1, khớp thuộc tính num trong XML)
            frame_idx = int(src.stem.replace("img", ""))
            dst_name = f"{camera_id}_{seq}_{frame_idx:06d}.jpg"
            dst = out_images / dst_name
            shutil.copy2(src, dst)  # copy, không move — giữ nguyên dữ liệu gốc

            with Image.open(dst) as im:
                width, height = im.size

            records.append(
                {
                    "image_id": dst_name,
                    "batch_id": batch_id,
                    "camera_id": camera_id,
                    "sequence_id": seq,
                    "frame_idx": frame_idx,
                    "timestamp_in_video_sec": round((frame_idx - 1) / src_fps, 3),
                    "weather": attrs["weather"],
                    "camera_state": attrs["camera_state"],
                    "source_path": str(src.relative_to(REPO_ROOT)),
                    "width": width,
                    "height": height,
                    "sha256": sha256_of(dst),
                    "sampled_at": datetime.now(timezone.utc).isoformat(),
                    "status": "sampled",
                }
            )

        log(f"{camera_id} / {seq} ({attrs['weather']}): {len(picked)}/{len(frames)} frame")

    with manifest_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    scenes_path.write_text(json.dumps(scenes, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Tổng: {len(records)} ảnh -> {out_images}")
    log(f"Manifest: {manifest_path}")
    log(f"Thuộc tính cảnh + ignored_region: {scenes_path}")

    if args.upload:
        upload_to_minio(cfg, out_images, records, batch_id)


def upload_to_minio(cfg: dict, img_dir: Path, records: list, batch_id: str) -> None:
    """Đẩy lô ảnh lên MinIO. Bucket tự tạo nếu chưa có."""
    try:
        from minio import Minio
    except ImportError:
        die("Chưa cài thư viện minio: pip install minio")

    import os

    env_path = REPO_ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, _, v = line.partition("=")
                os.environ.setdefault(k.strip(), v.strip().strip("'\""))

    st = cfg["storage"]
    user = os.environ.get("MINIO_ROOT_USER")
    pwd = os.environ.get("MINIO_ROOT_PASSWORD")
    if not user or not pwd:
        die("Thiếu MINIO_ROOT_USER / MINIO_ROOT_PASSWORD trong .env")

    client = Minio(st["endpoint"], access_key=user, secret_key=pwd, secure=bool(st["secure"]))
    bucket = st["bucket_raw_frames"]
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        log(f"Đã tạo bucket '{bucket}'")

    for i, r in enumerate(records, 1):
        client.fput_object(bucket, f"{batch_id}/{r['image_id']}", str(img_dir / r["image_id"]),
                           content_type="image/jpeg")
        if i % 25 == 0 or i == len(records):
            log(f"upload {i}/{len(records)}")
    log(f"Xong: s3://{bucket}/{batch_id}/")


if __name__ == "__main__":
    main()
