#!/usr/bin/env python3
"""Bước 0 — Nạp dữ liệu mới và lấy mẫu frame.

Đọc danh sách nguồn trong configs/pipeline.yaml (`sources`), lấy mẫu thưa theo
`sample_every_n`, ghi ảnh sang <work_dir>/images/ với tên chứa sẵn camera_id +
frame_idx, và ghi manifest.jsonl để các bước sau truy ngược nguồn gốc.

Hai loại nguồn:
    kind: video   -> tách frame bằng OpenCV, fps đọc thật từ file
    kind: images  -> thư mục ảnh đã tách sẵn, fps lấy từ sampling.source_fps

Nguyên tắc:
  - không ghi đè / không di chuyển dữ liệu gốc
  - giữ camera_id, sequence_id, frame_idx để MLOps chia train/val theo video
  - mọi tham số nằm trong file cấu hình

Dùng:
    .venv/bin/python scripts/00_ingest.py
    .venv/bin/python scripts/00_ingest.py --force
    .venv/bin/python scripts/00_ingest.py --camera cam05     # chỉ một nguồn
    .venv/bin/python scripts/00_ingest.py --upload           # đẩy lên MinIO
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

from _common import REPO_ROOT, add_io_args, load_dotenv, load_yaml, logger, sha256_of

log, die = logger("ingest")

TRAILING_DIGITS = re.compile(r"(\d+)$")


def frame_index_from_name(stem: str, position: int) -> int:
    """Số frame suy từ tên file, vd img00123 -> 123. Không có số thì dùng thứ tự trong thư mục.

    Giữ đúng cách đánh số của bộ dữ liệu gốc, để lấy mẫu lại ra cùng tên ảnh.
    """
    m = TRAILING_DIGITS.search(stem)
    return int(m.group(1)) if m else position


def sample_images(src_dir: Path, glob: str, every: int, limit: int):
    """Lấy mẫu từ thư mục ảnh. Trả về list (frame_idx, đường dẫn nguồn, hàm ghi ảnh)."""
    frames = sorted(src_dir.glob(glob))
    if not frames:
        die(f"thư mục {src_dir} không có file nào khớp '{glob}'")
    picked = frames[::every]
    if limit:
        picked = picked[:limit]
    out = []
    for pos, src in enumerate(picked, 1):
        idx = frame_index_from_name(src.stem, (pos - 1) * every + 1)
        out.append((idx, src, lambda dst, s=src: shutil.copy2(s, dst)))
    return out, len(frames), None


def sample_video(path: Path, every: int, limit: int):
    """Lấy mẫu từ file video. Trả về (list mẫu, tổng frame, fps thật)."""
    try:
        import cv2
    except ImportError:
        die("nguồn kind=video cần opencv: .venv/bin/pip install opencv-python")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        die(f"không mở được video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    out, idx = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        idx += 1
        if (idx - 1) % every:
            continue
        # cv2 trả BGR; đảo sang RGB trước khi ghi để màu không bị lộn
        out.append((idx, path, lambda dst, f=frame[:, :, ::-1].copy(): Image.fromarray(f).save(dst, quality=95)))
        if limit and len(out) >= limit:
            break
    cap.release()
    return out, total or idx, (fps if fps > 0 else None)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(ap, "config")
    ap.add_argument("--camera", help="chỉ nạp một camera, vd cam05")
    ap.add_argument("--force", action="store_true", help="xoá lô cũ và lấy mẫu lại")
    ap.add_argument("--upload", action="store_true", help="upload lên bucket MinIO sau khi lấy mẫu")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    sampling = cfg["sampling"]
    work_dir = REPO_ROOT / cfg["paths"]["work_dir"]
    out_images = work_dir / "images"

    sources = cfg.get("sources") or []
    if args.camera:
        sources = [s for s in sources if s["camera_id"] == args.camera]
    if not sources:
        die("không có nguồn nào trong configs/pipeline.yaml"
            + (f" cho camera {args.camera}" if args.camera else ""))

    if out_images.exists() and any(out_images.iterdir()):
        if not args.force:
            die(f"đã có dữ liệu ở {out_images}. Dùng --force để lấy mẫu lại.")
        shutil.rmtree(out_images)
    out_images.mkdir(parents=True, exist_ok=True)

    batch_id = datetime.now(timezone.utc).strftime("batch_%Y%m%d_%H%M%S")
    every = int(sampling["sample_every_n"])
    limit = int(sampling.get("max_per_camera") or 0)
    default_fps = float(sampling["source_fps"])
    image_glob = sampling.get("image_glob", "*.jpg")

    records = []
    for src_cfg in sources:
        camera_id = src_cfg["camera_id"]
        kind = src_cfg.get("kind", "images")
        src_path = REPO_ROOT / src_cfg["path"]
        sequence = src_cfg.get("sequence") or src_path.stem
        scene = src_cfg.get("scene") or {}

        if not src_path.exists():
            die(f"{camera_id}: không thấy nguồn {src_path}")

        if kind == "video":
            picked, total, real_fps = sample_video(src_path, every, limit)
            fps = real_fps or default_fps
        elif kind == "images":
            picked, total, _ = sample_images(src_path, image_glob, every, limit)
            fps = default_fps
        else:
            die(f"{camera_id}: kind='{kind}' không hợp lệ (chỉ nhận 'video' hoặc 'images')")

        for frame_idx, origin, write in picked:
            dst_name = f"{camera_id}_{sequence}_{frame_idx:06d}.jpg"
            dst = out_images / dst_name
            write(dst)  # copy (ảnh) hoặc ghi frame đã giải mã (video) — không đụng nguồn

            with Image.open(dst) as im:
                width, height = im.size

            records.append({
                "image_id": dst_name,
                "batch_id": batch_id,
                "camera_id": camera_id,
                "sequence_id": sequence,
                "frame_idx": frame_idx,
                "timestamp_in_video_sec": round((frame_idx - 1) / fps, 3),
                # metadata tự khai trong pipeline.yaml; hai khoá dưới luôn có mặt để bước
                # phát hành thống kê được, kể cả khi nguồn không khai báo gì.
                "weather": scene.get("weather"),
                "camera_state": scene.get("camera_state"),
                **{k: v for k, v in scene.items() if k not in ("weather", "camera_state")},
                "source_kind": kind,
                "source_path": str(origin.relative_to(REPO_ROOT)),
                "width": width,
                "height": height,
                "sha256": sha256_of(dst),
                "sampled_at": datetime.now(timezone.utc).isoformat(),
                "status": "sampled",
            })

        log(f"{camera_id} / {sequence} [{kind}]: {len(picked)}/{total} frame"
            + (f" ({scene['weather']})" if scene.get("weather") else ""))

    manifest_path = work_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    log(f"Tổng: {len(records)} ảnh -> {out_images}")
    log(f"Manifest: {manifest_path}")

    if args.upload:
        upload_to_minio(cfg, out_images, records, batch_id)


def upload_to_minio(cfg: dict, img_dir: Path, records: list, batch_id: str) -> None:
    """Đẩy lô ảnh lên MinIO. Bucket tự tạo nếu chưa có."""
    try:
        from minio import Minio
    except ImportError:
        die("chưa cài thư viện minio: pip install minio")

    load_dotenv()

    st = cfg["storage"]
    user = os.environ.get("MINIO_ROOT_USER")
    pwd = os.environ.get("MINIO_ROOT_PASSWORD")
    if not user or not pwd:
        die("thiếu MINIO_ROOT_USER / MINIO_ROOT_PASSWORD trong .env")

    client = Minio(st["endpoint"], access_key=user, secret_key=pwd, secure=bool(st["secure"]))
    bucket = st["bucket_raw_frames"]
    if not client.bucket_exists(bucket):
        client.make_bucket(bucket)
        log(f"đã tạo bucket '{bucket}'")

    for i, r in enumerate(records, 1):
        client.fput_object(bucket, f"{batch_id}/{r['image_id']}", str(img_dir / r["image_id"]),
                           content_type="image/jpeg")
        if i % 25 == 0 or i == len(records):
            log(f"upload {i}/{len(records)}")
    log(f"Xong: s3://{bucket}/{batch_id}/")


if __name__ == "__main__":
    main()
