#!/usr/bin/env python3
"""Chuyển nhãn gốc UA-DETRAC (XML) sang định dạng YOLO cho đúng các frame đã lấy mẫu.

GT này là THƯỚC ĐO, không phải nhãn của dataset phát hành: dùng để chấm điểm
nhãn sơ bộ do YOLO26 sinh (scripts/04_eval_prelabel.py).

Định dạng ra: data/prototype/gt/<image_id>.txt, mỗi dòng
    <class_id> <cx> <cy> <w> <h>     (toạ độ chuẩn hoá 0..1)

Dùng:
    .venv/bin/python scripts/02_xml_to_yolo_gt.py
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path

from _common import REPO_ROOT, load_label_spec, load_yaml


def log(msg: str) -> None:
    print(f"[gt] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[gt] LỖI: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def parse_sequence_targets(xml_path: Path) -> dict[int, list[dict]]:
    """Đọc XML DETRAC -> {frame_num: [target, ...]}. frame_num đánh số từ 1, khớp img00001.jpg."""
    root = ET.parse(xml_path).getroot()
    out: dict[int, list[dict]] = {}
    for frame in root.findall("frame"):
        num = int(frame.get("num"))
        targets = []
        for t in frame.iter("target"):
            box = t.find("box")
            attr = t.find("attribute")
            targets.append(
                {
                    "target_id": t.get("id"),
                    "vehicle_type": attr.get("vehicle_type") if attr is not None else None,
                    "truncation_ratio": float(attr.get("truncation_ratio", 0)) if attr is not None else 0.0,
                    "occlusion": t.find("occlusion") is not None,
                    "left": float(box.get("left")),
                    "top": float(box.get("top")),
                    "width": float(box.get("width")),
                    "height": float(box.get("height")),
                }
            )
        out[num] = targets
    return out


def to_yolo(box: dict, img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """left/top/width/height (pixel) -> cx/cy/w/h chuẩn hoá.

    DETRAC để box của xe bị cắt mép vượt ra ngoài ảnh (vẽ cả phần ngoài khung).
    CVAT và định dạng YOLO không biểu diễn được phần đó, nên cắt box theo biên ảnh
    TRƯỚC khi đổi toạ độ — để GT so được công bằng với box người vẽ trên CVAT.
    """
    x1 = min(max(box["left"], 0.0), img_w)
    y1 = min(max(box["top"], 0.0), img_h)
    x2 = min(max(box["left"] + box["width"], 0.0), img_w)
    y2 = min(max(box["top"] + box["height"], 0.0), img_h)
    w, h = x2 - x1, y2 - y1
    return (x1 + w / 2) / img_w, (y1 + h / 2) / img_h, w / img_w, h / img_h


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "configs" / "labels.json")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    work_dir = REPO_ROOT / cfg["paths"]["work_dir"]
    ann_root = REPO_ROOT / cfg["paths"]["detrac_annotations"]

    manifest_path = work_dir / "manifest.jsonl"
    if not manifest_path.is_file():
        die(f"Chưa có {manifest_path}. Chạy scripts/01_sample_frames.py trước.")

    classes = load_label_spec(args.labels)
    class_id = {name: i for i, name in enumerate(classes)}
    log(f"Lớp: {', '.join(f'{i}={n}' for n, i in class_id.items())}")

    gt_dir = work_dir / "gt"
    gt_dir.mkdir(parents=True, exist_ok=True)

    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    by_seq: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_seq[r["sequence_id"]].append(r)

    cls_count = Counter()
    per_cam = defaultdict(lambda: {"images": 0, "boxes": 0, "empty_images": 0, "truncated": 0, "no_gt": 0})
    no_gt: list[str] = []
    unknown_types = Counter()
    total_boxes = 0

    for seq, rows in sorted(by_seq.items()):
        xml_path = ann_root / f"{seq}.xml"
        if not xml_path.is_file():
            die(f"Không thấy annotation {xml_path}")
        frames = parse_sequence_targets(xml_path)

        for r in rows:
            targets = frames.get(r["frame_idx"])
            out_path = gt_dir / f"{Path(r['image_id']).stem}.txt"
            if targets is None:
                # UA-DETRAC bỏ trống cả đoạn frame không gán nhãn (vd MVI_39761: 897-1080).
                # Vắng mặt trong XML = KHÔNG CÓ GT, không phải "không có xe".
                # Không ghi file -> bước đánh giá bỏ qua ảnh này thay vì chấm sai.
                out_path.unlink(missing_ok=True)
                no_gt.append(r["image_id"])
                per_cam[r["camera_id"]]["no_gt"] += 1
                continue

            lines = []
            for t in targets:
                vt = t["vehicle_type"]
                if vt not in class_id:
                    unknown_types[vt] += 1
                    continue
                cx, cy, w, h = to_yolo(t, r["width"], r["height"])
                if w <= 0 or h <= 0:
                    continue
                lines.append(f"{class_id[vt]} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                cls_count[vt] += 1
                if t["truncation_ratio"] > 0:
                    per_cam[r["camera_id"]]["truncated"] += 1

            out_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")

            per_cam[r["camera_id"]]["images"] += 1
            per_cam[r["camera_id"]]["boxes"] += len(lines)
            if not lines:
                per_cam[r["camera_id"]]["empty_images"] += 1
            total_boxes += len(lines)

    # classes.txt để cvat/ultralytics đọc được thứ tự lớp
    (work_dir / "classes.txt").write_text("\n".join(classes) + "\n", encoding="utf-8")

    stats = {
        "num_images": len(records),
        "num_boxes": total_boxes,
        "boxes_per_image": round(total_boxes / len(records), 2) if records else 0,
        "classes": classes,
        "class_distribution": dict(cls_count),
        "per_camera": {k: dict(v) for k, v in sorted(per_cam.items())},
        "images_without_gt": len(no_gt),
        "images_with_gt": len(records) - len(no_gt),
        "unknown_vehicle_types": dict(unknown_types),
    }
    stats_path = work_dir / "gt_stats.json"
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    (work_dir / "gt_missing.json").write_text(json.dumps(sorted(no_gt), indent=2), encoding="utf-8")
    log(f"Đã ghi {len(records) - len(no_gt)} file nhãn -> {gt_dir}")
    log(f"Tổng box: {total_boxes} ({stats['boxes_per_image']} box/ảnh)")
    log(f"Phân bố lớp: {dict(cls_count)}")
    for cam, v in stats["per_camera"].items():
        log(f"  {cam}: {v['images']} ảnh có GT, {v['no_gt']} ảnh KHÔNG có GT, {v['boxes']} box, "
            f"{v['empty_images']} ảnh không có xe, {v['truncated']} box bị cắt")
    if unknown_types:
        log(f"CẢNH BÁO: loại xe không có trong label spec: {dict(unknown_types)}")
    if no_gt:
        log(f"Ghi chú: {len(no_gt)} ảnh rơi vào đoạn DETRAC không gán nhãn -> không có GT, "
            f"bước đánh giá sẽ bỏ qua (danh sách: gt_missing.json)")
    log(f"Thống kê: {stats_path}")


if __name__ == "__main__":
    main()
