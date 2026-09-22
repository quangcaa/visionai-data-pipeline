#!/usr/bin/env python3
"""Chuyển nhãn sơ bộ YOLO26 sang gói COCO 1.0 để nạp vào CVAT.

CVAT không đọc trực tiếp .txt của YOLO kèm cột conf, nên bước này đóng gói lại:
    prelabels_coco.zip
      └── annotations/instances_default.json

Dự đoán rơi vào vùng bỏ qua (configs/ignore_regions.json) bị loại, để đánh giá viên
không phải xoá tay hàng trăm box ở những vùng vốn không cần gán nhãn.

Dùng:
    .venv/bin/python scripts/02_prelabels_to_cvat.py
    .venv/bin/python scripts/02_prelabels_to_cvat.py --min-conf 0.4
    .venv/bin/python scripts/02_prelabels_to_cvat.py --no-drop-ignored
"""

from __future__ import annotations

import argparse
import json
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from _common import (REPO_ROOT, add_io_args, boxes_in_ignored, ignore_regions_xyxy,
                     load_ignore_regions, load_label_spec, load_yaml, load_yolo,
                     logger, read_manifest, yolo_to_xyxy)


log, die = logger("to-cvat")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(ap, "config", "prelabel-config", "labels")
    ap.add_argument("--min-conf", type=float, default=None,
                    help="ngưỡng conf tối thiểu (mặc định lấy export.min_conf trong prelabel.yaml)")
    ap.add_argument("--no-drop-ignored", action="store_true",
                    help="giữ cả dự đoán nằm trong vùng bỏ qua")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    pcfg = load_yaml(args.prelabel_config)
    classes = load_label_spec(args.labels)

    work = REPO_ROOT / cfg["paths"]["work_dir"]
    pred_dir = work / "prelabels"
    if not pred_dir.is_dir():
        die(f"Chưa có nhãn sơ bộ ở {pred_dir}. Chạy scripts/01_prelabel.py trước.")

    export = pcfg.get("export") or {}
    min_conf = args.min_conf if args.min_conf is not None else float(export.get("min_conf", 0.25))
    drop_ignored = bool(export.get("drop_in_ignore", True)) and not args.no_drop_ignored
    ig_thr = float((cfg.get("ignore_regions") or {}).get("overlap_threshold", 0.5))

    records = read_manifest(work)
    ignore = load_ignore_regions(cfg)

    # CVAT khớp ảnh theo file_name, nên giữ đúng tên file trong task.
    coco = {
        "licenses": [{"id": 0, "name": "", "url": ""}],
        "info": {
            "description": "Nhan so bo YOLO26 - CHUA XAC NHAN (buoc 2)",
            "date_created": datetime.now(timezone.utc).isoformat(),
        },
        "categories": [{"id": i + 1, "name": name, "supercategory": ""} for i, name in enumerate(classes)],
        "images": [],
        "annotations": [],
    }

    ann_id = 1
    n_dropped_conf = n_dropped_ignored = 0

    for img_id, r in enumerate(records, 1):
        stem = Path(r["image_id"]).stem
        w, h = r["width"], r["height"]
        coco["images"].append(
            {"id": img_id, "file_name": r["image_id"], "width": w, "height": h, "license": 0, "flickr_url": "",
             "coco_url": "", "date_captured": 0}
        )

        p = load_yolo(pred_dir / f"{stem}.txt", 6)
        if not len(p):
            continue

        cls = p[:, 0].astype(int)
        conf = p[:, 5]
        xyxy = yolo_to_xyxy(p[:, 1:5], w, h)

        keep = conf >= min_conf
        n_dropped_conf += int((~keep).sum())

        if drop_ignored:
            reg_box = ignore_regions_xyxy(ignore.get(r["camera_id"]), w, h)
            in_ig = boxes_in_ignored(xyxy, reg_box, ig_thr)
            n_dropped_ignored += int((keep & in_ig).sum())
            keep &= ~in_ig

        for i in np.where(keep)[0]:
            x1, y1, x2, y2 = xyxy[i]
            x1, y1 = max(0.0, x1), max(0.0, y1)
            x2, y2 = min(float(w), x2), min(float(h), y2)
            bw_i, bh_i = x2 - x1, y2 - y1
            if bw_i <= 1 or bh_i <= 1:
                continue
            coco["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls[i]) + 1,   # COCO đánh id lớp từ 1
                    "bbox": [round(x1, 2), round(y1, 2), round(bw_i, 2), round(bh_i, 2)],
                    "area": round(bw_i * bh_i, 2),
                    "segmentation": [],
                    "iscrowd": 0,
                    "attributes": {"occluded": False, "conf": round(float(conf[i]), 4)},
                }
            )
            ann_id += 1

    out_zip = args.out or (work / "prelabels_coco.zip")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations/instances_default.json", json.dumps(coco, ensure_ascii=False))

    log(f"{len(coco['images'])} ảnh, {len(coco['annotations'])} box sơ bộ -> {out_zip}")
    log(f"Đã loại: {n_dropped_conf} box dưới conf {min_conf}"
        + (f", {n_dropped_ignored} box trong vùng bỏ qua" if drop_ignored else ""))
    log("Nạp vào CVAT bằng scripts/03_cvat_create_task.py, hoặc trong UI: Task > Upload annotations > COCO 1.0")


if __name__ == "__main__":
    main()
