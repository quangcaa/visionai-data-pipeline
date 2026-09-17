#!/usr/bin/env python3
"""Chuyển nhãn sơ bộ YOLO26 sang gói COCO 1.0 để nạp vào CVAT.

CVAT không đọc trực tiếp .txt của YOLO kèm cột conf, nên bước này đóng gói lại:
    prelabels_coco.zip
      └── annotations/instances_default.json

Dự đoán rơi vào ignored_region của DETRAC bị loại (giống lúc chấm điểm ở
04_eval_prelabel.py) để đánh giá viên không phải xoá tay hàng trăm box ở
những vùng vốn không cần gán nhãn.

Dùng:
    .venv/bin/python scripts/05_prelabels_to_cvat.py
    .venv/bin/python scripts/05_prelabels_to_cvat.py --min-conf 0.4
    .venv/bin/python scripts/05_prelabels_to_cvat.py --no-drop-ignored
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[to-cvat] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[to-cvat] LỖI: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def inside_ignored(pred_xyxy: np.ndarray, regions: np.ndarray, thr: float) -> np.ndarray:
    """True nếu >= thr diện tích box nằm trong một vùng bỏ qua."""
    if pred_xyxy.size == 0 or regions.size == 0:
        return np.zeros(len(pred_xyxy), dtype=bool)
    x1 = np.maximum(pred_xyxy[:, None, 0], regions[None, :, 0])
    y1 = np.maximum(pred_xyxy[:, None, 1], regions[None, :, 1])
    x2 = np.minimum(pred_xyxy[:, None, 2], regions[None, :, 2])
    y2 = np.minimum(pred_xyxy[:, None, 3], regions[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip((pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]), 1e-9, None)
    return (inter / area[:, None]).max(axis=1) >= thr


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs" / "eval.yaml")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "configs" / "labels.json")
    ap.add_argument("--min-conf", type=float, default=None,
                    help="ngưỡng conf tối thiểu (mặc định lấy conf_report trong eval.yaml)")
    ap.add_argument("--no-drop-ignored", action="store_true",
                    help="giữ cả dự đoán trong ignored_region")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    ecfg = yaml.safe_load(args.eval_config.read_text(encoding="utf-8"))
    classes = [i["name"] for i in json.loads(args.labels.read_text(encoding="utf-8"))]

    work = REPO_ROOT / cfg["paths"]["work_dir"]
    pred_dir = work / "prelabels"
    if not pred_dir.is_dir():
        die(f"Chưa có nhãn sơ bộ ở {pred_dir}. Chạy scripts/03_prelabel_yolo26.py trước.")

    min_conf = args.min_conf if args.min_conf is not None else float(ecfg["conf_report"])
    drop_ignored = not args.no_drop_ignored
    ig_thr = float(ecfg["ignore_regions"]["overlap_threshold"])

    records = [json.loads(l) for l in (work / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    scenes = json.loads((work / "scenes.json").read_text(encoding="utf-8"))

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

        p_file = pred_dir / f"{stem}.txt"
        if not p_file.is_file() or p_file.stat().st_size == 0:
            continue
        p = np.loadtxt(p_file, ndmin=2)

        cls = p[:, 0].astype(int)
        cx, cy, bw, bh = p[:, 1] * w, p[:, 2] * h, p[:, 3] * w, p[:, 4] * h
        conf = p[:, 5]
        xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)

        keep = conf >= min_conf
        n_dropped_conf += int((~keep).sum())

        if drop_ignored:
            regs = scenes[r["sequence_id"]]["ignored_regions"]
            reg_box = np.array([[x["left"], x["top"], x["left"] + x["width"], x["top"] + x["height"]]
                                for x in regs]) if regs else np.zeros((0, 4))
            in_ig = inside_ignored(xyxy, reg_box, ig_thr)
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
        + (f", {n_dropped_ignored} box trong ignored_region" if drop_ignored else ""))
    log("Nạp vào CVAT bằng scripts/06_cvat_create_task.py, hoặc trong UI: Task > Upload annotations > COCO 1.0")


if __name__ == "__main__":
    main()
