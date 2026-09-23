#!/usr/bin/env python3
"""Bước 2 — Gán nhãn sơ bộ bằng YOLO26 (mục 4, docx).

Đầu vào : lô ảnh đã tuyển chọn ở Bước 1 (data/prototype/images)
Đầu ra  : nhãn SƠ BỘ ở data/prototype/prelabels/ + _provenance.json

Ràng buộc từ docx:
  - nhãn do model sinh KHÔNG đi thẳng vào dataset; để ở thư mục riêng, chờ người sửa
  - phải lưu model + version đã sinh nhãn (provenance)
  - tham số nằm trong configs/prelabel.yaml

Định dạng mỗi dòng: <class_id> <cx> <cy> <w> <h> <conf>   (toạ độ chuẩn hoá)
Cột conf là phần mở rộng so với YOLO chuẩn, dùng khi đánh giá và lọc; bỏ cột cuối là ra YOLO chuẩn.

Dùng:
    .venv/bin/python scripts/01_prelabel.py
    .venv/bin/python scripts/01_prelabel.py --force
"""

from __future__ import annotations

import argparse
import json
import platform
import shutil
import time
from collections import Counter
from datetime import datetime, timezone

from _common import REPO_ROOT, add_io_args, load_label_spec, load_yaml, logger


log, die = logger("prelabel")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(ap, "config", "prelabel-config", "labels")
    ap.add_argument("--force", action="store_true", help="ghi đè nhãn sơ bộ đã có")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    pcfg = load_yaml(args.prelabel_config)

    work_dir = REPO_ROOT / cfg["paths"]["work_dir"]
    img_dir = work_dir / "images"
    out_dir = work_dir / "prelabels"

    if not img_dir.is_dir() or not any(img_dir.glob("*.jpg")):
        die(f"Chưa có ảnh ở {img_dir}. Chạy scripts/00_ingest.py trước.")
    if out_dir.exists() and any(out_dir.glob("*.txt")):
        if not args.force:
            die(f"Đã có nhãn sơ bộ ở {out_dir}. Dùng --force để chạy lại.")
        # Xoá sạch chứ không ghi đè: lô trước có ảnh mà lô này không có thì nhãn
        # của nó nằm lại, vừa tốn đĩa vừa có thể bị đọc nhầm nếu tên frame trùng.
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = load_label_spec(args.labels)
    class_id = {name: i for i, name in enumerate(classes)}

    # COCO id -> chỉ số lớp DETRAC
    coco_to_detrac = {}
    for coco_id, detrac_name in pcfg["class_map"].items():
        if detrac_name not in class_id:
            die(f"class_map trỏ tới lớp '{detrac_name}' không có trong labels.json")
        coco_to_detrac[int(coco_id)] = class_id[detrac_name]

    try:
        import ultralytics
        from ultralytics import YOLO
    except ImportError:
        die("Chưa cài ultralytics: pip install ultralytics")

    log(f"Nạp model {pcfg['model']} (lần đầu sẽ tải về)...")
    model = YOLO(pcfg["model"])

    images = sorted(img_dir.glob("*.jpg"))
    log(f"{len(images)} ảnh, device={pcfg['device']}, conf={pcfg['conf']}, imgsz={pcfg['imgsz']}")

    started = time.time()
    results = model.predict(
        source=[str(p) for p in images],
        conf=float(pcfg["conf"]),
        iou=float(pcfg["iou"]),
        imgsz=int(pcfg["imgsz"]),
        device=str(pcfg["device"]),
        max_det=int(pcfg["max_det"]),
        classes=sorted(coco_to_detrac),
        stream=True,
        verbose=False,
    )

    cls_count = Counter()
    per_cam = Counter()
    total_boxes = 0
    empty_images = 0
    dropped = 0

    for i, (img_path, r) in enumerate(zip(images, results), 1):
        lines = []
        boxes = r.boxes
        if boxes is not None and len(boxes):
            for coco_cls, conf, xywhn in zip(
                boxes.cls.tolist(), boxes.conf.tolist(), boxes.xywhn.tolist()
            ):
                detrac_id = coco_to_detrac.get(int(coco_cls))
                if detrac_id is None:      # lớp COCO ngoài class_map -> bỏ
                    dropped += 1
                    continue
                cx, cy, w, h = xywhn
                lines.append(f"{detrac_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f} {conf:.4f}")
                cls_count[classes[detrac_id]] += 1
                per_cam[img_path.name.split("_")[0]] += 1

        (out_dir / f"{img_path.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
        )
        total_boxes += len(lines)
        if not lines:
            empty_images += 1
        if i % 25 == 0 or i == len(images):
            log(f"  {i}/{len(images)} ảnh, {total_boxes} box")

    elapsed = time.time() - started

    provenance = {
        "step": "2 - gan nhan so bo",
        "status": "preliminary_not_confirmed",
        "prelabel_model": pcfg["model"],
        "ultralytics_version": ultralytics.__version__,
        "params": {k: pcfg[k] for k in ("conf", "iou", "imgsz", "device", "max_det")},
        "class_map_coco_to_detrac": {str(k): v for k, v in pcfg["class_map"].items()},
        "classes": classes,
        "run_at": datetime.now(timezone.utc).isoformat(),
        "runtime_sec": round(elapsed, 1),
        "python": platform.python_version(),
        "num_images": len(images),
        "num_boxes": total_boxes,
        "boxes_per_image": round(total_boxes / len(images), 2) if images else 0,
        "empty_images": empty_images,
        "class_distribution": dict(cls_count),
        "boxes_per_camera": dict(per_cam),
        "dropped_out_of_map": dropped,
    }
    prov_path = out_dir / "_provenance.json"
    prov_path.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Xong sau {elapsed:.1f}s ({elapsed / max(len(images), 1):.2f}s/ảnh)")
    log(f"{total_boxes} box sơ bộ, {empty_images} ảnh không có box")
    log(f"Phân bố lớp: {dict(cls_count)}")
    log(f"Theo camera : {dict(per_cam)}")
    log(f"Provenance  : {prov_path}")
    log("LƯU Ý: đây là nhãn SƠ BỘ, chưa được xác nhận — chưa được đưa vào dataset.")


if __name__ == "__main__":
    main()
