#!/usr/bin/env python3
"""Đo chất lượng nhãn sơ bộ (Bước 2) bằng cách so với Ground Truth UA-DETRAC.

Trả lời câu hỏi: nhãn YOLO26 sinh ra tốt tới đâu, và vì sao vẫn cần người sửa.

Chỉ số:
  - AP theo lớp, mAP@0.5 và mAP@0.5:0.95 (kiểu COCO, nội suy toàn điểm)
  - Precision / Recall / F1 tại ngưỡng conf trong configs/eval.yaml
  - Tỷ lệ bỏ sót (GT không có dự đoán khớp) và số dự đoán thừa
  - Tách theo camera / điều kiện thời tiết
  - Ma trận nhầm lớp giữa các cặp box đã ghép đúng vị trí

Dự đoán nằm trong ignored_region của DETRAC bị loại trước khi chấm.

Dùng:
    .venv/bin/python scripts/04_eval_prelabel.py
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from _common import (REPO_ROOT, boxes_in_ignored, load_label_spec,
                     load_yolo, load_yaml, yolo_to_xyxy)


def log(msg: str) -> None:
    print(f"[eval] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[eval] LỖI: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU giữa mọi cặp (a_i, b_j)."""
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def average_precision(tp: np.ndarray, conf: np.ndarray, n_gt: int) -> float:
    """AP nội suy toàn điểm (all-point interpolation), như COCO/Ultralytics."""
    if n_gt == 0:
        return float("nan")
    if len(tp) == 0:
        return 0.0
    order = np.argsort(-conf)
    tp = tp[order]
    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(1 - tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.clip(tp_cum + fp_cum, 1e-9, None)
    # precision envelope: lấy max từ phải sang trái
    mrec = np.concatenate(([0.0], recall, [recall[-1]]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    return float(np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1]))


def match_at_iou(gt: np.ndarray, gt_cls: np.ndarray, pred: np.ndarray, pred_cls: np.ndarray,
                 pred_conf: np.ndarray, thr: float) -> tuple[np.ndarray, np.ndarray]:
    """Ghép greedy theo conf giảm dần, cùng lớp, IoU >= thr. Trả về (tp cho pred, gt đã khớp)."""
    tp = np.zeros(len(pred), dtype=np.float64)
    gt_used = np.zeros(len(gt), dtype=bool)
    if len(pred) == 0 or len(gt) == 0:
        return tp, gt_used
    ious = iou_matrix(pred, gt)
    for pi in np.argsort(-pred_conf):
        cand = np.where((~gt_used) & (gt_cls == pred_cls[pi]) & (ious[pi] >= thr))[0]
        if len(cand):
            best = cand[np.argmax(ious[pi, cand])]
            gt_used[best] = True
            tp[pi] = 1.0
    return tp, gt_used


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs" / "eval.yaml")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "configs" / "labels.json")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    ecfg = load_yaml(args.eval_config)
    classes = load_label_spec(args.labels)

    work = REPO_ROOT / cfg["paths"]["work_dir"]
    gt_dir, pred_dir = work / "gt", work / "prelabels"
    for d in (gt_dir, pred_dir):
        if not d.is_dir():
            die(f"Thiếu thư mục {d}")

    records = [json.loads(l) for l in (work / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    scenes = json.loads((work / "scenes.json").read_text(encoding="utf-8"))

    iou_match = float(ecfg["iou_match"])
    conf_report = float(ecfg["conf_report"])
    start, stop, step = ecfg["map_iou_range"]
    iou_thresholds = np.round(np.arange(start, stop + 1e-9, step), 2)
    use_ignore = bool(ecfg["ignore_regions"]["enabled"])
    ig_thr = float(ecfg["ignore_regions"]["overlap_threshold"])

    # gom theo lớp để tính AP, và theo ảnh để tính P/R
    per_class_conf = defaultdict(list)
    per_class_tp = defaultdict(lambda: defaultdict(list))   # [iou_thr][cls] -> tp
    n_gt_per_class = defaultdict(int)
    per_group = defaultdict(lambda: {"gt": 0, "pred": 0, "tp": 0, "missed": 0})
    confusion = np.zeros((len(classes), len(classes)), dtype=int)
    n_ignored_preds = 0
    n_no_gt = 0

    for r in records:
        stem = Path(r["image_id"]).stem
        w, h = r["width"], r["height"]
        if not (gt_dir / f"{stem}.txt").is_file():
            n_no_gt += 1      # đoạn DETRAC không gán nhãn: không có đáp án để chấm
            continue

        g = load_yolo(gt_dir / f"{stem}.txt", 5)
        p_file = pred_dir / f"{stem}.txt"
        p = load_yolo(p_file, 6)

        gt_cls = g[:, 0].astype(int) if len(g) else np.zeros(0, dtype=int)
        gt_box = yolo_to_xyxy(g[:, 1:5], w, h) if len(g) else np.zeros((0, 4))
        pr_cls = p[:, 0].astype(int) if len(p) else np.zeros(0, dtype=int)
        pr_box = yolo_to_xyxy(p[:, 1:5], w, h) if len(p) else np.zeros((0, 4))
        pr_conf = p[:, 5] if len(p) else np.zeros(0)

        if use_ignore:
            regs = scenes[r["sequence_id"]]["ignored_regions"]
            reg_box = np.array([[x["left"], x["top"], x["left"] + x["width"], x["top"] + x["height"]]
                                for x in regs]) if regs else np.zeros((0, 4))
            keep = ~boxes_in_ignored(pr_box, reg_box, ig_thr)
            n_ignored_preds += int((~keep).sum())
            pr_cls, pr_box, pr_conf = pr_cls[keep], pr_box[keep], pr_conf[keep]

        for c in gt_cls:
            n_gt_per_class[int(c)] += 1
        for thr in iou_thresholds:
            tp, _ = match_at_iou(gt_box, gt_cls, pr_box, pr_cls, pr_conf, float(thr))
            for ci in range(len(classes)):
                m = pr_cls == ci
                if m.any():
                    per_class_tp[float(thr)][ci].extend(tp[m].tolist())
        for ci in range(len(classes)):
            m = pr_cls == ci
            if m.any():
                per_class_conf[ci].extend(pr_conf[m].tolist())

        # thống kê tại ngưỡng báo cáo
        sel = pr_conf >= conf_report
        tp_r, gt_used = match_at_iou(gt_box, gt_cls, pr_box[sel], pr_cls[sel], pr_conf[sel], iou_match)
        for key in (r["camera_id"], r["weather"], "TOTAL"):
            per_group[key]["gt"] += len(gt_box)
            per_group[key]["pred"] += int(sel.sum())
            per_group[key]["tp"] += int(tp_r.sum())
            per_group[key]["missed"] += int((~gt_used).sum())

        # nhầm lớp: box khớp vị trí (IoU>=thr) nhưng khác lớp
        if len(gt_box) and sel.any():
            ious = iou_matrix(pr_box[sel], gt_box)
            used = np.zeros(len(gt_box), dtype=bool)
            for pi in np.argsort(-pr_conf[sel]):
                cand = np.where((~used) & (ious[pi] >= iou_match))[0]
                if len(cand):
                    gi = cand[np.argmax(ious[pi, cand])]
                    used[gi] = True
                    confusion[gt_cls[gi], pr_cls[sel][pi]] += 1

    # --- AP ---
    ap_table = {}
    for ci, name in enumerate(classes):
        conf = np.array(per_class_conf[ci])
        aps = []
        for thr in iou_thresholds:
            tp = np.array(per_class_tp[float(thr)][ci])
            aps.append(average_precision(tp, conf, n_gt_per_class[ci]))
        ap_table[name] = {
            "n_gt": n_gt_per_class[ci],
            "n_pred": len(conf),
            "AP50": aps[0],
            "AP50_95": float(np.nanmean(aps)) if n_gt_per_class[ci] else float("nan"),
        }

    valid = [v for v in ap_table.values() if v["n_gt"] > 0]
    map50 = float(np.mean([v["AP50"] for v in valid])) if valid else 0.0
    map5095 = float(np.mean([v["AP50_95"] for v in valid])) if valid else 0.0

    for v in per_group.values():
        v["precision"] = v["tp"] / v["pred"] if v["pred"] else 0.0
        v["recall"] = v["tp"] / v["gt"] if v["gt"] else 0.0
        v["f1"] = (2 * v["precision"] * v["recall"] / (v["precision"] + v["recall"])
                   if v["precision"] + v["recall"] else 0.0)
        v["miss_rate"] = v["missed"] / v["gt"] if v["gt"] else 0.0
        v["false_positives"] = v["pred"] - v["tp"]

    # --- in ra ---
    t = per_group["TOTAL"]
    log(f"Ảnh chấm: {len(records) - n_no_gt}/{len(records)} ({n_no_gt} ảnh không có GT, bỏ qua) | "
        f"GT: {t['gt']} box | dự đoán (conf>={conf_report}): {t['pred']} box")
    if use_ignore:
        log(f"Đã loại {n_ignored_preds} dự đoán rơi vào ignored_region của DETRAC")
    print()
    print(f"  mAP@0.5      = {map50:.3f}")
    print(f"  mAP@0.5:0.95 = {map5095:.3f}")
    print()
    print(f"  {'lớp':<8}{'GT':>7}{'pred':>7}{'AP50':>9}{'AP50-95':>10}")
    for name, v in ap_table.items():
        print(f"  {name:<8}{v['n_gt']:>7}{v['n_pred']:>7}{v['AP50']:>9.3f}{v['AP50_95']:>10.3f}")
    print()
    print(f"  Tại conf>={conf_report}, IoU>={iou_match}:")
    print(f"  {'nhóm':<10}{'GT':>7}{'pred':>7}{'TP':>7}{'FP':>7}{'sót':>7}{'P':>8}{'R':>8}{'F1':>8}")
    for key in list(per_group):
        v = per_group[key]
        print(f"  {key:<10}{v['gt']:>7}{v['pred']:>7}{v['tp']:>7}{v['false_positives']:>7}"
              f"{v['missed']:>7}{v['precision']:>8.3f}{v['recall']:>8.3f}{v['f1']:>8.3f}")
    print()
    print("  Ma trận nhầm lớp (hàng = GT, cột = dự đoán, chỉ box khớp vị trí):")
    print(f"  {'':<8}" + "".join(f"{c:>9}" for c in classes))
    for i, c in enumerate(classes):
        print(f"  {c:<8}" + "".join(f"{confusion[i, j]:>9}" for j in range(len(classes))))

    out = {
        "iou_match": iou_match,
        "conf_report": conf_report,
        "map_iou_thresholds": iou_thresholds.tolist(),
        "ignored_predictions": n_ignored_preds,
        "images_evaluated": len(records) - n_no_gt,
        "images_without_gt": n_no_gt,
        "mAP50": map50,
        "mAP50_95": map5095,
        "per_class": ap_table,
        "per_group": {k: dict(v) for k, v in per_group.items()},
        "confusion_gt_rows_pred_cols": confusion.tolist(),
        "classes": classes,
    }
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    path = reports / "prelabel_eval.json"
    path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print()
    log(f"Báo cáo: {path}")


if __name__ == "__main__":
    main()
