#!/usr/bin/env python3
"""Tạo project + task trên CVAT và nạp nhãn sơ bộ (chuyển giao Bước 2 -> Bước 3).

Sau khi chạy, task nằm ở stage `annotation`: đánh giá viên mở lên, sửa box sai,
xoá box thừa, VÀ thêm box bị bỏ sót (nguyên tắc 3 trong docx).

Thông tin truy xuất nguồn gốc (model, version, conf, thời điểm chạy) được ghi vào
phần mô tả của task, để sau này biết nhãn sơ bộ do đâu mà có.

Chuẩn bị: điền CVAT_USER / CVAT_PASSWORD vào .env, CVAT đang chạy ở localhost:8080.

Dùng:
    .venv/bin/python scripts/03_cvat_create_task.py
    .venv/bin/python scripts/03_cvat_create_task.py --camera cam03 --task-name "night demo"
    .venv/bin/python scripts/03_cvat_create_task.py --no-annotations   # tạo task trống
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from _common import (REPO_ROOT, add_io_args, cvat_client, cvat_config, load_json,
                     load_yaml, logger)


log, die = logger("cvat")


def filter_coco_zip(src_zip: Path, keep_names: set[str], dst_zip: Path) -> tuple[int, int]:
    """Lọc gói COCO xuống đúng danh sách ảnh của task.

    CVAT từ chối nạp nhãn nếu gói chứa ảnh không có trong task
    ("Could not match item id ... with any task frame"), nên phải cắt trước.
    """
    with zipfile.ZipFile(src_zip) as zf:
        coco = json.loads(zf.read("annotations/instances_default.json"))

    images = [im for im in coco["images"] if im["file_name"] in keep_names]
    keep_ids = {im["id"] for im in images}
    annotations = [a for a in coco["annotations"] if a["image_id"] in keep_ids]

    coco["images"], coco["annotations"] = images, annotations
    with zipfile.ZipFile(dst_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("annotations/instances_default.json", json.dumps(coco, ensure_ascii=False))
    return len(images), len(annotations)


def sync_project_labels(project, label_spec: list, models, prune: bool = True) -> None:
    """Thêm vào project những lớp có trong labels.json mà project chưa biết.

    Sửa configs/labels.json KHÔNG tự lan sang project đã tạo trên CVAT. Không đồng
    bộ trước thì CVAT từ chối nạp nhãn ("Label 'x' is not registered for this task")
    — và task rỗng đã kịp được tạo, thành rác.

    Lớp thừa (có trong project, không có trong labels.json) được xoá nếu CHƯA AI DÙNG.
    Còn nhãn dùng tới thì giữ lại và báo rõ — xoá đi là mất công sức người gán.
    """
    have = {lb.name for lb in project.get_labels()}
    want = [lb["name"] for lb in label_spec]

    missing = [lb for lb in label_spec if lb["name"] not in have]
    if missing:
        names = ", ".join(lb["name"] for lb in missing)
        try:
            project.update(models.PatchedProjectWriteRequest(
                labels=[models.PatchedLabelRequest(**lb) for lb in missing]))
        except Exception as exc:  # noqa: BLE001
            die(f"không thêm được lớp [{names}] vào project '{project.name}': {exc}")
        log(f"Đã thêm lớp mới vào project: {names}")

    labels = {lb.name: lb for lb in project.get_labels()}
    extra = sorted(set(labels) - set(want))
    if extra:
        used = count_label_usage(project, {labels[n].id: n for n in extra})
        unused = [n for n in extra if not used.get(n)]
        blocked = {n: used[n] for n in extra if used.get(n)}

        if unused and prune:
            try:
                project.update(models.PatchedProjectWriteRequest(
                    labels=[models.PatchedLabelRequest(id=labels[n].id, deleted=True)
                            for n in unused]))
                log(f"Đã xoá khỏi project lớp thừa, chưa ai dùng: {', '.join(unused)}")
            except Exception as exc:  # noqa: BLE001
                log(f"CẢNH BÁO: không xoá được lớp thừa {unused}: {exc}")
        elif unused:
            log(f"Lớp thừa chưa ai dùng, giữ lại theo --keep-extra-labels: {', '.join(unused)}")

        if blocked:
            detail = ", ".join(f"{n} ({c} nhãn)" for n, c in blocked.items())
            log(f"CẢNH BÁO: giữ lại lớp thừa vì đang có nhãn dùng tới: {detail}. "
                f"Xoá đi là mất số nhãn đó — hãy đổi chúng sang lớp khác trong CVAT trước, "
                f"nếu không bước phát hành sẽ từ chối cả lô")

    still = [n for n in want if n not in {lb.name for lb in project.get_labels()}]
    if still:
        die(f"project '{project.name}' vẫn thiếu lớp: {', '.join(still)}")


def count_label_usage(project, label_ids: dict) -> dict:
    """Đếm số nhãn đang dùng từng label_id, trên mọi task của project."""
    from collections import Counter

    n = Counter()
    for task in project.get_tasks():
        ann = task.get_annotations()
        for shape in ann.shapes:
            n[shape.label_id] += 1
        for track in ann.tracks:
            n[track.label_id] += 1
    return {name: n[lid] for lid, name in label_ids.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_io_args(ap, "config", "labels")
    ap.add_argument("--project-name", default=None,
                    help="mặc định lấy cvat.project_name trong pipeline.yaml")
    ap.add_argument("--task-name", default=None)
    ap.add_argument("--camera", action="append",
                    help="chỉ lấy ảnh của camera này; lặp lại để chọn nhiều")
    ap.add_argument("--segment-size", type=int, default=50, help="số ảnh mỗi job")
    ap.add_argument("--no-annotations", action="store_true", help="tạo task trống, không nạp nhãn sơ bộ")
    ap.add_argument("--keep-extra-labels", action="store_true",
                    help="giữ lại lớp có trong project nhưng không có trong labels.json")
    args = ap.parse_args()

    try:
        host, port, user, _ = cvat_config()
    except RuntimeError as exc:
        die(str(exc))

    cfg = load_yaml(args.config)
    project_name = args.project_name or (cfg.get("cvat") or {}).get("project_name", "Vehicle detection")
    work = REPO_ROOT / cfg["paths"]["work_dir"]
    img_dir = work / "images"

    images = sorted(img_dir.glob("*.jpg"))
    if args.camera:
        prefixes = tuple(f"{c}_" for c in args.camera)
        images = [p for p in images if p.name.startswith(prefixes)]
    if not images:
        die(f"Không có ảnh nào ở {img_dir}"
            + (f" cho camera {', '.join(args.camera)}" if args.camera else ""))

    label_spec = load_json(args.labels)

    ann_zip = work / "prelabels_coco.zip"
    use_ann = not args.no_annotations
    if use_ann and not ann_zip.is_file():
        die(f"Chưa có {ann_zip}. Chạy scripts/02_prelabels_to_cvat.py trước "
            f"(hoặc dùng --no-annotations).")
    tmp_dir = None
    if use_ann:
        # Luôn lọc gói nhãn theo đúng ảnh của task này.
        tmp_dir = tempfile.TemporaryDirectory()
        filtered = Path(tmp_dir.name) / "prelabels_filtered.zip"
        n_img, n_ann = filter_coco_zip(ann_zip, {p.name for p in images}, filtered)
        log(f"Lọc gói nhãn sơ bộ cho task: {n_img} ảnh, {n_ann} box")
        if n_img != len(images):
            log(f"CẢNH BÁO: {len(images) - n_img} ảnh của task không có trong gói nhãn sơ bộ")
        ann_zip = filtered

    try:
        from cvat_sdk import models
    except ImportError:
        die("Chưa cài cvat-sdk: pip install cvat-sdk")

    prov_path = work / "prelabels" / "_provenance.json"
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    params = prov.get("params", {})
    # CVAT 2.x không có trường 'description' cho task; 'subset' và 'bug_tracker'
    # là hai trường tự do duy nhất hiển thị được trong UI. Provenance đầy đủ nằm ở
    # reports/cvat_task_<id>.json và prelabels/_provenance.json trong repo.
    provenance_tag = (
        f"prelabel={prov.get('prelabel_model', 'n/a')}"
        f"@ultralytics-{prov.get('ultralytics_version', 'n/a')}"
        f",conf={params.get('conf')},imgsz={params.get('imgsz')}"
        f",run_at={prov.get('run_at', 'n/a')}"
    )

    batch_label = (cfg.get("dataset") or {}).get("name", "batch")
    cam_label = "-".join(args.camera) if args.camera else "all"
    task_name = args.task_name or f"{batch_label}-{cam_label}-{datetime.now().strftime('%Y%m%d-%H%M')}"

    log(f"Kết nối {host}:{port} với tài khoản '{user}'...")
    with cvat_client() as client:
        client.organization_slug = os.environ.get("CVAT_ORG", "")

        # Project giữ label spec dùng chung cho mọi task.
        project = next((p for p in client.projects.list() if p.name == project_name), None)
        if project is None:
            project = client.projects.create(
                models.ProjectWriteRequest(
                    name=project_name,
                    labels=[models.PatchedLabelRequest(**lb) for lb in label_spec],
                )
            )
            log(f"Đã tạo project '{project.name}' (id={project.id})")
        else:
            log(f"Dùng lại project '{project.name}' (id={project.id})")
            sync_project_labels(project, label_spec, models,
                                prune=not args.keep_extra_labels)

        log(f"Tạo task '{task_name}' với {len(images)} ảnh, segment_size={args.segment_size}...")
        try:
            task = client.tasks.create_from_data(
                spec=models.TaskWriteRequest(
                    name=task_name,
                    project_id=project.id,
                    segment_size=args.segment_size,
                ),
                resources=[str(p) for p in images],
                annotation_path=str(ann_zip) if use_ann else "",
                annotation_format="COCO 1.0",
            )
        except Exception as exc:  # noqa: BLE001
            # CVAT tạo task trước rồi mới nạp nhãn; nạp hỏng là task rỗng nằm lại
            die(f"tạo task thất bại: {exc}\n"
                f"  CVAT có thể đã tạo task rỗng — kiểm tra ở {host}:{port}/tasks "
                f"và xoá bằng nút 'Xoá task' trên web")
        log(f"Đã tạo task id={task.id}")

        # Ghi dấu vết nguồn gốc lên task. Không chặn luồng nếu CVAT từ chối,
        # vì nhãn đã nạp xong ở bước trên và provenance vẫn được lưu trong repo.
        try:
            task.update(models.PatchedTaskWriteRequest(bug_tracker=provenance_tag))
            log("Đã ghi provenance vào trường bug_tracker của task")
        except Exception as exc:  # noqa: BLE001
            log(f"CẢNH BÁO: không ghi được provenance lên CVAT ({exc.__class__.__name__}); "
                f"vẫn lưu đầy đủ trong reports/")

        jobs = task.get_jobs()
        n_shapes = len(task.get_annotations().shapes)
        log(f"Task có {len(jobs)} job, {n_shapes} box sơ bộ đã nạp")
        for j in jobs:
            log(f"  job {j.id}: frames {j.start_frame}-{j.stop_frame}, stage={j.stage}, state={j.state}")

    out = {
        "task_id": task.id,
        "task_name": task_name,
        "project_id": project.id,
        "num_images": len(images),
        "num_prelabel_shapes": n_shapes,
        "jobs": [j.id for j in jobs],
        "camera_filter": args.camera,
        "annotations_imported": use_ann,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prelabel_provenance": prov,
        "provenance_tag": provenance_tag,
    }
    reports = REPO_ROOT / "reports"
    reports.mkdir(exist_ok=True)
    out_path = reports / f"cvat_task_{task.id}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")

    log(f"Đã ghi {out_path}")
    log(f"Mở: {host}:{port}/tasks/{task.id}")
    log("Bước 3: mở job, sửa theo docs/labeling_guideline.md, Finish job; "
        "người duyệt đặt PASS (acceptance/completed) hoặc FAIL (annotation/rejected).")


if __name__ == "__main__":
    main()
