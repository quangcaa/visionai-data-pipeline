#!/usr/bin/env python3
"""Tạo project + task trên CVAT và nạp nhãn sơ bộ (chuyển giao Bước 2 -> Bước 3).

Sau khi chạy, task nằm ở stage `annotation`: đánh giá viên mở lên, sửa box sai,
xoá box thừa, VÀ thêm box bị bỏ sót (nguyên tắc 3 trong docx).

Thông tin truy xuất nguồn gốc (model, version, conf, thời điểm chạy) được ghi vào
phần mô tả của task, để sau này biết nhãn sơ bộ do đâu mà có.

Chuẩn bị: điền CVAT_USER / CVAT_PASSWORD vào .env, CVAT đang chạy ở localhost:8080.

Dùng:
    .venv/bin/python scripts/06_cvat_create_task.py
    .venv/bin/python scripts/06_cvat_create_task.py --camera cam03 --task-name "night demo"
    .venv/bin/python scripts/06_cvat_create_task.py --no-annotations   # tạo task trống
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def log(msg: str) -> None:
    print(f"[cvat] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"[cvat] LỖI: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip("'\""))


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "configs" / "labels.json")
    ap.add_argument("--project-name", default="UA-DETRAC vehicle detection")
    ap.add_argument("--task-name", default=None)
    ap.add_argument("--camera", default=None, help="chỉ lấy ảnh của một camera, vd cam03")
    ap.add_argument("--segment-size", type=int, default=50, help="số ảnh mỗi job")
    ap.add_argument("--no-annotations", action="store_true", help="tạo task trống, không nạp nhãn sơ bộ")
    args = ap.parse_args()

    load_dotenv(REPO_ROOT / ".env")
    host = os.environ.get("CVAT_HOST", "localhost")
    # cvat-sdk mặc định dùng https khi host không có scheme; CVAT local chạy http thường.
    if "://" not in host:
        host = f"http://{host}"
    port = int(os.environ.get("CVAT_PORT", 8080))
    user = os.environ.get("CVAT_USER")
    password = os.environ.get("CVAT_PASSWORD")
    if not user or not password:
        die("Thiếu CVAT_USER / CVAT_PASSWORD trong .env (tài khoản tạo bằng createsuperuser)")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    work = REPO_ROOT / cfg["paths"]["work_dir"]
    img_dir = work / "images"

    images = sorted(img_dir.glob("*.jpg"))
    if args.camera:
        images = [p for p in images if p.name.startswith(f"{args.camera}_")]
    if not images:
        die(f"Không có ảnh nào ở {img_dir}" + (f" cho camera {args.camera}" if args.camera else ""))

    label_spec = json.loads(args.labels.read_text(encoding="utf-8"))

    ann_zip = work / "prelabels_coco.zip"
    use_ann = not args.no_annotations
    if use_ann and not ann_zip.is_file():
        die(f"Chưa có {ann_zip}. Chạy scripts/05_prelabels_to_cvat.py trước "
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
        from cvat_sdk import make_client, models
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

    task_name = args.task_name or f"detrac-{args.camera or 'all'}-{datetime.now().strftime('%Y%m%d-%H%M')}"

    log(f"Kết nối {host}:{port} với tài khoản '{user}'...")
    with make_client(host=host, port=port, credentials=(user, password)) as client:
        client.organization_slug = os.environ.get("CVAT_ORG", "")

        # Project giữ label spec dùng chung cho mọi task.
        project = next((p for p in client.projects.list() if p.name == args.project_name), None)
        if project is None:
            project = client.projects.create(
                models.ProjectWriteRequest(
                    name=args.project_name,
                    labels=[models.PatchedLabelRequest(**lb) for lb in label_spec],
                )
            )
            log(f"Đã tạo project '{project.name}' (id={project.id})")
        else:
            log(f"Dùng lại project '{project.name}' (id={project.id})")

        log(f"Tạo task '{task_name}' với {len(images)} ảnh, segment_size={args.segment_size}...")
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
