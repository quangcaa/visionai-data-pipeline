#!/usr/bin/env python3
"""Bước 5 — Hình thành dataset phát hành (mục 7, docx).

Đầu vào : CHỈ các job người duyệt đã PASS trên CVAT (stage=acceptance, state=completed)
Xử lý   : 1. kéo nhãn đã xác nhận từ CVAT
          2. kiểm tra đủ bộ cho TỪNG ảnh: file ảnh + checksum, nhãn hợp lệ,
             metadata nguồn gốc (camera_id, video nguồn, frame), prelabel model,
             guideline version
          3. thiếu bất kỳ thứ gì -> DỪNG, không phát hành gì cả
          4. dựng dataset/ trong thư mục tạm rồi mới thay thế (không để bản dở dang)
Đầu ra  : dataset/
            images/  labels/  metadata.jsonl  data.yaml  MANIFEST.json  [datasheet.md]

Ranh giới: KHÔNG chia train/val. MLOps chia theo camera_id/sequence_id trong metadata.jsonl.

Dùng:
    .venv/bin/python scripts/04_build_release.py --version 1.0.0
    .venv/bin/python scripts/04_build_release.py --version 1.0.0 --task 4
    .venv/bin/python scripts/04_build_release.py --version 1.1.0 --force   # thay bản đang có
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from _common import (REPO_ROOT, add_io_args, cvat_client, load_label_spec, load_yaml,
                     logger, read_manifest, sha256_of)


log, die = logger("release")


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "scripts", "configs", "docs"],
                               cwd=REPO_ROOT, capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", required=True, help="phiên bản dataset, dạng MAJOR.MINOR.PATCH")
    ap.add_argument("--task", type=int, action="append", help="chỉ lấy job của task này (lặp được)")
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "dataset")
    ap.add_argument("--force", action="store_true", help="thay thế dataset/ đang có")
    add_io_args(ap, "config", "labels")
    args = ap.parse_args()

    if not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        die("--version phải có dạng MAJOR.MINOR.PATCH, ví dụ 1.0.0")

    if args.out.exists() and any(args.out.iterdir()):
        old = args.out / "MANIFEST.json"
        old_ver = json.loads(old.read_text(encoding="utf-8")).get("version") if old.is_file() else "?"
        if not args.force:
            die(f"{args.out} đã có bản {old_ver}. Nếu đã 'dvc add' bản đó thì dùng --force để thay "
                f"(bản cũ vẫn lấy lại được qua git tag + dvc checkout).")
        log(f"Sẽ thay bản {old_ver} bằng {args.version}")

    cfg = load_yaml(args.config)
    dataset_cfg = cfg.get("dataset") or {}
    dataset_name = dataset_cfg.get("name", "dataset")
    classes = load_label_spec(args.labels)
    work = REPO_ROOT / cfg["paths"]["work_dir"]
    img_dir = work / "images"

    manifest = {r["image_id"]: r for r in read_manifest(work)}

    prelabel_prov_path = work / "prelabels" / "_provenance.json"
    prelabel_prov = json.loads(prelabel_prov_path.read_text(encoding="utf-8")) if prelabel_prov_path.is_file() else {}

    guideline_version = (cfg.get("labeling") or {}).get("guideline_version")
    guideline_path = REPO_ROOT / "docs" / "labeling_guideline.md"

    errors: list[str] = []
    if not prelabel_prov.get("prelabel_model"):
        errors.append(f"thiếu provenance nhãn sơ bộ ({prelabel_prov_path.relative_to(REPO_ROOT)})")
    if not guideline_version:
        errors.append("thiếu labeling.guideline_version trong configs/pipeline.yaml")
    if not guideline_path.is_file():
        errors.append("thiếu docs/labeling_guideline.md — không thể phát hành nhãn không rõ gán theo quy tắc nào")

    images_out: list[dict] = []   # {src, name, label_lines, meta}

    try:
        client_cm = cvat_client()
    except RuntimeError as exc:
        die(str(exc))

    with client_cm as client:
        all_jobs = client.jobs.list()
        if args.task:
            all_jobs = [j for j in all_jobs if j.task_id in set(args.task)]
        if not all_jobs:
            die("Không tìm thấy job nào" + (f" trong task {args.task}" if args.task else ""))

        accepted = [j for j in all_jobs if str(j.stage) == "acceptance" and str(j.state) == "completed"]

        log(f"{len(all_jobs)} job, {len(accepted)} đã được chấp nhận:")
        for j in sorted(all_jobs, key=lambda x: x.id):
            mark = "LẤY" if j in accepted else "bỏ "
            log(f"  [{mark}] job {j.id} (task {j.task_id}) stage={j.stage} state={j.state}")
        if not accepted:
            die("Chưa có job nào được duyệt PASS (stage=acceptance, state=completed). "
                "Người duyệt đặt trạng thái này cho job trên CVAT.")

        seen_names: dict[str, int] = {}

        for job in sorted(accepted, key=lambda x: x.id):
            frames_info = job.get_frames_info()
            label_names = {lb.id: lb.name for lb in job.get_labels()}
            ann = job.get_annotations()
            if ann.tracks:
                errors.append(f"job {job.id}: có {len(ann.tracks)} track, bản phát hành chỉ nhận shape ảnh tĩnh")

            by_frame = defaultdict(list)
            for s in ann.shapes:
                by_frame[s.frame].append(s)

            offset = job.start_frame if len(frames_info) == job.stop_frame - job.start_frame + 1 else 0
            for fr in range(job.start_frame, job.stop_frame + 1):
                meta = frames_info[fr - offset]
                name, W, H = meta.name, meta.width, meta.height
                where = f"job {job.id} / {name}"

                if name in seen_names:
                    errors.append(f"{where}: ảnh trùng với job {seen_names[name]}")
                    continue
                seen_names[name] = job.id

                # --- ảnh + nguồn gốc ---
                src = img_dir / name
                rec = manifest.get(name)
                if rec is None:
                    errors.append(f"{where}: không có trong manifest.jsonl (mất nguồn gốc)")
                    continue
                for field in ("camera_id", "sequence_id", "frame_idx", "source_path", "sha256"):
                    if rec.get(field) in (None, ""):
                        errors.append(f"{where}: metadata thiếu '{field}'")
                if not src.is_file():
                    errors.append(f"{where}: không thấy file ảnh {src.relative_to(REPO_ROOT)}")
                    continue
                if sha256_of(src) != rec.get("sha256"):
                    errors.append(f"{where}: checksum ảnh không khớp manifest (ảnh bị thay đổi?)")
                    continue

                # --- nhãn đã xác nhận ---
                lines = []
                for s in by_frame.get(fr, []):
                    lname = label_names.get(s.label_id)
                    stype = getattr(s.type, "value", str(s.type))
                    if stype != "rectangle":
                        errors.append(f"{where}: shape '{stype}' không phải rectangle")
                        continue
                    if lname not in classes:
                        errors.append(f"{where}: lớp '{lname}' không có trong label spec")
                        continue
                    x1, y1, x2, y2 = s.points
                    x1, x2 = sorted((max(0.0, x1), min(float(W), x2)))
                    y1, y2 = sorted((max(0.0, y1), min(float(H), y2)))
                    if x2 - x1 <= 0 or y2 - y1 <= 0:
                        errors.append(f"{where}: box kích thước <= 0")
                        continue
                    cx, cy = (x1 + x2) / 2 / W, (y1 + y2) / 2 / H
                    lines.append(f"{classes.index(lname)} {cx:.6f} {cy:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")

                images_out.append({
                    "src": src,
                    "name": name,
                    "lines": lines,
                    "meta": {
                        **{k: rec.get(k) for k in ("image_id", "camera_id", "sequence_id", "frame_idx",
                                                   "timestamp_in_video_sec", "weather", "camera_state",
                                                   "source_path", "width", "height", "sha256", "batch_id")},
                        "num_boxes": len(lines),
                        "label_status": "accepted",
                        "cvat_task_id": job.task_id,
                        "cvat_job_id": job.id,
                        "prelabel_model": prelabel_prov.get("prelabel_model"),
                        "prelabel_ultralytics_version": prelabel_prov.get("ultralytics_version"),
                        "guideline_version": guideline_version,
                        "review_decision": "PASS",
                        "accepted_at": str(job.updated_date),
                        "reviewer_id": getattr(job.assignee, "id", None) if job.assignee else None,
                    },
                })

    # --- cổng chặn: thiếu gì cũng không phát hành ---
    if errors:
        log(f"KIỂM TRA ĐỦ BỘ THẤT BẠI — {len(errors)} lỗi, KHÔNG phát hành:")
        for e in errors[:50]:
            print(f"    - {e}", file=sys.stderr)
        if len(errors) > 50:
            print(f"    ... và {len(errors) - 50} lỗi nữa", file=sys.stderr)
        sys.exit(1)
    if not images_out:
        die("Không có ảnh nào để phát hành")

    log(f"Kiểm tra đủ bộ: {len(images_out)} ảnh đạt")

    # --- dựng trong thư mục tạm ---
    commit = git_commit()
    if commit is None:
        log("CẢNH BÁO: repo chưa có commit — MANIFEST không ghi được phiên bản code đã tạo ra dataset")
    elif commit.endswith("-dirty"):
        log("CẢNH BÁO: scripts/configs/docs có thay đổi chưa commit — nên commit trước khi phát hành")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=".dataset-build-", dir=args.out.parent))
    try:
        (tmp / "images").mkdir()
        (tmp / "labels").mkdir()

        cls_count, cam_count, weather_count = Counter(), Counter(), Counter()
        files_checksum = {}
        with (tmp / "metadata.jsonl").open("w", encoding="utf-8") as mf:
            for item in sorted(images_out, key=lambda x: x["name"]):
                shutil.copy2(item["src"], tmp / "images" / item["name"])
                label_name = f"{Path(item['name']).stem}.txt"
                (tmp / "labels" / label_name).write_text(
                    "\n".join(item["lines"]) + ("\n" if item["lines"] else ""), encoding="utf-8")
                mf.write(json.dumps(item["meta"], ensure_ascii=False) + "\n")

                files_checksum[f"images/{item['name']}"] = item["meta"]["sha256"]
                files_checksum[f"labels/{label_name}"] = sha256_of(tmp / "labels" / label_name)
                for ln in item["lines"]:
                    cls_count[classes[int(ln.split()[0])]] += 1
                cam_count[item["meta"]["camera_id"]] += 1
                weather_count[item["meta"]["weather"]] += 1

        data_yaml = (
            f"# {dataset_name} v{args.version}\n"
            "# KHÔNG chia train/val trong pipeline dữ liệu (ranh giới docx mục 7).\n"
            "# train và val cùng trỏ vào toàn bộ ảnh chỉ để file hợp lệ với Ultralytics.\n"
            "# MLOps phải tự chia theo camera_id / sequence_id trong metadata.jsonl,\n"
            "# KHÔNG chia ngẫu nhiên theo ảnh (các frame cùng video gần như giống nhau).\n"
            "path: .\n"
            "train: images\n"
            "val: images\n"
            f"nc: {len(classes)}\n"
            "names:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(classes))
        )
        (tmp / "data.yaml").write_text(data_yaml, encoding="utf-8")

        datasheet = REPO_ROOT / "docs" / "datasheet.md"
        if datasheet.is_file():
            shutil.copy2(datasheet, tmp / "datasheet.md")
        else:
            log("CẢNH BÁO: chưa có docs/datasheet.md — bản phát hành không kèm datasheet")

        jobs_used = sorted({(m["meta"]["cvat_task_id"], m["meta"]["cvat_job_id"], m["meta"]["accepted_at"])
                            for m in images_out})
        manifest_out = {
            "dataset": dataset_name,
            "version": args.version,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "code_commit": commit,
            "split": "none (MLOps chia theo camera_id/sequence_id)",
            "source": {
                "note": dataset_cfg.get("source_note"),
                "sources": cfg.get("sources", []),
                "sampling": cfg["sampling"],
            },
            "labeling": {
                "classes": classes,
                "prelabel": {k: prelabel_prov.get(k) for k in
                             ("prelabel_model", "ultralytics_version", "params", "run_at")},
                "guideline_version": guideline_version,
                "guideline_sha256": sha256_of(guideline_path),
                "review": "người duyệt đặt PASS trên CVAT (stage=acceptance, state=completed)",
                "cvat_jobs": [{"task_id": t, "job_id": j, "accepted_at": a} for t, j, a in jobs_used],
            },
            "stats": {
                "num_images": len(images_out),
                "num_boxes": sum(cls_count.values()),
                "empty_images": sum(1 for m in images_out if not m["lines"]),
                "class_distribution": dict(cls_count),
                "images_per_camera": dict(cam_count),
                "images_per_weather": dict(weather_count),
            },
            "files_sha256": files_checksum,
        }
        (tmp / "MANIFEST.json").write_text(json.dumps(manifest_out, ensure_ascii=False, indent=2), encoding="utf-8")

        # --- tự kiểm lại bản vừa dựng ---
        n_img = len(list((tmp / "images").glob("*.jpg")))
        n_lbl = len(list((tmp / "labels").glob("*.txt")))
        if not (n_img == n_lbl == len(images_out)):
            raise RuntimeError(f"bản dựng lệch: {n_img} ảnh, {n_lbl} nhãn, kỳ vọng {len(images_out)}")

        # --- thay thế nguyên khối ---
        if args.out.exists():
            shutil.rmtree(args.out)
        tmp.rename(args.out)
    except Exception:
        shutil.rmtree(tmp, ignore_errors=True)
        raise

    s = manifest_out["stats"]
    log(f"Đã phát hành {dataset_name} v{args.version} -> {args.out.relative_to(REPO_ROOT)}/")
    log(f"  {s['num_images']} ảnh, {s['num_boxes']} box, {s['empty_images']} ảnh không có xe")
    log(f"  lớp: {s['class_distribution']}")
    log(f"  camera: {s['images_per_camera']}")
    print()
    log("Tiếp theo — đóng phiên bản bằng DVC:")
    print("    dvc add dataset")
    print(f"    git add dataset.dvc .gitignore && git commit -m \"dataset v{args.version}\"")
    print(f"    git tag dataset-v{args.version}")
    print("    dvc push")


if __name__ == "__main__":
    main()
