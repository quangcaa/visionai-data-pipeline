#!/usr/bin/env python3
"""Chạy pipeline dữ liệu end-to-end bằng một lệnh.

    [tự động] download -> sample -> gt -> prelabel -> eval -> package -> cvat
    [chờ]     review : theo dõi CVAT tới khi MỌI job của task được duyệt PASS
    [tự động] release -> publish (dvc add, git commit + tag, dvc push)

Bước review là con người (docx mục 5-6), script chỉ chờ chứ không thay thế.
Job bị FAIL (annotation/rejected) thì script tiếp tục chờ tới khi được sửa và PASS.

Dùng:
    python scripts/run_pipeline.py --version 1.0.0                  # chạy hết
    python scripts/run_pipeline.py --version 1.0.0 --to cvat        # dừng sau khi tạo task
    python scripts/run_pipeline.py --version 1.0.0 --from review --task 4   # chạy tiếp
    python scripts/run_pipeline.py --version 1.0.0 --from release --task 4  # đã duyệt xong
    python scripts/run_pipeline.py --list                           # xem các bước

Mặc định KHÔNG push Git lên GitHub; thêm --git-push nếu muốn.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from _common import REPO_ROOT, load_dotenv, load_yaml
SCRIPTS = REPO_ROOT / "scripts"
PY = sys.executable

STEPS = ["download", "sample", "gt", "prelabel", "eval", "package", "cvat", "review", "release", "publish"]
STEP_HELP = {
    "download": "00 tải UA-DETRAC (bỏ qua nếu đã có)",
    "sample":   "01 lấy mẫu frame + upload MinIO",
    "gt":       "02 GT từ annotation DETRAC (thước đo)",
    "prelabel": "03 gán nhãn sơ bộ YOLO26",
    "eval":     "04 đánh giá nhãn sơ bộ",
    "package":  "05 đóng gói nhãn sơ bộ cho CVAT",
    "cvat":     "06 tạo task CVAT + nạp nhãn sơ bộ",
    "review":   "chờ người sửa + duyệt PASS trên CVAT",
    "release":  "07 dựng dataset/ từ job đã PASS",
    "publish":  "dvc add + git commit/tag + dvc push",
}


def log(msg: str) -> None:
    print(f"\n[pipeline] {msg}", flush=True)


def die(msg: str) -> None:
    print(f"\n[pipeline] LỖI: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def run(cmd: list[str], *, check: bool = True, capture: bool = False,
        secret: str | None = None) -> subprocess.CompletedProcess:
    shown = " ".join(str(c) for c in cmd).replace(str(REPO_ROOT) + "/", "")
    if secret:
        shown = shown.replace(secret, "****")
    print(f"  $ {shown}", flush=True)
    res = subprocess.run([str(c) for c in cmd], cwd=REPO_ROOT, text=True,
                         capture_output=capture)
    if check and res.returncode != 0:
        if capture:
            out = f"{res.stdout}\n{res.stderr}"
            print(out.replace(secret, "****") if secret else out, file=sys.stderr)
        die(f"lệnh thất bại (mã {res.returncode})")
    return res


def script(name: str, *args) -> None:
    run([PY, SCRIPTS / name, *args])


def cvat_client():
    from cvat_sdk import make_client
    host = os.environ.get("CVAT_HOST", "localhost")
    if "://" not in host:
        host = f"http://{host}"
    user, pwd = os.environ.get("CVAT_USER"), os.environ.get("CVAT_PASSWORD")
    if not user or not pwd:
        die("thiếu CVAT_USER / CVAT_PASSWORD trong .env")
    return make_client(host=host, port=int(os.environ.get("CVAT_PORT", 8080)), credentials=(user, pwd))


# --- kiểm tra trước khi chạy ------------------------------------------------

def preflight(steps: list[str], version: str | None) -> None:
    problems = []
    need_cvat = {"cvat", "review", "release"} & set(steps)
    if need_cvat:
        try:
            with cvat_client():
                pass
        except SystemExit:
            raise
        except Exception as exc:  # noqa: BLE001
            problems.append(f"không kết nối được CVAT ({exc.__class__.__name__}) — CVAT đã chạy chưa?")
    if "sample" in steps:
        import urllib.request
        cfg = yaml.safe_load((REPO_ROOT / "configs/pipeline.yaml").read_text(encoding="utf-8"))
        url = f"http://{cfg['storage']['endpoint']}/minio/health/live"
        try:
            urllib.request.urlopen(url, timeout=5)
        except Exception:  # noqa: BLE001
            problems.append(f"MinIO không phản hồi ở {url} — đã `docker compose up -d` chưa?")
    if {"release", "publish"} & set(steps):
        if not version:
            problems.append("bước release/publish cần --version, ví dụ --version 1.0.0")
        elif run(["git", "rev-parse", "-q", "--verify", f"refs/tags/dataset-v{version}"],
                 check=False, capture=True).returncode == 0:
            problems.append(f"tag dataset-v{version} đã tồn tại — phiên bản đã phát hành không được ghi đè, "
                            f"hãy dùng số phiên bản mới")
    if problems:
        die("chưa chạy được:\n  - " + "\n  - ".join(problems))


# --- các bước -----------------------------------------------------------------

def step_cvat(cfg: dict) -> int:
    reports = REPO_ROOT / "reports"
    before = set(reports.glob("cvat_task_*.json")) if reports.is_dir() else set()
    c = cfg["cvat"]
    args = ["--project-name", c["project_name"], "--segment-size", str(c["segment_size"])]
    if c.get("camera"):
        args += ["--camera", c["camera"]]
    script("06_cvat_create_task.py", *args)
    new = sorted(set(reports.glob("cvat_task_*.json")) - before)
    if not new:
        die("không xác định được task vừa tạo (không thấy reports/cvat_task_<id>.json mới)")
    task_id = json.loads(new[-1].read_text(encoding="utf-8"))["task_id"]
    log(f"Đã tạo task {task_id}")
    return task_id


def step_review(cfg: dict, task_id: int) -> None:
    host = os.environ.get("CVAT_HOST", "localhost")
    host = host if "://" in host else f"http://{host}"
    port = os.environ.get("CVAT_PORT", 8080)
    poll = int(cfg["review"]["poll_seconds"])
    timeout = int(cfg["review"]["timeout_minutes"]) * 60
    log(f"Chờ duyệt task {task_id}: {host}:{port}/tasks/{task_id}\n"
        f"  Người sửa làm theo docs/labeling_guideline.md; người duyệt đặt mỗi job\n"
        f"  PASS = stage acceptance + state completed. Ctrl+C để dừng, chạy tiếp bằng\n"
        f"  --from review --task {task_id}")
    started, last = time.time(), None
    while True:
        with cvat_client() as client:
            jobs = [j for j in client.jobs.list() if j.task_id == task_id]
        if not jobs:
            die(f"task {task_id} không có job nào (đã bị xoá?)")
        status = tuple(sorted((j.id, str(j.stage), str(j.state)) for j in jobs))
        passed = sum(1 for _, st, s in status if st == "acceptance" and s == "completed")
        if status != last:
            print(f"  [{time.strftime('%H:%M:%S')}] {passed}/{len(jobs)} job PASS: "
                  + ", ".join(f"job {i} {st}/{s}" for i, st, s in status), flush=True)
            last = status
        if passed == len(jobs):
            log("Mọi job đã được duyệt PASS")
            return
        if timeout and time.time() - started > timeout:
            die(f"hết thời gian chờ ({cfg['review']['timeout_minutes']} phút)")
        time.sleep(poll)


def step_release(version: str, task_id: int | None) -> None:
    args = ["--version", version]
    if task_id is not None:
        args += ["--task", str(task_id)]
    manifest = REPO_ROOT / "dataset" / "MANIFEST.json"
    if manifest.is_file():
        old = json.loads(manifest.read_text(encoding="utf-8")).get("version")
        log(f"dataset/ đang là bản {old}, sẽ được thay bằng {version} (bản cũ lấy lại qua git tag + dvc)")
        args.append("--force")
    script("07_build_release.py", *args)


def ensure_dvc(cfg: dict) -> None:
    d = cfg["dvc"]
    dvc = [PY, "-m", "dvc"]
    if not (REPO_ROOT / ".dvc").is_dir():
        run([*dvc, "init", "-q"])

    remotes = run([*dvc, "remote", "list"], capture=True).stdout
    if not re.search(rf"^{re.escape(d['remote_name'])}\s", remotes, flags=re.M):
        run([*dvc, "remote", "add", "-d", d["remote_name"], d["remote_url"]])

    if d["remote_url"].startswith("s3://"):
        user, pwd = os.environ.get("MINIO_ROOT_USER"), os.environ.get("MINIO_ROOT_PASSWORD")
        if not user or not pwd:
            die("thiếu MINIO_ROOT_USER / MINIO_ROOT_PASSWORD trong .env cho DVC remote")
        run([*dvc, "remote", "modify", d["remote_name"], "endpointurl", d["endpoint_url"]])
        # --local: credential nằm trong .dvc/config.local, không vào Git
        run([*dvc, "remote", "modify", "--local", d["remote_name"], "access_key_id", user],
            capture=True, secret=user)
        run([*dvc, "remote", "modify", "--local", d["remote_name"], "secret_access_key", pwd],
            capture=True, secret=pwd)

        from minio import Minio
        endpoint = d["endpoint_url"].split("://", 1)[1]
        bucket = d["remote_url"][len("s3://"):].split("/", 1)[0]
        client = Minio(endpoint, access_key=user, secret_key=pwd,
                       secure=d["endpoint_url"].startswith("https"))
        if not client.bucket_exists(bucket):
            client.make_bucket(bucket)
            log(f"Đã tạo bucket MinIO '{bucket}'")


def step_publish(cfg: dict, version: str, git_push: bool) -> None:
    if not (REPO_ROOT / "dataset" / "MANIFEST.json").is_file():
        die("chưa có dataset/ để phát hành — chạy bước release trước")
    ensure_dvc(cfg)
    dvc = [PY, "-m", "dvc"]
    run([*dvc, "add", "dataset"])

    to_commit = ["dataset.dvc", ".gitignore", ".dvc/config", ".dvc/.gitignore", ".dvcignore"]
    to_commit = [p for p in to_commit if (REPO_ROOT / p).exists()]
    run(["git", "add", "--", *to_commit])
    if run(["git", "diff", "--cached", "--quiet", "--", *to_commit], check=False).returncode != 0:
        run(["git", "commit", "-m", f"dataset v{version}", "--", *to_commit])
    else:
        log("dataset.dvc không đổi so với commit trước — không tạo commit mới")
    run(["git", "tag", "-a", f"dataset-v{version}", "-m", f"dataset v{version}"])
    run([*dvc, "push"])

    if git_push:
        run(["git", "push"])
        run(["git", "push", "origin", f"dataset-v{version}"])
    else:
        log(f"Chưa push Git. Khi sẵn sàng:\n  git push && git push origin dataset-v{version}")
    log(f"Đã phát hành dataset v{version}. Lấy lại ở máy khác:\n"
        f"  git clone <repo> && git checkout dataset-v{version} && dvc pull")


# --- điều phối ------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--version", help="phiên bản dataset phát hành, dạng MAJOR.MINOR.PATCH")
    ap.add_argument("--from", dest="start", choices=STEPS, default=STEPS[0])
    ap.add_argument("--to", dest="end", choices=STEPS, default=STEPS[-1])
    ap.add_argument("--task", type=int, help="id task CVAT (bắt buộc khi bắt đầu từ review/release)")
    ap.add_argument("--git-push", action="store_true", help="push commit + tag lên remote Git sau khi phát hành")
    ap.add_argument("--list", action="store_true", help="liệt kê các bước rồi thoát")
    args = ap.parse_args()

    if args.list:
        for i, s in enumerate(STEPS, 1):
            print(f"  {i:>2}. {s:<9} {STEP_HELP[s]}")
        return

    steps = STEPS[STEPS.index(args.start): STEPS.index(args.end) + 1]
    if not steps:
        die("--from đứng sau --to")
    if args.version and not re.fullmatch(r"\d+\.\d+\.\d+", args.version):
        die("--version phải có dạng MAJOR.MINOR.PATCH")
    if {"review", "release"} & set(steps) and "cvat" not in steps and args.task is None:
        die(f"bắt đầu từ '{args.start}' thì phải chỉ rõ --task <id>")

    load_dotenv()
    cfg = load_yaml(REPO_ROOT / "configs" / "pipeline.yaml")
    log("Các bước sẽ chạy: " + " -> ".join(steps))
    preflight(steps, args.version)

    task_id = args.task
    t0 = time.time()
    for s in steps:
        log(f"=== {s}: {STEP_HELP[s]} ===")
        if s == "download":
            script("00_download_detrac.py")
        elif s == "sample":
            script("01_sample_frames.py", "--force", "--upload")
        elif s == "gt":
            script("02_xml_to_yolo_gt.py")
        elif s == "prelabel":
            script("03_prelabel_yolo26.py", "--force")
        elif s == "eval":
            script("04_eval_prelabel.py")
        elif s == "package":
            script("05_prelabels_to_cvat.py")
        elif s == "cvat":
            task_id = step_cvat(cfg)
        elif s == "review":
            step_review(cfg, task_id)
        elif s == "release":
            step_release(args.version, task_id)
        elif s == "publish":
            step_publish(cfg, args.version, args.git_push)

    log(f"Hoàn tất {len(steps)} bước sau {time.time() - t0:.0f}s")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[pipeline] Đã dừng. Chạy tiếp bằng --from <bước>.", file=sys.stderr)
        sys.exit(130)
