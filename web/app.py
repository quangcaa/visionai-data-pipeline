#!/usr/bin/env python3
"""Web điều khiển pipeline dữ liệu — không phải gõ lệnh nữa.

Chạy:
    .venv/bin/python -m web.app
    .venv/bin/python -m web.app --port 8000 --host 127.0.0.1

Web gọi lại đúng các script trong scripts/, không nhân bản logic pipeline.
Chỉ nghe trên 127.0.0.1 và KHÔNG có xác thực — đây là công cụ chạy trên máy
của bạn, đừng phơi ra mạng ngoài.
"""

from __future__ import annotations

import argparse
import json
import queue
import re
import shutil
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from web import config_io as cio                                  # noqa: E402
from web.runner import REPO_ROOT, Runner, script                  # noqa: E402

STATIC = Path(__file__).resolve().parent / "static"
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg"}
SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

app = FastAPI(title="Vision AI data pipeline")
runner = Runner()


# --- tiện ích ----------------------------------------------------------------

def _cfg() -> dict:
    from _common import load_yaml
    return load_yaml(cio.PIPELINE_YAML)


def _work_dir() -> Path:
    return REPO_ROOT / _cfg()["paths"]["work_dir"]


def _inbox() -> Path:
    d = REPO_ROOT / (_cfg().get("paths", {}).get("inbox") or "data/inbox")
    d.mkdir(parents=True, exist_ok=True)
    return d


def _service_up(url: str) -> bool:
    import urllib.request
    try:
        urllib.request.urlopen(url, timeout=2)
        return True
    except Exception:       # noqa: BLE001 — chỉ cần biết có lên hay không
        return False


def _bad(msg: str, code: int = 400):
    raise HTTPException(status_code=code, detail=msg)


# --- trạng thái tổng quan ----------------------------------------------------

@app.get("/api/state")
def api_state():
    cfg = _cfg()
    work = _work_dir()
    images = work / "images"
    prelabels = work / "prelabels"
    dataset = REPO_ROOT / "dataset"

    n_images = len(list(images.glob("*.jpg"))) if images.is_dir() else 0
    n_prelabels = len(list(prelabels.glob("*.txt"))) if prelabels.is_dir() else 0

    manifest = work / "manifest.jsonl"
    per_camera: dict[str, int] = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            if line.strip():
                cam = json.loads(line).get("camera_id", "?")
                per_camera[cam] = per_camera.get(cam, 0) + 1

    released = None
    man = dataset / "MANIFEST.json"
    if man.is_file():
        m = json.loads(man.read_text(encoding="utf-8"))
        released = {"version": m.get("version"), "stats": m.get("stats", {})}

    endpoint = cfg.get("storage", {}).get("endpoint", "localhost:9000")
    return {
        "services": {
            "minio": _service_up(f"http://{endpoint}/minio/health/live"),
            "cvat": _service_up("http://localhost:8080/api/server/about"),
        },
        "sources": cio.list_sources(),
        "batch": {
            "images": n_images,
            "prelabels": n_prelabels,
            "per_camera": per_camera,
            "package": (work / "prelabels_coco.zip").is_file(),
        },
        "released": released,
        "job": runner.current.as_dict() if runner.current else None,
        "sampling": cfg.get("sampling", {}),
        "dataset_name": (cfg.get("dataset") or {}).get("name", "dataset"),
    }


# --- inbox: chọn / tải video lên ---------------------------------------------

@app.get("/api/inbox")
def api_inbox():
    out = []
    for p in sorted(_inbox().iterdir()):
        if p.is_file():
            out.append({"name": p.name, "size": p.stat().st_size,
                        "is_video": p.suffix.lower() in VIDEO_EXT})
        elif p.is_dir():
            n = sum(1 for f in p.iterdir() if f.is_file())
            out.append({"name": p.name, "size": None, "is_video": False, "images": n})
    return {"dir": str(_inbox().relative_to(REPO_ROOT)), "entries": out}


@app.post("/api/inbox/upload")
async def api_upload(file: UploadFile = File(...)):
    raw = Path(file.filename or "video").name
    name = SAFE_NAME.sub("_", raw).strip("._") or "video"
    if Path(name).suffix.lower() not in VIDEO_EXT:
        _bad(f"chỉ nhận video ({', '.join(sorted(VIDEO_EXT))})")

    dst = _inbox() / name
    stem, suf = Path(name).stem, Path(name).suffix
    i = 1
    while dst.exists():
        dst = _inbox() / f"{stem}_{i}{suf}"
        i += 1

    size = 0
    try:
        with dst.open("wb") as f:
            while chunk := await file.read(1 << 20):
                size += len(chunk)
                f.write(chunk)
    except BaseException:
        dst.unlink(missing_ok=True)
        raise
    return {"name": dst.name, "size": size,
            "path": str(dst.relative_to(REPO_ROOT))}


@app.delete("/api/inbox/{name}")
def api_inbox_delete(name: str):
    p = _inbox() / Path(name).name
    if not p.exists():
        _bad("không thấy file", 404)
    if any(s.get("path") == str(p.relative_to(REPO_ROOT)) for s in cio.list_sources()):
        _bad("file đang được một nguồn dùng — xoá nguồn đó trước")
    p.unlink() if p.is_file() else shutil.rmtree(p)
    return {"ok": True}


# --- nguồn dữ liệu -----------------------------------------------------------

@app.get("/api/sources")
def api_sources():
    return {"sources": cio.list_sources()}


@app.post("/api/sources")
async def api_source_upsert(payload: dict):
    try:
        cio.upsert_source(
            camera_id=(payload.get("camera_id") or "").strip(),
            kind=payload.get("kind") or "images",
            path=payload.get("path") or "",
            scene=payload.get("scene") or {},
            sequence=(payload.get("sequence") or "").strip() or None,
        )
    except ValueError as exc:
        _bad(str(exc))
    return {"ok": True, "sources": cio.list_sources()}


@app.patch("/api/sampling")
async def api_sampling(payload: dict):
    try:
        return {"ok": True, "sampling": cio.update_sampling(payload or {})}
    except ValueError as exc:
        _bad(str(exc))


@app.delete("/api/sources/{camera_id}")
def api_source_delete(camera_id: str):
    """Xoá nguồn kèm vùng bỏ qua của nó — vùng vẽ cho nguồn đã xoá thì vô nghĩa."""
    res = cio.delete_source(camera_id)
    if res is None:
        _bad("không thấy nguồn", 404)
    return {"ok": True, **res, "sources": cio.list_sources()}


@app.get("/api/regions/orphans")
def api_region_orphans():
    return {"orphans": cio.orphan_regions()}


@app.delete("/api/regions/{camera_id}")
def api_region_delete(camera_id: str):
    return {"ok": True, "dropped": cio.drop_regions(camera_id)}


# --- lớp nhãn ----------------------------------------------------------------

_model_cache: dict = {}


def _model_classes() -> dict[int, str]:
    """Bảng {id: tên} model thực sự phân biệt, đọc thẳng từ file trọng số.

    Không đoán theo COCO: người dùng có thể thay model khác với bộ lớp khác.
    Nạp model tốn vài giây nên nhớ lại theo (đường dẫn, mtime).
    """
    from _common import REPO_ROOT, load_yaml

    model = str(load_yaml(cio.PRELABEL_YAML).get("model", ""))
    path = REPO_ROOT / model
    key = (model, path.stat().st_mtime if path.is_file() else None)
    if _model_cache.get("key") == key:
        return _model_cache["names"]

    names: dict[int, str] = {}
    try:
        from ultralytics import YOLO
        names = {int(k): str(v) for k, v in YOLO(str(path) if path.is_file()
                                                 else model).names.items()}
    except Exception as exc:                      # noqa: BLE001
        print(f"[web] không đọc được lớp của model {model}: {exc}", flush=True)
    _model_cache.update(key=key, names=names)
    return names


@app.get("/api/labels")
def api_labels():
    return cio.load_labels(_model_classes())


@app.put("/api/labels")
async def api_labels_save(payload: dict):
    try:
        return {"ok": True, **cio.save_labels(payload.get("labels") or [],
                                              payload.get("class_map") or [],
                                              _model_classes())}
    except ValueError as exc:
        _bad(str(exc))


# --- ảnh xem trước để khoanh vùng --------------------------------------------

_preview_cache: dict[tuple, tuple[float, bytes]] = {}


@app.get("/api/preview/{camera_id}")
def api_preview(camera_id: str, pos: float = 0.1):
    src = next((s for s in cio.list_sources() if s["camera_id"] == camera_id), None)
    if src is None:
        _bad("không thấy nguồn", 404)
    path = REPO_ROOT / src["path"]
    if not path.exists():
        _bad(f"không thấy {src['path']}", 404)

    pos = min(max(pos, 0.0), 1.0)
    key = (str(path), round(pos, 3))
    mtime = path.stat().st_mtime
    hit = _preview_cache.get(key)
    if hit and hit[0] == mtime:
        return Response(hit[1], media_type="image/jpeg")

    if src["kind"] == "video":
        data = _frame_from_video(path, pos)
    else:
        glob = _cfg().get("sampling", {}).get("image_glob", "*.jpg")
        files = sorted(path.glob(glob))
        if not files:
            _bad(f"thư mục không có file nào khớp '{glob}'", 404)
        data = files[min(int(pos * (len(files) - 1)), len(files) - 1)].read_bytes()

    if len(_preview_cache) > 32:
        _preview_cache.clear()
    _preview_cache[key] = (mtime, data)
    return Response(data, media_type="image/jpeg")


def _frame_from_video(path: Path, pos: float) -> bytes:
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        _bad("không mở được video — codec không đọc được?", 422)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total > 1:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(pos * (total - 1)))
    ok, frame = cap.read()
    if not ok:                       # seek hỏng với vài codec, đọc lại từ đầu
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, frame = cap.read()
    cap.release()
    if not ok:
        _bad("không đọc được frame nào từ video", 422)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if not ok:
        _bad("không mã hoá được ảnh", 500)
    return buf.tobytes()


# --- vùng bỏ qua -------------------------------------------------------------

@app.get("/api/regions")
def api_regions():
    return cio.load_regions()


@app.put("/api/regions/{camera_id}")
async def api_regions_save(camera_id: str, payload: dict):
    try:
        n = cio.save_regions(camera_id, payload.get("regions") or [], payload.get("note"))
    except ValueError as exc:
        _bad(str(exc))
    return {"ok": True, "count": n}


# --- chạy các bước -----------------------------------------------------------

STEPS = {
    "ingest":   "Nạp nguồn + lấy mẫu frame",
    "prelabel": "Gán nhãn sơ bộ YOLO26",
    "package":  "Đóng gói nhãn sơ bộ cho CVAT",
    "cvat":     "Tạo task CVAT",
    "release":  "Dựng dataset từ job đã PASS",
    "publish":  "Đóng phiên bản bằng DVC + git tag",
}


@app.post("/api/run/{step}")
async def api_run(step: str, payload: dict | None = None):
    payload = payload or {}
    if step not in STEPS:
        _bad(f"bước không hợp lệ: {step}", 404)
    if runner.busy():
        _bad(f"đang chạy '{runner.current.step}', đợi xong đã", 409)

    cameras = payload.get("cameras") or []
    if isinstance(cameras, str):
        cameras = [cameras]

    if step == "ingest":
        argv = script("00_ingest.py", "--force")
        if payload.get("upload"):
            argv.append("--upload")
        for cam in cameras:
            argv += ["--camera", str(cam)]

    elif step == "prelabel":
        argv = script("01_prelabel.py", "--force")

    elif step == "package":
        argv = script("02_prelabels_to_cvat.py")

    elif step == "cvat":
        argv = script("03_cvat_create_task.py")
        for cam in cameras:
            argv += ["--camera", str(cam)]
        if payload.get("task_name"):
            argv += ["--task-name", str(payload["task_name"])]

    elif step == "release":
        version = str(payload.get("version") or "").strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            _bad("phiên bản phải dạng MAJOR.MINOR.PATCH, ví dụ 1.0.0")
        argv = script("04_build_release.py", "--version", version)
        if payload.get("task"):
            argv += ["--task", str(int(payload["task"]))]
        if (REPO_ROOT / "dataset" / "MANIFEST.json").is_file():
            argv.append("--force")

    else:  # publish — dùng run_pipeline vì logic dvc remote nằm ở đó
        version = str(payload.get("version") or "").strip()
        if not re.fullmatch(r"\d+\.\d+\.\d+", version):
            _bad("phiên bản phải dạng MAJOR.MINOR.PATCH, ví dụ 1.0.0")
        argv = script("run_pipeline.py", "--from", "publish", "--to", "publish",
                      "--version", version)

    try:
        job = runner.start(step, argv, STEPS[step])
    except RuntimeError as exc:
        _bad(str(exc), 409)
    return job.as_dict()


@app.get("/api/job")
def api_job():
    return {
        "current": runner.current.as_dict() if runner.current else None,
        "history": [j.as_dict() for j in runner.history],
    }


@app.post("/api/job/cancel")
def api_job_cancel():
    if not runner.current or not runner.current.cancel():
        _bad("không có job nào đang chạy")
    return {"ok": True}


@app.get("/api/job/{job_id}/stream")
def api_job_stream(job_id: int):
    job = runner.get(job_id)
    if job is None:
        _bad("không thấy job", 404)

    def gen():
        q = job.subscribe()
        try:
            while True:
                try:
                    line = q.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"          # giữ kết nối qua proxy
                    continue
                if line is None:
                    yield f"event: done\ndata: {json.dumps(job.as_dict())}\n\n"
                    return
                yield f"data: {json.dumps(line)}\n\n"
        finally:
            job.unsubscribe(q)

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache",
                                      "X-Accel-Buffering": "no"})


# --- CVAT --------------------------------------------------------------------

@app.get("/api/cvat")
def api_cvat():
    try:
        from _common import cvat_client, cvat_config
        host, port, _, _ = cvat_config()
        with cvat_client() as c:
            tasks = []
            jobs_by_task: dict[int, list] = {}
            for j in c.jobs.list():
                jobs_by_task.setdefault(j.task_id, []).append({
                    "id": j.id, "stage": str(j.stage), "state": str(j.state),
                    "passed": str(j.stage) == "acceptance" and str(j.state) == "completed",
                    "frames": [j.start_frame, j.stop_frame],
                })
            for t in c.tasks.list():
                js = sorted(jobs_by_task.get(t.id, []), key=lambda x: x["id"])
                tasks.append({
                    "id": t.id, "name": t.name, "size": t.size, "jobs": js,
                    "passed": bool(js) and all(x["passed"] for x in js),
                    "url": f"{host}:{port}/tasks/{t.id}",
                })
            return {"ok": True, "base_url": f"{host}:{port}",
                    "tasks": sorted(tasks, key=lambda x: -x["id"])}
    except Exception as exc:                      # noqa: BLE001
        return {"ok": False, "error": f"{exc.__class__.__name__}: {exc}", "tasks": []}


@app.delete("/api/cvat/task/{task_id}")
def api_cvat_task_delete(task_id: int):
    """Xoá hẳn một task trên CVAT, kèm mọi nhãn người đã sửa trong đó."""
    try:
        from _common import cvat_client
        with cvat_client() as c:
            task = c.tasks.retrieve(task_id)
            name = task.name
            task.remove()
    except Exception as exc:                      # noqa: BLE001
        _bad(f"{exc.__class__.__name__}: {exc}", 502)

    # báo cáo trong repo cũng hết ý nghĩa khi task không còn
    report = REPO_ROOT / "reports" / f"cvat_task_{task_id}.json"
    report.unlink(missing_ok=True)
    return {"ok": True, "task_id": task_id, "name": name}


@app.post("/api/cvat/job/{job_id}/review")
async def api_cvat_review(job_id: int, payload: dict):
    """Đặt PASS/FAIL cho một job. Tiện, nhưng việc soát vẫn phải làm trong CVAT."""
    decision = payload.get("decision")
    if decision not in ("pass", "fail"):
        _bad("decision phải là 'pass' hoặc 'fail'")
    stage, state = (("acceptance", "completed") if decision == "pass"
                    else ("annotation", "rejected"))
    try:
        from _common import cvat_client
        from cvat_sdk.api_client import models
        with cvat_client() as c:
            job = c.jobs.retrieve(job_id)
            job.update(models.PatchedJobWriteRequest(stage=stage, state=state))
    except Exception as exc:                      # noqa: BLE001
        _bad(f"{exc.__class__.__name__}: {exc}", 502)
    return {"ok": True, "stage": stage, "state": state}


# --- ảnh minh hoạ vùng bỏ qua + file tĩnh ------------------------------------

@app.get("/api/health")
def api_health():
    return {"ok": True, "repo": str(REPO_ROOT), "time": time.time()}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--reload", action="store_true")
    args = ap.parse_args()

    import uvicorn
    print(f"[web] Mở http://{args.host}:{args.port}", flush=True)
    uvicorn.run("web.app:app" if args.reload else app,
                host=args.host, port=args.port, reload=args.reload,
                log_level="warning")


if __name__ == "__main__":
    main()
