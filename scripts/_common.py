"""Helper dùng chung cho mọi script trong scripts/. Không chạy trực tiếp.

Các script tự `from _common import ...` để dùng. KHÔNG thêm CLI / argparse ở đây.
"""

from __future__ import annotations

import functools
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


def load_dotenv(path: Path | None = None) -> None:
    """Nạp các biến KAGGLE_* / MINIO_* / CVAT_* từ file .env.

    Chỉ `os.environ.setdefault` — không ghi đè biến đã có.
    """
    path = path or (REPO_ROOT / ".env")
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip("'\"")
        if value and key not in os.environ:
            os.environ[key] = value


# --- in ra màn hình ----------------------------------------------------------

def log(tag: str, msg: str) -> None:
    """In ra stdout, flush ngay (CI / log file cần thấy ngay)."""
    print(f"[{tag}] {msg}", flush=True)


def die(tag: str, msg: str) -> None:
    """In ra stderr rồi thoát mã 1."""
    print(f"[{tag}] LỖI: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


def logger(tag: str):
    """Trả về cặp (log, die) đã gắn sẵn tag của script.

    Dùng ở đầu mỗi script:  log, die = logger("sample")
    """
    return functools.partial(log, tag), functools.partial(die, tag)


# --- đọc file ----------------------------------------------------------------

def sha256_of(path: Path) -> str:
    """SHA-256 của file, đọc theo chunk 1 MiB."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_label_spec(path: Path) -> list[str]:
    """Đọc configs/labels.json -> danh sách tên lớp theo thứ tự."""
    return [item["name"] for item in load_json(path)]


def read_manifest(work_dir: Path) -> list[dict]:
    """Đọc manifest.jsonl của lô hiện tại -> danh sách record theo thứ tự ghi."""
    path = work_dir / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_ignore_regions(cfg: dict) -> dict[str, np.ndarray]:
    """Đọc file vùng bỏ qua khai báo tay -> {camera_id: mảng (N,4) toạ độ chuẩn hoá 0..1}.

    Đường dẫn file lấy từ `ignore_regions.file` trong pipeline.yaml. Camera không có
    mục trong file = không có vùng bỏ qua nào. File thiếu cũng không sao — coi như rỗng.
    """
    spec = cfg.get("ignore_regions") or {}
    path = REPO_ROOT / spec.get("file", "configs/ignore_regions.json")
    if not path.is_file():
        return {}
    out: dict[str, np.ndarray] = {}
    for camera_id, entry in (load_json(path).get("cameras") or {}).items():
        regions = (entry or {}).get("regions") or []
        if not regions:
            continue
        out[camera_id] = np.clip(
            np.array([[r["x1"], r["y1"], r["x2"], r["y2"]] for r in regions], dtype=float), 0.0, 1.0)
    return out


# --- hộp giới hạn ------------------------------------------------------------

def load_yolo(path: Path, n_cols: int) -> np.ndarray:
    """Đọc file .txt định dạng YOLO. Trả về mảng (N, n_cols) hoặc zeros nếu rỗng."""
    if not path.is_file() or path.stat().st_size == 0:
        return np.zeros((0, n_cols))
    return np.loadtxt(path, ndmin=2)


def yolo_to_xyxy(arr: np.ndarray, w: int, h: int) -> np.ndarray:
    """cx,cy,w,h chuẩn hoá -> x1,y1,x2,y2 pixel."""
    if arr.size == 0:
        return np.zeros((0, 4))
    cx, cy, bw, bh = arr[:, 0] * w, arr[:, 1] * h, arr[:, 2] * w, arr[:, 3] * h
    return np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], axis=1)


def ignore_regions_xyxy(regions_norm: np.ndarray | None, w: int, h: int) -> np.ndarray:
    """Vùng bỏ qua chuẩn hoá 0..1 -> toạ độ pixel x1,y1,x2,y2 của ảnh w x h."""
    if regions_norm is None or len(regions_norm) == 0:
        return np.zeros((0, 4))
    return regions_norm * np.array([w, h, w, h], dtype=float)


def boxes_in_ignored(pred_xyxy: np.ndarray, regions_xyxy: np.ndarray, thr: float) -> np.ndarray:
    """True nếu >= thr diện tích box dự đoán nằm trong một vùng bỏ qua."""
    if pred_xyxy.size == 0 or regions_xyxy.size == 0:
        return np.zeros(len(pred_xyxy), dtype=bool)
    x1 = np.maximum(pred_xyxy[:, None, 0], regions_xyxy[None, :, 0])
    y1 = np.maximum(pred_xyxy[:, None, 1], regions_xyxy[None, :, 1])
    x2 = np.minimum(pred_xyxy[:, None, 2], regions_xyxy[None, :, 2])
    y2 = np.minimum(pred_xyxy[:, None, 3], regions_xyxy[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = np.clip((pred_xyxy[:, 2] - pred_xyxy[:, 0]) * (pred_xyxy[:, 3] - pred_xyxy[:, 1]), 1e-9, None)
    return (inter / area[:, None]).max(axis=1) >= thr


# --- CVAT --------------------------------------------------------------------

def cvat_config() -> tuple[str, int, str, str]:
    """Đọc CVAT_* từ .env / môi trường -> (host, port, user, password).

    cvat-sdk mặc định dùng https khi host không có scheme; CVAT local chạy http thường.
    Ném RuntimeError nếu thiếu tài khoản — nơi gọi tự quyết định báo lỗi thế nào.
    """
    load_dotenv()
    host = os.environ.get("CVAT_HOST", "localhost")
    if "://" not in host:
        host = f"http://{host}"
    port = int(os.environ.get("CVAT_PORT", 8080))
    user, password = os.environ.get("CVAT_USER"), os.environ.get("CVAT_PASSWORD")
    if not user or not password:
        raise RuntimeError("thiếu CVAT_USER / CVAT_PASSWORD trong .env "
                           "(tài khoản tạo bằng createsuperuser)")
    return host, port, user, password


def cvat_client():
    """Context manager client CVAT đã đăng nhập. Dùng: `with cvat_client() as c:`"""
    from cvat_sdk import make_client

    host, port, user, password = cvat_config()
    return make_client(host=host, port=port, credentials=(user, password))


# --- CLI ---------------------------------------------------------------------

_IO_ARGS = {
    "config": ("--config", "configs/pipeline.yaml"),
    "labels": ("--labels", "configs/labels.json"),
    "prelabel-config": ("--prelabel-config", "configs/prelabel.yaml"),
    "ignore-regions": ("--ignore-regions", "configs/ignore_regions.json"),
}


def add_io_args(ap, *names: str) -> None:
    """Thêm các flag trỏ tới file cấu hình, mỗi script chỉ khai báo cái mình dùng.

        add_io_args(ap, "config", "labels")
    """
    for name in names:
        flag, default = _IO_ARGS[name]
        ap.add_argument(flag, type=Path, default=REPO_ROOT / default)


if __name__ == "__main__":
    die("common", "không chạy trực tiếp — đây là module helper")
