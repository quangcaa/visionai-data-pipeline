"""Helper dùng chung cho mọi script trong scripts/. Không chạy trực tiếp.

Các script tự `from _common import ...` để dùng. KHÔNG thêm CLI / argparse ở đây.
"""

from __future__ import annotations

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


def log(tag: str, msg: str) -> None:
    """In ra stdout, flush ngay (CI / log file cần thấy ngay)."""
    print(f"[{tag}] {msg}", flush=True)


def die(tag: str, msg: str) -> None:
    """In ra stderr rồi thoát mã 1."""
    print(f"[{tag}] LỖI: {msg}", file=sys.stderr, flush=True)
    sys.exit(1)


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


def add_io_args(ap) -> None:
    """Thêm 4 flag chuẩn: --config, --labels, --prelabel-config, --eval-config."""
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "configs" / "labels.json")
    ap.add_argument("--prelabel-config", type=Path, default=REPO_ROOT / "configs" / "prelabel.yaml")
    ap.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs" / "eval.yaml")


def add_io_args(ap) -> None:
    """Thêm 4 flag chuẩn: --config, --labels, --prelabel-config, --eval-config."""
    ap.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "pipeline.yaml")
    ap.add_argument("--labels", type=Path, default=REPO_ROOT / "configs" / "labels.json")
    ap.add_argument("--prelabel-config", type=Path, default=REPO_ROOT / "configs" / "prelabel.yaml")
    ap.add_argument("--eval-config", type=Path, default=REPO_ROOT / "configs" / "eval.yaml")


if __name__ == "__main__":
    die("common", "không chạy trực tiếp — đây là module helper")
