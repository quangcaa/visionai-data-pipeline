"""Đọc/ghi configs/ cho web, giữ nguyên comment.

configs/pipeline.yaml đầy chú thích giải thích từng tham số. Dump bằng PyYAML là
mất sạch, nên ở đây dùng ruamel.yaml chế độ round-trip.

Ghi file luôn qua thư mục tạm rồi os.replace, để không bao giờ để lại file dở
dang nếu tiến trình chết giữa chừng.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path

from ruamel.yaml import YAML
from ruamel.yaml.comments import CommentedMap

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIGS = REPO_ROOT / "configs"
PIPELINE_YAML = CONFIGS / "pipeline.yaml"
IGNORE_JSON = CONFIGS / "ignore_regions.json"
LABELS_JSON = CONFIGS / "labels.json"
PRELABEL_YAML = CONFIGS / "prelabel.yaml"

CAMERA_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,31}$")
LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9 _-]{0,31}$")

# Dự phòng khi không nạp được model để đọc tên lớp thật.
COCO_NAMES = {
    0: "person", 1: "bicycle", 2: "car", 3: "motorcycle", 5: "bus",
    6: "train", 7: "truck", 9: "traffic light", 11: "stop sign",
}

_yaml = YAML()
_yaml.preserve_quotes = True
_yaml.width = 4096          # đừng tự ngắt dòng các comment dài
_yaml.indent(mapping=2, sequence=4, offset=2)

# Mặc định ruamel ghi None thành dòng trống; giữ chữ "null" để sửa một nguồn
# không làm bẩn diff ở những dòng người dùng không đụng tới.
_yaml.representer.add_representer(
    type(None),
    lambda rep, data: rep.represent_scalar("tag:yaml.org,2002:null", "null"),
)


def _atomic_write(path: Path, write) -> None:
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            write(f)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


# --- pipeline.yaml -----------------------------------------------------------

def load_pipeline() -> CommentedMap:
    return _yaml.load(PIPELINE_YAML.read_text(encoding="utf-8"))


def save_pipeline(cfg: CommentedMap) -> None:
    _atomic_write(PIPELINE_YAML, lambda f: _yaml.dump(cfg, f))


def list_sources() -> list[dict]:
    cfg = load_pipeline()
    out = []
    for s in cfg.get("sources") or []:
        src = {k: (dict(v) if k == "scene" and v else v) for k, v in dict(s).items()}
        path = REPO_ROOT / src.get("path", "")
        src["exists"] = path.exists()
        src.setdefault("kind", "images")
        src["sequence_id"] = src.get("sequence") or path.stem
        out.append(src)

    regions = load_regions()["cameras"]
    for src in out:
        entry = regions.get(src["camera_id"]) or {}
        drawn_for = entry.get("source")
        src["regions"] = len(entry.get("regions") or [])
        # vùng vẽ cho đường dẫn khác => gần như chắc chắn là của dữ liệu cũ
        src["regions_stale"] = bool(src["regions"]) and drawn_for not in (None, src["path"])
        src["regions_source"] = drawn_for
    return out


def upsert_source(camera_id: str, kind: str, path: str, scene: dict | None,
                  sequence: str | None = None) -> None:
    """Thêm nguồn mới hoặc sửa nguồn đã có (khớp theo camera_id)."""
    if not CAMERA_ID_RE.match(camera_id):
        raise ValueError("camera_id chỉ được gồm chữ, số, '-' và '_' (tối đa 32 ký tự)")
    if kind not in ("video", "images"):
        raise ValueError("kind phải là 'video' hoặc 'images'")

    rel = _safe_repo_relative(path)
    if not (REPO_ROOT / rel).exists():
        raise ValueError(f"không thấy đường dẫn {rel}")

    cfg = load_pipeline()
    if "sources" not in cfg or cfg["sources"] is None:
        cfg["sources"] = []

    entry = CommentedMap()
    entry["camera_id"] = camera_id
    entry["kind"] = kind
    entry["path"] = rel
    if sequence:
        entry["sequence"] = sequence
    if scene:
        sc = CommentedMap()
        for k, v in scene.items():
            if v not in (None, ""):
                sc[k] = v
        if sc:
            sc.fa.set_flow_style()      # giữ kiểu {weather: sunny} một dòng
            entry["scene"] = sc

    for i, s in enumerate(cfg["sources"]):
        if s.get("camera_id") == camera_id:
            cfg["sources"][i] = entry
            break
    else:
        cfg["sources"].append(entry)

    save_pipeline(cfg)


SAMPLING_FIELDS = {
    # tên -> (kiểu, nhỏ nhất, mô tả khi báo lỗi)
    "sample_every_n":  (int,   1, "lấy 1 frame mỗi N frame, N >= 1"),
    "max_per_camera":  (int,   0, "giới hạn mỗi camera, 0 = không giới hạn"),
    "source_fps":      (float, 0.1, "fps giả định cho nguồn ảnh, > 0"),
}


def update_sampling(values: dict) -> dict:
    """Ghi lại mục `sampling` trong pipeline.yaml. Chỉ đụng các khoá được gửi lên."""
    cfg = load_pipeline()
    sampling = cfg.get("sampling")
    if sampling is None:
        raise ValueError("pipeline.yaml chưa có mục sampling")

    for key, raw in values.items():
        if key == "image_glob":
            glob = str(raw).strip()
            if not glob or "/" in glob or ".." in glob:
                raise ValueError("image_glob phải là một mẫu tên file, vd *.jpg")
            sampling[key] = glob
            continue
        if key not in SAMPLING_FIELDS:
            raise ValueError(f"tham số không sửa được: {key}")
        cast, lo, why = SAMPLING_FIELDS[key]
        try:
            v = cast(raw)
        except (TypeError, ValueError):
            raise ValueError(f"{key}: {why}") from None
        if v < lo:
            raise ValueError(f"{key}: {why}")
        # 25.0 ghi lại thành 25 — đừng biến số nguyên thành số thực trong file
        sampling[key] = int(v) if cast is float and v == int(v) else v

    save_pipeline(cfg)
    return dict(sampling)


def delete_source(camera_id: str) -> dict | None:
    """Xoá nguồn và vùng bỏ qua của nó.

    Vùng luôn bị xoá theo: nó được vẽ cho đúng nguồn này, nên giữ lại chỉ tạo mục
    mồ côi mà camera_id sau này dùng lại sẽ âm thầm thừa kế.
    """
    cfg = load_pipeline()
    sources = cfg.get("sources") or []
    for i, s in enumerate(sources):
        if s.get("camera_id") == camera_id:
            del sources[i]
            save_pipeline(cfg)
            break
    else:
        return None

    return {"camera_id": camera_id, "regions_dropped": drop_regions(camera_id)}


def orphan_regions() -> list[str]:
    """Camera còn vùng bỏ qua nhưng không còn nguồn nào khai báo."""
    have = {s["camera_id"] for s in list_sources()}
    return sorted(k for k, v in load_regions()["cameras"].items()
                  if k not in have and (v or {}).get("regions"))


def drop_regions(camera_id: str) -> int:
    data = load_regions()
    entry = data["cameras"].pop(camera_id, None)
    if entry is None:
        return 0
    _atomic_write(IGNORE_JSON, lambda f: json.dump(data, f, ensure_ascii=False, indent=2))
    return len(entry.get("regions") or [])


def _safe_repo_relative(path: str) -> str:
    """Chuyển đường dẫn về dạng tương đối gốc repo, chặn thoát ra ngoài repo."""
    p = Path(path)
    abs_p = (p if p.is_absolute() else REPO_ROOT / p).resolve()
    try:
        return str(abs_p.relative_to(REPO_ROOT.resolve()))
    except ValueError:
        raise ValueError("đường dẫn phải nằm trong repo") from None


# --- ignore_regions.json -----------------------------------------------------

def load_regions() -> dict:
    if not IGNORE_JSON.is_file():
        return {"cameras": {}}
    data = json.loads(IGNORE_JSON.read_text(encoding="utf-8"))
    data.setdefault("cameras", {})
    return data


def save_regions(camera_id: str, regions: list[dict], note: str | None = None) -> int:
    """Ghi vùng bỏ qua của một camera. Toạ độ chuẩn hoá 0..1, tự kẹp về biên.

    Ghi kèm đường dẫn nguồn đang dùng: `camera_id` có thể được dùng lại cho dữ
    liệu khác, và vùng vẽ cho camera này thì vô nghĩa với camera khác.
    """
    data = load_regions()
    clean = []
    for r in regions:
        try:
            x1, y1 = float(r["x1"]), float(r["y1"])
            x2, y2 = float(r["x2"]), float(r["y2"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("mỗi vùng phải có x1, y1, x2, y2 dạng số") from None
        x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
        y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
        if x2 - x1 < 1e-4 or y2 - y1 < 1e-4:
            continue                    # vùng bé như một cú click, bỏ
        clean.append({"x1": round(x1, 5), "y1": round(y1, 5),
                      "x2": round(x2, 5), "y2": round(y2, 5)})

    if clean or camera_id in data["cameras"]:
        src = next((x for x in list_sources() if x["camera_id"] == camera_id), None)
        entry = data["cameras"].get(camera_id, {})
        entry["note"] = note if note is not None else entry.get("note", "")
        entry["source"] = src["path"] if src else None
        entry["regions"] = clean
        data["cameras"][camera_id] = entry

    _atomic_write(IGNORE_JSON,
                  lambda f: json.dump(data, f, ensure_ascii=False, indent=2))
    return len(clean)


# --- labels.json + class_map trong prelabel.yaml ------------------------------

def load_labels(model_classes: dict | None = None) -> dict:
    """Danh sách lớp + ánh xạ. Hai thứ này phải khớp nhau nên trả về cùng lúc.

    `model_classes` là bảng {id: tên} đọc thật từ model; thiếu thì dùng bảng COCO
    rút gọn làm dự phòng.
    """
    known = model_classes or COCO_NAMES
    spec = json.loads(LABELS_JSON.read_text(encoding="utf-8"))
    class_map = (_yaml.load(PRELABEL_YAML.read_text(encoding="utf-8"))
                 .get("class_map") or {})
    names = [x["name"] for x in spec]
    return {
        "labels": [{"name": x["name"], "type": x.get("type", "rectangle")} for x in spec],
        "class_map": [{"coco_id": int(k), "coco_name": known.get(int(k)),
                       "label": v, "valid": v in names,
                       "in_model": int(k) in known}
                      for k, v in class_map.items()],
        "model_classes": {str(k): v for k, v in sorted(known.items())},
        "model_known": bool(model_classes),
    }


def save_labels(names: list, class_map: list, model_classes: dict | None = None) -> dict:
    """Ghi lại labels.json và class_map. Từ chối nếu hai bên không khớp.

    class_id trong nhãn YOLO chính là chỉ số trong danh sách này, nên đổi thứ tự
    hay xoá lớp sẽ làm lệch mọi nhãn đã sinh trước đó — nơi gọi phải cảnh báo.
    """
    clean: list[str] = []
    for raw in names:
        name = str(raw).strip()
        if not LABEL_RE.match(name):
            raise ValueError(f"tên lớp không hợp lệ: {name!r} "
                             "(chữ, số, khoảng trắng, '-' và '_', tối đa 32 ký tự)")
        if name in clean:
            raise ValueError(f"tên lớp bị trùng: {name!r}")
        clean.append(name)
    if not clean:
        raise ValueError("phải có ít nhất một lớp")

    pairs: dict[int, str] = {}
    for item in class_map or []:
        try:
            coco_id = int(item["coco_id"])
        except (KeyError, TypeError, ValueError):
            raise ValueError("mỗi dòng ánh xạ cần coco_id là số nguyên") from None
        if not 0 <= coco_id <= 999:
            raise ValueError(f"coco_id ngoài khoảng: {coco_id}")
        label = str(item.get("label", "")).strip()
        if label not in clean:
            raise ValueError(f"ánh xạ COCO {coco_id} trỏ tới lớp '{label}' không có "
                             "trong danh sách")
        pairs[coco_id] = label

    # giữ nguyên type/attributes của lớp đã có, lớp mới mặc định rectangle
    old = {x["name"]: x for x in json.loads(LABELS_JSON.read_text(encoding="utf-8"))}
    spec = [old.get(n, {"name": n, "type": "rectangle", "attributes": []}) for n in clean]
    for x in spec:
        x.setdefault("type", "rectangle")
        x.setdefault("attributes", [])

    # một lớp một dòng — file này người ta còn sửa tay, đừng bung ra 6 dòng mỗi lớp
    body = ",\n".join("  " + json.dumps(x, ensure_ascii=False) for x in spec)
    _atomic_write(LABELS_JSON, lambda f: f.write(f"[\n{body}\n]\n"))

    cfg = _yaml.load(PRELABEL_YAML.read_text(encoding="utf-8"))
    cm = cfg.get("class_map")
    if cm is None:
        cfg["class_map"] = pairs
    else:
        for k in [k for k in cm if int(k) not in pairs]:
            del cm[k]
        for k, v in pairs.items():
            changed = cm.get(k) != v
            cm[k] = v
            # comment cũ mô tả ánh xạ cũ; để nguyên là nó nói ngược với giá trị
            if changed:
                label = (model_classes or COCO_NAMES).get(k, k)
                cm.yaml_add_eol_comment(f"lớp {label} của model", k, column=18)
    _atomic_write(PRELABEL_YAML, lambda f: _yaml.dump(cfg, f))
    return load_labels(model_classes)

