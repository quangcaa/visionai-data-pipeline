# Thư mục trọng số model

Trọng số YOLO dùng cho bước gán nhãn sơ bộ (`scripts/01_prelabel.py`).
Khai báo trong [`configs/prelabel.yaml`](../configs/prelabel.yaml) qua khoá `model`.

- `yolo26n.pt` — YOLO26 nano (đã có sẵn, ~5.3 MB).
- Ultralytics tự tải nếu đường dẫn chỉ là tên (vd `yolo26s.pt`) về cache hệ thống (`~/.cache/Ultralytics/`).

Trọng số trong thư mục này **được commit vào Git** (override `.gitignore` `*.pt`) để
người dùng clone về chạy được ngay, không phải tải.
