# Chạy lần đầu — từ repo trắng tới dataset phát hành

Hướng dẫn đi thẳng một mạch, mỗi bước một lệnh. Chưa biết gì về repo cũng theo được.

Tra cứu chi tiết từng phần thì xem [README](../README.md).

---

## 0. Cần có sẵn trên máy

| | Kiểm tra bằng |
|---|---|
| Git | `git --version` |
| Python ≥ 3.10 | `python3 -V` |
| Docker + `docker compose` | `docker compose version` |

Docker phải **đang chạy** và tài khoản của bạn phải dùng được (`docker info` không báo lỗi quyền).

---

## 1. Lấy mã nguồn

```bash
git clone <địa-chỉ-repo> VisionAI-data-pipeline-test && cd VisionAI-data-pipeline-test
```

Quên `--recurse-submodules` cũng không sao — bước sau tự xử lý.

---

## 2. Cài đặt

```bash
./setup.sh
```

Một lệnh làm hết: submodule CVAT → `.venv` + thư viện → `.env` → `data/inbox/` → MinIO → CVAT → tài khoản CVAT.

Lần đầu mất **vài phút tới vài chục phút**: `torch` khoảng 2 GB, ảnh Docker của CVAT khoảng 6 GB.

Chạy lại lúc nào cũng được, bước nào xong thì bỏ qua.

### Điền `.env` rồi chạy lại

`setup.sh` tạo `.env` từ mẫu. Mở ra đặt mật khẩu của bạn:

```
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=<tự đặt, tối thiểu 8 ký tự>
CVAT_USER=<tên đăng nhập bạn muốn>
CVAT_PASSWORD=<mật khẩu bạn muốn>
```

Rồi chạy lại để nó tạo tài khoản CVAT:

```bash
./setup.sh
```

Kiểm tra mọi thứ đã sẵn sàng:

```bash
./setup.sh --check
```

Phải xanh cả 7 bước. Đỏ chỗ nào thì đọc dòng báo lỗi ở đó.

---

## 3. Đưa dữ liệu vào

Chép video của bạn vào `data/inbox/`:

```bash
cp ~/Videos/nga-tu-le-duan.mp4 data/inbox/
```

Nhận cả **file video** lẫn **thư mục ảnh đã tách frame sẵn**.

Chưa có dữ liệu thì tải bộ mẫu UA-DETRAC (9,3 GB, cần token Kaggle trong `.env`):

```bash
.venv/bin/python scripts/tools/download_detrac.py
```

---

## 4. Chạy — chọn một trong hai đường

Hai đường cho **cùng kết quả**. Web gọi lại đúng các script này.

### Đường A — Web (khuyến nghị cho người mới)

```bash
.venv/bin/python -m web.app
```

Mở http://127.0.0.1:8000 rồi đi từ khu 1 xuống khu 6. Mỗi khu một việc, log hiện trực tiếp.

Nhảy tới [bước 5](#5-soát-nhãn-trên-cvat) — phần còn lại của mục này là đường dòng lệnh.

### Đường B — Dòng lệnh

**B1. Khai nguồn.** Mở `configs/pipeline.yaml`, sửa mục `sources`:

```yaml
sources:
  - camera_id: cam01
    kind: video                       # hoặc images nếu trỏ vào thư mục ảnh
    path: data/inbox/nga-tu-le-duan.mp4
    scene: {weather: sunny}           # tuỳ chọn, khai gì cũng được
```

**B2. Chỉnh tham số lấy mẫu** (cùng file, mục `sampling`):

```yaml
sampling:
  sample_every_n: 25      # lấy 1 frame mỗi 25 frame
  max_per_camera: 50      # tối đa mỗi camera, 0 = không giới hạn
```

**B3. Chạy trọn luồng:**

```bash
.venv/bin/python scripts/run_pipeline.py --version 1.0.0
```

Script chạy `ingest → prelabel → package → cvat` rồi **dừng chờ** bạn soát nhãn trên CVAT.

Muốn chạy từng bước riêng:

```bash
.venv/bin/python scripts/00_ingest.py --force --upload
.venv/bin/python scripts/01_prelabel.py --force
.venv/bin/python scripts/02_prelabels_to_cvat.py
.venv/bin/python scripts/03_cvat_create_task.py
```

Xem tên các bước: `.venv/bin/python scripts/run_pipeline.py --list`

---

## 5. Soát nhãn trên CVAT

Đây là **việc của người**, không tự động được.

1. Mở http://localhost:8080, đăng nhập bằng `CVAT_USER` / `CVAT_PASSWORD`
2. Vào task vừa tạo, mở từng job
3. Sửa theo [`docs/labeling_guideline.md`](labeling_guideline.md) — giữ box đúng, chỉnh box lệch, đổi lớp sai, xoá box thừa, **và thêm box bị bỏ sót**
4. Xong một job thì `Ctrl+S` → Menu → **Finish the job**
5. Người duyệt đặt job thành **stage `acceptance` + state `completed`**

**Chỉ job đúng cặp trạng thái đó mới vào bản phát hành.** Đặt nhanh bằng nút PASS ở khu 5 của web.

---

## 6. Dựng dataset

### Trên web
Khu 6 → điền phiên bản → **Dựng dataset**.

### Dòng lệnh
```bash
.venv/bin/python scripts/04_build_release.py --version 1.0.0
```

Bước này kiểm tra đủ bộ từng ảnh: checksum, nguồn gốc, model sinh nhãn sơ bộ, phiên bản guideline. **Thiếu một thứ là dừng toàn bộ**, không phát hành gì cả.

Kết quả ở `dataset/`:

```
dataset/
  images/          ảnh
  labels/          nhãn YOLO: <class_id> <cx> <cy> <w> <h>, toạ độ 0..1
  data.yaml        khai báo lớp cho Ultralytics
  metadata.jsonl   nguồn gốc từng ảnh: camera, video, frame, ai duyệt
  MANIFEST.json    phiên bản + checksum từng file
  datasheet.md     tài liệu bộ dữ liệu
```

---

## 7. Đóng phiên bản

### Trên web
Khu 6 → **Đóng phiên bản (DVC + tag)**.

### Dòng lệnh
```bash
.venv/bin/python scripts/run_pipeline.py --from publish --to publish --version 1.0.0
```

Bước này làm 4 việc, và kết quả nằm ở 4 chỗ khác nhau:

| Chỗ | Có gì |
|---|---|
| `dataset/` trên đĩa | dữ liệu thật, **không vào Git** |
| `dataset.dvc` trong Git | con trỏ 6 dòng thay cho cả thư mục |
| git tag `dataset-v1.0.0` | neo phiên bản |
| MinIO bucket `dvc-store` | nội dung thật, lưu theo mã băm |

Kiểm tra đã đẩy đủ lên MinIO chưa:

```bash
.venv/bin/python -m dvc status --cloud
```

Phải thấy `Cache and remote 'minio' are in sync.`

Muốn người khác lấy được thì đẩy Git đi:

```bash
git push && git push origin dataset-v1.0.0
```

---

## 8. Lấy lại một phiên bản ở máy khác

```bash
git clone <repo> && cd VisionAI-data-pipeline-test
./setup.sh
git checkout dataset-v1.0.0
.venv/bin/python -m dvc pull
```

`git checkout` lấy con trỏ, `dvc pull` kéo nội dung từ MinIO về.

---

## Giao diện web

| | |
|---|---|
| Pipeline | http://127.0.0.1:8000 |
| CVAT | http://localhost:8080 |
| MinIO console | http://localhost:9001 |

---

## Vướng ở đâu

**`./setup.sh --check` báo đỏ** — đọc đúng dòng đỏ đó, nó nói thiếu gì.

**Không tạo được `.venv`** — máy thiếu `ensurepip`. Chọn một cách:
`sudo apt install python3.14-venv` hoặc `pip install --user virtualenv`.

**CVAT không lên sau 180 giây** — lần đầu phải tải ~6 GB ảnh Docker. Không phải lỗi, chạy lại `./setup.sh` là nó bắt được trạng thái đã lên.

**`Label 'x' is not registered for this task`** — bạn đổi `configs/labels.json` sau khi CVAT đã tạo project. Chạy lại bước tạo task, nó tự đồng bộ nhãn.

**`lớp 'x' không có trong label spec`** khi dựng dataset — trên CVAT còn nhãn mang tên lớp đã bị xoá khỏi `labels.json`. Đổi chúng sang lớp khác trong CVAT trước.

**Máy đầy đĩa** — `du -sh data/* .venv third_party` cho thấy chỗ nào nặng. `.venv` khoảng 6 GB, ảnh Docker CVAT khoảng 6 GB, `data/prototype/images` theo số ảnh đã lấy mẫu.

**Nhãn sơ bộ trỏ sai lớp** — bạn vừa đổi thứ tự lớp. `class_id` là **vị trí** trong danh sách, nên đổi thứ tự là đổi nghĩa. Chạy lại bước gán nhãn sơ bộ.
