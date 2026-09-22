# Vision AI — Pipeline dữ liệu phát hiện phương tiện

Pipeline biến video/ảnh giao thông thô thành **dataset có thể huấn luyện, có phiên bản và
truy ngược được nguồn gốc**: nạp nguồn → gán nhãn sơ bộ bằng YOLO26 → người sửa và duyệt
trên CVAT → đóng phiên bản bằng DVC + MinIO.

Không gắn với bộ dữ liệu nào cụ thể. Nguồn khai trong [`configs/pipeline.yaml`](configs/pipeline.yaml),
vùng bỏ qua khai trong [`configs/ignore_regions.json`](configs/ignore_regions.json).
Bản mặc định trỏ vào UA-DETRAC để chạy thử được ngay.

Bốn lớp: `car`, `bus`, `van`, `others` (xem [`configs/labels.json`](configs/labels.json)).

## Nguyên tắc

1. **Nhãn do model sinh không bao giờ đi thẳng vào dataset.** Chúng nằm ở `data/prototype/prelabels/`, chờ người sửa.
2. **Không phát hành thứ thiếu nguồn gốc.** [`04_build_release.py`](scripts/04_build_release.py) kiểm tra từng ảnh: checksum, camera/nguồn/frame, model sinh nhãn sơ bộ, phiên bản guideline. Thiếu một thứ là dừng toàn bộ.
3. **Chỉ lấy nhãn đã có người duyệt PASS** (`stage=acceptance` + `state=completed` trên CVAT).
4. **Không chia train/val.** Đó là việc của MLOps, và phải chia theo `camera_id`/`sequence_id` — chia ngẫu nhiên theo ảnh sẽ rò rỉ dữ liệu vì các frame cùng video gần như giống nhau.
5. **Mọi tham số nằm trong `configs/`**, không hard-code trong script.
6. **Không sửa dữ liệu gốc.** Bước nạp chỉ đọc và copy.

## Luồng

| # | Bước | Script | Ra |
|---|---|---|---|
| 0 | ingest | [`00_ingest.py`](scripts/00_ingest.py) | `images/` + `manifest.jsonl` (+ MinIO) |
| 1 | prelabel | [`01_prelabel.py`](scripts/01_prelabel.py) | `prelabels/*.txt` + `_provenance.json` |
| 2 | package | [`02_prelabels_to_cvat.py`](scripts/02_prelabels_to_cvat.py) | `prelabels_coco.zip` (COCO 1.0) |
| 3 | cvat | [`03_cvat_create_task.py`](scripts/03_cvat_create_task.py) | task CVAT đã nạp nhãn sơ bộ |
| — | **review** | *(người làm)* | sửa + duyệt theo [guideline](docs/labeling_guideline.md) |
| 4 | release | [`04_build_release.py`](scripts/04_build_release.py) | `dataset/` + `MANIFEST.json` |
| — | publish | [`run_pipeline.py`](scripts/run_pipeline.py) | `dvc add` + git tag + `dvc push` |

Bước review là con người. Script chỉ chờ, không thay thế.

## Khai báo nguồn dữ liệu

Trong [`configs/pipeline.yaml`](configs/pipeline.yaml), mục `sources`. Mỗi mục là **một camera**:

```yaml
sources:
  # thư mục ảnh đã tách frame sẵn
  - camera_id: cam01
    kind: images
    path: data/raw/ua-detrac/DETRAC-Images/DETRAC-Images/MVI_20011
    scene: {weather: sunny, camera_state: unstable}

  # file video — frame tách bằng OpenCV, fps đọc thật từ file
  - camera_id: cam05
    kind: video
    path: data/inbox/nga-tu-le-duan.mp4
    scene: {weather: sunny, location: "Lê Duẩn"}
```

- `sequence_id` mặc định lấy theo tên file/thư mục; ghi đè bằng khoá `sequence`.
- `scene` là metadata **tự khai, tuỳ chọn** — mọi khoá trong đó đi thẳng vào `metadata.jsonl` của bản phát hành. Thêm khoá tuỳ ý (`location`, `lane_count`, …).
- Ảnh ra có tên `{camera_id}_{sequence_id}_{frame_idx:06d}.jpg`. Với nguồn `images`, `frame_idx` suy từ số cuối tên file (`img00123.jpg` → 123); không có số thì lấy thứ tự trong thư mục.

Thả dữ liệu mới vào `data/inbox/` rồi trỏ `path` tới đó.

## Vùng bỏ qua

Những vùng ảnh **không gán nhãn**: đoạn đường quá xa, bãi đỗ ven đường, khu vực ngoài phạm vi
quan tâm. Dự đoán rơi vào đây bị loại trước khi nạp lên CVAT, để đánh giá viên không phải xoá tay.

Khai trong [`configs/ignore_regions.json`](configs/ignore_regions.json), **toạ độ chuẩn hoá 0..1**
nên đổi độ phân giải camera không phải tính lại:

```json
{
  "cameras": {
    "cam01": {
      "note": "đường ở xa + bãi đỗ ven đường",
      "regions": [{"x1": 0.81, "y1": 0.05, "x2": 1.0, "y2": 0.16}]
    }
  }
}
```

Camera không có mục ở đây = không có vùng bỏ qua nào. Ngưỡng diện tích chồng lấn đặt ở
`ignore_regions.overlap_threshold` trong `pipeline.yaml` (mặc định 0.5).

Kiểm tra bằng mắt sau khi sửa:

```bash
.venv/bin/python scripts/tools/draw_ignore_regions.py
```

Sinh ra `docs/ignore_regions/<camera_id>.jpg` — chính là ảnh minh hoạ nhúng trong guideline.

## Cài đặt

Một lệnh, sau đó không phải cài gì nữa:

```bash
./setup.sh
```

Script tự làm: khởi tạo submodule CVAT, dựng `.venv` + cài thư viện, tạo `.env` từ
`.env.example`, dựng `data/inbox/`, khởi động MinIO và CVAT, rồi tạo tài khoản CVAT
từ `CVAT_USER`/`CVAT_PASSWORD` trong `.env`. Chạy lại lúc nào cũng được — bước nào
xong rồi thì bỏ qua.

| Lệnh | Làm gì |
|---|---|
| `./setup.sh` | cài đầy đủ |
| `./setup.sh --check` | chỉ kiểm tra, không sửa gì |
| `./setup.sh --python-only` | chỉ `.venv` + thư viện, bỏ qua Docker |
| `./setup.sh --no-cvat` | có MinIO, bỏ qua CVAT (~10 container) |
| `./setup.sh --force-venv` | xoá `.venv` cũ và cài lại |

Clone chưa kèm submodule cũng không sao — `setup.sh` tự khởi tạo.

Môi trường ảo được tạo bằng `python3 -m venv`, hoặc `virtualenv` / `uv` nếu bản Python
của máy thiếu `ensurepip` (hay gặp trên Debian/Ubuntu). Không có cái nào thì script báo
rõ lệnh cần chạy.

Sau khi cài, **điền `.env`** nếu chưa có: `MINIO_ROOT_USER`/`MINIO_ROOT_PASSWORD` và
`CVAT_USER`/`CVAT_PASSWORD`, rồi chạy lại `./setup.sh` để nó tạo tài khoản CVAT.

Giao diện: MinIO console http://localhost:9001, CVAT http://localhost:8080.

Để CVAT đọc được ảnh từ MinIO, bỏ comment khối `cvat` trong `networks:` của
[`docker-compose.yml`](docker-compose.yml) sau khi CVAT đã chạy.

## Chạy

Một lệnh từ đầu tới cuối:

```bash
.venv/bin/python scripts/run_pipeline.py --version 1.0.0
```

Script dừng ở bước `review` và chờ tới khi mọi job được duyệt PASS trên CVAT.
`Ctrl+C` để dừng, chạy tiếp bằng `--from review --task <id>`.

```bash
.venv/bin/python scripts/run_pipeline.py --list                          # xem các bước
.venv/bin/python scripts/run_pipeline.py --version 1.0.0 --to cvat       # dừng sau khi tạo task
.venv/bin/python scripts/run_pipeline.py --version 1.0.0 --from release --task 4
```

Hoặc chạy từng bước — mọi script đều có `--help`:

```bash
.venv/bin/python scripts/00_ingest.py --force --upload
.venv/bin/python scripts/01_prelabel.py --force
.venv/bin/python scripts/04_build_release.py --version 1.0.0 --task 4
```

## Cấu trúc thư mục

```
setup.sh      cài đặt toàn bộ, chạy lại được nhiều lần   ─┐
configs/      tham số pipeline, label spec, vùng bỏ qua   │
scripts/      logic, đánh số theo bước                    │
  _common.py  helper dùng chung (không chạy trực tiếp)    ├─ vào Git
  tools/      tiện ích phụ trợ, không thuộc luồng chính   │
docs/         guideline gán nhãn + ảnh minh hoạ           │
models/       trọng số YOLO (cố ý commit)                ─┘
data/
  inbox/      nơi thả dữ liệu mới                        ─┐
  prototype/  thư mục làm việc của lô hiện tại            ├─ .gitignore
dataset/      bản phát hành (DVC theo dõi)                │
reports/      artifact mỗi lần chạy                      ─┘
third_party/  submodule CVAT
```

`data/` vứt đi lúc nào cũng được, dựng lại từ `configs/`. `dataset/` thì không —
nó là bản phát hành, có `MANIFEST.json` kèm checksum của từng file và được đóng phiên bản
bằng git tag + DVC.

## Lấy lại một phiên bản dataset

```bash
git checkout dataset-v1.0.0 && dvc pull
```

## Nhãn sơ bộ tốt tới đâu

Đo một lần trên lô prototype UA-DETRAC (166 ảnh có đáp án gốc để đối chiếu, 1.301 box):

| | mAP@0.5 | Precision | Recall | Bỏ sót |
|---|---|---|---|---|
| | 0,443 | 0,863 | 0,841 | 207 box |

| Lớp | GT | AP50 | Ghi chú |
|---|---|---|---|
| `car` | 1.080 | 0,841 | |
| `bus` | 156 | 0,913 | |
| `van` | 57 | **0,016** | 39/57 bị model gọi là `car` — COCO không có lớp `van` |
| `others` | 8 | 0,000 | không box nào khớp |

Đây chính là lý do bước review bắt buộc phải có người: vị trí box khá tốt, nhưng phân lớp
`van` gần như sai hoàn toàn. **Dữ liệu mới không có đáp án gốc để đo lại**, nên mặc định
phải coi nhãn sơ bộ là chưa đáng tin.

## Tài liệu

- [`docs/labeling_guideline.md`](docs/labeling_guideline.md) — quy tắc gán nhãn, bắt buộc đọc trước khi review
- [`docs/datasheet.md`](docs/datasheet.md) — datasheet của dataset phát hành
- [`models/README.md`](models/README.md) — trọng số model
