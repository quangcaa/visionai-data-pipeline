# Datasheet — dataset phát hành

Theo khung [Datasheets for Datasets](https://arxiv.org/abs/1803.09010) (Gebru et al.).
Bản sao của file này được đóng gói kèm mỗi lần phát hành (`dataset/datasheet.md`).

| | |
|---|---|
| **Sinh bởi** | [`scripts/04_build_release.py`](../scripts/04_build_release.py) |
| **Nguồn số liệu** | `dataset/MANIFEST.json` của bản phát hành tương ứng |

> Con số trong tài liệu này ứng với bản **v1.0.0**. Mỗi lần phát hành mới phải cập nhật lại,
> hoặc đối chiếu trực tiếp với `MANIFEST.json` của bản đó.

---

## 1. Động cơ

**Dataset này được tạo để làm gì?** Huấn luyện mô hình phát hiện phương tiện giao thông từ
camera cố định. Mục tiêu đồng thời là dựng và kiểm chứng một *pipeline dữ liệu* đầy đủ:
nạp nguồn → gán nhãn sơ bộ → người sửa → duyệt → phát hành có phiên bản.

**Ai tạo?** Nhóm Data Engineering, trong phạm vi prototype nội bộ.

**Đây là bản prototype.** Quy mô cố ý nhỏ để chạy được trên CPU và soát tay được toàn bộ.
Không đủ lớn để huấn luyện mô hình dùng thật.

## 2. Thành phần

**Mỗi mẫu là gì?** Một khung hình JPEG trích từ video giao thông, kèm file nhãn YOLO cùng
tên trong `labels/`.

**Bản v1.0.0 có bao nhiêu?**

| | |
|---|---|
| Ảnh | 50 |
| Box | 185 |
| Ảnh không có xe nào | 3 |
| Camera | `cam03` (sequence `MVI_39761`, 960×540) |
| Điều kiện | ban đêm (`night`) |

Phân bố lớp:

| Lớp | Số box |
|---|---|
| `car` | 165 |
| `bus` | 14 |
| `others` | 6 |
| `van` | **0** |

**Mất cân bằng nặng, và thiếu hẳn một lớp.** Bản v1.0.0 chỉ gồm một camera ban đêm nên
không có `van` nào, `others` thì quá ít để học. Không dùng bản này một mình để đánh giá
năng lực mô hình trên 4 lớp.

**Có gì kèm theo?**

| File | Nội dung |
|---|---|
| `images/` | ảnh JPEG |
| `labels/` | nhãn YOLO: `<class_id> <cx> <cy> <w> <h>`, toạ độ chuẩn hoá |
| `metadata.jsonl` | một dòng mỗi ảnh: camera, nguồn, frame, metadata tự khai, checksum, job CVAT đã duyệt |
| `data.yaml` | khai báo lớp cho Ultralytics |
| `MANIFEST.json` | phiên bản, nguồn, tham số, checksum SHA-256 của **từng** file |
| `datasheet.md` | tài liệu này |

**Có thiếu thông tin gì không?** Có. Bộ nguồn UA-DETRAC bỏ trống cả đoạn frame không gán
nhãn (với `MVI_39761` là frame 897–1080). Những ảnh rơi vào đoạn đó không có nhãn gốc để
đối chiếu.

**Có dữ liệu cá nhân không?** Ảnh chụp đường công cộng, có thể thấy biển số và người đi
bộ ở độ phân giải thấp. Không có định danh cá nhân nào được gán nhãn hay lưu trữ.

## 3. Quy trình thu thập

**Nguồn.** Khai trong `sources` của [`configs/pipeline.yaml`](../configs/pipeline.yaml).
Pipeline nhận hai loại nguồn — file video (`kind: video`) và thư mục ảnh đã tách frame
(`kind: images`) — nên không gắn với bộ dữ liệu nào cụ thể.

Bản v1.0.0 lấy từ UA-DETRAC (phân phối Kaggle `bratjay/ua-detrac-orig`) dưới dạng thư mục
ảnh. Dữ liệu gốc **không bị sửa hay di chuyển** — bước nạp chỉ đọc và copy.

**Tuyển chọn.** Lấy thưa 1 frame mỗi 25 frame (~1 ảnh/giây ở 25 fps), tối đa 50 ảnh mỗi
camera. Lấy thưa là có chủ đích: các frame liền nhau gần như giống hệt, gán nhãn chúng chỉ
tốn công mà không thêm thông tin.

Tập prototype phủ 4 điều kiện thời tiết, mỗi sequence là một camera:

| Camera | Sequence | Điều kiện |
|---|---|---|
| cam01 | MVI_20011 | sunny |
| cam02 | MVI_40131 | cloudy |
| cam03 | MVI_39761 | night |
| cam04 | MVI_63521 | rainy |

Bản v1.0.0 chỉ phát hành `cam03` vì mới có job của camera này được duyệt PASS.

**Truy ngược nguồn gốc.** Mỗi ảnh giữ `camera_id`, `sequence_id`, `frame_idx`,
`timestamp_in_video_sec`, `source_kind`, `source_path` và `sha256` trong `metadata.jsonl`.
Bước phát hành kiểm lại checksum và từ chối phát hành nếu ảnh bị đổi.

## 4. Gán nhãn

**Quy trình ba chặng:**

1. **Nhãn sơ bộ** — YOLO26 nano (`yolo26n.pt`, ultralytics 8.4.153), `conf=0.25`,
   `imgsz=960`, CPU. Lớp COCO được ánh xạ sang lớp đích; COCO không có `van` nên `truck`
   bị quy về `van` — một phép gần đúng, và chính là một lý do bắt buộc phải có người sửa.
2. **Người sửa** — trên CVAT, theo [`labeling_guideline.md`](labeling_guideline.md)
   **phiên bản 1.0**. Được phép và bắt buộc: giữ, chỉnh, đổi lớp, xoá, **và thêm box bị bỏ sót**.
3. **Người duyệt** — đặt PASS (`acceptance`/`completed`) hoặc FAIL (`annotation`/`rejected`).
   Chỉ job PASS mới vào bản phát hành.

**Nhãn sơ bộ tốt tới đâu?** Đo **một lần** trên lô prototype UA-DETRAC (166 ảnh có đáp án
gốc, 1.301 box). Pipeline không còn bước đo tự động — dữ liệu mới không có đáp án để đối chiếu:

| | mAP@0.5 | Precision | Recall | Bỏ sót |
|---|---|---|---|---|
| | 0,443 | 0,863 | 0,841 | 207 box |

| Lớp | GT | AP50 |
|---|---|---|
| `car` | 1.080 | 0,841 |
| `bus` | 156 | 0,913 |
| `van` | 57 | 0,016 |
| `others` | 8 | 0,000 |

39 trên 57 xe `van` bị model gọi là `car`. **Nhãn sơ bộ tự nó không dùng được** — số liệu này
là căn cứ định lượng cho việc bắt buộc có người ở chặng 2. Với camera mới, giả định mặc định
là nhãn sơ bộ còn kém hơn thế, vì model chưa từng thấy góc nhìn đó.

**Vùng bỏ qua.** Khai tay theo từng camera trong
[`configs/ignore_regions.json`](../configs/ignore_regions.json), toạ độ chuẩn hoá 0..1.
Dự đoán có từ 50% diện tích nằm trong vùng đó bị loại trước khi nạp lên CVAT, để không ai
phải xoá tay hàng trăm box thừa. Vùng của 4 camera UA-DETRAC được chuyển sẵn từ
`ignored_region` trong annotation gốc; camera mới phải tự khai.

**Nguồn gốc bản v1.0.0:** CVAT task 4, job 7 và 8, cùng duyệt PASS ngày 2026-09-16.

## 5. Dùng như thế nào

**Đã dùng vào việc gì?** Chưa. Mới phát hành, chưa huấn luyện mô hình nào.

**Dùng được vào việc gì?** Kiểm thử pipeline huấn luyện, kiểm thử tích hợp, demo. **Không**
dùng để đánh giá năng lực mô hình — quá nhỏ và thiếu lớp `van`.

**KHÔNG chia sẵn train/val.** Đây là ranh giới cố ý: việc chia là của MLOps, và **phải chia
theo `camera_id`/`sequence_id`**. Chia ngẫu nhiên theo ảnh sẽ rò rỉ dữ liệu, vì các frame cùng
một video gần như giống nhau — mô hình sẽ có điểm val cao giả tạo.

`data.yaml` để `train` và `val` cùng trỏ vào toàn bộ ảnh, chỉ nhằm cho file hợp lệ với
Ultralytics. Đừng dùng thẳng để đo.

**Thiên lệch cần biết trước:**

- Một camera, một góc nhìn, một điều kiện sáng (ban đêm)
- Không có `van`, `others` chỉ 6 box
- Xe bị che chiếm tỷ trọng đáng kể; quy ước là **vẽ box bao cả xe**, kể cả phần bị che
- Toàn bộ dữ liệu từ Trung Quốc, không đại diện cho giao thông nơi khác

## 6. Phân phối và bảo trì

**Lưu ở đâu?** Nội dung do DVC quản lý, đẩy lên MinIO. Con trỏ (`dataset.dvc`) và git tag
`dataset-v<phiên bản>` nằm trong Git. Lấy lại một phiên bản:

```bash
git checkout dataset-v1.0.0 && dvc pull
```

**Giấy phép.** Bản v1.0.0 dẫn xuất từ UA-DETRAC — công bố cho mục đích nghiên cứu. Kiểm tra
điều khoản của bộ dữ liệu gốc trước khi dùng ngoài phạm vi nghiên cứu. Nhãn trong bản phát
hành này do nhóm tạo ra.

**Ai bảo trì?** Nhóm Data Engineering.

**Cập nhật thế nào?** Mỗi bản phát hành là một lần chạy `04_build_release.py` với số phiên bản
mới, kèm git tag riêng. Bản cũ không bị ghi đè và luôn lấy lại được. Quy tắc tăng số phiên bản
của *guideline* nằm ở mục 8 của [`labeling_guideline.md`](labeling_guideline.md).

**Bản cũ có được giữ không?** Có — git tag + DVC. Mỗi `MANIFEST.json` ghi lại commit của code
đã tạo ra bản đó, nên tái dựng được.

---

## Lịch sử

| Phiên bản | Ngày | Thay đổi |
|---|---|---|
| 1.0.0 | 2026-09-16 | Phát hành lần đầu. 50 ảnh cam03 (night), 185 box, từ CVAT task 4 job 7+8. |
