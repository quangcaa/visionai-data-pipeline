# Hướng dẫn gán nhãn phương tiện giao thông

| | |
|---|---|
| **Phiên bản** | 1.0 |
| **Ngày ban hành** | 2026-09-16 |
| **Áp dụng cho** | CVAT project khai trong `cvat.project_name` ([`configs/pipeline.yaml`](../configs/pipeline.yaml)) |
| **Label spec** | [`configs/labels.json`](../configs/labels.json) |
| **Khai báo phiên bản** | `labeling.guideline_version` trong [`configs/pipeline.yaml`](../configs/pipeline.yaml) |

Tài liệu này là đầu vào bắt buộc của **Bước 3 — Đánh giá viên tinh chỉnh và duyệt**. Mọi job phải được gán nhãn theo đúng phiên bản ghi ở trên. Phiên bản guideline được ghi vào từng ảnh của dataset phát hành.

---

## 1. Nhãn sơ bộ là gợi ý, không phải đáp án

Mỗi job mở ra đã có sẵn box do model YOLO26 sinh. Các box này **chưa được ai xác nhận**. Đo trên lô prototype UA-DETRAC (tháng 9/2026, 166 ảnh có đáp án gốc để đối chiếu):

- Vị trí box khá tốt: AP50 của `car` 0.84, `bus` 0.91
- **Phân loại sai nhiều**: 39 trên 57 xe van bị model gọi là `car`
- Có box thừa (precision 0.86), có xe bị bỏ sót (207 box)

Đây là số đo một lần trên bộ có sẵn đáp án gốc. Dữ liệu mới không có đáp án để đo lại, nên **mặc định phải coi nhãn sơ bộ là chưa đáng tin** cho tới khi có người soát.

Đánh giá viên phải **xem từng box trên từng ảnh**. Không được bấm Finish khi chưa soát hết.

Năm thao tác được phép, và phải dùng đủ khi cần:

| Thao tác | Khi nào | Phím |
|---|---|---|
| Giữ nguyên | Box đúng lớp, bám sát xe | — |
| Chỉnh box | Đúng xe nhưng lệch hoặc thiếu/thừa diện tích | kéo góc |
| Đổi lớp | Đúng vị trí, sai lớp | panel phải |
| Xoá | Không phải xe, hoặc nằm trong vùng bỏ qua (mục 5) | `Delete` |
| **Thêm box** | **Xe có thật mà chưa có box** | `N` |

Thêm box bị bỏ sót là thao tác **bắt buộc**, không chỉ chấp nhận hoặc xoá.

---

## 2. Các lớp

Chỉ có 4 lớp. Không tạo lớp mới.

| Lớp | Là gì | Dấu hiệu nhận biết |
|---|---|---|
| `car` | Xe con: sedan, hatchback, SUV, taxi | Thân thấp, nắp capo và cốp tách rõ khỏi khoang người |
| `bus` | Xe buýt, xe khách lớn | Rất dài, cao, nhiều cửa sổ dọc thân. Box thường **cao hơn rộng** khi nhìn từ trên xuống (w/h trung vị 0.70) |
| `van` | Xe van, minibus, xe tải nhỏ dạng hộp | Thân cao, khoang liền khối từ sau kính lái tới đuôi, **không có cốp tách riêng** |
| `others` | Phương tiện 4 bánh trở lên không thuộc 3 lớp trên: xe tải lớn, xe chuyên dụng | Rất hiếm: 8 trên 1.301 box trong lô prototype |

### 2.1 Bẫy lớn nhất: `van` bị gán thành `car`

Model nhầm ở đây nhiều nhất, và mắt người lướt nhanh cũng dễ bỏ qua.

**Cách kiểm tra:** nhìn phần đuôi xe. Có bậc thụt xuống giữa khoang người và cốp → `car`. Mái kéo thẳng ra tận đuôi thành một khối hộp → `van`.

Khi phân vân, đối chiếu với ảnh minh hoạ của chính camera đó trong [`docs/ignore_regions/`](ignore_regions/).

### 2.2 Không gán nhãn

- Người đi bộ, xe máy, xe đạp. Model có thể đã vẽ box cho những đối tượng này dưới lớp `others`. **Xoá đi.** Trong lô prototype, không box `others` nào của model khớp với một box `others` thật.
- Phản chiếu của đèn xe trên mặt đường ướt, đèn đường, biển báo. Hay gặp ở camera chạng vạng/ban đêm (`cam03`): precision của model chỉ 0.717, thấp nhất trong 4 camera.

### 2.3 Khi phân vân về lớp

**Không đoán, và không dồn vào `others`.** Chọn lớp gần đúng nhất, rồi tạo **issue** tại box đó trong CVAT, ghi rõ lý do phân vân. Người duyệt sẽ quyết định. Các ca phân vân lặp lại là căn cứ để sửa guideline ở phiên bản sau.

---

## 3. Quy tắc vẽ box

### 3.1 Bám sát xe

Bốn cạnh box chạm đúng bốn điểm ngoài cùng của xe: gương, cản, bánh. Không chừa khoảng trống, không cắt vào thân xe.

### 3.2 Xe bị che → bao cả xe

Xe bị xe khác, cây hoặc cột che một phần: **vẽ box bao toàn bộ xe**, ước lượng phần bị che dựa vào hình dạng và kích thước xe. Không vẽ chỉ phần nhìn thấy.

Quy ước này theo đúng UA-DETRAC: 19,2% số box trong bộ dữ liệu là xe bị che, và box luôn bao cả xe.

### 3.3 Xe ra khỏi khung hình → vẽ tới mép ảnh

Xe mới vào hoặc sắp ra khỏi khung hình **vẫn phải gán nhãn**. Vẽ box tới đúng mép ảnh. 15% số box trong lô prototype chạm mép ảnh.

### 3.4 Kích thước tối thiểu

| Cạnh ngắn của xe | Quy tắc |
|---|---|
| **≥ 8 px** | **Bắt buộc** gán nhãn |
| < 8 px | Không bắt buộc |

Mốc 8 px lấy từ dữ liệu: 99% số box của UA-DETRAC có cạnh ngắn từ 8,5 px trở lên. Trung vị là 50 px, nhưng 5% số box có cạnh ngắn **dưới 20 px** — đó là xe ở xa, nhỏ nhưng **vẫn phải gán**.

Nếu chỉ còn là một chấm mờ không phân biệt được là xe gì, thường nó đã nhỏ hơn 8 px hoặc nằm trong vùng bỏ qua (mục 5).

---

## 4. Tạo box mới

1. Phím `N` (hoặc icon hình chữ nhật ở thanh công cụ trái)
2. Chọn lớp **trước** khi vẽ
3. Bấm một góc, kéo sang góc đối diện, thả chuột
4. Chỉnh lại cho sát theo mục 3.1

---

## 5. Vùng bỏ qua

Mỗi camera có những vùng **không gán nhãn**: đoạn đường quá xa, xe mờ chồng lên nhau, bãi đỗ xe ven đường. CVAT **không hiển thị** các vùng này, nên phải xem ảnh minh hoạ trước khi làm job của camera đó:

Ảnh minh hoạ nằm ở [`docs/ignore_regions/`](ignore_regions/), mỗi camera một file
`<camera_id>.jpg` — vùng đỏ là chỗ **không gán nhãn**. Xem ảnh của camera mình sắp làm
trước khi mở job.

Quy tắc: xe có **từ một nửa diện tích trở lên** nằm trong vùng đỏ thì **không gán nhãn**. Nhãn sơ bộ đã được lọc sẵn theo quy tắc này, nên đừng tự thêm box vào đó.

Vùng bỏ qua khai báo tay trong [`configs/ignore_regions.json`](../configs/ignore_regions.json), toạ độ chuẩn hoá 0..1 theo camera. Camera mới phải khai vùng của nó vào đây, nếu không sẽ không có vùng nào bị loại.

Ảnh minh hoạ sinh từ lô dữ liệu hiện tại — chạy lại sau mỗi lần sửa config hoặc nạp lô mới:

```
.venv/bin/python scripts/tools/draw_ignore_regions.py
```

---

## 6. Duyệt job: PASS hoặc FAIL

Người sửa và người duyệt nên là **hai người khác nhau**. Người duyệt mở job đã Finish, soát lại theo checklist ở mục 7, rồi đặt trạng thái cho job **trên CVAT**:

| Quyết định | Stage | State | Chuyện gì xảy ra tiếp |
|---|---|---|---|
| **PASS** | `acceptance` | `completed` | Job được đưa vào bản phát hành dataset |
| **FAIL** | `annotation` | `rejected` | Job trả về người sửa. Người duyệt **phải tạo issue** tại từng chỗ sai để người sửa biết sửa gì |

Cách đặt trong CVAT: trang task → dòng của job → đổi cột **Stage**; đổi **State** trong menu của job (Menu → Change job state).

Script phát hành (`scripts/04_build_release.py`) **chỉ lấy job có đúng cặp `acceptance` + `completed`**. Job ở bất kỳ trạng thái nào khác đều bị bỏ qua.

Lý do thường gặp để FAIL:

| Lỗi | Mục liên quan |
|---|---|
| Còn xe `van` bị để là `car` | 2.1 |
| Còn box cho xe máy, xe đạp, người, phản chiếu đèn | 2.2 |
| Box lỏng, không bám sát xe | 3.1 |
| Xe bị che chỉ khoanh phần nhìn thấy | 3.2 |
| Bỏ sót xe nhỏ ở xa hoặc xe cắt mép | 3.3, 3.4 |
| Có box trong vùng bỏ qua | 5 |

---

## 7. Kiểm tra trước khi Finish

- [ ] Đã xem **tất cả** ảnh trong job, không chỉ những ảnh có box sơ bộ
- [ ] Mọi xe có cạnh ngắn ≥ 8 px ngoài vùng bỏ qua đều có box
- [ ] Đã soát lại từng box `car` xem có phải `van` không
- [ ] Đã xoá box cho xe máy, xe đạp, người, phản chiếu đèn
- [ ] Xe bị che: box bao cả xe. Xe cắt mép: box tới mép ảnh
- [ ] Ca phân vân đã tạo issue
- [ ] `Ctrl+S` → Menu → **Finish the job**

---

## 8. Lịch sử phiên bản

| Phiên bản | Ngày | Thay đổi |
|---|---|---|
| 1.0 | 2026-09-16 | Ban hành lần đầu. Quy tắc che khuất và cắt mép đối chiếu theo annotation gốc UA-DETRAC. Ngưỡng kích thước tối thiểu 8 px lấy từ phân vị 1% cạnh ngắn của box GT. |

**Khi nào tăng phiên bản:**

- **Tăng số lớn (2.0)**: thêm/bớt/đổi định nghĩa lớp. Nhãn cũ không còn tương thích, phải gán nhãn lại.
- **Tăng số nhỏ (1.1)**: đổi quy tắc vẽ box hoặc ngưỡng kích thước, làm kết quả gán nhãn khác đi.
- **Chỉ sửa câu chữ, thêm ví dụ**: giữ nguyên phiên bản.

Mỗi lần tăng phiên bản phải cập nhật `labeling.guideline_version` trong `configs/pipeline.yaml`. Phiên bản này được ghi vào từng ảnh của dataset phát hành.
