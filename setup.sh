#!/usr/bin/env bash
#
# Cài đặt toàn bộ repo. Chạy một lần, sau đó chỉ việc dùng pipeline.
#
#   ./setup.sh                 cài đầy đủ (venv + thư viện + MinIO + CVAT)
#   ./setup.sh --check         chỉ kiểm tra, không sửa gì
#   ./setup.sh --python-only   chỉ venv + thư viện, bỏ qua Docker
#   ./setup.sh --no-cvat       có MinIO, bỏ qua CVAT (nặng, ~10 container)
#   ./setup.sh --force-venv    xoá venv cũ và cài lại từ đầu
#
# Chạy lại lúc nào cũng được: bước nào đã xong thì bỏ qua.

set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO"

VENV="$REPO/.venv"
PY="$VENV/bin/python"
CVAT_COMPOSE="$REPO/third_party/cvat/docker-compose.yml"
PY_MIN_MINOR=10          # cần Python >= 3.10 (cú pháp `str | None`)

CHECK_ONLY=0; PYTHON_ONLY=0; WITH_CVAT=1; FORCE_VENV=0
for arg in "$@"; do
  case "$arg" in
    --check)       CHECK_ONLY=1 ;;
    --python-only) PYTHON_ONLY=1 ;;
    --no-cvat)     WITH_CVAT=0 ;;
    --force-venv)  FORCE_VENV=1 ;;
    -h|--help)     sed -n '2,11p' "$0" | sed 's/^# \?//'; exit 0 ;;
    *) echo "Tham số không hiểu: $arg (xem ./setup.sh --help)" >&2; exit 2 ;;
  esac
done

# --- in ấn -------------------------------------------------------------------

if [ -t 1 ]; then
  B=$'\033[1m'; G=$'\033[32m'; Y=$'\033[33m'; R=$'\033[31m'; D=$'\033[2m'; N=$'\033[0m'
else
  B=""; G=""; Y=""; R=""; D=""; N=""
fi

STEP=0
FAILED=0
WARNED=0

step()  { STEP=$((STEP + 1)); printf '\n%s[%d/%d] %s%s\n' "$B" "$STEP" "$TOTAL" "$1" "$N"; }
ok()    { printf '      %s✓%s %s\n' "$G" "$N" "$1"; }
skip()  { printf '      %s·%s %s\n' "$D" "$N" "$1"; }
warn()  { printf '      %s!%s %s\n' "$Y" "$N" "$1"; WARNED=$((WARNED + 1)); }
fail()  { printf '      %s✗%s %s\n' "$R" "$N" "$1"; FAILED=$((FAILED + 1)); }
die()   { printf '\n%s✗ %s%s\n' "$R" "$1" "$N" >&2; exit 1; }

TOTAL=7
[ "$PYTHON_ONLY" -eq 1 ] && TOTAL=5

printf '%sCài đặt Vision AI data pipeline%s\n' "$B" "$N"
printf '%s%s%s\n' "$D" "$REPO" "$N"
[ "$CHECK_ONLY" -eq 1 ] && printf '%schế độ --check: chỉ kiểm tra, không thay đổi gì%s\n' "$Y" "$N"

# --- 1. công cụ hệ thống -----------------------------------------------------

step "Kiểm tra công cụ hệ thống"

command -v git >/dev/null || die "thiếu git"
ok "git $(git --version | awk '{print $3}')"

command -v python3 >/dev/null || die "thiếu python3"
PY_VER="$(python3 -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_MINOR="${PY_VER#*.}"
if [ "${PY_VER%%.*}" -lt 3 ] || [ "$PY_MINOR" -lt "$PY_MIN_MINOR" ]; then
  die "cần Python >= 3.$PY_MIN_MINOR, đang có $PY_VER"
fi
ok "python3 $PY_VER"

# Không phải bản Python nào cũng có ensurepip (Debian/Ubuntu tách ra gói riêng),
# nên dò xem công cụ nào tạo được môi trường ảo.
VENV_TOOL=""
if python3 -c 'import ensurepip' 2>/dev/null; then VENV_TOOL="venv"
elif command -v virtualenv >/dev/null 2>&1;  then VENV_TOOL="virtualenv"
elif command -v uv >/dev/null 2>&1;          then VENV_TOOL="uv"
fi
if [ -z "$VENV_TOOL" ]; then
  die "không tạo được môi trường ảo.
    python3 -m venv thiếu ensurepip. Chọn một cách:
      sudo apt install python${PY_VER}-venv     (khuyến nghị)
      pip install --user virtualenv
      pip install --user uv"
fi
ok "công cụ tạo venv: $VENV_TOOL"

if [ "$PYTHON_ONLY" -eq 0 ]; then
  if command -v docker >/dev/null && docker info >/dev/null 2>&1; then
    ok "docker $(docker --version | awk '{print $3}' | tr -d ,)"
    docker compose version >/dev/null 2>&1 \
      && ok "docker compose" \
      || die "thiếu 'docker compose' (plugin v2)"
  else
    die "docker chưa chạy hoặc không có quyền — thử: sudo systemctl start docker, hoặc dùng ./setup.sh --python-only"
  fi
fi

# --- 2. submodule CVAT -------------------------------------------------------

step "Submodule CVAT (third_party/cvat)"

if [ -f "$CVAT_COMPOSE" ]; then
  skip "đã có"
elif [ "$CHECK_ONLY" -eq 1 ]; then
  fail "chưa khởi tạo — chạy ./setup.sh để cài"
else
  git submodule update --init --recursive third_party/cvat \
    && ok "đã khởi tạo" \
    || fail "khởi tạo submodule thất bại"
fi

# --- 3. môi trường Python ----------------------------------------------------

step "Môi trường Python (.venv)"

if [ "$FORCE_VENV" -eq 1 ] && [ "$CHECK_ONLY" -eq 0 ] && [ -d "$VENV" ]; then
  rm -rf "$VENV"; ok "đã xoá venv cũ (--force-venv)"
fi

if [ -x "$PY" ]; then
  skip "đã có ($("$PY" -V 2>&1))"
elif [ "$CHECK_ONLY" -eq 1 ]; then
  fail "chưa có .venv — chạy ./setup.sh để cài"
else
  case "$VENV_TOOL" in
    venv)       python3 -m venv "$VENV" ;;
    virtualenv) virtualenv -q -p python3 "$VENV" ;;
    uv)         uv venv --python python3 "$VENV" >/dev/null ;;
  esac
  [ -x "$PY" ] || die "tạo venv bằng '$VENV_TOOL' thất bại"
  ok "đã tạo .venv bằng $VENV_TOOL"
fi

# --- 4. thư viện -------------------------------------------------------------

step "Thư viện Python (requirements.txt)"

MISSING=""
if [ -x "$PY" ]; then
  # tên module khi import, không phải tên gói trên PyPI
  for mod in PIL cv2 cvat_sdk fastapi kaggle minio numpy ruamel.yaml ultralytics uvicorn yaml; do
    "$PY" -c "import $mod" >/dev/null 2>&1 || MISSING="$MISSING $mod"
  done
  "$PY" -m dvc --version >/dev/null 2>&1 || MISSING="$MISSING dvc"
fi

if [ ! -x "$PY" ]; then
  fail "bỏ qua — chưa có venv"
elif [ -z "$MISSING" ]; then
  skip "đủ cả 12 phụ thuộc"
elif [ "$CHECK_ONLY" -eq 1 ]; then
  fail "thiếu:$MISSING"
else
  echo "      cài đặt (torch qua ultralytics khá nặng, lần đầu mất vài phút)..."
  "$PY" -m ensurepip --upgrade >/dev/null 2>&1 || true
  "$PY" -m pip install --quiet --upgrade pip \
    && "$PY" -m pip install --quiet -r requirements.txt \
    || die "pip install thất bại — chạy lại không có --quiet để xem chi tiết"
  STILL=""
  for mod in PIL cv2 cvat_sdk fastapi kaggle minio numpy ruamel.yaml ultralytics uvicorn yaml; do
    "$PY" -c "import $mod" >/dev/null 2>&1 || STILL="$STILL $mod"
  done
  [ -z "$STILL" ] && ok "đã cài đủ" || fail "vẫn thiếu:$STILL"
fi

# --- 5. cấu hình và thư mục --------------------------------------------------

step "Cấu hình (.env) và thư mục làm việc"

if [ -f .env ]; then
  skip ".env đã có"
elif [ "$CHECK_ONLY" -eq 1 ]; then
  fail ".env chưa có"
else
  cp .env.example .env && ok "đã tạo .env từ .env.example"
fi

if [ -f .env ]; then
  set -a; . ./.env 2>/dev/null || true; set +a
  [ -n "${MINIO_ROOT_USER:-}" ] && [ -n "${MINIO_ROOT_PASSWORD:-}" ] \
    && ok "MINIO_ROOT_USER / PASSWORD đã điền" \
    || warn "thiếu MINIO_ROOT_USER / MINIO_ROOT_PASSWORD trong .env"
  [ -n "${CVAT_USER:-}" ] && [ -n "${CVAT_PASSWORD:-}" ] \
    && ok "CVAT_USER / PASSWORD đã điền" \
    || warn "thiếu CVAT_USER / CVAT_PASSWORD trong .env — điền rồi chạy lại ./setup.sh"
fi

if [ "$CHECK_ONLY" -eq 0 ]; then
  mkdir -p data/inbox && ok "data/inbox/ sẵn sàng (nơi thả dữ liệu mới)"
fi

if [ -x "$PY" ] && [ -n "$MISSING" ] && [ "$CHECK_ONLY" -eq 1 ]; then
  skip "bỏ qua kiểm tra config — thiếu thư viện"
elif [ -x "$PY" ]; then
  "$PY" - <<'EOF' 2>/dev/null && ok "4 file config đọc được" || fail "config lỗi cú pháp"
import json, yaml
for f in ("configs/pipeline.yaml", "configs/prelabel.yaml"): yaml.safe_load(open(f))
for f in ("configs/labels.json", "configs/ignore_regions.json"): json.load(open(f))
EOF
fi

if [ "$PYTHON_ONLY" -eq 1 ]; then
  printf '\n%s--python-only: bỏ qua MinIO và CVAT%s\n' "$D" "$N"
else

# --- 6. MinIO ----------------------------------------------------------------

step "MinIO (lưu ảnh thô + DVC remote)"

minio_up() { curl -fsS --max-time 3 http://localhost:9000/minio/health/live >/dev/null 2>&1; }

if minio_up; then
  skip "đang chạy ở localhost:9000"
elif [ "$CHECK_ONLY" -eq 1 ]; then
  fail "không phản hồi ở localhost:9000"
else
  docker compose up -d minio >/dev/null 2>&1 || fail "docker compose up minio thất bại"
  printf '      chờ MinIO sẵn sàng'
  for _ in $(seq 1 30); do minio_up && break; printf '.'; sleep 2; done
  printf '\n'
  minio_up && ok "đang chạy — console http://localhost:9001" || fail "MinIO không lên sau 60s"
fi

# --- 7. CVAT -----------------------------------------------------------------

step "CVAT (công cụ gán nhãn)"

cvat_up() { curl -fsS --max-time 3 http://localhost:8080/api/server/about >/dev/null 2>&1; }

if [ "$WITH_CVAT" -eq 0 ]; then
  skip "bỏ qua (--no-cvat)"
elif [ ! -f "$CVAT_COMPOSE" ]; then
  fail "chưa có submodule CVAT"
elif cvat_up; then
  skip "đang chạy ở localhost:8080"
elif [ "$CHECK_ONLY" -eq 1 ]; then
  fail "không phản hồi ở localhost:8080"
else
  echo "      khởi động (~10 container, lần đầu phải tải image, mất vài phút)..."
  docker compose -f "$CVAT_COMPOSE" up -d >/dev/null 2>&1 || fail "docker compose up cvat thất bại"
  printf '      chờ CVAT sẵn sàng'
  for _ in $(seq 1 90); do cvat_up && break; printf '.'; sleep 2; done
  printf '\n'
  cvat_up && ok "đang chạy — http://localhost:8080" || fail "CVAT không lên sau 180s"
fi

# tài khoản CVAT: tạo từ .env để không phải gõ tay
if [ "$WITH_CVAT" -eq 1 ] && [ "$CHECK_ONLY" -eq 0 ] && cvat_up; then
  if [ -z "${CVAT_USER:-}" ] || [ -z "${CVAT_PASSWORD:-}" ]; then
    warn "chưa tạo được tài khoản — điền CVAT_USER/CVAT_PASSWORD vào .env rồi chạy lại"
  elif curl -fsS --max-time 5 -u "$CVAT_USER:$CVAT_PASSWORD" \
        http://localhost:8080/api/users/self >/dev/null 2>&1; then
    skip "tài khoản '$CVAT_USER' đã dùng được"
  else
    docker exec -e DJANGO_SUPERUSER_PASSWORD="$CVAT_PASSWORD" cvat_server \
      python3 /home/django/manage.py createsuperuser --noinput \
      --username "$CVAT_USER" --email "${CVAT_EMAIL:-$CVAT_USER@localhost}" >/dev/null 2>&1
    if curl -fsS --max-time 5 -u "$CVAT_USER:$CVAT_PASSWORD" \
        http://localhost:8080/api/users/self >/dev/null 2>&1; then
      ok "đã tạo tài khoản '$CVAT_USER'"
    else
      warn "không tạo được tài khoản tự động — tạo tay:
        docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'"
    fi
  fi
fi

fi  # PYTHON_ONLY

# --- tổng kết ----------------------------------------------------------------

printf '\n%s────────────────────────────────────────%s\n' "$D" "$N"

if [ "$FAILED" -gt 0 ]; then
  printf '%s✗ %d bước chưa xong%s' "$R" "$FAILED" "$N"
  [ "$WARNED" -gt 0 ] && printf ', %d cảnh báo' "$WARNED"
  printf '\n'
  [ "$CHECK_ONLY" -eq 1 ] && printf '  Chạy %s./setup.sh%s để cài.\n' "$B" "$N"
  exit 1
fi

if [ "$WARNED" -gt 0 ]; then
  printf '%s✓ Cài xong, %d cảnh báo ở trên cần xử lý.%s\n' "$Y" "$WARNED" "$N"
else
  printf '%s✓ Cài xong. Không cần cài lại nữa.%s\n' "$G" "$N"
fi

cat <<EOF

${B}Mở web điều khiển${N} — làm mọi bước không cần gõ lệnh:

  .venv/bin/python -m web.app        ${D}# rồi mở http://127.0.0.1:8000${N}

${B}Hoặc dùng dòng lệnh${N} — lấy dữ liệu vào ${B}data/inbox/${N} rồi khai trong ${B}configs/pipeline.yaml${N}:

  ${D}# dùng UA-DETRAC để chạy thử (tải 9,3 GB, cần token Kaggle trong .env)${N}
  .venv/bin/python scripts/tools/download_detrac.py

  ${D}# hoặc thả video / thư mục ảnh của bạn vào data/inbox/ và sửa mục 'sources'${N}

${B}Chạy pipeline${N}:

  .venv/bin/python scripts/run_pipeline.py --list
  .venv/bin/python scripts/run_pipeline.py --version 1.0.0

${B}Kiểm tra lại bất cứ lúc nào${N}:  ./setup.sh --check
EOF
