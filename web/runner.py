"""Chạy các script pipeline dưới dạng job nền, gom log để web xem trực tiếp.

Web KHÔNG nhân bản logic pipeline — nó gọi lại đúng các script trong scripts/,
nên chạy bằng web hay bằng dòng lệnh đều ra cùng một kết quả.

Mỗi lúc chỉ cho phép một job: các bước dùng chung data/prototype nên chạy song
song sẽ giẫm lên nhau.
"""

from __future__ import annotations

import itertools
import queue
import subprocess
import sys
import threading
import time
from collections import deque
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PY = str(REPO_ROOT / ".venv" / "bin" / "python")
MAX_LINES = 2000          # giữ lại bấy nhiêu dòng log gần nhất cho mỗi job


class Job:
    """Một lần chạy script. Log vừa được lưu lại vừa phát cho các subscriber."""

    _ids = itertools.count(1)

    def __init__(self, step: str, argv: list[str], label: str):
        self.id = next(Job._ids)
        self.step = step
        self.label = label
        self.argv = argv
        self.state = "running"          # running | ok | failed | cancelled
        self.returncode: int | None = None
        self.started = time.time()
        self.ended: float | None = None
        self.lines: deque[str] = deque(maxlen=MAX_LINES)
        self._subs: list[queue.Queue] = []
        self._lock = threading.Lock()
        self._proc: subprocess.Popen | None = None

    # --- phát log ---

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue()
        with self._lock:
            for line in self.lines:      # gửi lại phần đã có để không hụt đầu
                q.put(line)
            if self.state != "running":
                q.put(None)
            else:
                self._subs.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subs:
                self._subs.remove(q)

    def _emit(self, line: str) -> None:
        with self._lock:
            self.lines.append(line)
            for q in self._subs:
                q.put(line)

    def _finish(self, state: str, returncode: int | None) -> None:
        with self._lock:
            self.state = state
            self.returncode = returncode
            self.ended = time.time()
            for q in self._subs:
                q.put(None)
            self._subs.clear()

    # --- chạy ---

    def run(self) -> None:
        self._emit(f"$ {' '.join(self.argv)}")
        try:
            self._proc = subprocess.Popen(
                self.argv,
                cwd=REPO_ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                errors="replace",
            )
        except OSError as exc:
            self._emit(f"không chạy được: {exc}")
            self._finish("failed", None)
            return

        assert self._proc.stdout is not None
        for line in self._proc.stdout:
            self._emit(line.rstrip("\n"))
        rc = self._proc.wait()

        if self.state == "cancelled":
            self._emit("— đã dừng theo yêu cầu —")
            self._finish("cancelled", rc)
        else:
            self._finish("ok" if rc == 0 else "failed", rc)

    def cancel(self) -> bool:
        if self.state != "running" or self._proc is None:
            return False
        self.state = "cancelled"
        self._proc.terminate()
        return True

    def as_dict(self) -> dict:
        return {
            "id": self.id,
            "step": self.step,
            "label": self.label,
            "state": self.state,
            "returncode": self.returncode,
            "started": self.started,
            "ended": self.ended,
            "elapsed": round((self.ended or time.time()) - self.started, 1),
            "argv": self.argv,
        }


class Runner:
    """Giữ job đang chạy và lịch sử ngắn các job đã xong."""

    def __init__(self) -> None:
        self.current: Job | None = None
        self.history: deque[Job] = deque(maxlen=20)
        self._lock = threading.Lock()

    def busy(self) -> bool:
        return self.current is not None and self.current.state == "running"

    def start(self, step: str, argv: list[str], label: str) -> Job:
        with self._lock:
            if self.busy():
                raise RuntimeError(f"đang chạy bước '{self.current.step}', đợi xong đã")
            job = Job(step, argv, label)
            self.current = job
            self.history.appendleft(job)
        threading.Thread(target=job.run, daemon=True, name=f"job-{job.id}").start()
        return job

    def get(self, job_id: int) -> Job | None:
        if self.current and self.current.id == job_id:
            return self.current
        return next((j for j in self.history if j.id == job_id), None)


def script(name: str, *args: str) -> list[str]:
    """argv để chạy một script trong scripts/ bằng đúng interpreter của venv."""
    exe = PY if Path(PY).exists() else sys.executable
    return [exe, str(REPO_ROOT / "scripts" / name), *args]
