"""Task 05: replay clip index and cold page queries."""
import os
import shutil
import stat
from pathlib import Path

from . import clip_index, clip_query

TASK_ID = "05-arcade-clips"
ROOT = Path(__file__).parent
REPLAY = ROOT / "replay.jsonl"
PROMPT = ("В каталоге есть replay.jsonl и детерминированный движок engine.py. "
          + (clip_index.PROMPT + clip_query.PROMPT).strip())


def setup(folder: Path) -> None:
    for name in ("engine.py", "replay.jsonl"):
        shutil.copyfile(ROOT / name, folder / name)


def read_submission(path: Path) -> bytes:
    """Only bounded regular files can cross the workspace boundary."""
    limit = 4_000_000
    try:
        fd = os.open(path, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW)
    except OSError:
        return b""
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            return b""
        with open(fd, "rb", closefd=False) as stream:
            data = stream.read(limit + 1)
    finally:
        os.close(fd)
    return data if len(data) <= limit else b""


def grade(index: bytes, program: bytes) -> tuple[int, str]:
    if index != clip_index.build_index(REPLAY):
        return 0, "Invalid replay index"
    try:
        if program and clip_query.verify(program, REPLAY):
            return 1, "OK"
    except (ValueError, RecursionError, TimeoutError, BrokenPipeError, ConnectionResetError):
        pass
    return 0, "Index query failed"
