"""Cold-process query verifier. Only the player's program enters the sandbox."""
import asyncio
import contextlib
import json
import os
import random
import struct
import tempfile
from pathlib import Path

from .clip_index import build_index
from .engine import advance, frames, initial
from .sandbox import query_command


async def _query(root, data, n, left, right, k, expected):
    pages = (n + 127) // 128
    start = 12 + ((n + 63) // 128 + 1) * 258
    headers = [start + i * (4 + 20 * pages) for i in range(8)]
    zeros = [struct.unpack_from("<I", data, h)[0] for h in headers]
    argv = query_command(root / "query.py")
    proc = await asyncio.create_subprocess_exec(
        *argv, stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL, cwd=root,
        env={"PATH": os.defpath}, limit=4096)
    try:
        reply = [n, left, right, k, zeros]
        for reads in range(97):
            proc.stdin.write(json.dumps(reply).encode() + b"\n")
            await proc.stdin.drain()
            line = await proc.stdout.readline()
            if not line:
                return False
            msg = json.loads(line)
            if (isinstance(msg, list) and len(msg) == 3 and msg[0] == "answer"
                    and all(type(value) is int for value in msg[1:]) and msg[1:] == expected):
                return True
            if not (reads < 96 and isinstance(msg, list) and len(msg) == 3 and msg[0] == "page"
                    and type(msg[1]) is int and 0 <= msg[1] < 8
                    and type(msg[2]) is int and 0 <= msg[2] < pages):
                return False
            prefix, bits = struct.unpack_from("<I16s", data, headers[msg[1]] + 4 + 20 * msg[2])
            reply = [prefix, format(int.from_bytes(bits, "little"), "032x")]
    finally:
        if proc.returncode is None:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
        proc.stdin.close()
        await proc.communicate()


def verify(source, replay):
    rng = random.Random(533022)
    with tempfile.TemporaryDirectory(prefix="arcade-clip-query-") as name:
        root = Path(name)
        archive = root / "archive.jsonl"
        # Replays use the same controls with different initial conditions and lengths.
        rows = [row for _, _, row in frames(replay)]
        for offset, length in [(0, len(rows)), (3071, 17199), (9164, 8255)]:
            scores = [0]
            state = initial()
            with archive.open("w", encoding="utf-8", newline="\n") as out:
                for tick, row in enumerate(rows[offset:offset + length]):
                    row = dict(row, tick=tick)
                    out.write(json.dumps(row, separators=(",", ":")) + "\n")
                    advance(state, row)
                    scores.append(sum(p[2] for p in state))
            data = build_index(archive)
            gains = [a - b for a, b in zip(scores[64:], scores[:-64], strict=True)]
            n = len(gains)
            for left, right in [(0, n), (127, 129), (n - 1, n),
                                (rng.randrange(n // 3), rng.randrange(2 * n // 3, n))]:
                k = rng.randrange(right - left)
                tick = sorted(range(left, right), key=lambda i: (-gains[i], i))[k]
                # Each query gets a clean filesystem, interpreter and page budget.
                with tempfile.TemporaryDirectory(dir=root) as directory:
                    clean = Path(directory)
                    (clean / "query.py").write_bytes(source)
                    ok = asyncio.run(asyncio.wait_for(
                        _query(clean, data, n, left, right, k, [tick, gains[tick]]), 5))
                    if not ok:
                        return False
    return True


PROMPT = """
Также сдай query.py (Python 3.12, стандартная библиотека). Верификатор запускает
его отдельно для каждого запроса в пустом каталоге, в том числе для других
реплеев, индексированных по тому же формату. Файл индекса при поиске находится
на стороне верификатора. Каждый процесс получает в stdin одну строку JSON
[N,L,R,k,Z], Z — массив восьми чисел нулей по уровням от старшего бита.
Среди стартовых тиков L <= i < R упорядочь клипы по убыванию ключа, равные —
по возрастанию i. Найди элемент с номером k от нуля.
Чтобы прочитать страницу, выведи строку JSON ["page",уровень,страница], номера от
нуля, и сбрось буфер stdout. Из stdin придёт [число_единиц_перед_страницей,маска]:
маска — 32 hex-цифры числового значения little-endian поля (младший бит справа).
В конце выведи ["answer",i,ключ]. Доступно до 96 запросов страниц и 5 секунд
на запрос монтажёра, 1 <= N <= 30521, 0 <= L <= k+L < R <= N.
В stdout допустимы только сообщения протокола. Для проверки используются полные
и укороченные матчи, неполные страницы и равные ключи."""
