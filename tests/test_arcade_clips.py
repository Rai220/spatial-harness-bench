import json
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from spatial_harness_bench.arcade_clips.__main__ import run
from spatial_harness_bench.arcade_clips.clip_index import build_index
from spatial_harness_bench.arcade_clips.engine import advance, frames, initial
from spatial_harness_bench.arcade_clips.task import PROMPT, REPLAY, ROOT, TASK_ID, grade, read_submission


class ArcadeClipsTests(unittest.TestCase):
    def test_index_format_and_quantiles(self):
        raw = build_index(REPLAY)
        magic, n, total = struct.unpack_from("<4sII", raw)
        assert (magic, total, n) == (b"RIX1", 30584, 30521)
        state = initial()
        points = [0]
        for offset, _, frame in frames(REPLAY):
            group, pos = divmod(frame["tick"], 128)
            base, *diffs = struct.unpack_from("<I127H", raw, 12 + group * 258)
            assert base + sum(diffs[:pos]) == offset
            advance(state, frame)
            points.append(sum(rider[2] for rider in state))
        gains = [a - b for a, b in zip(points[64:], points[:-64], strict=True)]
        pages = (n + 127) // 128
        start = 12 + (total // 128 + 1) * 258
        for lo, hi, k in [(0, n, 129), (111, 1999, 14), (n - 5, n, 4)]:
            expected = sorted(gains[lo:hi])[k]
            value = 0
            for level in range(8):
                header = start + level * (4 + pages * 20)
                zeros, = struct.unpack_from("<I", raw, header)
                ranks = []
                for i in (lo, hi):
                    page, bit = divmod(i, 128)
                    prefix, mask = struct.unpack_from("<I16s", raw, header + 4 + page * 20)
                    ranks.append(prefix + (int.from_bytes(mask, "little") % (1 << bit)).bit_count())
                count = hi - lo - (ranks[1] - ranks[0])
                value <<= 1
                if k >= count:
                    k -= count
                    lo, hi = (zeros + rank for rank in ranks)
                    value += 1
                else:
                    lo -= ranks[0]
                    hi -= ranks[1]
            assert value == expected

    def test_runner_with_reference_solution(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # This adapter emulates a submission; the runner still grades it
            # through all twelve isolated queries.
            (root / "reference.rix").write_bytes(build_index(REPLAY))
            adapter = root / "submit.py"
            adapter.write_text(
                "import shutil, sys\nfrom pathlib import Path\n"
                "assert set(p.name for p in Path.cwd().iterdir()) == {'engine.py', 'replay.jsonl'}\n"
                "assert sys.stdin.read() == Path(sys.argv[3]).read_text()\n"
                "shutil.copyfile(sys.argv[1], 'clips.rix')\n"
                "shutil.copyfile(sys.argv[2], 'query.py')\n"
            )
            (root / "prompt.txt").write_text(PROMPT)
            output = root / "run"
            result = run([sys.executable, str(adapter), str(root / "reference.rix"),
                          str(ROOT / "clip_query_gold.py"), str(root / "prompt.txt")], output)
            self.assertEqual(result, json.loads((output / "result.json").read_text()))
            self.assertEqual((result["task"], result["score"], result["max_score"]),
                             (TASK_ID, 1, 1))
            self.assertEqual((output / "clips.rix").read_bytes(), (root / "reference.rix").read_bytes())

    def test_invalid_submissions(self):
        index = build_index(REPLAY)
        self.assertEqual(grade(index[:-1], b""), (0, "Invalid replay index"))
        for source in (b'import json; print(json.dumps(["answer", 0, 0]))', b'print("malformed")'):
            self.assertEqual(grade(index, source), (0, "Index query failed"))
        with TemporaryDirectory() as tmp:
            link = Path(tmp) / "query.py"
            link.symlink_to(ROOT / "clip_query_gold.py")
            self.assertEqual(read_submission(link), b"")
