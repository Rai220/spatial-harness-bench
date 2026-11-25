from functools import lru_cache
from itertools import accumulate
import json
import struct
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from spatial_harness_bench.arcade_clips.__main__ import run
from spatial_harness_bench.arcade_clips.clip_index import build_index
from spatial_harness_bench.arcade_clips.task import PROMPT, REPLAY, ROOT, TASK_ID, grade, read_submission


@lru_cache(maxsize=2)
def replay_oracle(path):
    """Apply the published scoring rules per rider, without the production engine."""
    lines = path.read_bytes().splitlines(keepends=True)
    rows = [json.loads(line) for line in lines]
    points = [0] * len(rows)
    for player, position in (("p1", 0), ("p2", 8)):
        energy = 12
        for tick, row in enumerate(rows):
            pad = row[player]
            position += (("left", "idle", "right").index(pad["direction"]) - 1) * (2 if pad["jump"] else 1)
            if not pad["fire"]:
                energy += energy < 12
            elif energy >= 3:
                energy -= 3
                distance = abs((position - tick // 7 + 8) % 17 - 8)
                points[tick] += {0: 4, 1: 1}.get(distance, 0)
    return tuple(accumulate(map(len, lines), initial=0)), tuple(
        sum(points[i:i+64]) for i in range(len(points) - 63)
    )


@lru_cache(maxsize=1)
def reference_index():
    """Encode the independently calculated offsets and keys by the RIX1 spec."""
    offsets, keys = replay_oracle(REPLAY)
    parts = [struct.pack("<4sII", b"RIX1", len(keys), len(offsets) - 1)]
    for i in range(0, len(offsets), 128):
        offsets_page = offsets[i:i+128]
        deltas = [b - a for a, b in zip(offsets_page, offsets_page[1:])]
        parts.append(struct.pack("<I127H", offsets_page[0], *deltas, *([0] * (127 - len(deltas)))))

    for shift in range(7, -1, -1):
        bits = [(key >> shift) & 1 for key in keys]
        parts.append(struct.pack("<I", bits.count(0)))
        prefix = 0
        for i in range(0, len(bits), 128):
            page = bits[i:i+128]
            packed = int("".join(map(str, reversed(page))), 2)
            parts.append(struct.pack("<I", prefix) + packed.to_bytes(16, "little"))
            prefix += sum(page)
        keys = sorted(keys, key=lambda key: (key >> shift) & 1)
    return b"".join(parts)


class ArcadeClipsTests(unittest.TestCase):
    def test_index_format_and_quantiles(self):
        raw = build_index(REPLAY)
        magic, n, total = struct.unpack_from("<4sII", raw)
        assert (magic, total, n) == (b"RIX1", 30584, 30521)
        offsets, gains = replay_oracle(REPLAY)
        for tick, offset in enumerate(offsets):
            group, pos = divmod(tick, 128)
            base, *diffs = struct.unpack_from("<I127H", raw, 12 + group * 258)
            self.assertEqual(base + sum(diffs[:pos]), offset)
        self.assertTrue(raw == reference_index(), "RIX1 bytes differ from the independent implementation")
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

    def test_independent_oracle_on_hand_scored_replay(self):
        # P1 gets 4 + 1 + 1 + 4, then cannot fire with no charge.
        # At tick 64 P2 (lane 8) shoots next to the target on lane 9.
        idle = {"direction": "idle", "jump": False, "fire": False}
        p1 = [
            {**idle, "fire": True},
            {**idle, "direction": "right", "fire": True},
            {"direction": "left", "jump": True, "fire": True},
            {**idle, "direction": "right", "fire": True},
            {**idle, "fire": True},
        ]
        rows = [{"tick": i, "p1": p1[i] if i < len(p1) else idle,
                 "p2": {**idle, "fire": True} if i == 64 else idle} for i in range(65)]
        with TemporaryDirectory() as tmp:
            replay = Path(tmp) / "small.jsonl"
            replay.write_text("".join(json.dumps(row) + "\n" for row in rows))
            self.assertEqual(replay_oracle(replay)[1], (10, 7))

    def test_runner_with_reference_solution(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            # This adapter emulates a submission; the runner still grades it
            # through all twelve isolated queries.
            (root / "reference.rix").write_bytes(reference_index())
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
        index = reference_index()
        self.assertEqual(grade(index[:-1], b""), (0, "Invalid replay index"))
        for source in (b'import json; print(json.dumps(["answer", 0, 0]))', b'print("malformed")'):
            self.assertEqual(grade(index, source), (0, "Index query failed"))
        with TemporaryDirectory() as tmp:
            link = Path(tmp) / "query.py"
            link.symlink_to(ROOT / "clip_query_gold.py")
            self.assertEqual(read_submission(link), b"")
