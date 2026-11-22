from io import BytesIO
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from xml.etree import ElementTree as ET

try:
    from spatial_harness_bench.svg_recall import hard, verifier
except ModuleNotFoundError as exc:
    if exc.name not in {"cairosvg", "cairocffi", "PIL", "defusedxml"}:
        raise
    raise unittest.SkipTest("Install the svg extra to run SVG recall tests.") from exc

from PIL import Image

from spatial_harness_bench.svg_recall.__main__ import run
from spatial_harness_bench.svg_recall.task import KITCHEN_PAIRS

NAMES = "кошка собака слон жираф ёж сова рыба зонт стол дом".split()


def replay(grid, instruction, *, fixed=False):
    """Move actual rows/columns/blocks using the Russian prompt, independently of the generator."""
    addresses = re.findall(r"\((\d+),(\d+)\)", instruction) if "предмет" in instruction else []
    if addresses:
        positions = []
        for row, col in addresses:
            initial = (int(row)-1)*4 + int(col)-1
            cell = initial if fixed else sum(grid, []).index(initial)
            positions.append(divmod(cell, 4))
        if instruction.startswith("поменять"):
            if "строки" in instruction:
                a, b = [p[0] for p in positions]
                grid[a], grid[b] = grid[b], grid[a]
            else:
                a, b = [p[1] for p in positions]
                for row in grid:
                    row[a], row[b] = row[b], row[a]
            return
        axis = 0 if instruction.startswith("строку") else 1
        instruction = f"{instruction.split()[0]} {positions[0][axis]+1} {instruction.split()[-1]}"
    parts = instruction.split()
    if parts[0] == "строку":
        row = grid[int(parts[1]) - 1]
        shift = 1 if parts[2] == "вправо" else -1
        row[:] = row[-shift:] + row[:-shift]
    elif parts[0] == "столбец":
        c = int(parts[1]) - 1
        col = [row[c] for row in grid]
        shift = 1 if parts[2] == "вниз" else -1
        for row, value in zip(grid, col[-shift:] + col[:-shift], strict=True):
            row[c] = value
    else:
        if parts[0] == "квадрат":
            r, c = (int(t) - 1 for t in parts[1].strip("()").split(","))
            size = 2
        else:
            r, c, size = 0, 0, len(grid)
        block = [row[c:c+size] for row in grid[r:r+size]]
        rotated = list(zip(*block[::-1])) if "против" not in parts else list(zip(*block))[::-1]
        for i, row in enumerate(rotated):
            grid[r+i][c:c+size] = row


class TestSvgHard(unittest.TestCase):
    def test_published_operations_solve_every_question(self):
        self.assertEqual(hard.make_questions(42), hard.make_questions(42))
        self.assertNotEqual(hard.make_questions(42), hard.make_questions(43))
        covered = set()
        for seed in range(25):
            board, questions = hard.make_questions(seed)
            self.assertEqual({board[q.origin] for q in questions}, set(range(10)))
            self.assertEqual(tuple(len(q.moves) for q in questions), hard.COUNTS)
            for q in questions:
                grid = [list(range(r, r + 4)) for r in range(0, 16, 4)]
                visited = {q.origin}
                for move in q.moves:
                    covered.add(move.text)
                    replay(grid, move.text)
                    current = sum(grid, []).index(q.origin)
                    self.assertNotIn(current, visited)
                    visited.add(current)
                self.assertEqual(sum(grid, [])[q.end], q.origin)
        self.assertEqual(covered, {m.text for m in hard.MOVES})

    def test_expert_needs_current_landmarks_and_uses_every_object_as_target(self):
        target_also_used_as_landmark = False
        for seed, level in [(s, "expert") for s in range(10)] + [(s, "master") for s in range(4)]:
            board, questions = hard.make_questions(seed, difficulty=level)
            self.assertEqual(board, hard.make_questions(seed)[0])
            self.assertEqual({board[q.origin] for q in questions}, set(range(10)))
            self.assertEqual(tuple(len(q.moves) for q in questions), hard.LEVEL_COUNTS[level])
            for q in questions:
                grid = [list(range(r, r+4)) for r in range(0, 16, 4)]
                fixed = [row.copy() for row in grid]
                for move in q.moves:
                    replay(grid, move.text)
                    replay(fixed, move.text, fixed=True)
                    if "предмет" in move.text:
                        r, c = hard.coordinates(q.origin)
                        target_also_used_as_landmark |= f"({r},{c})" in move.text
                self.assertEqual(sum(grid, [])[q.end], q.origin)
                self.assertNotEqual(sum(fixed, [])[q.end], q.origin)
        self.assertTrue(target_also_used_as_landmark)

    def test_kitchen_rejects_incomplete_pairs_even_with_ten_distinct_names(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            names = [name for pair in KITCHEN_PAIRS[:4] for name in pair]
            names += [KITCHEN_PAIRS[4][0], KITCHEN_PAIRS[5][1]]
            (folder / "desc.txt").write_text("\n".join(names))
            with self.assertRaisesRegex(ValueError, "пяти полных пар"):
                run([], folder / "report", "code", drawings_dir=folder, objects="kitchen")

    def test_composed_svg_and_pixels_match_even_with_repeated_gradient_ids(self):
        pictures = []
        for color in ["red", "blue"]:
            source = (f'<svg xmlns="{verifier.SVG}" viewBox="0 0 10 10"><defs>'
                      f'<linearGradient id="g1"><stop stop-color="{color}"/></linearGradient></defs>'
                      '<rect width="10" height="10" style="fill:url(#g1)"/></svg>').encode()
            pictures.append(verifier.Picture(source, b""))
        board = [None] * 16
        board[3], board[8] = 1, 0
        svg, png = hard.render_board(board, pictures)
        ids = [node.get("id") for node in ET.fromstring(svg).iter() if "id" in node.attrib]
        self.assertEqual(len(set(ids)), 2)
        image = Image.open(BytesIO(png)).convert("RGB")
        for i, want in [(3, (0, 0, 255)), (8, (255, 0, 0)), (4, (255, 255, 255))]:
            r, c = divmod(i, 4)
            self.assertEqual(image.getpixel((256*c + 128, 256*r + 128)), want)

    def test_origin_scores_names_are_reported_and_penalty_applies_once(self):
        board, questions = hard.make_questions(37)
        rows = []
        for q in questions:
            r, c = hard.coordinates(q.origin)
            rows.append(f"{r},{c} | {NAMES[board[q.origin]]}")
        issues = [None] * 10
        def score():
            return hard.grade(board, questions, NAMES, issues, "\n".join(rows).encode())
        self.assertEqual(score()["score"], 10)
        rows[0] = rows[0].split("|")[0] + "| другой предмет"
        rows[1] = "0,0 |" + rows[1].split("|")[1]
        issues[board[questions[2].origin]] = verifier.DrawingIssue("caption", "подпись")
        # Q1 has the right cell and a wrong name (1 point), Q2 the reverse (0), Q3 a caption (−1).
        self.assertEqual((score()["score"], score()["origin_hits"], score()["name_hits"], score()["full_hits"]),
                         (7, 8, 8, 7))
        cat = next(i for i, q in enumerate(questions) if NAMES[board[q.origin]] == "кошка")
        if cat > 2:  # a synonym still counts as a correct name
            rows[cat] = rows[cat].split("|")[0] + "| Кот"
            self.assertEqual(score()["name_hits"], 8)
        self.assertEqual(hard.grade(board, questions, NAMES, issues, b"")["score"], 0)

    def test_replay_passes_only_one_modality_and_public_questions(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            originals = root / "original"
            originals.mkdir()
            (originals / "desc.txt").write_text("\n".join(NAMES))
            for i in range(10):
                (originals / f"{i}.svg").write_text(
                    f'<svg xmlns="{verifier.SVG}" viewBox="0 0 20 20"><circle r="{i+1}"/></svg>')
            adapter = root / "fake.py"
            adapter.write_text(
                "import os,sys\nfrom pathlib import Path\np=sys.stdin.read()\n"
                "stage=os.environ['SVG_RECALL_STAGE'];assert stage == 'code'\n"
                "assert os.environ['SVG_RECALL_DIFFICULTY']=='hard'\n"
                "assert {f.name for f in Path.cwd().iterdir()} == {'board.svg'}\n"
                "assert 'Конечная клетка' in p and 'кошка' not in p and 'seed' not in p\n"
                "Path('desc.txt').write_text('1,1 | неизвестно\\n'*10)\n")
            output = root / "report"
            with patch.object(verifier, "pixel_caption", return_value=None):
                report = run([sys.executable, str(adapter)], output, "code", seed=42,
                             difficulty="hard", drawings_dir=originals)
            self.assertEqual(report["drawing_stage"], "reused")
            self.assertEqual(report, json.loads((output / "result.json").read_text()))
            self.assertEqual(set(report["modes"]), {"code"})
            self.assertEqual((output / "questions.txt").read_text().count("Конечная клетка"), 10)
