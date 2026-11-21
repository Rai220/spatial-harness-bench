import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from spatial_harness_bench.ascii_recall.task import FILES
from spatial_harness_bench.ascii_recall.verifier import check_drawing, parse_reference, score
from spatial_harness_bench.objects import assign


NAMES = "кошка собака слон жираф ёж сова рыба зонт стол дом".split()
REFERENCE = "\n".join(NAMES)


class TestAsciiRecall(unittest.TestCase):
    def test_picture_strokes_and_file_format(self):
        self.assertIsNone(check_drawing(b" /\\_/\\\r\n( o.o )\r\n", NAMES))
        for data in [b"", b"  \n", b"\x1b[0m", "◯".encode()]:
            with self.subTest(data=data):
                self.assertEqual(check_drawing(data, NAMES).kind, "invalid_art")

    def test_caption_check_independent_of_guess_or_language(self):
        for caption in ["<CAT>", "C A T", "кошка", "K.O.S.H.K.A", "|c|\n|a|\n|t|",
                        " /\\_/\\\n( o.o )\n   взгляд", "с т о л"]:
            with self.subTest(caption=caption):
                self.assertEqual(check_drawing(caption.encode(), NAMES).kind, "caption")

    def test_invalid_reference(self):
        for text in [REFERENCE.replace("сова", "ЁЖ"), REFERENCE.replace("сова", ""),
                     REFERENCE + "\nлишнее", REFERENCE.replace("сова", "кот"),
                     REFERENCE.replace("сова", "домик")]:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_reference(text.encode())

    def test_positional_scoring(self):
        valid = [None] * 10
        guesses = REFERENCE.upper().replace("ЁЖ", " ЕЖ ")
        self.assertEqual(score(NAMES, guesses.encode(), valid)["score"], 10)
        guesses = "\n".join([NAMES[1], NAMES[0], "", NAMES[3]])
        self.assertEqual(score(NAMES, guesses.encode(), valid)["score"], 1)
        self.assertEqual(score(NAMES, b"", valid)["score"], 0)
        # Word forms and listed synonyms count; a broader category does not.
        guesses = "\n".join(["кот", "собачка", "слоны", "жираф", "ёжик", "сова", "рыбка", "зонтик",
                             "стол", "здание"])
        self.assertEqual(score(NAMES, guesses.encode(), valid)["score"], 9)
        penalty = check_drawing(b"CAT", NAMES)
        self.assertEqual(score(NAMES, REFERENCE.encode(), [penalty] * 10)["score"], 0)
        self.assertEqual(score(NAMES, b"", [penalty] + valid[1:])["items"][0]["points"], -1)

    def test_caption_check_covers_synonyms(self):
        self.assertEqual(check_drawing(b"  KOT  \n /\\_/\\", NAMES).kind, "caption")

    def test_assigned_objects_replace_the_artists_choice(self):
        self.assertEqual(assign(5), assign(5))
        self.assertNotEqual(assign(5), assign(6))
        names = assign(5)
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = root / "harness.py"
            harness.write_text(
                "from pathlib import Path\nimport sys\n"
                f"names = {names!r}\n"
                "prompt = sys.stdin.read()\n"
                "if not list(Path.cwd().iterdir()):\n"
                "    assert 'Выбери' not in prompt\n"
                "    for i, name in enumerate(names):\n"
                "        assert f'{i}.txt — {name}' in prompt\n"
                "        Path(f'{i}.txt').write_text('/' * (i + 1))\n"
                "    Path('desc.txt').write_text('мусор')  # ignored: the runner owns the key\n"
                "else:\n"
                "    Path('desc.txt').write_text('\\n'.join(names[:3]))\n"
            )
            output = root / "report"
            result = subprocess.run([sys.executable, "-m", "spatial_harness_bench.ascii_recall",
                                     "--objects", "assigned", "--seed", "5", "--output", str(output),
                                     "--", sys.executable, str(harness)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            report = json.loads((output / "result.json").read_text())
            self.assertEqual((report["score"], report["objects"], report["seed"]), (3, "assigned", 5))
            self.assertEqual((output / "original/desc.txt").read_text().split(), names)

    def test_two_fresh_processes_and_workspace_handoff(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / "report"
            harness = root / "harness.py"
            harness.write_text(
                "from pathlib import Path\nimport sys\n"
                f"files = {FILES!r}\nnames = {REFERENCE!r}\n"
                "prompt = sys.stdin.read()\n"
                "if not list(Path.cwd().iterdir()):\n"
                "    assert 'Выбери' in prompt\n"
                "    Path('desc.txt').write_text(names)\n"
                "    Path('notes').write_text(names)\n"
                "    for i, f in enumerate(files):\n"
                "        Path(f).write_text('label' if i == 4 else '/' * (i+1))\n"
                "else:\n"
                "    assert 'Восстанови' in prompt\n"
                "    assert set(p.name for p in Path.cwd().iterdir()) == set(files)\n"
                "    for i, f in enumerate(files):\n"
                "        assert Path(f).read_text() == ('' if i == 4 else '/' * (i+1))\n"
                "    assert names not in prompt\n"
                "    Path('desc.txt').write_text(names)\n"
            )
            command = [sys.executable, "-m", "spatial_harness_bench.ascii_recall",
                       "--output", str(output), "--", sys.executable, str(harness)]
            result = subprocess.run(command, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual((output / "original" / "4.txt").read_text(), "label")
            self.assertEqual((output / "desc.txt").read_text(), REFERENCE)
            report = json.loads((output / "result.json").read_text())
            self.assertEqual(report["score"], 8)
            self.assertEqual(report["items"][4]["status"], "caption")
            self.assertNotEqual(subprocess.run(command, capture_output=True).returncode, 0)
