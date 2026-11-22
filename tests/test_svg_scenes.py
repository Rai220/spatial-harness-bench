from hashlib import sha256
import json
from pathlib import Path
import shutil
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    from spatial_harness_bench.svg_scenes import task
    from spatial_harness_bench.svg_scenes.__main__ import run
except ModuleNotFoundError as exc:
    if exc.name not in {"cairosvg", "cairocffi", "PIL", "defusedxml"}:
        raise
    raise unittest.SkipTest("Install the svg extra to run the fixed scene suite.") from exc


class FixedScenesTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scenes, cls.digest = task.load_scenes()

    def test_versioned_bank_and_positional_rubric(self):
        self.assertEqual(self.digest, "882289c9526a125c60e924a20e8debfe94b7c99c8242e4846cb03fed01a348a0")
        lines = [", ".join(group[0] for group in scene.answers) for scene in self.scenes]
        lines[0], lines[5], lines[8] = "КЛЮЧИК.", "три", "булавка и ножницы"
        self.assertEqual(task.grade(self.scenes, "\n".join(lines).encode())["score"], 10)
        lines[0], lines[3], lines[8] = "ключ или зонт", "", "ножницы, булавка"
        self.assertEqual(task.grade(self.scenes, "\n".join(lines).encode())["score"], 7)

    def test_only_svg_source_is_visible_in_recognition_workspace(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            code_hashes = [sha256(s.picture.source).hexdigest() for s in self.scenes]
            adapter = root / "adapter.py"
            adapter.write_text(
                "import hashlib, os, sys\nfrom pathlib import Path\n"
                "stage=os.environ['SVG_RECALL_STAGE']; assert stage == 'code'\n"
                "assert os.environ['SVG_RECALL_DIFFICULTY']=='fixed'\n"
                "assert {p.name for p in Path.cwd().iterdir()}=={f'{i}.svg' for i in range(10)}\n"
                f"hashes={code_hashes!r}\n"
                "assert [hashlib.sha256(Path(f'{i}.svg').read_bytes()).hexdigest() for i in range(10)]==hashes\n"
                "p=sys.stdin.read(); assert f'9.svg —' in p and 'монгольфьер' not in p and 'палатка' not in p\n"
                "with Path(sys.argv[1]).open('a') as f: print(stage, str(Path.cwd()), file=f)\n"
                "Path('desc.txt').write_text('ключ\\n')\n",
                encoding="utf-8",
            )
            log = root / "log"
            result = run([sys.executable, str(adapter), str(log)], root / "report", "code")
            sessions = [line.split() for line in log.read_text().splitlines()]
            self.assertEqual([s[0] for s in sessions], ["code"])
            with self.assertRaises(ValueError):
                run([], root / "removed", "both")
            self.assertEqual([row["score"] for row in result["modes"].values()], [1])
            self.assertEqual(result, json.loads((root/"report/result.json").read_text()))

    def test_defective_authored_picture_stops_before_grading(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            bank = Path(task.__file__).parent
            shutil.copy(bank/"items.json", folder)
            (folder/"art").mkdir()
            (folder/"art/0.svg").write_text('<svg viewBox="0 0 40 40"><title>ключ</title></svg>')
            with patch.object(task, "files", return_value=folder), self.assertRaisesRegex(ValueError, "0.svg"):
                run([], folder/"report", "code")
            self.assertFalse((folder/"report").exists())
