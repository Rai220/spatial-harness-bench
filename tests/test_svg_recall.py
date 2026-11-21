import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

try:
    from spatial_harness_bench.svg_recall import verifier
except ModuleNotFoundError as exc:
    if exc.name not in {"cairosvg", "cairocffi", "PIL", "defusedxml"}:
        raise
    raise unittest.SkipTest("Install the svg extra to run SVG recall tests.") from exc

from spatial_harness_bench.svg_recall.__main__ import run

NAMES = "кошка собака слон жираф ёж сова рыба зонт стол дом".split()
SHAPE = '<circle cx="64" cy="64" r="24" fill="#d02"/>'


def svg(body):
    return f'<svg xmlns="{verifier.SVG}" viewBox="0 0 128 128">{body}</svg>'.encode()


class TestSvgRecall(unittest.TestCase):
    def test_text_labels_are_checked_before_either_mode_even_if_invisible(self):
        for data in [
            svg('<text opacity="0">ёж</text>'), svg('<title>cat</title>'), svg('<desc>CAT</desc>'),
            svg('<metadata>кошка</metadata>'), svg('<!-- фонарик -->'), b'<!-- cat -->' + svg(SHAPE),
            svg('<g id="lighthouse"/>'), svg('<g class="dog"/>'), svg('<g aria-label="кошка"/>'),
            svg('<g data-x="CAT"/>'), svg('<g something="K.O.S.H.K.A"/>'),
            svg(SHAPE).replace(b"<svg ", '<svg xmlns:кот="urn:кошка" '.encode()),
            svg('<g fill="url(https://example.com)"/><text>&#x43a;от</text>'),
        ]:
            with self.subTest(data=data):
                picture = verifier.inspect(data, NAMES)
                self.assertEqual(picture.issue.kind, "caption")
                self.assertEqual(picture.png, verifier.BLANK_PNG)
                self.assertEqual(picture.source, verifier.BLANK_SVG)

    def test_valid_art_and_unrenderable_inputs(self):
        art = ('<defs><linearGradient id="g1"><stop offset="0" stop-color="blue"/>'
               '<stop offset="1" stop-color="red"/></linearGradient></defs>'
               '<g transform="translate(1,1)"><path d="M10 10 L100 10 L50 80 Z" '
               'style="fill:url(#g1);stroke:blue;stroke-width:2"/></g>')
        self.assertIsNone(verifier.inspect(svg(art), NAMES).issue)
        sized = svg(SHAPE).replace(b'viewBox="0 0 128 128"', b'width="128" height="128"')
        self.assertIsNone(verifier.inspect(sized, NAMES).issue)
        for data in [b"", b"bad", svg(""), svg('<circle r="0"/>'), svg('<script/>'),
                     svg('<image href="file:///never"/>'), svg('<rect fill="url(https://example.com)"/>'),
                     b'<!DOCTYPE svg>' + svg(SHAPE)]:
            with self.subTest(data=data):
                self.assertEqual(verifier.inspect(data, NAMES).issue.kind, "invalid_art")

    def test_caption_drawn_entirely_with_paths(self):
        # Turn glyphs into actual outlines before verification. The SVG has no text objects.
        cairo = verifier.cairocffi
        ctx = cairo.Context(cairo.ImageSurface(cairo.FORMAT_ARGB32, 128, 128))
        ctx.select_font_face("DejaVu Sans", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(31)
        ctx.move_to(10, 74)
        ctx.text_path("TIGER")
        commands = "MLCZ"
        d = " ".join(commands[k] + " ".join(f"{n:.3f}" for n in coords)
                     for k, coords in ctx.copy_path())
        picture = verifier.inspect(svg(f'<path d="{d}"/>'), NAMES)
        self.assertEqual(picture.issue.kind, "caption")
        self.assertIn("OCR", picture.issue.reason)

    def test_two_modes_one_draw_and_isolation(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            harness = root / "harness.py"
            harness.write_text(
                "from pathlib import Path\nimport sys, os\n"
                f"names = {NAMES!r}\nart = {svg(SHAPE)!r}\n"
                "stage = os.environ['SVG_RECALL_STAGE']; prompt = sys.stdin.read()\n"
                "with Path(sys.argv[1]).open('a') as log: print(stage, Path.cwd(), file=log)\n"
                "files = {p.name for p in Path.cwd().iterdir()}\n"
                "if stage == 'draw':\n"
                "    assert not files and 'потом самостоятельно узнать' in prompt\n"
                "    Path('secret').write_text('drawing notes')\n"
                "    for i in range(9):\n"
                "        Path(f'{i}.svg').write_bytes(art.replace(b'</svg>', b'<title>cat</title></svg>') if i == 4 else art)\n"
                "elif stage == 'code':\n"
                "    assert files == {f'{i}.svg' for i in range(10)}\n"
                "    assert 'title' not in Path('4.svg').read_text() and 'circle' in Path('0.svg').read_text()\n"
                "    names[0], names[1] = names[1], names[0]\n"
                "    Path('secret').write_text('first guesses')\n"
                "else:\n"
                "    assert files == {f'{i}.png' for i in range(10)}\n"
                "    assert all(Path(f).read_bytes().startswith(b'\\x89PNG') for f in files)\n"
                "    assert Path('4.png').read_bytes() == Path('9.png').read_bytes() != Path('0.png').read_bytes()\n"
                "    if len(sys.argv) > 2: sys.exit(2)\n"
                "Path('desc.txt').write_text('\\n'.join(names) + '\\n')\n",
                encoding="utf-8",
            )
            log = root / "sessions"
            cmd = [sys.executable, str(harness), str(log)]
            output = root / "out"
            with patch.object(verifier, "pixel_caption", return_value=None):
                report = run(cmd, output, "both")
                self.assertEqual(report, json.loads((output / "result.json").read_text()))
                self.assertEqual(report["checks"][4]["status"], "caption")
                self.assertEqual([r["score"] for r in report["modes"].values()], [5, 7])
                self.assertEqual(report["modes"]["code"]["items"][4]["points"], -1)
                self.assertEqual(report["modes"]["vision"]["items"][9]["status"], "invalid_art")
                calls = [s.split() for s in log.read_text().splitlines()]
                self.assertEqual([stage for stage, _ in calls], ["draw", "code", "vision"])
                self.assertEqual(len({folder for _, folder in calls}), 3)
                failed = run([*cmd, "no-image-support"], root / "failed", "vision")
                self.assertEqual(failed["modes"], {"vision": {"score": None, "status": "error", "returncode": 2}})
            self.assertTrue((output / "original/4.svg").read_bytes().endswith(b'</title></svg>'))
            self.assertTrue((output / "rendered/4.png").is_file())
            self.assertNotEqual((output / "code/desc.txt").read_text(), (output / "vision/desc.txt").read_text())
