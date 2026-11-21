"""Re-derive every generated key from the SVG source alone, independently of the generator."""

import json
import math
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
import unittest
from xml.etree import ElementTree as ET

try:
    from spatial_harness_bench.svg_scenes import generated
except ModuleNotFoundError as exc:
    if exc.name not in {"cairosvg", "cairocffi", "PIL", "defusedxml"}:
        raise
    raise unittest.SkipTest("Install the svg extra to run generated scene tests.") from exc

from PIL import Image, ImageDraw

from spatial_harness_bench.answers import lemmas, normalize, same
from spatial_harness_bench.svg_scenes.__main__ import run
from spatial_harness_bench.svg_scenes.task import generated_scenes, grade

NS = "{http://www.w3.org/2000/svg}"
HEX_TO_COLOR = {code: name for name, code in generated.COLORS.items()}
SEEDS = range(6)


def figures(source: bytes) -> list[dict]:
    """Shapes in paint order with colour, class, outline points and centre, read from the markup."""
    result = []
    for node in ET.fromstring(source).iter():
        tag = node.tag.removeprefix(NS)
        if node.get("fill") not in HEX_TO_COLOR:
            continue
        if tag == "circle":
            cx, cy, r = (float(node.get(k)) for k in ("cx", "cy", "r"))
            points = [(cx + r * math.cos(t / 48 * math.tau), cy + r * math.sin(t / 48 * math.tau)) for t in range(48)]
            kind = "круг"
        elif tag == "polygon":
            values = [float(v) for v in re.split(r"[ ,]", node.get("points"))]
            points = list(zip(values[::2], values[1::2]))
            kind = {3: "треугольник", 6: "шестиугольник", 10: "звезда", 12: "крест"}.get(len(points))
            if len(points) == 4:
                d1, d2 = math.dist(points[0], points[2]), math.dist(points[1], points[3])
                kind = "квадрат" if abs(d1 - d2) < 1 else "ромб"
        else:
            continue
        xs, ys = zip(*points)
        result.append({"color": HEX_TO_COLOR[node.get("fill")], "kind": kind, "points": points,
                       "centre": (sum(xs) / len(xs), sum(ys) / len(ys)), "bbox": (min(xs), max(xs))})
    return result


def word_to(options, word: str) -> str:
    return next(o for o in options if normalize(o) in lemmas(word))


class TestGeneratedScenes(unittest.TestCase):
    def items(self):
        for seed in SEEDS:
            for item in generated.generate(seed, per_tier=1):
                yield seed, item

    def test_reproducible_and_seed_dependent(self):
        a, b = generated.generate(3, 1), generated.generate(3, 1)
        self.assertEqual([i.picture.source for i in a], [i.picture.source for i in b])
        self.assertNotEqual([i.question for i in a], [i.question for i in generated.generate(4, 1)])
        self.assertEqual(generated_scenes(3, 1)[1], generated_scenes(3, 1)[1])
        self.assertEqual(len(a), len(generated.KINDS) * len(generated.TIERS))

    def test_source_has_no_hints(self):
        for _, item in self.items():
            root = ET.fromstring(item.picture.source)
            for node in root.iter():
                self.assertFalse({"id", "class"} & set(node.attrib))
                self.assertFalse((node.text or "").strip())

    def test_count_keys(self):
        for seed, item in self.items():
            if item.kind != "count":
                continue
            with self.subTest(seed=seed, tier=item.tier):
                m = re.match(r"Сколько (\S+) (\S+) находится целиком (левее|правее)", item.question)
                color, kind = word_to(generated.COLORS, m[1]), word_to(generated.SHAPES, m[2])
                line = float(re.search(r'<line x1="([\d.]+)"', item.picture.source.decode())[1])
                count = 0
                for f in figures(item.picture.source):
                    x0, x1 = f["bbox"]
                    self.assertTrue(x1 < line - 4 or x0 > line + 4 or (x0 < line - 6 and x1 > line + 6))
                    if (f["color"], f["kind"]) == (color, kind):
                        count += (x1 < line) if m[3] == "левее" else (x0 > line)
                self.assertIn(str(count), item.answers[0])

    def test_cords_keys_by_chaining_shuffled_pieces(self):
        for seed, item in self.items():
            if item.kind != "cords":
                continue
            with self.subTest(seed=seed, tier=item.tier):
                root = ET.fromstring(item.picture.source)
                pieces = {}
                for node in root.iter(f"{NS}path"):
                    d = node.get("d")
                    start = tuple(re.match(r"M([\d.]+) ([\d.]+)", d).groups())
                    if "H" in d:
                        end = (d.rsplit("H", 1)[1], start[1])
                    else:
                        end = tuple(d.split()[-2:])
                    pieces[start] = end
                shape = word_to(generated.SHAPES, re.search(r"шнур от (\S+)\?", item.question)[1])
                device = next(f for f in figures(item.picture.source)
                              if f["kind"] == shape and f["color"] == "серый" and f["centre"][0] < 60)
                point = ("58", f"{device['centre'][1]:.1f}")
                for _ in range(len(pieces)):
                    point = pieces[point]
                    if point[0] == "440":
                        break
                socket = next(n for n in root.iter(f"{NS}rect")
                              if n.get("x") == "440" and abs(float(n.get("y")) + 18 - float(point[1])) < 0.2)
                self.assertEqual(item.answers, [[HEX_TO_COLOR[socket.get("fill")]]])
                self.assertFalse(abs(float(point[1]) - device["centre"][1]) < 1, "cord must not run straight")

    def test_transform_keys_with_independent_rotations(self):
        ops = {
            "поверни карточку на 90° по часовой стрелке": lambda g: [list(r) for r in zip(*g[::-1])],
            "поверни карточку на 90° против часовой стрелки": lambda g: [list(r) for r in zip(*g)][::-1],
            "поверни карточку на 180°": lambda g: [r[::-1] for r in g[::-1]],
            "отрази карточку слева направо, как в зеркале": lambda g: [r[::-1] for r in g],
            "отрази карточку сверху вниз": lambda g: g[::-1],
        }
        for seed, item in self.items():
            if item.kind != "transform":
                continue
            with self.subTest(seed=seed, tier=item.tier):
                shapes = figures(item.picture.source)
                n = math.isqrt(len(shapes))
                grid = [[None] * n for _ in range(n)]
                for f in shapes:
                    c, r = (int((v - 56) / (400 / n)) for v in f["centre"])
                    grid[r][c] = f"{f['color']} {f['kind']}"
                steps = re.match(r"Мысленно (.+?)\. Какие", item.question)[1].split(", затем ")
                for step in steps:
                    grid = ops[step](grid)
                cells = re.findall(r"\((\d),(\d)\)", item.question.split("Какие")[1])[:2]
                for (r, c), aliases in zip(cells, item.answers, strict=True):
                    self.assertTrue(same(aliases[0], grid[int(r) - 1][int(c) - 1]))

    def test_occlusion_keys_by_paint_order(self):
        for seed, item in self.items():
            if item.kind != "occlusion":
                continue
            with self.subTest(seed=seed, tier=item.tier):
                shapes = figures(item.picture.source)
                labels = Image.new("L", (512, 512), 0)
                draw = ImageDraw.Draw(labels)
                for i, f in enumerate(shapes, 1):
                    draw.polygon(f["points"], fill=i)
                masks = []
                for f in shapes:
                    mask = Image.new("L", (512, 512), 0)
                    ImageDraw.Draw(mask).polygon(f["points"], fill=1)
                    masks.append(mask)
                label_data = labels.tobytes()
                mask_data = [m.tobytes() for m in masks]

                def over(top, bottom):
                    return top > bottom and sum(
                        1 for p, inside in zip(label_data, mask_data[bottom]) if inside and p == top + 1) >= 150

                names = [f"{f['color']} {f['kind']}" for f in shapes]
                m = re.match(r"Какая фигура лежит поверх (\S+) (\S+) и сама частично закрыта (\S+) (\S+)\?",
                             item.question)
                a = names.index(f"{word_to(generated.COLORS, m[1])} {word_to(generated.SHAPES, m[2])}")
                b = names.index(f"{word_to(generated.COLORS, m[3])} {word_to(generated.SHAPES, m[4])}")
                middle = [x for x in range(len(shapes)) if x not in (a, b) and over(x, a) and over(b, x)]
                self.assertEqual(len(middle), 1)
                self.assertTrue(same(item.answers[0][0], names[middle[0]]))

    def test_grading_accepts_word_forms_and_pairs(self):
        scenes, _ = generated_scenes(5, 1)
        lines = []
        for scene in scenes:
            if scene.kind == "cords":
                lines.append(_adjective_feminine(scene.answers[0][0]))
            else:
                lines.append(", ".join(a[0].upper() for a in scene.answers))
        report = grade(scenes, "\n".join(lines).encode())
        self.assertEqual(report["score"], len(scenes))
        self.assertEqual(set(report["by_kind"]), set(generated.KINDS))
        swapped = [", ".join(a[0] for a in s.answers[::-1]) for s in scenes]
        self.assertLess(grade(scenes, "\n".join(swapped).encode())["score"], len(scenes))

    def test_runner_with_a_blank_adapter(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            adapter = root / "adapter.py"
            adapter.write_text("import os, sys\nsys.stdin.read()\n"
                               "n = int(os.environ['SVG_RECALL_COUNT'])\n"
                               "assert os.environ['SVG_RECALL_DIFFICULTY'] == 'generated'\n"
                               "assert sorted(os.listdir()) == sorted(f'{i}.svg' for i in range(n))\n"
                               "open('desc.txt', 'w').write('\\n' * n)\n")
            report = run([sys.executable, str(adapter)], root / "out", "code",
                         scene_set="generated", seed=11, per_tier=1)
            self.assertEqual(report["modes"]["code"]["score"], 0)
            self.assertEqual(report["modes"]["code"]["max_score"], 12)
            saved = json.loads((root / "out/scenes/items.json").read_text())
            self.assertEqual(len(saved), 12)
            self.assertEqual(json.loads((root / "out/result.json").read_text())["seed"], 11)


def _adjective_feminine(color: str) -> str:
    from spatial_harness_bench.answers import _morph
    parse = next(p for p in _morph().parse(color) if p.tag.POS == "ADJF")
    return parse.inflect({"nomn", "femn"}).word


if __name__ == "__main__":
    unittest.main()
