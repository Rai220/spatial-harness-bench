import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from spatial_harness_bench.answers import duplicates, same
from spatial_harness_bench.objects import POOL
from spatial_harness_bench.stats import collect, summarize, wilson


class TestAnswers(unittest.TestCase):
    def test_equivalence(self):
        for answer, reference in [("кот", "кошка"), ("Ёлочка", "ель"), ("сиреневой", "фиолетовый"),
                                  ("синяя звезда", "синий звезда"), ("ЗОНТИКИ", "зонт")]:
            with self.subTest(answer=answer):
                self.assertTrue(same(answer, reference))
        for answer, reference in [("ёлка", "дерево"), ("ковш", "кастрюля"), ("сито", "дуршлаг"),
                                  ("кружка", "чашка"), ("", "кот"), ("кофейная чашка", "чашка")]:
            with self.subTest(answer=answer):
                self.assertFalse(same(answer, reference))

    def test_pool_is_free_of_synonyms(self):
        self.assertEqual(duplicates(list(POOL)), [])


class TestStats(unittest.TestCase):
    def test_wilson_matches_reference_values(self):
        lo, hi = wilson(5, 10)
        self.assertAlmostEqual(lo, 0.2366, places=3)
        self.assertAlmostEqual(hi, 0.7634, places=3)
        self.assertEqual(wilson(0, 0), (0.0, 1.0))

    def test_runs_items_and_nested_suites(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            for i, score in enumerate([6, 8, 7]):
                run = root / f"rep-{i}" / "model"
                run.mkdir(parents=True)
                items = [{"points": 1}] * score + [{"points": 0}] * (10 - score)
                (run / "result.json").write_text(json.dumps({"task": "04", "modes": {
                    "code": {"status": "complete", "score": score, "max_score": 10, "items": items},
                    "vision": {"status": "error", "score": None}}}))
                suite = root / f"rep-{i}" / "maze"
                (suite / "level-01").mkdir(parents=True)
                (suite / "level-01" / "result.json").write_text(json.dumps({"task": "02-maze-v2-01"}))
                (suite / "suite.json").write_text(json.dumps({
                    "suite": "02-maze-v2", "status": "complete", "score": 1, "max_score": 2, "efficiency": 0.25,
                    "levels": [{"status": "solved"}, {"status": "limit"}]}))
            groups = collect([f"m={root}/rep-*/model", f"z={root}/rep-*/maze"])
            code = summarize(("m", "04", "code"), groups[("m", "04", "code")])
            self.assertEqual((code["runs"], code["mean"], code["items"]), (3, 7, 30))
            half = 4.303 * 1 / math.sqrt(3)
            self.assertAlmostEqual(code["runs_ci95"][0], 7 - half)
            self.assertEqual(groups[("m", "04", "vision")].errors, 3)
            maze = summarize(("z", "02-maze-v2", "-"), groups[("z", "02-maze-v2", "-")])
            self.assertEqual((maze["runs"], maze["items"], maze["efficiency"]), (3, 6, 0.25))


if __name__ == "__main__":
    unittest.main()
