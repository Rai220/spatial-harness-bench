import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from spatial_harness_bench.building import task
from spatial_harness_bench.building.__main__ import run


# Hand-checked against the floor descriptions, in common building coordinates.
ANSWERS = [
    {"apartment": 203},
    {"apartments": [101, 102, 104]},
    {"cells": ["B3", "C3"]},
    {"count": 1, "cells": ["C3", "C4", "D3"]},
    {"inlet": "A5", "outlet": "A3", "apartments": [106, 204, 303, 503], "length": 2},
]


class BuildingTests(unittest.TestCase):
    def test_native_and_mirrored_descriptions(self):
        self.assertIn("A–B — 201", task.floor_description(2))
        third = task.floor_description(3)
        self.assertIn("с востока на запад", third)
        self.assertIn("Ряд 3: A — 304; B — холл; C–E — серверная; F — 303.", third)
        for level, answer in enumerate(ANSWERS, 1):
            with self.subTest(level=level):
                self.assertEqual(task.solutions(level), [answer])
                self.assertEqual(task.grade(level, json.dumps(answer))["points"], 1)

    def test_membership_geometry_and_optimization(self):
        self.assertEqual(task.grade(3, '{"cells":[" c3 ","b3"]}')["points"], 1)
        wrong = [
            (1, {"apartment": True}),
            (2, {"apartments": [101, 104]}),  # Misses the northern extension of the gym.
            (3, {"cells": ["C3"]}),  # Reading all sheets from west to east.
            (4, {"count": 0, "cells": ["C2"]}),  # Switchboard on floor 5.
            (4, {"count": 1, "cells": ["C3", "D3"]}),  # Missing an equally good shaft.
            (5, {"inlet": "F3", "outlet": "C3", "apartments": [105, 203, 304, 503], "length": 3}),
            (5, {**ANSWERS[4], "length": 3}),
        ]
        for level, answer in wrong:
            with self.subTest(level=level, answer=answer):
                self.assertEqual(task.grade(level, json.dumps(answer))["points"], 0)
        self.assertFalse(task.horizontal("A1", "B2"))
        self.assertEqual(task.horizontal("F3", "C3"), ["C3", "D3", "E3", "F3"])

    def test_cli_uses_fresh_rooms_and_preserves_submissions(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "suite"
            agent = ("import os,sys,json; from pathlib import Path; "
                     "assert not list(Path.cwd().iterdir()); assert 'answer.json' in sys.stdin.read(); "
                     f"a={ANSWERS!r}[int(os.environ['BUILDING_LEVEL'])-1]; "
                     "Path('answer.json').write_text(json.dumps(a))")
            subprocess.run([sys.executable, "-m", "spatial_harness_bench.building", "--output",
                            output, "--", sys.executable, "-c", agent], check=True, capture_output=True)
            report = json.loads((output / "result.json").read_text())
            self.assertEqual((report["score"], report["max_score"]), (5, 5))
            for level, item in enumerate(report["items"], 1):
                folder = output / item["task"]
                self.assertEqual(json.loads((folder / "answer.json").read_text()), ANSWERS[level - 1])
                self.assertEqual((folder / "prompt.txt").read_text(), task.prompt(level))

    def test_gateway_failure_and_missing_answer_differ(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, SPATIAL_GATEWAY_RETRIES="0"):
            gateway = run([sys.executable, "-c", "raise SystemExit(75)"], Path(tmp) / "fail", [1])
            self.assertIsNone(gateway["score"])
            self.assertEqual(gateway["items"][0]["reason"], "gateway")
            empty = run([sys.executable, "-c", "pass"], Path(tmp) / "empty", [1])
            self.assertEqual((empty["task"], empty["score"], empty["status"]),
                             ("06-building-above", 0, "complete"))
