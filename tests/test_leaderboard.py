import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.render_leaderboard import ranking
from scripts.score_leaderboard import overall_score


def scores():
    def subtask(maximum):
        return {"score": 0, "max_score": maximum}

    def svg(maximum):
        return {mode: subtask(maximum) for mode in ("code", "vision")}

    return {"01-ascii": subtask(10), "01-ascii-free": subtask(10), "02-maze": subtask(10),
            **{f"03-{level}": svg(10) for level in ("basic", "hard", "expert", "master")},
            "04-fixed": svg(10), "04-generated": svg(24), "05-arcade": subtask(1)}


class OverallScoreTests(unittest.TestCase):
    def test_equal_task_and_mode_weights(self):
        data = scores()
        data["05-arcade"]["score"] = 1
        data["04-fixed"]["code"]["score"] = 10
        data["04-generated"]["vision"]["score"] = 24
        self.assertEqual(overall_score(data), 30)

    def test_incomplete_result_cannot_enter_ranking(self):
        data = scores()
        data["03-expert"]["vision"].update(status="error", score=None)
        with self.assertRaises(ValueError):
            overall_score(data)
        del data["03-expert"]["vision"]
        with self.assertRaises(KeyError):
            overall_score(data)

    def test_ties_share_rank_and_incompatible_cohorts_are_rejected(self):
        common = dict(scoring_version="spatial-macro-v1", benchmark_package_sha256="test",
                      datasets={}, drawing_seed="1", maze_suite_seed="2", max_score=100, date="2026-11-21")
        with TemporaryDirectory() as tmp:
            paths = []
            for model, value in [("C", 53.2), ("B", 65), ("A", 65)]:
                path = Path(tmp) / f"{model}.json"
                path.write_text(json.dumps(dict(common, overall_score=value, display_model=model)))
                paths.append(path)
            self.assertEqual([(place, run["display_model"]) for place, run in ranking(paths)],
                             [(1, "A"), (1, "B"), (3, "C")])
            bad = json.loads(paths[0].read_text())
            bad["maze_suite_seed"] = "3"
            paths[0].write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                ranking(paths)
