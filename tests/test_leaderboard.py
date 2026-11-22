import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from scripts.render_leaderboard import ranking, table
from scripts.score_leaderboard import SCORING_VERSION, overall_score


def scores():
    def subtask(maximum):
        return {"score": 0, "max_score": maximum}

    def svg(maximum):
        return {"code": subtask(maximum)}

    return {"01-ascii": subtask(10), "01-ascii-free": subtask(10), "02-maze": subtask(10),
            **{f"03-{level}": svg(10) for level in ("basic", "hard", "expert", "master")},
            "04-fixed": svg(10), "04-generated": svg(24), "05-arcade": subtask(1)}


class OverallScoreTests(unittest.TestCase):
    def test_harness_is_visible_in_both_tables_even_for_a_single_run(self):
        run = dict(display_model="Example", harness="Codex", harness_version="1.2.3",
                   provider_label="OpenAI", result_id="example", overall_score=72.5)
        html, markdown = table([(1, run)])
        self.assertIn('<th scope="col">Харнес</th>', html)
        self.assertIn("<td>Codex 1.2.3 · OpenAI</td>", html)
        self.assertIn("| 1 | Example | Codex 1.2.3 · OpenAI | 72,50 |", markdown)

    def test_arcade_is_one_point_and_scene_sets_have_equal_weight(self):
        data = scores()
        data["05-arcade"]["score"] = 1
        self.assertEqual(overall_score(data), 1)
        data["04-fixed"]["code"]["score"] = 10
        data["04-generated"]["vision"] = {"score": 24, "max_score": 24}
        self.assertEqual(overall_score(data), 13.38)

    def test_perfect_spatial_score_leaves_one_point_for_arcade(self):
        data = scores()
        for task, result in data.items():
            if task != "05-arcade":
                for mode in ([result] if "score" in result else result.values()):
                    mode["score"] = mode["max_score"]
        self.assertEqual(overall_score(data), 99)
        data["05-arcade"]["score"] = 1
        self.assertEqual(overall_score(data), 100)

    def test_incomplete_result_cannot_enter_ranking(self):
        data = scores()
        data["03-expert"]["code"].update(status="error", score=None)
        with self.assertRaises(ValueError):
            overall_score(data)
        del data["03-expert"]["code"]
        with self.assertRaises(KeyError):
            overall_score(data)

    def test_profiles_share_a_ranking_and_different_map_seeds_are_rejected(self):
        common = dict(scoring_version=SCORING_VERSION, benchmark_package_sha256="test",
                      datasets={}, drawing_seed="1", maze_suite_seed="2", max_score=100, date="2026-11-21")
        with TemporaryDirectory() as tmp:
            paths = []
            for model, value in [("C", 53.2), ("B", 65), ("A", 65)]:
                path = Path(tmp) / f"{model}.json"
                row = dict(common, overall_score=value, display_model=model)
                if model == "B":
                    row.update(run_profile="medium-v1", benchmark_package_sha256="updated")
                path.write_text(json.dumps(row))
                paths.append(path)
            self.assertEqual([(place, run["display_model"]) for place, run in ranking(paths)],
                             [(1, "A"), (1, "B"), (3, "C")])
            bad = json.loads(paths[0].read_text())
            bad["maze_suite_seed"] = "3"
            paths[0].write_text(json.dumps(bad))
            with self.assertRaises(ValueError):
                ranking(paths)
