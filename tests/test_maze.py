import contextlib
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest

from scripts.maze_chat import use_file
from spatial_harness_bench.maze.__main__ import run
from spatial_harness_bench.maze.game import CLOSED, DIRECTIONS, Maze, OPEN, WALL, Walk
from spatial_harness_bench.maze.levels import (LEVELS, LOOPS, dfs_steps, explore_cost, generate_level,
                                               wall_followers_fail)


def open_edges(maze):
    return sum(state == OPEN for doors in maze.rooms for state in doors) // 2


class TestMaze(unittest.TestCase):
    def test_connected_symmetric_mazes_at_every_supported_size(self):
        for size in range(5, 16):
            seed = 7
            with self.subTest(size=size, seed=seed):
                maze = Maze.generate(size, seed)
                distances = {maze.start: 0}
                queue = [maze.start]
                for cell in queue:
                    for d, state in enumerate(maze.rooms[cell]):
                        neighbor = maze.neighbor(cell, d)
                        if neighbor is None:
                            self.assertEqual(state, WALL)
                            continue
                        self.assertEqual(maze.rooms[neighbor][(d + 2) % 4], state)
                        if state == OPEN and neighbor not in distances:
                            distances[neighbor] = distances[cell] + 1
                            queue.append(neighbor)
                self.assertEqual(len(distances), size * size)
                self.assertNotEqual(maze.start, maze.exit)
                self.assertEqual(maze.shortest_distance(), distances[maze.exit])

    def test_loops_keep_the_tree_and_its_closed_doors(self):
        for size in (5, 10, 15):
            tree, braided = Maze.generate(size, 3), Maze.generate(size, 3, LOOPS)
            self.assertEqual(open_edges(tree), size * size - 1)
            self.assertGreater(open_edges(braided), open_edges(tree))
            for a, b in zip(tree.rooms, braided.rooms):
                for x, y in zip(a, b):
                    self.assertTrue(x == y or (y == OPEN and x != OPEN))
            self.assertEqual(braided, Maze.generate(size, 3, LOOPS))
        with self.assertRaises(ValueError):
            Maze.generate(5, 0, 1.5)

    def test_seed_and_positions(self):
        self.assertEqual(Maze.generate(10, 31), Maze.generate(10, 31))
        self.assertNotEqual(Maze.generate(10, 31).rooms, Maze.generate(10, 32).rooms)
        maze = Maze.generate(10, 1)
        for cell in (maze.start, maze.exit):
            self.assertTrue(all(0 < n < 9 for n in divmod(cell, 10)))
        for size in (4, 16):
            with self.assertRaises(ValueError):
                Maze.generate(size, 0)

    def test_blocked_and_terminal_moves(self):
        maze = Maze.generate(10, 7)
        for state in (CLOSED, WALL):
            cell = next(i for i, doors in enumerate(maze.rooms) if state in doors and i != maze.exit)
            maze.start = cell
            walk = Walk(maze, 2)
            walk.step(DIRECTIONS[maze.rooms[cell].index(state)])
            walk.step("направо и ещё раз направо")
            self.assertEqual((walk.status, walk.cell, walk.attempts), ("limit", cell, 2))
            with self.assertRaises(ValueError):
                walk.step("влево")
        d = maze.rooms[maze.start].index(OPEN)
        maze.exit = maze.neighbor(maze.start, d)
        walk = Walk(maze, 1)
        walk.step(DIRECTIONS[d].upper().replace("Ё", "Е"))
        self.assertEqual(walk.status, "solved")

    def test_adapter_can_solve_from_local_observations_and_keep_notes(self):
        # A right-hand traversal visits every room of a tree (loops=0) in at most 2E steps.
        adapter = '''import json, sys
from pathlib import Path
directions = ("вперёд", "вправо", "назад", "влево")
heading = 0
for line in sys.stdin:
    room = json.loads(line)
    if room["type"] == "end":
        assert room["status"] in ("solved", "limit")
        break
    if room["turn"] == 1:
        assert set(room) == {"instructions","type","turn","text"}
        assert not list(Path.cwd().iterdir())
        Path("journal.txt").write_text("start")
    else:
        assert set(room) == {"type","turn","text"}
        assert Path("journal.txt").read_text() == "start"
    for offset in (1, 0, -1, 2):
        facing = (heading + offset) % 4
        if directions[facing] + " — открыта" in room["text"]:
            heading = facing
            print(json.dumps({"move": directions[facing]}), flush=True)
            break
'''
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "walker.py"
            script.write_text(adapter)
            result = run(Maze.generate(10, 8), 8, 200, [sys.executable, str(script)], root / "run")
            self.assertEqual(result["status"], "solved")
            self.assertLessEqual(result["shortest"], result["moves"])
            self.assertEqual(result["moves"], result["attempts"])
            self.assertEqual((root / "run/notes/journal.txt").read_text(), "start")
            trace = (root / "run/steps.jsonl").read_text().splitlines()
            self.assertEqual(json.loads(trace[-1])["to"], result["exit"])
            self.assertEqual(len(trace), result["attempts"])

            cli = [sys.executable, "-m", "spatial_harness_bench.maze"]
            seed = "8278864541757570430"
            # Version 2: a wall follower loops on every level; a remembering explorer solves all.
            suites = {}
            for name, adapter_cmd in [
                ("wall", ["--", sys.executable, str(script)]),
                ("dfs", ["--", sys.executable, "-m", "spatial_harness_bench.maze.baselines", "dfs"]),
            ]:
                suites[name] = root / name
                subprocess.run([*cli, "--suite", "--seed", seed, "--output", str(suites[name]), *adapter_cmd],
                               check=True, capture_output=True)
            wall = json.loads((suites["wall"] / "suite.json").read_text())
            self.assertEqual((wall["status"], wall["score"], wall["efficiency"]), ("complete", 0, 0.0))
            summary = json.loads((suites["dfs"] / "suite.json").read_text())
            self.assertEqual((summary["status"], summary["score"], summary["max_score"]),
                             ("complete", 10, 10))
            self.assertEqual(len(summary["levels"]), 10)
            self.assertTrue(0 < summary["efficiency"] < 1)
            for number, result in enumerate(summary["levels"], 1):
                folder = suites["dfs"] / f"level-{number:02d}"
                self.assertEqual(result["task"], f"02-maze-v2-{number:02d}")
                self.assertEqual(result, json.loads((folder / "result.json").read_text()))
                maze = Maze.generate(result["size"], result["seed"], LOOPS)
                self.assertEqual(result["attempts"], dfs_steps(maze, (0, 1, 2, 3)))
                self.assertEqual(result["efficiency"], round(result["shortest"] / result["attempts"], 4))
                self.assertEqual(result["limit"], 2 * result["size"]**2)
            single = root / "level-05"
            subprocess.run([*cli, "--level", "5", "--seed", seed, "--output", str(single),
                            "--", sys.executable, "-m", "spatial_harness_bench.maze.baselines", "dfs"],
                           check=True, capture_output=True)
            replay = json.loads((single / "result.json").read_text())
            self.assertEqual(replay, summary["levels"][4])
            self.assertEqual((single / "map.txt").read_text(),
                             Maze.generate(10, replay["seed"], LOOPS).render())

    def test_level_difficulty_and_replay(self):
        self.assertEqual(len(LEVELS), 10)
        self.assertEqual((LEVELS[0].size, LEVELS[-1].size), (5, 15))
        for lower, higher in zip(LEVELS, LEVELS[1:]):
            self.assertLess(lower.size, higher.size)
            self.assertLess(lower.shortest[1], higher.shortest[0])
            self.assertLess(lower.explore[1], higher.explore[0])
        for number, level in enumerate(LEVELS, 1):
            maze, map_seed, cost = generate_level(number, 123)
            self.assertEqual(maze, Maze.generate(level.size, map_seed, LOOPS))
            self.assertEqual((maze, map_seed, cost), generate_level(number, 123))
            self.assertLessEqual(level.shortest[0], maze.shortest_distance())
            self.assertLessEqual(maze.shortest_distance(), level.shortest[1])
            self.assertEqual(cost, explore_cost(maze))
            self.assertLessEqual(level.explore[0], cost)
            self.assertLessEqual(cost, level.explore[1])
            self.assertTrue(wall_followers_fail(maze, 2 * level.size**2))
            # Exploration with memory always fits the decision limit.
            self.assertLessEqual(max(dfs_steps(maze, o) for o in ((0, 1, 2, 3), (3, 2, 1, 0))),
                                 2 * level.size**2)

    def test_suite_distinguishes_limits_from_adapter_errors(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            cli = [sys.executable, "-m", "spatial_harness_bench.maze", "--suite", "--seed", "42",
                   "--limit", "1", "--output"]
            cmd = 'import sys; input(); print(\'{"move":"назад"}\', flush=True); input()'
            subprocess.run([*cli, str(root / "limits"), "--", sys.executable, "-c", cmd],
                           check=True, capture_output=True)
            report = json.loads((root / "limits/suite.json").read_text())
            self.assertEqual([r["status"] for r in report["levels"]], ["limit"] * 10)
            self.assertEqual((report["status"], report["score"]), ("complete", 0))
            run_error = subprocess.run([*cli, str(root / "error"), "--", sys.executable, "-c", "pass"],
                                       capture_output=True)
            self.assertNotEqual(run_error.returncode, 0)
            report = json.loads((root / "error/suite.json").read_text())
            self.assertEqual((report["status"], report["score"]), ("error", None))

    def test_note_files_stay_inside_workspace(self):
        def call(tool, **args):
            return use_file({"name": tool, "arguments": json.dumps(args)})

        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "outside").write_text("hidden")
            (root / "work").mkdir()
            with contextlib.chdir(root / "work"):
                Path("link").symlink_to(root / "outside")
                for name in ("../outside", str(root / "outside"), "link"):
                    self.assertNotEqual(call("read_file", name=name), "hidden")
                call("write_file", name="map", text="S")
                call("append_file", name="map", text="---")
                self.assertEqual(call("read_file", name="map"), "S---")
