import io
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch
from urllib.error import HTTPError

from scripts import maze_chat
from spatial_harness_bench import gateway
from spatial_harness_bench.ascii_recall.__main__ import run as ascii_run
from spatial_harness_bench.arcade_clips.__main__ import run as arcade_run
from spatial_harness_bench.maze.__main__ import run as maze_run
from spatial_harness_bench.maze.game import Maze
from spatial_harness_bench.stats import rows
from spatial_harness_bench.svg_recall.__main__ import run as svg_run
from spatial_harness_bench.svg_scenes.__main__ import run as scenes_run


class GatewayRetryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.enterContext(patch.dict(os.environ, SPATIAL_GATEWAY_RETRIES="2"))
        self.sleep = self.enterContext(patch.object(gateway, "sleep"))

    def command(self, code):
        return [sys.executable, "-c", code, str(self.root / "calls")]

    def test_retry_only_rejected_mode_from_pristine_inputs(self):
        cmd = self.command("""
import os, sys
from pathlib import Path
stage = os.environ['SVG_RECALL_STAGE']
log = Path(sys.argv[1])
previous = log.read_text().splitlines() if log.exists() else []
with log.open('a') as f: print(stage, file=f)
if stage == 'vision' and 'vision' not in previous:
    Path('0.png').unlink()
    Path('leftover').write_text('failed output')
    Path('desc.txt').write_text('discard me')
    print('API Error: 400 Invalid request parameter', file=sys.stderr)
    sys.exit(1)
assert not Path('leftover').exists() and not Path('desc.txt').exists()
if stage == 'vision': assert Path('0.png').read_bytes().startswith(b'\\x89PNG')
Path('desc.txt').write_text('valid but wrong')
""")
        result = scenes_run(cmd, self.root / "out", "both")
        self.assertEqual((self.root / "calls").read_text().splitlines(), ["code", "vision", "vision"])
        self.assertEqual([row["score"] for row in result["modes"].values()], [0, 0])
        self.sleep.assert_called_once_with(1)

    def test_exhausted_file_stages_never_grade_a_partial_submission(self):
        cmd = self.command("from pathlib import Path; import sys; "
                           "Path('desc.txt').write_text('partial'); sys.exit(75)")
        for name, run in [("ascii", ascii_run), ("arcade", arcade_run)]:
            with self.subTest(name=name):
                output = self.root / name
                result = run(cmd, output)
                self.assertEqual(result, json.loads((output / "result.json").read_text()))
                self.assertEqual((result["status"], result["reason"], result["score"], result["attempts"]),
                                 ("error", "gateway", None, 3))
                self.assertIsNone(list(rows(result))[0][2])
                self.assertFalse((output / "desc.txt").exists())

    def test_unavailable_svg_artist_blocks_both_modes(self):
        cmd = self.command("import sys; sys.exit(75)")
        output = self.root / "svg"
        report = svg_run(cmd, output, "both", objects="assigned", seed=2)
        self.assertEqual(report["status"], "error")
        for mode in ("code", "vision"):
            self.assertEqual(report["modes"][mode], {"status": "error", "reason": "gateway",
                             "attempts": 3, "stage": "draw", "score": None})
            self.assertFalse((output / mode).exists())

    def test_failed_cli_event_even_with_exit_zero_but_no_retry_for_model_errors(self):
        folder = self.root / "work"
        folder.mkdir()
        event = {"type": "result", "is_error": True, "result": "API Error: 422 Request failed"}
        with self.assertRaises(gateway.GatewayError) as cm:
            gateway.run_stage(self.command(f"print({json.dumps(event)!r})"), folder, "prompt")
        self.assertEqual(cm.exception.attempts, 3)
        self.sleep.reset_mock()
        with self.assertRaises(subprocess.CalledProcessError):
            gateway.run_stage(self.command("import sys; sys.exit(2)"), folder, "prompt")
        self.sleep.assert_not_called()
        self.assertFalse(gateway.gateway_failure(0, "HTTP 503", '{"type":"result","is_error":false}'))
        self.assertFalse(gateway.gateway_failure(0, "", "API Error: response exceeded the output token maximum"))
        self.assertTrue(gateway.gateway_failure(0, "", json.dumps(
            {"type": "turn.failed", "error": {"message": "unexpected status 503 Service Unavailable"}})))

    def test_chat_retries_the_same_request_after_gateway_rejection(self):
        reject = HTTPError("https://localhost", 400, "Rejected", {}, io.BytesIO(b""))
        with patch.object(maze_chat.urllib.request, "urlopen",
                          side_effect=[reject, io.BytesIO(b'{"choices": []}')]) as send:
            self.assertEqual(maze_chat.complete("https://localhost", "test", {"messages": []}), {"choices": []})
        self.assertIs(send.call_args_list[0].args[0], send.call_args_list[1].args[0])

    def test_jsonlines_adapter_restores_the_same_map_with_an_empty_notebook(self):
        cmd = self.command("""
from pathlib import Path
import sys, json
log = Path(sys.argv[1]); room = json.loads(sys.stdin.readline())
if not log.exists():
    log.write_text(json.dumps(room))
    Path('notes').write_text('abandoned')
    sys.exit(75)
assert room == json.loads(log.read_text()) and not Path('notes').exists()
print(json.dumps({'move': 'wrong'}), flush=True)
assert json.loads(sys.stdin.readline())['type'] == 'end'
""")
        dest = self.root / "maze"
        result = maze_run(Maze.generate(5, 17), 17, 1, cmd, dest)
        self.assertEqual((result["status"], result["attempts"]), ("limit", 1))
        self.assertEqual(len((dest / "steps.jsonl").read_text().splitlines()), 1)
        self.sleep.assert_called_once_with(1)

    def test_exhausted_maze_request_does_not_restart_the_map_or_consume_a_move(self):
        cmd = self.command("""import json,sys
sys.stdin.readline()
print(json.dumps({'type':'gateway_error','exhausted':True,'attempts':3}), flush=True)
sys.exit(75)
""")
        result = maze_run(Maze.generate(5, 17), 17, 1, cmd, self.root / "maze")
        self.assertEqual((result["status"], result["score"], result["reason"], result["attempts"],
                          result["gateway_attempts"]), ("error", None, "gateway", 0, 3))
        self.sleep.assert_not_called()
