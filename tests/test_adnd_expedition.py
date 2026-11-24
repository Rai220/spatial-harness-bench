import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from spatial_harness_bench.adnd_expedition.__main__ import canonical_decision, run
from spatial_harness_bench.adnd_expedition.engine import Game
from spatial_harness_bench.adnd_expedition.pi import _unique_pairs as pi_unique_pairs
from spatial_harness_bench.stats import collect, rows as stats_rows


class ShortExpeditionTests(unittest.TestCase):
    def test_action_schema_and_duplicate_keys(self):
        obs = Game("public-test", max_turns=2, enemy_count=2).observe()
        self.assertEqual(canonical_decision(b'{"action":{"type":"move","direction":"north"}}', obs),
                         {"type": "move", "direction": "north"})
        for bad in (b'{"action":{"type":"talk","notes":"extra"}}',
                    b'{"action":{"type":"talk","type":"attack"}}',
                    b'{"action":{"type":"ask","topic":"unknown"}}', b'not json'):
            self.assertEqual(canonical_decision(bad, obs), {"type": "invalid"})
        with self.assertRaises(ValueError):
            json.loads('{"action":{"type":"talk","type":"attack"}}', object_pairs_hook=pi_unique_pairs)

    def test_two_turn_run_private_replay_and_model_visibility(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "mock.py"
            # Each action is based only on a visible observation. The adapter
            # asserts the seed is not part of any message it receives.
            script.write_text('''import json, sys\nfor line in sys.stdin:\n r=json.loads(line)\n assert "private-example-seed" not in line\n if r["type"]=="end": break\n print(json.dumps({"action":{"type":"talk"}}),flush=True)\n''')
            folder = root / "result"
            result = run("private-example-seed", 2, [sys.executable, "-u", str(script)], folder, timeout=5)
            self.assertEqual(result["status"], "turn_limit")
            self.assertEqual(result["score"], 0)
            self.assertNotIn("seed", result)
            self.assertEqual(list(stats_rows(result))[0][:6], ("adnd-expedition-short-v1-t2", "-", 0, 1, 0, 1))
            group = collect([f"model={folder}"])[("model", "adnd-expedition-short-v1-t2", "-")]
            self.assertEqual((group.scores, group.items, group.errors), ([0], 1, 0))
            self.assertEqual(folder.stat().st_mode & 0o777, 0o700)
            for name in ("result.json", "replay.json", "trace.jsonl"):
                self.assertEqual((folder / name).stat().st_mode & 0o777, 0o600)
            rows = [json.loads(line) for line in (folder / "trace.jsonl").read_text().splitlines()]
            game = Game("private-example-seed", max_turns=2, enemy_count=2)
            for row in rows:
                self.assertEqual(game.observe(), row["observation"])
                self.assertEqual(game.step(row["action"]), row["next_observation"])
            self.assertTrue(game.done)
            with self.assertRaises(FileExistsError):
                run("another", 1, [sys.executable, str(script)], folder, timeout=5)

    def test_gateway_error_is_unscored_and_timeout_is_bounded(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "mock.py"
            script.write_text('''import sys\nsys.stdin.readline()\nprint('{"type":"gateway_error"}',flush=True)\n''')
            result = run("dev", 1, [sys.executable, "-u", str(script)], root / "error", timeout=2)
            self.assertEqual(result["status"], "technical_error")
            self.assertEqual(result["reason"], "gateway")
            self.assertIsNone(result["score"])
            group = collect([f"model={root / 'error'}"])[("model", "adnd-expedition-short-v1-t1", "-")]
            self.assertEqual((group.scores, group.items, group.errors), ([], 0, 1))
            # Regression: EOF may precede the adapter's delayed EX_TEMPFAIL.
            script.write_text('''import os,sys,time\nsys.stdin.readline()\nos.close(1)\ntime.sleep(.3)\nsys.exit(75)\n''')
            result = run("dev", 1, [sys.executable, "-u", str(script)], root / "delayed", timeout=2)
            self.assertEqual(result["reason"], "gateway")
            self.assertIsNone(result["score"])
            script.write_text('''import sys, time\nsys.stdin.readline()\ntime.sleep(10)\n''')
            result = run("dev", 1, [sys.executable, "-u", str(script)], root / "timeout", timeout=0.2)
            self.assertEqual(result["reason"], "timeout")
            self.assertIsNone(result["score"])

    def test_adapter_that_never_reads_input_cannot_block_writer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "stalled.py"
            script.write_text('''import json,sys,time\nfor i in range(2000):\n print(json.dumps({"action":{"type":"talk"}}),flush=True)\ntime.sleep(10)\n''')
            result = run("dev", 2000, [sys.executable, "-u", str(script)], root / "result", timeout=0.25)
            self.assertEqual(result["status"], "technical_error")
            self.assertEqual(result["reason"], "timeout")
            self.assertIsNone(result["score"])

    def test_harness_response_limit_is_scored_and_distinct_from_gateway_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "adapter.py"
            script.write_text('''import sys\nsys.stdin.readline()\nprint('{"type":"response_limit","responses":32}',flush=True)\n''')
            result = run("dev", 5, [sys.executable, "-u", str(script)], root / "result", timeout=2)
            self.assertEqual((result["status"], result["reason"], result["terminal_reason"]),
                             ("limit", "response_limit", "response_limit"))
            self.assertEqual(result["score"], 0)
            group = collect([f"model={root / 'result'}"])[("model", "adnd-expedition-short-v1-t5", "-")]
            self.assertEqual((group.scores, group.items, group.errors), ([0], 1, 0))
            script.write_text('''import sys\nsys.stdin.readline()\nprint('{"type":"response_limit","responses":31}',flush=True)\n''')
            bad = run("dev", 5, [sys.executable, "-u", str(script)], root / "bad", timeout=2)
            self.assertEqual(bad["status"], "technical_error")
            self.assertIsNone(bad["score"])

    def test_cleanup_kills_only_this_adapters_lingering_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); marker = root / "child-survived"
            script = root / "adapter.py"
            child = f'import time; from pathlib import Path; time.sleep(.7); Path({str(marker)!r}).write_text("survived")'
            script.write_text(f'''import json,subprocess,sys\nsubprocess.Popen([sys.executable,"-c",{child!r}])\nfor line in sys.stdin:\n r=json.loads(line)\n if r.get("type")=="end": break\n print(json.dumps({{"action":{{"type":"talk"}}}}),flush=True)\n''')
            result = run("dev", 1, [sys.executable, "-u", str(script)], root / "result", timeout=3)
            self.assertEqual(result["status"], "turn_limit")
            time.sleep(1)
            self.assertFalse(marker.exists(), "descendant survived process-group cleanup")

    def test_stats_do_not_pool_different_turn_budgets(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); script = root / "adapter.py"
            script.write_text('''import json,sys\nfor line in sys.stdin:\n r=json.loads(line)\n if r.get("type")=="end": break\n print(json.dumps({"action":{"type":"talk"}}),flush=True)\n''')
            for limit in (1, 2):
                run("dev", limit, [sys.executable, "-u", str(script)], root / f"run-{limit}", timeout=3)
            groups = collect([f"same-label={root}/run-*"])
            self.assertIn(("same-label", "adnd-expedition-short-v1-t1", "-"), groups)
            self.assertIn(("same-label", "adnd-expedition-short-v1-t2", "-"), groups)

    def test_pi_rpc_bridge_drives_the_external_harness(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mock_pi = root / "pi.py"
            mock_pi.write_text('''import json,sys\nfor line in sys.stdin:\n event={"type":"message_end","message":{"role":"assistant","content":[{"type":"text","text":json.dumps({"action":{"type":"talk"}})}]}}\n print(json.dumps(event),flush=True)\n print(json.dumps({"type":"agent_end"}),flush=True)\n''')
            # The installed package is on sys.path in normal use. Set it
            # explicitly here because the harness runs in a temporary cwd.
            repo = str(Path(__file__).resolve().parents[1])
            env_path = repo + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
            with patch.dict(os.environ, {"PYTHONPATH": env_path}):
                result = run("public-test", 2, [sys.executable, "-u", "-m",
                             "spatial_harness_bench.adnd_expedition.pi", "--",
                             sys.executable, "-u", str(mock_pi)], root / "result", timeout=5)
            self.assertEqual(result["status"], "turn_limit")
            self.assertEqual(result["turns"], 2)

    def test_module_cli_with_external_jsonlines_adapter(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            script = root / "adapter.py"
            script.write_text('''import json,sys\nfor line in sys.stdin:\n request=json.loads(line)\n if request.get("type")=="end": break\n print(json.dumps({"action":{"type":"talk"}}),flush=True)\n''')
            result = subprocess.run([sys.executable, "-m", "spatial_harness_bench.adnd_expedition",
                                     "--seed", "public-dev-only", "--max-turns", "2", "--output",
                                     str(root / "run"), "--", sys.executable, "-u", str(script)],
                                    capture_output=True, text=True, timeout=10, check=True)
            report = json.loads(result.stdout)
            self.assertEqual(report["status"], "turn_limit")
            self.assertNotIn("public-dev-only", result.stdout)

    def test_chat_adapter_with_mock_gateway_and_file_tools(self):
        received = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                data = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                received.append(data)
                if len(received) == 1:
                    message = {"content": None, "tool_calls": [{"id": "call-1", "type": "function",
                               "function": {"name": "write_file", "arguments": json.dumps({"name": "notes.txt", "text": "remember"})}}]}
                else:
                    message = {"content": json.dumps({"action": {"type": "talk"}})}
                body = json.dumps({"choices": [{"message": message}]}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(thread.join)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as tmp:
            repo = str(Path(__file__).resolve().parents[1])
            env_path = repo + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
            env = {"PYTHONPATH": env_path, "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "OPENAI_API_KEY": "dummy-private"}
            with patch.dict(os.environ, env):
                result = run("private-seed-example", 2, [sys.executable, "-u", "-m",
                             "spatial_harness_bench.adnd_expedition.chat", "--model", "test-model", "--notes"],
                             Path(tmp) / "result", timeout=5)
            self.assertEqual(result["status"], "turn_limit")
            self.assertEqual(len(received), 3)
            self.assertTrue(all(r["model"] == "test-model" and r["temperature"] == 0 for r in received))
            self.assertIn("tools", received[0])
            self.assertTrue(any(m["role"] == "tool" and m["content"] == "Сохранено." for m in received[1]["messages"]))
            self.assertNotIn("private-seed-example", json.dumps(received))
            self.assertNotIn("dummy-private", json.dumps(received))

    def test_chat_length_is_unscored_even_when_content_is_nonempty_or_valid_json(self):
        seen = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                self.rfile.read(int(self.headers["Content-Length"]))
                # Both an incomplete JSON prefix and a valid-looking prefix
                # are unscored when the provider says it truncated output.
                content = '{"action":{"type":"talk"' if not seen else '{"action":{"type":"talk"}}'
                seen.append(content)
                body = json.dumps({"choices": [{"message": {"content": content}, "finish_reason": "length"}]}).encode()
                self.send_response(200); self.send_header("Content-Length", str(len(body))); self.end_headers(); self.wfile.write(body)
            def log_message(self, *_):
                pass
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True); thread.start()
        self.addCleanup(server.server_close); self.addCleanup(thread.join); self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory() as tmp:
            repo = str(Path(__file__).resolve().parents[1])
            env_path = repo + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")
            with patch.dict(os.environ, {"PYTHONPATH": env_path,
                    "OPENAI_BASE_URL": f"http://127.0.0.1:{server.server_port}/v1", "OPENAI_API_KEY": "dummy-private"}):
                for i in range(2):
                    result = run("dev", 5, [sys.executable, "-u", "-m",
                                 "spatial_harness_bench.adnd_expedition.chat", "--model", "test-model"],
                                 Path(tmp) / f"run-{i}", timeout=5)
                    self.assertEqual(result["status"], "technical_error")
                    self.assertEqual(result["reason"], "output_limit")
                    self.assertIsNone(result["score"])
                    self.assertEqual(result["turns"], 0)
        self.assertEqual(len(seen), 2)


if __name__ == "__main__":
    unittest.main()
