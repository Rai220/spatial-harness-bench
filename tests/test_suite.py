import io
import json
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from spatial_harness_bench import suite
from spatial_harness_bench import suite_adapter as adapter
from spatial_harness_bench import suite_native as native


def completed(events, code=0):
    return subprocess.CompletedProcess([], code, "\n".join(map(json.dumps, events)).encode(), b"")


def event(kind, **kwargs):
    return dict(type=kind, **kwargs)


class SuiteTest(unittest.TestCase):
    def test_full_plan_selectors_and_modes(self):
        blocks = suite.catalog()
        self.assertEqual(len([b for b in blocks if b.family == "svg_recall"]), 4)
        self.assertEqual([b.name for b in suite.select("building,svg_scenes.generated", blocks)],
                         ["svg_scenes.generated", "building"])
        self.assertEqual(len({b.family for b in blocks}), 9 + any(b.mode == "geometry" for b in blocks))
        self.assertEqual({b.family for b in blocks if b.mode == "session"},
                         {"maze", "adnd_memory", "adnd_expedition"})
        with self.assertRaises(ValueError):
            suite.select("building,typo", blocks)

    def test_seed_only_reaches_seeded_blocks(self):
        by_name = {b.name: b for b in suite.catalog()}
        self.assertEqual(suite.options(by_name["ascii_recall.free"], False, 2), ())
        self.assertEqual(suite.options(by_name["maze"], True, 2), ("--level", "1", "--seed", "2"))
        if "engineering_geometry" in by_name:
            self.assertEqual(suite.options(by_name["engineering_geometry"], False),
                             ("--family", "all", "--count", "5"))

    def test_preflight_incompatible_request_never_launches_docker(self):
        with patch.object(suite, "preflight") as check, patch("sys.stderr", new=io.StringIO()):
            with self.assertRaises(SystemExit) as exc:
                suite.main(["--select", "building,maze", "--image", "local", "--output", "new", "--", "custom"])
            self.assertEqual(exc.exception.code, 2)
            check.assert_not_called()

    def test_report_distinguishes_missing_infrastructure_and_zero(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp) / "building"
            output.mkdir()
            self.assertEqual(suite.summary(output, 1)["status"], "technical_error")
            result = output / "result.json"
            result.write_text(json.dumps({"status": "complete", "score": 0, "max_score": 5}))
            self.assertEqual(suite.summary(output, 0)["score"], 0)
            result.write_text(json.dumps({"status": "error", "score": None, "max_score": 5, "reason": "gateway"}))
            self.assertEqual(suite.summary(output, 1)["reason"], "gateway")
            result.write_text(json.dumps({"task": "02-maze-v2-01", "status": "solved"}))
            self.assertEqual(suite.summary(output, 0)["score"], 1)

    def test_file_exchange_only_accepts_regular_artifacts(self):
        with TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "answer.json").symlink_to("/etc/passwd")
            self.assertIsNone(adapter.safe_artifact(folder, "answer.json"))
            (folder / "answer.json").unlink()
            (folder / "answer.json").write_text("{}")
            self.assertEqual(adapter.safe_artifact(folder, "answer.json"), b"{}")

    def test_source_response_accepts_only_expected_shape(self):
        with TemporaryDirectory() as tmp:
            output = Path(tmp)
            adapter.source_artifacts(output, "code", 2, '{"answers":["one","two"]}')
            self.assertEqual((output / "desc.txt").read_text(), "one\ntwo\n")
            adapter.source_artifacts(output, "code", 2, '{"answers":["one","two\n"]}')
            self.assertEqual((output / "desc.txt").read_text(), "\n")
            adapter.source_artifacts(output, "draw", 1, '{"drawings":[{"name":"one","svg":"<svg/>"}]}')
            self.assertEqual((output / "0.svg").read_text(), "<svg/>")


class NativeTest(unittest.TestCase):
    def test_codex_records_current_thread_and_resumes_it(self):
        first = completed([event("thread.started", thread_id="thread-one"),
                           event("item.completed", item={"type": "agent_message", "text": "right"}),
                           event("turn.completed")])
        second = completed([event("item.completed", item={"type": "agent_message", "text": "left"}),
                            event("turn.completed")])
        transport = native.Native("codex", "dummy", ["codex"], 10)
        with patch.object(native.subprocess, "run", side_effect=[first, second]) as run:
            self.assertEqual(transport.turn("first"), "right")
            self.assertEqual(transport.turn("second"), "left")
        one, two = [call.args[0] for call in run.call_args_list]
        self.assertNotIn("--ephemeral", one)
        self.assertEqual(two[:3], ["codex", "exec", "resume"])
        self.assertEqual(two[-2:], ["thread-one", "-"])

    def test_opencode_rejects_session_change_and_tools(self):
        transcript = [event("step_start", sessionID="one"),
                      event("text", sessionID="one", part={"messageID": "a", "id": "x", "text": "go"})]
        self.assertEqual(native.parse_turn("opencode", completed(transcript), "one"), ("one", "go"))
        with self.assertRaises(native.TransportError):
            native.parse_turn("opencode", completed(transcript), "two")
        with self.assertRaises(native.ToolViolation):
            native.parse_turn("opencode", completed([*transcript, event("tool_use", sessionID="one")]), "one")

    def test_opencode_preserves_server_credentials_and_session_between_turns(self):
        transport = native.Native("opencode", "provider/test", ["opencode"], 10)
        answer = completed([event("text", sessionID="episode", part={"text": "rest"})])
        with (
            patch.object(native.subprocess, "Popen") as server,
            patch.object(native.socket, "create_connection"),
            patch.object(native.subprocess, "run", return_value=answer) as run,
        ):
            server.return_value.poll.return_value = None
            transport.start_server()
            self.assertEqual(transport.turn("first"), "rest")
            self.assertEqual(transport.turn("next"), "rest")
        credentials = server.call_args.kwargs["env"]
        self.assertTrue(credentials["OPENCODE_PASSWORD"])
        self.assertTrue(all(call.kwargs["env"] is credentials for call in run.call_args_list))
        self.assertEqual(run.call_args.args[0][-2:], ["--session", "episode"])

    def test_provider_error_and_bad_model_action_are_distinct(self):
        with self.assertRaises(native.ProviderError):
            native.parse_turn("opencode", completed([event("error", error={"type": "provider.network"})]), None)
        self.assertEqual(native.response({"type": "observation"}, "{bad json}"), {"action": {"type": "invalid"}})
        self.assertEqual(native.response({"type": "observation"}, '{"action":{"type":"rest"}}'),
                         {"action": {"type": "rest"}})
