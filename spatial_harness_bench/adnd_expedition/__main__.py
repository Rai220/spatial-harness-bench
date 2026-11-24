"""Play a short, uncalibrated expedition with a line-oriented harness adapter."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import select
import signal
import subprocess
import sys
import tempfile
import time
from typing import Any

from ..gateway import gateway_failure, tail
from ..maze import MAX_RESPONSES_PER_MOVE
from . import DEFAULT_MAX_TURNS, VERSION
from .engine import ENGINE_VERSION, Game, SYSTEM_PROMPT

MAX_LINE = 65536
INSTRUCTIONS = SYSTEM_PROMPT + "\n" + (
    'Return one JSON object per turn, wrapped as {"action": ACTION}. '
    'The ACTION must contain exactly its required fields. Invalid actions use a turn. '
    'The adapter may provide tools for notes and maps, subject to its settings. Only '
    'observed information is available. This is a short exploratory task, not a calibrated score.'
)


def _unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise ValueError("duplicate key")
        out[key] = value
    return out


def canonical_decision(line: bytes, observation: dict[str, Any]) -> dict[str, Any]:
    """Match the original calibration action policy, including duplicate-key rejection."""
    try:
        result = json.loads(line, object_pairs_hook=_unique_pairs)
        if not isinstance(result, dict) or set(result) != {"action"}:
            return {"type": "invalid"}
        action = result["action"]
        if not isinstance(action, dict) or not isinstance(action.get("type"), str):
            return {"type": "invalid"}
        kind = action["type"]
        if kind in ("talk", "search", "attack", "defend", "potion", "rest") and set(action) == {"type"}:
            return action
        if kind == "move" and set(action) == {"type", "direction"} and action.get("direction") in (
            "north", "south", "east", "west", "up", "down"
        ):
            return action
        if kind == "ask" and set(action) == {"type", "topic"}:
            topics = {t.get("id") for t in observation.get("topics", []) if isinstance(t, dict)}
            if isinstance(action.get("topic"), str) and action["topic"] in topics:
                return action
        if kind == "ability" and set(action) == {"type", "ability"}:
            abilities = (observation.get("character") or {}).get("abilities", [])
            ids = {a.get("id") for a in abilities if isinstance(a, dict)}
            if isinstance(action.get("ability"), str) and action["ability"] in ids:
                return action
    except (TypeError, ValueError, RecursionError, UnicodeDecodeError):
        pass
    return {"type": "invalid"}


class _Lines:
    """Bound line length and waiting time without blocking on partial adapter output."""
    def __init__(self, stream):
        self.stream = stream
        self.buffer = b""

    def read(self, timeout: float) -> bytes:
        deadline = time.monotonic() + timeout
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not select.select([self.stream], [], [], remaining)[0]:
                raise TimeoutError("adapter timed out")
            part = os.read(self.stream.fileno(), 4096)
            if not part:
                raise EOFError("adapter closed output")
            self.buffer += part
            if len(self.buffer) > MAX_LINE:
                raise ValueError("adapter output exceeds limit")
        line, self.buffer = self.buffer.split(b"\n", 1)
        return line


def _send(stream, data: bytes, deadline: float) -> None:
    """Write a complete request without blocking past the turn deadline."""
    written = 0
    while written < len(data):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not select.select([], [stream], [], remaining)[1]:
            raise TimeoutError("adapter did not read input")
        try:
            written += os.write(stream.fileno(), data[written:])
        except BlockingIOError:
            continue


def _write_private(path: Path, data: dict[str, Any]) -> None:
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.write("\n")


def run(seed: str, max_turns: int, command: list[str], output: Path, *, timeout: float = 120) -> dict[str, Any]:
    """Run one episode; the seed and trace remain in owner-only files, never in model input."""
    if not isinstance(seed, str) or not seed or not command or max_turns < 1 or timeout <= 0:
        raise ValueError("invalid run arguments")
    output = Path(output)
    output.mkdir(parents=True, mode=0o700, exist_ok=False)
    if output.stat().st_mode & 0o077:
        raise ValueError("output directory is not private")
    game = Game(seed, max_turns=max_turns, enemy_count=2)
    _write_private(output / "replay.json", {"seed": seed, "max_turns": max_turns,
                    "engine_version": ENGINE_VERSION, "component_versions": game.component_versions,
                    "protocol": VERSION,
                    "config": game.config})
    error = None
    response_limited = False
    fd = os.open(output / "trace.jsonl", os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as trace, tempfile.TemporaryDirectory(prefix="adnd-harness-") as workspace, tempfile.TemporaryFile() as diagnostics:
        agent = None
        try:
            agent = subprocess.Popen(command, cwd=workspace, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=diagnostics, bufsize=0, start_new_session=True)
            assert agent.stdin is not None and agent.stdout is not None
            os.set_blocking(agent.stdin.fileno(), False)
            lines = _Lines(agent.stdout)
            while not game.done:
                obs = game.observe()
                request = {"type": "observation", "turn": game.turn + 1, "observation": obs}
                if game.turn == 0:
                    request["instructions"] = INSTRUCTIONS
                deadline = time.monotonic() + timeout
                _send(agent.stdin, (json.dumps(request, ensure_ascii=False) + "\n").encode("utf-8"), deadline)
                raw = lines.read(max(0.0, deadline - time.monotonic()))
                try:
                    special = json.loads(raw, object_pairs_hook=_unique_pairs)
                except (ValueError, TypeError, UnicodeDecodeError):
                    special = None
                if isinstance(special, dict) and special.get("type") == "gateway_error":
                    error = "gateway"
                    break
                if isinstance(special, dict) and special.get("type") == "output_limit":
                    error = "output_limit"
                    break
                if isinstance(special, dict) and special.get("type") == "response_limit":
                    if special.get("responses") == MAX_RESPONSES_PER_MOVE:
                        response_limited = True
                    else:
                        error = "adapter"
                    break
                action = canonical_decision(raw, obs)
                after = game.step(action)
                trace.write(json.dumps({"observation": obs, "action": action, "next_observation": after},
                                       ensure_ascii=False) + "\n")
                trace.flush()
            if game.done:
                try:
                    _send(agent.stdin, (json.dumps({"type": "end", "status": game.terminal_reason}) + "\n").encode(),
                          time.monotonic() + 1)
                    agent.stdin.close()
                    agent.wait(timeout=5)
                except (BrokenPipeError, TimeoutError, subprocess.TimeoutExpired):
                    # A valid terminal action remains valid even if shutdown is unclean.
                    pass
        except TimeoutError:
            error = "timeout"
        except (EOFError, OSError, ValueError):
            # An adapter may close stdout just before exiting with EX_TEMPFAIL.
            # Give it a bounded chance to publish its status before classifying.
            if agent is not None and agent.poll() is None:
                try:
                    agent.wait(timeout=min(1.0, timeout))
                except subprocess.TimeoutExpired:
                    pass
            diagnostic = tail(diagnostics)
            error = "gateway" if agent and gateway_failure(agent.poll(), diagnostic) else "adapter"
        finally:
            if agent is not None:
                # The adapter may exit while leaving an RPC child alive.
                # This session/group was created exclusively for this run;
                # clean it even when its leader has already exited.
                try:
                    os.killpg(agent.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                agent.wait()
                for stream in (agent.stdin, agent.stdout):
                    if stream is not None:
                        stream.close()
    result = game.result()
    safe = {key: result[key] for key in ("engine_version", "component_versions", "turns", "terminal_reason",
        "main_quest_completed", "quests_completed", "quests_total", "side_quests_completed",
        "side_quests_total", "rooms_visited", "rooms_total", "invalid_actions", "damage_taken", "potions_used")}
    safe.update(task=f"adnd-expedition-short-v1-t{max_turns}", protocol=VERSION, limit=max_turns,
                status="technical_error" if error else result["terminal_reason"],
                score=None if error else int(result["main_quest_completed"]), max_score=1)
    if response_limited:
        safe.update(status="limit", reason="response_limit", terminal_reason="response_limit", score=0)
    if error:
        safe["reason"] = error
    # One scored item represents one whole world; a technical failure is
    # preserved as None and excluded by spatial_harness_bench.stats.collect.
    safe["items"] = [{"task": safe["task"], "status": safe["status"], "points": safe["score"],
                      "terminal_reason": safe["terminal_reason"], "turns": safe["turns"]}]
    _write_private(output / "result.json", safe)
    return safe


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", help="reproduce a world; omit to generate a private random seed")
    parser.add_argument("--max-turns", type=int, default=DEFAULT_MAX_TURNS)
    parser.add_argument("--timeout", type=float, default=120, help="per-action timeout in seconds")
    parser.add_argument("--output", type=Path, required=True, help="new private output directory")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Provide the harness adapter command after --")
    try:
        result = run(args.seed if args.seed is not None else secrets.token_hex(16), args.max_turns,
                     command, args.output, timeout=args.timeout)
    except (OSError, ValueError):
        parser.exit(1, "Could not initialize the expedition.\n")
    print(json.dumps(result, ensure_ascii=False))
    return int(result["status"] == "technical_error")


if __name__ == "__main__":
    sys.exit(main())
