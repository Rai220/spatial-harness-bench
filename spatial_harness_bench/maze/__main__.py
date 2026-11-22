"""Run a JSON-lines maze adapter. Only room observations cross into the adapter."""

import argparse
import json
from pathlib import Path
import secrets
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile

from ..gateway import GatewayError, gateway_failure, relay, retry, tail

from . import MAX_RESPONSES_PER_MOVE
from .game import MAX_SIZE, MIN_SIZE, Maze, Walk, instructions
from .levels import LEVELS, LOOPS, VERSION, explore_cost, generate_level


def run(maze: Maze, seed: int, limit: int | None, command: list[str], output: Path, *,
        task: str = "02-maze", explore: int | None = None, loops: float = 0.0) -> dict:
    output.mkdir(parents=True, exist_ok=False)

    def clear_notes():
        notes = output / "notes"
        if notes.exists():
            shutil.rmtree(notes)

    try:
        return retry(lambda: _walk_once(maze, seed, limit, command, output, task, explore, loops),
                     reset=clear_notes)
    except GatewayError as exc:
        path = output / "result.json"
        result = json.loads(path.read_text())
        result["gateway_attempts"] = exc.attempts
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
        return result


def _walk_once(maze, seed, limit, command, output, task, explore, loops):
    if limit is None:
        limit = 2 * maze.size**2
    walk = Walk(maze, limit)
    shortest = maze.shortest_distance()
    visited = {walk.cell}
    moves = 0
    feedback = ""
    error = None
    response_limit = False
    gateway_error = None
    agent = None
    with TemporaryDirectory(prefix="maze-notes-") as work, TemporaryFile() as diagnostics:
        try:
            with (output / "steps.jsonl").open("w", encoding="utf-8") as trace, subprocess.Popen(
                command, cwd=work, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1, stderr=diagnostics,
            ) as agent:
                try:
                    while walk.status == "running":
                        room = {"type": "room", "turn": walk.attempts + 1,
                                "text": (feedback + " " + walk.observation()).strip()}
                        if not walk.attempts:
                            room["instructions"] = instructions(maze.size, limit, loops=loops > 0)
                        print(json.dumps(room, ensure_ascii=False), file=agent.stdin, flush=True)
                        line = agent.stdout.readline()
                        if not line:
                            agent.wait(timeout=2)
                            raise RuntimeError("Адаптер завершился до следующего решения.")
                        decision = json.loads(line)
                        if isinstance(decision, dict) and decision.get("type") == "response_limit":
                            if decision.get("responses") != MAX_RESPONSES_PER_MOVE:
                                raise ValueError("Неверное число ответов при исчерпании лимита.")
                            response_limit = True
                            break
                        if isinstance(decision, dict) and decision.get("type") == "gateway_error":
                            failure = GatewayError("Inference request rejected or interrupted")
                            if decision.get("exhausted") is True:
                                failure.exhausted = True
                                failure.attempts = int(decision["attempts"])
                            raise failure
                        action = decision.get("move") if isinstance(decision, dict) else None
                        if not isinstance(action, str):
                            raise ValueError("Поле move должно быть строкой.")
                        before = walk.cell
                        feedback = walk.step(action)
                        visited.add(walk.cell)
                        moves += walk.cell != before
                        print(json.dumps({"turn": walk.attempts, "from": before, "to": walk.cell,
                                          "move": action, "observation": room["text"], "result": feedback},
                                         ensure_ascii=False), file=trace, flush=True)
                    if not response_limit:
                        print(json.dumps({"type": "end", "status": walk.status}),
                              file=agent.stdin, flush=True)
                    agent.stdin.close()
                    if agent.wait(timeout=15):
                        raise RuntimeError(f"Адаптер завершился с кодом {agent.returncode}.")
                finally:
                    if agent.poll() is None:
                        agent.kill()
        except (OSError, ValueError, KeyError, RuntimeError, subprocess.TimeoutExpired) as exc:
            error = str(exc)
            diagnostic = tail(diagnostics)
            if isinstance(exc, GatewayError):
                gateway_error = exc
            elif agent is not None and gateway_failure(agent.returncode, diagnostic):
                gateway_error = GatewayError("Inference request rejected or interrupted")
        if gateway_error is None:
            relay(diagnostics, sys.stderr)
        # The adapter can use its private workspace for persistent notes.
        if any(Path(work).iterdir()):
            shutil.copytree(work, output / "notes", symlinks=True)
    if explore is None:
        explore = explore_cost(maze)
    solved = not error and walk.status == "solved"
    result = {"task": task, "size": maze.size, "seed": seed, "loops": loops, "start": maze.start,
              "exit": maze.exit, "status": "error" if error else walk.status,
              "limit": limit, "attempts": walk.attempts, "moves": moves,
              "visited": len(visited), "shortest": shortest, "explore_steps": explore,
              # 1.0 means the agent walked the shortest path; 0 for an unsolved attempt.
              "efficiency": round(shortest / walk.attempts, 4) if solved and walk.attempts else 0.0,
              # Decisions relative to a remembering depth-first explorer; below 1 beats it.
              "explore_ratio": round(walk.attempts / explore, 4) if solved and explore else None}
    if error:
        result["error"] = error
    elif response_limit:
        result.update(status="limit", reason="response_limit", responses=MAX_RESPONSES_PER_MOVE,
                      stalled_turn=walk.attempts + 1)
    if gateway_error:
        result.update(reason="gateway", score=None, error=str(gateway_error))
    (output / "result.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    (output / "map.txt").write_text(maze.render(), encoding="utf-8")
    if gateway_error:
        raise gateway_error
    return result


def run_level(number: int, seed: int, limit: int | None, command: list[str], output: Path) -> dict:
    maze, map_seed, explore = generate_level(number, seed)
    return run(maze, map_seed, limit, command, output,
               task=f"02-maze-v{VERSION}-{number:02d}", explore=explore, loops=LOOPS)


def run_suite(seed: int, limit: int | None, command: list[str], output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    summary = {"suite": f"02-maze-v{VERSION}", "seed": seed, "status": "running", "score": None,
               "max_score": len(LEVELS), "efficiency": None, "levels": []}

    def save_progress():
        tmp = output / "suite.tmp"
        tmp.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        tmp.replace(output / "suite.json")

    save_progress()
    for number in range(1, len(LEVELS) + 1):
        result = run_level(number, seed, limit, command, output / f"level-{number:02d}")
        summary["levels"].append(result)
        if result["status"] == "error":
            summary["status"] = "error"
        elif number == len(LEVELS):
            summary["status"] = "complete"
            summary["score"] = sum(r["status"] == "solved" for r in summary["levels"])
            summary["efficiency"] = round(sum(r["efficiency"] for r in summary["levels"]) / len(LEVELS), 4)
        save_progress()
        if result["status"] == "error":
            break
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--suite", action="store_true", help="run all ten benchmark levels in order")
    mode.add_argument("--level", type=int, choices=range(1, len(LEVELS) + 1),
                      help="run one benchmark level")
    mode.add_argument("--size", type=int, default=10,
                      help=f"single uncalibrated map, {MIN_SIZE}–{MAX_SIZE} inclusive (default: 10)")
    parser.add_argument("--seed", type=int, help="replay the same map or suite")
    parser.add_argument("--loops", type=float, default=0.0,
                        help="single map only: share of interior walls opened as extra doors (0–1)")
    parser.add_argument("--limit", type=int, help="maximum decisions (default: 2 per room)")
    parser.add_argument("--output", type=Path, required=True, help="new result directory")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Укажите команду адаптера после --")
    if args.limit is not None and args.limit < 1:
        parser.error("Лимит ходов должен быть положительным.")
    seed = args.seed if args.seed is not None else secrets.randbits(63)
    try:
        if args.suite:
            result = run_suite(seed, args.limit, command, args.output)
        elif args.level is not None:
            result = run_level(args.level, seed, args.limit, command, args.output)
        else:
            result = run(Maze.generate(args.size, seed, args.loops), seed, args.limit, command,
                         args.output, loops=args.loops)
    except (OSError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, ensure_ascii=False))
    return int(result["status"] == "error")


if __name__ == "__main__":
    sys.exit(main())
