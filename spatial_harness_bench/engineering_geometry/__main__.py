"""Run isolated file-based engineering geometry editing sessions."""

import argparse
from dataclasses import asdict
import json
from math import isfinite
from pathlib import Path
import secrets
import sys

from .model import TASK_ID
from .sandbox import execute
from .svg import render_svg
from .task import FAMILIES, MAX_SEED, make_instance, prompt_for
from .verifier import verify


MAX_COUNT = 1000


def _write_json(path: Path, report: dict) -> None:
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run(command: list[str], output: Path, *, image: str, family: str, seed: int, count: int = 1,
        timeout: float = 180, network: str = "none") -> dict:
    if family not in (*FAMILIES, "all"):
        raise ValueError("unknown engineering geometry family")
    if not 1 <= count <= MAX_COUNT or not 0 <= seed <= seed + count - 1 <= MAX_SEED:
        raise ValueError(f"count must be between 1 and {MAX_COUNT}; all seeds must be in [0, 2^63-1]")
    if not isfinite(timeout) or timeout <= 0:
        raise ValueError("timeout must be positive and finite")
    if not command or not image:
        raise ValueError("a Docker image and harness command are required")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    rows = []
    for current_family in FAMILIES if family == "all" else (family,):
        for offset in range(count):
            instance = make_instance(current_family, seed + offset)
            source = render_svg(instance.original)
            prompt = prompt_for(instance)
            name = f"{len(rows) + 1:04d}"
            folder = output / name
            folder.mkdir(mode=0o700)
            (folder / "input.svg").write_bytes(source)
            (folder / "prompt.txt").write_text(prompt, encoding="utf-8")
            row = {"family": current_family, "seed": instance.seed, "artifacts": name,
                   "score": None, "max_score": 1}
            try:
                outcome = execute(image, command, source, prompt, timeout=timeout, network=network)
            except ValueError as exc:
                row.update(status="technical_error", code="harness_config", detail=str(exc))
            else:
                row.update(returncode=outcome.returncode, logs_truncated=outcome.logs_truncated)
                for filename, data in (("stdout.log", outcome.stdout), ("stderr.log", outcome.stderr),
                                       ("output.svg", outcome.output_svg), ("status.json", outcome.status_json)):
                    if data is not None:
                        (folder / filename).write_bytes(data)
                if outcome.technical_error:
                    row.update(status="technical_error", code="harness_execution", detail=outcome.technical_error)
                elif outcome.agent_error:
                    row.update(status="agent_error", score=0, code="invalid_artifact", detail=outcome.agent_error)
                else:
                    verdict = verify(instance, outcome.output_svg, outcome.status_json)
                    row.update(asdict(verdict), status="passed" if verdict.passed else "agent_error",
                               score=int(verdict.passed))
            _write_json(folder / "result.json", {"task": TASK_ID, **row})
            rows.append(row)
    passed = sum(row["status"] == "passed" for row in rows)
    agent_errors = sum(row["status"] == "agent_error" for row in rows)
    technical_errors = len(rows) - passed - agent_errors
    graded = len(rows) - technical_errors
    report = {"task": TASK_ID, "family": family, "seed": seed, "count_per_family": count,
              "image": image, "total": len(rows), "passed": passed, "agent_errors": agent_errors,
              "technical_errors": technical_errors, "score": passed if graded else None,
              "max_score": graded, "items": rows}
    _write_json(output / "result.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", required=True, choices=(*FAMILIES, "all"))
    parser.add_argument("--seed", type=int, help="first seed (default: a randomized private seed)")
    parser.add_argument("--count", type=int, default=1, help="consecutive seeds per selected family")
    parser.add_argument("--image", required=True, help="prebuilt local Docker harness image")
    parser.add_argument("--network", default="none", help="none (default) or evaluator-managed bridge")
    parser.add_argument("--timeout", type=float, default=180, help="harness deadline per instance, in seconds")
    parser.add_argument("--output", type=Path, required=True, help="new evaluator-only artifact directory")
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...] inside the image")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("provide a harness command after --")
    try:
        report = run(command, args.output, image=args.image, family=args.family,
                     seed=args.seed if args.seed is not None else secrets.randbits(62), count=args.count,
                     timeout=args.timeout, network=args.network)
    except (OSError, ValueError) as exc:
        print(exc, file=sys.stderr)
        return 2
    print(f"{report['passed']}/{report['max_score']} passed; "
          f"{report['agent_errors']} agent errors, {report['technical_errors']} technical errors "
          f"({report['total']} planned)")
    return 2 if report["technical_errors"] else int(bool(report["agent_errors"]))


if __name__ == "__main__":
    sys.exit(main())
