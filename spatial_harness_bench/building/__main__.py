"""Run five independent, text-only floor-plan tasks against a file-capable harness."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

from ..gateway import GatewayError, run_stage
from .task import TASKS, VERSION, grade, prompt


def run(command, output, levels):
    output.mkdir(parents=True, exist_ok=False)
    items = []
    for level in levels:
        task = TASKS[level - 1]
        folder = output / task.id
        folder.mkdir()
        question = prompt(level)
        (folder / "prompt.txt").write_text(question, encoding="utf-8")
        item = {"task": task.id, "level": level,
                "prompt_sha256": hashlib.sha256(question.encode()).hexdigest()}
        with TemporaryDirectory(prefix="building-room-") as tmp:
            work = Path(tmp)
            try:
                run_stage(command, work, question, env={**os.environ, "BUILDING_LEVEL": str(level)})
            except GatewayError as exc:
                item.update(exc.report("solve"), points=None)
            except (OSError, subprocess.CalledProcessError) as exc:
                item.update(status="error", reason="harness", error=str(exc), points=None)
            else:
                path = work / "answer.json"
                data = path.read_bytes() if not path.is_symlink() and path.is_file() else b""
                (folder / "answer.json").write_bytes(data)
                item.update(status="complete", **grade(level, data.decode("utf-8", errors="replace")))
        items.append(item)
    complete = all(item["status"] == "complete" for item in items)
    report = {"task": items[0]["task"] if len(items) == 1 else "building-v1",
              "version": VERSION, "status": "complete" if complete else "error",
              "score": sum(item["points"] for item in items) if complete else None,
              "max_score": len(items), "items": items}
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                        encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--level", type=int, choices=range(1, 6), help="single level (default: all five)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Укажите команду харнеса после --")
    try:
        report = run(command, args.output, [args.level] if args.level else range(1, 6))
    except OSError as exc:
        parser.exit(1, f"{exc}\n")
    for item in report["items"]:
        print(f"{item['task']}: {item['points'] if item['status'] == 'complete' else item['reason']}")
    if report["status"] == "complete":
        print(f"{report['score']}/{report['max_score']}")
    return int(report["status"] != "complete")


if __name__ == "__main__":
    sys.exit(main())
