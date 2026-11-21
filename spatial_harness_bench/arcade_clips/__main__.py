"""Run the replay-index task with a harness that reads its prompt from stdin."""
import argparse
import json
import os
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

from .sandbox import query_command
from .task import PROMPT, TASK_ID, grade, read_submission, setup


def run(command: list[str], output: Path) -> dict:
    if output.exists():
        raise FileExistsError(f"Каталог результата уже существует: {output}")
    with TemporaryDirectory(prefix="arcade-clips-") as tmp:
        folder = Path(tmp)
        # Detect unavailable isolation before spending a model run.
        probe = folder / "probe.py"
        probe.write_text("import json\n", encoding="utf-8")
        subprocess.run(query_command(probe), env={"PATH": os.defpath}, check=True)
        probe.unlink()
        setup(folder)
        timed_out = False
        with subprocess.Popen(command, cwd=folder, stdin=subprocess.PIPE, start_new_session=True) as proc:
            try:
                proc.communicate(PROMPT.encode(), timeout=3600)
            except subprocess.TimeoutExpired:
                timed_out = True
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            if not timed_out and proc.returncode:
                raise subprocess.CalledProcessError(proc.returncode, command)
        index = read_submission(folder / "clips.rix")
        program = read_submission(folder / "query.py")
    score, message = (0, "Harness timed out after 3600s") if timed_out else grade(index, program)
    result = {"task": TASK_ID, "score": score, "max_score": 1, "message": message}
    output.mkdir(parents=True, exist_ok=False)
    for name, data in (("clips.rix", index), ("query.py", program)):
        if data:
            (output / name).write_bytes(data)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...]")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Укажите команду харнеса после --")
    try:
        result = run(command, args.output)
    except (OSError, subprocess.SubprocessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    print(f"{result['score']}/1  {args.output / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
