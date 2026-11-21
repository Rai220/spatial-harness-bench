"""Run the two prompts through the same command in independent workspaces."""

import argparse
import json
from pathlib import Path
import secrets
import subprocess
import sys
from tempfile import TemporaryDirectory

from ..gateway import GatewayError, run_stage, save_error
from .. import objects as assigned_objects
from .task import DRAW_PROMPT, FILES, RECALL_PROMPT, TASK_ID

ASSIGNED_INTRO = """Нарисуй в ASCII десять назначенных объектов реального мира так, чтобы потом
узнать каждый по одному рисунку, когда история разговора и список названий будут скрыты.
"""
from .verifier import check_drawing, parse_reference, score


def read_output(folder: Path, name: str) -> bytes:
    path = folder / name
    if path.is_symlink() or not path.is_file():
        return b""
    return path.read_bytes()


def run(command: list[str], output: Path, *, objects: str = "free", seed: int | None = None) -> dict:
    if output.exists():
        raise FileExistsError(f"Каталог результата уже существует: {output}")
    if objects not in {"free", "assigned"}:
        raise ValueError(f"Неизвестный набор объектов: {objects}")
    if objects == "free" and seed is not None:
        raise ValueError("Seed применяется к набору assigned.")
    if objects == "assigned":
        seed = secrets.randbits(63) if seed is None else seed
        assigned = assigned_objects.assign(seed)
        prompt = ASSIGNED_INTRO + "\n" + DRAW_PROMPT.split("\n\n", 1)[1] + assigned_objects.draw_prompt(assigned, "txt")
    else:
        prompt = DRAW_PROMPT

    with TemporaryDirectory(prefix="ascii-draw-") as tmp:
        folder = Path(tmp)
        try:
            run_stage(command, folder, prompt)
        except GatewayError as exc:
            return save_error(output, TASK_ID, 10, "draw", exc, objects=objects, seed=seed)
        if objects == "assigned":
            # The runner owns the reference; the artist's copy is irrelevant.
            reference = ("\n".join(assigned) + "\n").encode()
        else:
            reference = read_output(folder, "desc.txt")
        names = parse_reference(reference)
        drawings = [read_output(folder, name) for name in FILES]

    # The first process has exited and its workspace has been deleted. Only
    # picture contents cross this boundary. Even file metadata is recreated.
    checks = [check_drawing(data, names) for data in drawings]
    with TemporaryDirectory(prefix="ascii-recall-") as tmp:
        folder = Path(tmp)
        for name, data, issue in zip(FILES, drawings, checks, strict=True):
            (folder / name).write_bytes(data if issue is None else b"")
        try:
            run_stage(command, folder, RECALL_PROMPT)
        except GatewayError as exc:
            return save_error(output, TASK_ID, 10, "recall", exc, objects=objects, seed=seed)
        answer = read_output(folder, "desc.txt")

    result = score(names, answer, checks)
    result["objects"] = objects
    if seed is not None:
        result["seed"] = seed
    # Publish the original files for inspection only after recognition ends.
    output.mkdir(parents=True, exist_ok=False)
    originals = output / "original"
    originals.mkdir()
    for name, data in zip(FILES, drawings, strict=True):
        (originals / name).write_bytes(data)
    (originals / "desc.txt").write_bytes(reference)
    (output / "desc.txt").write_bytes(answer)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path, help="new result directory")
    parser.add_argument("--objects", choices=("free", "assigned"), default="free",
                        help="free: the model picks its subjects; assigned: the runner picks them by seed")
    parser.add_argument("--seed", type=int, help="replay the assigned object list")
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="-- COMMAND [ARG ...] (fresh harness session, prompt on stdin)")
    args = parser.parse_args()
    command = args.command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("Укажите команду запуска харнеса после --")
    try:
        result = run(command, args.output, objects=args.objects, seed=args.seed)
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    if result.get("status") == "error":
        print("Gateway error. No score.", file=sys.stderr)
        return 1
    print(f"{result['score']}/10  {args.output / 'result.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
