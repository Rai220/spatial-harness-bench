"""Draw once, then recognize the SVG source in a fresh session."""

import argparse
import json
import os
from pathlib import Path
import secrets
import subprocess
import sys
from tempfile import TemporaryDirectory

from ..gateway import GatewayError, run_stage, save_error
from .. import objects as assigned_objects
from ..answers import same
from ..ascii_recall.__main__ import read_output
from ..ascii_recall.verifier import parse_reference, score
from . import hard
from .task import DRAW_PROMPT, FILES, KITCHEN_DRAW, KITCHEN_PAIRS, KITCHEN_RECALL, PNG_FILES, TASK_ID, recall_prompt
from .verifier import inspect


ASSIGNED_INTRO = """Нарисуй в SVG десять назначенных объектов реального мира так, чтобы потом узнать
каждый по одному рисунку, когда список названий и история будут скрыты. Объект должен быть
узнаваем по геометрии в коде SVG.

"""


def session(command: list[str], folder: Path, stage: str, prompt: str, difficulty: str,
            count: int = 10) -> None:
    env = {**os.environ, "SVG_RECALL_STAGE": stage, "SVG_RECALL_DIFFICULTY": difficulty,
           "SVG_RECALL_COUNT": str(count)}
    run_stage(command, folder, prompt, env=env)


def recognize(command: list[str], folder: Path, prompt: str, difficulty: str, count: int = 10):
    try:
        session(command, folder, "code", prompt, difficulty, count=count)
    except GatewayError as exc:
        return None, exc.report("code")
    except subprocess.CalledProcessError as exc:
        return None, {"status": "error", "score": None, "returncode": exc.returncode}
    return read_output(folder, "desc.txt"), {"status": "complete"}


def run(command: list[str], output: Path, mode: str = "code", *, difficulty: str = "basic",
        seed: int | None = None, drawings_dir: Path | None = None, objects: str = "free") -> dict:
    if mode != "code":
        raise ValueError(f"Неизвестный режим: {mode}")
    if output.exists():
        raise FileExistsError(f"Каталог результата уже существует: {output}")
    if difficulty not in {"basic", *hard.LEVEL_COUNTS}:
        raise ValueError(f"Неизвестный уровень: {difficulty}")
    if objects not in {"free", "kitchen", "assigned"}:
        raise ValueError(f"Неизвестный набор предметов: {objects}")
    spatial = difficulty != "basic"
    if not spatial and seed is not None and objects != "assigned":
        raise ValueError("Seed применяется к уровням hard, expert, master и к набору assigned.")
    if objects == "assigned" and drawings_dir is not None:
        raise ValueError("Набор assigned рисуется заново; для готовых рисунков укажите их набор.")
    seed = secrets.randbits(63) if seed is None and (spatial or objects == "assigned") else seed
    assigned = assigned_objects.assign(seed) if objects == "assigned" else None
    languages = subprocess.check_output(["tesseract", "--list-langs"], text=True).splitlines()
    if not {"rus", "eng"}.issubset(languages):
        raise RuntimeError("Для проверки подписей нужны языки Tesseract rus и eng.")

    with TemporaryDirectory(prefix="svg-draw-") as tmp:
        folder = Path(tmp)
        if drawings_dir is None:
            draw_prompt = DRAW_PROMPT
            if objects == "kitchen":
                draw_prompt += KITCHEN_DRAW
            elif assigned:
                draw_prompt = (ASSIGNED_INTRO + DRAW_PROMPT.split("\n\n", 1)[1]
                               + assigned_objects.draw_prompt(assigned, "svg"))
            if spatial:
                draw_prompt += "\nТвои рисунки поместят в клетки общего поля для задач на мысленные перемещения."
            try:
                session(command, folder, "draw", draw_prompt, difficulty)
            except GatewayError as exc:
                task = f"{TASK_ID}-{difficulty}" if spatial else TASK_ID
                return save_error(output, task, 10, "draw", exc, objects=objects, seed=seed,
                                  modes={"code": exc.report("draw")})
        else:
            folder = drawings_dir
        if assigned:
            reference = ("\n".join(assigned) + "\n").encode()
        else:
            reference = read_output(folder, "desc.txt")
        names = parse_reference(reference)
        if objects == "kitchen":
            complete = sum(all(any(same(name, item) for name in names) for item in pair)
                           for pair in KITCHEN_PAIRS)
            listed = all(any(same(name, item) for pair in KITCHEN_PAIRS for item in pair) for name in names)
            if complete != 5 or not listed:
                raise ValueError("Набор kitchen требует десять предметов из пяти полных пар.")
        drawings = [read_output(folder, name) for name in FILES]

    pictures = [inspect(data, names) for data in drawings]
    issues = [picture.issue for picture in pictures]
    report = {
        "task": f"{TASK_ID}-{difficulty}" if spatial else TASK_ID,
        "checks": [{"file": name, "status": issue.kind if issue else "ok",
                    "reason": issue.reason if issue else None}
                   for name, issue in zip(FILES, issues, strict=True)],
        "modes": {},
        "objects": objects,
    }
    if seed is not None:
        report["seed"] = seed
    if spatial:
        board, questions = hard.make_questions(seed, difficulty=difficulty)
        board_svg, board_png = hard.render_board(board, pictures)
        instructions = hard.questions_text(questions, expert=difficulty != "hard")
        report.update(seed=seed, board=board, drawing_stage="reused" if drawings_dir else "fresh")
    with TemporaryDirectory(prefix="svg-code-") as tmp:
        folder = Path(tmp)
        if spatial:
            (folder / "board.svg").write_bytes(board_svg)
            prompt = "Прочитай board.svg как текст и определи предметы по геометрии SVG. Не рендери поле.\n" + instructions
        else:
            for name, picture in zip(FILES, pictures, strict=True):
                (folder / name).write_bytes(picture.source)
            prompt = recall_prompt("code")
        if objects == "kitchen":
            prompt += KITCHEN_RECALL
        answer, result = recognize(command, folder, prompt, difficulty)
    if answer is not None:
        if spatial:
            result.update(hard.grade(board, questions, names, issues, answer, task=report["task"]))
        else:
            result.update(score(names, answer, issues, files=FILES, task=TASK_ID))
    report["modes"]["code"] = result

    output.mkdir(parents=True, exist_ok=False)
    original = output / "original"
    rendered = output / "rendered"
    original.mkdir()
    rendered.mkdir()
    (original / "desc.txt").write_bytes(reference)
    if spatial:
        (output / "board.svg").write_bytes(board_svg)
        (output / "board.png").write_bytes(board_png)
        (output / "questions.txt").write_text(instructions, encoding="utf-8")
    for name, png_name, data, picture in zip(FILES, PNG_FILES, drawings, pictures, strict=True):
        (original / name).write_bytes(data)
        (rendered / png_name).write_bytes(picture.png)
    if answer is not None:
        (output / "code").mkdir()
        (output / "code/desc.txt").write_bytes(answer)
    (output / "result.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("code",), default="code")
    parser.add_argument("--difficulty", choices=("basic", *hard.LEVEL_COUNTS), default="basic")
    parser.add_argument("--seed", type=int, help="reproduce a board and its questions, or an assigned object list")
    parser.add_argument("--drawings", type=Path, help="reuse a drawing session's original/ directory")
    parser.add_argument("--objects", choices=("free", "kitchen", "assigned"), default="free")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...]")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Укажите команду харнеса после --")
    try:
        report = run(command, args.output, args.mode, difficulty=args.difficulty,
                     seed=args.seed, drawings_dir=args.drawings, objects=args.objects)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for mode, result in report["modes"].items():
        print(f"{mode}: {result['score']}/10" if result["status"] == "complete" else f"{mode}: ошибка харнеса")
    return int(any(r["status"] != "complete" for r in report["modes"].values()))


if __name__ == "__main__":
    sys.exit(main())
