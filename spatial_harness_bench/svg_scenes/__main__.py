"""Run SVG scene questions from source, from pixels, or both.

--set fixed      ten authored scenes (public, v1)
--set generated  procedural scenes from a seed: keep the seed private for a closed test set
"""

import argparse
import json
from pathlib import Path
import secrets
import subprocess
import sys
from tempfile import TemporaryDirectory

from ..gateway import GatewayError
from ..ascii_recall.__main__ import read_output
from ..svg_recall.__main__ import session
from .task import GENERATED_TASK_ID, TASK_ID, generated_scenes, grade, load_scenes, prompt_for


def run(command: list[str], output: Path, mode: str, *, scene_set: str = "fixed",
        seed: int | None = None, per_tier: int = 2) -> dict:
    if output.exists():
        raise FileExistsError(f"Каталог результата уже существует: {output}")
    modes = ("code", "vision") if mode == "both" else (mode,)
    if scene_set == "generated":
        from .generated import legend

        seed = secrets.randbits(63) if seed is None else seed
        scenes, digest = generated_scenes(seed, per_tier)
        report = {"task": GENERATED_TASK_ID, "seed": seed, "per_tier": per_tier,
                  "dataset_sha256": digest, "modes": {}}
        difficulty = "generated"
    elif scene_set == "fixed":
        if seed is not None:
            raise ValueError("Seed применяется к набору generated.")
        scenes, digest = load_scenes()
        report = {"task": TASK_ID, "dataset_sha256": digest, "modes": {}}
        difficulty, legend = "fixed", None
    else:
        raise ValueError(f"Неизвестный набор: {scene_set}")
    answers = {}
    for current in modes:
        prompt = prompt_for(current, scenes, legend(current) if legend else "")
        with TemporaryDirectory(prefix=f"fixed-svg-{current}-") as tmp:
            folder = Path(tmp)
            for i, scene in enumerate(scenes):
                if current == "code":
                    (folder / scene.file).write_bytes(scene.picture.source)
                else:
                    (folder / f"{i}.png").write_bytes(scene.picture.png)
            try:
                session(command, folder, current, prompt, difficulty, count=len(scenes))
            except GatewayError as exc:
                report["modes"][current] = exc.report(current)
                continue
            except subprocess.CalledProcessError as exc:
                report["modes"][current] = {"status": "error", "score": None, "returncode": exc.returncode}
                continue
            answers[current] = read_output(folder, "desc.txt")
            report["modes"][current] = {"status": "complete", **grade(scenes, answers[current])}
    output.mkdir(parents=True, exist_ok=False)
    for current, answer in answers.items():
        (output / current).mkdir()
        (output / current / "desc.txt").write_bytes(answer)
    if scene_set == "generated":
        # The scenes are reproducible from the seed; keep a copy for inspection.
        scenes_dir = output / "scenes"
        scenes_dir.mkdir()
        for i, scene in enumerate(scenes):
            (scenes_dir / scene.file).write_bytes(scene.picture.source)
            (scenes_dir / f"{i}.png").write_bytes(scene.picture.png)
        (scenes_dir / "items.json").write_text(json.dumps(
            [{"file": s.file, "kind": s.kind, "tier": s.tier, "question": s.question, "answers": s.answers}
             for s in scenes], ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "result.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("code", "vision", "both"), default="code")
    parser.add_argument("--set", dest="scene_set", choices=("fixed", "generated"), default="fixed")
    parser.add_argument("--seed", type=int, help="generated set: the seed that defines all items")
    parser.add_argument("--per-tier", type=int, default=2,
                        help="generated set: items per kind and tier (4 kinds × 3 tiers)")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="-- COMMAND [ARG ...]")
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("Укажите команду харнеса после --")
    try:
        if args.per_tier < 1:
            parser.error("--per-tier должен быть положительным.")
        report = run(command, args.output, args.mode, scene_set=args.scene_set,
                     seed=args.seed, per_tier=args.per_tier)
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(exc, file=sys.stderr)
        return 1
    for name, row in report["modes"].items():
        print(f"{name}: {row['score']}/{row['max_score']}" if row["status"] == "complete" else f"{name}: ошибка харнеса")
    return int(any(row["status"] != "complete" for row in report["modes"].values()))


if __name__ == "__main__":
    sys.exit(main())
