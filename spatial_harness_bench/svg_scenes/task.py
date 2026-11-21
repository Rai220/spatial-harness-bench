"""Author-owned fixtures and an ordered, closed answer key. A respondent never sees the key."""

from dataclasses import dataclass
from hashlib import sha256
from importlib.resources import files
import json
import re
import subprocess

from ..answers import MATCHER_VERSION, matches_any, normalize
from ..svg_recall.verifier import Picture, inspect

TASK_ID = "04-svg-scenes-v1"
GENERATED_TASK_ID = "04-svg-scenes-gen-v1"


@dataclass(frozen=True)
class Scene:
    file: str
    question: str
    answers: list[list[str]]
    picture: Picture
    kind: str = "fixed"
    tier: int = 0


def generated_scenes(seed: int, per_tier: int) -> tuple[list[Scene], str]:
    from . import generated

    items = generated.generate(seed, per_tier)
    digest = sha256(f"{generated.GENERATOR_VERSION}:{seed}:{per_tier}".encode())
    scenes = []
    for i, item in enumerate(items):
        digest.update(sha256(item.picture.source).digest())
        digest.update(json.dumps([item.question, item.answers], ensure_ascii=False).encode())
        scenes.append(Scene(f"{i}.svg", item.question, item.answers, item.picture, item.kind, item.tier))
    return scenes, digest.hexdigest()


def load_scenes() -> tuple[list[Scene], str]:
    available = subprocess.check_output(["tesseract", "--list-langs"], text=True).splitlines()
    if not {"rus", "eng"}.issubset(available):
        raise RuntimeError("Для проверки подписей нужны языки Tesseract rus и eng.")
    package = files(__package__)
    manifest = package.joinpath("items.json").read_bytes()
    items = json.loads(manifest)
    if [item["file"] for item in items] != [f"{i}.svg" for i in range(10)]:
        raise ValueError("Нужны десять последовательных рисунков 0.svg … 9.svg.")
    digest = sha256(manifest)
    vocabulary = [alias for item in items for part in item["answers"] for alias in part]
    scenes = []
    for item in items:
        name = item["file"]
        source = package.joinpath("art", name).read_bytes()
        picture = inspect(source, vocabulary)
        if picture.issue:
            raise ValueError(f"Ошибка в материале теста {name}: {picture.issue.reason}")
        digest.update(sha256(source).digest())
        scenes.append(Scene(name, item["question"], item["answers"], picture))
    return scenes, digest.hexdigest()


def prompt_for(mode: str, scenes: list[Scene], legend: str = "") -> str:
    if mode == "code":
        instruction = "Прочитай десять SVG-файлов как текст и определи ответ по геометрии. Не рендери изображения."
        ext = "svg"
    elif mode == "vision":
        instruction = "Рассмотри десять PNG-изображений с помощью зрения."
        ext = "png"
    else:
        raise ValueError(f"Неизвестный режим {mode}")
    count = len(scenes)
    instruction = instruction.replace("десять", str(count)) if count != 10 else instruction
    rows = [instruction, "Рисунки уже готовы. Для каждого ответь на его вопрос по-русски. "
            f"Сохрани ответы в desc.txt по одному на строку в порядке 0…{count - 1}. Дай только название, цвет, число "
            "или запрошенную пару через запятую. Слева и справа — со стороны зрителя. "
            "Для пропуска оставь соответствующую строку пустой."]
    if legend:
        rows.append(legend)
    for i, scene in enumerate(scenes):
        rows.append(f"{i}.{ext} — {scene.question}")
    return "\n".join(rows) + "\n"


def grade(scenes: list[Scene], data: bytes) -> dict:
    lines = data.decode("utf-8-sig", errors="replace").splitlines()
    items = []
    for i, scene in enumerate(scenes):
        raw = lines[i] if i < len(lines) else ""
        value = normalize(raw).strip(" .!?…")
        parts = re.split(r"\s*[,;]\s*|\s+и\s+", value) if len(scene.answers) > 1 else [value]
        ok = len(parts) == len(scene.answers) and all(
            matches_any(part, aliases) for part, aliases in zip(parts, scene.answers)
        )
        row = {"level": i+1, "file": scene.file, "question": scene.question, "guess": raw,
               "expected": ", ".join(aliases[0] for aliases in scene.answers), "points": int(ok)}
        if scene.kind != "fixed":
            row.update(kind=scene.kind, tier=scene.tier)
        items.append(row)
    report = {"score": sum(i["points"] for i in items), "max_score": len(scenes),
              "matcher": MATCHER_VERSION, "items": items}
    if any(scene.kind != "fixed" for scene in scenes):
        breakdown = {}
        for item in items:
            cell = breakdown.setdefault(item["kind"], {}).setdefault(str(item["tier"]), [0, 0])
            cell[0] += item["points"]
            cell[1] += 1
        report["by_kind"] = {kind: {tier: f"{a}/{b}" for tier, (a, b) in tiers.items()}
                             for kind, tiers in breakdown.items()}
    return report
