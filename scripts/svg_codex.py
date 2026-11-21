"""File adapter for SVG recall using a fresh, tool-free Codex prompt per stage."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spatial_harness_bench.gateway import TEMPFAIL, gateway_failure

STRING = {"type": "string"}


def obj(properties):
    return {"type": "object", "properties": properties, "required": list(properties), "additionalProperties": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", default="medium")
    args = parser.parse_args()
    stage = os.environ["SVG_RECALL_STAGE"]
    difficulty = os.environ.get("SVG_RECALL_DIFFICULTY")
    spatial = difficulty in {"hard", "expert", "master"}
    count = int(os.environ.get("SVG_RECALL_COUNT", "10"))
    folder = Path.cwd()
    prompt = sys.stdin.read()
    images = []
    if stage == "draw":
        field = "drawings"
        item = obj({"name": STRING, "svg": STRING})
        prompt += "\nАдаптер сохранит файлы за тебя. Верни массив drawings в порядке 0…9, у каждого name и полный svg."
    else:
        item = STRING
        if spatial:
            field = "answers"
            prompt += "\nАдаптер запишет desc.txt за тебя. Верни десять строк «строка,столбец | название» в массиве answers по порядку вопросов 1…10."
            inputs = ["board"]
        else:
            field = "answers" if difficulty in {"fixed", "generated"} else "names"
            prompt += f"\nАдаптер запишет desc.txt за тебя. Верни массив {field} в порядке файлов 0…{count - 1}, для пропуска верни пустую строку."
            inputs = [str(i) for i in range(count)]
        if stage == "code":
            for stem in inputs:
                prompt += f"\n\n--- {stem}.svg ---\n" + (folder / f"{stem}.svg").read_text(encoding="utf-8")
        elif stage == "vision":
            images = [folder / f"{stem}.png" for stem in inputs]
            if not spatial:
                prompt += f"\nКартинки приложены в порядке 0.png … {count - 1}.png, считай с нуля."
        else:
            raise ValueError(f"Неизвестный этап: {stage}")
    prompt += "\nВесь материал уже в сообщении. Выполни задачу без инструментов."
    size = count if stage != "draw" and not spatial else 10
    schema = obj({field: {"type": "array", "items": item, "minItems": size, "maxItems": size}})
    with TemporaryDirectory(prefix="svg-codex-") as tmp:
        work = Path(tmp)
        (work / "schema.json").write_text(json.dumps(schema), encoding="utf-8")
        command = ["codex", "exec", "--ignore-user-config", "--ephemeral", "--skip-git-repo-check",
                   "--sandbox", "read-only", "--model", args.model, "-c", f'model_reasoning_effort="{args.effort}"',
                   "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0", "--json",
                   "--output-schema", str(work / "schema.json"), "-o", str(work / "answer.json")]
        for feature in ("shell_tool", "unified_exec", "apps", "plugins", "hooks", "memories",
                        "multi_agent", "multi_agent_v2", "code_mode", "skill_search"):
            command.extend(["--disable", feature])
        for path in images:
            command.extend(["-i", str(path)])
        result = subprocess.run([*command, "-"], cwd=work, input=prompt, capture_output=True, text=True)
        if gateway_failure(result.returncode, result.stderr, result.stdout):
            sys.exit(TEMPFAIL)
        if result.returncode:
            raise RuntimeError(f"Codex ({stage}): {result.stderr[-1600:]} {result.stdout[-800:]}")
        for line in result.stdout.splitlines():
            event = json.loads(line)
            if "item" in event and event["item"]["type"] not in {"agent_message", "reasoning"}:
                raise RuntimeError(f"Codex использовал инструмент на этапе {stage}: {event['item']['type']}")
        payload = json.loads((work / "answer.json").read_text(encoding="utf-8"))[field]
    if len(payload) != size:
        raise ValueError(f"Ожидалось {size} позиций в ответе Codex.")
    if stage == "draw":
        for i, entry in enumerate(payload):
            (folder / f"{i}.svg").write_text(entry["svg"], encoding="utf-8")
        names = [entry["name"] for entry in payload]
    else:
        names = payload
    (folder / "desc.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
