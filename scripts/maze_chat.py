"""Chat Completions adapter for the maze JSON-lines protocol (standard library)."""

import argparse
from collections import deque
import json
import os
from pathlib import Path
import sys
import time
import urllib.error
import urllib.request


def file_tool(name, description, properties):
    return {"type": "function", "function": {"name": name, "description": description,
            "parameters": {"type": "object", "properties": {
                key: {"type": "string", "description": value} for key, value in properties.items()
            }, "required": list(properties), "additionalProperties": False}}}


TOOLS = [
    file_tool("read_file", "Прочитать свою заметку.", {"name": "Имя файла"}),
    file_tool("write_file", "Создать или заменить свою заметку.",
              {"name": "Имя файла", "text": "Содержимое"}),
    file_tool("append_file", "Дописать текст в конец своей заметки.",
              {"name": "Имя файла", "text": "Добавляемый текст"}),
]


def use_file(call):
    try:
        args = json.loads(call["arguments"])
        name = args["name"]
        if not isinstance(name, str) or not name or Path(name).name != name or name in (".", ".."):
            return "Используй простое имя файла внутри рабочей папки."
        path = Path(name)
        if path.is_symlink():
            return "Недоступный файл."
        if call["name"] == "read_file":
            return path.read_text(encoding="utf-8")
        if call["name"] not in ("append_file", "write_file") or not isinstance(args["text"], str):
            return "Неизвестная операция с файлом."
        with path.open("a" if call["name"] == "append_file" else "w", encoding="utf-8") as f:
            f.write(args["text"])
        return "Сохранено."
    except (ValueError, OSError, KeyError, TypeError) as exc:
        return f"Ошибка заметки: {exc}"


def complete(endpoint, key, payload):
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers)
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=180) as response:
                return json.load(response)
        except urllib.error.HTTPError as exc:
            if exc.code not in (408, 429, 500, 502, 503, 504) or attempt == 3:
                raise RuntimeError(f"HTTP {exc.code}: {exc.read(500).decode(errors='replace')}") from exc
        except (TimeoutError, urllib.error.URLError):
            if attempt == 3:
                raise
        time.sleep(2**attempt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--effort", help="optional reasoning_effort supported by the provider")
    args = parser.parse_args()
    endpoint = os.environ["OPENAI_BASE_URL"].rstrip("/") + "/chat/completions"
    key = os.environ.get("OPENAI_API_KEY")
    history = deque()
    history_chars = 0
    system = ""
    model_ids = set()
    for line in sys.stdin:
        room = json.loads(line)
        if room["type"] == "end":
            break
        if "instructions" in room:
            system = (room["instructions"] + " Ты можешь читать и писать свои файлы инструментами. "
                      "Для хода ответь ровно одним словом: вперёд, назад, вправо или влево.")
        turn = [{"role": "user", "content": f"Ход {room['turn']}. {room['text']}"}]
        for _ in range(32):
            messages = [{"role": "system", "content": system}]
            for _, previous in history:
                messages.extend(previous)
            messages.extend(turn)
            payload = {"model": args.model, "messages": messages, "tools": TOOLS}
            if args.effort:
                payload["reasoning_effort"] = args.effort
            response = complete(endpoint, key, payload)
            reported = response.get("model")
            if reported not in model_ids:
                print(f"Requested {args.model}, provider returned {reported}", file=sys.stderr)
                model_ids.add(reported)
            msg = response["choices"][0]["message"]
            answer = {"role": "assistant", "content": msg.get("content") or ""}
            turn.append(answer)
            if calls := msg.get("tool_calls"):
                answer["tool_calls"] = calls
                for call in calls:
                    turn.append({"role": "tool", "tool_call_id": call["id"],
                                 "content": use_file(call["function"])})
                continue
            print(json.dumps({"move": answer["content"]}, ensure_ascii=False), flush=True)
            break
        else:
            raise RuntimeError("Модель не выбрала ход за 32 ответа.")
        length = len(json.dumps(turn, ensure_ascii=False))
        history.append((length, turn))
        history_chars += length
        # Evict complete turns with their tool results on very long walks.
        # The current room, task instructions and notebook files persist.
        while history_chars > 180_000 and history:
            history_chars -= history.popleft()[0]


if __name__ == "__main__":
    main()
