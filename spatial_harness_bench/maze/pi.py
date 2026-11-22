"""Drive a configured Pi CLI in RPC mode from the maze JSON-lines protocol."""

import argparse
import json
import subprocess
import sys

from ..gateway import GatewayError, TEMPFAIL
from . import MAX_RESPONSES_PER_MOVE


class MoveResponseLimit(Exception):
    """The last available assistant reply requested tools instead of submitting a move."""


def read_move(events):
    """Return the final assistant message for one room. Pi executes its own tools."""
    replies = 0
    last = None
    for event in events:
        kind = event["type"]
        if kind == "response" and not event.get("success"):
            raise RuntimeError(event.get("error", "Pi rejected the RPC command"))
        if kind == "message_end" and event["message"].get("role") == "assistant":
            last = event["message"]
            if last.get("stopReason") in ("error", "aborted"):
                raise GatewayError(last.get("errorMessage") or "Pi inference interrupted")
            replies += 1
            tools = any(block["type"] == "toolCall" for block in last["content"])
            if replies > MAX_RESPONSES_PER_MOVE or (replies == MAX_RESPONSES_PER_MOVE and tools):
                raise MoveResponseLimit
        if kind == "agent_end":
            if last is None:
                raise RuntimeError("Pi ended the turn without an assistant message")
            return last
    raise RuntimeError("Pi closed the RPC stream during a move")


def room_prompt(room):
    instructions = ""
    if room["turn"] == 1:
        instructions = (room["instructions"] + " Можно вести заметки в файле. "
                        "Ответь одним словом: вперёд, назад, вправо или влево.\n")
    return instructions + f"Ход {room['turn']}. " + room["text"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER,
                        help="Pi launcher with file access isolated to the current notes directory")
    command = parser.parse_args().command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("Укажите команду запуска изолированного Pi после --")
    command += [
        "--mode", "rpc", "--no-session", "--no-context-files", "--no-extensions", "--no-skills",
        "--no-prompt-templates", "--no-themes", "--tools", "read,write",
    ]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", bufsize=1) as pi:
        events = (json.loads(line) for line in pi.stdout)
        try:
            for line in sys.stdin:
                room = json.loads(line)
                if room["type"] == "end":
                    break
                print(json.dumps({"type": "prompt", "message": room_prompt(room)}, ensure_ascii=False),
                      file=pi.stdin, flush=True)
                msg = read_move(events)
                move = "".join(b["text"] for b in msg["content"] if b["type"] == "text").strip()
                print(json.dumps({"move": move}, ensure_ascii=False), flush=True)
        except MoveResponseLimit:
            print(json.dumps({"type": "response_limit", "responses": MAX_RESPONSES_PER_MOVE}), flush=True)
        except GatewayError:
            print(json.dumps({"type": "gateway_error"}), flush=True)
            return TEMPFAIL
        finally:
            pi.terminate()
            try:
                pi.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pi.kill()
    return 0


if __name__ == "__main__":
    sys.exit(main())
