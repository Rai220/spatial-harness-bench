"""Bridge a Pi RPC process to the short-expedition JSON-lines protocol."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys

from ..gateway import GatewayError, TEMPFAIL
from ..maze import MAX_RESPONSES_PER_MOVE
from ..maze.pi import MoveResponseLimit, read_move


def _unique_pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate key")
        result[key] = value
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", nargs=argparse.REMAINDER, help="Pi launcher with read/write tools")
    command = parser.parse_args().command
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        parser.error("Provide the Pi command after --")
    command += ["--mode", "rpc", "--no-session", "--no-context-files", "--no-extensions", "--no-skills",
                "--no-prompt-templates", "--no-themes", "--tools", "read,write"]
    with subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                          text=True, encoding="utf-8", bufsize=1) as pi:
        assert pi.stdin is not None and pi.stdout is not None
        events = (json.loads(line) for line in pi.stdout)
        try:
            for line in sys.stdin:
                request = json.loads(line)
                if request.get("type") == "end":
                    break
                if request.get("type") != "observation":
                    raise ValueError("unknown request")
                prefix = request.get("instructions", "")
                if prefix:
                    prefix += " You can use the supplied read/write tools for your own notes."
                prompt = (prefix + "\n" if prefix else "") + "Observation: " + json.dumps(
                    request["observation"], ensure_ascii=False, sort_keys=True)
                print(json.dumps({"type": "prompt", "message": prompt}, ensure_ascii=False), file=pi.stdin, flush=True)
                message = read_move(events)
                text = "".join(block["text"] for block in message["content"] if block.get("type") == "text").strip()
                try:
                    decision = json.loads(text, object_pairs_hook=_unique_pairs)
                except ValueError:
                    decision = {"action": {"type": "invalid"}}
                print(json.dumps(decision, ensure_ascii=False), flush=True)
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
                pi.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
