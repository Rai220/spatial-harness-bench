"""OpenAI-compatible chat adapter for the short-expedition JSONL protocol."""
from __future__ import annotations

import argparse
from collections import deque
import json
import os
import sys

from ..adnd_memory.chat import TOOLS, _endpoint, _file_tool, complete
from ..gateway import GatewayError, TEMPFAIL
from ..maze import MAX_RESPONSES_PER_MOVE
from .__main__ import canonical_decision


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--effort", help="optional provider reasoning_effort")
    parser.add_argument("--context-turns", type=int, default=20)
    parser.add_argument("--notes", action="store_true", help="enable read/write/append file tools")
    args = parser.parse_args()
    if not 0 <= args.temperature <= 2 or args.context_turns < 1 or (args.max_tokens is not None and args.max_tokens < 1):
        parser.error("invalid generation settings")
    endpoint, key = _endpoint(), os.environ.get("OPENAI_API_KEY")
    history = deque(maxlen=args.context_turns - 1 or 1)
    system = None
    for line in sys.stdin:
        packet = json.loads(line)
        if packet.get("type") == "end":
            return 0
        if packet.get("type") != "observation" or not isinstance(packet.get("observation"), dict):
            raise ValueError("invalid observation packet")
        if system is None:
            system = packet["instructions"]
            system += (" File tools are available for your own notes." if args.notes
                       else " File tools are not available in this mode.")
        current = [{"role": "user", "content": json.dumps(packet["observation"], ensure_ascii=False, sort_keys=True)}]
        for _ in range(MAX_RESPONSES_PER_MOVE):
            messages = [{"role": "system", "content": system}]
            if args.context_turns > 1:
                for turn in history:
                    messages.extend(turn)
            messages.extend(current)
            payload = {"model": args.model, "messages": messages, "temperature": args.temperature}
            if args.effort:
                payload["reasoning_effort"] = args.effort
            if args.max_tokens is not None:
                payload["max_tokens"] = args.max_tokens
            if args.notes:
                payload["tools"] = TOOLS
            body = complete(endpoint, key, payload)
            choice = body["choices"][0]
            # A cut-off response is not a model decision, even if its prefix
            # happens to be valid JSON. Preserve this as an unscored failure.
            if choice.get("finish_reason") == "length":
                print(json.dumps({"type": "output_limit"}), flush=True)
                return 0
            msg = choice["message"]
            content = msg.get("content")
            calls = msg.get("tool_calls") or []
            if calls:
                if not args.notes:
                    raise ValueError("unexpected tool calls")
                current.append({"role": "assistant", "content": content or "", "tool_calls": calls})
                for call in calls:
                    current.append({"role": "tool", "tool_call_id": call["id"],
                                    "content": _file_tool(call["function"])})
                continue
            # A provider that consumed its output budget without text is an
            # inference failure, not an invalid game action or loss.
            if not isinstance(content, str):
                raise GatewayError("model returned no text")
            action = canonical_decision(content.encode("utf-8"), packet["observation"])
            decision = {"action": action}
            current.append({"role": "assistant", "content": json.dumps(decision, ensure_ascii=False, sort_keys=True)})
            print(json.dumps(decision, ensure_ascii=False), flush=True)
            break
        else:
            print(json.dumps({"type": "response_limit", "responses": MAX_RESPONSES_PER_MOVE}), flush=True)
            return 0
        history.append(current)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except GatewayError:
        print(json.dumps({"type": "gateway_error"}), flush=True)
        sys.exit(TEMPFAIL)
    except (ValueError, KeyError, TypeError, IndexError, OSError):
        print("Adapter configuration or response error.", file=sys.stderr)
        sys.exit(1)
