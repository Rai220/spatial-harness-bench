"""Native, isolated Codex and OpenCode transports. No benchmark or verifier imports."""

import json
import os
import secrets
import socket
import subprocess
import sys
from time import monotonic, sleep


class TransportError(Exception):
    pass


class ProviderError(TransportError):
    pass


class ToolViolation(TransportError):
    pass


def codex_config(model, *, tool_free=False):
    flags = ["--model", model, "-c", 'web_search="disabled"', "-c", "project_doc_max_bytes=0"]
    disabled = ["apps", "plugins", "hooks", "memories", "multi_agent", "multi_agent_v2", "code_mode", "skill_search"]
    if tool_free:
        disabled += ["shell_tool", "unified_exec"]
    for feature in disabled:
        flags.extend(("--disable", feature))
    return flags


def strict_pairs(items):
    obj = {}
    for key, value in items:
        if key in obj:
            raise ValueError("duplicate JSON key")
        obj[key] = value
    return obj


def json_answer(text):
    text = text.strip()
    if text.startswith("```") and text.endswith("```") and "\n" in text:
        text = text.split("\n", 1)[1][:-3].strip()
    return json.loads(text, object_pairs_hook=strict_pairs)


def events(data):
    for line in data.splitlines():
        try:
            value = json.loads(line)
        except ValueError:
            continue
        if isinstance(value, dict):
            yield value


def event_failure(records):
    return any(e.get("type") in {"error", "session.error", "turn.failed"} for e in records)


def parse_turn(agent, result, requested_id):
    records = list(events(result.stdout))
    if event_failure(records):
        if any(isinstance(err := e.get("error"), dict) and (
                str(err.get("type", "")).startswith("provider.") or err.get("message"))
               for e in records if e.get("type") in {"error", "session.error", "turn.failed"}):
            raise ProviderError
        raise TransportError
    if result.returncode:
        raise TransportError
    if agent == "codex":
        ids = {e["thread_id"] for e in records if e.get("type") == "thread.started" and e.get("thread_id")}
        if any("item" in e and (e["item"] or {}).get("type") not in {"agent_message", "reasoning"} for e in records):
            raise ToolViolation
        # Codex emits thread.started on initial execution. Resume may omit it.
        ids = ids or ({requested_id} if requested_id else set())
        messages = [e["item"].get("text", "") for e in records if e.get("type") == "item.completed"
                    and e.get("item", {}).get("type") == "agent_message"]
        text = messages[-1] if messages else ""
        complete = any(e.get("type") == "turn.completed" for e in records)
    else:
        ids = {e["sessionID"] for e in records if e.get("sessionID")}
        if any(e.get("type") == "tool_use" for e in records):
            raise ToolViolation
        messages = {}
        for e in records:
            if e.get("type") == "text":
                part = e.get("part", {})
                group = messages.setdefault(part.get("messageID"), {})
                group[part.get("id", part.get("partID", len(group)))] = part.get("text", "")
        text = "".join(list(messages.values())[-1].values()) if messages else ""
        # OpenCode 2.0.20 can exit normally after the final text event
        # without emitting step_finish, even for a complete streamed reply.
        complete = bool(messages) or any(e.get("type") == "step_finish" for e in records)
    if len(ids) != 1 or (requested_id and ids != {requested_id}) or not complete:
        raise TransportError
    return next(iter(ids)), text


def response(request, text):
    if request["type"] == "room":
        return {"move": text.strip()}
    try:
        answer = json_answer(text)
        if isinstance(answer, dict):
            if request.get("observation", {}).get("scenario") == "memory":
                # Memory asks the model for a bare action. Its outer envelope
                # belongs to the evaluator protocol, not the model response.
                if isinstance(answer.get("type"), str):
                    return {"action": answer}
            elif set(answer) == {"action"}:
                return answer
    except (ValueError, RecursionError):
        pass
    return {"action": {"type": "invalid"}}


def turn_prompt(request):
    instructions = request.get("instructions", "")
    if request["type"] == "room":
        return (f"{instructions}\nХод {request['turn']}. {request['text']}\n"
                "Ответь одним направлением: вперёд, вправо, назад или влево.")
    if request["type"] != "observation":
        raise TransportError
    return f"{instructions}\nObservation: " + json.dumps(request["observation"], ensure_ascii=False)


class Native:
    def __init__(self, agent, model, command, timeout):
        self.agent, self.model, self.command, self.timeout = agent, model, command, timeout
        self.session = None
        self.server = None
        self.address = None
        self.env = None

    def start_server(self):
        # The listener is private to this worker container. It is never published.
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        # A native OpenCode 2 server authenticates even on loopback. Set an
        # episode-only password and pass it to both the server and clients.
        self.env = {**os.environ, "OPENCODE_PASSWORD": secrets.token_urlsafe(32)}
        self.server = subprocess.Popen([*self.command, "serve", "--hostname", "127.0.0.1", "--port", str(port)],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, env=self.env)
        self.address = f"http://127.0.0.1:{port}"
        deadline = monotonic() + min(10, self.timeout)
        while monotonic() < deadline:
            if self.server.poll() is not None:
                break
            try:
                with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                    return
            except OSError:
                sleep(0.05)
        raise TransportError

    def close(self):
        if self.server is not None and self.server.poll() is None:
            self.server.terminate()
            try:
                self.server.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.server.kill()
                self.server.wait()

    def turn(self, prompt, *, ephemeral=False):
        if self.agent == "codex":
            flags = ["--json", "--skip-git-repo-check", "--ignore-rules", *codex_config(self.model, tool_free=True)]
            if self.session:
                cmd = [*self.command, "exec", "resume", *flags, self.session, "-"]
            else:
                cmd = [*self.command, "exec", *flags, "--sandbox", "read-only"]
                if ephemeral:
                    cmd += ["--ephemeral"]
                cmd += ["-"]
        else:
            flags = ["--standalone"] if ephemeral else ["--server", self.address]
            cmd = [*self.command, "run", *flags, "--format", "json", "--model", self.model,
                   "--agent", "svg-source"]
            if self.session:
                cmd += ["--session", self.session]
            elif not ephemeral:
                cmd += ["--title", "Spatial Benchmark Episode"]
        result = subprocess.run(cmd, input=prompt.encode(), check=False, capture_output=True,
                                timeout=self.timeout, env=self.env)
        self.session, text = parse_turn(self.agent, result, self.session)
        return text


def main():
    mode, agent, model, seconds, *command = sys.argv[1:]
    transport = Native(agent, model, command, float(seconds))
    try:
        if mode == "source":
            print(transport.turn(sys.stdin.read(), ephemeral=True))
        else:
            if agent == "opencode":
                transport.start_server()
            for line in sys.stdin:
                req = json.loads(line, object_pairs_hook=strict_pairs)
                if req.get("type") == "end":
                    break
                text = transport.turn(turn_prompt(req))
                print(json.dumps(response(req, text), ensure_ascii=False), flush=True)
    except ProviderError:
        if mode == "session":
            print('{"type":"gateway_error","exhausted":true,"attempts":1}', flush=True)
        return 75
    except (TransportError, subprocess.TimeoutExpired, OSError, ValueError):
        print("Native CLI did not complete a valid tool-free turn.", file=sys.stderr)
        return 70
    finally:
        transport.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
