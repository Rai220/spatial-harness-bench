"""Retry an ungraded inference failure with the original inputs."""
from collections.abc import Callable
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory, TemporaryFile
from time import sleep

# Adapters can signal an inference transport failure unambiguously with EX_TEMPFAIL.
TEMPFAIL = 75
HTTP_ERROR = re.compile(
    r"\b(?:api\s*error|http(?:\s*error|/\d(?:\.\d)?)?|status(?:[ _]code)?|error code|unexpected status)"
    r"\s*[:=]?\s*[45]\d\d\b", re.I)
CONNECTION_ERROR = re.compile(
    r"\b(?:ECONNRESET|ETIMEDOUT|connection (?:reset|refused)|read timed out|"
    r"remote end closed connection|stream disconnected before completion)\b", re.I)


class GatewayError(RuntimeError):
    attempts = 1
    exhausted = False

    def report(self, stage):
        return {"status": "error", "reason": "gateway", "stage": stage,
                "score": None, "attempts": self.attempts}


def retry(operation: Callable, *, reset=None):
    retries = int(os.environ.get("SPATIAL_GATEWAY_RETRIES", "3"))
    if retries < 0:
        raise ValueError("SPATIAL_GATEWAY_RETRIES must be nonnegative")
    for attempt in range(retries + 1):
        try:
            return operation()
        except GatewayError as exc:
            if exc.exhausted:
                raise
            exc.attempts = attempt + 1
            if attempt == retries:
                exc.exhausted = True
                raise
            print(f"Gateway failure. Retrying ({attempt + 1}/{retries}).", file=sys.stderr)
            sleep(2 ** min(attempt, 4))
            if reset:
                reset()


def tail(stream):
    stream.seek(max(0, stream.seek(0, os.SEEK_END) - 65536))
    return stream.read().decode(errors="replace").strip()


def relay(stream, target):
    stream.seek(0)
    if hasattr(target, "buffer"):
        shutil.copyfileobj(stream, target.buffer)
        target.buffer.flush()
    else:
        for line in stream:
            target.write(line.decode(errors="replace"))


def gateway_failure(code, stderr, stdout=""):
    if code == TEMPFAIL:
        return True
    if code and any(HTTP_ERROR.search(log) or CONNECTION_ERROR.search(log)
                    for log in (stderr, stdout)):
        return True
    # Some CLIs return zero and put the API failure in the final JSON event.
    lines = (stdout.strip() or stderr.strip()).splitlines()
    if lines:
        if lines[-1].lower().startswith("api error:") and HTTP_ERROR.match(lines[-1]):
            return True
        try:
            event = json.loads(lines[-1])
        except ValueError:
            return False
        if isinstance(event, dict):
            if event.get("type") == "result" and event.get("is_error"):
                message = str(event.get("result", ""))
            elif event.get("type") == "turn.failed":
                message = str(event.get("error", ""))
            else:
                return False
            return bool(HTTP_ERROR.search(message) or CONNECTION_ERROR.search(message))
    return False


def run_stage(command, folder, prompt, *, env=None, timeout=None, show_stdout=False):
    """Run a file-based stage. Restore pristine inputs before each new session."""
    with TemporaryDirectory(prefix="stage-inputs-") as tmp:
        snapshot = Path(tmp) / "inputs"
        shutil.copytree(folder, snapshot, symlinks=True)

        def restore():
            shutil.rmtree(folder)
            shutil.copytree(snapshot, folder, symlinks=True)

        def attempt():
            with TemporaryFile() as stdout, TemporaryFile() as stderr, subprocess.Popen(
                command, cwd=folder, env=env, stdin=subprocess.PIPE, stdout=stdout,
                stderr=stderr, start_new_session=True,
            ) as agent:
                try:
                    agent.communicate(prompt.encode(), timeout=timeout)
                except BaseException:
                    if agent.poll() is None:
                        os.killpg(agent.pid, signal.SIGKILL)
                    agent.wait()
                    raise
                diagnostic = tail(stderr)
                if gateway_failure(agent.returncode, diagnostic, tail(stdout)):
                    raise GatewayError("Inference request rejected or interrupted")
                relay(stderr, sys.stderr)
                if show_stdout:
                    relay(stdout, sys.stdout)
                if agent.returncode:
                    raise subprocess.CalledProcessError(agent.returncode, command)
        retry(attempt, reset=restore)


def save_error(output, task, maximum, stage, exc, **metadata):
    result = {"task": task, "max_score": maximum, **metadata, **exc.report(stage)}
    output.mkdir(parents=True, exist_ok=False)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
