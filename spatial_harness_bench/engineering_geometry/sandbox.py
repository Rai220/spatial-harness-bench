"""Docker transport for an evaluator-supplied harness image (no host mounts)."""

from dataclasses import dataclass, replace
from io import BytesIO
import json
import os
import re
from selectors import DefaultSelector, EVENT_READ
import subprocess
import tarfile
from tempfile import TemporaryFile
from time import monotonic
from uuid import uuid4

from .svg import MAX_SVG_BYTES
from .verifier import MAX_STATUS_BYTES


SUPERVISOR_TIMEOUT = 20
LOG_LIMIT = 64 * 1024


@dataclass(frozen=True)
class SandboxResult:
    output_svg: bytes | None = None
    status_json: bytes | None = None
    stdout: bytes = b""
    stderr: bytes = b""
    technical_error: str | None = None
    agent_error: str | None = None
    returncode: int | None = None
    logs_truncated: bool = False


@dataclass(frozen=True)
class _DockerResult:
    returncode: int | None
    stdout: bytes
    stderr: bytes
    truncated: bool


def _docker(*args: str, stdin: bytes = b"", timeout: float = SUPERVISOR_TIMEOUT,
            limit: int = LOG_LIMIT) -> _DockerResult:
    # Drain both pipes without storing unlimited output. A file for finite stdin
    # avoids deadlock when the subprocess writes before reading the prompt.
    deadline = monotonic() + timeout
    streams = [bytearray(), bytearray()]
    truncated = False
    with TemporaryFile() as payload:
        payload.write(stdin)
        payload.seek(0)
        with subprocess.Popen(["docker", *args], stdin=payload, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE) as process, DefaultSelector() as selector:
            try:
                assert process.stdout is not None and process.stderr is not None
                selector.register(process.stdout, EVENT_READ, streams[0])
                selector.register(process.stderr, EVENT_READ, streams[1])
                while selector.get_map():
                    left = deadline - monotonic()
                    if left <= 0:
                        raise subprocess.TimeoutExpired(args, timeout)
                    for key, _ in selector.select(left):
                        chunk = os.read(key.fd, 64 * 1024)
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        remaining = limit - len(key.data)
                        key.data.extend(chunk[:remaining])
                        truncated |= len(chunk) > remaining
                returncode = process.wait(timeout=max(0, deadline - monotonic()))
            except subprocess.TimeoutExpired:
                process.kill()
                returncode = None
            except BaseException:
                process.kill()
                raise
    return _DockerResult(returncode, bytes(streams[0]), bytes(streams[1]), truncated)


def _inputs(svg: bytes, prompt: bytes) -> bytes:
    archive = BytesIO()
    with tarfile.open(fileobj=archive, mode="w") as bundle:
        folder = tarfile.TarInfo("task")
        folder.type, folder.mode = tarfile.DIRTYPE, 0o1777
        bundle.addfile(folder)
        for name, data in (("input.svg", svg), ("prompt.txt", prompt)):
            info = tarfile.TarInfo(f"task/{name}")
            info.mode, info.size = 0o444, len(data)
            bundle.addfile(info, BytesIO(data))
    return archive.getvalue()


class _TransportError(RuntimeError):
    pass


class _OutputError(ValueError):
    pass


def _required(result: _DockerResult, stage: str) -> bytes:
    if result.returncode is None:
        raise _TransportError(f"{stage}: Docker operation timed out")
    if result.returncode != 0:
        detail = result.stderr.decode("utf-8", errors="replace").strip()[:1500]
        raise _TransportError(f"{stage}: {detail or 'Docker operation failed'}")
    return result.stdout


def _artifact(container: str, name: str, max_bytes: int) -> bytes | None:
    result = _docker("cp", f"{container}:/task/{name}", "-", limit=max_bytes + 16 * 1024)
    if result.returncode and f"Could not find the file /task/{name} in container ".encode() in result.stderr:
        return None
    archive = _required(result, f"read {name}")
    if result.truncated:
        raise _OutputError(f"{name} exceeds the artifact limit")
    # Never extract untrusted archives or dereference links on the evaluator host.
    try:
        with tarfile.open(fileobj=BytesIO(archive), mode="r:") as bundle:
            entry = bundle.next()
            if (entry is None or entry.name != name or entry.type not in (tarfile.REGTYPE, tarfile.AREGTYPE)
                    or entry.sparse is not None or not 0 <= entry.size <= max_bytes):
                raise _OutputError(f"{name} must be one bounded regular file")
            stream = bundle.extractfile(entry)
            if stream is None:
                raise _OutputError(f"{name} must be a regular file")
            data = stream.read(max_bytes + 1)
            if len(data) != entry.size or bundle.next() is not None:
                raise _OutputError(f"invalid archive for {name}")
            return data
    except (tarfile.TarError, OSError, ValueError) as exc:
        if isinstance(exc, _OutputError):
            raise
        raise _OutputError(f"invalid archive for {name}") from exc


def execute(image: str, command: list[str], input_svg: bytes, prompt: str, *,
            timeout: float, network: str = "none") -> SandboxResult:
    """Run one fresh job. An empty result artifact is distinct from a missing one."""
    if not image or image.startswith("-") or not command or not command[0]:
        raise ValueError("a Docker image and harness command are required")
    if not 0 < timeout < float("inf"):
        raise ValueError("timeout must be positive and finite")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", network) or network.startswith("-") or network == "host":
        raise ValueError("use 'none' or a dedicated Docker bridge network")
    container = "geometry-" + uuid4().hex
    prompt_bytes = prompt.encode("utf-8")
    result = SandboxResult()
    create_attempted = False
    try:
        if network != "none":
            driver = _required(_docker("network", "inspect", "--format", "{{.Driver}}", network),
                               "inspect network").strip()
            if driver != b"bridge":
                raise _TransportError("only Docker bridge networking is allowed for harnesses")
        create_attempted = True
        _required(_docker(
            "create", "--pull=never", "--name", container, "--workdir", "/task",
            "--interactive", "--user", "65532:65532", "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges", "--network", network,
            "--memory", "1g", "--memory-swap", "1g", "--cpus", "2", "--pids-limit", "256",
            "--ulimit", "fsize=67108864:67108864", "--log-driver", "none",
            "--env", "HOME=/tmp", "--entrypoint", command[0], image, *command[1:],
        ), "create container")
        _required(_docker("cp", "-", f"{container}:/", stdin=_inputs(input_svg, prompt_bytes)), "stage inputs")
        attached = _docker("start", "-ai", container, stdin=prompt_bytes, timeout=timeout)
        result = replace(result, stdout=attached.stdout, stderr=attached.stderr, logs_truncated=attached.truncated)
        if attached.returncode is None:
            raise _TransportError("harness exceeded its deadline")
        state = json.loads(_required(_docker("inspect", "--format", "{{json .State}}", container), "inspect container"))
        if state["Status"] != "exited" or state["Error"]:
            raise _TransportError(f"container did not run normally: {state['Error'] or state['Status']}")
        result = replace(result, returncode=state["ExitCode"])
        if state["OOMKilled"]:
            raise _TransportError("harness ran out of sandbox memory")
        if result.returncode != 0 or attached.returncode != 0:
            raise _TransportError(f"harness process did not finish normally (exit {result.returncode})")
        result = replace(result, output_svg=_artifact(container, "output.svg", MAX_SVG_BYTES))
        result = replace(result, status_json=_artifact(container, "status.json", MAX_STATUS_BYTES))
    except _OutputError as exc:
        result = replace(result, agent_error=str(exc))
    except (OSError, _TransportError, json.JSONDecodeError, KeyError) as exc:
        result = replace(result, technical_error=str(exc))
    finally:
        if create_attempted:
            try:
                removed = _docker("rm", "-fv", container)
                if b"No such container:" not in removed.stderr:
                    _required(removed, "remove container")
            except (OSError, _TransportError) as exc:
                prior = f"{result.technical_error}; " if result.technical_error else ""
                result = replace(result, technical_error=f"{prior}{exc}")
    return result
