"""Isolate native agents from legacy evaluators and exchange only public inputs."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from selectors import EVENT_READ, DefaultSelector
from stat import S_ISREG
from tempfile import TemporaryDirectory, TemporaryFile
from time import monotonic
from uuid import uuid4

# No verifier code or checkout is imported by the process inside Docker.
if __package__:
    from . import suite_native as native
else:
    import suite_native as native


FILE_OUTPUTS = {
    "ascii_recall": ("desc.txt", *(f"{i}.txt" for i in range(10))),
    "arcade_clips": ("clips.rix", "query.py"),
    "building": ("answer.json",),
    "adnd_route": ("answer.json",),
}
CAP = 64 * 1024 * 1024


def container_args(network, name, run_id=None):
    flags = ["docker", "run", "--rm", "--pull=never", "--name", name, "-i", "--network", network,
             "--user", "65532:65532", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
             "--memory", "1g", "--memory-swap", "1g", "--cpus", "2", "--pids-limit", "256",
             "--ulimit", f"fsize={CAP}:{CAP}", "--log-driver", "none", "--env", "HOME=/tmp",
             "--workdir", "/tmp"]
    if run_id:
        flags.extend(("--label", f"spatial-suite={run_id}"))
    return flags


def file_command(agent, model, command):
    if agent == "codex":
        return [*command, "exec", "--ephemeral", "--json", "--skip-git-repo-check", "--ignore-rules",
                "--sandbox", "danger-full-access", "-c", 'approval_policy="never"',
                *native.codex_config(model), "-"]
    if agent == "opencode":
        return [*command, "run", "--standalone", "--auto", "--format", "json", "--model", model]
    return command


def safe_artifact(folder, name):
    # The container has exited. Do not treat links, devices, pipes or oversized
    # files as submissions, and never copy back unrequested paths.
    try:
        path = folder / name
        attrs = path.lstat()
        if S_ISREG(attrs.st_mode) and attrs.st_nlink == 1 and attrs.st_size <= CAP:
            with path.open("rb") as fh:
                return fh.read(CAP + 1)
    except OSError:
        pass
    return None


def docker_command(image, network, run_id, command, *, stage=None, helper=False):
    name = "spatial-" + uuid4().hex
    flags = container_args(network, name, run_id)
    if stage is not None:
        flags += ["--mount", f"type=bind,src={stage},dst=/task", "--workdir", "/task"]
    if helper:
        source = Path(native.__file__).resolve()
        flags += ["--mount", f"type=bind,src={source},dst=/suite_native.py,readonly"]
        command = ["python3", "/suite_native.py", *command]
    return name, [*flags, "--entrypoint", command[0], image, *command[1:]]


def bounded_run(argv, prompt, timeout):
    """Drain both output pipes without buffering an unlimited agent transcript."""
    deadline = monotonic() + timeout
    with TemporaryFile() as stdin, DefaultSelector() as streams:
        stdin.write(prompt)
        stdin.seek(0)
        with subprocess.Popen(argv, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE) as child:
            output, diagnostics = bytearray(), bytearray()
            streams.register(child.stdout, EVENT_READ, output)
            streams.register(child.stderr, EVENT_READ, diagnostics)
            try:
                while streams.get_map():
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, timeout)
                    for key, _ in streams.select(remaining):
                        chunk = os.read(key.fd, 65536)
                        if not chunk:
                            streams.unregister(key.fileobj)
                        else:
                            key.data.extend(chunk)
                            if len(key.data) > (8 * 1024 * 1024 if key.data is output else 128 * 1024):
                                raise OSError("agent logs exceeded transport limit")
                return subprocess.CompletedProcess(argv, child.wait(timeout=max(0, deadline - monotonic())),
                                                   bytes(output), bytes(diagnostics))
            except BaseException:
                child.kill()
                raise


def invoke(image, network, run_id, command, *, stage=None, helper=False, prompt=None, timeout=None):
    name, argv = docker_command(image, network, run_id, command, stage=stage, helper=helper)
    try:
        if prompt is None:
            return subprocess.CompletedProcess(argv, subprocess.run(argv, check=False).returncode, b"", b"")
        result = bounded_run(argv, prompt, timeout)
        if result.stderr:
            sys.stderr.buffer.write(result.stderr)
        return result
    finally:
        # Handling a timeout in docker(1) does not necessarily stop the daemon job.
        removed = subprocess.run(["docker", "rm", "-fv", name], check=False, capture_output=True, timeout=15)
        if removed.returncode and b"No such container" not in removed.stderr:
            raise OSError("cannot clean up agent container")


def source_request(folder, prompt):
    stage, difficulty = (os.environ[k] for k in ("SVG_RECALL_STAGE", "SVG_RECALL_DIFFICULTY"))
    count = int(os.environ["SVG_RECALL_COUNT"])
    if stage == "draw":
        return (stage, count,
                prompt + '\nWrite no files. Return only JSON {"drawings":[{"name":"…","svg":"…"}]}, '
                f'one entry for each of {count} drawings in order. No tools.')
    if stage != "code":
        raise ValueError("invalid source stage")
    names = ["board"] if difficulty in {"hard", "expert", "master"} else [str(i) for i in range(count)]
    source = "\n".join(f"\n--- {name}.svg ---\n" + (folder / f"{name}.svg").read_text(encoding="utf-8")
                       for name in names)
    return (stage, count, prompt +
            f'\nReturn only JSON {{"answers":[...]}}, exactly {count} strings in question/file order. '
            'Use an empty string to skip. No tools.\n' + source)


def source_artifacts(folder, stage, count, data):
    try:
        payload = native.json_answer(data)
        key = "drawings" if stage == "draw" else "answers"
        rows = payload[key]
        if set(payload) != {key} or not isinstance(rows, list) or len(rows) != count:
            raise ValueError
        if stage == "draw":
            if any(not isinstance(x, dict) or set(x) != {"svg", "name"} or
                   not all(isinstance(v, str) for v in x.values()) for x in rows):
                raise ValueError
            names = [row["name"] for row in rows]
        else:
            names = rows
        if any(not isinstance(name, str) or "\n" in name or "\r" in name for name in names):
            raise ValueError
    except (TypeError, KeyError, ValueError, RecursionError):
        names, rows = [], []
    if stage == "draw":
        # Provide absent/invalid drawings as empty SVG data for the stock scorer.
        for i in range(count):
            (folder / f"{i}.svg").write_text(rows[i]["svg"] if i < len(rows) else "", encoding="utf-8")
    (folder / "desc.txt").write_text("\n".join(names) + "\n", encoding="utf-8")


def main():
    image, network, seconds, run_id, family, mode, agent, model, *command = sys.argv[1:]
    timeout = float(seconds)
    try:
        if mode == "session":
            return invoke(image, network, run_id,
                          ["session", agent, model, seconds, *command], helper=True).returncode
        if mode == "stream":
            # The stock Pi bridge owns the pipes and enforces its protocol.
            return invoke(image, network, run_id, command).returncode
        folder = Path.cwd()
        if mode == "source":
            stage, count, prompt = source_request(folder, sys.stdin.read())
            result = invoke(image, network, run_id, ["source", agent, model, seconds, *command],
                            helper=True, prompt=prompt.encode(), timeout=timeout + 15)
            if result.returncode == 0:
                source_artifacts(folder, stage, count, result.stdout.decode("utf-8"))
            return result.returncode
        prompt = sys.stdin.buffer.read()
        with TemporaryDirectory(prefix="spatial-public-") as tmp:
            stage = Path(tmp) / "work"
            shutil.copytree(folder, stage, symlinks=True)
            stage.chmod(0o1777)
            for path in stage.rglob("*"):
                if path.is_symlink():
                    raise ValueError("public input is a symlink")
                if path.is_dir():
                    path.chmod(0o755)
                else:
                    path.chmod(0o644)
            result = invoke(image, network, run_id, file_command(agent, model, command),
                          stage=stage.resolve(), prompt=prompt, timeout=timeout)
            if agent in {"codex", "opencode"} and native.event_failure(native.events(result.stdout)):
                return 75
            if result.returncode == 0:
                for filename in FILE_OUTPUTS[family]:
                    data = safe_artifact(stage, filename)
                    if data is not None:
                        (folder / filename).write_bytes(data)
            return result.returncode
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"Agent transport failed: {type(exc).__name__}", file=sys.stderr, flush=True)
        return 70


if __name__ == "__main__":
    raise SystemExit(main())
