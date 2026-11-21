"""Run each submitted query with just its source and the system Python runtime."""
import sysconfig
from pathlib import Path


def query_command(source: Path) -> list[str]:
    interpreter = "/usr/bin/python3.12"
    stdlib = "/usr/lib/python3.12"
    libraries = Path("/lib") / sysconfig.get_config_var("MULTIARCH")
    args = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session",
            "--ro-bind", interpreter, interpreter, "--ro-bind", stdlib, stdlib,
            "--ro-bind", str(libraries), str(libraries)]
    if Path("/lib64").is_dir():
        args += ["--ro-bind", "/lib64", "/lib64"]
    for name in ("dist-packages", "site-packages"):
        packages = Path(stdlib) / name
        if packages.is_dir():
            args += ["--tmpfs", str(packages)]
    return [*args, "--dev", "/dev", "--tmpfs", "/tmp",
            "--ro-bind", str(source), "/work/query.py", "--chdir", "/work",
            "--", interpreter, "-I", "-S", "/work/query.py"]
