"""Aggregate repeated runs: mean score with a 95% interval per model, task and mode.

    python -m spatial_harness_bench.stats sol='runs/rep-*/sol' haiku='runs/rep-*/haiku'

Each argument is LABEL=GLOB or a bare GLOB (then the glob is the label). A glob may name
result.json/suite.json files or directories, which are searched recursively; per-level files
inside a maze suite are not counted twice. Two intervals are reported:

* runs  — Student's t over whole-run scores. Needs at least two runs. This is the honest
          interval for comparing models, since it includes seed and session variance.
* items — Wilson interval for the share of fully correct items, pooled over runs. It treats
          items as independent and is narrower; with one run it is the only estimate.
"""

import argparse
from collections import defaultdict
from dataclasses import dataclass, field
import glob
import json
import math
from pathlib import Path
import statistics
import sys

T95 = (12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160,
       2.145, 2.131, 2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056,
       2.052, 2.048, 2.045, 2.042)


def t95(df: int) -> float:
    return T95[df - 1] if df <= len(T95) else 1.96


def wilson(hits: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = hits / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass
class Group:
    scores: list[float] = field(default_factory=list)
    max_score: int = 0
    hits: int = 0
    items: int = 0
    errors: int = 0
    efficiency: list[float] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


def rows(data: dict):
    """Yield (task, mode, score, max_score, hits, items, extra) for each completed run in a report."""
    if "levels" in data and "suite" in data:
        if data.get("status") != "complete":
            yield data["suite"], "-", None, data["max_score"], 0, 0, {}
            return
        hits = sum(level["status"] == "solved" for level in data["levels"])
        yield (data["suite"], "-", data["score"], data["max_score"], hits, len(data["levels"]),
               {"efficiency": data.get("efficiency")})
        return
    if "modes" in data:
        for mode, result in data["modes"].items():
            if result.get("status") != "complete":
                yield data["task"], mode, None, 0, 0, 0, {}
                continue
            items = result.get("items", [])
            yield (result.get("task", data["task"]), mode, result["score"], result["max_score"],
                   sum(item["points"] == 1 for item in items), len(items), {})
        return
    if data.get("reason") == "gateway" and data.get("score") is None:
        yield data["task"], "-", None, data.get("max_score", 0), 0, 0, {}
        return
    if "score" in data and "items" in data:
        yield (data["task"], "-", data["score"], data["max_score"],
               sum(item["points"] == 1 for item in data["items"]), len(data["items"]), {})
    elif data.get("task") == "05-arcade-clips":
        yield data["task"], "-", data["score"], 1, data["score"], 1, {}
    elif data.get("task", "").startswith("02-maze") and "status" in data:
        solved = data["status"] == "solved"
        if data["status"] == "error":
            yield data["task"], "-", None, 1, 0, 0, {}
        else:
            yield data["task"], "-", int(solved), 1, int(solved), 1, {"efficiency": data.get("efficiency")}


def report_files(pattern: str) -> list[Path]:
    found: list[Path] = []
    for match in sorted(glob.glob(pattern, recursive=True)):
        path = Path(match)
        if path.is_file():
            found.append(path)
            continue
        suites = sorted(path.rglob("suite.json"))
        roots = [s.parent for s in suites]
        found.extend(suites)
        for result in sorted(path.rglob("result.json")):
            if not any(root in result.parents for root in roots):
                found.append(result)
    return found


def collect(specs: list[str]) -> dict[tuple[str, str, str], Group]:
    groups: dict[tuple[str, str, str], Group] = defaultdict(Group)
    for spec in specs:
        label, sep, pattern = spec.partition("=")
        if not sep:
            label, pattern = spec, spec
        files = report_files(pattern)
        if not files:
            raise FileNotFoundError(f"Нет отчётов по шаблону {pattern}")
        for path in files:
            data = json.loads(path.read_text(encoding="utf-8"))
            for task, mode, score, max_score, hits, items, extra in rows(data):
                group = groups[(label, task, mode)]
                group.files.append(str(path))
                if score is None:
                    group.errors += 1
                    continue
                group.scores.append(score)
                group.max_score = max(group.max_score, max_score)
                group.hits += hits
                group.items += items
                if extra.get("efficiency") is not None:
                    group.efficiency.append(extra["efficiency"])
    return groups


def summarize(key: tuple[str, str, str], group: Group) -> dict:
    label, task, mode = key
    n = len(group.scores)
    mean = statistics.fmean(group.scores) if n else None
    sd = statistics.stdev(group.scores) if n > 1 else None
    half = t95(n - 1) * sd / math.sqrt(n) if sd is not None else None
    lo, hi = wilson(group.hits, group.items)
    row = {"label": label, "task": task, "mode": mode, "runs": n, "errors": group.errors,
           "max_score": group.max_score, "mean": mean, "sd": sd,
           "runs_ci95": ([max(0.0, mean - half), min(float(group.max_score), mean + half)]
                         if half is not None else None),
           "item_accuracy": group.hits / group.items if group.items else None,
           "items": group.items, "items_ci95": [lo, hi] if group.items else None}
    if group.efficiency:
        row["efficiency"] = statistics.fmean(group.efficiency)
    return row


def table(summaries: list[dict]) -> str:
    def fmt(x, digits=2):
        return "—" if x is None else f"{x:.{digits}f}"

    lines = ["| Модель | Задача | Режим | Прогонов | Балл | 95% ДИ (прогоны) | Доля верных | 95% ДИ (пункты) | Эффективность |",
             "|---|---|---|---:|---:|---|---:|---|---:|"]
    for s in summaries:
        runs_ci = "—" if s["runs_ci95"] is None else f"{fmt(s['runs_ci95'][0])}…{fmt(s['runs_ci95'][1])}"
        items_ci = "—" if s["items_ci95"] is None else f"{fmt(s['items_ci95'][0])}…{fmt(s['items_ci95'][1])}"
        runs = f"{s['runs']}" + (f" (+{s['errors']} err)" if s["errors"] else "")
        score = "—" if s["mean"] is None else f"{fmt(s['mean'])}/{s['max_score']}"
        lines.append(f"| {s['label']} | {s['task']} | {s['mode']} | {runs} | {score} | {runs_ci} | "
                     f"{fmt(s['item_accuracy'])} | {items_ci} | {fmt(s.get('efficiency'), 3)} |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("specs", nargs="+", metavar="[LABEL=]GLOB")
    parser.add_argument("--json", action="store_true", help="print JSON instead of a Markdown table")
    args = parser.parse_args()
    try:
        groups = collect(args.specs)
    except (OSError, ValueError, KeyError) as exc:
        print(exc, file=sys.stderr)
        return 1
    summaries = [summarize(key, group) for key, group in sorted(groups.items())]
    print(json.dumps(summaries, ensure_ascii=False, indent=2) if args.json else table(summaries))
    return 0


if __name__ == "__main__":
    sys.exit(main())
