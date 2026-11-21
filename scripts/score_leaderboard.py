"""Export a public overall score from a complete, verified private run report."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
from statistics import mean

SCORING_VERSION = "spatial-macro-v2"


def fraction(result, maximum):
    score = result["score"]
    if result.get("status", "complete") != "complete" or result["max_score"] != maximum:
        raise ValueError("Incomplete result or unexpected maximum")
    if type(score) is not int or not 0 <= score <= maximum:
        raise ValueError(f"Invalid score {score!r}")
    return Fraction(score, maximum)


def overall_score(scores):
    def svg(key, maximum):
        return mean(fraction(scores[key][mode], maximum) for mode in ("code", "vision"))

    spatial_tasks = [
        mean(fraction(scores[key], 10) for key in ("01-ascii", "01-ascii-free")),
        fraction(scores["02-maze"], 10),
        mean(svg(f"03-{level}", 10) for level in ("basic", "hard", "expert", "master")),
        mean([svg("04-fixed", 10), svg("04-generated", 24)]),
    ]
    points = 99 * mean(spatial_tasks) + fraction(scores["05-arcade"], 1)
    hundredths = int(points * 100 + Fraction(1, 2))
    return hundredths / 100


def public_summary(report):
    fields = ("benchmark", "date", "model", "display_model", "harness", "harness_version",
              "provider_label", "benchmark_source_commit", "benchmark_package_sha256",
              "drawing_seed", "maze_suite_seed")
    public = {field: report[field] for field in fields}
    public["datasets"] = {
        key: {field: report["datasets"][key][field] for field in ("version", "dataset_sha256")}
        for key in ("fixed", "generated")
    }
    public.update(scoring_version=SCORING_VERSION, overall_score=overall_score(report["scores"]),
                  max_score=100)
    return public


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    data = public_summary(json.loads(args.report.read_text(encoding="utf-8")))
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
