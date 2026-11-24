"""Export a public overall score from a complete, verified private run report."""

import argparse
from fractions import Fraction
import json
from pathlib import Path
from statistics import mean

SCORING_VERSION = "spatial-macro-v4"


def fraction(result, maximum, statuses=("complete",)):
    score = result["score"]
    if result.get("status", "complete") not in statuses or result["max_score"] != maximum:
        raise ValueError("Incomplete result or unexpected maximum")
    if type(score) is not int or not 0 <= score <= maximum:
        raise ValueError(f"Invalid score {score!r}")
    return Fraction(score, maximum)


def overall_score(scores, additional_scores=None):
    def svg(key, maximum):
        return fraction(scores[key]["code"], maximum)

    spatial_tasks = [
        mean(fraction(scores[key], 10) for key in ("01-ascii", "01-ascii-free")),
        fraction(scores["02-maze"], 10),
        mean(svg(f"03-{level}", 10) for level in ("basic", "hard", "expert", "master")),
        mean([svg("04-fixed", 10), svg("04-generated", 24)]),
    ]
    spatial_weight = 99
    points = fraction(scores["05-arcade"], 1)
    if additional_scores is not None:
        for name, maximum in (("building", 5), ("route", 4), ("memory", 1), ("expedition", 1)):
            statuses = ("complete", "completed", "turn_limit", "death", "limit") if name == "expedition" else ("complete",)
            points += maximum * fraction(additional_scores[name], maximum, statuses)
            spatial_weight -= maximum
    points += spatial_weight * mean(spatial_tasks)
    hundredths = int(points * 100 + Fraction(1, 2))
    return hundredths / 100


def public_summary(report):
    if not isinstance(report["additional_scores"], dict):
        raise ValueError("Missing additional task scores")
    fields = ("benchmark", "date", "model", "display_model", "harness", "harness_version",
              "provider_label", "benchmark_source_commit", "benchmark_package_sha256",
              "drawing_seed", "maze_suite_seed")
    public = {field: report[field] for field in fields}
    for optional in ("run_profile", "maze_source_commit", "additional_source_commit", "answer_gallery"):
        if optional in report:
            public[optional] = report[optional]
    public["datasets"] = {
        key: {field: report["datasets"][key][field] for field in ("version", "dataset_sha256")}
        for key in ("fixed", "generated", "additional")
    }
    public.update(scoring_version=SCORING_VERSION,
                  overall_score=overall_score(report["scores"], report["additional_scores"]),
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
