"""Render the same overall ranking for GitHub Pages and the repository."""

from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO = "https://github.com/Rai220/spatial-harness-bench"
SITE = "https://krestnikov.org/spatial-harness-bench/"
METHOD = "Первые четыре задачи дают до 24,75 балла каждая. За аркадную задачу начисляется 1 балл при успехе и 0 при ошибке."


def ranking(paths):
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    if not runs:
        raise ValueError("В docs/results нет опубликованных прогонов.")
    for field in ("scoring_version", "benchmark_package_sha256", "datasets", "drawing_seed",
                  "maze_suite_seed", "max_score"):
        if any(run[field] != runs[0][field] for run in runs):
            raise ValueError(f"Несопоставимые условия замера: {field}")
    if runs[0]["max_score"] != 100 or runs[0]["scoring_version"] != "spatial-macro-v2":
        raise ValueError("Неизвестная шкала лидерборда.")
    runs.sort(key=lambda r: (-r["overall_score"], r["display_model"].casefold(), r["date"]))
    rows = []
    for index, run in enumerate(runs, 1):
        score = run["overall_score"]
        if type(score) not in (int, float) or not 0 <= score <= 100:
            raise ValueError(f"Недопустимый балл: {score!r}")
        if index == 1 or score != runs[index - 2]["overall_score"]:
            place = index
        rows.append((place, run))
    return rows


def main():
    rows = ranking(sorted((DOCS / "results").glob("*.json")))
    platforms = sorted({f'{run["harness"]} {run["harness_version"]} · {run["provider_label"]}'
                        for _, run in rows})
    shared = platforms[0] if len(platforms) == 1 else ""
    html_rows = []
    markdown = ["# Spatial Harness Bench · Лидерборд", "", f"[GitHub Pages]({SITE})", "",
                METHOD, ""]
    if shared:
        markdown.extend([shared, ""])
    markdown.extend(["| Место | Модель | Балл из 100 |", "| ---: | --- | ---: |"])
    for place, run in rows:
        label = run["display_model"]
        if not shared:
            label += f' · {run["harness"]} {run["harness_version"]} · {run["provider_label"]}'
        value = run["overall_score"]
        score = f"{value:.2f}".replace(".", ",")
        html_rows.append(f'<tr><td>{place}</td><th scope="row">{escape(label)}</th>'
                         f'<td class="numeric"><span class="score">{score}'
                         f'<meter aria-hidden="true" min="0" max="100" value="{value}"></meter>'
                         f'</span></td></tr>')
        label = label.replace("|", r"\|")
        markdown.append(f"| {place} | {label} | {score} |")

    subtitle = f'<p>{escape(shared)}</p>' if shared else ""
    page = f'''<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Spatial Harness Bench · Лидерборд</title>
  <meta name="description" content="Сравнение моделей по общей оценке Spatial Harness Bench.">
  <link rel="stylesheet" href="leaderboard.css">
</head>
<body>
<header><a class="brand" href="./">Spatial Harness Bench<span>.</span></a>
<nav aria-label="Навигация"><a href="{REPO}">GitHub</a></nav></header>
<main><div class="intro"><h1>Лидерборд</h1>{subtitle}</div>
<div class="table-wrap"><table aria-label="Общий рейтинг моделей"><thead><tr>
<th scope="col">Место</th><th scope="col">Модель</th><th class="numeric" scope="col">Балл из 100</th>
</tr></thead><tbody>{''.join(html_rows)}</tbody></table></div>
<p>{METHOD} <a href="{REPO}#общая-оценка">Методика</a></p></main>
<footer><span>Spatial Harness Bench</span><a href="{REPO}/blob/main/LEADERBOARD.md">Таблица в репозитории</a></footer>
</body></html>
'''
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (ROOT / "LEADERBOARD.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
