"""Render the same overall ranking for GitHub Pages and the repository."""

from html import escape
import json
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO = "https://github.com/Rai220/spatial-harness-bench"
SITE = "https://krestnikov.org/spatial-harness-bench/"
METHOD = "Первые четыре задачи дают до 24,75 балла каждая. В SVG-задачах оцениваются ответы по исходному коду. За аркадную задачу начисляется 1 балл при успехе и 0 при ошибке."
EXAMPLE_INTRO = "В пяти задачах модель работает с рисунками, комнатами лабиринта и записью игры. Попробуйте ответить на вопросы по иллюстрациям."
EXAMPLES = (
    ("ASCII-рисунки", "examples/ascii.svg", "Два рисунка из символов, сделанные GPT-6.2 Astra.",
     "Модель рисует десять предметов, а в новой сессии узнаёт их по силуэтам. Какие два предмета нарисовала Astra?",
     "Парусник и кактус."),
    ("Лабиринт", "examples/maze.svg", "Карта 5×5 с флажком и увеличенная стартовая комната. Из неё открыты двери вперёд и вправо.",
     "Модель видит только описание дверей в текущей комнате. Какой первый ход от зелёной точки ведёт к флажку кратчайшим путём?",
     "Вперёд. Весь путь займёт пять ходов — вперёд, вперёд, влево, вперёд, влево."),
    ("Повороты на поле SVG", "examples/rotation.svg", "Квадрат из четырёх клеток до и после поворота по часовой стрелке.",
     "Модель следит за своими SVG-рисунками при сдвигах и поворотах. Где окажется парусник после поворота квадрата на 90° по часовой стрелке?",
     "В верхней правой клетке."),
    ("Вопросы по SVG-сценам", "answers/opus-claude-code-20261121/shelf.svg", "Две кухонные полки с посудой и кофемолкой.",
     "Модель читает код готовой SVG-сцены. Что стоит под чайником и слева от бирюзовой кружки?",
     "Ручная кофемолка."),
    ("Монтажный индекс", "examples/replay.svg", "Общий счёт игроков за два клипа по 64 кадра. На границах клипов он равен 0, 6 и 15.",
     "Модель строит индекс игровых клипов по 64 кадра. В каком из двух отрезков сильнее вырос общий счёт игроков?",
     "Во втором, с кадра 64 по 127. Игроки набрали 9 очков, в первом отрезке — 6."),
)


def examples():
    cards = []
    markdown = ["## Примеры задач", "", EXAMPLE_INTRO, "",
                "| Иллюстрация | Задача |", "| --- | --- |"]
    for title, image, alt, question, answer in EXAMPLES:
        cards.append(f'''<article class="example">
<img class="example-art" src="{escape(image)}" alt="{escape(alt)}" loading="lazy">
<h3>{escape(title)}</h3><p>{escape(question)}</p>
<details><summary>Показать ответ</summary><p>{escape(answer)}</p></details>
</article>''')
        markdown.append(f'| <img src="docs/{image}" width="220" alt="{escape(alt)}"> '
                        f'| **{title}**<br>{question} |')
    html = f'''<section class="examples" id="examples" aria-labelledby="examples-title">
<h2 id="examples-title">Примеры задач</h2><p class="examples-intro">{escape(EXAMPLE_INTRO)}</p>
<div class="example-grid">{''.join(cards)}</div></section>'''
    return html, markdown


def ranking(paths):
    runs = [dict(json.loads(path.read_text(encoding="utf-8")), result_id=path.stem) for path in paths]
    if not runs:
        raise ValueError("В docs/results нет опубликованных прогонов.")
    for field in ("scoring_version", "datasets", "drawing_seed",
                  "maze_suite_seed", "max_score"):
        if any(run[field] != runs[0][field] for run in runs):
            raise ValueError(f"Несопоставимые условия замера: {field}")
    if runs[0]["max_score"] != 100 or runs[0]["scoring_version"] != "spatial-macro-v3":
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


def table(rows):
    html_rows = []
    markdown = ["| Место | Модель | Харнес | Балл из 100 |",
                "| ---: | --- | --- | ---: |"]
    for place, run in rows:
        label = run["display_model"]
        harness = f'{run["harness"]} {run["harness_version"]} · {run["provider_label"]}'
        value = run["overall_score"]
        score = f"{value:.2f}".replace(".", ",")
        html_label = escape(label)
        label = label.replace("|", r"\|")
        answer_dir = Path("answers") / run["result_id"]
        if (DOCS / answer_dir / "index.html").is_file():
            href = quote(answer_dir.as_posix()) + "/"
            html_label += f' <a class="answer-link" href="{href}">Разбор ответов</a>'
            label += f" · [Разбор ответов]({SITE}{href})"
        html_rows.append(f'<tr><td>{place}</td><th scope="row">{html_label}</th>'
                         f'<td>{escape(harness)}</td><td class="numeric"><span class="score">{score}'
                         f'<meter aria-hidden="true" min="0" max="100" value="{value}"></meter>'
                         f'</span></td></tr>')
        markdown.append(f"| {place} | {label} | {harness} | {score} |")

    html = f'''<div class="table-wrap"><table aria-label="Рейтинг моделей"><thead><tr>
<th scope="col">Место</th><th scope="col">Модель</th><th scope="col">Харнес</th>
<th class="numeric" scope="col">Балл из 100</th>
</tr></thead><tbody>{''.join(html_rows)}</tbody></table></div>'''
    return html, markdown


def main():
    paths = sorted((DOCS / "results").glob("*.json"))
    html_table, table_lines = table(ranking(paths))
    html_examples, example_lines = examples()
    markdown = ["# Spatial Harness Bench · Лидерборд", "", f"[GitHub Pages]({SITE})", "", METHOD, "",
                *table_lines, "", "[Пять дополнительных задач по поэтажным планам](README.md#building) "
                "· Отдельный зачёт 0–5.", "", *example_lines]
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
<main><div class="intro"><h1>Лидерборд</h1></div>
{html_table}
<p>{METHOD} <a href="{REPO}#общая-оценка">Методика</a></p>
<p><a href="{REPO}#building">Пять дополнительных задач по поэтажным планам</a> · Отдельный зачёт 0–5.</p>
{html_examples}</main>
<footer><span>Spatial Harness Bench</span><a href="{REPO}/blob/main/LEADERBOARD.md">Таблица в репозитории</a></footer>
</body></html>
'''
    (DOCS / "index.html").write_text(page, encoding="utf-8")
    (ROOT / "LEADERBOARD.md").write_text("\n".join(markdown).rstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
