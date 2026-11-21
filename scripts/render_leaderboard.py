"""Render the repository leaderboard and the GitHub Pages site from public run reports."""

from datetime import date
from html import escape
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
REPO = "https://github.com/Rai220/spatial-harness-bench"
SITE = "https://krestnikov.org/spatial-harness-bench/"
INTRO = ("Пять задач проверяют, как модель узнаёт свои рисунки, находит путь и рассуждает "
         "о сценах. У каждой задачи своя шкала.")
STANDARD = (
    ("01-ascii", "01 · ASCII-рисунки", "Назначенные предметы"),
    ("01-ascii-free", "01 · ASCII-рисунки", "Предметы выбирала модель"),
    ("02-maze", "02 · Лабиринты v2", "Карты, на которых найден выход"),
    ("05-arcade", "05 · Аркадный реплей", "Точный индекс и программа для запросов"),
)
SVG = (
    ("03-basic", "03 · Свои SVG", "Распознавание"),
    ("03-hard", "03 · Свои SVG", "Исходная клетка · hard"),
    ("03-expert", "03 · Свои SVG", "Исходная клетка · expert"),
    ("03-master", "03 · Свои SVG", "Исходная клетка · master"),
    ("04-fixed", "04 · Готовые сцены", "Публичные · v1"),
    ("04-generated", "04 · Готовые сцены", "Закрытые · генератор v1"),
)


def score_text(cell):
    return f"{cell['score']}/{cell['max_score']}"


def html_cell(cell):
    if isinstance(cell, dict):
        return (f'<span class="score">{score_text(cell)}<meter aria-hidden="true" '
                f'min="0" max="{int(cell["max_score"])}" value="{int(cell["score"])}"></meter></span>')
    return escape(str(cell))


def tables(columns, rows, numeric):
    """A cell is either literal text or a scored result with score and max_score."""
    def align(i):
        return ' class="numeric"' if i in numeric else ''

    headings = ''.join(f'<th scope="col"{align(i)}>{escape(name)}</th>'
                       for i, name in enumerate(columns))
    body = []
    markdown = [columns, ["---"] * len(columns)]
    for row in rows:
        cells = [f'<th scope="row">{html_cell(row[0])}</th>']
        cells.extend(f'<td{align(i)}>{html_cell(cell)}</td>' for i, cell in enumerate(row[1:], 1))
        body.append('<tr>' + ''.join(cells) + '</tr>')
        markdown.append([score_text(v) if isinstance(v, dict) else str(v).replace('|', r'\|')
                         for v in row])
    html = (f'<div class="table-wrap"><table><thead><tr>{headings}</tr></thead>'
            f'<tbody>{"".join(body)}</tbody></table></div>')
    return html, '\n'.join('| ' + ' | '.join(row) + ' |' for row in markdown)


def svg_table(scores):
    rows = [(task, variant, scores[key]["code"], scores[key]["vision"])
            for key, task, variant in SVG if key in scores]
    return tables(["Задача", "Режим", "Код", "PNG"], rows, {2, 3})


def render_run(path):
    data = json.loads(path.read_text(encoding="utf-8"))
    scores, levels = data["scores"], data["maze_levels"]
    title = f'{data["display_model"]} × {data["harness"]}'
    day = date.fromisoformat(data["date"]).strftime("%d.%m.%Y")
    caption = f'{data["provider"]} · {data["harness"]} {data["harness_version"]} · {day}'
    run_id = escape(path.stem, quote=True)
    json_path = escape(path.relative_to(DOCS).as_posix(), quote=True)
    html = [f'<article id="{run_id}"><div class="run"><strong>{escape(title)}</strong>'
            f'<span>{escape(caption)}</span></div>']
    md = [f'## {title}', f'{caption} · [Данные в JSON](docs/{json_path})']

    def section(name, table, detail="", folded=False):
        html_table, md_table = table
        explanation = f'<p>{escape(detail)}</p>' if detail else ''
        if folded:
            content = f'<details><summary>{escape(name)}</summary>{explanation}{html_table}</details>'
        else:
            content = f'<h2>{escape(name)}</h2>{explanation}{html_table}'
        html.append(f'<section class="measure">{content}</section>')
        md.extend(t for t in (f'### {name}', detail, md_table) if t)

    rows = [(task, variant, scores[key]) for key, task, variant in STANDARD if key in scores]
    section("Рисунки, лабиринты и реплей", tables(["Задача", "Что оценивается", "Балл"], rows, {2}))
    section("SVG по коду и по изображению", svg_table(scores),
            "В основном замере для каждого уровня и способа подачи одна сессия получала весь набор вопросов.")

    maze_rows = [(f'{l["size"]}×{l["size"]}', "Выход найден" if l["status"] == "solved" else "Исчерпан лимит",
                  f'{l["attempts"]} / {l["limit"]}', l["shortest"], l["note_writes"]) for l in levels]
    efficiency = f'{100 * scores["02-maze"]["mean_efficiency"]:.1f}'.replace(".", ",")
    writes = sum(l["note_writes"] for l in levels)
    errors = sum(l["format_errors"] for l in levels)
    detail = (f"Средняя эффективность — {efficiency}%. Для пройденной карты это длина кратчайшего пути, "
              f"делённая на число решений. За непройденную карту начисляется 0. "
              f"Записей в файл заметок — {writes}. Ответов с неверным форматом — {errors}.")
    section("Лабиринты по размерам",
            tables(["Карта", "Итог", "Решения / лимит", "Кратчайший путь", "Записей в файл"], maze_rows, {2, 3, 4}), detail)
    if "individual_questions" in data:
        extra = data["individual_questions"]
        section("Дополнительный замер SVG по одному вопросу", svg_table(extra["scores"]),
                extra["description_ru"], folded=True)

    digest = data["datasets"]["generated"]["dataset_sha256"]
    methods = data["method_ru"]
    notes = ''.join(f'<p>{escape(note)}</p>' for note in methods)
    href = f'{REPO}/tree/{data["benchmark_source_commit"]}'
    html.append(f'<section class="method" id="method-{run_id}"><h2>Методика и данные</h2>{notes}'
                f'<details><summary>Отпечаток закрытого SVG-набора</summary><p class="release">{escape(digest)}</p></details>'
                f'<p><a href="{json_path}">Скачать результаты в JSON</a> · '
                f'<a href="{escape(href)}">Исходники измеренной версии</a></p></section></article>')
    md.extend(["### Методика", *methods, f"SHA-256 закрытого SVG-набора — `{digest}`.",
               f"[Исходники измеренной версии]({href})"])
    return '\n'.join(html), '\n\n'.join(md), title


def main():
    paths = sorted((DOCS / "results").glob("*.json"))
    if not paths:
        raise SystemExit("В docs/results нет опубликованных прогонов.")
    runs = [render_run(p) for p in paths]
    if len(runs) == 1:
        nav = (f'<a href="#method-{escape(paths[0].stem)}">Методика</a>'
               f'<a href="results/{escape(paths[0].name)}">JSON</a>')
    else:
        nav = ''.join(f'<a href="#{escape(path.stem)}">{escape(title)}</a>'
                      for path, (_, _, title) in zip(paths, runs, strict=True))
    body = '\n'.join(fragment for fragment, _, _ in runs)
    html = f'''<!doctype html>
<html lang="ru">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Spatial Harness Bench · Лидерборд</title>
  <meta name="description" content="Результаты моделей и харнесов по пяти задачам Spatial Harness Bench.">
  <link rel="stylesheet" href="leaderboard.css">
</head>
<body>
<header><a class="brand" href="./">Spatial Harness Bench<span>.</span></a><nav aria-label="Навигация">{nav}</nav></header>
<main><div class="intro"><h1>Лидерборд</h1><p>{INTRO}</p></div>
{body}
</main>
<footer><span>Spatial Harness Bench</span><a href="{REPO}/blob/main/LEADERBOARD.md">Таблица в репозитории</a></footer>
</body></html>
'''
    (DOCS / "index.html").write_text(html, encoding="utf-8")
    markdown = ["# Spatial Harness Bench · Лидерборд", f"[Открыть на GitHub Pages]({SITE})", INTRO]
    markdown.extend(text for _, text, _ in runs)
    (ROOT / "LEADERBOARD.md").write_text('\n\n'.join(markdown) + '\n', encoding="utf-8")


if __name__ == "__main__":
    main()
