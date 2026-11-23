"""Textual floor surveys and exact geometry for tasks 06–10."""

from dataclasses import dataclass
from itertools import groupby, product
import json
import re

from .house import COLUMNS, FLOORS, floor_cells

VERSION = 1
INLETS = ("A1", "A5", "B1", "E1", "F3")
OUTLETS = ("A3", "B4", "C3", "D1", "E5")
BLOCKED = {"ядро", "щитовая"}


@dataclass(frozen=True)
class Task:
    id: str
    title: str
    floors: tuple[int, ...]
    question: str
    fields: str


TASKS = (
    Task("06-building-above", "Сосед сверху", (1, 2),
         "В самой северной полосе квартиры №105 на первом этаже выбери крайнюю восточную клетку "
         "этой квартиры. Какая квартира находится непосредственно над этой клеткой?",
         '"apartment" — номер квартиры'),
    Task("07-building-noise", "Стук в потолок", (1, 2),
         "Во всех клетках спортзала на втором этаже стучат по полу. По условию стук в потолок "
         "слышен только в клетке прямо под источником на соседнем этаже. Перечисли все квартиры, "
         "в которых слышен стук хотя бы в одной клетке.",
         '"apartments" — массив номеров квартир'),
    Task("08-building-alignment", "Три помещения на одной вертикали", (2, 3, 4),
         "Мастерской на втором этаже нужна точка связи с серверной на третьем и диспетчерской "
         "на четвёртом. Перечисли все клетки, в которых одна и та же вертикаль проходит "
         "внутри всех трёх помещений. Ответ дай в общих координатах дома.",
         '"cells" — массив обозначений клеток'),
    Task("09-building-shaft", "Прямой воздуховод", tuple(FLOORS),
         "Воздуховод занимает одну и ту же клетку на всех пяти этажах. Войти из подвала и выйти "
         "на крышу можно в любой клетке. Ядро и щитовая непроходимы. Квартира считается "
         "затронутой, если трасса проходит хотя бы через одну её клетку. Один номер считается "
         "один раз по всей трассе. Найди минимальное число затронутых квартир и ВСЕ положения "
         "прямого воздуховода, достигающие минимума. Используй общие координаты.",
         '"count" — минимум квартир, "cells" — массив всех оптимальных клеток'),
    Task("10-building-offset", "Воздуховод со смещением", tuple(FLOORS),
         f"Вводы из подвала доступны в клетках {', '.join(INLETS)}. Выходы на крышу доступны в "
         f"клетках {', '.join(OUTLETS)}. Все обозначения отверстий даны в общих координатах. "
         "Трасса поднимается по клетке ввода до третьего этажа. На третьем она проходит по одному "
         "прямому горизонтальному отрезку вдоль ряда или столбца, включая оба конца. Затем "
         "поднимается по клетке выхода до крыши. Ядро и щитовая непроходимы на всех участках. "
         "Сначала минимизируй число РАЗЛИЧНЫХ квартир на всей трассе, включая горизонтальный "
         "участок. При равенстве минимизируй длину горизонтального участка в шагах между "
         "центрами соседних клеток. Дай один оптимальный маршрут и все затронутые им квартиры.",
         '"inlet" и "outlet" — клетки ввода и выхода в общих координатах, '
         '"apartments" — массив номеров затронутых квартир, "length" — длина смещения'),
)


def floor_description(number):
    # Store the survey once in building coordinates. Only this local sheet reads east to west.
    mirrored = number == 3
    direction = "с востока на запад" if mirrored else "с запада на восток"
    lines = [f"Этаж {number}. На этом листе колонки A–F идут {direction}, ряды 1–5 — с севера на юг."]
    for row, line in enumerate(FLOORS[number], 1):
        names = line.split()
        if mirrored:
            names.reverse()
        spans = []
        for name, group in groupby(zip(COLUMNS, names, strict=True), key=lambda pair: pair[1]):
            letters = [col for col, _ in group]
            span = letters[0] if len(letters) == 1 else f"{letters[0]}–{letters[-1]}"
            spans.append(f"{span} — {name}")
        lines.append(f"Ряд {row}: " + "; ".join(spans) + ".")
    return "\n".join(lines)


def prompt(level):
    task = TASKS[level - 1]
    return "\n\n".join([
        f"Дом «Маяк». {task.title}.",
        "Все этажи занимают один прямоугольник 6 × 5. В общих координатах латинские колонки "
        "A–F идут с запада на восток, ряды 1–5 — с севера на юг. A1 — северо-западный угол. "
        "У каждого поэтажного листа указано своё направление колонок. Диапазоны включают оба "
        "края. Число в описании обозначает квартиру. Одинаковый номер на разных этажах "
        "принадлежит одной двухуровневой квартире. Слова обозначают нежилые помещения.",
        *(floor_description(n) for n in task.floors), task.question,
        f"Запиши в answer.json один JSON-объект. Поля: {task.fields}. "
        "Номера и длины записывай целыми числами, клетки — строками. Порядок в массивах свободный.",
    ])


def apartments(names):
    return sorted({int(name) for name in names if name.isdigit()})


def horizontal(a, b):
    """The occupied cells on a straight run, including its two ends."""
    if a[0] == b[0]:
        lo, hi = sorted((int(a[1]), int(b[1])))
        return [a[0] + str(row) for row in range(lo, hi + 1)]
    if a[1] == b[1]:
        lo, hi = sorted((COLUMNS.index(a[0]), COLUMNS.index(b[0])))
        return [col + a[1] for col in COLUMNS[lo:hi + 1]]
    return []


def solutions(level):
    plans = {n: floor_cells(n) for n in FLOORS}
    if level == 1:
        cells = [c for c, room in plans[1].items() if room == "105"]
        north = min(c[1] for c in cells)
        target = max(c for c in cells if c[1] == north)
        return [{"apartment": int(plans[2][target])}]
    if level == 2:
        below = (plans[1][c] for c, room in plans[2].items() if room == "спортзал")
        return [{"apartments": apartments(below)}]
    if level == 3:
        cells = [c for c in plans[2] if (plans[2][c], plans[3][c], plans[4][c])
                 == ("мастерская", "серверная", "диспетчерская")]
        return [{"cells": sorted(cells)}]
    if level == 4:
        costs = {}
        for cell in plans[1]:
            names = {floor[cell] for floor in plans.values()}
            if names.isdisjoint(BLOCKED):
                costs[cell] = len(apartments(names))
        best = min(costs.values())
        return [{"count": best, "cells": sorted(c for c, cost in costs.items() if cost == best)}]
    if level != 5:
        raise ValueError(f"Неизвестный уровень {level}")
    routes = []
    for inlet, outlet in product(INLETS, OUTLETS):
        segment = horizontal(inlet, outlet)
        if not segment:
            continue
        names = {plans[1][inlet], plans[2][inlet], plans[4][outlet], plans[5][outlet]}
        names.update(plans[3][c] for c in segment)
        if names.isdisjoint(BLOCKED):
            routes.append({"inlet": inlet, "outlet": outlet, "length": len(segment) - 1,
                           "apartments": apartments(names)})
    def cost(route):
        return len(route["apartments"]), route["length"]
    best = min(map(cost, routes))
    return [route for route in routes if cost(route) == best]


def normalize(field, value):
    if field in ("apartments", "cells"):
        if not isinstance(value, list):
            raise ValueError("Ожидался массив")
        singular = "apartment" if field == "apartments" else "cell"
        return sorted({normalize(singular, item) for item in value})
    if field in ("count", "apartment", "length"):
        if type(value) is not int:
            raise ValueError("Ожидалось целое число")
        return value
    if not isinstance(value, str) or not re.fullmatch(r"[A-F][1-5]", cell := value.strip().upper()):
        raise ValueError("Неизвестная клетка")
    return cell


def grade(level, text):
    expected = solutions(level)
    try:
        raw = json.loads(text)
        if not isinstance(raw, dict):
            raise ValueError("Ожидался JSON-объект")
        answer = {key: normalize(key, raw[key]) for key in expected[0]}
    except (ValueError, KeyError) as exc:
        return {"points": 0, "expected": expected, "answer": None, "error": str(exc)}
    return {"points": int(answer in expected), "expected": expected, "answer": answer}
