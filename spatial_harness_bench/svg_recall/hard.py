"""Trace objects through row, column and square permutations on a shared board."""

from dataclasses import dataclass
from functools import cache
from itertools import combinations
import random
import re
from xml.etree import ElementTree as ET

import cairosvg

from ..answers import MATCHER_VERSION, same
from ..ascii_recall.verifier import DrawingIssue
from .task import FILES
from .verifier import Picture, SVG

TASK_ID = "03-svg-recall-hard"
SIDE = 4
COUNTS = (2, 3, 4, 5, 6, 7, 8, 9, 10, 12)
EXPERT_COUNTS = (4, 5, 6, 7, 8, 9, 10, 12, 14, 16)
MASTER_COUNTS = (8, 10, 12, 14, 16, 18, 20, 22, 25, 28)
LEVEL_COUNTS = {"hard": COUNTS, "expert": EXPERT_COUNTS, "master": MASTER_COUNTS}


def coordinates(cell: int) -> tuple[int, int]:
    r, c = divmod(cell, SIDE)
    return r + 1, c + 1


@dataclass(frozen=True)
class Move:
    text: str
    destination: tuple[int, ...]
    fixed_landmarks: tuple[int, ...] | None = None


def moves() -> tuple[Move, ...]:
    result = []

    def pair(cycles, forward, reverse):
        dest = list(range(SIDE**2))
        for cycle in cycles:
            for src, dst in zip(cycle, cycle[1:] + cycle[:1]):
                dest[src] = dst
        result.append(Move(forward, tuple(dest)))
        result.append(Move(reverse, tuple(dest.index(i) for i in range(len(dest)))))

    for n in range(SIDE):
        row = [n * SIDE + k for k in range(SIDE)]
        col = [k * SIDE + n for k in range(SIDE)]
        pair([row], f"строку {n+1} вправо", f"строку {n+1} влево")
        pair([col], f"столбец {n+1} вниз", f"столбец {n+1} вверх")
    for row in range(SIDE - 1):
        for col in range(SIDE - 1):
            cell = row * SIDE + col
            square = [cell, cell + 1, cell + SIDE + 1, cell + SIDE]
            name = f"квадрат ({row+1},{col+1})"
            pair([square], name + " по часовой", name + " против часовой")
    turn = [col * SIDE + SIDE - 1 - row for row in range(SIDE) for col in range(SIDE)]
    result.append(Move("всё поле по часовой", tuple(turn)))
    result.append(Move("всё поле против часовой", tuple(turn.index(i) for i in range(len(turn)))))
    return tuple(result)


MOVES = moves()
SHIFTS = {move.text: move.destination for move in MOVES}


@dataclass(frozen=True)
class Question:
    origin: int
    end: int
    moves: tuple[Move, ...]


@cache
def swap_lines(axis: int, first: int, second: int) -> tuple[int, ...]:
    dest = []
    swap = {first: second, second: first}
    for cell in range(SIDE**2):
        rc = list(divmod(cell, SIDE))
        rc[axis] = swap.get(rc[axis], rc[axis])
        dest.append(rc[0] * SIDE + rc[1])
    return tuple(dest)


def anchored_moves(positions: list[int], anchors: list[int]):
    """Reference subjects are drawn from all objects, independently of the expected answer."""
    for axis, singular, plural, directions in (
        (0, "строку", "строки", ("влево", "вправо")),
        (1, "столбец", "столбцы", ("вверх", "вниз")),
    ):
        lines = {a: divmod(positions[a], SIDE)[axis] for a in anchors}
        initial = {a: divmod(a, SIDE)[axis] for a in anchors}
        for a in anchors:
            if lines[a] == initial[a]:
                continue
            r, c = coordinates(a)
            for direction in directions:
                text = f"{singular} с предметом из ({r},{c}) {direction}"
                yield Move(text, SHIFTS[f"{singular} {lines[a]+1} {direction}"],
                           SHIFTS[f"{singular} {initial[a]+1} {direction}"])
        for a, b in combinations(anchors, 2):
            first, second = lines[a], lines[b]
            if first == second or {first, second} == {initial[a], initial[b]}:
                continue
            ra, ca = coordinates(a)
            rb, cb = coordinates(b)
            text = f"поменять местами {plural} с предметами из ({ra},{ca}) и ({rb},{cb})"
            yield Move(text, swap_lines(axis, first, second), swap_lines(axis, initial[a], initial[b]))


def expert_question(rng: random.Random, origins: list[int], origin: int, length: int) -> Question:
    # Generate commands without consulting the target. Reject a chain if it merely pads the
    # question or if interpreting all landmark references as fixed would give the right answer.
    while True:
        positions = list(range(SIDE**2))  # initial cell -> current cell
        sequence = []
        fixed_target = origin
        moved = 0
        for step in range(length):
            if step == 0:
                options = MOVES[-2:]  # Move the landmarks before the first reference.
            elif step % 4 == 3 and step < length - 1:
                options = MOVES
            else:
                options = list(anchored_moves(positions, origins)) or MOVES[-2:]
            move = rng.choice(options)
            sequence.append(move)
            target = positions[origin]
            positions = [move.destination[cell] for cell in positions]
            fixed_target = (move.fixed_landmarks or move.destination)[fixed_target]
            moved += positions[origin] != target
        end = positions[origin]
        if end not in (origin, fixed_target) and moved * 2 >= length:
            return Question(origin, end, tuple(sequence))


def make_questions(seed: int, *, expert: bool = False, difficulty: str | None = None
                   ) -> tuple[list[int | None], list[Question]]:
    """hard: fixed moves only; expert and master: commands refer to objects' current positions."""
    difficulty = difficulty or ("expert" if expert else "hard")
    rng = random.Random(seed)
    board = list(range(len(FILES))) + [None] * (SIDE**2 - len(FILES))
    rng.shuffle(board)
    origins = [i for i, obj in enumerate(board) if obj is not None]
    rng.shuffle(origins)
    if difficulty != "hard":
        return board, [expert_question(rng, origins, origin, n)
                       for origin, n in zip(origins, LEVEL_COUNTS[difficulty], strict=True)]

    def path(cell: int, remaining: int, visited: set[int]) -> tuple[tuple[Move, ...], int] | None:
        if remaining == 0:
            return (), cell
        choices = [m for m in MOVES if m.destination[cell] not in visited]
        rng.shuffle(choices)
        for move in choices:
            dest = move.destination[cell]
            tail = path(dest, remaining - 1, visited | {dest})
            if tail is not None:
                sequence, end = tail
                return (move, *sequence), end
        return None

    questions = []
    for origin, length in zip(origins, COUNTS, strict=True):
        result = path(origin, length, {origin})
        assert result is not None  # Row/column shifts allow a path through all 16 cells.
        sequence, end = result
        questions.append(Question(origin, end, sequence))
    return board, questions


def render_board(board: list[int | None], pictures: list[Picture]) -> tuple[bytes, bytes]:
    tile = 256
    width = tile * SIDE
    root = ET.Element(f"{{{SVG}}}svg", {"viewBox": f"0 0 {width} {width}"})
    next_id = 0
    for cell, index in enumerate(board):
        if index is None or pictures[index].issue is not None:
            continue
        row, col = divmod(cell, SIDE)
        icon = ET.fromstring(pictures[index].source)
        # Independent source files can reuse the same IDs for gradients and clips.
        ids = {}
        for node in icon.iter():
            if local_id := node.get("id"):
                next_id += 1
                ids[local_id] = f"g{next_id}"
        for node in icon.iter():
            for key, value in node.attrib.items():
                if key == "id":
                    node.set(key, ids[value])
                elif "url(#" in value:
                    node.set(key, re.sub(r"url\(#([A-Za-z]\d+)\)",
                                         lambda m: f"url(#{ids[m[1]]})" if m[1] in ids else "none", value))
        icon.attrib.update(x=str(col * tile + 8), y=str(row * tile + 8),
                           width=str(tile - 16), height=str(tile - 16))
        root.append(icon)
    for i in range(SIDE + 1):
        offset = i * tile
        ET.SubElement(root, "path", {"d": f"M0 {offset}H{width} M{offset} 0V{width}",
                                      "stroke": "#a5adb4", "stroke-width": "2"})
    source = ET.tostring(root, encoding="utf-8")
    return source, cairosvg.svg2png(bytestring=source, output_width=width, output_height=width,
                                    background_color="white")


def questions_text(questions: list[Question], *, expert: bool = False) -> str:
    lines = [
        "Перед тобой исходное поле 4×4 с десятью объектами. Каждый вопрос начинается с этого же поля.",
        "Координаты (строка,столбец) считаются с 1: строки сверху вниз, столбцы слева направо.",
        "Мысленно выполни команды по порядку. Строка или столбец сдвигается на одну клетку циклически, "
        "крайняя клетка переходит на другой конец. Любой поворот — на 90°. Для квадрата 2×2 указана его "
        "верхняя левая клетка. Номера относятся к текущему положению на экране после предыдущих команд.",
        "Для указанной конечной клетки найди, из какой исходной клетки пришёл её предмет, и узнай этот предмет. "
        "В desc.txt дай 10 строк по порядку вопросов в формате «строка,столбец | название». "
        "Укажи исходные координаты и одно простое русское название в именительном падеже. "
        "Если часть ответа неизвестна, оставь её пустой, сохраняя строку и разделитель. "
        "Балл даёт верная исходная клетка. Название учитывается отдельно.",
    ]
    if expert:
        lines.append("«Предмет из (r,c)» означает тот предмет, который стоял в (r,c) в начале вопроса. "
                     "Найди его текущее положение непосредственно перед данной командой. Например, "
                     "«строку с предметом из (3,2) вправо» сдвигает целиком текущую строку этого предмета. "
                     "При обмене строк или столбцов определи обе линии до обмена и поменяй их целиком, одновременно.")
    for i, question in enumerate(questions, 1):
        r, c = coordinates(question.end)
        lines.append(f"{i}. " + " → ".join(m.text for m in question.moves) + f". Конечная клетка ({r},{c}).")
    return "\n".join(lines) + "\n"


def grade(board: list[int | None], questions: list[Question], names: list[str],
          issues: list[DrawingIssue | None], answer: bytes, *, task: str = TASK_ID) -> dict:
    lines = answer.decode("utf-8-sig", errors="replace").splitlines()
    items = []
    for i, q in enumerate(questions):
        index = board[q.origin]
        issue = issues[index]
        line = lines[i] if i < len(lines) else ""
        location, sep, name = line.partition("|")
        match = re.fullmatch(r"\s*([1-4])\s*,\s*([1-4])\s*", location)
        guess_origin = tuple(map(int, match.groups())) if sep and match else None
        origin_ok = guess_origin == coordinates(q.origin)
        name_ok = bool(sep) and same(name, names[index])
        # The board task measures tracking. Recognition is reported, not scored, here.
        points = int(origin_ok)
        status = "match" if points else "miss"
        if issue:
            status, points = issue.kind, -int(issue.kind == "caption")
        items.append({"question": i + 1, "file": FILES[index], "end": list(coordinates(q.end)),
                      "moves": [m.text for m in q.moves], "expected_origin": list(coordinates(q.origin)),
                      "guess_origin": list(guess_origin) if guess_origin else None, "expected": names[index],
                      "guess": name.strip(), "raw": line, "origin_match": origin_ok, "name_match": name_ok,
                      "status": status, "points": points, "reason": issue.reason if issue else None})
    eligible = [item for item in items if item["status"] in {"match", "miss"}]
    return {"task": task, "score": max(0, sum(i["points"] for i in items)), "max_score": len(items),
            "scoring": "origin", "matcher": MATCHER_VERSION, "eligible": len(eligible),
            "origin_hits": sum(i["origin_match"] for i in eligible),
            "name_hits": sum(i["name_match"] for i in eligible),
            "full_hits": sum(i["origin_match"] and i["name_match"] for i in eligible), "items": items}
