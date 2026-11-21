"""Independent checks on original picture bytes and positional scoring."""

from dataclasses import dataclass
import re

from ..answers import MATCHER_VERSION, duplicates, expand, normalize, same
from .task import FILES, TASK_ID


def parse_reference(data: bytes) -> list[str]:
    names = data.decode("utf-8-sig").splitlines()
    keys = {normalize(name) for name in names}
    if len(names) != len(FILES) or len(keys) != len(FILES) or "" in keys:
        raise ValueError("На первом этапе desc.txt должен содержать 10 разных непустых названий.")
    if pairs := duplicates(names):
        same_pairs = ", ".join(f"«{names[i]}» и «{names[j]}»" for i, j in pairs)
        raise ValueError(f"На первом этапе названия совпадают по смыслу: {same_pairs}.")
    return names


@dataclass(frozen=True)
class DrawingIssue:
    kind: str
    reason: str


# Common transliteration catches a Russian name even when its ASCII letters
# are separated by strokes. Joined words are also checked independently.
_TRANSLIT = str.maketrans(dict(zip(
    "абвгдежзийклмнопрстуфхцчшщъыьэюя",
    "a b v g d e zh z i y k l m n o p r s t u f kh ts ch sh shch _ y _ e yu ya".split(),
    strict=True,
)))
_WORD = re.compile(r"[^\W\d_]+")
_SPELLED_OUT = re.compile(r"(?<![^\W\d_])[^\W\d_]([ ._-])[^\W\d_](?:\1[^\W\d_])+(?![^\W\d_])")


def compact(text: str) -> str:
    return "".join(c for c in normalize(text) if c.isalnum())


def caption_keys(names: list[str]) -> dict[str, str]:
    keys = {}
    for name in expand(names):
        for spelling in (name, normalize(name).translate(_TRANSLIT)):
            key = compact(spelling)
            if key:
                keys[key] = name
    return keys


def find_caption(text: str, names: list[str]) -> str | None:
    """Inspect rows and columns; isolated or repeated letter strokes are OK."""
    keys = caption_keys(names)

    def inspect(line: str, min_word: int) -> str | None:
        joined = compact(line)
        for key, name in keys.items():
            if key in joined:
                return f"Название в рисунке: {name}"
        for match in _WORD.finditer(line):
            word = match.group()
            if len(word) >= min_word and len(set(word.casefold())) > 1:
                return f"Текст в рисунке: {word}"
        for match in _SPELLED_OUT.finditer(line):
            if len(set(compact(match.group()))) > 1:
                return f"Текст в рисунке: {match.group()}"
        return None

    # Keep columns sparse so ragged drawings take space proportional to the
    # input. A gap is a word boundary, including a gap at a short line end.
    columns: dict[int, list[str]] = {}
    last_row: dict[int, int] = {}
    for row, line in enumerate(text.splitlines()):
        if reason := inspect(line, 2):
            return reason
        for col, char in enumerate(line):
            if char.isalnum():
                column = columns.setdefault(col, [])
                if last_row.get(col) != row - 1:
                    column.append(" ")
                column.append(char)
                last_row[col] = row
    for column in columns.values():
        if reason := inspect("".join(column), 3):
            return reason
    return None


def check_drawing(data: bytes, names: list[str]) -> DrawingIssue | None:
    text = data.decode("utf-8", errors="replace").replace("\r\n", "\n")
    if reason := find_caption(text, names):
        return DrawingIssue("caption", reason)
    if not text.strip() or any(c != "\n" and not (" " <= c <= "~") for c in text):
        return DrawingIssue("invalid_art", "Нужен непустой рисунок из печатных символов ASCII.")
    return None


def score(names: list[str], answer: bytes, issues: list[DrawingIssue | None], *,
          files: tuple[str, ...] = FILES, task: str = TASK_ID) -> dict:
    guesses = answer.decode("utf-8-sig", errors="replace").splitlines()
    items = []
    for i, (file, name, issue) in enumerate(zip(files, names, issues, strict=True)):
        guess = guesses[i] if i < len(guesses) else ""
        if issue:
            status = issue.kind
            points = -1 if status == "caption" else 0
        else:
            points = int(same(guess, name))
            status = "match" if points else "miss"
        items.append({"file": file, "expected": name, "guess": guess, "status": status,
                      "reason": issue.reason if issue else None, "points": points})
    return {"task": task, "score": max(0, sum(item["points"] for item in items)),
            "max_score": len(files), "matcher": MATCHER_VERSION, "items": items}
