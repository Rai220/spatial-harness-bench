"""Answer equivalence shared by every task: case, «ё», word forms and curated synonyms.

Two answers are equal when, after normalization, they have the same number of words and
each pair of words shares a dictionary lemma, or when both phrases belong to the same
synonym group. Hypernyms are not merged: «ёлка» is not «дерево».
"""

from functools import cache
from importlib.resources import files
import json
import re
import unicodedata

import pymorphy3

MATCHER_VERSION = "lemma-synonyms-1"
_WORD = re.compile(r"[^\W_]+(?:-[^\W_]+)*")


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFC", text).casefold().replace("ё", "е")
    return " ".join(text.split())


@cache
def _morph() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


@cache
def lemmas(word: str) -> frozenset[str]:
    """Every dictionary lemma of one word, so a rare competing parse cannot hide the right one."""
    forms = {normalize(word)}
    for parse in _morph().parse(word):
        forms.add(normalize(parse.normal_form))
    return frozenset(forms)


def words(text: str) -> tuple[str, ...]:
    return tuple(_WORD.findall(normalize(text)))


@cache
def _groups() -> tuple[dict[tuple[str, ...], int], dict[str, int]]:
    """Map each listed phrase to its group, both literally and by the lemma of a single word."""
    data = json.loads(files(__package__).joinpath("synonyms.json").read_text(encoding="utf-8"))
    phrases: dict[tuple[str, ...], int] = {}
    single: dict[str, int] = {}
    for index, group in enumerate(data["groups"]):
        for phrase in group:
            key = words(phrase)
            if phrases.setdefault(key, index) != index:
                raise ValueError(f"Синоним {phrase!r} указан в двух группах.")
            if len(key) == 1:
                single.setdefault(key[0], index)
    return phrases, single


@cache
def word_group(word: str) -> int | None:
    single = _groups()[1]
    for lemma in sorted(lemmas(word)):
        if lemma in single:
            return single[lemma]
    return None


def synonym_group(text: str) -> int | None:
    key = words(text)
    phrases = _groups()[0]
    if key in phrases:
        return phrases[key]
    return word_group(key[0]) if len(key) == 1 else None


def _same_word(x: str, y: str) -> bool:
    if lemmas(x) & lemmas(y):
        return True
    group = word_group(x)
    return group is not None and group == word_group(y)


def same(answer: str, reference: str) -> bool:
    a, b = words(answer), words(reference)
    if not a or not b:
        return False
    if a == b:
        return True
    if len(a) == len(b) and all(_same_word(x, y) for x, y in zip(a, b)):
        return True
    group = synonym_group(reference)
    return group is not None and group == synonym_group(answer)


def matches_any(answer: str, references) -> bool:
    return any(same(answer, reference) for reference in references)


def duplicates(names: list[str]) -> list[tuple[int, int]]:
    """Pairs of positions whose names would be accepted as the same answer."""
    return [(i, j) for i in range(len(names)) for j in range(i + 1, len(names))
            if same(names[i], names[j])]


def expand(names: list[str]) -> list[str]:
    """Reference names plus their listed synonyms, for the caption detectors."""
    data = json.loads(files(__package__).joinpath("synonyms.json").read_text(encoding="utf-8"))
    result = list(names)
    for name in names:
        group = synonym_group(name)
        if group is not None:
            result.extend(p for p in data["groups"][group] if p not in result)
    return result


NUMBERS = ("ноль один два три четыре пять шесть семь восемь девять десять одиннадцать двенадцать "
           "тринадцать четырнадцать пятнадцать шестнадцать семнадцать восемнадцать девятнадцать "
           "двадцать").split()


def number_aliases(value: int) -> list[str]:
    aliases = [str(value)]
    if 0 <= value < len(NUMBERS):
        aliases.append(NUMBERS[value])
    return aliases
