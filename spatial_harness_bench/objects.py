"""Runner-assigned objects: the artist no longer chooses its own, easiest subjects."""

import random

from .answers import duplicates

# Mixed difficulty on purpose: some subjects differ from others only by a few details.
POOL = (
    "кошка", "собака", "рыба", "сова", "улитка", "жираф", "паук", "бабочка", "черепаха", "утка",
    "ёлка", "кактус", "гриб", "яблоко", "груша", "морковь", "дом", "маяк", "палатка", "лестница",
    "зонт", "ключ", "чашка", "чайник", "бутылка", "свеча", "лампочка", "колокол", "корона", "якорь",
    "ножницы", "молоток", "очки", "часы", "гитара", "книга", "стул", "велосипед", "самолёт", "лодка",
)
COUNT = 10

if duplicates(list(POOL)):
    raise RuntimeError("В пуле объектов есть синонимы.")


def assign(seed: int) -> list[str]:
    return random.Random(seed).sample(POOL, COUNT)


def draw_prompt(names: list[str], ext: str) -> str:
    rows = "\n".join(f"{i}.{ext} — {name}" for i, name in enumerate(names))
    return ("\nОбъекты назначены заранее, выбирать их не нужно. Нарисуй каждый в своём файле:\n"
            f"{rows}\nВ desc.txt перепиши эти десять названий в том же порядке.\n")
