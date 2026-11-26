"""Versioned pilot instances and public instructions. No answer is sent to the harness."""

from dataclasses import dataclass
from math import cos, radians, sin, sqrt
from random import Random

from .model import Cable, Circle, CORE_IDS


FAMILIES = ("shell-thickness", "core-radii", "core-spacing", "add-core", "remove-core", "incompatible")
PILOT_SEEDS = tuple(range(5))
MAX_SEED = (1 << 63) - 1


@dataclass(frozen=True)
class Thickness:
    value: float
    keep: str


@dataclass(frozen=True)
class Radii:
    values: tuple[float, float, float]


@dataclass(frozen=True)
class Spacing:
    side: float


@dataclass(frozen=True)
class Addition:
    radius: float
    anchor: str
    percent: int


@dataclass(frozen=True)
class Removal:
    core_id: str


Edit = Thickness | Radii | Spacing | Addition | Removal


@dataclass(frozen=True)
class Instance:
    family: str
    seed: int
    original: Cable
    edit: Edit


def make_instance(family: str, seed: int) -> Instance:
    if family not in FAMILIES or not 0 <= seed <= MAX_SEED:
        raise ValueError("unknown family or seed outside [0, 2^63-1]")
    rng = Random(seed)
    x, y = (rng.randint(-12, 12) / 4 for _ in range(2))
    inner = rng.randint(104, 124) / 4
    outer = inner + rng.randint(12, 22) / 4
    gx, gy = (v + rng.randint(-6, 6) / 4 for v in (x, y))
    side = rng.randint(68, 88) / 4
    angle = radians(rng.randrange(360))
    cores = tuple(
        (name, Circle(gx + side / sqrt(3) * cos(angle + radians(120 * i)),
                      gy + side / sqrt(3) * sin(angle + radians(120 * i)),
                      rng.randint(14, 21) / 4))
        for i, name in enumerate(CORE_IDS[:3])
    )
    original = Cable(Circle(x, y, outer), Circle(x, y, inner), cores)
    edit: Edit
    if family == "shell-thickness":
        delta = rng.choice((-1, 1)) * rng.randint(3, 8) / 4
        edit = Thickness(outer - inner + delta, "inner" if seed % 2 == 0 else "outer")
    elif family == "core-radii":
        a, b, c = (core.r + rng.choice((-1, 1)) * rng.randint(2, 6) / 4 for _, core in cores)
        edit = Radii((a, b, c))
    elif family == "core-spacing":
        edit = Spacing(side + rng.choice((-1, 1)) * rng.randint(8, 14) / 4)
    elif family == "add-core":
        edit = Addition(rng.randint(7, 12) / 4, rng.choice(CORE_IDS[:3]), rng.choice((0, 10, 15)))
    elif family == "remove-core":
        edit = Removal(cores[seed % 3][0])
    else:
        variant = seed % 5
        if variant == 0:
            edit = Radii((side / 2 + 1, side / 2 + 1, cores[2][1].r))
        elif variant == 1:
            edit = Thickness(outer - 8, "outer")
        elif variant == 2:
            edit = Spacing(cores[0][1].r + cores[1][1].r - 1)
        elif variant == 3:
            edit = Addition(2, "core-b", 100)
        else:
            edit = Addition(2, "core-c", 400)
    return Instance(family, seed, original, edit)


FORMAT = """Формат результата:
- Если задача разрешима, запиши output.svg и status.json с единственным полем {"status":"solved"}.
- Если ограничения геометрически несовместимы, запиши только status.json с {"status":"infeasible"}.
  Не создавай output.svg при отказе. Входные файлы не изменяй.
- Оба файла результата должны быть обычными файлами UTF-8: до 64 КиБ для SVG и 4 КиБ для JSON.
- В SVG числа в координатах пользователя выражены в мм; ось x направлена вправо, y вниз.
  Сохрани width="100mm", height="100mm" и viewBox="-50 -50 100 100". Центры и радиусы
  указывай с достаточной точностью (допуск размеров 0.00001 мм, общих границ 0.000001 мм;
  лучше вычислять и писать не менее 9 знаков после запятой).
- Материалы — непосредственные дочерние элементы <svg> в пространстве имён SVG. Сохраняй
  id и fill материалов; для новой жилы используй id="core-d" и fill="#b66038".
  Оболочка shell — <path fill-rule="evenodd"> с двумя концентрическими контурами.
  Изоляция insulation — <path fill-rule="evenodd"> с внешним контуром и отдельным круглым
  отверстием под каждую жилу. Отверстия и контуры жил должны совпадать геометрически;
  внешняя граница изоляции должна совпадать с внутренней границей оболочки.
- Жила — <circle cx="…" cy="…" r="…"> или <path> с одним круглым контуром.
  Контур пути: M/m, не менее двух явных A/a с одинаковыми положительными rx и ry,
  один обход в одном направлении, возвращение в начальную точку, Z/z. Удобно
  использовать четыре четверти окружности. Пример с двумя полукругами:
  M 4 0 A 4 4 0 0 1 -4 0 A 4 4 0 0 1 4 0 Z. Флаги 0/1 пиши отдельными токенами.
  Допустимы только указанные элементы и атрибуты (у пути жилы правило evenodd необязательно).
  В частности, нельзя использовать группы, преобразования, стили, обводки, маски и ссылки.
- Жилы не должны касаться друг друга или оболочки. Внутри наружной границы оболочки
  не должно быть пустот или перекрытия материалов; снаружи не должно быть материалов.
  Не меняй геометрию, изменение которой не требуется поручением.
"""


def prompt_for(instance: Instance) -> str:
    edit = instance.edit
    match edit:
        case Thickness(value, keep):
            fixed = "внутреннюю" if keep == "inner" else "наружную"
            request = (f"Установи толщину кольцевой оболочки ровно {value:g} мм "
                       f"(разность наружного и внутреннего радиусов), сохранив {fixed} границу "
                       "на месте и концентричность. Не изменяй жилы; обнови изоляцию по общей границе.")
        case Radii(values):
            targets = ", ".join(f"{name}: {r:g} мм" for name, r in zip(CORE_IDS[:3], values, strict=True))
            request = (f"Установи радиусы жил: {targets}. Сохрани все центры и оболочку. "
                       "Обнови соответствующие отверстия изоляции.")
        case Spacing(side):
            request = (f"Сделай расстояние между каждой парой центров жил ровно {side:g} мм. "
                       "Сохрани общий центр (среднее трёх центров), радиусы жил и оболочку. "
                       "Каждый центр должен остаться на том же луче из общего центра (без поворота, "
                       "отражения или перестановки). Обнови отверстия изоляции.")
        case Addition(radius, anchor, percent):
            location = ("в среднем арифметическом координат трёх исходных центров"
                        if percent == 0 else
                        f"на луче от среднего арифметического трёх исходных центров к центру {anchor}, "
                        f"на расстоянии {percent}% от длины отрезка между этими двумя точками")
            request = (f"Добавь жилу core-d радиусом {radius:g} мм с центром {location}. "
                       "Не меняй исходные жилы и оболочку; сделай новое отверстие в изоляции.")
        case Removal(core_id):
            request = (f"Удали элемент {core_id} и соответствующее отверстие изоляции, "
                       "чтобы изоляция заполнила освободившийся круг. Не меняй остальные жилы и оболочку.")
    return ("Отредактируй инженерное поперечное сечение в input.svg. "
            "Это плоская геометрическая задача; электрические свойства не учитываются.\n\n"
            f"{request}\n\n{FORMAT}")
