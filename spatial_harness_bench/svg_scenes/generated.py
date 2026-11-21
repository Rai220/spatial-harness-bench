"""Procedural scenes: unlimited items with keys that are correct by construction.

A run is defined by its seed. Keep the seed private to obtain a closed test set; publish it to
let anyone replay the same items. Four kinds of question grow harder across three tiers:

count      shapes of one class fully on one side of a line; distractors share colour or shape,
           some objects straddle the line
occlusion  which shape lies over A and is itself partly covered by B (visible pixels decide)
cords      trace a cord through crossings; in SVG source each cord is cut into shuffled pieces
transform  rotate and mirror a card mentally, then name the shapes in two cells
"""

from dataclasses import dataclass
from functools import cache
import math
import random

import cairosvg
from PIL import Image, ImageChops, ImageDraw

from ..answers import number_aliases
from ..svg_recall.verifier import Picture

GENERATOR_VERSION = 1
KINDS = ("count", "occlusion", "cords", "transform")
TIERS = (1, 2, 3)
SIDE = 512
BACKGROUND = "#fbf8f1"
OUTLINE = "#2b2b2b"
COLORS = {
    "красный": "#d7263d", "оранжевый": "#f08a24", "жёлтый": "#f2c418", "зелёный": "#2e9e4f",
    "синий": "#2b59c3", "фиолетовый": "#8e5bb5", "розовый": "#f48fb1", "серый": "#8a8f98",
}
SHAPES = ("круг", "квадрат", "треугольник", "ромб", "звезда", "шестиугольник", "крест")
# Rotation limits keep a square from turning into a rhombus and a cross into an «x».
MAX_TURN = {"круг": 0, "квадрат": 12, "ромб": 8, "крест": 15, "треугольник": 180,
            "звезда": 180, "шестиугольник": 180}


@dataclass(frozen=True)
class Item:
    kind: str
    tier: int
    question: str
    answers: list[list[str]]
    picture: Picture


# ---------------------------------------------------------------- grammar

@cache
def _morph():
    import pymorphy3
    return pymorphy3.MorphAnalyzer()


def _noun(word: str):
    return next(p for p in _morph().parse(word) if p.tag.POS == "NOUN" and {"nomn", "sing"} <= p.tag.grammemes)


def _adj(word: str):
    return next(p for p in _morph().parse(word) if p.tag.POS == "ADJF")


def phrase(color: str | None, shape: str, case: str, plural: bool = False) -> str:
    noun = _noun(shape)
    number = {"plur"} if plural else set()
    words = [noun.inflect({case, *number}).word]
    if color:
        agreement = {case, "plur"} if plural else {case, noun.tag.gender}
        words.insert(0, _adj(color).inflect(agreement).word)
    return " ".join(words)


# ---------------------------------------------------------------- geometry and drawing

@dataclass(frozen=True)
class Shape:
    color: str
    kind: str
    cx: float
    cy: float
    r: float
    turn: float = 0.0

    def points(self) -> list[tuple[float, float]]:
        r = self.r
        if self.kind == "круг":
            local = [(r * math.cos(2 * math.pi * k / 48), r * math.sin(2 * math.pi * k / 48)) for k in range(48)]
        elif self.kind == "квадрат":
            h = r / math.sqrt(2)
            local = [(-h, -h), (h, -h), (h, h), (-h, h)]
        elif self.kind == "ромб":
            local = [(0, -r), (0.62 * r, 0), (0, r), (-0.62 * r, 0)]
        elif self.kind == "крест":
            w = 0.33 * r
            local = [(-w, -r), (w, -r), (w, -w), (r, -w), (r, w), (w, w), (w, r), (-w, r), (-w, w),
                     (-r, w), (-r, -w), (-w, -w)]
        else:
            count, inner = {"треугольник": (3, None), "шестиугольник": (6, None), "звезда": (5, 0.45)}[self.kind]
            local = []
            for k in range(count * (2 if inner else 1)):
                radius = r * inner if inner and k % 2 else r
                angle = -math.pi / 2 + math.pi * k / count * (1 if inner else 2)
                local.append((radius * math.cos(angle), radius * math.sin(angle)))
        a = math.radians(self.turn)
        return [(self.cx + x * math.cos(a) - y * math.sin(a), self.cy + x * math.sin(a) + y * math.cos(a))
                for x, y in local]

    def bbox(self) -> tuple[float, float, float, float]:
        xs, ys = zip(*self.points())
        return min(xs), min(ys), max(xs), max(ys)

    def svg(self) -> str:
        paint = f'fill="{COLORS[self.color]}" stroke="{OUTLINE}" stroke-width="3" stroke-linejoin="round"'
        if self.kind == "круг":
            return f'<circle cx="{self.cx:.1f}" cy="{self.cy:.1f}" r="{self.r:.1f}" {paint}/>'
        points = " ".join(f"{x:.1f},{y:.1f}" for x, y in self.points())
        return f'<polygon points="{points}" {paint}/>'

    def mask(self) -> Image.Image:
        image = Image.new("L", (SIDE, SIDE), 0)
        ImageDraw.Draw(image).polygon(self.points(), fill=255)
        return image


def document(body: list[str]) -> bytes:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {SIDE} {SIDE}">'
            f'<rect width="{SIDE}" height="{SIDE}" fill="{BACKGROUND}"/>' + "".join(body) + "</svg>").encode()


def picture(source: bytes) -> Picture:
    png = cairosvg.svg2png(bytestring=source, output_width=SIDE, output_height=SIDE, background_color="white")
    return Picture(source, png)


def random_shape(rng: random.Random, color: str, kind: str, cx: float, cy: float, r: float) -> Shape:
    limit = MAX_TURN[kind]
    return Shape(color, kind, cx, cy, r, rng.uniform(-limit, limit) if limit else 0.0)


class Retry(Exception):
    pass


# ---------------------------------------------------------------- count

COUNT_TIERS = {1: dict(n=10, radius=(18, 26), straddle=0), 2: dict(n=16, radius=(15, 22), straddle=2),
               3: dict(n=24, radius=(13, 19), straddle=3)}


def count_item(rng: random.Random, tier: int) -> Item:
    spec = COUNT_TIERS[tier]
    line = rng.randint(200, 312)
    side = rng.choice(("left", "right"))
    color, kind = rng.choice(list(COLORS)), rng.choice(SHAPES)
    wanted = rng.randint(2, 5)
    plan = [("target", side)] * wanted + [("target", "right" if side == "left" else "left")] * rng.randint(1, 2)
    plan += [("target", "straddle")] * spec["straddle"]
    while len(plan) < spec["n"]:
        plan.append(("other", rng.choice(("left", "right", "right", "left", "straddle"))))
    placed: list[Shape] = []
    for role, where in plan:
        for _ in range(400):
            if role == "target":
                c, k = color, kind
            else:  # Most distractors share the colour or the shape of the target class.
                c, k = rng.choice([(color, rng.choice([s for s in SHAPES if s != kind])),
                                   (rng.choice([x for x in COLORS if x != color]), kind),
                                   (rng.choice([x for x in COLORS if x != color]),
                                    rng.choice([s for s in SHAPES if s != kind]))])
            r = rng.uniform(*spec["radius"])
            shape = random_shape(rng, c, k, rng.uniform(r + 12, SIDE - r - 12), rng.uniform(r + 12, SIDE - r - 12), r)
            x0, _, x1, _ = shape.bbox()
            if where == "left" and x1 > line - 7 or where == "right" and x0 < line + 7:
                continue
            if where == "straddle" and not (x0 < line - 8 and x1 > line + 8):
                continue
            if any(math.dist((shape.cx, shape.cy), (o.cx, o.cy)) < shape.r + o.r + 6 for o in placed):
                continue
            placed.append(shape)
            break
        else:
            raise Retry
    rng.shuffle(placed)
    body = [s.svg() for s in placed]
    body.append(f'<line x1="{line}" y1="8" x2="{line}" y2="{SIDE - 8}" stroke="{OUTLINE}" '
                'stroke-width="4" stroke-dasharray="12 8"/>')
    direction = "левее" if side == "left" else "правее"
    question = (f"Сколько {phrase(color, kind, 'gent', plural=True)} находится целиком {direction} "
                "пунктирной линии? Фигуры, которые пересекает линия, не считай. Ответь числом.")
    return Item("count", tier, question, [number_aliases(wanted)], picture(document(body)))


# ---------------------------------------------------------------- occlusion

OCCLUSION_TIERS = {1: dict(n=5, radius=(60, 85)), 2: dict(n=8, radius=(50, 75)), 3: dict(n=11, radius=(45, 68))}
MIN_VISIBLE_OVERLAP = 180


def occlusion_item(rng: random.Random, tier: int) -> Item:
    spec = OCCLUSION_TIERS[tier]
    classes = rng.sample([(c, k) for c in COLORS for k in SHAPES], spec["n"])
    shapes = []
    for c, k in classes:
        r = rng.uniform(*spec["radius"])
        # Pull shapes toward the centre so that they overlap.
        cx = rng.gauss(SIDE / 2, 85)
        cy = rng.gauss(SIDE / 2, 85)
        cx, cy = min(max(cx, r + 10), SIDE - r - 10), min(max(cy, r + 10), SIDE - r - 10)
        shapes.append(random_shape(rng, c, k, cx, cy, r))
    masks = [s.mask() for s in shapes]
    n = len(shapes)
    # Binary masks in mode L: darker = intersection, lighter = union, invert = complement.
    hidden = [Image.new("L", (SIDE, SIDE), 0) for _ in range(n)]  # covered by a later shape
    for i in range(n - 2, -1, -1):
        hidden[i] = ImageChops.lighter(hidden[i + 1], masks[i + 1])

    def area(image: Image.Image) -> int:
        return image.histogram()[255]

    def shown(image: Image.Image, i: int) -> Image.Image:
        return ImageChops.darker(image, ImageChops.invert(hidden[i]))

    for i in range(n):  # every shape must stay recognisable
        if area(shown(masks[i], i)) < 0.35 * area(masks[i]):
            raise Retry

    def over(top: int, bottom: int) -> bool:
        if top <= bottom:
            return False
        return area(shown(ImageChops.darker(masks[top], masks[bottom]), top)) >= MIN_VISIBLE_OVERLAP

    relation = {(a, b): over(a, b) for a in range(n) for b in range(n) if a != b}
    options = []
    for a in range(n):
        for b in range(n):
            if a == b:
                continue
            middle = [x for x in range(n) if x not in (a, b) and relation[(x, a)] and relation[(b, x)]]
            if len(middle) == 1:
                # Prefer pairs where each clue alone is ambiguous.
                partial = sum(relation[(x, a)] != relation[(b, x)] for x in range(n) if x not in (a, b))
                options.append((partial, a, b, middle[0]))
    if not options:
        raise Retry
    best = max(o[0] for o in options)
    _, a, b, x = rng.choice([o for o in options if o[0] == best])
    question = (f"Какая фигура лежит поверх {phrase(*classes[a], 'gent')} и сама частично закрыта "
                f"{phrase(*classes[b], 'ablt')}? Назови её цвет и форму.")
    return Item("occlusion", tier, question, [[phrase(*classes[x], "nomn")]],
                picture(document([s.svg() for s in shapes])))


# ---------------------------------------------------------------- cords

CORD_TIERS = {1: dict(k=3, stages=3, swaps=3), 2: dict(k=5, stages=5, swaps=7), 3: dict(k=7, stages=7, swaps=12)}


def cords_item(rng: random.Random, tier: int) -> Item:
    spec = CORD_TIERS[tier]
    k, stages = spec["k"], spec["stages"]
    ys = [70 + j * (SIDE - 140) / (k - 1) for j in range(k)]
    xs = [100 + b * (412 - 100) / stages for b in range(stages + 1)]
    # Adjacent swaps per stage; never two swaps sharing a track inside one stage.
    plan: list[list[int]] = [[] for _ in range(stages)]
    total = 0
    while total < spec["swaps"]:
        stage = rng.randrange(stages)
        j = rng.randrange(k - 1)
        if any(abs(j - other) < 2 for other in plan[stage]):
            continue
        plan[stage].append(j)
        total += 1
    tracks = [[j] for j in range(k)]  # cord c starts at track c; tracks[c][b] = position at boundary b
    crossings = [0] * k
    order = list(range(k))  # order[position] = cord
    for swaps in plan:
        for j in swaps:
            crossings[order[j]] += 1
            crossings[order[j + 1]] += 1
            order[j], order[j + 1] = order[j + 1], order[j]
        for position, cord in enumerate(order):
            tracks[cord].append(position)
    end = {cord: tracks[cord][-1] for cord in range(k)}
    moved = [c for c in range(k) if end[c] != c]
    if not moved:
        raise Retry
    asked = max(moved, key=lambda c: (crossings[c], rng.random()))
    devices = rng.sample(SHAPES, k)
    sockets = rng.sample([c for c in COLORS if c != "серый"], k)
    segments = []
    for cord in range(k):
        y0 = ys[tracks[cord][0]]
        segments.append(f"M58 {y0:.1f}H{xs[0]:.1f}")
        for b in range(stages):
            ya, yb = ys[tracks[cord][b]], ys[tracks[cord][b + 1]]
            xm = (xs[b] + xs[b + 1]) / 2
            segments.append(f"M{xs[b]:.1f} {ya:.1f}C{xm:.1f} {ya:.1f} {xm:.1f} {yb:.1f} {xs[b + 1]:.1f} {yb:.1f}")
        segments.append(f"M{xs[-1]:.1f} {ys[end[cord]]:.1f}H440")
    rng.shuffle(segments)  # in source, a cord must be followed piece by piece
    body = [f'<path d="{d}" fill="none" stroke="#3a3f47" stroke-width="5" stroke-linecap="round"/>' for d in segments]
    for j in range(k):
        body.append(random_shape(rng, "серый", devices[j], 38, ys[j], 18).svg())
        body.append(f'<rect x="440" y="{ys[j] - 18:.1f}" width="36" height="36" rx="6" fill="{COLORS[sockets[j]]}" '
                    f'stroke="{OUTLINE}" stroke-width="3"/>')
        body.append(f'<circle cx="458" cy="{ys[j]:.1f}" r="6" fill="{OUTLINE}"/>')
    question = (f"К розетке какого цвета ведёт шнур от {phrase(None, devices[asked], 'gent')}? "
                "Слева серые приборы, справа цветные розетки. На пересечениях каждый шнур идёт своей линией. "
                "Ответь цветом.")
    return Item("cords", tier, question, [[sockets[end[asked]]]], picture(document(body)))


# ---------------------------------------------------------------- transform

OPS = {
    "поверни карточку на 90° по часовой стрелке": lambda r, c, n: (c, n - 1 - r),
    "поверни карточку на 90° против часовой стрелки": lambda r, c, n: (n - 1 - c, r),
    "поверни карточку на 180°": lambda r, c, n: (n - 1 - r, n - 1 - c),
    "отрази карточку слева направо, как в зеркале": lambda r, c, n: (r, n - 1 - c),
    "отрази карточку сверху вниз": lambda r, c, n: (n - 1 - r, c),
}
TRANSFORM_TIERS = {1: dict(n=3, ops=1), 2: dict(n=3, ops=2), 3: dict(n=4, ops=4)}


def apply(ops: list[str], cell: tuple[int, int], n: int) -> tuple[int, int]:
    for op in ops:
        cell = OPS[op](*cell, n)
    return cell


def transform_item(rng: random.Random, tier: int) -> Item:
    spec = TRANSFORM_TIERS[tier]
    n = spec["n"]
    classes = rng.sample([(c, k) for c in COLORS for k in SHAPES], n * n)
    grid = {(r, c): classes[r * n + c] for r in range(n) for c in range(n)}
    ops = [rng.choice(list(OPS)) for _ in range(spec["ops"])]

    def landing(sequence):
        return {apply(sequence, cell, n): obj for cell, obj in grid.items()}

    final = landing(ops)
    # Wrong but tempting readings: no move, only the last or first command, commands reversed.
    decoys = [landing(seq) for seq in ([], ops[-1:], ops[:1], ops[::-1]) if seq != ops]
    cells = [cell for cell in final if all(final[cell] != d[cell] for d in decoys)]
    if len(cells) < 2:
        raise Retry
    asked = rng.sample(cells, 2)
    card, size = 56, 400
    step = size / n
    body = [f'<rect x="{card}" y="{card}" width="{size}" height="{size}" rx="18" fill="#fffdf7" '
            f'stroke="{OUTLINE}" stroke-width="4"/>']
    for i in range(1, n):
        offset = card + i * step
        body.append(f'<path d="M{card} {offset:.1f}H{card + size} M{offset:.1f} {card}V{card + size}" '
                    'stroke="#b9b2a5" stroke-width="2"/>')
    for (r, c), (color, kind) in grid.items():
        body.append(random_shape(rng, color, kind, card + (c + 0.5) * step, card + (r + 0.5) * step,
                                 step * 0.32).svg())
    steps = ", затем ".join(ops)
    where = " и ".join(f"({r + 1},{c + 1})" for r, c in asked)
    question = (f"Мысленно {steps}. Какие фигуры окажутся в клетках {where}? Клетки заданы как (строка,столбец), "
                "строки считаются сверху, столбцы слева, с единицы. Назови цвет и форму обеих через запятую, в этом порядке.")
    answers = [[phrase(*final[cell], "nomn")] for cell in asked]
    return Item("transform", tier, question, answers, picture(document(body)))


BUILDERS = {"count": count_item, "occlusion": occlusion_item, "cords": cords_item, "transform": transform_item}


def generate(seed: int, per_tier: int = 2) -> list[Item]:
    items = []
    for kind in KINDS:
        for tier in TIERS:
            for k in range(per_tier):
                rng = random.Random(f"{GENERATOR_VERSION}-{seed}-{kind}-{tier}-{k}")
                while True:
                    try:
                        items.append(BUILDERS[kind](rng, tier))
                        break
                    except Retry:
                        continue
    return items


def legend(mode: str) -> str:
    shapes = ", ".join(SHAPES)
    if mode == "code":
        colors = ", ".join(f"{name} {code}" for name, code in COLORS.items())
    else:
        colors = ", ".join(COLORS)
    return (f"Фигуры на рисунках: {shapes}. Цвета: {colors}. Контур у всех фигур тёмный. "
            "Называй фигуру цветом и формой, например «синий ромб».")
