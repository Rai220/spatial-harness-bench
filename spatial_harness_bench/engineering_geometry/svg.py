"""Serialize and analytically decode the pilot's restricted SVG format."""

from dataclasses import dataclass
from decimal import Context, Decimal, InvalidOperation, localcontext
from math import atan2, hypot, pi, sqrt, tau
import re
from xml.etree import ElementTree as ET

from .model import BOUNDARY_EPS, FILLS, HALF_VIEW, Circle, Cable


SVG_NS = "{http://www.w3.org/2000/svg}"
MAX_SVG_BYTES = 64 * 1024
ARC_EPS = BOUNDARY_EPS / 4
NUMBER = r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?"
TOKEN = re.compile(rf"{NUMBER}|[MmAaZz]")
WS = " \t\n\r"
SEPARATOR = re.compile(r"[ \t\n\r]*,?[ \t\n\r]*")
VIEWBOX = re.compile(
    r"[ \t\n\r]*"
    + r"(?:[ \t\n\r]+|[ \t\n\r]*,[ \t\n\r]*)".join(f"({NUMBER})" for _ in range(4))
    + r"[ \t\n\r]*"
)


class SVGError(ValueError):
    pass


@dataclass(frozen=True)
class Regions:
    outer: Circle
    inner: Circle
    insulation: Circle
    holes: tuple[Circle, ...]
    cores: dict[str, Circle]


def _fmt(value: float) -> str:
    return f"{value:.9f}".rstrip("0").rstrip(".") or "0"


def _contour(circle: Circle) -> str:
    # Four quarter arcs avoid poorly conditioned endpoint-to-centre conversion
    # for a rounded, nearly diametral chord.
    x, y, r = circle.x, circle.y, circle.r
    start = (x + r, y)
    points = [(x, y + r), (x - r, y), (x, y - r), start]
    d = f"M {_fmt(start[0])} {_fmt(start[1])}"
    for px, py in points:
        d += f" A {_fmt(r)} {_fmt(r)} 0 0 1 {_fmt(px)} {_fmt(py)}"
    return d + " Z"


def render_svg(cable: Cable) -> bytes:
    lines = ['<svg xmlns="http://www.w3.org/2000/svg" width="100mm" height="100mm" viewBox="-50 -50 100 100">']
    for name, contours in (
        ("shell", (cable.outer, cable.inner)),
        ("insulation", (cable.inner, *(c for _, c in cable.cores))),
    ):
        d = " ".join(map(_contour, contours))
        lines.append(f'  <path id="{name}" fill="{FILLS[name]}" fill-rule="evenodd" d="{d}"/>')
    for name, c in cable.cores:
        lines.append(f'  <circle id="{name}" fill="{FILLS[name]}" cx="{_fmt(c.x)}" cy="{_fmt(c.y)}" r="{_fmt(c.r)}"/>')
    return ("\n".join([*lines, "</svg>"]) + "\n").encode("ascii")


def _number(text: str) -> float:
    if not re.fullmatch(NUMBER, text):
        raise SVGError("invalid SVG number")
    value = float(text)
    if not -1000 <= value <= 1000:  # includes non-finite values
        raise SVGError("SVG number out of range")
    return value


def _tokens(source: str) -> list[str]:
    """Tokenize without silently dropping unknown syntax or malformed commas."""
    result = []
    end = 0
    for match in TOKEN.finditer(source):
        token = match[0]
        sep = source[end:match.start()]
        if not SEPARATOR.fullmatch(sep) or (
            "," in sep and (not result or result[-1].isalpha() or token.isalpha())
        ):
            raise SVGError("invalid path syntax")
        result.append(token)
        end = match.end()
    if source[end:].strip(WS):
        raise SVGError("invalid path syntax")
    return result


def _precise_number(text: str) -> Decimal:
    value = _number(text)
    # Values below this precision cannot affect the circle tolerances. Dropping
    # them also bounds the decimal arithmetic for very small exponents.
    return Decimal(text) if abs(value) >= 1e-24 else Decimal(0)


def _arc(start: tuple[Decimal, Decimal], end: tuple[Decimal, Decimal], radius: Decimal,
         large: str, sweep: str) -> tuple[Circle, float]:
    """SVG endpoint-to-centre conversion specialized to equal radii."""
    dx, dy = (start[0] - end[0]) / 2, (start[1] - end[1]) / 2
    half_squared = dx * dx + dy * dy
    half, r = sqrt(float(half_squared)), float(radius)
    if half <= 1e-9 or half - r > ARC_EPS:
        raise SVGError("degenerate arc or implicit radius scaling")
    # SVG scales an undersized radius to half the chord; permit only rounding.
    # Compute the difference before converting to float: an exact diameter
    # must not pick up a spurious perpendicular offset from cancellation.
    height = sqrt(max(0.0, float(radius * radius - half_squared)))
    signed_height = height if large != sweep else -height
    c = Circle(float((start[0] + end[0]) / 2) + float(dy) / half * signed_height,
               float((start[1] + end[1]) / 2) - float(dx) / half * signed_height, max(r, half))
    angle = 2 * atan2(half, height)
    return c, tau - angle if large == "1" else angle


def _circles(d: str, maximum: int) -> tuple[Circle, ...]:
    tokens = iter(_tokens(d))
    rings = []
    position = (Decimal(0), Decimal(0))

    def end_point(cmd: str) -> tuple[Decimal, Decimal]:
        x, y = (_precise_number(next(tokens, "")), _precise_number(next(tokens, "")))
        if cmd.islower():
            x, y = x + position[0], y + position[1]
        if max(abs(x), abs(y)) > 1000:
            raise SVGError("path position out of range")
        return x, y

    while (cmd := next(tokens, None)) is not None:
        if cmd not in ("M", "m") or len(rings) == maximum:
            raise SVGError("expected a circular subpath beginning with M/m")
        position = start = end_point(cmd)
        carrier = None
        direction = None
        angle = 0.0
        count = 0
        while (cmd := next(tokens, None)) in ("A", "a"):
            rx, ry = (_precise_number(next(tokens, "")), _precise_number(next(tokens, "")))
            _number(next(tokens, ""))  # axis rotation has no effect on a circle
            large, sweep = next(tokens, ""), next(tokens, "")
            if rx != ry or rx <= 0 or large not in ("0", "1") or sweep not in ("0", "1"):
                raise SVGError("expected circular arc with explicit 0/1 flags")
            end = end_point(cmd)
            circle, span = _arc(position, end, rx, large, sweep)
            if carrier is None:
                carrier, direction = circle, sweep
            elif sweep != direction or boundary_delta(circle, carrier) > ARC_EPS:
                raise SVGError("arcs do not share a circle and direction")
            position = end
            angle += span
            count += 1
        if (cmd not in ("Z", "z") or count < 2 or carrier is None or
                hypot(position[0] - start[0], position[1] - start[1]) > 1e-9 or
                abs(angle - 2 * pi) * carrier.r > 4 * count * ARC_EPS):
            raise SVGError("a contour must explicitly close after exactly one full turn")
        position = start
        rings.append(carrier)
    if not rings:
        raise SVGError("empty circular path")
    return tuple(rings)


def boundary_delta(a: Circle, b: Circle) -> float:
    """Hausdorff distance between two complete circle boundaries."""
    return a.distance(b) + abs(a.r - b.r)


def parse_svg(data: bytes) -> Regions:
    if len(data) > MAX_SVG_BYTES:
        raise SVGError("SVG is too large")
    try:
        source = data.decode("utf-8-sig")
        if ("\x00" in source or "<!DOCTYPE" in source or "<!ENTITY" in source or
                re.search(r"<\?(?!xml[ \t\r\n])", source) or
                re.match(r"""<\?xml[^?]*encoding\s*=\s*['"](?!utf-8['"])""", source, re.I)):
            raise SVGError("DTD, processing instructions or non-UTF-8 encoding are not permitted")
        root = ET.fromstring(source)
    except (ET.ParseError, UnicodeError, RecursionError) as exc:
        raise SVGError("invalid UTF-8 SVG XML") from exc
    if root.tag != SVG_NS + "svg" or set(root.attrib) != {"width", "height", "viewBox"}:
        raise SVGError("unsupported SVG root")
    for dim in ("width", "height"):
        raw = root.attrib[dim].strip(WS)
        if not raw.endswith("mm") or _number(raw[:-2]) != 2 * HALF_VIEW:
            raise SVGError("incorrect physical dimensions in mm")
    box = VIEWBOX.fullmatch(root.attrib["viewBox"])
    if not box or tuple(map(_number, box.groups())) != (-HALF_VIEW, -HALF_VIEW, 2 * HALF_VIEW, 2 * HALF_VIEW):
        raise SVGError("incorrect SVG viewBox")

    def whitespace_only(value: str | None) -> bool:
        return not value or not value.strip(WS)

    if not whitespace_only(root.text):
        raise SVGError("unsupported SVG text")
    parts: dict[str, tuple[Circle, ...]] = {}
    for el in root:
        name = el.get("id", "")
        if name not in FILLS or name in parts or el.get("fill", "").lower() != FILLS[name]:
            raise SVGError("unknown, duplicate or incorrectly colored material")
        if len(el) or not whitespace_only(el.text) or not whitespace_only(el.tail):
            raise SVGError("unsupported SVG content")
        attrs = set(el.attrib) - {"id", "fill"}
        if el.tag == SVG_NS + "path":
            if attrs not in ({"d"}, {"d", "fill-rule"}):
                raise SVGError("unsupported path attributes")
            rule = el.get("fill-rule")
            if (rule not in (None, "evenodd") or
                    (name in ("shell", "insulation") and rule != "evenodd")):
                raise SVGError("shell and insulation paths require fill-rule=evenodd")
            maximum = {"shell": 2, "insulation": 5}.get(name, 1)
            try:
                with localcontext(Context(prec=40)):
                    parts[name] = _circles(el.attrib["d"], maximum)
            except InvalidOperation as exc:
                raise SVGError("invalid path coordinate") from exc
        elif el.tag == SVG_NS + "circle" and name not in ("shell", "insulation") and attrs == {"cx", "cy", "r"}:
            c = Circle(*(_number(el.attrib[key].strip(WS)) for key in ("cx", "cy", "r")))
            if c.r <= 0:
                raise SVGError("circle radius must be positive")
            parts[name] = (c,)
        else:
            raise SVGError("unsupported material element or attributes")
    if len(parts.get("shell", ())) != 2 or "insulation" not in parts:
        raise SVGError("missing shell contours or insulation")
    outer, inner = sorted(parts.pop("shell"), key=lambda c: c.r, reverse=True)
    insulation, *holes = sorted(parts.pop("insulation"), key=lambda c: c.r, reverse=True)
    return Regions(outer, inner, insulation, tuple(holes), {name: rings[0] for name, rings in parts.items()})
