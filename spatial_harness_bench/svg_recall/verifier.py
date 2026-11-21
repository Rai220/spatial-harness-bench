"""Inspect the XML before rendering, then check the same pixels in both modes."""

from dataclasses import dataclass
from io import BytesIO
import math
import re
import subprocess
from xml.etree import ElementTree as ET

import cairosvg
import cairocffi
from defusedxml.ElementTree import DefusedXMLParser
from defusedxml.common import DefusedXmlException
from PIL import Image, ImageColor

from ..answers import expand
from ..ascii_recall.verifier import DrawingIssue, caption_keys, compact, normalize

SVG = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG)
SIDE = 512
BLANK_SVG = b'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512"/>'
NUM = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
ID = r"[a-zA-Z]\d+"
LENGTH = re.compile(rf"{NUM}(?:px|pt|mm|cm|in|%)?")
UNIT_PIXELS = {"": 1, "px": 1, "pt": 96 / 72, "mm": 96 / 25.4, "cm": 96 / 2.54, "in": 96, "%": SIDE / 100}
TAGS = set("svg g path rect circle ellipse line polyline polygon defs linearGradient radialGradient stop clipPath".split())
LENGTHS = set("""x y x1 x2 y1 y2 cx cy r rx ry fx fy width height stroke-width
    stroke-dashoffset stroke-miterlimit opacity fill-opacity stroke-opacity offset stop-opacity""".split())
CHOICES = {
    "stroke-linecap": {"butt", "round", "square"}, "stroke-linejoin": {"round", "bevel", "miter"},
    "fill-rule": {"nonzero", "evenodd"}, "clip-rule": {"nonzero", "evenodd"},
    "gradientUnits": {"userSpaceOnUse", "objectBoundingBox"},
    "clipPathUnits": {"userSpaceOnUse", "objectBoundingBox"},
    "spreadMethod": {"pad", "reflect", "repeat"}, "version": {"1.0", "1.1", "2.0"},
}
PAINT = {"fill", "stroke", "color", "stop-color"}
COLORS = set(ImageColor.colormap) | {"none", "transparent", "currentcolor"}
TRANSFORM = re.compile(r"(?:(?:matrix|translate|scale|rotate|skewX|skewY)\s*\([\s,\d.eE+\-]+\)[\s,]*)+")
STYLE_KEYS = PAINT | LENGTHS | set(CHOICES) | {"clip-path", "stroke-dasharray"}
BLANK_ISSUE = DrawingIssue("invalid_art", "На белом фоне нет различимого рисунка.")


@dataclass
class Picture:
    source: bytes
    png: bytes
    issue: DrawingIssue | None = None


def white_png() -> bytes:
    out = BytesIO()
    Image.new("RGB", (SIDE, SIDE), "white").save(out, format="PNG")
    return out.getvalue()


BLANK_PNG = white_png()


class Caption(ValueError):
    pass


class Builder(ET.TreeBuilder):
    def __init__(self, names: list[str]):
        super().__init__()
        self.keys = caption_keys(names)

    def start_ns(self, prefix, uri):
        if uri == SVG and prefix in {"", "svg"}:
            return
        spelling = compact(f"{prefix} {uri}")
        for key, name in self.keys.items():
            if key in spelling:
                raise Caption(f"Название в объявлении пространства имён: {name}")

    def data(self, text):
        if text.strip():
            raise Caption("Текст внутри SVG.")
        super().data(text)

    # Also called for comments/processing instructions outside the root.
    def comment(self, text):
        if text.strip():
            raise Caption("Текст в XML-комментарии.")

    def pi(self, target, text):
        raise ValueError("Инструкции обработки XML не поддерживаются.")


def check_attribute(key: str, value: str) -> bool:
    if key == "id" or key == "class":
        if not re.fullmatch(ID, value):
            raise Caption(f"Текстовая метка в {key}.")
        return True
    if key.startswith(("aria-", "data-")) or key in {"label", "title", "alt", "name"}:
        if value:
            raise Caption(f"Текстовая метка в {key}.")
        return False
    if key in LENGTHS:
        return LENGTH.fullmatch(value) is not None
    if key in CHOICES:
        return value in CHOICES[key]
    if key == "viewBox":
        parts = re.split(r"[\s,]+", value)
        return (len(parts) == 4 and all(re.fullmatch(NUM, p) and math.isfinite(float(p)) for p in parts)
                and float(parts[2]) > 0 and float(parts[3]) > 0)
    if key == "preserveAspectRatio":
        return bool(re.fullmatch(r"(?:none|x(?:Min|Mid|Max)Y(?:Min|Mid|Max)(?:\s+(?:meet|slice))?)", value))
    if key == "d":
        return bool(re.fullmatch(r"[MmZzLlHhVvCcSsQqTtAa0-9eE.,\s+\-]+", value))
    if key in {"points", "stroke-dasharray"}:
        return bool(re.fullmatch(r"[\d+\-.,%\seE]+", value) or (key == "stroke-dasharray" and value == "none"))
    if key in {"transform", "gradientTransform"}:
        return bool(TRANSFORM.fullmatch(value))
    if key == "clip-path" or key in PAINT:
        if re.fullmatch(rf"url\(#{ID}\)", value):
            return True
        if key in PAINT:
            return bool(value.lower() in COLORS or re.fullmatch(r"#(?:[\da-fA-F]{3,4}|[\da-fA-F]{6}|[\da-fA-F]{8})", value)
                        or re.fullmatch(r"rgba?\([\d.,%\s]+\)", value))
        return value == "none"
    return False


def style_allowed(value: str) -> bool:
    if "/*" in value:
        raise Caption("Текст в CSS-комментарии.")
    declarations = [part.partition(":") for part in value.split(";") if part.strip()]
    return bool(declarations) and all(
        colon and prop.strip() in STYLE_KEYS and check_attribute(prop.strip(), val.strip())
        for prop, colon, val in declarations
    )


def checked_source(data: bytes, names: list[str]) -> bytes:
    if len(data) > 512_000:
        raise ValueError("SVG превышает 512 КБ.")
    parser = DefusedXMLParser(target=Builder(names), forbid_dtd=True)
    parser.feed(data.decode("utf-8-sig"))
    root = parser.close()
    invalid = []
    for node in root.iter():
        tag = node.tag.removeprefix(f"{{{SVG}}}")
        if tag not in TAGS:
            invalid.append((tag, f"Неподдерживаемый элемент {tag}."))
        for key, raw in list(node.attrib.items()):
            value = raw.strip()
            attr = key.rsplit("}", 1)[-1]
            allowed = style_allowed(value) if key == "style" else check_attribute(attr, value)
            if not allowed or key.startswith("{"):
                invalid.append((f"{key} {value}", f"Неподдерживаемый атрибут или значение {key}."))
    if root.tag not in {f"{{{SVG}}}svg", "svg"}:
        invalid.append(("", "Нужен корневой элемент svg."))
    if invalid:
        keys = caption_keys(names)
        for raw, _ in invalid:
            for key, name in keys.items():
                if key in compact(raw):
                    raise Caption(f"Название в коде SVG: {name}")
        raise ValueError(invalid[0][1])
    if "viewBox" not in root.attrib:
        size = []
        for axis in ("width", "height"):
            length = root.get(axis, str(SIDE)).strip()
            number = re.match(NUM, length).group()
            pixels = float(number) * UNIT_PIXELS[length[len(number):]]
            if not math.isfinite(pixels) or pixels <= 0:
                raise ValueError("Размер холста должен быть конечным и положительным.")
            size.append(str(pixels))
        root.set("viewBox", "0 0 " + " ".join(size))
    # Drop unused namespace declarations, XML headers and inter-element whitespace.
    for node in root.iter():
        node.text = node.tail = None
    return ET.tostring(root, encoding="utf-8")


def pixel_caption(png: bytes, names: list[str]) -> DrawingIssue | None:
    process = subprocess.run(["tesseract", "stdin", "stdout", "-l", "rus+eng", "--psm", "11", "tsv"],
                             input=png, capture_output=True, check=True)
    keys = {normalize(name) for name in expand(names)}
    for line in process.stdout.decode("utf-8").splitlines()[1:]:
        fields = line.split("\t", 11)
        if len(fields) != 12 or fields[0] != "5":
            continue
        word = fields[11].strip(" .,:;!?()[]{}«»\"'\\|_-")
        confidence = float(fields[10])
        if ((normalize(word) in keys and confidence >= 55) or
                (re.fullmatch(r"[^\W\d_]{2,}", word) and confidence >= 85)):
            return DrawingIssue("caption", f"Текст на отрисованном изображении (OCR): {word}")
    return None


def inspect(data: bytes, names: list[str]) -> Picture:
    try:
        source = checked_source(data, names)
        png = cairosvg.svg2png(bytestring=source, output_width=SIDE, output_height=SIDE, background_color="white")
        with Image.open(BytesIO(png)) as image:
            uniform = all(lo == hi for lo, hi in image.convert("RGB").getextrema())
        issue = BLANK_ISSUE if uniform else pixel_caption(png, names)
    except Caption as exc:
        issue = DrawingIssue("caption", str(exc))
    except (ET.ParseError, DefusedXmlException, ValueError, ArithmeticError, RecursionError,
            cairocffi.CairoError, IndexError) as exc:
        issue = DrawingIssue("invalid_art", str(exc))
    if issue is not None:
        return Picture(BLANK_SVG, BLANK_PNG, issue)
    return Picture(source, png)
