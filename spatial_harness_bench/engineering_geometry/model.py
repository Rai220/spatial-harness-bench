"""Material boundaries in millimetres (SVG axes: right and down)."""

from dataclasses import dataclass
from math import hypot


TASK_ID = "engineering-geometry-cable-v1"
HALF_VIEW = 50
# Task dimensions and shared boundaries are deliberately checked separately.
DIM_EPS = 1e-5
BOUNDARY_EPS = 1e-6
CORE_IDS = ("core-a", "core-b", "core-c", "core-d")
FILLS = {"shell": "#687582", "insulation": "#f1ddb2",
         **dict.fromkeys(CORE_IDS, "#b66038")}


@dataclass(frozen=True)
class Circle:
    x: float
    y: float
    r: float

    def distance(self, other: "Circle") -> float:
        return hypot(self.x - other.x, self.y - other.y)


@dataclass(frozen=True)
class Cable:
    outer: Circle
    inner: Circle
    cores: tuple[tuple[str, Circle], ...]

    def core_map(self) -> dict[str, Circle]:
        return dict(self.cores)

    def centroid(self) -> tuple[float, float]:
        return (sum(c.x for _, c in self.cores) / len(self.cores),
                sum(c.y for _, c in self.cores) / len(self.cores))
