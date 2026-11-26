"""Trusted example construction, not used by the verifier or the runner."""

from dataclasses import replace

from .model import Cable, Circle
from .task import Addition, Instance, Radii, Removal, Spacing, Thickness


def construct(instance: Instance) -> Cable:
    """Construct the requested dimensions; impossible requests may yield invalid geometry."""
    original = instance.original
    cores = original.core_map()
    match instance.edit:
        case Thickness(value, "inner"):
            return replace(original, outer=replace(original.outer, r=original.inner.r + value))
        case Thickness(value, "outer"):
            return replace(original, inner=replace(original.inner, r=original.outer.r - value))
        case Radii(values):
            for key, value in zip(cores, values, strict=True):
                cores[key] = replace(cores[key], r=value)
        case Spacing(side):
            gx, gy = original.centroid()
            a, b, _ = cores.values()
            factor = side / a.distance(b)
            for key, c in cores.items():
                cores[key] = Circle(gx + (c.x - gx) * factor, gy + (c.y - gy) * factor, c.r)
        case Addition(radius, anchor, percent):
            c = cores[anchor]
            gx, gy = original.centroid()
            weight = percent / 100
            cores["core-d"] = Circle((1 - weight) * gx + weight * c.x,
                                     (1 - weight) * gy + weight * c.y, radius)
        case Removal(core_id):
            del cores[core_id]
    return replace(original, cores=tuple(cores.items()))
