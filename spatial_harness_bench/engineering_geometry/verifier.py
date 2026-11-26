"""Grade actual SVG geometry using independent predicates, never a gold SVG diff."""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations
import json
from math import hypot, sqrt

from .model import BOUNDARY_EPS, DIM_EPS, HALF_VIEW, Circle
from .svg import Regions, SVGError, boundary_delta, parse_svg
from .task import Addition, Instance, Radii, Removal, Spacing, Thickness


MAX_STATUS_BYTES = 4096


@dataclass(frozen=True)
class Verdict:
    passed: bool
    code: str
    detail: str
    submission_status: str | None = None


def _centroid(cores: Sequence[Circle]) -> tuple[float, float]:
    return (sum(c.x for c in cores) / len(cores), sum(c.y for c in cores) / len(cores))


def _inside_view(circle: Circle) -> bool:
    return max(abs(circle.x), abs(circle.y)) + circle.r <= HALF_VIEW


def conflict(instance: Instance) -> tuple[str, str] | None:
    """Prove a contradiction from necessary bounds and the trusted input.

    The v1 constraints fix all centres/radii uniquely. Proof is independent
    of the submitted geometry and does not construct or read any gold SVG.
    """
    original = instance.original
    cores = original.core_map()
    match instance.edit:
        case Thickness(value, keep):
            if value <= 0 or (keep == "outer" and value >= original.outer.r):
                return "shell_thickness", "The mandated annulus cannot have positive inner radius and thickness."
            if keep == "inner":
                if max(abs(original.inner.x), abs(original.inner.y)) + original.inner.r + value > HALF_VIEW:
                    return "outside_view", "The mandatory outer shell exceeds the viewport."
            else:
                limit = original.outer.r - value
                for name, c in cores.items():
                    if c.distance(original.outer) + c.r >= limit:
                        return "core_outside_insulation", f"{name} needs more clearance than the mandatory inner radius."
        case Radii(values):
            for (name, c), r in zip(cores.items(), values, strict=True):
                if r <= 0:
                    return "nonpositive_radius", f"{name} requires a nonpositive radius."
                if c.distance(original.inner) + r >= original.inner.r:
                    return "core_outside_insulation", f"{name} cannot fit inside the fixed shell at its fixed centre."
            for ((name_a, a), ra), ((name_b, b), rb) in combinations(zip(cores.items(), values, strict=True), 2):
                if ra + rb >= a.distance(b):
                    return "core_overlap", f"{name_a} and {name_b}: fixed centre distance is no greater than the required sum of radii."
        case Spacing(side):
            for (name_a, a), (name_b, b) in combinations(cores.items(), 2):
                if a.r + b.r >= side:
                    return "core_overlap", f"{name_a} and {name_b}: required centre distance is no greater than the fixed sum of radii."
            gx, gy = _centroid(tuple(cores.values()))
            for name, c in cores.items():
                dx, dy = c.x - gx, c.y - gy
                scale = side / (sqrt(3) * hypot(dx, dy))
                if hypot(gx - original.inner.x + scale * dx,
                         gy - original.inner.y + scale * dy) + c.r >= original.inner.r:
                    return "core_outside_insulation", f"{name} cannot fit at the mandatory circumradius and direction."
        case Addition(radius, anchor, percent):
            if radius <= 0:
                return "nonpositive_radius", "The new core requires a nonpositive radius."
            gx, gy = _centroid(tuple(cores.values()))
            base = cores[anchor]
            t = percent / 100
            x, y = gx + t * (base.x - gx), gy + t * (base.y - gy)
            disk = Circle(x, y, radius)
            if disk.distance(original.inner) + radius >= original.inner.r:
                return "core_outside_insulation", "The mandated new disk does not fit inside the insulation boundary."
            for name, c in cores.items():
                if disk.distance(c) <= radius + c.r:
                    return "core_overlap", f"The mandated position cannot separate core-d from {name}."
    return None


def _topology(regions: Regions) -> tuple[str, str] | None:
    outer, inner, insulation, holes, cores = (
        regions.outer, regions.inner, regions.insulation, regions.holes, regions.cores
    )
    if not _inside_view(outer):
        return "outside_view", "The external boundary exceeds the viewport."
    if outer.distance(inner) > BOUNDARY_EPS or outer.r - inner.r <= BOUNDARY_EPS:
        return "invalid_shell", "Shell contours must be concentric and strictly nested."
    if boundary_delta(inner, insulation) > BOUNDARY_EPS:
        return "shared_boundary", "Shell and insulation contours do not coincide."
    for name, core in cores.items():
        if any(core.distance(cavity) + core.r >= cavity.r - BOUNDARY_EPS for cavity in (inner, insulation)):
            return "core_outside_insulation", f"{name} is not strictly inside the insulation."
    for (name_a, a), (name_b, b) in combinations(cores.items(), 2):
        if a.distance(b) <= a.r + b.r + BOUNDARY_EPS:
            return "core_overlap", f"{name_a} and {name_b} touch or overlap."
    unmatched = list(holes)
    for name, c in cores.items():
        matches = [i for i, h in enumerate(unmatched) if boundary_delta(c, h) <= BOUNDARY_EPS]
        if len(matches) != 1:
            return "shared_boundary", f"{name} needs exactly one coincident insulation hole."
        unmatched.pop(matches[0])
    if unmatched:
        return "shared_boundary", "There is an unfilled insulation hole."
    for h in holes:
        if h.distance(insulation) + h.r >= insulation.r - BOUNDARY_EPS:
            return "hole_outside_insulation", "A hole is not strictly inside the insulation."
    for a, b in combinations(holes, 2):
        if a.distance(b) <= a.r + b.r + BOUNDARY_EPS:
            return "hole_overlap", "Insulation holes touch or overlap."
    return None


def _request(instance: Instance, regions: Regions) -> tuple[str, str] | None:
    original = instance.original
    before, after = original.core_map(), regions.cores
    edit = instance.edit
    names = set(before)
    if isinstance(edit, Removal):
        names.remove(edit.core_id)
    if isinstance(edit, Addition):
        names.add("core-d")
    if set(after) != names:
        return "wrong_core_ids", "Core identities do not match the requested edit."

    for side, actual, previous in (
        ("inner", regions.inner, original.inner), ("outer", regions.outer, original.outer)
    ):
        if isinstance(edit, Thickness) and side != edit.keep:
            continue
        if boundary_delta(actual, previous) > DIM_EPS:
            return "changed_shell", f"The {side} shell boundary was required to stay fixed."

    for name in sorted(before.keys() & after.keys()):
        if not isinstance(edit, Spacing) and before[name].distance(after[name]) > DIM_EPS:
            return "changed_center", f"The centre of {name} was required to stay fixed."
        if not isinstance(edit, Radii) and abs(before[name].r - after[name].r) > DIM_EPS:
            return "changed_radius", f"The radius of {name} was required to stay fixed."

    match edit:
        case Thickness(value, _):
            if abs(regions.outer.r - regions.inner.r - value) > DIM_EPS:
                return "wrong_thickness", "The thickness is not the requested value."
        case Radii(values):
            for name, radius in zip(before, values, strict=True):
                if abs(after[name].r - radius) > DIM_EPS:
                    return "wrong_radius", f"{name} has the wrong radius."
        case Spacing(side):
            gx, gy = _centroid(tuple(before.values()))
            ax, ay = _centroid(tuple(after.values()))
            if hypot(gx - ax, gy - ay) > DIM_EPS:
                return "changed_centroid", "The common centre was required to stay fixed."
            if any(abs(a.distance(b) - side) > DIM_EPS for a, b in combinations(after.values(), 2)):
                return "wrong_spacing", "All three centre distances must match the requested side length."
            for name, c in before.items():
                dx, dy = c.x - gx, c.y - gy
                x, y = after[name].x - gx, after[name].y - gy
                if x * dx + y * dy <= 0 or abs(x * dy - y * dx) / hypot(dx, dy) > DIM_EPS:
                    return "changed_ray", f"{name} must stay on its original ray."
        case Addition(radius, anchor, percent):
            c = after["core-d"]
            reference = before[anchor]
            gx, gy = _centroid(tuple(before.values()))
            t = percent / 100
            if hypot(c.x - (gx + t * (reference.x - gx)), c.y - (gy + t * (reference.y - gy))) > DIM_EPS:
                return "wrong_added_center", "The new core is not in the requested position."
            if abs(c.r - radius) > DIM_EPS:
                return "wrong_added_radius", "The new core does not have the requested radius."
    return None


def _status(data: bytes | None) -> str | None:
    if data is None or len(data) > MAX_STATUS_BYTES:
        return None

    def object_from_pairs(pairs):
        if len(pairs) != 1 or pairs[0][0] != "status":
            raise ValueError("status object must have exactly one field")
        return dict(pairs)

    try:
        value = json.loads(data.decode("utf-8"), object_pairs_hook=object_from_pairs)
    except (ValueError, UnicodeError, RecursionError):
        return None
    if isinstance(value, dict) and value["status"] in ("solved", "infeasible"):
        return value["status"]
    return None


def verify(instance: Instance, output_svg: bytes | None, status_json: bytes | None) -> Verdict:
    status = _status(status_json)
    if status is None:
        return Verdict(False, "invalid_status", "Provide status.json with exactly one allowed status.")
    proof = conflict(instance)
    if status == "infeasible":
        if output_svg is not None:
            return Verdict(False, "unexpected_output", "Do not submit an SVG with an infeasible status.", status)
        if not proof:
            return Verdict(False, "false_refusal", "The requested edit is feasible.", status)
        return Verdict(True, proof[0], proof[1], status)
    if proof:
        return Verdict(False, "false_success", f"The request is infeasible: {proof[1]}", status)
    if output_svg is None:
        return Verdict(False, "missing_svg", "Provide output.svg for a solved status.", status)
    try:
        regions = parse_svg(output_svg)
    except SVGError as exc:
        return Verdict(False, "invalid_svg", str(exc), status)
    if issue := _topology(regions) or _request(instance, regions):
        return Verdict(False, *issue, status)
    return Verdict(True, "accepted", "Geometry and requested edit verified.", status)
