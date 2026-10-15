"""Data model, input parsing and orientation clustering for the layout solver."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from . import geom
from .geom import Point, Poly

KIND_PRIORITY = {"fridge": 0, "iceMaker": 1, "shelf": 2, "overShelf": 3}
KIND_LABEL = {"fridge": "冰箱", "iceMaker": "制冰机", "shelf": "货架", "overShelf": "离地架", "unknown": "未知"}


def item_kind(name: str) -> str:
    n = name.lower()
    if "fridge" in n:
        return "fridge"
    if "icemaker" in n:
        return "iceMaker"
    if "overshelf" in n:
        return "overShelf"
    if "shelf" in n:
        return "shelf"
    return "unknown"


@dataclass
class Item:
    name: str
    kind: str
    length: float
    width: float

    @property
    def area(self) -> float:
        return self.length * self.width


@dataclass
class Door:
    p0: Point
    p1: Point
    width: float
    is_open_inward: bool
    inward: Point  # unit normal pointing into the room

    def zone_corners(self) -> Optional[Poly]:
        """N x N square inside the door when it opens inward, else None."""
        if not self.is_open_inward:
            return None
        n = self.inward
        w = self.width
        return [self.p0, self.p1, geom.add(self.p1, geom.mul(n, w)), geom.add(self.p0, geom.mul(n, w))]

    def segment(self) -> Tuple[Point, Point]:
        return (self.p0, self.p1)


@dataclass
class Room:
    raw_boundary: List[Point]
    poly: List[Point]
    cleanup_report: List[str]
    area: float
    bbox: Tuple[float, float, float, float]
    allowed_angles: List[float]      # item angles allowed by H3
    orientation_reps: List[float]    # representative directions (mod 90)

    def edges(self) -> List[Tuple[Point, Point]]:
        n = len(self.poly)
        return [(self.poly[i], self.poly[(i + 1) % n]) for i in range(n)]


@dataclass
class Problem:
    room: Room
    door: Door
    items: List[Item]
    source: str = ""


# --- orientation clustering -------------------------------------------------
def _weighted_mean(angles: List[float], weights: List[float]) -> float:
    if not angles:
        return 0.0
    s = sum(weights)
    return sum(a * w for a, w in zip(angles, weights)) / s if s else 0.0


def _mod90_dist(a: float, b: float) -> float:
    d = abs(a - b) % 90.0
    return min(d, 90.0 - d)


def orientation_angles(poly: Sequence[Point], min_wall_len: float = 500.0,
                       cluster_tol: float = geom.CLUSTER_TOL,
                       axis_snap: float = geom.AXIS_SNAP_TOL) -> Tuple[List[float], List[float]]:
    """Allowed item angles (H3) and the representative directions (mod 90).

    Edge directions are reduced mod 90, snapped to 0 when within ``axis_snap``
    of an axis, clustered (circular, ``cluster_tol`` apart), and a cluster is
    kept only when its total wall length is at least ``min_wall_len``.
    """
    entries: List[Tuple[float, float]] = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        ang = geom.angle_of(a, b) % 90.0
        if ang <= axis_snap or ang >= 90.0 - axis_snap:
            ang = 0.0
        entries.append((ang, geom.dist(a, b)))

    entries.sort(key=lambda t: t[0])
    clusters: List[Dict[str, List[float]]] = []
    for ang, L in entries:
        placed = False
        for c in clusters:
            rep = _weighted_mean(c["angles"], c["weights"])
            if _mod90_dist(ang, rep) <= cluster_tol:
                c["angles"].append(ang)
                c["weights"].append(L)
                placed = True
                break
        if not placed:
            clusters.append({"angles": [ang], "weights": [L]})

    reps: List[float] = []
    for c in clusters:
        if sum(c["weights"]) >= min_wall_len:
            reps.append(_weighted_mean(c["angles"], c["weights"]))
    reps = sorted({round(r, 6) for r in reps})

    allowed: List[float] = []
    for r in reps:
        for a in (r, r + 90.0):
            a = round(geom.normalize_angle(a), 6)
            if all(abs(a - x) > 1e-6 for x in allowed):
                allowed.append(a)
    return allowed, reps


# --- door localisation ------------------------------------------------------
def door_inward_normal(poly: Sequence[Point], door_pts: Sequence[Point]) -> Point:
    p0 = tuple(map(float, door_pts[0]))
    p1 = tuple(map(float, door_pts[1]))
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        if geom.point_on_segment(p0, a, b, 1e-3) and geom.point_on_segment(p1, a, b, 1e-3):
            return geom.interior_normal(a, b, poly)
    best = None
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        d = geom.dist_point_segment(p0, a, b)
        if best is None or d < best[0]:
            best = (d, a, b)
    return geom.interior_normal(best[1], best[2], poly)


# --- parsing ----------------------------------------------------------------
def parse_problem(data: Dict, source: str = "") -> Problem:
    boundary = [tuple(map(float, p)) for p in data["boundary"]]
    poly, report = geom.polygon_cleanup(boundary)
    door_pts = [tuple(map(float, p)) for p in data["door"]]
    is_open_inward = bool(data.get("isOpenInward", False))

    items = []
    for name, dims in data["algoToPlace"].items():
        items.append(Item(name, item_kind(name), float(dims[0]), float(dims[1])))

    inward = door_inward_normal(poly, door_pts)
    door = Door(door_pts[0], door_pts[1], geom.dist(door_pts[0], door_pts[1]), is_open_inward, inward)

    allowed, reps = orientation_angles(poly)
    room = Room(boundary, poly, report, geom.polygon_area(poly), geom.bounding_box(poly), allowed, reps)
    return Problem(room, door, items, source)


def load_problem(path: str) -> Problem:
    import json
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return parse_problem(data, source=path)
