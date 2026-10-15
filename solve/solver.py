"""Constructive wall-hugging solver: candidate generation + beam search + fallbacks."""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, replace
from typing import Dict, List, Optional, Sequence, Tuple

from . import geom, model
from .geom import Point, Poly
from .model import Item, Problem, Room, Door, KIND_PRIORITY

CONTACT_MEASURE_TOL = 1e-3  # mm, for measuring wall/item contact lengths


@dataclass
class Config:
    orientation: str = "axis"                 # axis | auto
    fridge_clearance: Optional[float] = None  # None -> length/2
    min_wall_len: float = 500.0
    beam: int = 40
    branch: int = 50
    grid_step: float = 50.0
    max_grid_candidates: int = 240
    wall_step: float = 100.0
    max_pool: int = 500
    node_budget: int = 1500000
    dfs_nodes: int = 0     # 0 = DFS only as a completeness fallback (>0 adds a quality pass)
    time_limit: float = 25.0
    seed: int = 0
    _grid_enabled: bool = False


@dataclass
class Placement:
    item: Item
    center: Point
    angle: float
    corners: Poly
    front_sign: int = 0       # fridge: +1 front at +v side, -1 at -v side
    wall_mm: float = 0.0      # cached wall-contact length
    dmin: float = 0.0         # cached min distance from center to a wall edge
    bbox: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)

    def front_band(self, depth: float) -> Optional[Poly]:
        if self.item.kind != "fridge" or depth <= 0.0:
            return None
        v = (-math.sin(math.radians(self.angle)), math.cos(math.radians(self.angle)))
        n = geom.mul(v, self.front_sign)
        cA, cB = (self.corners[2], self.corners[3]) if self.front_sign == 1 else (self.corners[0], self.corners[1])
        off = geom.mul(n, depth)
        return [cA, cB, geom.add(cB, off), geom.add(cA, off)]

    def front_normal(self) -> Point:
        v = (-math.sin(math.radians(self.angle)), math.cos(math.radians(self.angle)))
        return geom.mul(v, self.front_sign)


@dataclass
class Result:
    feasible: bool
    placements: List[Placement]
    reason: Optional[str]
    clearance_mode: str
    used_grid: bool
    strategy: str
    elapsed: float
    stats: dict


def fridge_depth(problem: Problem, config: Config) -> float:
    if config.fridge_clearance is not None:
        return float(config.fridge_clearance)
    for it in problem.items:
        if it.kind == "fridge":
            return it.length / 2.0
    return 0.0


def _item_orders(items: List[Item]) -> List[List[Item]]:
    s1 = sorted(items, key=lambda i: (KIND_PRIORITY.get(i.kind, 9), -i.area, i.name))
    s2 = sorted(items, key=lambda i: (-i.area, i.name))
    s3 = sorted(items, key=lambda i: (i.area, i.name))
    orders = [s1, s2, s3]
    out, seen = [], set()
    for s in orders:
        key = tuple(i.name for i in s)
        if key not in seen:
            seen.add(key)
            out.append(s)
    return out


# --- geometry helpers -------------------------------------------------------
def _allowed_angles(problem: Problem, config: Config) -> List[float]:
    """Item orientations for interior (grid) candidates, normalised to [0, 180)."""
    src = [0.0, 90.0] if config.orientation == "axis" else list(problem.room.allowed_angles)
    out: List[float] = []
    for a in src:
        a = a % 180.0
        if all(abs(a - x) > 1e-6 for x in out):
            out.append(a)
    return out


def _axes(angle: float) -> Tuple[Point, Point]:
    th = math.radians(angle)
    u = (math.cos(th), math.sin(th))
    v = (-math.sin(th), math.cos(th))
    return u, v


def _annotate(pl: Placement, room: Room) -> Placement:
    w, _ = geom.wall_contact(pl.corners, room.poly, CONTACT_MEASURE_TOL)
    pl.wall_mm = w
    pl.dmin = min(geom.dist_point_segment(pl.center, a, b) for (a, b) in room.edges())
    pl.bbox = geom.rect_bbox(pl.corners)
    return pl


def _fits_surface(item: Item, t: Point, u: Point, v: Point) -> Optional[Tuple[float, float]]:
    """Return (span, perp) if the item can sit with its back against a surface of direction t."""
    dot_u = abs(geom.dot(u, t))
    dot_v = abs(geom.dot(v, t))
    cos_tol = math.cos(math.radians(geom.ANGLE_TOL))
    if dot_u >= dot_v and dot_u > cos_tol:
        return (item.length, item.width)
    if dot_v > cos_tol and item.kind != "fridge":
        return (item.width, item.length)
    return None


def _edge_anchored(item: Item, angle: float, a: Point, b: Point, n: Point,
                   span: float, perp: float, s: float, front_sign: int) -> Optional[Placement]:
    t = geom.unit(geom.sub(b, a))
    L = geom.dist(a, b)
    if span > L + 1e-6:
        return None
    s = max(0.0, min(s, L - span))
    wall_pt = geom.add(a, geom.mul(t, s + span / 2.0))
    center = geom.add(wall_pt, geom.mul(n, perp / 2.0))
    corners = geom.rect_corners(center, item.length, item.width, angle)
    return Placement(item, center, angle, corners, front_sign)


def _static_ok(pl: Placement, room: Room, door: Door, config: Config) -> bool:
    if not geom.rect_inside_polygon(pl.corners, room.poly):
        return False
    zone = door.zone_corners()
    if zone is not None:
        if geom.rect_rect_overlap_area(pl.corners, zone) > geom.PENETRATION_TOL:
            return False
    elif geom.segment_overlap_with_rect(door.p0, door.p1, pl.corners) > geom.SEG_OVERLAP_TOL:
        return False  # outward door: no item may be flush across the opening
    if pl.item.kind == "fridge":
        depth = fridge_depth_current(pl.item, config)
        band = pl.front_band(depth)
        if band is None or not geom.rect_inside_polygon(band, room.poly):
            return False
    return True


def fridge_depth_current(item: Item, config: Config) -> float:
    return float(config.fridge_clearance) if config.fridge_clearance is not None else item.length / 2.0


def _dedup(placements: List[Placement]) -> List[Placement]:
    seen, out = set(), []
    for p in placements:
        key = (round(p.center[0], 3), round(p.center[1], 3), round(p.angle, 3), p.front_sign)
        if key not in seen:
            seen.add(key)
            out.append(p)
    return out


def _surface_angles(edge_ang: float, config: Config) -> Tuple[float, ...]:
    """Item orientations to try against a surface of direction ``edge_ang``.

    Angles are normalised to [0, 180): a rectangle is invariant under a 180 deg
    rotation, so this removes geometrically duplicate candidates.
    """
    if config.orientation == "axis":
        # strict 0/90 mode: a slanted surface cannot host a flush axis-aligned item
        if min(geom.mod90_diff(edge_ang, 0.0), geom.mod90_diff(edge_ang, 90.0)) > geom.AXIS_SNAP_TOL:
            return ()
        return (0.0, 90.0)
    return (edge_ang % 180.0, (edge_ang + 90.0) % 180.0)


def _wall_candidates(item: Item, room: Room, door: Door, config: Config) -> List[Placement]:
    door_zone = door.zone_corners()
    out: List[Placement] = []
    for (a, b) in room.edges():
        t = geom.unit(geom.sub(b, a))
        n = geom.interior_normal(a, b, room.poly)
        L = geom.dist(a, b)
        edge_ang = geom.angle_of(a, b)
        for ang in _surface_angles(edge_ang, config):
            u, v = _axes(ang)
            fit = _fits_surface(item, t, u, v)
            if fit is None:
                continue
            span, perp = fit
            offs = {0.0, L - span, (L - span) / 2.0}
            step = config.wall_step
            if step > 0:
                for k in range(int(L // step) + 2):
                    offs.add(min(L, k * step))
            for (px, py) in room.poly:
                pr = geom.dot(geom.sub((px, py), a), t)
                offs.add(pr)
                offs.add(pr - span)
            for dp in (door.p0, door.p1):
                pr = geom.dot(geom.sub(dp, a), t)
                offs.add(pr)
                offs.add(pr - span)
            if door_zone is not None:
                for c in door_zone:
                    pr = geom.dot(geom.sub(c, a), t)
                    offs.add(pr)
                    offs.add(pr - span)
            for s in offs:
                if s < -1e-6 or s > L - span + 1e-6:
                    continue
                for fs in ((1, -1) if item.kind == "fridge" else (0,)):
                    pl = _edge_anchored(item, ang, a, b, n, span, perp, s, fs)
                    if pl is not None and _static_ok(pl, room, door, config):
                        out.append(_annotate(pl, room))
    return out


def _object_anchored(item: Item, placed: List[Placement], room: Room,
                     door: Door, config: Config, cap: int = 24) -> List[Placement]:
    if not placed:
        return []
    out: List[Placement] = []
    for o in placed[-1:]:          # anchor only against the most recent item
        for i in range(4):
            p, q = o.corners[i], o.corners[(i + 1) % 4]
            d = geom.sub(q, p)
            L = geom.dist(p, q)
            if L < 1e-6:
                continue
            t = geom.unit(d)
            m = (t[1], -t[0])      # outward normal of a CCW side
            side_ang = geom.angle_of(p, q)
            for ang in _surface_angles(side_ang, config):
                u, v = _axes(ang)
                fit = _fits_surface(item, t, u, v)
                if fit is None:
                    continue
                span, perp = fit
                for s in {0.0, L - span, (L - span) / 2.0}:
                    if s < -1e-6 or s > L - span + 1e-6:
                        continue
                    base = geom.add(p, geom.mul(t, s + span / 2.0))
                    center = geom.add(base, geom.mul(m, perp / 2.0))
                    corners = geom.rect_corners(center, item.length, item.width, ang)
                    for fs in ((1, -1) if item.kind == "fridge" else (0,)):
                        pl = Placement(item, center, ang, corners, fs)
                        if _static_ok(pl, room, door, config):
                            out.append(_annotate(pl, room))
    out.sort(key=lambda p: (-p.wall_mm, p.dmin, p.center[0], p.center[1]))
    return out[:cap]


def _grid_candidates(item: Item, angle: float, room: Room, door: Door,
                     config: Config) -> List[Placement]:
    x0, y0, x1, y1 = room.bbox
    step = config.grid_step
    out: List[Placement] = []
    if (x1 - x0) * (y1 - y0) / (step * step) > 20000:
        step = max(step, min(x1 - x0, y1 - y0) / 80.0)
    y = y0
    while y <= y1:
        x = x0
        while x <= x1:
            for fs in ((1, -1) if item.kind == "fridge" else (0,)):
                corners = geom.rect_corners((x, y), item.length, item.width, angle)
                pl = Placement(item, (x, y), angle, corners, fs)
                if _static_ok(pl, room, door, config):
                    out.append(_annotate(pl, room))
            x += step
        y += step
    out.sort(key=lambda p: (-p.wall_mm, p.dmin, p.center[0], p.center[1]))
    return out[:config.max_grid_candidates]


# --- scoring ----------------------------------------------------------------
def _placement_score(pl: Placement, placed: List[Placement], room: Room, door: Door) -> Tuple[float, ...]:
    wall_touch = 1.0 if pl.wall_mm > geom.CONTACT_TOL else 0.0
    neighbor = sum(geom.rect_rect_contact_mm(pl.corners, o.corners, CONTACT_MEASURE_TOL) for o in placed)
    facing = 0.0
    if pl.item.kind == "fridge":
        door_mid = ((door.p0[0] + door.p1[0]) / 2, (door.p0[1] + door.p1[1]) / 2)
        toward = geom.unit(geom.sub(door_mid, pl.center))
        facing = max(0.0, geom.dot(pl.front_normal(), toward))
    return (wall_touch, pl.wall_mm, neighbor, -pl.dmin, facing, -pl.center[0], -pl.center[1])


def _add(a: Tuple[float, ...], b: Tuple[float, ...]) -> Tuple[float, ...]:
    if not a:
        return b
    return tuple(x + y for x, y in zip(a, b))


def _collides(pl: Placement, placed: List[Placement], config: Config) -> bool:
    for o in placed:
        if geom.bboxes_overlap(pl.bbox, o.bbox) and \
                geom.rect_rect_overlap_area(pl.corners, o.corners) > geom.PENETRATION_TOL:
            return True
    if pl.item.kind == "fridge":
        band = pl.front_band(fridge_depth_current(pl.item, config))
        if band is not None:
            bb = geom.rect_bbox(band)
            for o in placed:
                if geom.bboxes_overlap(bb, o.bbox) and \
                        geom.rect_rect_overlap_area(band, o.corners) > geom.PENETRATION_TOL:
                    return True
    for o in placed:
        if o.item.kind == "fridge":
            oband = o.front_band(fridge_depth_current(o.item, config))
            if oband is not None:
                obb = geom.rect_bbox(oband)
                if geom.bboxes_overlap(pl.bbox, obb) and \
                        geom.rect_rect_overlap_area(pl.corners, oband) > geom.PENETRATION_TOL:
                    return True
    return False


# --- beam search ------------------------------------------------------------
def _static_key(p: Placement) -> Tuple[float, ...]:
    return (-p.wall_mm, p.dmin, p.center[0], p.center[1])


def _dfs(problem: Problem, order: List[Item], config: Config, deadline: float, node_cap: int):
    """Depth-first search with backtracking; keeps the best layout, stops early on all-on-wall."""
    room, door = problem.room, problem.door
    allowed = _allowed_angles(problem, config)
    pools: Dict[str, List[Placement]] = {}
    for it in order:
        cands: List[Placement] = _wall_candidates(it, room, door, config)
        if config._grid_enabled:
            for ang in allowed:
                cands += _grid_candidates(it, ang, room, door, config)
        cands = _dedup(cands)
        cands.sort(key=_static_key)
        pools[it.name] = cands[:config.max_pool]
        if not pools[it.name]:
            return None

    nodes = [0]
    best = [None, None]   # (placements, score)
    done = [False]
    n_items = len(order)

    def rec(idx: int, placed: List[Placement], score: Tuple[float, ...]):
        nodes[0] += 1
        if done[0] or nodes[0] > node_cap or time.time() > deadline:
            return
        if idx == n_items:
            best[0] = list(placed)
            best[1] = score
            if score[0] >= n_items:   # all items touching a wall -> optimal for wall preference
                done[0] = True
            return
        it = order[idx]
        chosen: List[Placement] = []
        for c in pools[it.name]:
            if not _collides(c, placed, config):
                chosen.append(c)
        for c in _object_anchored(it, placed, room, door, config):
            if not _collides(c, placed, config):
                chosen.append(c)
        chosen = _dedup(chosen)
        scored = [(c, _placement_score(c, placed, room, door)) for c in chosen]
        scored.sort(key=lambda t: t[1], reverse=True)
        for c, contrib in scored:
            rec(idx + 1, placed + [c], _add(score, contrib))
            if done[0] or time.time() > deadline:
                return

    rec(0, [], ())
    if best[0] is None:
        return None
    return best[0], best[1]


def _beam(problem: Problem, order: List[Item], config: Config, deadline: float):
    """Beam search (breadth-first, keeps best states) — good at optimisation."""
    room, door = problem.room, problem.door
    allowed = _allowed_angles(problem, config)
    pools: Dict[str, List[Placement]] = {}
    for it in order:
        cands: List[Placement] = _wall_candidates(it, room, door, config)
        if config._grid_enabled:
            for ang in allowed:
                cands += _grid_candidates(it, ang, room, door, config)
        cands = _dedup(cands)
        cands.sort(key=_static_key)
        pools[it.name] = cands[:config.max_pool]
        if not pools[it.name]:
            return None

    n_items = len(order)
    states: List[Tuple[List[Placement], Tuple[float, ...]]] = [([], ())]
    best = [None, None]
    for it in order:
        new_states: List[Tuple[List[Placement], Tuple[float, ...]]] = []
        for placed, score in states:
            chosen: List[Placement] = []
            pool = pools[it.name]
            stride = max(1, len(pool) // (config.branch + 1))
            for idx in range(0, len(pool), stride):
                c = pool[idx]
                if not _collides(c, placed, config):
                    chosen.append(c)
                if len(chosen) >= config.branch:
                    break
            for c in _object_anchored(it, placed, room, door, config):
                if not _collides(c, placed, config):
                    chosen.append(c)
            chosen = _dedup(chosen)
            scored = [(c, _placement_score(c, placed, room, door)) for c in chosen]
            scored.sort(key=lambda t: t[1], reverse=True)
            for c, contrib in scored[:config.branch]:
                new_states.append((placed + [c], _add(score, contrib)))
        if not new_states:
            return None
        new_states.sort(key=lambda t: t[1], reverse=True)
        states = new_states[:config.beam]
        if time.time() > deadline:
            return None
    if states:
        best = (states[0][0], states[0][1])
    return best if best[0] is not None else None


def _try_strategies(problem: Problem, config: Config, deadline: float):
    labels = ["fridge-first", "area-desc", "area-asc"]
    orders = _item_orders(problem.items)
    best = None
    # 1) beam first: fast, and good at optimising the wall-hugging score
    for order, label in zip(orders, labels):
        if time.time() > deadline:
            break
        r = _beam(problem, order, config, deadline)
        if r is not None and (best is None or r[1] > best[1]):
            best = (r[0], r[1], label + "/beam")
    if best is not None and best[1][0] >= len(problem.items):
        return best  # every item already touches a wall -> nothing left to optimise
    # 2) DFS: completeness fallback when the beam could not place everything; with
    #    --dfs-nodes > 0 it also acts as an optional (slower) quality pass.
    if best is None:
        r = _dfs(problem, orders[0], config, deadline, config.node_budget)
        if r is not None:
            return (r[0], r[1], labels[0] + "/dfs")
        return None
    if config.dfs_nodes > 0 and time.time() <= deadline:
        r = _dfs(problem, orders[0], config, deadline, config.dfs_nodes)
        if r is not None and r[1] > best[1]:
            best = (r[0], r[1], labels[0] + "/dfs")
    return best


# --- top level --------------------------------------------------------------
def solve(problem: Problem, config: Optional[Config] = None) -> Result:
    config = config or Config()
    t0 = time.time()
    deadline = t0 + config.time_limit
    depth = fridge_depth(problem, config)

    def run(mode_label: str, clearance: float, use_grid: bool):
        cfg = replace(config, fridge_clearance=clearance, _grid_enabled=use_grid)
        best = _try_strategies(problem, cfg, deadline)
        if best is None:
            return None
        placements, score, strategy = best
        return Result(True, placements, None, mode_label, use_grid, strategy,
                      time.time() - t0, _stats(placements, problem))

    res = run("fridge-%gmm" % depth, depth, False)
    if res:
        return res
    res = run("fridge-%gmm+grid" % depth, depth, True)
    if res:
        return res
    res = run("literal-0", 0.0, False)
    if res:
        return res
    res = run("literal-0+grid", 0.0, True)
    if res:
        return res
    rng = random.Random(config.seed)
    cfg = replace(config, fridge_clearance=0.0, _grid_enabled=True)
    items = list(problem.items)
    for _ in range(3):
        if time.time() > deadline:
            break
        order = items[:]
        rng.shuffle(order)
        r = _dfs(problem, order, cfg, deadline)
        if r is not None:
            return Result(True, r[0], None, "literal-0+grid", True, "shuffle",
                          time.time() - t0, _stats(r[0], problem))
    return Result(False, [], "no feasible layout found within time budget",
                  "fridge-%gmm (and literal-0 fallback)" % depth, False, "",
                  time.time() - t0, {})


def _stats(placements: List[Placement], problem: Problem) -> dict:
    n_wall = 0
    total_wall = 0.0
    for p in placements:
        if p.wall_mm > geom.CONTACT_TOL:
            n_wall += 1
        total_wall += p.wall_mm
    fridge_front = next((p.item.length / 2.0 for p in placements if p.item.kind == "fridge"), None)
    return {
        "itemsPlaced": len(placements),
        "itemsTotal": len(problem.items),
        "itemsTouchingWall": n_wall,
        "wallContactMm": round(total_wall, 2),
        "fridgeClearanceMm": round(fridge_front, 2) if fridge_front is not None else None,
    }
