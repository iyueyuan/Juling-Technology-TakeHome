"""Pure-stdlib geometry kernel for the storage-room layout solver.

All quantities are in millimetres. Angles are in degrees (0..360), measured
counter-clockwise in the input coordinate frame (the same frame the input JSON
uses); an item angle of 0 means ``length`` lies along +x and ``width`` along +y.
"""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

Point = Tuple[float, float]
Poly = List[Point]

# --- tolerances -------------------------------------------------------------
GEOM_EPS = 1e-6          # generic geometric epsilon (mm)
ON_BOUNDARY_TOL = 1e-6   # distance (mm) within which a point counts as on a line
CONTACT_TOL = 1e-6       # (mm) two touching (non-overlapping) bodies are legal
PENETRATION_TOL = 1e-3   # (mm^2) overlap area above which two rects "overlap"
SEG_OVERLAP_TOL = 1e-3   # (mm) overlap length above which a door/front is blocked
ANGLE_TOL = 0.5          # (deg) angle tolerance
CLUSTER_TOL = 1.0        # (deg) tolerance when clustering edge directions
AXIS_SNAP_TOL = 0.5      # (deg) snap near-axis cluster reps to 0/90


# --- vectors ----------------------------------------------------------------
def add(a: Point, b: Point) -> Point:
    return (a[0] + b[0], a[1] + b[1])


def sub(a: Point, b: Point) -> Point:
    return (a[0] - b[0], a[1] - b[1])


def mul(a: Point, k: float) -> Point:
    return (a[0] * k, a[1] * k)


def dot(a: Point, b: Point) -> float:
    return a[0] * b[0] + a[1] * b[1]


def cross(a: Point, b: Point) -> float:
    return a[0] * b[1] - a[1] * b[0]


def norm(a: Point) -> float:
    return math.hypot(a[0], a[1])


def unit(a: Point) -> Point:
    n = norm(a)
    if n == 0.0:
        return (1.0, 0.0)
    return (a[0] / n, a[1] / n)


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def dist_point_segment(p: Point, a: Point, b: Point) -> float:
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    if dx == 0.0 and dy == 0.0:
        return dist(p, a)
    t = ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)
    t = max(0.0, min(1.0, t))
    return dist(p, (ax + t * dx, ay + t * dy))


# --- polygon basics ---------------------------------------------------------
def signed_area(poly: Sequence[Point]) -> float:
    s = 0.0
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        s += x1 * y2 - x2 * y1
    return s / 2.0


def polygon_area(poly: Sequence[Point]) -> float:
    return abs(signed_area(poly))


def ensure_ccw(poly: Poly) -> Poly:
    """Return the polygon traversed so that signed_area >= 0."""
    if signed_area(poly) < 0:
        return list(reversed(poly))
    return list(poly)


def bounding_box(poly: Sequence[Point]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in poly]
    ys = [p[1] for p in poly]
    return min(xs), min(ys), max(xs), max(ys)


def angle_of(a: Point, b: Point) -> float:
    return math.degrees(math.atan2(b[1] - a[1], b[0] - a[0])) % 360.0


def normalize_angle(deg: float) -> float:
    return deg % 360.0


def mod90_diff(a: float, b: float) -> float:
    """Minimum angular distance between two directions modulo 90 deg (0..45)."""
    d = abs((a - b) % 90.0)
    return min(d, 90.0 - d)


# --- polygon cleanup --------------------------------------------------------
def polygon_cleanup(pts: Sequence[Point], tol: float = GEOM_EPS) -> Tuple[Poly, List[str]]:
    """Normalise a possibly-degenerate boundary into a simple polygon.

    Runs to a fixpoint: drops consecutive/closing duplicate points, removes
    antiparallel spikes (an edge immediately retraced), and merges collinear
    consecutive edges. Returns (polygon, report).
    """
    report: List[str] = []
    out: List[Point] = [tuple(map(float, p)) for p in pts]

    changed = True
    while changed:
        changed = False

        # 1) consecutive duplicates + closing duplicate
        ded: List[Point] = []
        for p in out:
            if not ded or dist(p, ded[-1]) > tol:
                ded.append(p)
        if len(ded) > 1 and dist(ded[0], ded[-1]) <= tol:
            ded.pop()
        if len(ded) != len(out):
            report.append("removed %d duplicate point(s)" % (len(out) - len(ded)))
            out = ded
            changed = True
            continue

        n = len(out)
        if n < 4:
            break

        # 2) antiparallel spikes (edge out then back)
        spike = False
        for i in range(n):
            a, b, c = out[i - 1], out[i], out[(i + 1) % n]
            e1 = sub(b, a)
            e2 = sub(c, b)
            if abs(e1[0] + e2[0]) < tol and abs(e1[1] + e2[1]) < tol:
                report.append("removed spike at %s" % (_fmt(b)))
                out.pop(i)
                spike = True
                break
        if spike:
            changed = True
            continue

        # 3) collinear consecutive edges (same direction)
        merged = False
        for i in range(n):
            a, b, c = out[i - 1], out[i], out[(i + 1) % n]
            e1 = sub(b, a)
            e2 = sub(c, b)
            cr = cross(e1, e2)
            if abs(cr) < tol * max(1.0, norm(e1) * norm(e2)) and dot(e1, e2) > 0:
                report.append("merged collinear point at %s" % (_fmt(b)))
                out.pop(i)
                merged = True
                break
        if merged:
            changed = True
    return out, report


def _fmt(p: Point) -> str:
    return "(%.4f, %.4f)" % (p[0], p[1])


# --- point-in-polygon -------------------------------------------------------
def point_on_segment(p: Point, a: Point, b: Point, tol: float = ON_BOUNDARY_TOL) -> bool:
    return dist_point_segment(p, a, b) <= tol


def point_in_polygon(p: Point, poly: Sequence[Point], tol: float = ON_BOUNDARY_TOL) -> bool:
    """True if p is strictly inside, or on the boundary within ``tol``."""
    x, y = p
    n = len(poly)
    for i in range(n):
        if point_on_segment(p, poly[i], poly[(i + 1) % n], tol):
            return True
    inside = False
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
            if x < xint:
                inside = not inside
    return inside


def point_strict_inside(p: Point, poly: Sequence[Point], tol: float = ON_BOUNDARY_TOL) -> bool:
    if not point_in_polygon(p, poly, tol):
        return False
    n = len(poly)
    for i in range(n):
        if point_on_segment(p, poly[i], poly[(i + 1) % n], tol):
            return False
    return True


def interior_normal(a: Point, b: Point, poly: Sequence[Point], tol: float = GEOM_EPS) -> Point:
    """Unit normal to edge (a,b) pointing toward the polygon interior.

    Never assumes a particular winding (the input mixes CW and CCW), so the
    direction is decided by probing slightly offset points.
    """
    dx, dy = b[0] - a[0], b[1] - a[1]
    L = math.hypot(dx, dy)
    if L == 0.0:
        return (1.0, 0.0)
    ux, uy = dx / L, dy / L
    candidates = ((uy, -ux), (-uy, ux))
    scores = []
    for cand in candidates:
        good = bad = 0
        for frac in (0.5, 0.3, 0.7):
            mx = a[0] + frac * dx
            my = a[1] + frac * dy
            for d in (1.0, 2.0, 0.5):
                px = mx + cand[0] * d
                py = my + cand[1] * d
                if point_strict_inside((px, py), poly, tol):
                    good += 1
                else:
                    bad += 1
        scores.append((good, bad, cand))
    scores.sort(key=lambda t: (-t[0], t[1]))
    return scores[0][2]


# --- rectangles -------------------------------------------------------------
def rect_corners(center: Point, length: float, width: float, angle_deg: float) -> Poly:
    """Four corners of an axis-aligned-in-local-frame rectangle, CCW order."""
    cx, cy = center
    th = math.radians(angle_deg)
    ux, uy = math.cos(th), math.sin(th)
    vx, vy = -math.sin(th), math.cos(th)
    hl, hw = length / 2.0, width / 2.0
    return [
        (cx - ux * hl - vx * hw, cy - uy * hl - vy * hw),
        (cx + ux * hl - vx * hw, cy + uy * hl - vy * hw),
        (cx + ux * hl + vx * hw, cy + uy * hl + vy * hw),
        (cx - ux * hl + vx * hw, cy - uy * hl + vy * hw),
    ]


def rect_edges(corners: Poly) -> List[Tuple[Point, Point]]:
    return [(corners[i], corners[(i + 1) % 4]) for i in range(4)]


def rect_bbox(corners: Poly) -> Tuple[float, float, float, float]:
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    return (min(xs), min(ys), max(xs), max(ys))


def bboxes_overlap(b1: Tuple[float, float, float, float], b2: Tuple[float, float, float, float]) -> bool:
    return b1[0] <= b2[2] + 1e-9 and b2[0] <= b1[2] + 1e-9 and b1[1] <= b2[3] + 1e-9 and b2[1] <= b1[3] + 1e-9


# --- segment intersection ---------------------------------------------------
def _seg_cross_value(a: Point, b: Point, c: Point) -> float:
    return cross(sub(b, a), sub(c, a))


def segments_properly_intersect(a: Point, b: Point, c: Point, d: Point, tol: float = GEOM_EPS) -> bool:
    """True only for a proper (interior-interior) crossing with penetration > tol (mm)."""
    ab = norm(sub(b, a))
    cd = norm(sub(d, c))
    if ab < 1e-9 or cd < 1e-9:
        return False
    o1 = cross(sub(b, a), sub(c, a)) / ab
    o2 = cross(sub(b, a), sub(d, a)) / ab
    o3 = cross(sub(d, c), sub(a, c)) / cd
    o4 = cross(sub(d, c), sub(b, c)) / cd
    return ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and \
           ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol))


def rect_inside_polygon(corners: Poly, poly: Sequence[Point], tol: float = ON_BOUNDARY_TOL) -> bool:
    """True if the rectangle lies entirely inside-or-on ``poly`` (tol in mm)."""
    for c in corners:
        if not point_in_polygon(c, poly, tol):
            return False
    n = len(poly)
    for (a, b) in rect_edges(corners):
        for i in range(n):
            if segments_properly_intersect(a, b, poly[i], poly[(i + 1) % n], tol):
                return False
        # safety: edge midpoint must also be inside/on
        if not point_in_polygon(((a[0] + b[0]) / 2, (a[1] + b[1]) / 2), poly, tol):
            return False
    return True


# --- convex clipping (Sutherland-Hodgman) -----------------------------------
def _clip_intersect(s: Point, e: Point, a: Point, b: Point) -> Point:
    dx, dy = e[0] - s[0], e[1] - s[1]
    v = sub(b, a)
    w = sub(e, s)
    den = cross(v, w)
    if abs(den) < 1e-12:
        return e
    t = -cross(v, sub(s, a)) / den
    return (s[0] + t * dx, s[1] + t * dy)


def convex_clip(subject: Sequence[Point], clip: Sequence[Point]) -> Poly:
    """Clip ``subject`` by the convex polygon ``clip`` (CCW)."""
    clip = ensure_ccw(list(clip))
    output = list(subject)
    for i in range(len(clip)):
        a = clip[i]
        b = clip[(i + 1) % len(clip)]
        ab = sub(b, a)
        if not output:
            break
        inp = output
        output = []
        for j in range(len(inp)):
            s = inp[j]
            e = inp[(j + 1) % len(inp)]
            si = cross(ab, sub(s, a)) >= -GEOM_EPS
            ei = cross(ab, sub(e, a)) >= -GEOM_EPS
            if si and ei:
                output.append(e)
            elif si and not ei:
                output.append(_clip_intersect(s, e, a, b))
            elif not si and ei:
                output.append(_clip_intersect(s, e, a, b))
                output.append(e)
    return output


def convex_overlap_area(p1: Sequence[Point], p2: Sequence[Point]) -> float:
    clipped = convex_clip(p1, p2)
    if len(clipped) < 3:
        return 0.0
    return polygon_area(clipped)


def rect_rect_overlap_area(c1: Sequence[Point], c2: Sequence[Point]) -> float:
    return convex_overlap_area(c1, c2)


# --- segment vs convex polygon (for zero-depth door/front rules) ------------
def clip_segment_convex(p: Point, q: Point, convex: Sequence[Point]) -> Optional[Tuple[Point, Point]]:
    """Clip segment p->q by a CCW convex polygon; return the kept sub-segment or None."""
    conv = ensure_ccw(list(convex))
    tmin, tmax = 0.0, 1.0
    d = sub(q, p)
    for i in range(len(conv)):
        a = conv[i]
        b = conv[(i + 1) % len(conv)]
        ab = sub(b, a)
        c = cross(ab, sub(p, a))
        dd = cross(ab, d)
        if abs(dd) < 1e-12:
            if c < -GEOM_EPS:
                return None
            continue
        t = -c / dd
        if dd > 0:
            tmin = max(tmin, t)
        else:
            tmax = min(tmax, t)
        if tmin > tmax:
            return None
    pa = (p[0] + tmin * d[0], p[1] + tmin * d[1])
    pb = (p[0] + tmax * d[0], p[1] + tmax * d[1])
    return (pa, pb)


def segment_overlap_with_rect(seg_a: Point, seg_b: Point, rect: Sequence[Point]) -> float:
    kept = clip_segment_convex(seg_a, seg_b, rect)
    if kept is None:
        return 0.0
    return dist(kept[0], kept[1])


# --- wall contact -----------------------------------------------------------
def point_line_distance(p: Point, a: Point, b: Point) -> float:
    v = sub(b, a)
    n = norm(v)
    if n == 0.0:
        return dist(p, a)
    return abs(cross(v, sub(p, a))) / n


def _collinear_overlap_length(p: Point, q: Point, a: Point, b: Point, tol: float) -> float:
    if dist(p, q) < tol or dist(a, b) < tol:
        return 0.0
    # all four points on the same line?
    if point_line_distance(a, p, q) > tol or point_line_distance(b, p, q) > tol:
        return 0.0
    if point_line_distance(p, a, b) > tol or point_line_distance(q, a, b) > tol:
        return 0.0
    u = unit(sub(b, a))
    origin = a
    def proj(x: Point) -> float:
        return dot(sub(x, origin), u)
    a1, a2 = proj(a), proj(b)
    b1, b2 = proj(p), proj(q)
    lo = max(min(a1, a2), min(b1, b2))
    hi = min(max(a1, a2), max(b1, b2))
    return max(0.0, hi - lo)


def rect_rect_contact_mm(c1: Poly, c2: Poly, tol: float = CONTACT_TOL) -> float:
    """Total length (mm) of shared collinear boundary between two rectangles."""
    total = 0.0
    for (p, q) in rect_edges(c1):
        for (r, s) in rect_edges(c2):
            total += _collinear_overlap_length(p, q, r, s, tol)
    return total


def wall_contact(corners: Poly, poly: Sequence[Point], tol: float = CONTACT_TOL) -> Tuple[float, List[Tuple[int, float]]]:
    """Total length (mm) of contact between the rect's sides and boundary edges."""
    total = 0.0
    per_edge: List[Tuple[int, float]] = []
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        best = 0.0
        for (p, q) in rect_edges(corners):
            L = _collinear_overlap_length(p, q, a, b, tol)
            best = max(best, L)
        if best > tol:
            total += best
            per_edge.append((i, best))
    return total, per_edge
