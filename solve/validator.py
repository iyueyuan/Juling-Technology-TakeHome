"""Independent constraint checker.

Re-derives every requirement straight from the raw input JSON + the result JSON
(no reliance on solver internals). Each check returns PASS/FAIL plus the
measured value, so the "是否可行" answer is auditable.
"""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

from . import geom, model
from .geom import Point, Poly
from .model import Problem

CONTAINMENT_TOL = 1e-2   # mm  (absorbs the ~1e-3 mm rounding of the serialised output)
OVERLAP_TOL = 1.0        # mm^2 (absorbs rounding; a real overlap is >= hundreds of mm^2)
ANGLE_TOL = 0.5          # deg


def _build_rect(p: Dict) -> Poly:
    center = (float(p["center"][0]), float(p["center"][1]))
    ang = float(p["rotation"])
    length = float(p["length"])
    width = float(p["width"])
    return geom.rect_corners(center, length, width, ang)


def _fridge_front_face(corners: Poly, sign: int) -> Tuple[Point, Point]:
    # corners are CCW: +v face = (c2, c3); -v face = (c0, c1)
    if sign == 1:
        return corners[2], corners[3]
    return corners[0], corners[1]


def _front_band(corners: Poly, sign: int, depth: float, angle_deg: float) -> Poly:
    cA, cB = _fridge_front_face(corners, sign)
    u = (math.cos(math.radians(angle_deg)), math.sin(math.radians(angle_deg)))
    v = (-u[1], u[0])
    n = geom.mul(v, sign)
    off = geom.mul(n, depth)
    return [cA, cB, geom.add(cB, off), geom.add(cA, off)]


def validate(problem: Problem, result: Dict, fridge_clearance: Optional[float] = None) -> Dict:
    room, door = problem.room, problem.door
    placements = result.get("placements", [])
    feasible_flag = bool(result.get("feasible", False))
    checks: List[Dict] = []

    # V1 polygon validity
    cleanup_notes = list(room.cleanup_report)
    v1_ok = len(room.poly) >= 3 and room.area > 0
    checks.append({"id": "V1", "rule": "轮廓有效性(退化顶点清理后为简单多边形且面积>0)",
                   "status": "PASS" if v1_ok else "FAIL",
                   "detail": "cleaned %d->%d vertices; area=%.2f mm^2; %s" % (
                       len(room.raw_boundary), len(room.poly), room.area,
                       "; ".join(cleanup_notes) if cleanup_notes else "no degenerate vertices")})

    # build rects and check completeness (V8 first, cheap)
    by_name = {it.name: it for it in problem.items}
    rects: Dict[str, Poly] = {}
    v8_problems = []
    seen_names = []
    for p in placements:
        name = p.get("name")
        if name not in by_name:
            v8_problems.append("unknown item '%s'" % name)
            continue
        seen_names.append(name)
        try:
            rects[name] = _build_rect(p)
        except (KeyError, TypeError, ValueError) as e:
            v8_problems.append("item '%s' missing/invalid fields: %s" % (name, e))
    for nm in by_name:
        if nm not in seen_names:
            v8_problems.append("missing item '%s'" % nm)
    for nm, c in rects.items():
        for v in c:
            if not all(math.isfinite(x) for x in v):
                v8_problems.append("item '%s' has non-finite coordinate" % nm)
                break
    v8_ok = not v8_problems
    checks.append({"id": "V8", "rule": "完整性(名称/尺寸一一对应、坐标有限)",
                   "status": "PASS" if v8_ok else "FAIL",
                   "detail": "; ".join(v8_problems) if v8_problems else "%d/%d items present" % (len(rects), len(by_name))})

    # V2 containment
    v2_bad = []
    for nm, c in rects.items():
        if not geom.rect_inside_polygon(c, room.poly, CONTAINMENT_TOL):
            v2_bad.append(nm)
    checks.append({"id": "V2", "rule": "全部物体在轮廓内(容差 %.1e mm)" % CONTAINMENT_TOL,
                   "status": "PASS" if not v2_bad else "FAIL",
                   "detail": "all inside" if not v2_bad else "outside: %s" % ", ".join(v2_bad)})

    # V3 pairwise non-overlap
    names = list(rects.keys())
    v3_bad = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = geom.rect_rect_overlap_area(rects[names[i]], rects[names[j]])
            if a > OVERLAP_TOL:
                v3_bad.append("%s×%s=%.4f mm^2" % (names[i], names[j], a))
    checks.append({"id": "V3", "rule": "两两不重叠(交叠面积 ≤ %.1e mm^2，允许贴合)" % OVERLAP_TOL,
                   "status": "PASS" if not v3_bad else "FAIL",
                   "detail": "no overlaps" if not v3_bad else "; ".join(v3_bad)})

    # V4 angle parallel/perpendicular to a boundary edge
    v4_bad = []
    for nm, p in zip(seen_names, placements):
        if nm not in rects:
            continue
        ang = float(p["rotation"]) % 90.0
        ok = False
        for (a, b) in room.edges():
            e = geom.angle_of(a, b) % 90.0
            d = abs(ang - e) % 90.0
            if min(d, 90.0 - d) <= ANGLE_TOL + 1e-9:
                ok = True
                break
        if not ok:
            v4_bad.append("%s=%.3f°" % (nm, float(p["rotation"])))
    checks.append({"id": "V4", "rule": "旋转角与某条轮廓边平行/垂直(容差 %.1f°)" % ANGLE_TOL,
                   "status": "PASS" if not v4_bad else "FAIL",
                   "detail": "all aligned" if not v4_bad else "; ".join(v4_bad)})

    # V5 door not blocked
    zone = door.zone_corners()
    v5_bad = []
    if zone is not None:
        for nm, c in rects.items():
            if geom.rect_rect_overlap_area(c, zone) > OVERLAP_TOL:
                v5_bad.append("%s overlaps inward N×N zone" % nm)
    for nm, c in rects.items():
        if geom.segment_overlap_with_rect(door.p0, door.p1, c) > geom.SEG_OVERLAP_TOL:
            v5_bad.append("%s is flush across the door opening" % nm)
    checks.append({"id": "V5", "rule": "门未被遮挡(内开 N×N 无物体；外开不压门洞)",
                   "status": "PASS" if not v5_bad else "FAIL",
                   "detail": "door clear" if not v5_bad else "; ".join(v5_bad)})

    # V6 fridge door side free (clearance = length/2 by default)
    clearance_used = fridge_clearance
    v6_bad = []
    v6_info = []
    for nm, p in zip(seen_names, placements):
        it = by_name.get(nm)
        if it is None or it.kind != "fridge":
            continue
        dep = fridge_clearance if fridge_clearance is not None else it.length / 2.0
        if clearance_used is None:
            clearance_used = dep
        corners = rects[nm]
        ang = float(p["rotation"])
        ok = False
        for sign in (1, -1):
            band = _front_band(corners, sign, dep, ang)
            inside = geom.rect_inside_polygon(band, room.poly, CONTAINMENT_TOL)
            if not inside:
                continue
            free = True
            for onm, oc in rects.items():
                if onm == nm:
                    continue
                if geom.rect_rect_overlap_area(band, oc) > OVERLAP_TOL:
                    free = False
                    break
            if free:
                ok = True
                break
        if ok:
            v6_info.append("%s front %.0fmm clear" % (nm, dep))
        else:
            v6_bad.append("%s front not clear (%.0fmm)" % (nm, dep))
    checks.append({"id": "V6", "rule": "冰箱开门边(面宽=length)前方净空 %.0fmm 在房间内且无任何物体" % (clearance_used or 0),
                   "status": "PASS" if not v6_bad else "FAIL",
                   "detail": "; ".join(v6_info) if not v6_bad else "; ".join(v6_bad)})

    # V7 shelf/overShelf may touch (no clearance): touching is not an overlap (already covered by V3)
    checks.append({"id": "V7", "rule": "货架/离地架四边可贴放物品(无净空要求)",
                   "status": "PASS",
                   "detail": "flushing contact is permitted and is not counted as overlap (V3)"})

    passed = all(c["status"] == "PASS" for c in checks)

    # metrics
    n_wall = 0
    total_wall = 0.0
    for nm, c in rects.items():
        w, _ = geom.wall_contact(c, room.poly, 1e-3)
        total_wall += w
        if w > geom.CONTACT_TOL:
            n_wall += 1
    items_area = sum(by_name[nm].area for nm in rects)
    free_area = room.area - items_area - (geom.polygon_area(zone) if zone else 0.0)
    metrics = {
        "itemsTouchingWall": n_wall,
        "wallContactMm": round(total_wall, 2),
        "freeAreaMm2": round(free_area, 2),
        "fridgeClearanceMm": fridge_clearance if fridge_clearance is not None else round(next(
            (by_name[nm].length / 2.0 for nm in rects if by_name[nm].kind == "fridge"), 0.0), 2),
    }
    return {"passed": passed, "checks": checks, "metrics": metrics,
            "feasibleFlag": feasible_flag,
            "feasibleFlagConsistent": (not feasible_flag) or passed}
