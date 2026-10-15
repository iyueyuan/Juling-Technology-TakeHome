"""Optional Pillow renderer: draws the room, door, items and fridge clearance."""
from __future__ import annotations

import math
from typing import Dict, List, Optional

from . import geom, model
from .model import Problem

TYPE_COLORS = {
    "fridge": ("#BFE3F2", "#2F6F8F"),
    "iceMaker": ("#E4D4F2", "#6F4F9F"),
    "shelf": ("#CFEBCF", "#3F7F3F"),
    "overShelf": ("#F4E5C4", "#9F7F2F"),
    "unknown": ("#E8E8E8", "#666666"),
}
TYPE_LABEL = {"fridge": "冰箱", "iceMaker": "制冰机", "shelf": "货架", "overShelf": "离地架", "unknown": "未知"}


def _hex(h: str) -> tuple:
    h = h.lstrip("#")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def render_layout(problem: Problem, placements: List[Dict], out_png: str, max_dim: int = 1400) -> None:
    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        print("Pillow not installed; skipping PNG render.")
        return

    room = problem.room
    door = problem.door
    (x0, y0, x1, y1) = room.bbox
    pad = 60.0
    scale = min(max_dim / (x1 - x0 + 2 * pad), max_dim / (y1 - y0 + 2 * pad))
    scale = max(scale, 0.02)

    def P(pt):
        return (round((pt[0] - x0 + pad) * scale), round((pt[1] - y0 + pad) * scale))

    W = round((x1 - x0 + 2 * pad) * scale)
    H = round((y1 - y0 + 2 * pad) * scale)
    img = Image.new("RGB", (W, H), "white")
    d = ImageDraw.Draw(img)

    # room boundary
    d.polygon([P(p) for p in room.poly], outline="#222222", fill="#FAFAFA")
    # raw boundary (dashed is not available; draw thin gray)
    if len(room.raw_boundary) > len(room.poly):
        d.polygon([P(p) for p in room.raw_boundary], outline="#BBBBBB")

    # inward door zone
    zone = door.zone_corners()
    if zone is not None:
        d.polygon([P(p) for p in zone], outline="#CC4444", fill="#F5DADA")

    # door: a thick marker offset INTO the room (so nothing is drawn outside the
    # outline). Its outer edge lies exactly on the wall segment = the doorway line.
    n_in = door.inward
    thick = 55.0
    marker = [door.p0, door.p1,
              geom.add(door.p1, geom.mul(n_in, thick)),
              geom.add(door.p0, geom.mul(n_in, thick))]
    d.polygon([P(x) for x in marker], outline="#CC2222", fill="#E86666")
    # inward swing indicator
    if door.is_open_inward:
        mid = ((door.p0[0] + door.p1[0]) / 2, (door.p0[1] + door.p1[1]) / 2)
        tip = (mid[0] + n_in[0] * door.width * 0.9, mid[1] + n_in[1] * door.width * 0.9)
        d.line([P(mid), P(tip)], fill="#CC2222", width=2)

    # fridge clearance band: draw ONLY the effective (room-side) face band, so the
    # picture never shows a band sticking out through the wall
    by_name = {it.name: it for it in problem.items}
    for p in placements:
        it = by_name.get(p.get("name"))
        if it is None or it.kind != "fridge":
            continue
        ang = float(p["rotation"])
        length, width = float(p["length"]), float(p["width"])
        center = (float(p["center"][0]), float(p["center"][1]))
        corners = geom.rect_corners(center, length, width, ang)
        u = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
        v = (-u[1], u[0])
        chosen = None
        for sign in (1, -1):
            cA, cB = (corners[2], corners[3]) if sign == 1 else (corners[0], corners[1])
            off = geom.mul(geom.mul(v, sign), length / 2.0)
            band = [cA, cB, geom.add(cB, off), geom.add(cA, off)]
            if chosen is None and geom.rect_inside_polygon(band, room.poly, 1e-2):
                chosen = band          # the room-side face = the effective clearance
        if chosen is not None:
            d.polygon([P(x) for x in chosen], outline="#F28E2B", fill="#FCE7C8")

    # items
    for p in placements:
        it = by_name.get(p.get("name"))
        kind = it.kind if it else "unknown"
        fill, outline = TYPE_COLORS.get(kind, TYPE_COLORS["unknown"])
        center = (float(p["center"][0]), float(p["center"][1]))
        corners = geom.rect_corners(center, float(p["length"]), float(p["width"]), float(p["rotation"]))
        d.polygon([P(x) for x in corners], outline=outline, fill=fill)
        # label
        cx, cy = P(center)
        txt = "%s\n%sx%s" % (p.get("name"), int(float(p["length"])), int(float(p["width"])))
        d.text((cx - 20, cy - 12), txt, fill="#111111")

    # legend
    ly = H - 96
    d.text((10, ly), "房间轮廓 | 门/内开区 | 物品(冰箱=蓝/制冰机=紫/货架=绿/离地架=黄) | 橙=冰箱开门净空(生效侧，画在房间内的那条)", fill="#333333")

    img.save(out_png)
    print("rendered", out_png)
