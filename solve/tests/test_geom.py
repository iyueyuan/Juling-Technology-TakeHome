import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solve import geom, model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT = os.path.join(ROOT, "Input")


def _load(k):
    with open(os.path.join(INPUT, "example%d.json" % k), encoding="utf-8") as fh:
        return json.load(fh)


class TestGeom(unittest.TestCase):
    def test_example4_cleanup_removes_degenerate_spike(self):
        data = _load(4)
        poly, report = geom.polygon_cleanup([tuple(map(float, p)) for p in data["boundary"]])
        self.assertEqual(len(poly), 10)
        self.assertTrue(any("spike" in r for r in report))
        self.assertLess(abs(geom.signed_area(poly) - 6572952.24), 1.0)
        n = len(poly)
        for i in range(n):
            for j in range(i + 1, n):
                if j == i or (j + 1) % n == i or (i + 1) % n == j:
                    continue
                self.assertFalse(geom.segments_properly_intersect(
                    poly[i], poly[(i + 1) % n], poly[j], poly[(j + 1) % n], 1e-6))

    def test_interior_normal_for_clockwise_polygon(self):
        for k in (1, 3):
            p = model.load_problem(os.path.join(INPUT, "example%d.json" % k))
            mid = ((p.door.p0[0] + p.door.p1[0]) / 2, (p.door.p0[1] + p.door.p1[1]) / 2)
            probe = (mid[0] + p.door.inward[0] * 5, mid[1] + p.door.inward[1] * 5)
            self.assertTrue(geom.point_strict_inside(probe, p.room.poly))

    def test_convex_overlap_area(self):
        a = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        b = [(5.0, 0.0), (15.0, 0.0), (15.0, 10.0), (5.0, 10.0)]
        self.assertAlmostEqual(geom.convex_overlap_area(a, b), 50.0, places=6)
        c = [(10.0, 0.0), (20.0, 0.0), (20.0, 10.0), (10.0, 10.0)]
        self.assertLess(geom.convex_overlap_area(a, c), 1e-9)
        d = [(30.0, 30.0), (40.0, 30.0), (40.0, 40.0), (30.0, 40.0)]
        self.assertLess(geom.convex_overlap_area(a, d), 1e-9)

    def test_rect_inside_polygon(self):
        poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        inside = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]
        self.assertTrue(geom.rect_inside_polygon(inside, poly))
        flush = [(90.0, 0.0), (100.0, 0.0), (100.0, 10.0), (90.0, 10.0)]
        self.assertTrue(geom.rect_inside_polygon(flush, poly))
        poke = [(-1.0, 0.0), (10.0, 0.0), (10.0, 10.0), (-1.0, 10.0)]
        self.assertFalse(geom.rect_inside_polygon(poke, poly, 1e-6))

    def test_rect_inside_polygon_rejects_slit_crossing(self):
        """Regression: a 100mm slit (like example4's) must reject a rect straddling it.

        Every corner/edge-midpoint of this rectangle lies on or inside the polygon,
        and no edge properly crosses, so only the "polygon edge cuts the rectangle
        interior" check can catch it.
        """
        poly = [(0.0, 0.0), (200.0, 0.0), (200.0, 100.0),
                (150.0, 100.0), (150.0, 200.0), (100.0, 200.0),
                (100.0, 100.0), (0.0, 100.0)]          # 50mm-wide slit from (100,100) to (150,100)
        # left edge collinear with the slit wall (x=100), straddles the slit
        straddle = [(100.0, 100.0), (300.0, 100.0), (300.0, 300.0), (100.0, 300.0)]
        self.assertFalse(geom.rect_inside_polygon(straddle, poly, 1e-6))
        # fully inside the left lobe is fine
        inside = [(10.0, 10.0), (90.0, 10.0), (90.0, 90.0), (10.0, 90.0)]
        self.assertTrue(geom.rect_inside_polygon(inside, poly, 1e-6))
        # flush against the slit wall but NOT crossing it is fine
        flush = [(50.0, 10.0), (100.0, 10.0), (100.0, 90.0), (50.0, 90.0)]
        self.assertTrue(geom.rect_inside_polygon(flush, poly, 1e-6))

    def test_example4_slit_straddling_placement_is_rejected(self):
        """The exact placement that used to slip through in example4 (shelf-1)."""
        p = model.load_problem(os.path.join(INPUT, "example4.json"))
        cs = geom.rect_corners((184573.592, 30142.723), 1000.0, 400.0, 90.0)
        self.assertFalse(geom.rect_inside_polygon(cs, p.room.poly, 1e-2))

    def test_wall_contact(self):
        poly = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
        # touches only the bottom wall (length 40)
        rect = [(10.0, 0.0), (50.0, 0.0), (50.0, 10.0), (10.0, 10.0)]
        total, _ = geom.wall_contact(rect, poly, 1e-3)
        self.assertLess(abs(total - 40.0), 1e-3)
        # a corner rect touches two walls: bottom 40 + left 10
        rect2 = [(0.0, 0.0), (40.0, 0.0), (40.0, 10.0), (0.0, 10.0)]
        total2, per2 = geom.wall_contact(rect2, poly, 1e-3)
        self.assertLess(abs(total2 - 50.0), 1e-3)
        self.assertEqual(len(per2), 2)


if __name__ == "__main__":
    unittest.main()
