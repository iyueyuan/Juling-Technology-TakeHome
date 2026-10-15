import copy
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solve import geom, model, solver, validator
from solve.tests.helpers import load, serialize, solve_cached

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT = os.path.join(ROOT, "Input")


def _solve(k):
    """Cached solve (deep-copied so tests may mutate the result freely)."""
    problem, _result, out = solve_cached(k, orientation="auto", dfs_nodes=0)
    return problem, copy.deepcopy(out)


def _status(vr, cid):
    for c in vr["checks"]:
        if c["id"] == cid:
            return c["status"]
    return None


class TestValidatorNegative(unittest.TestCase):
    def test_valid_solution_passes(self):
        problem, out = _solve(2)
        vr = validator.validate(problem, out)
        self.assertTrue(vr["passed"])

    def test_overlap_is_detected(self):
        problem, out = _solve(2)
        bad = copy.deepcopy(out)
        p1 = next(p for p in bad["placements"] if p["name"] == "shelf-1")
        p2 = next(p for p in bad["placements"] if p["name"] == "shelf-2")
        p1["center"] = p2["center"][:]
        p1["rotation"] = p2["rotation"]
        self.assertEqual(_status(validator.validate(problem, bad), "V3"), "FAIL")

    def test_outside_is_detected(self):
        problem, out = _solve(2)
        bad = copy.deepcopy(out)
        p = next(p for p in bad["placements"] if p["name"] == "shelf-1")
        p["center"] = [p["center"][0] + 5000.0, p["center"][1]]
        self.assertEqual(_status(validator.validate(problem, bad), "V2"), "FAIL")

    def test_bad_angle_is_detected(self):
        problem, out = _solve(2)
        bad = copy.deepcopy(out)
        p = next(p for p in bad["placements"] if p["name"] == "shelf-1")
        p["rotation"] = 45.0
        self.assertEqual(_status(validator.validate(problem, bad), "V4"), "FAIL")

    def test_door_blocked_is_detected(self):
        problem = model.load_problem(os.path.join(INPUT, "example1.json"))
        out = {"feasible": True, "placements": [{
            "name": "shelf-1", "type": "shelf",
            "center": [7121.7357, 32246.0], "rotation": 90.0,
            "length": 1000.0, "width": 400.0}]}
        self.assertEqual(_status(validator.validate(problem, out), "V5"), "FAIL")

    def test_fridge_front_blocked_is_detected(self):
        problem, out = _solve(2)
        bad = copy.deepcopy(out)
        fridge = next(p for p in bad["placements"] if p["type"] == "fridge")
        corners = geom.rect_corners((fridge["center"][0], fridge["center"][1]),
                                    fridge["length"], fridge["width"], fridge["rotation"])
        ang = fridge["rotation"]
        depth = fridge["length"] / 2.0
        u = (math.cos(math.radians(ang)), math.sin(math.radians(ang)))
        v = (-u[1], u[0])
        shelf = None
        for sign in (1, -1):
            cA, cB = (corners[2], corners[3]) if sign == 1 else (corners[0], corners[1])
            n = (v[0] * sign, v[1] * sign)
            band = [cA, cB, (cB[0] + n[0] * depth, cB[1] + n[1] * depth),
                    (cA[0] + n[0] * depth, cA[1] + n[1] * depth)]
            if geom.rect_inside_polygon(band, problem.room.poly, 1e-2):
                shelf = {"name": "shelf-1", "type": "shelf",
                         "center": [round((cA[0] + cB[0]) / 2 + n[0] * depth / 2, 3),
                                    round((cA[1] + cB[1]) / 2 + n[1] * depth / 2, 3)],
                         "rotation": 90.0, "length": 1000.0, "width": 400.0}
                break
        self.assertIsNotNone(shelf)
        bad["placements"].append(shelf)
        self.assertEqual(_status(validator.validate(problem, bad), "V6"), "FAIL")


if __name__ == "__main__":
    unittest.main()
