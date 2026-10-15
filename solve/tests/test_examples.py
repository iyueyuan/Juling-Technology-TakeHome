import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solve import model, solver, validator

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT = os.path.join(ROOT, "Input")


def _serialize(result):
    return {
        "feasible": result.feasible,
        "reason": result.reason,
        "unit": "mm",
        "placements": [
            {"name": p.item.name, "type": p.item.kind,
             "center": [round(p.center[0], 3), round(p.center[1], 3)],
             "rotation": round(p.angle, 4),
             "length": p.item.length, "width": p.item.width}
            for p in result.placements
        ],
    }


def _fridge_clearance(problem):
    for it in problem.items:
        if it.kind == "fridge":
            return it.length / 2.0
    return 0.0


class TestEndToEnd(unittest.TestCase):
    def test_all_examples_feasible_and_valid(self):
        for k in (1, 2, 3, 4):
            with self.subTest(example=k):
                problem = model.load_problem(os.path.join(INPUT, "example%d.json" % k))
                result = solver.solve(problem, solver.Config(time_limit=30))
                self.assertTrue(result.feasible, "example%d not feasible" % k)
                vr = validator.validate(problem, _serialize(result))
                self.assertTrue(vr["passed"], "example%d failed: %s" % (k, vr["checks"]))
                self.assertEqual(vr["metrics"]["itemsTouchingWall"], len(problem.items))
                self.assertAlmostEqual(vr["metrics"]["fridgeClearanceMm"], _fridge_clearance(problem), places=6)

    def test_determinism(self):
        for k in (1, 2, 3, 4):
            with self.subTest(example=k):
                problem = model.load_problem(os.path.join(INPUT, "example%d.json" % k))
                r1 = solver.solve(problem, solver.Config(time_limit=30, seed=0))
                r2 = solver.solve(problem, solver.Config(time_limit=30, seed=0))
                self.assertEqual(_serialize(r1), _serialize(r2))


if __name__ == "__main__":
    unittest.main()
