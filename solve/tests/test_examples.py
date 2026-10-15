import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solve import model, solver, validator
from solve.tests.helpers import load, serialize, solve_cached


def _fridge_clearance(problem):
    for it in problem.items:
        if it.kind == "fridge":
            return it.length / 2.0
    return 0.0


class TestEndToEnd(unittest.TestCase):
    def test_all_examples_feasible_valid_and_wall_hugging(self):
        """Default config: every example feasible, valid, and fully wall-hugging."""
        for k in (1, 2, 3, 4):
            with self.subTest(example=k):
                problem, result, out = solve_cached(k)
                self.assertTrue(result.feasible, "example%d not feasible" % k)
                vr = validator.validate(problem, out)
                self.assertTrue(vr["passed"], "example%d failed: %s" % (k, vr["checks"]))
                self.assertEqual(vr["metrics"]["itemsTouchingWall"], len(problem.items))
                self.assertAlmostEqual(vr["metrics"]["fridgeClearanceMm"], _fridge_clearance(problem), places=6)

    def test_axis_orientation_costs_wall_contact_on_slanted_room(self):
        """example1 has slanted walls: strict 0/90 cannot hug them."""
        p_auto, _, out_auto = solve_cached(1, orientation="auto", dfs_nodes=0)
        p_axis, _, out_axis = solve_cached(1, orientation="axis", dfs_nodes=0)
        v_auto = validator.validate(p_auto, out_auto)["metrics"]["itemsTouchingWall"]
        v_axis = validator.validate(p_axis, out_axis)["metrics"]["itemsTouchingWall"]
        self.assertGreater(v_auto, v_axis)
        self.assertEqual(v_auto, len(p_auto.items))

    def test_determinism(self):
        for k in (1, 4):
            with self.subTest(example=k):
                problem, r1, out1 = solve_cached(k)
                r2 = solver.solve(problem, solver.Config(time_limit=30,
                                                         orientation="auto", dfs_nodes=15000))
                self.assertEqual(out1, serialize(r2))

    def test_orientation_axis_restricts_to_multiples_of_90(self):
        """axis mode must only produce 0/90 angles."""
        for k in (1, 2, 3, 4):
            with self.subTest(example=k):
                problem, result, out = solve_cached(k, orientation="axis", dfs_nodes=0)
                self.assertTrue(result.feasible)
                for pl in result.placements:
                    a = pl.angle % 180.0
                    dist_to_axis = min(a, 180.0 - a, abs(a - 90.0))
                    self.assertLess(dist_to_axis, 0.5,
                                    "axis mode produced %s deg for %s" % (pl.angle, pl.item.name))


class TestInfeasibleInput(unittest.TestCase):
    """The last-resort path (shuffled DFS retries) must report feasible=false."""

    def test_infeasible_input_reports_false_instead_of_crashing(self):
        data = {
            "boundary": [[0, 0], [1000, 0], [1000, 1000], [0, 1000]],
            "door": [[400, 1000], [600, 1000]],
            "isOpenInward": False,
            "algoToPlace": {"fridge": [2000, 2000]},   # far too big for the room
        }
        problem = model.parse_problem(data, "infeasible-test")
        result = solver.solve(problem, solver.Config(time_limit=10))
        self.assertFalse(result.feasible)
        self.assertEqual(result.placements, [])
        self.assertIsNotNone(result.reason)


if __name__ == "__main__":
    unittest.main()
