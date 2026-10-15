import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solve import geom, render
from solve.tests.helpers import load, solve_cached

try:
    import PIL  # noqa: F401
    HAVE_PIL = True
except ImportError:
    HAVE_PIL = False

PAD, MAX_DIM = 60.0, 1400
ORANGE = [(252, 231, 200), (242, 142, 43)]        # fridge clearance band
RED = [(232, 102, 102), (204, 34, 34), (245, 218, 218), (204, 68, 68)]  # door / N x N zone


def _near(px, target, tol=14):
    return all(abs(px[i] - target[i]) <= tol for i in range(3))


@unittest.skipUnless(HAVE_PIL, "Pillow not installed")
class TestRenderInsideOutline(unittest.TestCase):
    """Regression: the layout PNG must not draw the door marker or the fridge
    clearance band outside the room outline (that made the picture misleading)."""

    def _check(self, k):
        from PIL import Image
        problem, _result, out = solve_cached(k, dfs_nodes=0)
        x0, y0, x1, y1 = problem.room.bbox
        with tempfile.TemporaryDirectory() as td:
            png = os.path.join(td, "layout.png")
            render.render_layout(problem, out["placements"], png)
            img = Image.open(png).convert("RGB")
            W, H = img.size
            px = img.load()
            scale = max(min(MAX_DIM / (x1 - x0 + 2 * PAD), MAX_DIM / (y1 - y0 + 2 * PAD)), 0.02)
            outside = {"orange": 0, "red": 0}
            stride = 3   # sampling: any real overhang is thousands of pixels wide
            for yy in range(0, H, stride):
                for xx in range(0, W, stride):
                    c = px[xx, yy]
                    is_o = any(_near(c, t) for t in ORANGE)
                    is_r = any(_near(c, t) for t in RED)
                    if not (is_o or is_r):
                        continue
                    # pixel -> mm, with a tolerance of one pixel size
                    mm = (xx / scale - PAD + x0, yy / scale - PAD + y0)
                    if not geom.point_in_polygon(mm, problem.room.poly, 1.0 / scale + 1.0):
                        if is_o:
                            outside["orange"] += 1
                        if is_r:
                            outside["red"] += 1
        return outside

    def test_no_band_or_door_outside_outline(self):
        for k in (1, 2, 3, 4):
            with self.subTest(example=k):
                outside = self._check(k)
                self.assertEqual(outside["orange"], 0, "fridge band drawn outside the room")
                self.assertEqual(outside["red"], 0, "door marker drawn outside the room")


if __name__ == "__main__":
    unittest.main()
