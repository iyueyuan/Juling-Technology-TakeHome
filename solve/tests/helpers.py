"""Shared helpers for the test modules (caches solves so the suite stays fast)."""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from solve import model, solver

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
INPUT = os.path.join(ROOT, "Input")

_CACHE = {}


def load(k):
    return model.load_problem(os.path.join(INPUT, "example%d.json" % k))


def serialize(result):
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


def solve_cached(k, orientation="auto", dfs_nodes=15000, time_limit=30.0):
    """Solve example k once per (orientation, dfs_nodes) and cache the outcome."""
    key = (k, orientation, dfs_nodes)
    if key not in _CACHE:
        problem = load(k)
        cfg = solver.Config(time_limit=time_limit, orientation=orientation, dfs_nodes=dfs_nodes)
        result = solver.solve(problem, cfg)
        _CACHE[key] = (problem, result, serialize(result))
    return _CACHE[key]
