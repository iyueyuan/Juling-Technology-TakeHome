"""CLI helpers and (de)serialisation shared by the entry-point scripts."""
from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List, Optional

from . import model, render, solver, validator
from .model import Problem


def make_config(args) -> solver.Config:
    cfg = solver.Config(
        orientation=getattr(args, "orientation", "auto"),
        fridge_clearance=getattr(args, "fridge_front_clearance_mm", None),
        time_limit=getattr(args, "time_limit", 25.0),
        beam=getattr(args, "beam", 40),
        branch=getattr(args, "branch", 50),
        seed=getattr(args, "seed", 0),
    )
    if hasattr(args, "wall_step"):
        cfg.wall_step = args.wall_step
    if hasattr(args, "grid_step"):
        cfg.grid_step = args.grid_step
    return cfg


def serialize_result(result: solver.Result, problem: Problem) -> Dict:
    placements = []
    for pl in result.placements:
        placements.append({
            "name": pl.item.name,
            "type": pl.item.kind,
            "center": [round(pl.center[0], 3), round(pl.center[1], 3)],
            "rotation": round(pl.angle, 4),
            "length": pl.item.length,
            "width": pl.item.width,
        })
    return {
        "feasible": result.feasible,
        "reason": result.reason,
        "unit": "mm",
        "clearanceMode": result.clearance_mode,
        "angleConvention": "deg; 0 deg = length along +x, width along +y; counter-clockwise positive",
        "placements": placements,
    }


def _add_common_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--orientation", choices=["auto", "axis"], default="auto")
    ap.add_argument("--fridge-front-clearance-mm", type=float, default=None)
    ap.add_argument("--beam", type=int, default=40)
    ap.add_argument("--branch", type=int, default=50)
    ap.add_argument("--time-limit", type=float, default=25.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--wall-step", type=float, default=100.0)
    ap.add_argument("--grid-step", type=float, default=50.0)


def solve_one(input_path: str, output_path: str, args) -> Optional[Dict]:
    problem = model.load_problem(input_path)
    cfg = make_config(args)
    result = solver.solve(problem, cfg)
    out = serialize_result(result, problem)
    if output_path:
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False, indent=2)
    return out


def validate_one(problem: Problem, result: Dict, fridge_clearance: Optional[float] = None) -> Dict:
    return validator.validate(problem, result, fridge_clearance)


def render_one(problem: Problem, result: Dict, png_path: str) -> None:
    if not png_path:
        return
    os.makedirs(os.path.dirname(os.path.abspath(png_path)), exist_ok=True)
    render.render_layout(problem, result.get("placements", []), png_path)


def _input_files(input_dir: str) -> List[str]:
    return sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".json") and not f.startswith("."))


def run_examples(input_dir: str, output_dir: str, args) -> int:
    files = _input_files(input_dir)
    summary_rows = []
    report: Dict[str, Dict] = {}
    exit_code = 0
    for f in files:
        stem = os.path.splitext(f)[0]
        input_path = os.path.join(input_dir, f)
        result_path = os.path.join(output_dir, stem + ".result.json")
        png_path = os.path.join(output_dir, "figures", stem + ".layout.png")
        print("=== %s ===" % f)
        problem = model.load_problem(input_path)
        out = solve_one(input_path, result_path, args)
        vr = validate_one(problem, out, args.fridge_front_clearance_mm)
        render_one(problem, out, png_path)
        report[stem] = {"input": f, "result": stem + ".result.json", "validation": vr}
        ok = vr["passed"]
        if not ok:
            exit_code = 1
        row = {
            "example": f,
            "feasible": out["feasible"],
            "mode": out["clearanceMode"],
            "validation": "PASS" if ok else "FAIL",
            "items": "%d/%d" % (len(out["placements"]), len(problem.items)),
            "itemsOnWall": "%d/%d" % (vr["metrics"]["itemsTouchingWall"], len(out["placements"])),
            "wallContactMm": vr["metrics"]["wallContactMm"],
            "fridgeClearanceMm": vr["metrics"]["fridgeClearanceMm"],
        }
        summary_rows.append(row)
        print("   feasible=%s validation=%s itemsOnWall=%d/%d" % (
            out["feasible"], "PASS" if ok else "FAIL",
            vr["metrics"]["itemsTouchingWall"], len(out["placements"])))

    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "validation_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, ensure_ascii=False, indent=2)
    _write_validation_md(report, os.path.join(output_dir, "validation.md"))
    _write_summary_md(summary_rows, os.path.join(output_dir, "summary.md"))
    print("\nvalidation report -> %s" % os.path.join(output_dir, "validation_report.json"))
    print("summary          -> %s" % os.path.join(output_dir, "summary.md"))
    return exit_code


def _write_validation_md(report: Dict, path: str) -> None:
    lines = ["# 校验报告 (validation)", ""]
    for stem, r in report.items():
        vr = r["validation"]
        lines += ["## %s" % stem, "", "校验结果: **%s**" % ("通过" if vr["passed"] else "未通过"), ""]
        lines += ["| 规则 | 结论 | 说明 |", "|---|---|---|"]
        for c in vr["checks"]:
            lines.append("| %s | %s | %s |" % (c["id"], c["status"], c["detail"]))
        lines += ["", "指标: 贴墙 %d 件, 贴墙总长 %s mm, 自由面积 %.2f m², 冰箱净空 %s mm" % (
            vr["metrics"]["itemsTouchingWall"], vr["metrics"]["wallContactMm"],
            vr["metrics"]["freeAreaMm2"] / 1e6, vr["metrics"]["fridgeClearanceMm"]), ""]
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def _write_summary_md(rows: List[Dict], path: str) -> None:
    lines = ["# 结果汇总 (summary)", ""]
    lines += ["| 样例 | 可行 | 净空口径 | 校验 | 物品 | 贴墙 | 贴墙总长(mm) | 冰箱净空(mm) |",
              "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (
            r["example"], "是" if r["feasible"] else "否", r["mode"], r["validation"],
            r["items"], r["itemsOnWall"], r["wallContactMm"], r["fridgeClearanceMm"]))
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


def main_solve() -> None:
    ap = argparse.ArgumentParser(description="Solve a storage-room layout")
    ap.add_argument("input", help="input JSON or a directory (with --batch)")
    ap.add_argument("-o", "--output", default=None)
    ap.add_argument("--batch", action="store_true", help="treat input as a directory")
    ap.add_argument("--render-dir", default=None, help="directory for layout PNGs")
    _add_common_args(ap)
    args = ap.parse_args()
    if args.batch:
        outdir = args.output or "Output"
        exit(run_examples(args.input, outdir, args))
    out = solve_one(args.input, args.output, args)
    print(json.dumps(out, ensure_ascii=False, indent=2))
    if args.render_dir:
        render_one(model.load_problem(args.input), out, os.path.join(args.render_dir,
                   os.path.splitext(os.path.basename(args.input))[0] + ".layout.png"))


def main_validate() -> None:
    ap = argparse.ArgumentParser(description="Independently validate result JSON(s)")
    ap.add_argument("input_dir")
    ap.add_argument("output_dir")
    ap.add_argument("--fridge-front-clearance-mm", type=float, default=None)
    args = ap.parse_args()
    files = _input_files(args.input_dir)
    exit_code = 0
    for f in files:
        stem = os.path.splitext(f)[0]
        result_path = os.path.join(args.output_dir, stem + ".result.json")
        if not os.path.exists(result_path):
            print("%s: missing result file -> FAIL" % f)
            exit_code = 1
            continue
        problem = model.load_problem(os.path.join(args.input_dir, f))
        with open(result_path, "r", encoding="utf-8") as fh:
            out = json.load(fh)
        vr = validate_one(problem, out, args.fridge_front_clearance_mm)
        status = "PASS" if vr["passed"] else "FAIL"
        print("%s: %s" % (f, status))
        for c in vr["checks"]:
            print("   %s %s  %s" % (c["id"], c["status"], c["detail"]))
        if not vr["passed"]:
            exit_code = 1
    exit(exit_code)


def main_run_examples() -> None:
    ap = argparse.ArgumentParser(description="Solve + validate + render all inputs")
    ap.add_argument("--input", default="Input")
    ap.add_argument("--output", default="Output")
    _add_common_args(ap)
    args = ap.parse_args()
    exit(run_examples(args.input, args.output, args))
