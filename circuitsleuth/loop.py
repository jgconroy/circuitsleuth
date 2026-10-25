"""Closed-loop ("self-driving") spectroscopy on a digital twin.

Start from a partial sweep, identify, ask the experimental-design module for the most informative next sweep,
"measure" it (take those columns from the full map, which stands in for the instrument), refit, repeat.
`measure()` is the one function a lab swaps for an instrument driver.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

from . import agent, design, extract, fit, models


def _narrow_bounds(params: dict) -> dict:
    """Bounds for the next round's refit: the previous solution is a strong prior."""
    b = {}
    for k, v in params.items():
        if k == "d":
            b[k] = (max(0.0, v - 0.15), min(0.9, v + 0.15))
        elif k == "period":
            b[k] = (0.9 * v, 1.1 * v)
        elif k in ("offset", "x0"):
            b[k] = (v - 0.1 * abs(params.get("period", 1.0)), v + 0.1 * abs(params.get("period", 1.0)))
        elif k in ("theta", "phi", "f0"):
            b[k] = (v - 0.2, v + 0.2)
        elif v > 0:
            b[k] = (v / 1.5, v * 1.5)
    return b


def measure(twin_points: np.ndarray, x_lo: float, x_hi: float, x_grid: np.ndarray) -> np.ndarray:
    """Return the digital twin's features in the columns [x_lo, x_hi] (one half-step margin each side)."""
    dx = float(np.median(np.diff(x_grid))) if len(x_grid) > 1 else 0.0
    m = (twin_points[:, 0] >= x_lo - 0.5 * dx) & (twin_points[:, 0] <= x_hi + 0.5 * dx)
    return twin_points[m]


def run_loop(image, x, f, families: tuple, rounds: int = 3, initial_frac: float = 0.4, band=None, progress: Callable | None = None,
             cfg_kwargs: dict | None = None) -> dict:
    """Returns dict(rounds=[{round, n_points, columns_measured, params, sigma, hypothesis, block, runtime_s, lines, visible_x}], twin_points)."""
    x = np.asarray(x, float)
    band = band or (float(f[0]), float(f[-1]))
    twin = extract.extract_points(image, x, f)
    cols = np.unique(twin[:, 0])
    n0 = max(8, int(round(initial_frac * len(cols))))
    visible = set(float(v) for v in cols[:n0])
    platform = models.platform_of(families[0])
    base = dict(use_classifier=platform == "circuit", families=families, try_coupled="auto" if platform == "circuit" else "never", holdout=True)
    base.update(cfg_kwargs or {})
    records = []
    prev_best = None
    for r in range(rounds + 1):
        t0 = time.time()
        pts = twin[np.isin(twin[:, 0], sorted(visible))]
        if progress:
            progress("round_start", dict(round=r, n_points=int(len(pts)), n_columns=len(visible)))
        cfg_r = dict(base)
        if prev_best is not None:  # later rounds refine, they do not re-open the family question
            cfg_r.update(families=(prev_best["family"],), try_coupled="always" if prev_best.get("coupled") else "never", use_classifier=False,
                         param_bounds=_narrow_bounds(prev_best["params"]), n_starts=1)
        rep = agent.identify(points=pts, x=x, band=band, config=agent.AgentConfig(**cfg_r))
        if rep.get("status") != "ok":
            records.append(dict(round=r, error=rep.get("error"), n_points=int(len(pts))))
            break
        best = rep["best"]
        prev_best = best
        try:  # design restricted to columns not measured yet
            dsg = design.design_next_measurements(rep, band=band, measured_x=sorted(visible))
            rep["design"] = dsg
        except Exception:
            dsg = rep.get("design") or {}
        cands = dsg.get("candidates") or []
        block = None
        for c in cands:
            new_cols = [v for v in cols if c["x_lo"] - 1e-9 <= v <= c["x_hi"] + 1e-9 and float(v) not in visible]
            if len(new_cols) >= 3:
                block = dict(c, new_columns=len(new_cols))
                break
        rec = dict(round=r, n_points=int(len(pts)), columns_measured=len(visible), hypothesis=best["hypothesis"], params=best["params"], sigma=best["sigma"],
                   explained=best.get("explained_dynamic"), holdout=(best.get("holdout") or {}).get("inlier_frac"), block=block, runtime_s=round(time.time() - t0, 1),
                   lines=rep["lines"], x_grid=rep["x_grid"], visible_x=sorted(visible), inlier_mask=rep["inlier_mask"], points=rep["points"],
                   correlation=(rep.get("design") or {}).get("correlation"))
        records.append(rec)
        if progress:
            progress("round_done", rec)
        if block is None or r == rounds:
            break
        for v in cols:
            if block["x_lo"] - 1e-9 <= v <= block["x_hi"] + 1e-9:
                visible.add(float(v))
    return dict(rounds=records, twin_points=twin.tolist(), families=list(families), band=list(band))


def sigma_table(result: dict) -> list[dict]:
    """Per-round relative uncertainties, for the UI and for tests."""
    out = []
    for rec in result["rounds"]:
        if "params" not in rec:
            continue
        row = dict(round=rec["round"], n_points=rec["n_points"], columns=rec["columns_measured"])
        for k, v in rec["params"].items():
            s = rec["sigma"].get(k, float("nan"))
            row[k] = v
            row[k + "_rel_sigma"] = float(s / abs(v)) if v else float("nan")
        out.append(row)
    return out
