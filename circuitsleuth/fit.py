"""Robust identification of a circuit hypothesis from a spectroscopy point cloud.

Each hypothesis = (family, with/without readout resonator). Fitted variables are the circuit
parameters (log-space for energies) plus, for flux-tunable circuits, the flux calibration
(period and offset of the control axis in 'volts').  The loss is Geman-McClure on the distance of
every observed feature to the nearest predicted line, so spurious features cost a bounded amount
and are reported as unexplained instead of distorting the fit.

Pipeline: subsample -> differential evolution (global) -> least_squares (local, soft-L1) ->
inlier/outlier split -> hold-out verification on flux columns never used in the local fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import differential_evolution, least_squares

from . import models

LOG_PARAMS = {"EJ", "EC", "EL", "EJmax", "fr", "g", "period", "field_scale", "lever_arm", "t_c", "E"}
MODEL_KW = {"ncut": 20, "cutoff": 60, "kq": 5, "kr": 3}


@dataclass(frozen=True)
class Hypothesis:
    family: str
    coupled: bool = False

    @property
    def flux_dependent(self) -> bool:  # kept for backward compatibility: "depends on the control axis"
        return models.FAMILIES[self.family]["flux_dependent"]

    control_dependent = flux_dependent

    @property
    def platform(self) -> str:
        return models.platform_of(self.family)

    @property
    def names(self) -> list[str]:
        return models.hypothesis_params(self.family, self.coupled) + list(models.control_params(self.family))

    @property
    def label(self) -> str:
        return self.family.replace("_", " ") + (" + readout resonator" if self.coupled else "")


ALL_HYPOTHESES = [Hypothesis(f, c) for f in models.FAMILIES for c in (False, True)]


# ----------------------------------------------------------------------------- parameterisation
def default_bounds(hyp: Hypothesis, x: np.ndarray) -> dict[str, tuple[float, float]]:
    b = dict(models.hypothesis_bounds(hyp.family, hyp.coupled))
    span = float(np.max(x) - np.min(x))
    ctrl = models.control_params(hyp.family)
    if ctrl == ("period", "offset"):
        pmax = span / 0.4
        b["period"] = (span / 3.0, pmax)
        b["offset"] = (float(np.min(x)) - pmax / 2, float(np.max(x)) + pmax / 2)
    elif ctrl == ("field_scale", "x0"):
        b["field_scale"] = models.CONTROL_BOUNDS["field_scale"]
        b["x0"] = (float(np.min(x)) - span, float(np.max(x)) + span)
    elif ctrl == ("lever_arm", "x0"):
        b["lever_arm"] = models.CONTROL_BOUNDS["lever_arm"]
        b["x0"] = (float(np.min(x)) - span / 2, float(np.max(x)) + span / 2)
    return b


def to_theta(params: dict, names: list[str]) -> np.ndarray:
    return np.array([np.log(params[n]) if n in LOG_PARAMS else params[n] for n in names], float)


def from_theta(theta: np.ndarray, names: list[str]) -> dict:
    return {n: (float(np.exp(t)) if n in LOG_PARAMS else float(t)) for n, t in zip(names, theta)}


def to_theta_partial(init: dict | None, names: list[str], tb: list[tuple[float, float]]) -> np.ndarray:
    """theta from a possibly partial init; missing entries start at the middle of their bounds."""
    th = []
    for n, (lo, hi) in zip(names, tb):
        if init is not None and n in init:
            v = np.log(init[n]) if n in LOG_PARAMS else init[n]
        else:
            v = 0.5 * (lo + hi)
        th.append(float(np.clip(v, lo, hi)))
    return np.array(th)


def theta_bounds(bounds: dict, names: list[str]) -> list[tuple[float, float]]:
    out = []
    for n in names:
        lo, hi = bounds[n]
        if lo > hi:  # defensive: an inverted narrowed interval collapses to a tiny one around its midpoint
            lo, hi = hi, lo
        if hi - lo < 1e-6 * max(abs(hi), 1.0):
            lo, hi = lo - 1e-6 * max(abs(lo), 1.0), hi + 1e-6 * max(abs(hi), 1.0)
        out.append((np.log(max(lo, 1e-9)), np.log(max(hi, 1e-9))) if n in LOG_PARAMS else (lo, hi))
    return out


def split_params(params: dict, hyp: Hypothesis):
    ctrl = set(models.control_params(hyp.family))
    circuit = {k: v for k, v in params.items() if k not in ctrl}
    period = params.get("period", 1.0)
    offset = params.get("offset", 0.0)
    return circuit, period, offset


def predict_lines(hyp: Hypothesis, params: dict, x: np.ndarray, model_kw: dict | None = None) -> dict[str, np.ndarray]:
    circuit, _, _ = split_params(params, hyp)
    ctrl_val = models.control_from_x(hyp.family, params, np.asarray(x, float))
    kw = dict(MODEL_KW)
    kw.update(model_kw or {})
    return models.transitions(hyp.family, circuit, ctrl_val, coupled=hyp.coupled, **kw)


# ----------------------------------------------------------------------------- objective
class Objective:
    """Callable loss on a fixed point cloud; picklable so differential_evolution can use workers."""

    def __init__(self, hyp: Hypothesis, points: np.ndarray, band: tuple[float, float], c: float, model_kw: dict | None = None, line_filter: list[str] | None = None):
        self.hyp = hyp
        self.line_filter = line_filter
        self.names = hyp.names
        self.points = points
        self.x_cols, self.col_idx = np.unique(points[:, 0], return_inverse=True)
        self.f = points[:, 1]
        self.w = points[:, 2] / max(points[:, 2].sum(), 1e-12)
        self.band = band
        self.c = c
        self.model_kw = model_kw

    def line_matrix(self, theta: np.ndarray) -> np.ndarray:
        params = from_theta(theta, self.names)
        lines = predict_lines(self.hyp, params, self.x_cols, self.model_kw)
        keys = [k for k in lines if (self.line_filter is None or k in self.line_filter)]
        L = np.stack([lines[k] for k in keys])  # (nlines, ncols)
        return np.where(np.isfinite(L), L, np.inf)

    def residuals(self, theta: np.ndarray) -> np.ndarray:
        L = self.line_matrix(theta)
        d = np.abs(self.f[None, :] - L[:, self.col_idx])  # (nlines, npoints)
        return d.min(axis=0)

    def __call__(self, theta: np.ndarray) -> float:
        try:
            L = self.line_matrix(theta)
        except (np.linalg.LinAlgError, ValueError):
            return 1e3
        d = np.abs(self.f[None, :] - L[:, self.col_idx]).min(axis=0)
        rho = d**2 / (d**2 + self.c**2)
        loss = float(np.sum(self.w * rho))
        # weak coverage term: the 0-1 line, where it lies inside the measured band, should have data
        l01 = L[0]
        inband = (l01 > self.band[0]) & (l01 < self.band[1])
        if inband.any():
            has_data = np.zeros(len(self.x_cols), bool)
            near = d < 3 * self.c
            # a column "has data" on the 0-1 line if some point sits within 3c of it
            d01 = np.abs(self.f - l01[self.col_idx])
            np.logical_or.at(has_data, self.col_idx[d01 < 3 * self.c], True)
            miss = np.mean(~has_data[inband])
            loss += 0.15 * miss
        return loss


# ----------------------------------------------------------------------------- results
@dataclass
class FitResult:
    hyp: Hypothesis
    params: dict
    theta: np.ndarray
    sigma: dict
    loss: float
    residuals: np.ndarray
    inlier_mask: np.ndarray
    inlier_frac: float
    rms_inliers: float
    x_cols: np.ndarray
    lines: dict
    holdout: dict = field(default_factory=dict)
    history: list = field(default_factory=list)
    n_evals: int = 0

    @property
    def score(self) -> float:
        """Explained fraction on held-out columns (falls back to in-sample) minus a complexity penalty."""
        base = self.holdout.get("inlier_frac", self.inlier_frac)
        return float(base - 0.02 * len(self.params))

    def summary(self) -> dict:
        return dict(
            hypothesis=self.hyp.label,
            family=self.hyp.family,
            coupled=self.hyp.coupled,
            params={k: round(v, 5) for k, v in self.params.items()},
            sigma={k: round(v, 5) for k, v in self.sigma.items()},
            inlier_frac=round(self.inlier_frac, 4),
            rms_inliers_MHz=round(1e3 * self.rms_inliers, 2),
            holdout=self.holdout,
            score=round(self.score, 4),
            loss=round(self.loss, 5),
        )


def subsample(points: np.ndarray, max_cols: int = 41, max_points: int = 3000) -> np.ndarray:
    """Thin the flux columns (cost scales with columns) but keep faint points: they pin EC and EL."""
    cols = np.unique(points[:, 0])
    if len(cols) > max_cols:
        keep = cols[np.linspace(0, len(cols) - 1, max_cols).round().astype(int)]
        points = points[np.isin(points[:, 0], keep)]
    if len(points) > max_points:
        idx = np.argsort(points[:, 2])[::-1][:max_points]
        points = points[np.sort(idx)]
    return points


def _fit_once(
    points: np.ndarray,
    hyp: Hypothesis,
    band: tuple[float, float],
    bounds: dict | None = None,
    init: dict | None = None,
    c_global: float = 0.08,
    c_local: float = 0.012,
    maxiter: int = 60,
    popsize: int = 15,
    seed: int = 0,
    workers: int = 1,
    callback: Callable[[dict, float], None] | None = None,
    model_kw: dict | None = None,
    do_global: bool = True,
    n_starts: int = 1,
    line_filter: list[str] | None = None,
) -> FitResult:
    x_all = np.unique(points[:, 0])
    bounds = bounds or default_bounds(hyp, x_all)
    names = hyp.names
    tb = theta_bounds(bounds, names)
    history: list = []

    # ---- global search on a subsample, annealing the robust scale (coarse -> fine) and carrying
    #      the population between stages so the search narrows instead of restarting
    sub = subsample(points, max_cols=31)
    if hyp.family == "fluxonium":  # harder landscape: more diversity
        popsize = max(popsize, 20)
        maxiter = int(maxiter * 1.5)
    global_kw = dict(model_kw or {})
    global_kw.setdefault("cutoff", 40)  # converged to <1e-8 GHz except at the extreme EC/EL corner; local stage uses 60
    if do_global:
        lo_b = np.array([b[0] for b in tb])
        hi_b = np.array([b[1] for b in tb])
        stages = [(4 * c_global, maxiter // 2), (c_global, maxiter), (max(c_global / 4, 1.5 * c_local), maxiter // 2)]
        population = None
        theta0 = None
        n_evals = 0
        for c_stage, it in stages:
            obj_g = Objective(hyp, sub, band, c_stage, global_kw, line_filter)
            de_kw = dict(bounds=tb, maxiter=it, popsize=popsize, tol=1e-3, seed=seed, polish=False)  # stop a stage once the population has converged
            de_kw["init"] = "sobol" if population is None else population
            if init is not None and population is None:
                x0 = to_theta_partial(init, names, tb)
                eps = 1e-9 * (hi_b - lo_b)
                de_kw["x0"] = np.clip(x0, lo_b + eps, hi_b - eps)
            if workers != 1:
                de_kw.update(workers=workers, updating="deferred")

            def _cb(xk, convergence=0.0, _obj=obj_g):
                p = from_theta(xk, names)
                history.append(p)
                if callback:
                    callback(p, float(_obj(xk)))

            res = differential_evolution(obj_g, callback=_cb, **de_kw)
            population = res.population
            theta0 = res.x
            n_evals += int(res.nfev)
    else:
        theta0 = to_theta_partial(init, names, tb)
        n_evals = 0

    # ---- local refinement on all points
    obj_l = Objective(hyp, points, band, c_local, model_kw, line_filter)
    lo = np.array([b[0] for b in tb])
    hi = np.array([b[1] for b in tb])
    theta0 = np.clip(theta0, lo + 1e-9, hi - 1e-9)
    # EJ-EC degeneracy along constant f01 (transmon families): the global search often settles anywhere
    # on that valley; scan it at the fine scale and restart the polish from the best point
    ej_name = "EJ" if "EJ" in names else ("EJmax" if "EJmax" in names else None)
    if ej_name and "EC" in names:
        i_ej, i_ec = names.index(ej_name), names.index("EC")
        p0 = from_theta(theta0, names)
        f01 = np.sqrt(8 * p0["EC"] * p0[ej_name]) - p0["EC"]
        obj_sweep = Objective(hyp, points, band, 3 * c_local, model_kw, line_filter)
        best_t, best_v = theta0, obj_sweep(theta0)
        for ec in np.exp(np.linspace(np.log(p0["EC"] * 0.5), np.log(p0["EC"] * 2.0), 25)):
            ej = (f01 + ec) ** 2 / (8 * ec)
            t = theta0.copy()
            t[i_ec], t[i_ej] = np.log(ec), np.log(ej)
            if np.all(t >= lo) and np.all(t <= hi):
                v = obj_sweep(t)
                if v < best_v:
                    best_t, best_v = t, v
        theta0 = best_t
    sw = np.sqrt(obj_l.w * len(obj_l.w))

    def resid(theta):
        return sw * obj_l.residuals(theta)

    # two-stage polish: a wider robust scale first pulls in lines that are ~50-100 MHz off, then the fine scale
    ls = least_squares(resid, theta0, bounds=(lo, hi), loss="soft_l1", f_scale=3 * c_local, xtol=1e-9, ftol=1e-9, max_nfev=200)
    ls = least_squares(resid, ls.x, bounds=(lo, hi), loss="soft_l1", f_scale=c_local, xtol=1e-10, ftol=1e-10, max_nfev=400)
    theta = ls.x
    params = from_theta(theta, names)
    r = obj_l.residuals(theta)
    inlier = r < 3 * c_local
    inlier_frac = float(np.sum(obj_l.w[inlier]))
    rms = float(np.sqrt(np.mean(r[inlier] ** 2))) if inlier.any() else float("nan")
    # parameter uncertainties from the Jacobian restricted to inliers (log-space -> relative)
    sigma = {}
    try:
        J = ls.jac[inlier]
        s2 = np.sum(ls.fun[inlier] ** 2) / max(inlier.sum() - len(theta), 1)
        cov = s2 * np.linalg.pinv(J.T @ J)
        sd = np.sqrt(np.clip(np.diag(cov), 0, None))
        for n, t, s in zip(names, theta, sd):
            sigma[n] = float(np.exp(t) * s) if n in LOG_PARAMS else float(s)
    except Exception:
        sigma = {n: float("nan") for n in names}
    lines = predict_lines(hyp, params, x_all, model_kw)
    return FitResult(hyp, params, theta, sigma, float(obj_l(theta)), r, inlier, inlier_frac, rms, x_all, lines, history=history, n_evals=n_evals + int(ls.nfev))


def brightest_points(points: np.ndarray) -> np.ndarray:
    """One point per flux column: the brightest feature (almost always the 0-1 line)."""
    out = []
    for c in np.unique(points[:, 0]):
        m = points[:, 0] == c
        out.append(points[m][np.argmax(points[m, 2])])
    return np.array(out)


def fit_hypothesis(points, hyp, band, bounds=None, init=None, c_global=0.08, c_local=0.012, maxiter=60, popsize=15, seed=0, workers=1,
                   callback=None, model_kw=None, do_global=True, n_starts=1, staged=True) -> FitResult:
    """Robust driver.  (1) staged: fit the 0-1 line alone to the brightest feature per column (one line,
    no assignment ambiguity -> smooth landscape), then fit all lines with bounds narrowed around that;
    (2) multi-start: the plain global+local fit from `n_starts` seeds.  Lowest fine-scale loss wins."""
    best = None
    x_all = np.unique(points[:, 0])
    base_bounds = bounds or default_bounds(hyp, x_all)
    if do_global and staged and hyp.flux_dependent and hyp.platform == "circuit" and len(x_all) >= 12:
        bp = brightest_points(points)
        resA = _fit_once(bp, hyp, band, bounds=base_bounds, init=init, c_global=c_global, c_local=c_local, maxiter=max(20, int(maxiter * 0.7)), popsize=popsize,
                         seed=seed + 7, workers=workers, callback=callback, model_kw=model_kw, line_filter=["0-1"])
        nar = {}
        P = resA.params.get("period", 1.0)
        for n, (lo, hi) in base_bounds.items():
            v = resA.params[n]
            if n == "period":
                nar[n] = (max(lo, 0.9 * v), min(hi, 1.1 * v))
            elif n == "offset":
                nar[n] = (v - 0.06 * P, v + 0.06 * P)
            elif n in ("fr", "g"):
                nar[n] = (lo, hi)
            elif n in ("EC", "EJ", "EJmax") and hyp.family != "fluxonium":
                nar[n] = (lo, hi)  # the 0-1 line alone cannot separate EJ from EC: leave both open
            elif n in LOG_PARAMS:
                nar[n] = (max(lo, v / 1.6), min(hi, v * 1.6))
            else:
                nar[n] = (max(lo, v - 0.2), min(hi, v + 0.2))
        best = _fit_once(points, hyp, band, bounds=nar, init=resA.params, c_global=c_global, c_local=c_local, maxiter=max(20, int(maxiter * 0.7)), popsize=popsize,
                         seed=seed + 13, workers=workers, callback=callback, model_kw=model_kw, do_global=True)
        if best.inlier_frac > 0.97:
            return _normalise_offset(best, x_all)
    for k in range(max(1, n_starts) if do_global else 1):
        res = _fit_once(points, hyp, band, bounds=base_bounds, init=init, c_global=c_global, c_local=c_local, maxiter=maxiter, popsize=popsize,
                        seed=seed + 1000 * k, workers=workers, callback=callback, model_kw=model_kw, do_global=do_global)
        if best is None or res.loss < best.loss:
            best = res
        if best.inlier_frac > 0.97:
            break
    return _normalise_offset(best, x_all)


def _normalise_offset(res: "FitResult", x_all: np.ndarray) -> "FitResult":
    """The flux offset is defined modulo the period: report the equivalent value nearest the data centre."""
    if "period" in res.params and "offset" in res.params:
        P = res.params["period"]
        mid = 0.5 * (float(np.min(x_all)) + float(np.max(x_all)))
        off = res.params["offset"]
        off2 = off - P * np.round((off - mid) / P)
        if off2 != off:
            res.params["offset"] = float(off2)
            names = res.hyp.names
            res.theta = to_theta(res.params, names)
    return res


def holdout_check(points: np.ndarray, result: FitResult, band: tuple[float, float], frac: float = 0.3, c_local: float = 0.012, model_kw: dict | None = None) -> dict:
    """Refit locally on ~70% of flux columns, score the untouched 30%."""
    cols = np.unique(points[:, 0])
    test_cols = cols[::max(int(round(1 / frac)), 2)]
    is_test = np.isin(points[:, 0], test_cols)
    train, test = points[~is_test], points[is_test]
    if len(test) < 5 or len(train) < 10:
        return {}
    r_train = fit_hypothesis(train, result.hyp, band, init=result.params, do_global=False, c_local=c_local, model_kw=model_kw)
    obj_t = Objective(result.hyp, test, band, c_local, model_kw)
    r = obj_t.residuals(r_train.theta)
    inl = r < 3 * c_local
    out = dict(
        n_test_points=int(len(test)),
        inlier_frac=float(np.sum(obj_t.w[inl])),
        rms_MHz=float(1e3 * np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float("nan"),
        param_shift_rel={n: float(abs(r_train.params[n] - result.params[n]) / (abs(result.params[n]) + 1e-12)) for n in result.params},
    )
    result.holdout = out
    return out


def track_score(points: np.ndarray, track_ids: np.ndarray, flat: np.ndarray, inlier_mask: np.ndarray) -> float:
    """Fraction of track weight explained by a fit (features count, not pixels)."""
    from .extract import track_weights

    tw = track_weights(points, track_ids, flat)
    tot = sum(tw.values())
    if tot <= 0:
        return 0.0
    expl = 0.0
    for t, w in tw.items():
        m = track_ids == t
        expl += w * float(np.mean(inlier_mask[m]))
    return expl / tot
