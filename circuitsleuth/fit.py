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

LOG_PARAMS = {"EJ", "EC", "EL", "EJmax", "fr", "g", "period", "field_scale", "lever_arm", "t_c", "E", "f_rad0", "volt_scale", "alpha1", "alpha2", "dfs"}
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
        b["field_scale"] = models.control_bounds(hyp.family)["field_scale"]
        b["x0"] = (float(np.min(x)) - span, float(np.max(x)) + span)
    elif ctrl == ("lever_arm", "x0"):
        b["lever_arm"] = models.control_bounds(hyp.family)["lever_arm"]
        b["x0"] = (float(np.min(x)) - span / 2, float(np.max(x)) + span / 2)
    elif ctrl == ("volt_scale", "x0"):
        b["volt_scale"] = models.control_bounds(hyp.family)["volt_scale"]
        b["x0"] = (float(np.min(x)) - 2 * span, float(np.min(x)))  # confinement vanishes below the measured window
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
    alternatives: list = field(default_factory=list)  # other labellings of the same features (see relabel_alternatives)

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
            rms_inliers_MHz=round(models.to_MHz(self.hyp.family) * self.rms_inliers, 2),
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


def relabel_alternatives(points: np.ndarray, res: FitResult, band: tuple[float, float], bounds: dict, c_local: float = 0.012, c_global: float = 0.08,
                         model_kw: dict | None = None, workers: int = 1) -> list[FitResult]:
    """Label ambiguity of a control-independent spectrum (fixed transmon): every line is flat, so a second track can be
    read two ways, e.g. the two-photon 0-2 line at f01 - EC/2 or the thermal 1-2 line at f01 - EC.  Both pass through the
    same points, so the robust loss cannot tell them apart and the search lands on either (EC differs by 2x).  For each
    (used weaker line, unused brighter line) pair, solve for the parameters that put the brighter line on the track while
    keeping the 0-1 line fixed, polish locally, and return the refits: the caller ranks them with the visibility-aware
    score, where the labelling that leaves a bright predicted line with nothing under it loses."""
    hyp = res.hyp
    if hyp.platform != "circuit" or hyp.flux_dependent:
        return []
    obj = Objective(hyp, points, band, c_local, model_kw)
    keys = list(predict_lines(hyp, res.params, obj.x_cols[:1], model_kw).keys())
    if "0-1" not in keys:
        return []
    L = obj.line_matrix(res.theta)
    D = np.abs(obj.f[None, :] - L[:, obj.col_idx])
    assign = np.argmin(D, axis=0)
    inl = D[assign, np.arange(len(assign))] < 3 * c_local
    counts = {k: int(np.sum(inl & (assign == i))) for i, k in enumerate(keys)}
    n_in = {k: int(np.sum(np.isfinite(L[i]) & (L[i] > band[0]) & (L[i] < band[1]))) for i, k in enumerate(keys)}
    if counts["0-1"] == 0:
        return []
    vis = models.line_weights(hyp.family)
    # "unused": a predicted in-band line with (almost) nothing under it; a couple of stray noise points do not count
    missing = [k for k in keys if k != "res" and n_in[k] >= 3 and counts[k] <= max(2, 0.1 * n_in[k])]
    used = [k for k in keys if k not in ("0-1", "res") and counts[k] >= 3]
    f01 = float(np.median(obj.f[inl & (assign == keys.index("0-1"))]))
    names = hyp.names
    tb = theta_bounds(bounds, names)
    lo = np.array([b[0] for b in tb])
    hi = np.array([b[1] for b in tb])
    x0 = obj.x_cols[:1]
    out: list[FitResult] = []
    for u in used:
        f_u = float(np.median(obj.f[inl & (assign == keys.index(u))]))
        for m in missing:
            if vis.get(m, 0.0) <= vis.get(u, 0.0):
                continue

            def resid(theta, _m=m):
                lines = predict_lines(hyp, from_theta(theta, names), x0, model_kw)
                return np.nan_to_num(np.array([lines["0-1"][0] - f01, lines[_m][0] - f_u]), nan=1e3, posinf=1e3, neginf=-1e3)

            try:
                sol = least_squares(resid, np.clip(res.theta, lo + 1e-9, hi - 1e-9), bounds=(lo, hi), xtol=1e-10, ftol=1e-10, max_nfev=200)
                if np.max(np.abs(sol.fun)) > 3 * c_local:
                    continue
                alt = _fit_once(points, hyp, band, bounds=bounds, init=from_theta(sol.x, names), c_global=c_global, c_local=c_local, model_kw=model_kw,
                                do_global=False, workers=workers)
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                continue
            if all(np.max(np.abs(alt.theta - o.theta)) > 1e-3 for o in out) and np.max(np.abs(alt.theta - res.theta)) > 1e-3:
                alt.history = [dict(relabel=f"{u} -> {m}")]
                out.append(alt)
    return out


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
    if do_global:
        best.alternatives = relabel_alternatives(points, best, band, base_bounds, c_local=c_local, c_global=c_global, model_kw=model_kw, workers=workers)
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
        rms_MHz=float(models.to_MHz(result.hyp.family) * np.sqrt(np.mean(r[inl] ** 2))) if inl.any() else float("nan"),
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


# ----------------------------------------------------------------------------- posterior sampling (calibrated uncertainties)
def result_from_report(report: dict, points: np.ndarray | None = None) -> FitResult:
    """Rebuild a FitResult from a saved report (``report['best']``) so the posterior sampler and the design code can run
    on stored results.  Only hyp / params / theta / sigma / inlier_mask / lines / holdout are meaningful; residuals are NaN."""
    best = report["best"]
    hyp = Hypothesis(best["family"], bool(best.get("coupled", False)))
    pts = np.asarray(report["points"] if points is None else points, float)
    params = {n: float(best["params"][n]) for n in hyp.names}
    theta = to_theta(params, hyp.names)
    x_all = np.unique(pts[:, 0])
    if points is None and "inlier_mask" in report and len(report["inlier_mask"]) == len(pts):
        inl = np.asarray(report["inlier_mask"], bool)
    else:
        inl = np.ones(len(pts), bool)
    lines = {k: np.array([np.nan if v is None else v for v in arr], float) for k, arr in (report.get("lines") or {}).items()}
    rms = float(best.get("rms_inliers_MHz", np.nan)) / models.to_MHz(hyp.family)
    return FitResult(hyp, params, theta, {k: float(v) for k, v in (best.get("sigma") or {}).items()}, float(best.get("loss", np.nan)), np.full(len(pts), np.nan),
                     inl, float(best.get("inlier_frac", np.nan)), rms, x_all, lines, holdout=dict(best.get("holdout") or {}))


def soft_l1(u: np.ndarray) -> np.ndarray:
    """The soft-L1 loss used by the polish stage (scipy ``loss='soft_l1'``): quadratic for |u| << 1, linear beyond."""
    return 2.0 * (np.sqrt(1.0 + u * u) - 1.0)


def robust_loglik(obj: Objective, theta: np.ndarray, w: np.ndarray, c: float) -> float:
    """log-likelihood  -sum_i w_i rho(r_i / c)  with rho = soft-L1 and r_i the distance to the nearest predicted line."""
    try:
        r = obj.residuals(theta)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return -np.inf
    r = np.where(np.isfinite(r), r, 1e3)  # a point with no finite line nearby costs a large, finite amount
    return -float(np.sum(w * soft_l1(r / c)))


def signed_residual_jacobian(obj: Objective, theta: np.ndarray, eps: float = 1e-4):
    """Central-difference Jacobian d r_i / d theta of the SIGNED residual f_i - L_a(i)(theta), where a(i) is the line nearest
    to point i at `theta` (fixed during differentiation, so the |.| kink of the nearest-line distance does not bite).
    Returns (r0 (n,), J (n, n_theta))."""
    L0 = obj.line_matrix(theta)
    D0 = obj.f[None, :] - L0[:, obj.col_idx]
    assign = np.argmin(np.abs(D0), axis=0)
    idx = (assign, obj.col_idx)  # indexes the (n_lines, n_cols) line matrix at each point's assigned line and column
    r0 = obj.f - L0[idx]
    J = np.zeros((len(r0), len(theta)))
    for k in range(len(theta)):
        h = eps * max(1.0, abs(float(theta[k])))
        tp, tm = theta.copy(), theta.copy()
        tp[k] += h
        tm[k] -= h
        rp = obj.f - obj.line_matrix(tp)[idx]
        rm = obj.f - obj.line_matrix(tm)[idx]
        J[:, k] = (rp - rm) / (2 * h)
    J = np.where(np.isfinite(J), J, 0.0)
    r0 = np.where(np.isfinite(r0), r0, 1e3)
    return r0, J


def laplace_covariance(obj: Objective, theta: np.ndarray, w: np.ndarray, c: float) -> np.ndarray | None:
    """Gauss-Newton inverse Hessian of  sum_i w_i rho(r_i/c)  at `theta` (theta space).  rho''(u) = 2 (1+u^2)^(-3/2) down-weights
    outliers, so far-away features contribute no curvature, exactly as they contribute no information to the fit."""
    try:
        r0, J = signed_residual_jacobian(obj, theta)
    except (np.linalg.LinAlgError, ValueError, FloatingPointError):
        return None
    u = r0 / c
    curv = w * 2.0 / (1.0 + u * u) ** 1.5 / c**2
    H = (J * curv[:, None]).T @ J
    if not np.all(np.isfinite(H)) or np.trace(H) <= 0:
        return None
    reg = H + np.eye(len(theta)) * (1e-9 * np.trace(H) / len(theta))
    cov = np.linalg.pinv(reg)
    cov = 0.5 * (cov + cov.T)
    return cov if np.all(np.isfinite(cov)) else None


def gauss_newton_recentre(obj: Objective, theta: np.ndarray, w: np.ndarray, c: float, lo: np.ndarray, hi: np.ndarray, n_iter: int = 4) -> np.ndarray:
    """A few damped Gauss-Newton steps on  sum_i w_i rho(r_i/c)  so the chain starts at the mode of the (subsampled) likelihood
    it samples rather than at the full-data MAP, which can sit a few stiff-direction sigmas away.  Each step is accepted only
    if the loss decreases; returns the (possibly unchanged) theta."""
    th = theta.copy()
    nll = -robust_loglik(obj, th, w, c)
    for _ in range(n_iter):
        try:
            r0, J = signed_residual_jacobian(obj, th)
        except (np.linalg.LinAlgError, ValueError, FloatingPointError):
            break
        u = r0 / c
        g = J.T @ (w * 2.0 * u / np.sqrt(1.0 + u * u) / c)  # gradient of the NLL
        curv = w * 2.0 / (1.0 + u * u) ** 1.5 / c**2
        H = (J * curv[:, None]).T @ J
        if not np.all(np.isfinite(H)) or np.trace(H) <= 0:
            break
        H = H + np.eye(len(th)) * (1e-9 * np.trace(H) / len(th))
        step = -np.linalg.pinv(H) @ g
        improved = False
        for damp in (1.0, 0.5, 0.25, 0.1):
            cand = np.clip(th + damp * step, lo + 1e-9, hi - 1e-9)
            v = -robust_loglik(obj, cand, w, c)
            if v < nll - 1e-9:
                th, nll, improved = cand, v, True
                break
        if not improved or np.max(np.abs(step) / np.maximum(np.sqrt(np.clip(np.diag(np.linalg.pinv(H)), 1e-300, None)), 1e-12)) < 0.05:
            break
    return th


def _theta_to_param_matrix(theta: np.ndarray, names: list[str]) -> np.ndarray:
    """Jacobian of params w.r.t. theta at theta (diagonal: exp(theta) for log parameters, 1 otherwise)."""
    return np.diag([float(np.exp(t)) if n in LOG_PARAMS else 1.0 for n, t in zip(names, theta)])


def sample_posterior(points: np.ndarray, result: FitResult, band: tuple[float, float], c_local: float, n_steps: int = 3000, seed: int = 0, max_cols: int = 31,
                     proposal_scale: float = 0.3, model_kw: dict | None = None, exclude_flat: bool = True, burn_in: float = 0.1) -> dict:
    """Random-walk Metropolis in theta space (the fit's parameterisation: log for energies) over the robust likelihood
    implied by the soft-L1 polish:  log L(theta) = -sum_i w_i rho(r_i(theta) / c_local),  rho(u) = 2 (sqrt(1+u^2) - 1),
    r_i = distance of feature i to the nearest predicted line, flat prior inside ``default_bounds``.

    * Columns are thinned to `max_cols` for speed (``subsample``); the weights of the kept points are normalised to sum to the
      number of points in the FULL cloud, so the subsample stands in for all the data instead of widening the posterior.
    * The chain starts at the mode of the subsampled likelihood (a few Gauss-Newton steps from the MAP; ``mode`` in the output,
      usually < 1 sigma from ``map``), the first `burn_in` fraction of steps is discarded and the rest thinned to ~750 samples.
    * Proposal covariance = `proposal_scale` x Laplace covariance (Gauss-Newton inverse Hessian at that mode; falls back to a
      (0.02 |theta| + 1e-3)^2 diagonal), capped at half the prior width per axis.

    * `exclude_flat` applies the same rule as the agent before fitting: a control-dependent hypothesis is sampled on the
      control-dependent features only (flat lines are pumps/spurs), provided >= max(10, 20 %) of the points are dynamic.
      Pass the full report point cloud and the sampler will drop the spurs itself.

    `result` needs ``.hyp`` and ``.theta`` (a FitResult or ``result_from_report(report)``).

    Returns dict(samples (n_kept, n_theta) in PARAMETER units, theta_samples, names, acceptance_rate, sd, laplace_sd, map, mode, mean,
    quantiles (16/50/84 %), cov (sample covariance, parameter units), laplace_cov (theta space, at the mode), laplace_cov_params,
    family, n_steps, n_kept, thin, burn_in, n_points, n_points_used, seconds)."""
    import time as _time

    t0 = _time.perf_counter()
    hyp = result.hyp
    names = list(hyp.names)
    pts_all = np.asarray(points, float)
    if exclude_flat and hyp.flux_dependent:
        from .extract import flat_feature_mask

        flat = flat_feature_mask(pts_all, tol=models.platform_scales(hyp.platform)["flat_tol"])
        if int((~flat).sum()) >= max(10, 0.2 * len(pts_all)):
            pts_all = pts_all[~flat]
    x_all = np.unique(pts_all[:, 0])
    tb = theta_bounds(default_bounds(hyp, x_all), names)
    lo = np.array([b[0] for b in tb])
    hi = np.array([b[1] for b in tb])
    theta0 = np.clip(np.asarray(result.theta, float), lo + 1e-9, hi - 1e-9)
    sub = subsample(pts_all, max_cols=max_cols)
    obj = Objective(hyp, sub, band, c_local, model_kw)
    w = sub[:, 2] / max(float(sub[:, 2].sum()), 1e-12) * len(pts_all)
    d = len(theta0)
    theta_m = gauss_newton_recentre(obj, theta0, w, c_local, lo, hi)

    cov = laplace_covariance(obj, theta_m, w, c_local)
    if cov is None:
        cov = np.diag((0.02 * np.abs(theta_m) + 1e-3) ** 2)
    prop = proposal_scale * cov
    # cap the proposal at half the prior width along each axis (an unidentified parameter would otherwise propose off the box)
    s = np.sqrt(np.clip(np.diag(prop), 1e-300, None))
    cap = np.minimum(1.0, 0.5 * (hi - lo) / s)
    prop = prop * np.outer(cap, cap)
    vals, vecs = np.linalg.eigh(0.5 * (prop + prop.T))
    Lch = vecs * np.sqrt(np.clip(vals, 0.0, None))[None, :]

    rng = np.random.default_rng(seed)
    n_burn = int(round(np.clip(burn_in, 0.0, 0.9) * int(n_steps)))
    thin = max(1, (int(n_steps) - n_burn) // 750)
    theta, ll = theta_m.copy(), robust_loglik(obj, theta_m, w, c_local)
    accepted = 0
    kept = []
    for step in range(int(n_steps)):
        cand = theta + Lch @ rng.standard_normal(d)
        if np.all(cand >= lo) and np.all(cand <= hi):
            llc = robust_loglik(obj, cand, w, c_local)
            if np.log(rng.uniform()) < llc - ll:
                theta, ll = cand, llc
                accepted += 1
        if step >= n_burn and (step + 1 - n_burn) % thin == 0:
            kept.append(theta.copy())
    T = np.array(kept) if kept else theta_m[None, :]
    is_log = np.array([n in LOG_PARAMS for n in names])
    P = np.where(is_log[None, :], np.exp(T), T)
    D = _theta_to_param_matrix(theta_m, names)
    lap_cov_p = D @ cov @ D
    lap_sd = np.sqrt(np.clip(np.diag(lap_cov_p), 0, None))
    sd = P.std(axis=0, ddof=1) if len(P) > 1 else np.full(d, np.nan)
    q = np.percentile(P, [16, 50, 84], axis=0)
    return dict(
        samples=P, theta_samples=T, names=names,
        acceptance_rate=float(accepted / max(int(n_steps), 1)),
        sd={n: float(v) for n, v in zip(names, sd)},
        laplace_sd={n: float(v) for n, v in zip(names, lap_sd)},
        map=from_theta(theta0, names), mode=from_theta(theta_m, names),
        mean={n: float(v) for n, v in zip(names, P.mean(axis=0))},
        quantiles={n: [float(q[0, i]), float(q[1, i]), float(q[2, i])] for i, n in enumerate(names)},
        cov=np.cov(P.T) if len(P) > 1 else np.full((d, d), np.nan),
        laplace_cov=cov, laplace_cov_params=lap_cov_p, family=hyp.family,
        n_steps=int(n_steps), n_kept=int(len(P)), thin=int(thin), burn_in=int(n_burn), n_points=int(len(pts_all)), n_points_used=int(len(sub)),
        seconds=float(_time.perf_counter() - t0),
    )
