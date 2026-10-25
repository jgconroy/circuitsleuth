"""Optimal next measurement by Fisher information (Bayesian experimental design, Laplace approximation).

Given the fitted hypothesis, every detected feature contributes to the Fisher information
F0 = sum_i J_i^T J_i / sigma_f^2 with J_i = d(line frequency)/d(theta) at the point's column (theta = the
fit's parameterisation, log for energies). A candidate measurement is "sweep columns [x_lo, x_hi] and look for
line l"; its expected information gain is the log-det ratio
    dI = 1/2 * [log det(F0 + F_S) - log det(F0)]  (nats, 1.4427 bits per nat),
discounted by the line's nominal visibility (probability it shows up) and by the number of columns (cost).
We also report which parameter's variance each candidate shrinks most, and the current parameter
correlation matrix (from F0^-1), which is where degeneracies such as EJ-EC show up.
"""
from __future__ import annotations

import numpy as np

from . import fit, models


def _theta_jacobian(hyp: fit.Hypothesis, params: dict, x_cols: np.ndarray, line_names: list[str], eps: float = 1e-4):
    """Finite-difference Jacobian of every line frequency w.r.t. theta: returns array (n_lines, n_cols, n_theta)."""
    names = hyp.names
    theta0 = fit.to_theta(params, names)
    base = fit.predict_lines(hyp, params, x_cols)
    L0 = np.stack([base[n] for n in line_names])
    J = np.zeros(L0.shape + (len(theta0),))
    for k in range(len(theta0)):
        th = theta0.copy()
        h = eps * max(1.0, abs(th[k]))
        th[k] += h
        pk = fit.from_theta(th, names)
        Lk = fit.predict_lines(hyp, pk, x_cols)
        J[:, :, k] = (np.stack([Lk[n] for n in line_names]) - L0) / h
    J = np.where(np.isfinite(J), J, 0.0)
    return L0, J, theta0


def fisher_information(report: dict, sigma_f: float | None = None):
    """F0 from the explained points of the best hypothesis. Returns dict(F0, names, theta, lines, L0, J, x_cols, sigma_f)."""
    best = report["best"]
    hyp = fit.Hypothesis(best["family"], best.get("coupled", False))
    pl = hyp.platform
    sigma_f = sigma_f or models.platform_scales(pl)["c_local"]
    pts = np.asarray(report["points"], float)
    inl = np.asarray(report["inlier_mask"], bool)
    x_cols = np.unique(pts[:, 0])
    line_names = [n for n in report["lines"] if n != "res" or best.get("coupled", False)]
    L0, J, theta = _theta_jacobian(hyp, best["params"], x_cols, line_names)
    col_idx = np.searchsorted(x_cols, pts[:, 0])
    F0 = np.zeros((len(theta), len(theta)))
    for i in np.flatnonzero(inl):
        c = col_idx[i]
        d = np.abs(L0[:, c] - pts[i, 1])
        d = np.where(np.isfinite(d), d, np.inf)
        l = int(np.argmin(d))
        if not np.isfinite(d[l]) or d[l] > 5 * sigma_f:
            continue
        j = J[l, c]
        F0 += np.outer(j, j) / sigma_f**2
    return dict(F0=F0, names=hyp.names, theta=theta, lines=line_names, L0=L0, J=J, x_cols=x_cols, sigma_f=sigma_f, hyp=hyp)


def correlation_matrix(F0: np.ndarray, names: list[str]):
    """Parameter correlation matrix from the Laplace covariance F0^-1 (regularised)."""
    reg = F0 + np.eye(len(names)) * (1e-9 * max(np.trace(F0) / len(names), 1e-12))
    cov = np.linalg.pinv(reg)
    sd = np.sqrt(np.clip(np.diag(cov), 1e-300, None))
    corr = cov / np.outer(sd, sd)
    return np.clip(corr, -1, 1), cov


def design_next_measurements(report: dict, band: tuple[float, float] | None = None, block: int = 6, top: int = 5, sigma_f: float | None = None) -> dict:
    """Rank candidate sweeps (line, column block) by expected information gain.

    Returns dict(candidates=[{line, x_lo, x_hi, f_lo, f_hi, gain_bits, gain_per_column_bits, best_parameter,
    variance_reduction}], gain_map={line: [bits per column]}, correlation=(names, matrix), sigma_f)."""
    info = fisher_information(report, sigma_f)
    F0, J, L0, x_cols, names, lines = info["F0"], info["J"], info["L0"], info["x_cols"], info["names"], info["lines"]
    band = band or tuple(report.get("band_GHz", (float(np.nanmin(L0)), float(np.nanmax(L0)))))
    vis = models.line_weights(info["hyp"].family)
    s2 = info["sigma_f"] ** 2
    reg = F0 + np.eye(len(names)) * (1e-9 * max(np.trace(F0) / len(names), 1e-12))
    sign0, logdet0 = np.linalg.slogdet(reg)
    cov0 = np.linalg.pinv(reg)
    var0 = np.clip(np.diag(cov0), 1e-300, None)
    gain_map: dict[str, np.ndarray] = {}
    cands = []
    n = len(x_cols)
    for li, name in enumerate(lines):
        inband = np.isfinite(L0[li]) & (L0[li] > band[0]) & (L0[li] < band[1])
        per_col = np.zeros(n)
        for c in range(n):
            if not inband[c]:
                continue
            j = J[li, c]
            Fc = reg + np.outer(j, j) / s2
            per_col[c] = 0.5 * (np.linalg.slogdet(Fc)[1] - logdet0) * 1.4427  # bits
        gain_map[name] = per_col
        # contiguous blocks of `block` columns
        for a in range(0, n, max(block // 2, 1)):
            b = min(a + block, n)
            cols = [c for c in range(a, b) if inband[c]]
            if len(cols) < max(3, block // 2):
                continue
            FS = reg.copy()
            for c in cols:
                FS += np.outer(J[li, c], J[li, c]) / s2
            gain = 0.5 * (np.linalg.slogdet(FS)[1] - logdet0) * 1.4427 * vis.get(name, 0.5)
            if gain <= 0.05:
                continue
            covS = np.linalg.pinv(FS)
            red = 1 - np.clip(np.diag(covS), 1e-300, None) / var0
            # name the parameter helped most among those that are not already pinned (relative sigma > 1e-3)
            theta_abs = np.maximum(np.abs(info["theta"]), 1e-3)
            free = np.sqrt(var0) / theta_abs > 1e-3
            k = int(np.argmax(np.where(free, red, -np.inf))) if free.any() else int(np.argmax(red))
            cands.append(dict(line=name, x_lo=float(x_cols[cols[0]]), x_hi=float(x_cols[cols[-1]]), f_lo=float(np.nanmin(L0[li, cols])), f_hi=float(np.nanmax(L0[li, cols])),
                              gain_bits=float(gain), gain_per_column_bits=float(gain / len(cols)), best_parameter=names[k], variance_reduction={names[i]: float(red[i]) for i in range(len(names))}))
    # greedy, non-overlapping selection by gain per column
    cands.sort(key=lambda d: -d["gain_per_column_bits"])
    chosen: list[dict] = []
    for cnd in cands:
        if any(o["line"] == cnd["line"] and not (cnd["x_hi"] < o["x_lo"] or cnd["x_lo"] > o["x_hi"]) for o in chosen):
            continue
        chosen.append(cnd)
        if len(chosen) >= top:
            break
    corr, _ = correlation_matrix(F0, names)
    return dict(candidates=chosen, gain_map={k: v.tolist() for k, v in gain_map.items()}, x_cols=x_cols.tolist(), correlation=(names, corr.tolist()), sigma_f=info["sigma_f"],
                current_sigma={names[i]: float(np.sqrt(var0[i])) for i in range(len(names))})
