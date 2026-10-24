"""Orchestration: extract -> propose -> fit -> verify -> revise -> report.

This is the "agent loop".  It is deliberately deterministic and inspectable: every hypothesis that was
tried is kept in the report with its score, every unexplained feature is listed, and the parameter
numbers always come from the physics fit (never from the neural proposer or from an LLM).
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from . import extract, fit, models

Progress = Callable[[str, dict], None]


@dataclass
class AgentConfig:
    top_k: int = 3
    workers: int = 1
    maxiter: int = 60
    popsize: int = 15
    use_classifier: bool = True
    classifier_path: str | None = None
    try_coupled: str = "auto"  # "auto" | "always" | "never"
    holdout: bool = True
    seed: int = 0
    c_local: float = 0.012
    extra_hypotheses: tuple = ()


def _emit(progress: Progress | None, kind: str, **payload):
    if progress:
        progress(kind, payload)


def rasterize(points: np.ndarray, nf: int = 400, nx: int | None = None, lw: float = 0.02):
    """Build a map from a point cloud so the CNN proposer (which looks at images) can be used on CSV input."""
    xs = np.unique(points[:, 0])
    nx = nx or len(xs)
    x = np.linspace(xs.min(), xs.max(), nx)
    f = np.linspace(points[:, 1].min() - 0.2, points[:, 1].max() + 0.2, nf)
    img = np.zeros((nf, nx))
    for xi, fi, wi in points:
        c = int(np.argmin(np.abs(x - xi)))
        img[:, c] += wi * lw**2 / ((f - fi) ** 2 + lw**2)
    return img, x, f


def _fit_points_for(hyp: fit.Hypothesis, pts_all: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Flux-dependent hypotheses are fitted on the flux-dependent features only; a fixed-frequency
    circuit's lines are flat, so it is fitted on everything."""
    if hyp.flux_dependent and (~flat).sum() >= 10:
        return pts_all[~flat]
    return pts_all


def _score(res: fit.FitResult, pts_all, ids, flat, band, c_local, model_kw=None) -> tuple[float, np.ndarray]:
    obj = fit.Objective(res.hyp, pts_all, band, c_local, model_kw)
    r = obj.residuals(res.theta)
    inl = r < 3 * c_local
    ts = fit.track_score(pts_all, ids, flat, inl)
    return float(ts - 0.02 * len(res.params)), inl


def unexplained_features(pts_all, ids, inlier_all, flat) -> list[dict]:
    out = []
    for s in extract.summarize_tracks(pts_all, ids):
        m = ids == s["track"]
        frac_out = 1.0 - float(np.mean(inlier_all[m]))
        if frac_out < 0.6 or m.sum() < 3:
            continue
        s = dict(s, unexplained_frac=round(frac_out, 2))
        if s["flat"] or flat[m].mean() > 0.5:
            s["interpretation"] = "flux-independent line: likely pump/LO leakage, a mixer spur, or a fixed-frequency neighbour (resonator, TLS, other qubit)"
        else:
            s["interpretation"] = "flux-dependent feature not in the model: candidate neighbouring tunable mode, higher transition, or sideband of a qubit line"
        out.append(s)
    return out


def next_measurements(res: fit.FitResult, pts_all, band, x_grid, c_local, line_weights=None) -> list[dict]:
    """Where to look next: predicted lines inside the band with no data, lines just outside the band,
    and the specific line that would pin down any poorly determined parameter."""
    lines = fit.predict_lines(res.hyp, res.params, x_grid)
    cols = np.unique(pts_all[:, 0])
    out = []
    for name, L in lines.items():
        if name == "res":
            continue
        inband = np.isfinite(L) & (L > band[0]) & (L < band[1])
        if not inband.any():
            # entirely out of band: suggest extending if it is close
            Lf = L[np.isfinite(L)]
            if Lf.size and (Lf.min() < band[1] + 3 or Lf.max() > band[0] - 3):
                lo, hi = float(np.nanmin(L)), float(np.nanmax(L))
                out.append(dict(kind="extend_band", line=name, f_lo=round(lo, 3), f_hi=round(hi, 3), why=f"the {name} transition is predicted at {lo:.2f}-{hi:.2f} GHz, outside the measured band"))
            continue
        # columns where the line is in band but no feature within 3c
        missing = []
        for xi in x_grid[inband]:
            near = pts_all[np.isclose(pts_all[:, 0], xi)]
            Li = float(np.interp(xi, x_grid, np.where(np.isfinite(L), L, np.nan)))
            if not near.size or np.min(np.abs(near[:, 1] - Li)) > 3 * c_local:
                missing.append((xi, Li))
        if len(missing) >= 0.5 * inband.sum() and len(missing) >= 3:
            xs = np.array([m[0] for m in missing])
            fs = np.array([m[1] for m in missing])
            why = f"the model predicts the {name} line here but nothing was detected"
            if name == "0-2/2ph":
                why += "; raise the drive power to bring out the two-photon transition"
            elif name == "1-2":
                why += "; a weak thermal population line, visible with higher power or a pre-pulse"
            out.append(dict(kind="look_here", line=name, x_lo=round(float(xs.min()), 4), x_hi=round(float(xs.max()), 4), f_lo=round(float(fs.min()), 3), f_hi=round(float(fs.max()), 3), why=why))
    # poorly determined parameters
    for n, s in res.sigma.items():
        v = res.params.get(n, np.nan)
        if np.isfinite(s) and abs(v) > 0 and s / abs(v) > 0.1:
            hint = {"EC": "EC is set by the anharmonicity: the 0-2/2ph (f01 - EC/2) or 1-2 (f01 - EC) lines pin it",
                    "EJ": "EJ is degenerate with EC along constant f01; the 0-2/2ph line breaks the degeneracy",
                    "EJmax": "EJmax is degenerate with EC along constant f01; the 0-2/2ph line breaks the degeneracy",
                    "EL": "EL is best constrained by the 0-1 line near half flux and by the 0-2 transition",
                    "d": "the junction asymmetry d is set by the minimum of the 0-1 line at half flux: extend the flux sweep",
                    "g": "the coupling g is set by the avoided-crossing splitting: measure finely where the qubit crosses the resonator",
                    "fr": "the resonator frequency: a single-tone scan pins it"}.get(n, "")
            out.append(dict(kind="constrain_parameter", parameter=n, rel_sigma=round(float(s / abs(v)), 3), why=hint))
    return out


def identify(image: np.ndarray | None = None, x: np.ndarray | None = None, f: np.ndarray | None = None, points: np.ndarray | None = None,
             band: tuple[float, float] | None = None, config: AgentConfig | None = None, progress: Progress | None = None) -> dict:
    cfg = config or AgentConfig()
    t_start = time.time()
    # ---- 1. features
    if points is None:
        assert image is not None and x is not None and f is not None, "need image+x+f or points"
        points = extract.extract_points(image, x, f)
        band = band or (float(f[0]), float(f[-1]))
    else:
        points = np.asarray(points, float)
        if image is None:
            image, x, f = rasterize(points)
        band = band or (float(points[:, 1].min()) - 0.1, float(points[:, 1].max()) + 0.1)
    x_grid = np.asarray(x, float) if x is not None else np.unique(points[:, 0])
    if len(points) < 10:
        return dict(error="fewer than 10 spectral features found; lower the detection threshold or check the map", n_points=int(len(points)))
    ids = extract.link_tracks(points)
    flat = extract.flat_feature_mask(points)
    flat_lines = extract.flat_lines(points)
    _emit(progress, "features", n_points=int(len(points)), n_tracks=int(len(np.unique(ids))), flat_lines=flat_lines)

    # ---- 2. propose
    proposals = []
    net = None
    if cfg.use_classifier:
        try:
            from . import classify

            net = classify.load(cfg.classifier_path or classify.DEFAULT_PATH)
            if net is not None:
                proposals = classify.propose(net, image, x_grid, top_k=6)
        except Exception as e:  # classifier is optional
            _emit(progress, "warning", message=f"classifier unavailable: {e}")
    if proposals:
        order = []
        for p in proposals:
            h = fit.Hypothesis(p["family"], p["coupled"])
            if h not in [o[0] for o in order]:
                order.append((h, p))
        bare = [(h, p) for h, p in order if not h.coupled][: cfg.top_k]
        p_coupled = proposals[0]["p_coupled"]
    else:
        bare = [(fit.Hypothesis(fam, False), None) for fam in models.FAMILIES]
        p_coupled = 0.5
    for h in cfg.extra_hypotheses:
        if h not in [b[0] for b in bare]:
            bare.append((h, None))
    _emit(progress, "proposals", hypotheses=[dict(h=h.label, prob=(p or {}).get("prob")) for h, p in bare], p_coupled=p_coupled)

    # ---- 3. fit bare hypotheses
    results: list[tuple[fit.FitResult, float, np.ndarray]] = []

    def run(h: fit.Hypothesis, init: dict | None, bounds: dict | None = None):
        pts_fit = _fit_points_for(h, points, flat)
        _emit(progress, "fit_start", hypothesis=h.label, n_points=int(len(pts_fit)))

        def cb(params, loss):
            _emit(progress, "fit_progress", hypothesis=h.label, params=params, loss=loss, lines={k: v.tolist() for k, v in fit.predict_lines(h, params, x_grid).items()})

        res = fit.fit_hypothesis(pts_fit, h, band, init=init, bounds=bounds, maxiter=cfg.maxiter, popsize=cfg.popsize, seed=cfg.seed, workers=cfg.workers, callback=cb, c_local=cfg.c_local)
        score, inl = _score(res, points, ids, flat, band, cfg.c_local)
        results.append((res, score, inl))
        _emit(progress, "fit_done", hypothesis=h.label, score=score, summary=res.summary(), lines={k: v.tolist() for k, v in res.lines.items()})
        return res, score

    for h, p in bare:
        if h.flux_dependent and (~flat).sum() < 10:
            continue  # nothing flux-dependent to fit
        init = None
        if p is not None:
            init = {k: v for k, v in p["seed"].items() if k in h.names}
        run(h, init)
    if not results:
        return dict(error="no hypothesis could be fitted", n_points=int(len(points)))
    results.sort(key=lambda r: -r[1])
    best, best_score, best_inl = results[0]

    # ---- 4. revise: try the resonator-coupled version of the best family
    want_coupled = cfg.try_coupled == "always" or (cfg.try_coupled == "auto" and (p_coupled > 0.25 or best_score < 0.85))
    if want_coupled and not best.hyp.coupled:
        hc = fit.Hypothesis(best.hyp.family, True)
        fr_guess = None
        for prop in proposals:
            if prop["family"] == best.hyp.family and prop["coupled"]:
                fr_guess, g_guess = prop["seed"].get("fr"), prop["seed"].get("g")
                break
        if fr_guess is None:
            cands = [fl for fl in flat_lines if 4.0 < fl < 10.0]
            fr_guess, g_guess = (cands[0] if cands else 7.0), 0.1
        init = dict(best.params, fr=fr_guess, g=g_guess)
        _emit(progress, "revise", reason=f"trying {hc.label} (p_coupled={p_coupled:.2f}, best score {best_score:.3f})")
        run(hc, init)
        results.sort(key=lambda r: -r[1])
        best, best_score, best_inl = results[0]

    # ---- 5. verify on held-out columns
    if cfg.holdout:
        pts_fit = _fit_points_for(best.hyp, points, flat)
        fit.holdout_check(pts_fit, best, band, c_local=cfg.c_local)
        _emit(progress, "verify", holdout=best.holdout)

    # ---- 6. report
    unexpl = unexplained_features(points, ids, best_inl, flat)
    nxt = next_measurements(best, points, band, x_grid, cfg.c_local)
    report = dict(
        status="ok",
        runtime_s=round(time.time() - t_start, 1),
        n_points=int(len(points)),
        n_tracks=int(len(np.unique(ids))),
        band_GHz=[round(float(band[0]), 3), round(float(band[1]), 3)],
        flat_lines_GHz=[round(v, 3) for v in flat_lines],
        proposals=[dict(hypothesis=fit.Hypothesis(p["family"], p["coupled"]).label, prob=round(p["prob"], 3)) for p in proposals],
        hypotheses=[dict(res.summary(), track_score=round(sc, 4)) for res, sc, _ in results],
        best=dict(res_summary(best), track_score=round(best_score, 4)),
        unexplained=unexpl,
        next_measurements=nxt,
        lines={k: [None if not np.isfinite(v) else round(float(v), 5) for v in arr] for k, arr in best.lines.items()},
        x_grid=[float(v) for v in x_grid],
        inlier_mask=best_inl.tolist(),
        points=points.tolist(),
        track_ids=ids.tolist(),
    )
    _emit(progress, "done", report=report)
    return report


def res_summary(res: fit.FitResult) -> dict:
    s = res.summary()
    s["params_pretty"] = {k: f"{v:.4g} ± {res.sigma.get(k, float('nan')):.2g}" + (" GHz" if k in fit.LOG_PARAMS and k != "period" else "") for k, v in res.params.items()}
    return s


def to_json(report: dict) -> str:
    return json.dumps(report, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
