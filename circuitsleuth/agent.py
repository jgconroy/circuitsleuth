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
    c_local: float | None = None  # None: platform default (circuits 12 MHz, NV 1.5 MHz)
    extra_hypotheses: tuple = ()
    families: tuple = ()  # restrict candidate families (empty = all)
    flux_calibrated: bool = False  # x axis already in flux quanta: period = 1, offset = 0 (tight bounds)
    n_starts: int = 2  # random restarts of the global search (best fine-scale loss wins)
    smooth_cols: int = 1  # average neighbouring flux columns before peak finding (3 helps very noisy maps)
    param_bounds: dict = field(default_factory=dict)  # user knowledge: name -> (lo, hi); lo == hi fixes the value
    coupled_margin: float = 0.05  # a resonator model must beat the bare circuit by this much to be accepted
    coupled_trials: int = 2  # how many of the best bare families get a resonator trial
    prior_weight: float = 0.05  # weight of the CNN family probability in the score (0 = ignore); the CNN is confidently wrong on some resonator-crossing maps


def _emit(progress: Progress | None, kind: str, **payload):
    """UI callbacks must never be able to abort a fit: swallow their exceptions (they are cosmetic)."""
    if progress:
        try:
            progress(kind, payload)
        except Exception:
            pass


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


def _apply_user_bounds(bounds: dict, cfg: "AgentConfig") -> dict:
    """Narrow or fix parameters the user already knows (design values, a single-tone resonator scan...)."""
    b = dict(bounds)
    for k, v in (cfg.param_bounds or {}).items():
        if k not in b:
            continue
        lo, hi = (float(v), float(v)) if np.isscalar(v) else (float(v[0]), float(v[1]))
        if lo == hi:
            lo, hi = lo * (1 - 1e-4) if lo != 0 else -1e-6, hi * (1 + 1e-4) if hi != 0 else 1e-6
        b[k] = (min(lo, hi), max(lo, hi))
    return b


def _fit_points_for(hyp: fit.Hypothesis, pts_all: np.ndarray, flat: np.ndarray) -> np.ndarray:
    """Flux-dependent circuit hypotheses are fitted on the flux-dependent features only (flat lines are
    pumps/spurs/fixed neighbours); a fixed-frequency circuit's own lines are flat, so it is fitted on
    everything, and so are the other platforms (a charge qubit far from zero detuning is nearly flat)."""
    n_dyn = int((~flat).sum())
    if hyp.flux_dependent and n_dyn >= max(10, 0.2 * len(pts_all)):
        return pts_all[~flat]
    return pts_all


def missing_line_penalty(res: fit.FitResult, pts_all: np.ndarray, band, c_local: float) -> float:
    """Visibility-weighted fraction of predicted lines that run through measured columns with nothing
    under them. A model that predicts a strong 0-1 line where the data shows nothing is implausible;
    an unseen faint 0-3 line costs little."""
    vis = models.line_weights(res.hyp.family)
    cols = np.unique(pts_all[:, 0])
    lines = fit.predict_lines(res.hyp, res.params, cols)
    tot, miss = 0.0, 0.0
    for name, Li in lines.items():
        if name == "res":
            continue
        inband = np.isfinite(Li) & (Li > band[0]) & (Li < band[1])
        if inband.sum() < 3:
            continue
        has = 0
        for c, Lc in zip(cols[inband], Li[inband]):
            near = pts_all[np.isclose(pts_all[:, 0], c), 1]
            if near.size and np.min(np.abs(near - Lc)) < 3 * c_local:
                has += 1
        frac_missing = 1.0 - has / inband.sum()
        tot += vis[name]
        miss += vis[name] * frac_missing
    return miss / tot if tot > 0 else 0.0


def _score(res: fit.FitResult, pts_all, ids, flat, band, c_local, model_kw=None) -> tuple[float, np.ndarray]:
    obj = fit.Objective(res.hyp, pts_all, band, c_local, model_kw)
    r = obj.residuals(res.theta)
    inl = r < 3 * c_local
    ts = fit.track_score(pts_all, ids, flat, inl)
    ts -= 0.3 * missing_line_penalty(res, pts_all, band, c_local)
    # complexity penalty + precision tie-break: a model that explains the same features with a 4 MHz residual
    # beats a look-alike that needs 15 MHz
    rms_pen = min(float(np.nan_to_num(res.rms_inliers, nan=0.03)) * 1e3, 30.0) / 300.0
    return float(ts - 0.02 * len(res.params) - rms_pen), inl


def unexplained_features(pts_all, ids, inlier_all, flat, qubit_line=None) -> list[dict]:
    out = []
    for s in extract.summarize_tracks(pts_all, ids):
        m = ids == s["track"]
        frac_out = 1.0 - float(np.mean(inlier_all[m]))
        if frac_out < 0.6 or m.sum() < 3:
            continue
        s = dict(s, unexplained_frac=round(frac_out, 2))
        if s["flat"] or flat[m].mean() > 0.5:
            s["interpretation"] = "flux-independent line: likely pump/LO leakage, a mixer spur, or a fixed-frequency neighbour (resonator, TLS, other qubit)"
            if qubit_line is not None:
                d = np.asarray(qubit_line, float) - s["f_mean"]
                ok = np.isfinite(d)
                if ok.sum() > 1 and np.any(d[ok][:-1] * d[ok][1:] <= 0):
                    s["interpretation"] = "flux-independent line that the qubit crosses with no visible avoided crossing: a pump/LO/mixer tone or an uncoupled mode (g below the linewidth), not a coupled resonator"
        else:
            s["interpretation"] = "flux-dependent feature not in the model: candidate neighbouring tunable mode, higher transition, or sideband of a qubit line"
        out.append(s)
    return out


def prune_look_here(look: list[dict], x_range: tuple[float, float], band: tuple[float, float], max_area_frac: float = 0.2) -> list[dict]:
    """A suggestion is only useful if it points somewhere specific: drop boxes covering more than
    `max_area_frac` of the map and the weak thermal 1-2 line unless its box is small."""
    area = max((x_range[1] - x_range[0]) * (band[1] - band[0]), 1e-12)
    keep = []
    for o in look:
        a = (o["x_hi"] - o["x_lo"]) * (o["f_hi"] - o["f_lo"]) / area
        if a > max_area_frac:
            continue
        if o["line"] == "1-2" and a > 0.05:
            continue
        keep.append(o)
    return keep


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
            # split the missing columns into contiguous runs and report the longest two as boxes
            xs = np.array([m[0] for m in missing])
            fs = np.array([m[1] for m in missing])
            order = np.argsort(xs)
            xs, fs = xs[order], fs[order]
            dxg = np.median(np.diff(np.unique(x_grid))) if len(x_grid) > 1 else 1.0
            runs, start = [], 0
            for i in range(1, len(xs) + 1):
                if i == len(xs) or xs[i] - xs[i - 1] > 2.5 * dxg:
                    runs.append((start, i))
                    start = i
            runs.sort(key=lambda ab: -(ab[1] - ab[0]))
            why = f"the model predicts the {name} line here but nothing was detected"
            if name.endswith("2ph"):
                why += "; raise the drive power to bring out the two-photon transition"
            elif name == "1-2":
                why += "; a weak thermal population line, visible with higher power or a pre-pulse"
            for a, b_ in runs[:2]:
                if b_ - a < 3:
                    continue
                out.append(dict(kind="look_here", line=name, x_lo=round(float(xs[a]), 4), x_hi=round(float(xs[b_ - 1]), 4), f_lo=round(float(fs[a:b_].min()), 3), f_hi=round(float(fs[a:b_].max()), 3), why=why))
    # keep the most useful suggestions: 0-1 first, then by run length; drop boxes that would cover the map
    prio = {"0-1": 0, "0-2": 1, "0-2/2ph": 2, "1-2": 3, "0-3": 4}
    look = [o for o in out if o["kind"] == "look_here"]
    look = prune_look_here(look, (float(np.min(x_grid)), float(np.max(x_grid))), band)
    look.sort(key=lambda o: (prio.get(o["line"], 9), -(o["x_hi"] - o["x_lo"])))
    out = [o for o in out if o["kind"] != "look_here"] + look[:4]
    # poorly determined parameters
    for n, s in res.sigma.items():
        v = res.params.get(n, np.nan)
        if np.isfinite(s) and abs(v) > 0 and s / abs(v) > 0.1:
            s = min(s, 10 * abs(v))  # cap: 'unconstrained' rather than an absurd percentage
            hint = {"EC": "EC is set by the anharmonicity: the 0-2/2ph (f01 - EC/2) or 1-2 (f01 - EC) lines pin it",
                    "EJ": "EJ is degenerate with EC along constant f01; the 0-2/2ph line breaks the degeneracy",
                    "EJmax": "EJmax is degenerate with EC along constant f01; the 0-2/2ph line breaks the degeneracy",
                    "EL": "EL is best constrained by the 0-1 line near half flux and by the 0-2 transition",
                    "d": "the junction asymmetry d is set by the minimum of the 0-1 line at half flux: extend the flux sweep",
                    "g": "the coupling g is set by the avoided-crossing splitting: measure finely where the qubit crosses the resonator",
                    "fr": "the resonator frequency: a single-tone scan pins it",
                    "D": "the zero-field splitting D is read off at B = 0: include the zero-field point in the sweep",
                    "E": "the strain splitting E is the line separation at B = 0: sweep through zero field with fine steps",
                    "theta": "the field polar angle sets how the four NV orientations fan out: a larger field range separates them",
                    "phi": "the field azimuth separates the orientations that currently overlap: tilt the field slightly",
                    "t_c": "the tunnel coupling is the minimum of the hyperbola at zero detuning: measure finely around it",
                    "lever_arm": "the lever arm is the slope of the line far from zero detuning: extend the detuning range",
                    "x0": "the zero of the control axis: measure on both sides of the minimum",
                    "f_rad0": "the radial secular frequency: raise the drive power until the radial sidebands appear",
                    "volt_scale": "the axial frequency per sqrt(volt): extend the endcap-voltage range",
                    "alpha1": "polarisability of the |mj| = 1/2 component: extend the field range so the parabola curves clearly",
                    "alpha2": "polarisability of the |mj| = 3/2 component: same, and resolve the two components at low field",
                    "dfs": "the zero-field splitting of the two components: fine steps around the stray-field null",
                    "f0": "the zero-field line centre: fine steps around the stray-field null"}.get(n, "")
            out.append(dict(kind="constrain_parameter", parameter=n, rel_sigma=round(float(s / abs(v)), 3), why=hint))
    return out


def identify(image: np.ndarray | None = None, x: np.ndarray | None = None, f: np.ndarray | None = None, points: np.ndarray | None = None,
             band: tuple[float, float] | None = None, config: AgentConfig | None = None, progress: Progress | None = None) -> dict:
    cfg = config or AgentConfig()
    platform = models.platform_of(cfg.families[0]) if cfg.families else "circuit"
    scales = models.platform_scales(platform)
    if cfg.c_local is None:
        cfg = AgentConfig(**{**cfg.__dict__, "c_local": scales["c_local"]})
    t_start = time.time()
    # ---- 1. features
    if points is None:
        assert image is not None and x is not None and f is not None, "need image+x+f or points"
        points = extract.extract_points(image, x, f, smooth_cols=cfg.smooth_cols)
        band = band or (float(f[0]), float(f[-1]))
    else:
        points = np.asarray(points, float)
        if image is None:
            image, x, f = rasterize(points)
        band = band or (float(points[:, 1].min()) - 0.1, float(points[:, 1].max()) + 0.1)
    x_grid = np.asarray(x, float) if x is not None else np.unique(points[:, 0])
    if len(points) < 10:
        return dict(error="fewer than 10 spectral features found; lower the detection threshold or check the map", n_points=int(len(points)))
    ids = extract.link_tracks(points, df_tol=scales["df_tol"])
    flat = extract.flat_feature_mask(points, tol=scales["flat_tol"])
    flat_lines = extract.flat_lines(points, tol=scales["flat_tol"])
    _emit(progress, "features", n_points=int(len(points)), n_tracks=int(len(np.unique(ids))), flat_lines=flat_lines)

    # ---- 2. propose
    proposals = []
    net = None
    circuit_only = all(models.platform_of(f) == "circuit" for f in (cfg.families or ("transmon",)))
    if cfg.use_classifier and circuit_only:
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
        fams = cfg.families or tuple(models.PLATFORMS["circuit"]["families"])
        bare = [(fit.Hypothesis(fam, False), None) for fam in fams]
        p_coupled = 0.5
    for h in cfg.extra_hypotheses:
        if h not in [b[0] for b in bare]:
            bare.append((h, None))
    if cfg.families:
        bare = [(h, p) for h, p in bare if h.family in cfg.families] or [(fit.Hypothesis(f, False), None) for f in cfg.families]
    _emit(progress, "proposals", hypotheses=[dict(h=h.label, prob=(p or {}).get("prob")) for h, p in bare], p_coupled=p_coupled)
    p_family: dict[str, float] = {}
    for prop in proposals:
        p_family[prop["family"]] = p_family.get(prop["family"], 0.0) + prop["prob"]

    def prior(h: fit.Hypothesis) -> float:
        if not p_family or cfg.prior_weight <= 0:
            return 0.0
        return cfg.prior_weight * (p_family.get(h.family, 1 / 3) - 1 / 3)

    # ---- 3. fit bare hypotheses
    results: list[tuple[fit.FitResult, float, np.ndarray]] = []

    def run_one(h: fit.Hypothesis, pts_fit, init, bounds, maxiter, n_starts=None):
        bounds = _apply_user_bounds(bounds if bounds is not None else fit.default_bounds(h, pts_fit[:, 0]), cfg) if cfg.param_bounds else bounds

        def cb(params, loss):
            _emit(progress, "fit_progress", hypothesis=h.label, params=params, loss=loss, lines={k: v.tolist() for k, v in fit.predict_lines(h, params, x_grid).items()})

        return fit.fit_hypothesis(pts_fit, h, band, init=init, bounds=bounds, maxiter=maxiter, popsize=cfg.popsize, seed=cfg.seed, workers=cfg.workers, callback=cb,
                                  c_local=cfg.c_local, c_global=scales["c_global"], n_starts=cfg.n_starts if n_starts is None else n_starts)

    def run(h: fit.Hypothesis, init: dict | None):
        pts_fit = _fit_points_for(h, points, flat)
        _emit(progress, "fit_start", hypothesis=h.label, n_points=int(len(pts_fit)))
        candidates: list[tuple[dict | None, dict | None, int]] = [(None, init, cfg.maxiter)]  # (bounds, init, maxiter); wide search first
        if h.flux_dependent and cfg.flux_calibrated:
            b = fit.default_bounds(h, pts_fit[:, 0])
            b["period"] = (0.97, 1.03)
            b["offset"] = (-0.03, 0.03)
            candidates = [(b, dict(init or {}, period=1.0, offset=0.0), cfg.maxiter)]
        elif h.flux_dependent and h.platform == "circuit":
            cal = extract.estimate_flux_calibration(pts_fit)
            if cal["period"]:
                P = cal["period"]
                for ax in cal["axes"][:1]:
                    if ax["type"] == "unknown":
                        continue
                    off = ax["x"] if ax["type"] == "max" else ax["x"] - P / 2
                    b = fit.default_bounds(h, pts_fit[:, 0])
                    b["period"] = (0.85 * P, 1.15 * P)
                    b["offset"] = (off - 0.08 * P, off + 0.08 * P)
                    ini = dict(init or {})
                    ini.update(period=P, offset=off)
                    candidates.append((b, ini, max(20, int(cfg.maxiter * 0.6))))
                _emit(progress, "calibration", period=P, axes=cal["axes"])
        best_res, best_sc, best_inl = None, -np.inf, None
        for b, ini, mi in candidates:
            try:
                res = run_one(h, pts_fit, ini, b, mi, n_starts=None if b is None else 1)
            except Exception as e:  # a narrowed candidate must never take the whole run down
                _emit(progress, "warning", message=f"{h.label}: candidate search failed ({type(e).__name__}: {str(e)[:80]})")
                continue
            score, inl = _score(res, points, ids, flat, band, cfg.c_local)
            score += prior(h)
            if score > best_sc:
                best_res, best_sc, best_inl = res, score, inl
        if best_res is None:
            return None, -np.inf
        best_res.lines = fit.predict_lines(h, best_res.params, x_grid)
        results.append((best_res, best_sc, best_inl))
        _emit(progress, "fit_done", hypothesis=h.label, score=best_sc, summary=best_res.summary(), lines={k: v.tolist() for k, v in best_res.lines.items()})
        return best_res, best_sc

    n_dyn = int((~flat).sum())
    for h, p in bare:
        if h.platform == "circuit" and h.flux_dependent and n_dyn < max(10, 0.2 * len(points)):
            _emit(progress, "skip", hypothesis=h.label, reason=f"only {n_dyn} of {len(points)} features vary with flux")
            continue  # (almost) everything is flux-independent: a flux-tunable circuit cannot be the explanation
        init = None
        if p is not None:
            init = {k: v for k, v in p["seed"].items() if k in h.names}
        run(h, init)
    if not results:
        return dict(error="no hypothesis could be fitted", n_points=int(len(points)))
    results.sort(key=lambda r: -r[1])
    best, best_score, best_inl = results[0]

    # ---- 4. revise: try the resonator-coupled version of the best family.
    #      Seeds for fr: where the bare model fails most (an avoided crossing leaves a cluster of
    #      unexplained flux-dependent points at the resonator frequency), then the proposer, then flat lines.
    want_coupled = cfg.try_coupled == "always" or (cfg.try_coupled == "auto" and (p_coupled > 0.25 or best_score < 0.85))
    bare_sorted = [r for r in results if not r[0].hyp.coupled]
    for base_res, base_score, base_inl in bare_sorted[: cfg.coupled_trials]:  # the best bare families get a resonator trial
        if not want_coupled or base_score < best_score - 0.35 or base_res.hyp.platform != "circuit":
            continue
        hc = fit.Hypothesis(base_res.hyp.family, True)
        fr_cands: list[float] = []
        bad = (~base_inl) & (~flat)
        if bad.sum() >= 5:
            hist, edges = np.histogram(points[bad, 1], bins=np.arange(band[0], band[1] + 0.2, 0.2), weights=points[bad, 2])
            if hist.max() > 0:
                fr_cands.append(float(0.5 * (edges[np.argmax(hist)] + edges[np.argmax(hist) + 1])))
        for prop in proposals:
            if prop["family"] == hc.family and prop["coupled"] and "fr" in prop["seed"]:
                fr_cands.append(float(prop["seed"]["fr"]))
                break
        fr_cands += [fl for fl in flat_lines if 4.0 < fl < 10.0]
        fr_cands.append(7.0)
        seen, cands = set(), []
        for v in fr_cands:
            v = float(np.clip(v, *models.RESONATOR_BOUNDS["fr"]))
            if round(v, 1) not in seen:
                seen.add(round(v, 1))
                cands.append(v)
        pts_fit = _fit_points_for(hc, points, flat)
        _emit(progress, "revise", reason=f"trying {hc.label} (p_coupled={p_coupled:.2f}, bare score {base_score:.3f}); resonator candidates {', '.join(f'{v:.2f}' for v in cands[:2])} GHz")
        _emit(progress, "fit_start", hypothesis=hc.label, n_points=int(len(pts_fit)))
        cbest, csc, cinl = None, -np.inf, None
        for fr0 in cands[: max(1, cfg.coupled_trials)]:
            b = fit.default_bounds(hc, pts_fit[:, 0])
            for k, v in base_res.params.items():  # circuit + calibration stay near the bare solution
                if k in b and k not in ("d", "offset", "period"):
                    b[k] = (max(0.5 * v, b[k][0]), min(2.0 * v, b[k][1]))
                elif k == "period":
                    b[k] = (0.9 * v, 1.1 * v)
                elif k == "offset":
                    b[k] = (v - 0.1 * base_res.params.get("period", 1.0), v + 0.1 * base_res.params.get("period", 1.0))
            b["fr"] = (max(fr0 - 0.6, models.RESONATOR_BOUNDS["fr"][0]), min(fr0 + 0.6, models.RESONATOR_BOUNDS["fr"][1]))
            init = dict(base_res.params, fr=fr0, g=0.1)
            try:
                res = run_one(hc, pts_fit, init, b, cfg.maxiter, n_starts=1)
            except Exception as e:
                _emit(progress, "warning", message=f"{hc.label}: search failed ({type(e).__name__})")
                continue
            score, inl = _score(res, points, ids, flat, band, cfg.c_local)
            score += prior(hc)
            if score > csc:
                cbest, csc, cinl = res, score, inl
        if cbest is None:
            continue
        cbest.lines = fit.predict_lines(hc, cbest.params, x_grid)
        accepted = csc > base_score + cfg.coupled_margin
        reason = f"gain {csc - base_score:+.3f} < margin {cfg.coupled_margin}"
        g_sig = cbest.sigma.get("g", float("nan"))
        if accepted and np.isfinite(g_sig) and cbest.params.get("g", 0) < 2 * g_sig:
            accepted, reason = False, "coupling g not significant"
        if accepted:
            # a resonator must buy its place with FLUX-DEPENDENT evidence (a bend, a crossing, a dispersive pull):
            # explaining one more flat line is not evidence, any tone does that
            dyn_mask = ~flat
            if dyn_mask.sum() >= 10:
                gain_dyn = fit.track_score(points[dyn_mask], ids[dyn_mask], flat[dyn_mask], cinl[dyn_mask]) - fit.track_score(points[dyn_mask], ids[dyn_mask], flat[dyn_mask], base_inl[dyn_mask])
                if gain_dyn < 0.02:
                    accepted, reason = False, f"no flux-dependent evidence of coupling (gain on flux-dependent features {gain_dyn:+.3f}); the extra line could be any tone"
        results.append((cbest, csc if accepted else min(csc, base_score - 1e-6), cinl))
        _emit(progress, "fit_done", hypothesis=hc.label, score=csc, summary=cbest.summary(), lines={k: v.tolist() for k, v in cbest.lines.items()},
              note=("accepted" if accepted else "rejected: " + reason))
    results.sort(key=lambda r: -r[1])
    best, best_score, best_inl = results[0]

    # coherent line segments (>= 5 points): the features a physicist would count; isolated noise fragments are not
    seg_ok = np.zeros(len(points), bool)
    for t_id in np.unique(ids):
        m = ids == t_id
        if m.sum() >= 5:
            seg_ok[m] = True

    # ---- 5. verify on held-out columns (coherent segments only, so noise fragments do not dilute the score)
    if cfg.holdout:
        pts_fit = _fit_points_for(best.hyp, points, flat)
        coh = pts_fit[np.isin(pts_fit[:, 0] * 1e6 + pts_fit[:, 1], (points[seg_ok, 0] * 1e6 + points[seg_ok, 1]))] if seg_ok.sum() >= 20 else pts_fit
        used = coh if len(coh) >= 20 else pts_fit
        fit.holdout_check(used, best, band, c_local=cfg.c_local)
        _cols = np.unique(used[:, 0])
        holdout_columns = [float(v) for v in _cols[::3]]  # same rule as fit.holdout_check
    else:
        holdout_columns = []
        _emit(progress, "verify", holdout=best.holdout)

    # ---- 6. report
    note = ""
    if best.hyp.family == "tunable_transmon" and best.params.get("d", 0) > 0.95:
        note = "junction asymmetry d > 0.95: the spectrum is effectively flux-insensitive, i.e. a fixed-frequency transmon with EJ = EJmax"
    # confidence from the score gap to the best *other family* (same-family resonator variants do not count)
    others = [sc for res, sc, _ in results[1:] if res.hyp.family != best.hyp.family]
    gap = best_score - max(others) if others else 1.0
    runner = next((res.hyp.label for res, sc, _ in results[1:] if res.hyp.family != best.hyp.family), None)
    if not others:
        confidence = dict(level="single family", gap=None, text="only one circuit family was considered, so no alternative was tested")
    elif gap > 0.15:
        confidence = dict(level="high", gap=round(float(gap), 3), text="the best hypothesis beats every other circuit family clearly")
    elif gap > 0.05:
        confidence = dict(level="medium", gap=round(float(gap), 3), text=f"{runner} is a plausible alternative; the look-here suggestions would separate them")
    else:
        confidence = dict(level="low", gap=round(float(gap), 3), text=f"ambiguous between {best.hyp.label} and {runner}: more data is needed before trusting the parameters")
    # user-facing: of the points that belong to coherent line segments (spurs excluded), the weighted
    # fraction lying on a predicted transition
    dyn = (~flat) & seg_ok
    if dyn.sum() < 5:
        dyn = seg_ok if seg_ok.sum() >= 5 else np.ones(len(points), bool)
    w = points[dyn, 2]
    explained_dynamic = float(np.sum(w * best_inl[dyn]) / max(np.sum(w), 1e-12))
    unexpl = unexplained_features(points, ids, best_inl, flat, qubit_line=best.lines.get("0-1"))
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
        best=dict(res_summary(best), track_score=round(best_score, 4), explained_dynamic=round(float(explained_dynamic), 4), note=note, platform=best.hyp.platform, confidence=confidence),
        unexplained=unexpl,
        next_measurements=nxt,
        lines={k: [None if not np.isfinite(v) else round(float(v), 5) for v in arr] for k, arr in best.lines.items()},
        x_grid=[float(v) for v in x_grid],
        inlier_mask=best_inl.tolist(),
        points=points.tolist(),
        track_ids=ids.tolist(),
        holdout_columns=holdout_columns,
    )
    try:
        from . import derived

        xr = (float(np.min(x_grid)), float(np.max(x_grid)))
        report["derived"] = [dict(d, value=(None if not np.isfinite(d["value"]) else round(float(d["value"]), 6))) for d in derived.derived_quantities(report["best"], xr)]
        report["scqubits_code"] = derived.scqubits_code(report["best"])
    except Exception as e:  # derived quantities are a convenience, never a failure
        report["derived"] = []
        report["scqubits_code"] = f"# unavailable: {type(e).__name__}: {e}"
    _emit(progress, "done", report=report)
    return report


def res_summary(res: fit.FitResult) -> dict:
    s = res.summary()
    s["params_pretty"] = {k: f"{v:.4g} ± {res.sigma.get(k, float('nan')):.2g}" + (" GHz" if k in fit.LOG_PARAMS and k != "period" else "") for k, v in res.params.items()}
    return s


def to_json(report: dict) -> str:
    return json.dumps(report, indent=1, default=lambda o: float(o) if isinstance(o, (np.floating, np.integer)) else str(o))
