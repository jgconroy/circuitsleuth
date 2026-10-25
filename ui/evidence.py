"""Proof-of-correctness figures for the CircuitSleuth UI.

Residual strip to sit under the map, hold-out map, avoided-crossing zoom, pull plot against known truth, and the
data for the spec cards. Styling follows ui/plots.py: its palette, fonts, heat colourscale and per-line colours are
imported from there (with the same values as fallbacks), legends sit outside the plotting area (below, horizontal),
fonts are >= 12 px. The module imports numpy, plotly and circuitsleuth only (never streamlit), so every figure can be
built and inspected from a plain script; each figure carries a small summary in `fig.layout.meta` for that purpose.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from circuitsleuth import extract, fit, models

try:  # one source of truth for the look: ui/plots.py (no streamlit in there)
    from . import plots as _plots
except Exception:  # pragma: no cover - evidence must still work if plots is mid-edit
    _plots = None


def _const(name: str, default):
    return getattr(_plots, name, default) if _plots is not None else default


SURFACE, TEXT, MUTED, GRID = _const("SURFACE", "#1a1a19"), _const("TEXT", "#f0efec"), _const("MUTED", "#c3c2b7"), _const("GRID", "#2a2a28")
ACCENT, BLUE, LIGHT_BLUE = _const("ACCENT", "#fab219"), _const("BLUE", "#3987e5"), _const("LIGHT_BLUE", "#86b6ef")
GREEN, RED = _const("GREEN", "#0ca30c"), _const("RED", "#d03b3b")
FONT = _const("FONT", 'Inter, "Helvetica Neue", Helvetica, Arial, sans-serif')
MONO = _const("MONO", FONT)
_HEAT = ["#1a1a19", "#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]  # single hue, surface -> light blue
HEAT_SCALE = _const("HEAT_SCALE", [[i / (len(_HEAT) - 1), c] for i, c in enumerate(_HEAT)])
LINE_STYLE = _const("LINE_STYLE", {
    "0-1": dict(color="#d95926", width=3),
    "0-2": dict(color="#c98500", width=2),
    "0-3": dict(color="#9085e9", width=2),
    "0-2/2ph": dict(color="#d55181", width=2, dash="dash"),
    "1-2": dict(color="#199e70", width=2, dash="dot"),
    "res": dict(color="#f0efec", width=2, dash="dashdot"),
})
OTHER_LINE = _const("OTHER_LINE", dict(color=MUTED, width=2))
HOVER_PT = "control %{x:.3f}<br>%{y:.3f} GHz<extra>%{fullData.name}</extra>"


# ----------------------------------------------------------------------------- shared helpers
def _style(fig: go.Figure, height: int, title: str | None = None, legend: bool = True, margin: dict | None = None, legend_offset_px: int = 46) -> go.Figure:
    """House layout as in ui/plots._style: legend horizontal below the plotting area (placed `legend_offset_px` under
    the axis line so it clears the tick labels and the axis title whatever the figure height), mono face for numbers."""
    margin = margin or dict(l=62, r=14, t=40 if title else 14, b=92)
    plot_h = max(height - margin.get("t", 0) - margin.get("b", 0), 1)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family=FONT, size=13, color=TEXT), margin=margin,
        legend=dict(orientation="h", x=0, xanchor="left", y=-legend_offset_px / plot_h, yanchor="top", bgcolor="rgba(0,0,0,0)", borderwidth=0,
                    font=dict(size=12, color=TEXT, family=FONT), itemsizing="constant", tracegroupgap=2),
        showlegend=legend, hovermode="closest", uirevision="keep",
        hoverlabel=dict(bgcolor="#222221", bordercolor=GRID, font=dict(family=MONO, size=12, color=TEXT)),
    )
    if title:
        from ui.theme import title_case as _tc
        title = title if "=" in title else _tc(title)
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=14, color=MUTED, family=FONT)))
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=GRID, showline=True, ticks="outside", tickcolor=GRID,
                tickfont=dict(size=12, color=MUTED, family=MONO), title_font=dict(size=13, color=MUTED, family=FONT))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    from ui.plots import capitalize_labels

    return capitalize_labels(fig)


def _finite(vals) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in vals], float)


def _points(report: dict) -> np.ndarray:
    return np.asarray(report.get("points") or [], float).reshape(-1, 3)


def _inlier_mask(report: dict, n: int) -> np.ndarray | None:
    m = report.get("inlier_mask")
    if m is None or len(m) != n:
        return None
    return np.asarray(m, bool)


def _line_arrays(report: dict) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """x_grid (sorted) and the predicted lines of the best hypothesis as float arrays with NaN gaps."""
    xg = np.asarray(report.get("x_grid") or [], float)
    order = np.argsort(xg)
    lines = {}
    for name, vals in (report.get("lines") or {}).items():
        v = _finite(vals)
        if len(v) == len(xg) and np.isfinite(v).any():
            lines[name] = v[order]
    return xg[order], lines


def _interp(xg: np.ndarray, v: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Linear interpolation of a line with NaN gaps: NaN wherever either bracketing grid value is missing."""
    out = np.full(len(xq), np.nan)
    n = len(xg)
    if n == 0:
        return out
    j = np.clip(np.searchsorted(xg, xq, side="right") - 1, 0, n - 1)
    k = np.clip(j + 1, 0, n - 1)
    span = xg[-1] - xg[0] if n > 1 else 1.0
    tol = 1e-9 * max(abs(span), 1.0)
    inside = (xq >= xg[0] - tol) & (xq <= xg[-1] + tol)
    dx = np.where(k != j, xg[k] - xg[j], 1.0)
    t = np.where(k != j, (xq - xg[j]) / dx, 0.0)
    both = np.isfinite(v[j]) & np.isfinite(v[k]) & inside
    out[both] = (v[j] + t * (v[k] - v[j]))[both]
    exact = ~both & inside & np.isfinite(v[j]) & (np.abs(xq - xg[j]) <= tol)
    out[exact] = v[j][exact]
    return out


def _nearest_line(report: dict, P: np.ndarray):
    """Per point: residual to the nearest predicted line (GHz), that line's label, and its predicted frequency."""
    xg, lines = _line_arrays(report)
    n = len(P)
    resid = np.full(n, np.nan)
    pred = np.full(n, np.nan)
    label = np.array([""] * n, dtype=object)
    if not lines or n == 0:
        return resid, label, pred
    names = list(lines)
    L = np.stack([_interp(xg, lines[k], P[:, 0]) for k in names])  # (nlines, npoints)
    D = P[None, :, 1] - L
    A = np.where(np.isfinite(D), np.abs(D), np.inf)
    best = np.argmin(A, axis=0)
    ok = np.isfinite(A[best, np.arange(n)])
    resid[ok] = D[best, np.arange(n)][ok]
    pred[ok] = L[best, np.arange(n)][ok]
    for i in np.flatnonzero(ok):
        label[i] = names[best[i]]
    return resid, label, pred


def _on_columns(xs: np.ndarray, cols: np.ndarray) -> np.ndarray:
    if len(cols) == 0 or len(xs) == 0:
        return np.zeros(len(xs), bool)
    span = float(np.ptp(xs)) or 1.0
    return np.min(np.abs(xs[:, None] - cols[None, :]), axis=1) <= 1e-6 * span


def _holdout_subsets(report: dict, P: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """Candidate point sets the agent may have handed to fit.holdout_check, from the least to the most selective:
    everything; flux-dependent features only (flat spurs dropped, as agent._fit_points_for does); and those
    further restricted to coherent tracks of >= 5 points (the current agent.identify)."""
    out = [("all", np.ones(len(P), bool))]
    best = report.get("best") or {}
    try:
        fam = best["family"]
        hyp = fit.Hypothesis(fam, bool(best.get("coupled", False)))
        scales = models.platform_scales(models.platform_of(fam))
        flat = extract.flat_feature_mask(P, tol=scales["flat_tol"])
        n_dyn = int((~flat).sum())
        fit_mask = ~flat if (hyp.flux_dependent and n_dyn >= max(10, 0.2 * len(P))) else np.ones(len(P), bool)
        out.append(("flux-dependent", fit_mask))
        ids = report.get("track_ids")
        if ids is not None and len(ids) == len(P):
            ids = np.asarray(ids)
            seg_ok = np.zeros(len(P), bool)
            for t in np.unique(ids):
                m = ids == t
                if m.sum() >= 5:
                    seg_ok[m] = True
            coh = fit_mask & seg_ok if seg_ok.sum() >= 20 else fit_mask
            used = coh if coh.sum() >= 20 else fit_mask
            out.append(("coherent", used))
    except Exception:
        pass
    return out


def _holdout_info(report: dict) -> tuple[np.ndarray, np.ndarray]:
    """(held-out column x values, boolean mask of the points that were actually scored on them).

    Uses report['holdout_columns'] when present. Otherwise every 3rd unique column (fit.holdout_check's rule) of the
    candidate point set whose count on those columns reproduces holdout['n_test_points'], falling back to every 3rd
    unique x of all points."""
    P = _points(report)
    if len(P) == 0:
        return np.array([]), np.zeros(0, bool)
    n_test = ((report.get("best") or {}).get("holdout") or {}).get("n_test_points")
    subsets = _holdout_subsets(report, P)
    explicit = report.get("holdout_columns")
    if explicit:
        cols = np.sort(np.asarray(explicit, float))
        on = _on_columns(P[:, 0], cols)
        for _, sub in subsets:
            if n_test is None or int((sub & on).sum()) == int(n_test):
                return cols, sub & on
        return cols, on
    for _, sub in subsets:
        if not sub.any():
            continue
        cols = np.unique(P[sub, 0])[::3]
        m = sub & _on_columns(P[:, 0], cols)
        if n_test is None or int(m.sum()) == int(n_test):
            return cols, m
    cols = np.unique(P[:, 0])[::3]
    return cols, _on_columns(P[:, 0], cols)


def holdout_columns(report: dict) -> np.ndarray:
    """Control values of the columns the hold-out check never fitted (see _holdout_info)."""
    return _holdout_info(report)[0]


def _heatmap(image, x, f) -> go.Heatmap:
    img = np.clip(extract.normalise(np.asarray(image, float)), 0, 10)
    return go.Heatmap(z=img, x=np.asarray(x, float), y=np.asarray(f, float), zmin=0, zmax=10, colorscale=HEAT_SCALE, showscale=False, name="map",
                      hovertemplate="control %{x:.3f}<br>%{y:.3f} GHz<extra></extra>")


def _add_lines(fig: go.Figure, report: dict, **kw):
    xg, lines = _line_arrays(report)
    for name, v in lines.items():
        fig.add_trace(go.Scatter(x=xg, y=v, mode="lines", name=_pretty(name), line=LINE_STYLE.get(name, OTHER_LINE), connectgaps=False,
                                 hovertemplate=f"{name}: %{{y:.3f}} GHz at %{{x:.3f}}<extra></extra>"), **kw)


def capitalize_labels(fig):
    from ui.plots import capitalize_labels as _cl

    return _cl(fig)


def _pretty(name: str) -> str:
    """Legend text for a transition key (0-1 -> 0→1, res -> resonator)."""
    from ui.plots import pretty

    return pretty(str(name))


def _param_html(name: str) -> str:
    from ui.theme import PARAM_HTML

    return PARAM_HTML.get(str(name), str(name))


def _label_order(labels) -> list[str]:
    present = {str(l) for l in labels if l}
    return [k for k in LINE_STYLE if k in present] + sorted(present - set(LINE_STYLE))


# ----------------------------------------------------------------------------- 1. residual strip
def _to_MHz(report) -> float:
    try:
        from circuitsleuth import models as _m
        return _m.to_MHz(report["best"]["family"])
    except Exception:
        return 1e3


def residual_figure(report: dict, c_local: float = 0.012, height: int = 220) -> go.Figure | None:
    """data - model (MHz) for every detected feature against the nearest predicted line, on the map's x range.

    Points are coloured by the line they were assigned to, hollow when they sit on a held-out column. The blue band
    is the inlier threshold +-3 c_local, the dashed lines are +-RMS over the inliers; the y axis is clipped to
    +-max(5 RMS, 60 MHz) and points beyond it are clamped to the edge as triangles and counted in the annotation."""
    P = _points(report)
    xg, lines = _line_arrays(report)
    if len(P) == 0 or not lines:
        return None
    resid, label, pred = _nearest_line(report, P)
    r = _to_MHz(report) * resid
    ok = np.isfinite(r)
    inl = _inlier_mask(report, len(P))
    if inl is None:
        inl = ok & (np.abs(resid) < 3 * c_local)
    rms = float(np.sqrt(np.mean(r[inl & ok] ** 2))) if (inl & ok).any() else float("nan")
    band = 3e3 * c_local
    clip = max(5 * rms if np.isfinite(rms) else 0.0, 60.0)
    cols, _ = _holdout_info(report)
    held = _on_columns(P[:, 0], cols)
    outside = ok & (np.abs(r) > clip)

    fig = go.Figure()
    fig.add_hrect(y0=-band, y1=band, fillcolor=BLUE, opacity=0.12, line_width=0, layer="below")
    fig.add_hline(y=0, line=dict(color=GRID, width=1))
    if np.isfinite(rms):
        for s in (1, -1):
            fig.add_hline(y=s * rms, line=dict(color=MUTED, width=1, dash="dash"))
        fig.add_annotation(xref="paper", x=0.004, y=rms, xanchor="left", yanchor="bottom", text=f"RMS {rms:.1f} MHz ({int((inl & ok).sum())} inliers)",
                           showarrow=False, font=dict(size=12, color=MUTED, family=MONO), yshift=1)
    fig.add_annotation(xref="paper", x=0.004, y=-band, xanchor="left", yanchor="top", text=f"±{band:.0f} MHz inlier band",
                       showarrow=False, font=dict(size=12, color=LIGHT_BLUE, family=MONO), yshift=-1)
    for lab in _label_order(label):
        m = ok & (label == lab) & ~outside
        if not m.any():
            continue
        color = LINE_STYLE.get(lab, OTHER_LINE)["color"]
        symbols = np.where(held[m], "circle-open", "circle").tolist()
        custom = np.column_stack([pred[m], np.where(held[m], "held-out column", "fit column")]).tolist()
        fig.add_trace(go.Scatter(x=P[m, 0], y=r[m], mode="markers", name=f"{_pretty(lab)} ({int((ok & (label == lab)).sum())})", customdata=custom,
                                 marker=dict(symbol=symbols, size=8, color=color, line=dict(width=1.5, color=color)),
                                 hovertemplate=f"control %{{x:.3f}}<br>residual %{{y:+.1f}} MHz<br>nearest line {lab} at %{{customdata[0]:.3f}} GHz"
                                               "<br>%{customdata[1]}<extra></extra>"))
    if outside.any():
        sgn = np.sign(r[outside])
        fig.add_trace(go.Scatter(x=P[outside, 0], y=0.94 * clip * sgn, mode="markers", name="beyond range", showlegend=False, customdata=r[outside],
                                 marker=dict(symbol=np.where(sgn > 0, "triangle-up", "triangle-down").tolist(), size=8, color=MUTED, line=dict(width=0)),
                                 hovertemplate="control %{x:.3f}<br>residual %{customdata:+.0f} MHz (off scale)<extra></extra>"))
    if held.any():
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name="hollow = held-out column",
                                 marker=dict(symbol="circle-open", size=8, color=TEXT, line=dict(width=1.5, color=TEXT)), hoverinfo="skip"))
    n_out = int(outside.sum())
    fig.add_annotation(xref="paper", yref="paper", x=0.996, y=0.96, xanchor="right", yanchor="top", showarrow=False,
                       text=(f"{n_out} unexplained points beyond ±{clip:.0f} MHz" if n_out else f"all points within ±{clip:.0f} MHz"),
                       font=dict(size=12, color=RED if n_out else MUTED, family=MONO))
    _style(fig, height, margin=dict(l=62, r=14, t=14, b=72), legend_offset_px=26)  # same l/r as the map above it
    fig.update_xaxes(range=[float(xg[0]), float(xg[-1])], showgrid=False)
    fig.update_yaxes(range=[-clip, clip], title_text="data − model (MHz)")
    fig.update_layout(meta=dict(kind="residual", n_points=int(len(P)), n_assigned=int(ok.sum()), n_heldout=int(held.sum()), n_outside=n_out,
                                rms_MHz=None if not np.isfinite(rms) else round(rms, 3), clip_MHz=round(clip, 1), band_MHz=round(band, 1),
                                n_heldout_columns=int(len(cols))))
    return capitalize_labels(fig)


# ----------------------------------------------------------------------------- 2. hold-out map
def holdout_figure(report: dict, image, x, f, height: int = 420) -> go.Figure | None:
    """The map with the never-fitted columns shaded, the fitted lines, the held-out features as hollow squares
    (green within 3 sigma of a predicted line, red outside, by report['inlier_mask']) and the rest as dim dots."""
    P = _points(report)
    x, f = np.asarray(x, float), np.asarray(f, float)
    if len(P) == 0 or image is None:
        return None
    cols, test = _holdout_info(report)
    inl = _inlier_mask(report, len(P))
    if inl is None:
        resid, _, _ = _nearest_line(report, P)
        inl = np.abs(resid) < 0.036
    fig = go.Figure()
    fig.add_trace(_heatmap(image, x, f))
    ux = np.unique(x)
    dx = float(np.median(np.diff(ux))) if len(ux) > 1 else 0.02
    for c in cols:
        fig.add_vrect(x0=float(c) - dx / 2, x1=float(c) + dx / 2, fillcolor=TEXT, opacity=0.08, line_width=0, layer="above")
    if len(cols):
        fig.add_trace(go.Scatter(x=[None], y=[None], mode="markers", name=f"held-out columns ({len(cols)})",
                                 marker=dict(symbol="square", size=12, color=TEXT, opacity=0.3, line=dict(width=0)), hoverinfo="skip"))
    rest = ~test
    if rest.any():
        fig.add_trace(go.Scatter(x=P[rest, 0], y=P[rest, 1], mode="markers", name=f"not held out ({int(rest.sum())})",
                                 marker=dict(size=4, color=MUTED, opacity=0.6, line=dict(width=0)), hovertemplate=HOVER_PT))
    for m, color, nm in ((test & inl, GREEN, "held-out · within 3σ"), (test & ~inl, RED, "held-out · outside 3σ")):
        if m.any():
            fig.add_trace(go.Scatter(x=P[m, 0], y=P[m, 1], mode="markers", name=f"{_pretty(nm)} ({int(m.sum())})",
                                     marker=dict(symbol="square-open", size=9, color=color, line=dict(width=1.5, color=color)), hovertemplate=HOVER_PT))
    _add_lines(fig, report)
    hd = (report.get("best") or {}).get("holdout") or {}
    if "inlier_frac" in hd:
        title = f"held-out columns: {hd['inlier_frac']:.0%} of features within 3σ, RMS {hd.get('rms_MHz', float('nan')):.1f} MHz"
    else:
        title = "held-out columns: no hold-out verification in this report"
    _style(fig, height, title=title, margin=dict(l=62, r=14, t=40, b=108))
    fig.update_xaxes(title_text="control (flux bias, arbitrary units)", range=[float(x[0]), float(x[-1])], showgrid=False)
    fig.update_yaxes(title_text="drive frequency (GHz)", range=[float(f[0]), float(f[-1])], showgrid=False)
    fig.update_layout(meta=dict(kind="holdout", n_columns=int(len(cols)), n_test=int(test.sum()), n_test_inlier=int((test & inl).sum()),
                                n_rest=int(rest.sum()), title=title, reported=dict(hd)))
    return capitalize_labels(fig)


# ----------------------------------------------------------------------------- 3. avoided-crossing zoom
def _bare_crossings(xs: np.ndarray, bare01: np.ndarray, fr: float) -> list[float]:
    d = bare01 - fr
    out = []
    for i in range(len(xs) - 1):
        if np.isfinite(d[i]) and np.isfinite(d[i + 1]) and d[i] * d[i + 1] <= 0 and d[i] != d[i + 1]:
            if d[i] == 0 and out and abs(out[-1] - xs[i]) < 1e-9:
                continue
            out.append(float(xs[i] - d[i] * (xs[i + 1] - xs[i]) / (d[i + 1] - d[i])))
    return out


def _break_swaps(xw: np.ndarray, a: np.ndarray, b: np.ndarray, g: float):
    """The dressed lines are labelled by maximum bare-state overlap, so at the crossing '0-1' and 'res' swap
    branches with a jump of ~2g in opposite directions. Insert a NaN there so neither draws a vertical segment."""
    xs, aa, bb = list(xw), list(a), list(b)
    cut = []
    for i in range(len(xw) - 1):
        da, db = a[i + 1] - a[i], b[i + 1] - b[i]
        if np.isfinite(da) and np.isfinite(db) and da * db < 0 and min(abs(da), abs(db)) > 0.5 * g:
            cut.append(i)
    for i in reversed(cut):
        xs.insert(i + 1, 0.5 * (xw[i] + xw[i + 1]))
        aa.insert(i + 1, np.nan)
        bb.insert(i + 1, np.nan)
    return np.array(xs), np.array(aa), np.array(bb)


def crossing_zoom_figure(report: dict, image, x, f, height: int = 360) -> go.Figure | None:
    """Zoom on where the bare 0-1 transition would cross the resonator: data crop, bare 0-1 (dashed), fr (dashed),
    the dressed 0-1 and resonator branches, the features in the window and a bracket for the splitting.
    None unless the best hypothesis is resonator-coupled and the bare line crosses fr inside the measured range."""
    best = report.get("best") or {}
    if not best.get("coupled"):
        return None
    params = dict(best.get("params") or {})
    fr, g = params.get("fr"), params.get("g")
    if fr is None or g is None or not (g > 0):
        return None
    sigma_g = float((best.get("sigma") or {}).get("g", float("nan")))
    family = best["family"]
    xg, lines = _line_arrays(report)
    if len(xg) < 2:
        return None
    x, f = np.asarray(x, float), np.asarray(f, float)
    img = np.clip(extract.normalise(np.asarray(image, float)), 0, 10)
    P = _points(report)
    inl = _inlier_mask(report, len(P))
    if inl is None:
        inl = np.ones(len(P), bool)
    hyp_bare, hyp_dressed = fit.Hypothesis(family, False), fit.Hypothesis(family, True)
    bare_params = {k: v for k, v in params.items() if k not in ("fr", "g")}
    x_fine = np.linspace(float(xg[0]), float(xg[-1]), max(4 * len(xg), 401))
    try:
        bare_fine = fit.predict_lines(hyp_bare, bare_params, x_fine)["0-1"]
    except Exception:
        return None
    crossings = _bare_crossings(x_fine, bare_fine, fr)
    if not crossings:
        return None
    hw, fw = 0.12, 4.0 * g
    f_lo, f_hi = fr - fw, fr + fw

    def n_in_window(xc):
        return int(((np.abs(P[:, 0] - xc) <= hw) & (P[:, 1] >= f_lo) & (P[:, 1] <= f_hi)).sum()) if len(P) else 0

    ranked = sorted(crossings, key=lambda xc: (-n_in_window(xc), xc))
    chosen = sorted(ranked[:2])
    n = len(chosen)
    fig = make_subplots(rows=1, cols=n, shared_yaxes=True, horizontal_spacing=0.06,
                        subplot_titles=[f"crossing at control = {xc:.3f}" for xc in chosen])
    fig.update_annotations(font=dict(size=13, color=MUTED, family=FONT))
    gaps = []
    for c, xc in enumerate(chosen, start=1):
        kw = dict(row=1, col=c)
        first = c == 1
        x0, x1 = max(xc - hw, float(xg[0])), min(xc + hw, float(xg[-1]))
        ix = (x >= xc - hw) & (x <= xc + hw)
        iy = (f >= f_lo) & (f <= f_hi)
        if ix.sum() >= 2 and iy.sum() >= 2:
            fig.add_trace(go.Heatmap(z=img[np.ix_(iy, ix)], x=x[ix], y=f[iy], zmin=0, zmax=10, colorscale=HEAT_SCALE, showscale=False, name="map",
                                     hovertemplate="control %{x:.3f}<br>%{y:.3f} GHz<extra></extra>"), **kw)
        xw = np.linspace(x0, x1, 161)
        try:
            bare_w = fit.predict_lines(hyp_bare, bare_params, xw)["0-1"]
        except Exception:
            bare_w = _interp(x_fine, bare_fine, xw)
        try:
            dressed = fit.predict_lines(hyp_dressed, params, xw)
            d01, dres = np.asarray(dressed["0-1"], float), np.asarray(dressed["res"], float)
        except Exception:  # fall back to the report's lines on the coarse grid
            d01 = _interp(xg, lines.get("0-1", np.full(len(xg), np.nan)), xw)
            dres = _interp(xg, lines.get("res", np.full(len(xg), np.nan)), xw)
        xd, d01b, dresb = _break_swaps(xw, d01, dres, g)
        fig.add_trace(go.Scatter(x=xw, y=bare_w, mode="lines", name="bare 0→1 (no resonator)", legendgroup="bare", showlegend=first,
                                 line=dict(color=MUTED, width=2, dash="dash"), hovertemplate="bare 0-1: %{y:.3f} GHz at %{x:.3f}<extra></extra>"), **kw)
        fig.add_trace(go.Scatter(x=[x0, x1], y=[fr, fr], mode="lines", name=f"f<sub>r</sub> = {fr:.3f} GHz", legendgroup="fr", showlegend=first,
                                 line=dict(color=TEXT, width=1.5, dash="dash"), hoverinfo="skip"), **kw)
        fig.add_trace(go.Scatter(x=xd, y=d01b, mode="lines", name="dressed 0→1", legendgroup="d01", showlegend=first, connectgaps=False,
                                 line=dict(color=LINE_STYLE["0-1"]["color"], width=3), hovertemplate="dressed 0-1: %{y:.3f} GHz at %{x:.3f}<extra></extra>"), **kw)
        fig.add_trace(go.Scatter(x=xd, y=dresb, mode="lines", name="dressed resonator", legendgroup="res", showlegend=first, connectgaps=False,
                                 line=dict(color=LINE_STYLE["res"]["color"], width=2), hovertemplate="dressed resonator: %{y:.3f} GHz at %{x:.3f}<extra></extra>"), **kw)
        if len(P):
            win = (P[:, 0] >= xc - hw) & (P[:, 0] <= xc + hw) & (P[:, 1] >= f_lo) & (P[:, 1] <= f_hi)
            for m, color, sym, nm, sz in ((win & inl, GREEN, "circle", "explained", 7), (win & ~inl, RED, "x", "unexplained", 8)):
                if m.any():
                    fig.add_trace(go.Scatter(x=P[m, 0], y=P[m, 1], mode="markers", name=nm, legendgroup=nm, showlegend=first and True,
                                             marker=dict(size=sz, color=color, symbol=sym, opacity=0.95, line=dict(width=0)), hovertemplate=HOVER_PT), **kw)
        # bracket: the two dressed branches exactly at the crossing
        try:
            at = fit.predict_lines(hyp_dressed, params, np.array([xc]))
            ya, yb = float(at["0-1"][0]), float(at["res"][0])
        except Exception:
            ya, yb = float(_interp(xw, d01, np.array([xc]))[0]), float(_interp(xw, dres, np.array([xc]))[0])
        if np.isfinite(ya) and np.isfinite(yb):
            ylo, yhi = min(ya, yb), max(ya, yb)
            gap = _to_MHz(report) * (yhi - ylo)
            gaps.append(round(gap, 1))
            cap = 0.012
            for (xa, xb, y0, y1) in ((xc, xc, ylo, yhi), (xc - cap, xc + cap, ylo, ylo), (xc - cap, xc + cap, yhi, yhi)):
                fig.add_shape(type="line", x0=xa, x1=xb, y0=y0, y1=y1, line=dict(color=ACCENT, width=2), **kw)
            text = f"2g ≈ {2000 * g:.0f} ± {2000 * sigma_g:.0f} MHz" if np.isfinite(sigma_g) else f"2g ≈ {2000 * g:.0f} MHz"
            if abs(gap - 2000 * g) > 0.05 * 2000 * g:
                text += f"<br><span style='color:{MUTED}'>gap here {gap:.0f} MHz</span>"  # 2g·|⟨0|n|1⟩| at this flux
            # centred at the top of the sub-plot: the text then never runs into the neighbouring panel
            fig.add_annotation(x=xc, y=f_hi - 0.03 * (f_hi - f_lo), text=text, showarrow=False, xanchor="center", yanchor="top",
                               font=dict(size=12, color=ACCENT, family=MONO), align="center", **kw)
        fig.update_xaxes(range=[xc - hw, xc + hw], title_text="control", showgrid=False, **kw)
    title = f"Avoided crossing{'s' if n > 1 else ''} at f<sub>r</sub> = {fr:.3f} GHz: dressed branches split by 2g ≈ {2000 * g:.0f} MHz"
    _style(fig, height, title=title, margin=dict(l=62, r=14, t=64, b=108))
    fig.update_yaxes(range=[f_lo, f_hi], showgrid=False)
    fig.update_yaxes(title_text="drive frequency (GHz)", row=1, col=1)
    fig.update_layout(meta=dict(kind="crossing", n_crossings=len(crossings), crossings=[round(v, 4) for v in crossings], shown=[round(v, 4) for v in chosen],
                                two_g_MHz=round(2000 * g, 1), sigma_two_g_MHz=None if not np.isfinite(sigma_g) else round(2000 * sigma_g, 1), branch_gap_MHz=gaps))
    return capitalize_labels(fig)


# ----------------------------------------------------------------------------- 4. pull plot
def pull_figure(report: dict, truth_params: dict, height: int = 260) -> go.Figure | None:
    """(fit - truth) / sigma per parameter shared by the fit and `truth_params`; 'offset' is compared modulo the fitted
    period. Bands mark +-1 sigma (green) and +-2 sigma (amber); pulls beyond +-6 are clamped and listed."""
    best = report.get("best") or {}
    p, s = best.get("params") or {}, best.get("sigma") or {}
    if not truth_params:
        return None
    period = p.get("period")
    rows = []
    for k, v in p.items():
        if k not in truth_params:
            continue
        sig = s.get(k)
        if sig is None or not np.isfinite(sig) or sig <= 0:
            continue
        diff = float(v) - float(truth_params[k])
        if k == "offset" and period:
            diff = (diff + period / 2) % period - period / 2
        rows.append((k, float(v), float(truth_params[k]), float(sig), diff / sig))
    if not rows:
        return None
    lim = 6.0
    names = [r[0] for r in rows]
    pulls = np.array([r[4] for r in rows])
    clipped = np.clip(pulls, -0.96 * lim, 0.96 * lim)
    beyond = np.abs(pulls) > lim
    colors = [RED if abs(v) > 3 else (ACCENT if abs(v) > 2 else BLUE) for v in pulls]
    symbols = ["triangle-right" if (b and v > 0) else ("triangle-left" if b else "circle") for v, b in zip(pulls, beyond)]
    ys = np.arange(len(rows))
    fig = go.Figure()
    fig.add_vrect(x0=-2, x1=2, fillcolor=ACCENT, opacity=0.08, line_width=0, layer="below")
    fig.add_vrect(x0=-1, x1=1, fillcolor=GREEN, opacity=0.10, line_width=0, layer="below")
    fig.add_vline(x=0, line=dict(color=GRID, width=1))
    for y, v, c in zip(ys, clipped, colors):
        fig.add_shape(type="line", x0=0, x1=float(v), y0=float(y), y1=float(y), line=dict(color=c, width=2))
    custom = [[r[1], r[2], r[3], r[4], r[0]] for r in rows]  # fit, truth, sigma, pull, name
    fig.add_trace(go.Scatter(x=clipped, y=ys, mode="markers+text", customdata=custom, text=[f"{v:+.1f}σ" for v in pulls],
                             textposition=["middle right" if v >= 0 else "middle left" for v in clipped], textfont=dict(size=12, color=TEXT, family=MONO), cliponaxis=False,
                             marker=dict(size=10, color=colors, symbol=symbols, line=dict(color=SURFACE, width=1.5)), showlegend=False,
                             hovertemplate="%{customdata[4]}<br>fit %{customdata[0]:.5g} · truth %{customdata[1]:.5g}<br>σ %{customdata[2]:.3g} → pull %{customdata[3]:+.2f}<extra></extra>"))
    for xv, txt, col in ((1, "±1σ", GREEN), (2, "±2σ", ACCENT)):
        fig.add_annotation(x=xv, yref="paper", y=1.0, yanchor="bottom", xanchor="left", xshift=2, text=txt, showarrow=False, font=dict(size=12, color=col, family=MONO))
    if beyond.any():
        fig.add_annotation(xref="paper", yref="paper", x=0.996, y=0.02, xanchor="right", yanchor="bottom", showarrow=False, align="right",
                           text="off scale: " + ", ".join(f"{_param_html(n)} {v:+.1f}σ" for n, v, b in zip(names, pulls, beyond) if b), font=dict(size=12, color=RED, family=MONO))
    _style(fig, height, title="pull per parameter", legend=False, margin=dict(l=72, r=16, t=52, b=48))
    fig.update_xaxes(title_text="(fit − truth) / σ", range=[-lim, lim], dtick=1, showgrid=True)
    fig.update_yaxes(tickvals=ys, ticktext=[_param_html(n) for n in names], range=[len(rows) - 0.4, -0.6], showgrid=False, tickfont=dict(size=12, color=TEXT))  # first parameter on top
    fig.update_layout(meta=dict(kind="pull", pulls={n: round(float(v), 3) for n, v in zip(names, pulls)}, n_beyond=int(beyond.sum())))
    return capitalize_labels(fig)


# ----------------------------------------------------------------------------- 5. spec cards
_CONTROL_KEYS = ("control", "sensitivity", "period")
_READOUT_KEYS = ("detuning", "g /", "δ", "dispersive", "purcell")


def _fmt_value(v, unit: str) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    try:
        s = f"{float(v):.4g}"
    except (TypeError, ValueError):
        s = str(v)
    if "e" in s:  # 1.234e+04 -> 1.234e4
        mant, exp = s.split("e")
        s = f"{mant}e{int(exp)}"
    return f"{s} {unit}".strip() if unit else s


def spec_card_data(report: dict) -> dict[str, list[tuple[str, str]]]:
    """Three lists of (label, value string) for the Qubit, Control axis and Readout cards, from report['derived'].

    Keyword routing on the quantity name: control / sensitivity / period / 'sweet spot ...' -> Control axis;
    detuning / 'g /' / Δ / dispersive / Purcell -> Readout; everything else (f01, anharmonicity, EJ, EC, EL,
    plasma, charge dispersion, ...) -> Qubit. Values are shown with 4 significant digits plus their unit."""
    out: dict[str, list[tuple[str, str]]] = {"Qubit": [], "Control axis": [], "Readout": []}
    for d in report.get("derived") or []:
        name = str(d.get("name", "")).strip()
        if not name:
            continue
        low = name.lower()
        value = _fmt_value(d.get("value"), str(d.get("unit") or ""))
        if any(k in low for k in _CONTROL_KEYS) or low.startswith("sweet spot"):
            bucket = "Control axis"
        elif any(k in low for k in _READOUT_KEYS):
            bucket = "Readout"
        else:
            bucket = "Qubit"
        out[bucket].append((name, value))
    return out
