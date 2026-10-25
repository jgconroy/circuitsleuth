"""Experimental-design panel: expected information gain per candidate sweep and the parameter correlation matrix."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

SURFACE, TEXT, MUTED, GRID, BLUE, AMBER = "#1a1a19", "#f0efec", "#c3c2b7", "#2a2a28", "#3987e5", "#fab219"
LINE_COLORS = {"0-1": "#d95926", "0-2": "#c98500", "0-3": "#9085e9", "0-2/2ph": "#d55181", "1-2": "#199e70", "res": "#f0efec"}


def _p(name: str) -> str:
    """Typeset parameter name for Plotly labels (HTML subscripts)."""
    from ui.theme import PARAM_HTML

    return PARAM_HTML.get(name, name)


def _layout(fig, title, xlab, ylab, height):
    fig.update_layout(title=dict(text=(title if (not title or "=" in title) else __import__("ui.theme", fromlist=["title_case"]).title_case(title)), font=dict(size=14, color=TEXT), x=0.01), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(family="IBM Plex Sans, Inter, Helvetica Neue, Arial, sans-serif", color=TEXT, size=12), margin=dict(l=56, r=16, t=40, b=84), height=height,
                      xaxis=dict(title=dict(text=xlab, standoff=8), gridcolor=GRID, zerolinecolor=GRID, color=MUTED), yaxis=dict(title=ylab, gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
                      legend=dict(orientation="h", y=-0.42, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    from ui.plots import capitalize_labels

    return capitalize_labels(fig)


def gain_figure(design: dict, xlabel: str = "control", height: int = 260):
    """Expected information gain (bits) of one extra column, per line, versus control value."""
    x = np.asarray(design.get("x_cols", []), float)
    fig = go.Figure()
    for name, g in design.get("gain_map", {}).items():
        g = np.asarray(g, float)
        if not np.any(g > 0):
            continue
        fig.add_trace(go.Scatter(x=x, y=g, mode="lines", name=__import__("ui.plots", fromlist=["pretty"]).pretty(name), line=dict(color=LINE_COLORS.get(name, MUTED), width=2),
                                 hovertemplate=f"{name}<br>%{{x:.3f}}<br>%{{y:.2f}} bits<extra></extra>"))
    for i, c in enumerate(design.get("candidates", [])[:3], start=1):
        fig.add_vrect(x0=c["x_lo"], x1=c["x_hi"], fillcolor=AMBER, opacity=0.12, line_width=0)
        fig.add_annotation(x=0.5 * (c["x_lo"] + c["x_hi"]), y=1.0, yref="paper", text=str(i), showarrow=False, font=dict(color=AMBER, size=12), bgcolor=SURFACE)
    return _layout(fig, "Expected information gain of one more column, per transition (bits)", xlabel, "bits", height)


def correlation_figure(design: dict, height: int = 300):
    names, corr = design["correlation"]
    corr = np.asarray(corr, float)
    fig = go.Figure(go.Heatmap(z=corr, x=[_p(n) for n in names], y=[_p(n) for n in names], zmin=-1, zmax=1, colorscale=[[0, "#1c5cab"], [0.5, SURFACE], [1, "#d03b3b"]],
                               text=[[f"{v:+.2f}" for v in row] for row in corr], texttemplate="%{text}", textfont=dict(size=11, family="IBM Plex Mono, monospace"),
                               hovertemplate="%{y} · %{x}: %{z:+.2f}<extra></extra>", showscale=False))
    fig = _layout(fig, "Parameter correlations (Laplace covariance of the fit)", "", "", height)
    fig.update_yaxes(autorange="reversed")
    return fig


def sigma_vs_round_figure(loop_result: dict, height: int = 280):
    """Relative 1-sigma uncertainty of every parameter after each autopilot round (log scale)."""
    rows = [r for r in loop_result["rounds"] if "params" in r]
    if not rows:
        return None
    names = [k for k in rows[0]["params"] if k not in ("offset", "x0")]
    palette = ["#d95926", "#3987e5", "#199e70", "#c98500", "#d55181", "#9085e9", "#f0efec", "#4cae4c"]
    fig = go.Figure()
    for i, n in enumerate(names):
        y = [abs(r["sigma"].get(n, np.nan) / r["params"][n]) if r["params"][n] else np.nan for r in rows]
        fig.add_trace(go.Scatter(x=[r["round"] for r in rows], y=y, mode="lines+markers", name=_p(n), line=dict(color=palette[i % len(palette)], width=2), marker=dict(size=7),
                                 hovertemplate=f"{n}<br>round %{{x}}<br>σ/|value| = %{{y:.2%}}<extra></extra>"))
    fig = _layout(fig, "Relative uncertainty per autopilot round", "round (0 = initial partial sweep)", "σ / |value|", height)
    fig.update_yaxes(type="log", tickformat=".0%")
    fig.update_xaxes(dtick=1)
    return fig


def posterior_pair_choice(names) -> tuple[str, str]:
    """The most interesting parameter pair to show: the EJ-EC valley when it exists, else the first two parameters."""
    names = list(names)
    for pair in (("EJmax", "EC"), ("EJ", "EC"), ("EJ", "EL")):
        if pair[0] in names and pair[1] in names:
            return pair
    if len(names) >= 2:
        return (names[0], names[1])
    return (names[0], names[0]) if names else ("", "")


def _sigma_ellipse(cov2: np.ndarray, centre, n_sigma: float = 1.0, n: int = 121):
    vals, vecs = np.linalg.eigh(0.5 * (cov2 + cov2.T))
    vals = np.sqrt(np.clip(vals, 0.0, None))
    t = np.linspace(0, 2 * np.pi, n)
    circ = np.stack([np.cos(t), np.sin(t)])
    pts = np.asarray(centre, float)[:, None] + n_sigma * (vecs @ (vals[:, None] * circ))
    return pts[0], pts[1]


def posterior_figure(samples_dict: dict, pair: tuple[str, str] = ("EJmax", "EC"), laplace_cov=None, height: int = 320):
    """2-D scatter of posterior samples (``fit.sample_posterior``) for `pair`: faint blue points, the 1-sigma Laplace
    ellipse (amber, dashed) from `laplace_cov` (parameter units, full matrix over ``names``; defaults to the dict's own
    ``laplace_cov_params``), and the MAP (amber diamond)."""
    from circuitsleuth import models

    names = list(samples_dict["names"])
    S = np.asarray(samples_dict["samples"], float)
    if pair is None or pair[0] not in names or pair[1] not in names:
        pair = posterior_pair_choice(names)
    i, j = names.index(pair[0]), names.index(pair[1])
    fam = samples_dict.get("family")

    def label(n):
        u = models.param_unit(fam, n) if fam else models.PARAM_UNITS.get(n, "")
        return f"{_p(n)} ({u})" if u else _p(n)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=S[:, i], y=S[:, j], mode="markers", name="posterior samples",
                             marker=dict(size=5, color=BLUE, opacity=0.35, line=dict(width=0)),
                             hovertemplate=f"{pair[0]} %{{x:.5g}}<br>{pair[1]} %{{y:.5g}}<extra>sample</extra>"))
    cov = laplace_cov if laplace_cov is not None else samples_dict.get("laplace_cov_params")
    map_ = samples_dict.get("map")
    centre = samples_dict.get("mode") or map_  # the Laplace covariance is evaluated at the mode of the sampled likelihood
    if cov is not None and centre is not None:
        cov = np.asarray(cov, float)
        sub = cov[np.ix_([i, j], [i, j])] if cov.shape[0] == len(names) else cov[:2, :2]
        if np.all(np.isfinite(sub)):
            ex, ey = _sigma_ellipse(sub, [centre[pair[0]], centre[pair[1]]])
            fig.add_trace(go.Scatter(x=ex, y=ey, mode="lines", name="1σ Laplace (fit covariance)", line=dict(color=AMBER, width=2, dash="dash"), hoverinfo="skip"))
    if map_ is not None:
        fig.add_trace(go.Scatter(x=[map_[pair[0]]], y=[map_[pair[1]]], mode="markers", name="best fit (MAP)",
                                 marker=dict(symbol="diamond", size=12, color=AMBER, line=dict(color=SURFACE, width=1)),
                                 hovertemplate=f"MAP<br>{pair[0]} %{{x:.5g}}<br>{pair[1]} %{{y:.5g}}<extra></extra>"))
    acc = samples_dict.get("acceptance_rate")
    sub_t = f" · {len(S)} MCMC samples" + (f", acceptance {acc:.0%}" if acc is not None else "")
    fig = _layout(fig, f"Posterior of {_p(pair[0])} and {_p(pair[1])} (robust likelihood of the fit){sub_t}", label(pair[0]), label(pair[1]), height)
    fig.update_layout(hovermode="closest", legend=dict(orientation="h", y=-0.3, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    fig.update_xaxes(tickfont=dict(family="IBM Plex Mono, monospace", size=11, color=MUTED))
    fig.update_yaxes(tickfont=dict(family="IBM Plex Mono, monospace", size=11, color=MUTED))
    return fig
