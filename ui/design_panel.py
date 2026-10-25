"""Experimental-design panel: expected information gain per candidate sweep and the parameter correlation matrix."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

SURFACE, TEXT, MUTED, GRID, BLUE, AMBER = "#1a1a19", "#f0efec", "#c3c2b7", "#2a2a28", "#3987e5", "#fab219"
LINE_COLORS = {"0-1": "#d95926", "0-2": "#c98500", "0-3": "#9085e9", "0-2/2ph": "#d55181", "1-2": "#199e70", "res": "#f0efec"}


def _layout(fig, title, xlab, ylab, height):
    fig.update_layout(title=dict(text=title, font=dict(size=14, color=TEXT), x=0.01), paper_bgcolor=SURFACE, plot_bgcolor=SURFACE,
                      font=dict(family="IBM Plex Sans, Inter, Helvetica Neue, Arial, sans-serif", color=TEXT, size=12), margin=dict(l=56, r=16, t=40, b=44), height=height,
                      xaxis=dict(title=xlab, gridcolor=GRID, zerolinecolor=GRID, color=MUTED), yaxis=dict(title=ylab, gridcolor=GRID, zerolinecolor=GRID, color=MUTED),
                      legend=dict(orientation="h", y=-0.28, x=0, bgcolor="rgba(0,0,0,0)", font=dict(size=11)))
    return fig


def gain_figure(design: dict, xlabel: str = "control", height: int = 260):
    """Expected information gain (bits) of one extra column, per line, versus control value."""
    x = np.asarray(design.get("x_cols", []), float)
    fig = go.Figure()
    for name, g in design.get("gain_map", {}).items():
        g = np.asarray(g, float)
        if not np.any(g > 0):
            continue
        fig.add_trace(go.Scatter(x=x, y=g, mode="lines", name=name.replace("-", "→"), line=dict(color=LINE_COLORS.get(name, MUTED), width=2),
                                 hovertemplate=f"{name}<br>%{{x:.3f}}<br>%{{y:.2f}} bits<extra></extra>"))
    for i, c in enumerate(design.get("candidates", [])[:3], start=1):
        fig.add_vrect(x0=c["x_lo"], x1=c["x_hi"], fillcolor=AMBER, opacity=0.12, line_width=0)
        fig.add_annotation(x=0.5 * (c["x_lo"] + c["x_hi"]), y=1.0, yref="paper", text=str(i), showarrow=False, font=dict(color=AMBER, size=12), bgcolor=SURFACE)
    return _layout(fig, "Expected information gain of one more column, per transition (bits)", xlabel, "bits", height)


def correlation_figure(design: dict, height: int = 300):
    names, corr = design["correlation"]
    corr = np.asarray(corr, float)
    fig = go.Figure(go.Heatmap(z=corr, x=names, y=names, zmin=-1, zmax=1, colorscale=[[0, "#1c5cab"], [0.5, SURFACE], [1, "#d03b3b"]],
                               text=[[f"{v:+.2f}" for v in row] for row in corr], texttemplate="%{text}", textfont=dict(size=11, family="IBM Plex Mono, monospace"),
                               hovertemplate="%{y} · %{x}: %{z:+.2f}<extra></extra>", showscale=False))
    fig = _layout(fig, "Parameter correlations (Laplace covariance of the fit)", "", "", height)
    fig.update_yaxes(autorange="reversed")
    return fig
