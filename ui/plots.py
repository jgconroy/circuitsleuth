"""Plotly figures for the CircuitSleuth UI.

Every figure sits on the surface colour #1a1a19 with text #f0efec, hairline grid #2a2a28 and a system sans.
Transition lines carry fixed colours per label so a line never changes colour between hypotheses.
"""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from circuitsleuth import extract

SURFACE, TEXT, MUTED, GRID = "#1a1a19", "#f0efec", "#c3c2b7", "#2a2a28"
ACCENT, BLUE, LIGHT_BLUE, GREEN, RED = "#fab219", "#3987e5", "#86b6ef", "#0ca30c", "#d03b3b"
FONT = 'Inter, "Helvetica Neue", Helvetica, Arial, sans-serif'
HEAT = ["#1a1a19", "#0d366b", "#1c5cab", "#3987e5", "#86b6ef", "#cde2fb"]  # single hue, surface -> light blue
HEAT_SCALE = [[i / (len(HEAT) - 1), c] for i, c in enumerate(HEAT)]
LINE_STYLE = {
    "0-1": dict(color="#d95926", width=3),
    "0-2": dict(color="#c98500", width=2),
    "0-3": dict(color="#9085e9", width=2),
    "0-2/2ph": dict(color="#d55181", width=2, dash="dash"),
    "1-2": dict(color="#199e70", width=2, dash="dot"),
    "res": dict(color="#f0efec", width=2, dash="dashdot"),
}
OTHER_LINE = dict(color=MUTED, width=2)
CONFIG = dict(displaylogo=False, modeBarButtonsToRemove=["select2d", "lasso2d", "autoScale2d", "toggleSpikelines"])
PI_TICKS = {k * np.pi / 2: t for k, t in zip(range(-6, 7), ["−3π", "", "−2π", "", "−π", "−π/2", "0", "π/2", "π", "", "2π", "", "3π"])}


def _style(fig: go.Figure, height: int, title: str | None = None, legend: bool = True, margin: dict | None = None) -> go.Figure:
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family=FONT, size=13, color=TEXT),
        margin=margin or dict(l=60, r=16, t=40 if title else 16, b=52),
        legend=dict(x=0.985, y=0.975, xanchor="right", yanchor="top", bgcolor="rgba(26,26,25,0.62)", bordercolor=GRID, borderwidth=1,
                    font=dict(size=12, color=TEXT), itemsizing="constant", tracegroupgap=2),
        showlegend=legend, hovermode="closest", uirevision="keep",
        hoverlabel=dict(bgcolor="#222221", bordercolor=GRID, font=dict(family=FONT, size=12, color=TEXT)),
    )
    if title:
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=14, color=TEXT)))
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=GRID, showline=True, ticks="outside", tickcolor=GRID,
                tickfont=dict(size=12, color=MUTED), title_font=dict(size=13, color=MUTED))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _finite(vals) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in vals], float)


def map_figure(image, x, f, points=None, inlier=None, lines=None, x_grid=None, look_here=None, flat_lines=None, title=None, xlabel="control (flux bias, arbitrary units)",
               markers=None) -> go.Figure:
    """Spectroscopy map with detected features, fitted transitions, 'look here' segments and flat spur lines.

    `markers` is an optional list of (control, frequency) operating points drawn as small amber diamonds."""
    x, f = np.asarray(x, float), np.asarray(f, float)
    img = np.clip(extract.normalise(np.asarray(image, float)), 0, 10)
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=img, x=x, y=f, zmin=0, zmax=10, colorscale=HEAT_SCALE, showscale=False, name="map",
                             hovertemplate="control %{x:.3f}<br>%{y:.3f} GHz<extra></extra>"))
    if points is not None and len(points):
        P = np.asarray(points, float)
        pt_hover = "control %{x:.3f}<br>%{y:.3f} GHz<extra>%{fullData.name}</extra>"
        if inlier is None:
            fig.add_trace(go.Scatter(x=P[:, 0], y=P[:, 1], mode="markers", name=f"detected features ({len(P)})",
                                     marker=dict(size=6, color=LIGHT_BLUE, opacity=0.9, line=dict(width=0)), hovertemplate=pt_hover))
        else:
            inl = np.asarray(inlier, bool)
            fig.add_trace(go.Scatter(x=P[inl, 0], y=P[inl, 1], mode="markers", name=f"explained ({inl.sum()})",
                                     marker=dict(size=6, color=GREEN, symbol="circle", opacity=0.9, line=dict(width=0)), hovertemplate=pt_hover))
            fig.add_trace(go.Scatter(x=P[~inl, 0], y=P[~inl, 1], mode="markers", name=f"unexplained ({(~inl).sum()})",
                                     marker=dict(size=8, color=RED, symbol="x", opacity=0.95, line=dict(width=0)), hovertemplate=pt_hover))
    xg = np.asarray(x_grid, float) if x_grid is not None else None
    if lines and xg is not None:
        for name, vals in lines.items():
            v = _finite(vals)
            if not np.isfinite(v).any():
                continue
            fig.add_trace(go.Scatter(x=xg, y=v, mode="lines", name=name, line=LINE_STYLE.get(name, OTHER_LINE), connectgaps=False,
                                     hovertemplate=f"{name}: %{{y:.3f}} GHz at %{{x:.3f}}<extra></extra>"))
    first = True
    for b in look_here or []:
        name = b.get("line")
        if not (lines and xg is not None and name in lines):
            continue
        v = _finite(lines[name])
        m = (xg >= b["x_lo"]) & (xg <= b["x_hi"]) & np.isfinite(v)
        if not m.any():
            continue
        fig.add_trace(go.Scatter(x=xg[m], y=v[m], mode="lines", name="look here", legendgroup="look", showlegend=first,
                                 line=dict(color=ACCENT, width=8), opacity=0.45, hoverinfo="skip"))
        fig.add_annotation(x=float(xg[m][0]), y=float(v[m][0]), text=f"look here: {name}", showarrow=False, xanchor="left", yanchor="bottom",
                           xshift=2, yshift=6, font=dict(size=12, color=ACCENT))
        first = False
    for fl in flat_lines or []:
        fig.add_hline(y=float(fl), line=dict(color=RED, width=1), opacity=0.4)
    if markers:
        M = np.asarray(markers, float).reshape(-1, 2)
        fig.add_trace(go.Scatter(x=M[:, 0], y=M[:, 1], mode="markers", name="operating point", marker=dict(symbol="diamond", size=10, color=ACCENT, line=dict(color=SURFACE, width=1)),
                                 hovertemplate="operating point<br>control %{x:.3f}<br>%{y:.3f} GHz<extra></extra>"))
    _style(fig, 520, title=title, margin=dict(l=62, r=14, t=40 if title else 14, b=54))
    fig.update_xaxes(title_text=xlabel, range=[float(x[0]), float(x[-1])], showgrid=False)
    fig.update_yaxes(title_text="drive frequency (GHz)", range=[float(f[0]), float(f[-1])], showgrid=False)
    return fig


def hypothesis_bars(hypotheses) -> go.Figure:
    """Horizontal bars of track_score per hypothesis, best first; the winner in amber, the rest in blue."""
    hs = sorted(hypotheses, key=lambda h: h["track_score"], reverse=True)
    labels = [h["hypothesis"] for h in hs]
    vals = np.array([float(h["track_score"]) for h in hs])
    colors = [ACCENT] + [BLUE] * (len(hs) - 1)
    fig = go.Figure(go.Bar(x=vals, y=labels, orientation="h", marker=dict(color=colors, line=dict(width=0)),
                           text=[f"{v:.3f}" for v in vals], textposition="outside", cliponaxis=False,
                           textfont=dict(size=12, color=TEXT), hovertemplate="%{y}<br>score %{x:.4f}<extra></extra>"))
    lo, hi = min(0.0, float(vals.min())), max(0.0, float(vals.max()))
    pad = 0.2 * ((hi - lo) or 1.0)
    _style(fig, max(110, 36 * len(hs) + 30), legend=False, margin=dict(l=8, r=56, t=10, b=14))
    fig.update_layout(bargap=0.55)
    fig.update_xaxes(showgrid=False, showticklabels=False, showline=False, ticks="", zeroline=True, zerolinecolor=GRID, zerolinewidth=1,
                     range=[lo - (pad if lo < 0 else 0), hi + pad])
    fig.update_yaxes(showgrid=False, showline=False, ticks="", autorange="reversed", automargin=True, tickfont=dict(size=12, color=TEXT))
    return fig


def convergence_figure(history) -> go.Figure | None:
    """Sparkline of the search loss; `history` is a list of (iteration, loss)."""
    if not history:
        return None
    it = [int(h[0]) for h in history]
    loss = [float(h[1]) for h in history]
    fig = go.Figure(go.Scatter(x=it, y=loss, mode="lines", line=dict(color=BLUE, width=2), hovertemplate="iteration %{x}<br>loss %{y:.4f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[it[-1]], y=[loss[-1]], mode="markers+text", text=[f"{loss[-1]:.3f}"], textposition="middle right",
                             marker=dict(color=BLUE, size=8, line=dict(color=SURFACE, width=2)), textfont=dict(size=12, color=TEXT), hoverinfo="skip"))
    _style(fig, 160, legend=False, margin=dict(l=10, r=58, t=10, b=28))
    fig.update_xaxes(showgrid=False, tickfont=dict(size=12, color=MUTED), nticks=6)
    fig.update_yaxes(showgrid=False, showticklabels=False, showline=False, ticks="", rangemode="tozero")
    return fig


def _segments(phi: np.ndarray, U: np.ndarray, E: float):
    """Classical turning points: the outermost phi where U <= E (None when the level is below the potential)."""
    m = U <= E
    return (float(phi[m].min()), float(phi[m].max())) if m.any() else None


def _label_groups(levels: np.ndarray, span: float, tol: float = 0.1) -> list[tuple[float, str]]:
    """Merge the labels of near-degenerate levels ('E0, E1') so they never overprint."""
    out, group = [], [0]
    for i in range(1, len(levels)):
        if levels[i] - levels[group[-1]] < tol * span:
            group.append(i)
        else:
            out.append(group)
            group = [i]
    out.append(group)
    return [(float(np.mean(levels[g])), ", ".join(f"E{i}" for i in g)) for g in out]


def _draw_well(fig, phi, U, levels, row=None, col=None, xmax=None):
    kw = dict(row=row, col=col) if row else {}
    fig.add_trace(go.Scatter(x=phi, y=U, mode="lines", line=dict(color=MUTED, width=2), name="U(φ)", showlegend=False,
                             hovertemplate="φ = %{x:.2f}<br>U = %{y:.2f} GHz<extra></extra>"), **kw)
    for n, E in enumerate(levels):
        seg = _segments(phi, U, E)
        if seg is None:
            continue
        fig.add_trace(go.Scatter(x=list(seg), y=[E, E], mode="lines", line=dict(color=BLUE, width=2), name=f"E{n}", showlegend=False,
                                 hovertemplate=f"E{n} = {E:.3f} GHz<extra></extra>"), **kw)
    span = float(levels.max() - U.min()) or 1.0
    for y, text in _label_groups(np.asarray(levels, float), span):
        fig.add_annotation(x=xmax if xmax is not None else float(phi.max()), y=y, text=text, showarrow=False, xanchor="right", yanchor="middle",
                           xshift=-4, font=dict(size=12, color=LIGHT_BLUE), **kw)


def _ticks(fig, lo: float, hi: float, **kw):
    vals = [v for v in PI_TICKS if lo - 1e-9 <= v <= hi + 1e-9 and PI_TICKS[v]]
    fig.update_xaxes(tickvals=vals, ticktext=[PI_TICKS[v] for v in vals], range=[lo, hi], **kw)


def potential_figure(family: str, params: dict, coupled: bool = False) -> go.Figure | None:
    """Potential U(φ) of the identified circuit with its lowest four levels drawn between the classical turning points."""
    try:
        from circuitsleuth import models

        tail = f"  ·  readout resonator fr = {params['fr']:.3g} GHz" if coupled and "fr" in params else ""
        if family in ("transmon", "tunable_transmon"):
            EJ = float(params["EJmax"] if family == "tunable_transmon" else params["EJ"])
            EC = float(params["EC"])
            phi = np.linspace(-np.pi, np.pi, 400)
            U = -EJ * np.cos(phi)
            levels = np.asarray(models.qubit_energies(family, params, np.array([0.0]), nlev=4))[0]
            fig = go.Figure()
            _draw_well(fig, phi, U, levels)
            name = "EJmax (at Φ = 0)" if family == "tunable_transmon" else "EJ"
            _style(fig, 320, title=f"U(φ) = −EJ cos φ  ·  {name} = {EJ:.3g} GHz, EC = {EC:.3g} GHz{tail}", legend=False,
                   margin=dict(l=62, r=16, t=40, b=48))
            _ticks(fig, -np.pi, np.pi, title_text="phase φ")
            top = max(EJ, float(levels.max()) + 0.1 * EJ)
            fig.update_yaxes(title_text="energy (GHz)", range=[-1.08 * EJ, 1.08 * top])
            return fig
        if family == "fluxonium":
            EJ, EC, EL = float(params["EJ"]), float(params["EC"]), float(params["EL"])
            phi = np.linspace(-3 * np.pi, 3 * np.pi, 700)
            fig = make_subplots(rows=1, cols=2, shared_yaxes=True, subplot_titles=("Φ = 0", "Φ = 0.5"), horizontal_spacing=0.05)
            fig.update_annotations(font=dict(size=13, color=MUTED))
            Umin, Emax = np.inf, -np.inf
            for col, Phi in ((1, 0.0), (2, 0.5)):
                U = 0.5 * EL * phi**2 - EJ * np.cos(phi - 2 * np.pi * Phi)
                levels = np.asarray(models.qubit_energies("fluxonium", params, np.array([Phi]), nlev=4))[0]
                _draw_well(fig, phi, U, levels, row=1, col=col, xmax=3 * np.pi)
                Umin, Emax = min(Umin, float(U.min())), max(Emax, float(levels.max()))
            _style(fig, 320, title=f"U(φ) = ½ EL φ² − EJ cos(φ − 2πΦ)  ·  EJ = {EJ:.3g}, EC = {EC:.3g}, EL = {EL:.3g} GHz{tail}", legend=False,
                   margin=dict(l=62, r=16, t=64, b=48))
            for col in (1, 2):
                _ticks(fig, -3 * np.pi, 3 * np.pi, row=1, col=col, title_text="phase φ")
            R = (Emax - Umin) or 1.0
            fig.update_yaxes(range=[Umin - 0.1 * R, Emax + 0.55 * R])
            fig.update_yaxes(title_text="energy (GHz)", row=1, col=1)
            return fig
    except Exception:
        return None
    return None


def levels_vs_control_figure(x_grid, lines) -> go.Figure | None:
    """Predicted transition frequencies against the control axis, without the data underneath."""
    if not lines or x_grid is None:
        return None
    xg = np.asarray(x_grid, float)
    fig = go.Figure()
    for name, vals in lines.items():
        v = _finite(vals)
        if np.isfinite(v).any():
            fig.add_trace(go.Scatter(x=xg, y=v, mode="lines", name=name, line=LINE_STYLE.get(name, OTHER_LINE), connectgaps=False,
                                     hovertemplate=f"{name}: %{{y:.3f}} GHz<extra></extra>"))
    if not fig.data:
        return None
    _style(fig, 260, margin=dict(l=62, r=14, t=14, b=50))
    fig.update_xaxes(title_text="control (flux bias, arbitrary units)")
    fig.update_yaxes(title_text="transition frequency (GHz)")
    return fig
