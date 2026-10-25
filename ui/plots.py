"""Plotly figures for the CircuitSleuth UI.

Every figure sits on the surface colour #1a1a19 with text #f0efec, hairline grid #2a2a28, IBM Plex Sans for words and
IBM Plex Mono for every number (tick labels, annotations).  Transition lines carry fixed colours per label so a line never
changes colour between hypotheses; legends sit outside the plotting area (below, horizontal).
"""
from __future__ import annotations

import re

import numpy as np
import plotly.graph_objects as go

from circuitsleuth import extract

SURFACE, TEXT, MUTED, GRID, INK = "#1a1a19", "#f0efec", "#c3c2b7", "#2a2a28", "#111214"
ACCENT, BLUE, LIGHT_BLUE, GREEN, RED = "#fab219", "#3987e5", "#86b6ef", "#4cae4c", "#d03b3b"
FONT = '"IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace'
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


def pretty(name: str) -> str:
    """Legend text for a transition key: '0-1' -> '0→1', '0-2/2ph' -> '0→2 two-photon', 'res' -> 'resonator'. Report keys are untouched."""
    if name == "res":
        return "resonator"
    s = re.sub(r"(\d)-(\d)", r"\1→\2", name)
    return s.replace("/2ph", " two-photon")


def line_color(name: str) -> str:
    return LINE_STYLE.get(name, OTHER_LINE)["color"]


def _style(fig: go.Figure, height: int, title: str | None = None, legend: bool = True, margin: dict | None = None, legend_below: bool = True) -> go.Figure:
    if legend_below:
        leg = dict(orientation="h", x=0, y=-0.14, xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)", borderwidth=0,
                   font=dict(size=12, color=TEXT), itemsizing="constant", tracegroupgap=2)
    else:
        leg = dict(x=0.985, y=0.975, xanchor="right", yanchor="top", bgcolor="rgba(26,26,25,0.62)", bordercolor=GRID, borderwidth=1,
                   font=dict(size=12, color=TEXT), itemsizing="constant", tracegroupgap=2)
    fig.update_layout(
        template="plotly_dark", paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, height=height,
        font=dict(family=FONT, size=13, color=TEXT),
        margin=margin or dict(l=60, r=16, t=40 if title else 16, b=52),
        legend=leg, showlegend=legend, hovermode="closest", uirevision="keep",
        hoverlabel=dict(bgcolor="#222221", bordercolor=GRID, font=dict(family=MONO, size=12, color=TEXT)),
    )
    if title:
        from ui.theme import title_case as _tc
        title = title if "=" in title else _tc(title)  # formula titles keep their symbols
        fig.update_layout(title=dict(text=title, x=0, xanchor="left", y=0.985, yanchor="top", font=dict(size=14, color=MUTED, family=FONT)))
    axis = dict(gridcolor=GRID, zeroline=False, linecolor=GRID, showline=True, ticks="outside", tickcolor=GRID,
                tickfont=dict(size=12, color=MUTED, family=MONO), title_font=dict(size=13, color=MUTED, family=FONT))
    fig.update_xaxes(**axis)
    fig.update_yaxes(**axis)
    return fig


def _finite(vals) -> np.ndarray:
    return np.array([np.nan if v is None else v for v in vals], float)


def map_figure(image, x, f, points=None, inlier=None, lines=None, x_grid=None, look_here=None, flat_lines=None, title=None, xlabel="flux bias (arbitrary units)",
               markers=None, height: int = 560, ylabel="drive frequency (GHz)") -> go.Figure:
    """Spectroscopy map with detected features, fitted transitions, numbered 'look here' boxes and flat spur lines.

    Explained features are filled green discs, unexplained ones hollow red rings. Each `look_here` box (x_lo, x_hi, f_lo, f_hi)
    is a hatched amber rectangle with a numbered badge 1.. at its top-left; the numbers match the recommendation cards.
    `markers` is an optional list of (control, frequency) operating points drawn as amber diamonds. The legend sits below the plot."""
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
                                     marker=dict(size=6, color=GREEN, symbol="circle", opacity=0.95, line=dict(width=0)), hovertemplate=pt_hover))
            fig.add_trace(go.Scatter(x=P[~inl, 0], y=P[~inl, 1], mode="markers", name=f"unexplained ({(~inl).sum()})",
                                     marker=dict(size=8, color="rgba(0,0,0,0)", symbol="circle-open", opacity=0.95, line=dict(width=1.5, color=RED)), hovertemplate=pt_hover))
    xg = np.asarray(x_grid, float) if x_grid is not None else None
    if lines and xg is not None:
        for name, vals in lines.items():
            v = _finite(vals)
            if not np.isfinite(v).any():
                continue
            fig.add_trace(go.Scatter(x=xg, y=v, mode="lines", name=pretty(name), line=LINE_STYLE.get(name, OTHER_LINE), connectgaps=False,
                                     hovertemplate=f"{pretty(name)}: %{{y:.3f}} GHz at %{{x:.3f}}<extra></extra>"))
    band = float(f[-1]) - float(f[0])
    dx = float(np.median(np.diff(x))) if len(x) > 1 else 0.0
    for i, b in enumerate(look_here or [], start=1):
        x_lo, x_hi = float(b["x_lo"]) - 0.5 * dx, float(b["x_hi"]) + 0.5 * dx
        pad = 0.012 * band
        y_lo, y_hi = max(float(b["f_lo"]) - pad, float(f[0])), min(float(b["f_hi"]) + pad, float(f[-1]))
        fig.add_trace(go.Scatter(x=[x_lo, x_hi, x_hi, x_lo, x_lo], y=[y_lo, y_lo, y_hi, y_hi, y_lo], mode="lines", fill="toself",
                                 fillpattern=dict(shape="/", fgcolor=ACCENT, bgcolor="rgba(250,178,25,0.05)", size=8, solidity=0.14),
                                 line=dict(color=ACCENT, width=1, dash="dot"), opacity=0.8, name="look here", legendgroup="look", showlegend=(i == 1),
                                 hovertemplate=f"{i}: look here, {pretty(b.get('line', ''))}<br>%{{x:.3f}}, %{{y:.3f}} GHz<extra></extra>"))
        fig.add_annotation(x=x_lo, y=y_hi, text=f"<b>{i}</b>", showarrow=False, xanchor="left", yanchor="top", xshift=3, yshift=-3,
                           font=dict(size=12, color=INK, family=MONO), bgcolor=ACCENT, bordercolor=ACCENT, borderpad=3, borderwidth=1)
    for fl in flat_lines or []:
        fig.add_hline(y=float(fl), line=dict(color=RED, width=1), opacity=0.4)
    if markers:
        M = np.asarray(markers, float).reshape(-1, 2)
        fig.add_trace(go.Scatter(x=M[:, 0], y=M[:, 1], mode="markers", name="operating point", marker=dict(symbol="diamond", size=10, color=ACCENT, line=dict(color=SURFACE, width=1)),
                                 hovertemplate="operating point<br>control %{x:.3f}<br>%{y:.3f} GHz<extra></extra>"))
    _style(fig, height, title=title, margin=dict(l=62, r=14, t=40 if title else 14, b=92))
    fig.update_xaxes(title_text=xlabel, range=[float(x[0]), float(x[-1])], showgrid=False)
    fig.update_yaxes(title_text=ylabel, range=[float(f[0]), float(f[-1])], showgrid=False)
    return fig


# ------------------------------------------------------------------ potential well
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


def _draw_well(fig, phi, U, levels, xmax=None):
    fig.add_trace(go.Scatter(x=phi, y=U, mode="lines", line=dict(color=MUTED, width=2), name="U(φ)", showlegend=False,
                             hovertemplate="φ = %{x:.2f}<br>U = %{y:.2f} GHz<extra></extra>"))
    segs = []
    for n, E in enumerate(levels):
        seg = _segments(phi, U, E)
        segs.append(seg)
        if seg is None:
            continue
        fig.add_trace(go.Scatter(x=list(seg), y=[E, E], mode="lines", line=dict(color=BLUE, width=2), name=f"E<sub>{n}</sub>", showlegend=False,
                                 hovertemplate=f"E<sub>{n}</sub> = {E:.3f} GHz<extra></extra>"))
    span = float(levels.max() - U.min()) or 1.0
    for y, text in _label_groups(np.asarray(levels, float), span):
        fig.add_annotation(x=xmax if xmax is not None else float(phi.max()), y=y, text=re.sub(r"E(\d)", r"E<sub>\1</sub>", text), showarrow=False, xanchor="right", yanchor="middle",
                           xshift=-4, font=dict(size=12, color=LIGHT_BLUE, family=MONO))
    return segs


def _arrows(fig, levels, segs, phi_centre: float):
    """Transition arrows 0→1 and 1→2 (and 0→2 when it fits) between the levels, in the map's line colours, labelled in GHz."""
    E = np.asarray(levels, float)
    seg1 = segs[1] if len(segs) > 1 and segs[1] else (phi_centre - 1, phi_centre + 1)
    w = max(seg1[1] - seg1[0], 0.6)
    # (name, from, to, arrow x, side of the label): 0→2 sits left of 0→1 with its label on its own left, so labels never cross arrows
    specs = [("0-1", 0, 1, phi_centre - 0.12 * w, "left"), ("1-2", 1, 2, phi_centre + 0.28 * w, "left")]
    if len(E) > 3 and (E[2] - E[0]) < 0.9 * (E[3] - E[0]):
        specs.append(("0-2", 0, 2, phi_centre - 0.42 * w, "right"))
    for name, i, j, xa, side in specs:
        if j >= len(E):
            continue
        col = line_color(name)
        fig.add_annotation(x=xa, y=float(E[j]), ax=xa, ay=float(E[i]), xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=3, arrowsize=1.1, arrowwidth=1.5, arrowcolor=col, text="")
        fig.add_annotation(x=xa, y=0.5 * float(E[i] + E[j]), text=f"{pretty(name)}  {E[j] - E[i]:.3f} GHz", showarrow=False, yanchor="middle",
                           xanchor="left" if side == "left" else "right", xshift=6 if side == "left" else -6,
                           font=dict(size=12, color=col, family=MONO), bgcolor="rgba(26,26,25,0.75)", borderpad=2)


def _ticks(fig, lo: float, hi: float, **kw):
    vals = [v for v in PI_TICKS if lo - 1e-9 <= v <= hi + 1e-9 and PI_TICKS[v]]
    fig.update_xaxes(tickvals=vals, ticktext=[PI_TICKS[v] for v in vals], range=[lo, hi], **kw)


def potential_figure(family: str, params: dict, coupled: bool = False, phi_ext: float = 0.0) -> go.Figure | None:
    """Potential U(φ) of the identified circuit at external flux `phi_ext` (in Φ₀) with its lowest four levels drawn between the
    classical turning points and the 0→1 / 1→2 transition arrows labelled in GHz."""
    try:
        from circuitsleuth import models

        tail = f"  ·  f<sub>r</sub> = {params['fr']:.3g} GHz" if coupled and "fr" in params else ""
        if family in ("transmon", "tunable_transmon"):
            EC = float(params["EC"])
            if family == "tunable_transmon":
                EJ = float(models.tunable_EJ(params["EJmax"], params["d"], np.array([phi_ext]))[0])
                name = f"E<sub>J</sub>(Φ = {phi_ext:.2f}) = {EJ:.3g} GHz"
                levels = np.asarray(models.qubit_energies(family, params, np.array([phi_ext]), nlev=4))[0]
            else:
                EJ = float(params["EJ"])
                name = f"E<sub>J</sub> = {EJ:.3g} GHz"
                levels = np.asarray(models.qubit_energies(family, params, np.array([0.0]), nlev=4))[0]
            phi = np.linspace(-np.pi, np.pi, 400)
            U = -EJ * np.cos(phi)
            fig = go.Figure()
            segs = _draw_well(fig, phi, U, levels)
            _arrows(fig, levels, segs, 0.0)
            _style(fig, 360, title=f"U(φ) = −E<sub>J</sub> cos φ  ·  {name}, E<sub>C</sub> = {EC:.3g} GHz{tail}", legend=False, margin=dict(l=62, r=16, t=40, b=48))
            _ticks(fig, -np.pi, np.pi, title_text="phase φ")
            top = max(EJ, float(levels.max()) + 0.1 * EJ)
            fig.update_yaxes(title_text="energy (GHz)", range=[-1.08 * EJ, 1.08 * top])
            return fig
        if family == "fluxonium":
            EJ, EC, EL = float(params["EJ"]), float(params["EC"]), float(params["EL"])
            phi = np.linspace(-3 * np.pi, 3 * np.pi, 700)
            U = 0.5 * EL * phi**2 - EJ * np.cos(phi - 2 * np.pi * phi_ext)
            levels = np.asarray(models.qubit_energies("fluxonium", params, np.array([phi_ext]), nlev=4))[0]
            fig = go.Figure()
            segs = _draw_well(fig, phi, U, levels, xmax=3 * np.pi)
            _arrows(fig, levels, segs, 2 * np.pi * phi_ext if phi_ext < 0.45 else 0.0)
            _style(fig, 360, title=f"U(φ) = ½ E<sub>L</sub> φ² − E<sub>J</sub> cos(φ − 2πΦ) at Φ = {phi_ext:.2f}  ·  E<sub>J</sub> = {EJ:.3g}, E<sub>C</sub> = {EC:.3g}, E<sub>L</sub> = {EL:.3g} GHz{tail}", legend=False,
                   margin=dict(l=62, r=16, t=40, b=48))
            _ticks(fig, -3 * np.pi, 3 * np.pi, title_text="phase φ")
            Umin, Emax = float(U.min()), float(levels.max())
            R = (Emax - Umin) or 1.0
            fig.update_yaxes(title_text="energy (GHz)", range=[Umin - 0.1 * R, Emax + 0.55 * R])
            return fig
    except Exception:
        return None
    return None
