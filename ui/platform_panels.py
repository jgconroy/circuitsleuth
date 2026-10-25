"""Physics panels for the non-circuit platforms (Plotly). Kept separate from ui/plots.py."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from circuitsleuth import models

SURFACE = "#1a1a19"
TEXT = "#f0efec"
MUTED = "#c3c2b7"
GRID = "#2a2a28"
BLUE = "#3987e5"
AMBER = "#fab219"
ORANGE = "#d95926"


FONT = '"IBM Plex Sans", "Helvetica Neue", Arial, sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, Menlo, monospace'


def _layout(fig, title, xlab, ylab, height=360):
    """Surface colours, Plex Sans for words, Plex Mono for every number, legend below the plot."""
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=MUTED, family=FONT), x=0.0, xanchor="left"),
        paper_bgcolor=SURFACE, plot_bgcolor=SURFACE, font=dict(family=FONT, color=TEXT, size=12),
        margin=dict(l=56, r=16, t=40, b=84), height=height,
        xaxis=dict(title=xlab, gridcolor=GRID, zerolinecolor=GRID, color=MUTED, tickfont=dict(family=MONO, size=12)),
        yaxis=dict(title=ylab, gridcolor=GRID, zerolinecolor=GRID, color=MUTED, tickfont=dict(family=MONO, size=12)),
        legend=dict(orientation="h", x=0, y=-0.16, xanchor="left", yanchor="top", bgcolor="rgba(0,0,0,0)", borderwidth=0, font=dict(size=12)),
        hoverlabel=dict(bgcolor="#222221", bordercolor=GRID, font=dict(family=MONO, size=12, color=TEXT)),
    )
    return fig


def nv_levels_figure(params: dict, B_max: float = 300.0):
    """S = 1 energy levels of the NV orientation most aligned with the field, versus field magnitude,
    with the two ODMR transitions marked at B_max / 2."""
    b = np.array([np.sin(params["theta"]) * np.cos(params["phi"]), np.sin(params["theta"]) * np.sin(params["phi"]), np.cos(params["theta"])])
    proj = models.NV_AXES @ b
    k = int(np.argmax(np.abs(proj)))
    bz, bperp = abs(float(proj[k])), float(np.sqrt(max(1 - proj[k] ** 2, 0)))
    B = np.linspace(0, B_max, 200)
    H0 = params["D"] * (models._SZ @ models._SZ) + params["E"] * (models._SX @ models._SX - models._SY @ models._SY)
    Hz = models.GAMMA_E * (bz * models._SZ + bperp * models._SX)
    ev = np.linalg.eigvalsh(H0[None] + B[:, None, None] * Hz[None])
    fig = go.Figure()
    names = ["m_s = 0", "m_s = -1", "m_s = +1"]
    cols = [MUTED, BLUE, ORANGE]
    for i in range(3):
        fig.add_trace(go.Scatter(x=B, y=ev[:, i], mode="lines", name=names[i], line=dict(color=cols[i], width=2), hovertemplate="B = %{x:.0f} G<br>E = %{y:.4f} GHz<extra></extra>"))
    j = len(B) // 2
    for i, col in ((1, BLUE), (2, ORANGE)):
        fig.add_trace(go.Scatter(x=[B[j], B[j]], y=[ev[j, 0], ev[j, i]], mode="lines", line=dict(color=AMBER, width=1.5, dash="dot"), showlegend=False, hoverinfo="skip"))
        fig.add_annotation(x=B[j], y=0.5 * (ev[j, 0] + ev[j, i]), text=f"{ev[j, i] - ev[j, 0]:.3f} GHz", showarrow=False, font=dict(color=AMBER, size=11, family=MONO), xshift=46)
    return _layout(fig, f"NV orientation {k + 1} (most aligned, B·n = {proj[k]:+.2f}): spin-1 levels vs field", "field magnitude (G)", "energy (GHz)")


def dqd_figure(params: dict, eps_max: float | None = None):
    """Charge-qubit eigenenergies versus detuning with the 2 t_c anticrossing marked."""
    t = params["t_c"]
    eps_max = eps_max or 6 * t
    eps = np.linspace(-eps_max, eps_max, 300)
    E = 0.5 * np.sqrt(eps**2 + 4 * t**2)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eps, y=0.5 * eps, mode="lines", name="bare (left dot)", line=dict(color=GRID, width=1, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=eps, y=-0.5 * eps, mode="lines", name="bare (right dot)", line=dict(color=GRID, width=1, dash="dash"), hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=eps, y=E, mode="lines", name="excited", line=dict(color=ORANGE, width=2), hovertemplate="ε = %{x:.2f} GHz<br>E = %{y:.3f} GHz<extra></extra>"))
    fig.add_trace(go.Scatter(x=eps, y=-E, mode="lines", name="ground", line=dict(color=BLUE, width=2), hovertemplate="ε = %{x:.2f} GHz<br>E = %{y:.3f} GHz<extra></extra>"))
    fig.add_trace(go.Scatter(x=[0, 0], y=[-t, t], mode="lines", line=dict(color=AMBER, width=1.5, dash="dot"), showlegend=False, hoverinfo="skip"))
    fig.add_annotation(x=0, y=0, text=f"2 t_c = {2 * t:.3f} GHz", showarrow=False, font=dict(color=AMBER, size=12, family=MONO), xshift=70)
    return _layout(fig, "Double quantum dot: hybridised charge states vs detuning", "detuning ε (GHz)", "energy (GHz)")
