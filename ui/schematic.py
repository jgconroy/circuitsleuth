"""Hand-drawn inline SVG circuit schematics with the fitted values written next to each component.

All circuit drawings share one geometry: two horizontal rails (top y = 44, bottom y = 124) with the elements as vertical
branches between them, a label row under the bottom rail and a value row under the labels.  Strokes are 1.5 px on screen
whatever the scale (vector-effect: non-scaling-stroke), no fills except the amber Josephson-junction box.  Fitted values
fade in one after the other (150 ms stagger) through a <style> block inside the SVG.
"""
from __future__ import annotations

import html

STROKE = "#f0efec"
ACCENT = "#fab219"
MUTED = "#c3c2b7"
INK = "#111214"
MONO = "'IBM Plex Mono', ui-monospace, Menlo, monospace"
TOP, BOT = 44, 124  # rail heights
LAB, VAL = BOT + 20, BOT + 37  # label and value baselines
H = BOT + 48
TARGET_W = 440  # on-screen width the type sizes are tuned for

_MARKER = ('<defs><marker id="cs-ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">'
           f'<path d="M0,0 L8,4 L0,8 z" fill="{STROKE}" stroke="none"/></marker>'
           '<marker id="cs-ah-amber" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto" markerUnits="userSpaceOnUse">'
           f'<path d="M0,0 L8,4 L0,8 z" fill="{ACCENT}" stroke="none"/></marker></defs>')
_STYLE = ("<style>.cs-sch line,.cs-sch polyline,.cs-sch path,.cs-sch rect,.cs-sch circle{vector-effect:non-scaling-stroke}"
          ".cs-sch .cs-v{opacity:0;animation:cs-vin .45s ease-out forwards}"
          "@keyframes cs-vin{from{opacity:0}to{opacity:1}}</style>")


def _svg(body: str, w: int, h: int = H, max_width: int = TARGET_W) -> str:
    fs = 14.0 * w / max_width  # 14 px on screen when the drawing is shown at max_width
    return (f'<svg xmlns="http://www.w3.org/2000/svg" class="cs-sch" viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-width:{max_width}px;display:block;margin:0 auto;height:auto" fill="none" stroke="{STROKE}" stroke-width="1.5" '
            f'stroke-linecap="round" stroke-linejoin="round" font-family="{MONO}" font-size="{fs:.2f}">{_STYLE}{_MARKER}{body}</svg>')


def _wire(x1, y1, x2, y2, color=STROKE) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"/>'


def _sub(base: str, sub: str, rest: str = "") -> str:
    """base + subscript (+ rest) as SVG tspans (dy shifts, no baseline-shift dependence)."""
    return f'{html.escape(base)}<tspan dy="4" font-size="75%">{html.escape(sub)}</tspan>' + (f'<tspan dy="-4">{html.escape(rest)}</tspan>' if rest else "")


LABEL_SVG = {  # component labels typeset for the label row
    "EC": _sub("E", "C"), "EJ": _sub("E", "J"), "EL": _sub("E", "L"), "EJmax · d": _sub("E", "J,max", " · d"), "fr  (L‖C)": _sub("f", "r", "  (L‖C)"),
    "t_c": _sub("t", "c"), "f_rad0": _sub("f", "r,0"), "√V scale": _sub("κ", "V"), "Δ_fs": _sub("Δ", "fs"), "ω_r": _sub("ω", "r"), "ω_z": _sub("ω", "z"),
}


def _text(x, y, s, anchor="middle", color=MUTED, size=None, cls="", delay_ms: int | None = None) -> str:
    sz = f' font-size="{size}"' if size else ""
    c = f' class="{cls}"' if cls else ""
    d = f' style="animation-delay:{delay_ms}ms"' if delay_ms is not None else ""
    body = LABEL_SVG.get(str(s), html.escape(str(s)))
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" stroke="none"{sz}{c}{d}>{body}</text>'


def _cap_v(x, yt=TOP, yb=BOT, half=15, gap=8) -> str:
    """Capacitor on a vertical branch (two horizontal plates)."""
    ym = (yt + yb) / 2
    return (_wire(x, yt, x, ym - gap / 2) + _wire(x - half, ym - gap / 2, x + half, ym - gap / 2)
            + _wire(x - half, ym + gap / 2, x + half, ym + gap / 2) + _wire(x, ym + gap / 2, x, yb))


def _cap_h(y, xl, xr, half=11, gap=8) -> str:
    """Capacitor on a horizontal wire (two vertical plates)."""
    xm = (xl + xr) / 2
    return (_wire(xl, y, xm - gap / 2, y) + _wire(xm - gap / 2, y - half, xm - gap / 2, y + half)
            + _wire(xm + gap / 2, y - half, xm + gap / 2, y + half) + _wire(xm + gap / 2, y, xr, y))


def _jj_v(x, yt=TOP, yb=BOT, s=26) -> str:
    """Josephson junction: amber box with a cross, on a vertical branch."""
    ym = (yt + yb) / 2
    x0, y0 = x - s / 2, ym - s / 2
    return (_wire(x, yt, x, y0)
            + f'<rect x="{x0}" y="{y0}" width="{s}" height="{s}" rx="2" fill="{ACCENT}" stroke="{ACCENT}"/>'
            + _wire(x0 + 4, y0 + 4, x0 + s - 4, y0 + s - 4, INK) + _wire(x0 + s - 4, y0 + 4, x0 + 4, y0 + s - 4, INK)
            + _wire(x, y0 + s, x, yb))


def _ind_v(x, yt=TOP, yb=BOT, n=4, h=40) -> str:
    """Inductor as a run of coil arcs (not a zigzag, which reads as a resistor) on a vertical branch."""
    ym = (yt + yb) / 2
    y0, r = ym - h / 2, h / (2 * n)
    arcs = "".join(f" a{r:.1f},{r:.1f} 0 0 1 0,{2 * r:.1f}" for _ in range(n))
    return _wire(x, yt, x, y0) + f'<path d="M {x},{y0}{arcs}"/>' + _wire(x, y0 + h, x, yb)


def _ground(x, y=BOT) -> str:
    return _wire(x, y, x, y + 6) + _wire(x - 9, y + 6, x + 9, y + 6) + _wire(x - 5.5, y + 10, x + 5.5, y + 10) + _wire(x - 2, y + 14, x + 2, y + 14)


def _flux(x, y, label="Φ") -> str:
    """Flux threading a loop: an upward arrow with its label to the left."""
    return f'<line x1="{x}" y1="{y + 15}" x2="{x}" y2="{y - 15}" marker-end="url(#cs-ah)"/>' + _text(x - 7, y + 4, label, "end", STROKE)


def _rails(xl, xr) -> str:
    return _wire(xl, TOP, xr, TOP) + _wire(xl, BOT, xr, BOT)


def _labelled(x, label, key, values, order, anchor="middle") -> str:
    """Component label on the label row with its fitted value (if known) fading in on the value row."""
    s = _text(x, LAB, label, anchor)
    v = (values or {}).get(key)
    if v:
        order.append(key)
        s += _text(x, VAL, v, anchor, ACCENT, cls="cs-v", delay_ms=150 * len(order))  # staggered fade-in, 150 ms apart
    return s


def _resonator(x1: int, values, order) -> tuple[str, int]:
    """Coupling capacitor from the qubit's top node at x1 to an L‖C readout resonator on the right. Returns (svg, right edge)."""
    xc0, xc1 = x1, x1 + 60  # coupling capacitor between the two top nodes
    xl, xr = xc1, xc1 + 110  # resonator rails
    xm = (xc0 + xc1) / 2
    g_val = (values or {}).get("g")
    g_txt = f'<text x="{xm}" y="{TOP - 16}" text-anchor="middle" fill="{MUTED}" stroke="none">g'
    if g_val:
        order.append("g")
        g_txt += f' <tspan class="cs-v" style="animation-delay:{150 * len(order)}ms" fill="{ACCENT}">{html.escape(g_val)}</tspan>'
    g_txt += "</text>"
    body = (_cap_h(TOP, xc0, xc1) + g_txt
            + _rails(xl, xr) + _wire(x1, BOT, xl, BOT)
            + _ind_v(xl + 30) + _cap_v(xr - 30)
            + _labelled((xl + xr) / 2, "fr  (L‖C)", "fr", values, order))
    return body, xr + 20


def _qubit(family: str, values, order) -> tuple[str, int, int]:
    """The bare circuit between two rails. Returns (svg, left rail x, right rail x)."""
    if family == "transmon":
        xl, xr = 60, 240
        body = (_rails(xl, xr) + _cap_v(105) + _jj_v(195) + _ground(150)
                + _labelled(105, "EC", "EC", values, order) + _labelled(195, "EJ", "EJ", values, order))
        return body, xl, xr
    if family == "tunable_transmon":
        xl, xr = 40, 260
        # SQUID loop: two junction branches joined by short wires, hung from the rails at x = 200
        body = (_rails(xl, xr) + _cap_v(80) + _ground(135)
                + _wire(200, TOP, 200, TOP + 17) + _wire(165, TOP + 17, 235, TOP + 17) + _wire(165, BOT - 17, 235, BOT - 17) + _wire(200, BOT - 17, 200, BOT)
                + _jj_v(165, TOP + 17, BOT - 17, 20) + _jj_v(235, TOP + 17, BOT - 17, 20) + _flux(206, (TOP + BOT) / 2)
                + _labelled(80, "EC", "EC", values, order) + _labelled(200, "EJmax · d", "EJmax", values, order))
        return body, xl, xr
    if family == "fluxonium":
        xl, xr = 30, 270
        body = (_rails(xl, xr) + _cap_v(70) + _jj_v(150) + _ind_v(230) + _ground(110) + _flux(190, (TOP + BOT) / 2)
                + _labelled(70, "EC", "EC", values, order) + _labelled(150, "EJ", "EJ", values, order) + _labelled(230, "EL", "EL", values, order))
        return body, xl, xr
    raise KeyError(family)


def _nv_ensemble(values, order) -> str:
    ox, oy = 130, BOT
    body = (f'<circle cx="{ox}" cy="{oy}" r="3.5" fill="{STROKE}" stroke="none"/>'
            + f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="{TOP}" marker-end="url(#cs-ah)"/>'
            + _text(ox + 9, TOP + 12, "NV axis", "start", STROKE)
            + f'<line x1="{ox}" y1="{oy}" x2="{ox + 80}" y2="{TOP + 22}" stroke="{ACCENT}" marker-end="url(#cs-ah-amber)"/>'
            + _text(ox + 88, TOP + 20, "B", "start", ACCENT)
            + f'<path d="M {ox},{oy - 30} A 30,30 0 0 1 {ox + 23},{oy - 19}" stroke-dasharray="3 3" stroke="{MUTED}"/>'
            + _text(ox + 20, oy - 34, "θ", "start")
            + _labelled(70, "D", "D", values, order) + _labelled(160, "E", "E", values, order) + _labelled(250, "θ, φ", "theta", values, order))
    return _svg(body, 320)


def _charge_qubit(values, order) -> str:
    yc = (TOP + BOT) / 2 + 6
    body = (f'<circle cx="105" cy="{yc}" r="22"/><circle cx="195" cy="{yc}" r="22"/>'
            + f'<circle cx="105" cy="{yc}" r="4" fill="{ACCENT}" stroke="none"/>'
            + f'<rect x="142" y="{yc - 18}" width="16" height="36" rx="2"/>'
            + _wire(80, TOP, 220, TOP) + _wire(150, TOP, 150, TOP - 16) + _text(158, TOP - 6, "ε", "start", STROKE)
            + _labelled(105, "dot L", "", values, order) + _labelled(195, "dot R", "", values, order) + _labelled(150, "t_c", "t_c", values, order))
    return _svg(body, 300)


def _trapped_ion(values, order) -> str:
    yc = (TOP + BOT) / 2 + 4
    body = (_wire(70, yc - 34, 230, yc - 34) + _wire(70, yc + 34, 230, yc + 34)  # RF rods
            + f'<rect x="56" y="{yc - 16}" width="14" height="32" rx="2"/><rect x="230" y="{yc - 16}" width="14" height="32" rx="2"/>'  # endcaps
            + f'<circle cx="150" cy="{yc}" r="5" fill="{ACCENT}" stroke="none"/>'
            + f'<line x1="150" y1="{yc}" x2="196" y2="{yc}" stroke="{ACCENT}" marker-end="url(#cs-ah-amber)"/>' + _text(200, yc + 4, "ω_z", "start", ACCENT)
            + f'<line x1="150" y1="{yc}" x2="150" y2="{yc - 26}" stroke="{MUTED}" marker-end="url(#cs-ah)"/>' + _text(156, yc - 20, "ω_r", "start", MUTED)
            + _text(63, yc + 50, "V", "middle", STROKE) + _text(237, yc + 50, "V", "middle", STROKE)
            + _labelled(95, "f_rad0", "f_rad0", values, order) + _labelled(205, "√V scale", "volt_scale", values, order))
    return _svg(body, 310)


def _rydberg_atom(values, order) -> str:
    yc = (TOP + BOT) / 2 + 4
    body = (_wire(80, yc - 40, 220, yc - 40) + _wire(80, yc + 40, 220, yc + 40)  # field plates
            + f'<circle cx="150" cy="{yc}" r="22" stroke-dasharray="4 3" stroke="{MUTED}"/><circle cx="150" cy="{yc}" r="4" fill="{ACCENT}" stroke="none"/>'
            + f'<circle cx="172" cy="{yc}" r="3" fill="{STROKE}" stroke="none"/>'
            + f'<line x1="240" y1="{yc - 34}" x2="240" y2="{yc + 30}" stroke="{ACCENT}" marker-end="url(#cs-ah-amber)"/>' + _text(246, yc + 2, "E", "start", ACCENT)
            + _text(150, yc - 48, "+V", "middle", STROKE) + _text(150, yc + 56, "−V", "middle", STROKE)
            + _labelled(100, "α", "alpha1", values, order) + _labelled(200, "Δ_fs", "dfs", values, order))
    return _svg(body, 300)


def schematic_svg(family: str, coupled: bool = False, values: dict | None = None) -> str:
    """Inline SVG for a circuit family; coupled=True appends a capacitively coupled L‖C readout resonator.
    `values` maps parameter keys (EJ, EC, EL, EJmax, d, fr, g, D, E, theta, t_c) to display strings written next to the symbols."""
    values = dict(values or {})
    if "EJmax" in values and "d" in values:
        values["EJmax"] = f"{values['EJmax']} · {values['d']}"
    order: list[str] = []
    if family == "nv_ensemble":
        svg = _nv_ensemble(values, order)
    elif family == "charge_qubit":
        svg = _charge_qubit(values, order)
    elif family == "trapped_ion":
        svg = _trapped_ion(values, order)
    elif family == "rydberg_atom":
        svg = _rydberg_atom(values, order)
    else:
        try:
            body, xl, xr = _qubit(family, values, order)
        except KeyError:
            return _svg(_text(150, (TOP + BOT) / 2, family, "middle", STROKE), 300)
        width = xr + 30
        if coupled:
            res, width = _resonator(xr, values, order)
            body += res
        svg = _svg(body, width)
    return svg
