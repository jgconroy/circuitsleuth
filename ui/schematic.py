"""Hand-drawn inline SVG circuit schematics.

All drawings share one geometry: two horizontal rails (top y = 25, bottom y = 105) with the circuit elements
as vertical branches between them.  Stroke #f0efec 2 px, no fills except the amber Josephson-junction box.
"""
from __future__ import annotations

STROKE = "#f0efec"
ACCENT = "#fab219"
MUTED = "#c3c2b7"
INK = "#111214"
FONT = "Inter, 'Helvetica Neue', Helvetica, Arial, sans-serif"
TOP, BOT = 25, 105  # rail heights

_MARKER = ('<defs><marker id="cs-ah" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
           f'<path d="M0,0 L8,4 L0,8 z" fill="{STROKE}" stroke="none"/></marker>'
           '<marker id="cs-ah-amber" markerWidth="8" markerHeight="8" refX="7" refY="4" orient="auto">'
           f'<path d="M0,0 L8,4 L0,8 z" fill="{ACCENT}" stroke="none"/></marker></defs>')


def _svg(body: str, w: int, h: int = 130) -> str:
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="100%" '
            f'style="max-height:140px;display:block;margin:.2rem auto .4rem" fill="none" stroke="{STROKE}" stroke-width="2" '
            f'stroke-linecap="round" stroke-linejoin="round" font-family="{FONT}" font-size="13">{_MARKER}{body}</svg>')


def _wire(x1, y1, x2, y2, color=STROKE) -> str:
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"/>'


def _text(x, y, s, anchor="middle", color=MUTED, size=13) -> str:
    return f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" stroke="none" font-size="{size}">{s}</text>'


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


def _ind_v(x, yt=TOP, yb=BOT, n=5, amp=7, h=40) -> str:
    """Inductor as a zigzag on a vertical branch."""
    ym = (yt + yb) / 2
    y0, step = ym - h / 2, h / (2 * n)
    pts = [(x, y0)] + [(x + (amp if i % 2 == 0 else -amp), y0 + (i + 0.5) * step) for i in range(2 * n)] + [(x, y0 + h)]
    poly = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    return _wire(x, yt, x, y0) + f'<polyline points="{poly}"/>' + _wire(x, y0 + h, x, yb)


def _ground(x, y=BOT) -> str:
    return _wire(x, y, x, y + 7) + _wire(x - 10, y + 7, x + 10, y + 7) + _wire(x - 6, y + 12, x + 6, y + 12) + _wire(x - 2, y + 17, x + 2, y + 17)


def _flux(x, y, label="Φ") -> str:
    """Flux threading a loop: an upward arrow with its label to the left."""
    return f'<line x1="{x}" y1="{y + 15}" x2="{x}" y2="{y - 15}" marker-end="url(#cs-ah)"/>' + _text(x - 7, y + 5, label, "end", STROKE, 14)


def _rails(xl, xr) -> str:
    return _wire(xl, TOP, xr, TOP) + _wire(xl, BOT, xr, BOT)


def _resonator(x1: int) -> tuple[str, int]:
    """Coupling capacitor from the qubit's top node at x1 to an LC resonator on the right. Returns (svg, right edge)."""
    xc0, xc1 = x1, x1 + 60  # coupling capacitor between the two top nodes
    xl, xr = xc1, xc1 + 110  # resonator rails
    body = (_cap_h(TOP, xc0, xc1) + _text((xc0 + xc1) / 2, TOP - 12, "g")
            + _rails(xl, xr) + _wire(x1, BOT, xl, BOT)
            + _ind_v(xl + 30) + _cap_v(xr - 30)
            + _text((xl + xr) / 2, BOT + 20, "fr"))
    return body, xr + 20


def _qubit(family: str) -> tuple[str, int, int]:
    """The bare circuit between two rails. Returns (svg, left rail x, right rail x)."""
    if family == "transmon":
        xl, xr = 70, 230
        body = _rails(xl, xr) + _cap_v(110) + _jj_v(190) + _ground(150) + _text(86, 69, "EC", "end") + _text(212, 69, "EJ", "start")
        return body, xl, xr
    if family == "tunable_transmon":
        xl, xr = 50, 250
        # SQUID loop: two junction branches joined by short wires, hung from the rails at x = 200
        body = (_rails(xl, xr) + _cap_v(85) + _text(61, 69, "EC", "end") + _ground(135)
                + _wire(200, TOP, 200, 42) + _wire(165, 42, 235, 42) + _wire(165, 88, 235, 88) + _wire(200, 88, 200, BOT)
                + _jj_v(165, 42, 88, 20) + _jj_v(235, 42, 88, 20) + _flux(206, 65)
                + _text(200, BOT + 20, "EJmax, d"))
        return body, xl, xr
    if family == "fluxonium":
        xl, xr = 40, 260
        body = (_rails(xl, xr) + _cap_v(75) + _jj_v(150) + _ind_v(225) + _ground(112) + _flux(190, 65)
                + _text(75, BOT + 20, "EC") + _text(150, BOT + 20, "EJ") + _text(225, BOT + 20, "EL"))
        return body, xl, xr
    raise KeyError(family)


def _nv_ensemble() -> str:
    ox, oy = 140, 108
    body = (f'<circle cx="{ox}" cy="{oy}" r="3.5" fill="{STROKE}" stroke="none"/>'
            + f'<line x1="{ox}" y1="{oy}" x2="{ox}" y2="26" marker-end="url(#cs-ah)"/>'
            + _text(ox + 9, 40, "NV axis", "start", STROKE) + _text(ox + 9, 57, "D, E", "start")
            + f'<line x1="{ox}" y1="{oy}" x2="{ox + 76}" y2="46" stroke="{ACCENT}" marker-end="url(#cs-ah-amber)"/>'
            + _text(ox + 84, 44, "B", "start", ACCENT, 14)
            + f'<path d="M {ox},{oy - 30} A 30,30 0 0 1 {ox + 23},{oy - 19}" stroke-dasharray="3 3" stroke="{MUTED}"/>'
            + _text(ox + 17, oy - 34, "θ", "start"))
    return _svg(body, 300)


def _charge_qubit() -> str:
    body = (f'<circle cx="105" cy="80" r="22"/><circle cx="195" cy="80" r="22"/>'
            + f'<circle cx="105" cy="80" r="4" fill="{ACCENT}" stroke="none"/>'
            + '<rect x="142" y="62" width="16" height="36" rx="2"/>' + _text(150, BOT + 20, "t_c")
            + _wire(80, 36, 220, 36) + _wire(150, 36, 150, 16) + _text(158, 22, "ε", "start", STROKE, 14)
            + _text(105, BOT + 20, "dot L") + _text(195, BOT + 20, "dot R"))
    return _svg(body, 300)


def schematic_svg(family: str, coupled: bool = False) -> str:
    """Inline SVG for a circuit family; coupled=True appends a capacitively coupled LC readout resonator."""
    if family == "nv_ensemble":
        return _nv_ensemble()
    if family == "charge_qubit":
        return _charge_qubit()
    try:
        body, xl, xr = _qubit(family)
    except KeyError:
        return _svg(_text(150, 70, family, "middle", STROKE), 300)
    width = xr + 40
    if coupled:
        res, width = _resonator(xr)
        body += res
    return _svg(body, width)
