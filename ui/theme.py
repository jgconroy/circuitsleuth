"""One CSS block (IBM Plex Sans / IBM Plex Mono, 8 px rhythm, flat cards) and the HTML builders for the pieces Streamlit has
no widget for: the verdict band, the hypothesis race, the pipeline stepper, spec-sheet cards, numbered recommendation
cards, the truth strip and compact hairline tables.

Palette: background #111214, surface #1a1a19, hairline rgba(240,239,236,.08), text #f0efec, muted #c3c2b7, accent #fab219.
Type: display 44/500/-0.02em, section 20/600, eyebrow 11 uppercase .12em, body 15/24, mono 14, KPI 32 mono.
Every number is set in Plex Mono with tabular figures; ± and units are muted.
"""
from __future__ import annotations

import html
import re

import streamlit as st

SANS = '"IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace'
FONT = SANS  # kept for older imports

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap');
:root { --bg:#111214; --surface:#1a1a19; --line:rgba(240,239,236,.08); --line2:rgba(240,239,236,.16); --text:#f0efec; --muted:#c3c2b7;
  --amber:#fab219; --green:#4cae4c; --red:#d03b3b; --sans:"IBM Plex Sans","Helvetica Neue",Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace; }
/* ---- chrome: no toolbar, no running indicator, no toasts, no footer, no shadows */
#MainMenu, footer, [data-testid="stDecoration"], .stAppDeployButton, [data-testid="stToolbar"], [data-testid="stToolbarActions"],
[data-testid="stStatusWidget"], [data-testid="stToast"], [data-testid="stToastContainer"], [data-testid="stAppToastContainer"],
div[class*="stToast"], [data-testid="stHeaderActionElements"] { display:none !important; }
header[data-testid="stHeader"] { background:transparent; box-shadow:none; height:2.25rem; }
.block-container { max-width:1200px; padding:24px 32px 64px; }
section[data-testid="stSidebar"] { width:280px !important; min-width:280px !important; max-width:280px !important; }
section[data-testid="stSidebar"] > div:first-child { width:280px !important; }
section[data-testid="stSidebar"] .block-container { padding-top:1.25rem; }
.stApp * { box-shadow:none !important; }
/* ---- type */
.stApp, .stApp p, .stApp li, .stApp label, .stApp summary, .stApp button, .stApp [data-testid="stMarkdownContainer"],
.stApp [data-testid="stWidgetLabel"] p { font-family:var(--sans); }
.stApp [data-testid="stMarkdownContainer"] p, .stApp [data-testid="stMarkdownContainer"] li { font-size:15px; line-height:24px; }
.stApp h1 { font-family:var(--sans); font-size:44px; font-weight:500; letter-spacing:-0.02em; line-height:1.1; }
.stApp h2, .stApp h3 { font-family:var(--sans); font-size:20px; font-weight:600; letter-spacing:-0.01em; color:var(--text); padding:0 0 8px; margin:0; }
.stApp [data-testid="stCaptionContainer"] p { font-size:13px; line-height:20px; color:var(--muted); }
.stApp [data-testid="stExpander"] summary p { font-size:15px; font-weight:500; }
.stApp [data-testid="stExpander"] details, .stApp [data-testid="stExpander"] > div { border:1px solid var(--line) !important; border-radius:6px; background:var(--surface); }
.stApp [data-testid="stVerticalBlockBorderWrapper"] { border:1px solid var(--line) !important; border-radius:6px; background:var(--surface); }
.stApp [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] { padding:8px; }
.stApp code, .stApp pre, .stApp kbd, .stApp [data-testid="stCode"] *, .cs-mono, .cs-num,
.stApp [data-testid="stNumberInputContainer"] input, .stApp [data-testid="stSliderThumbValue"], .stApp [data-testid="stSliderTickBarMin"],
.stApp [data-testid="stSliderTickBarMax"], .stApp [data-testid="stNumberInput"] input, .stApp input[type="number"]
  { font-family:var(--mono) !important; font-variant-numeric:tabular-nums; }
.stApp [data-testid="stCode"] pre, .stApp [data-testid="stCode"] code { font-size:14px; line-height:22px; }
.stApp .stButton button, .stApp .stDownloadButton button { font-family:var(--sans); font-weight:500; border-radius:6px; }
.stApp [data-testid="stMarkdownContainer"] svg { display:block; }
/* ---- rhythm */
.cs-section { margin:64px 0 24px; padding:0; }
.cs-section .cs-h2 { font-family:var(--sans); font-size:20px; font-weight:600; letter-spacing:-0.01em; color:var(--text); margin:0; line-height:28px; }
.cs-section .cs-eyebrow { margin-bottom:4px; }
.cs-section .cs-sub { font-size:13px; line-height:20px; color:var(--muted); margin-top:4px; }
.cs-block { margin-bottom:32px; }
.cs-eyebrow { font-family:var(--sans); font-size:11px; font-weight:500; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); line-height:16px; }
.cs-eyebrow.amber { color:var(--amber); }
.cs-eyebrow .cs-num { font-family:var(--mono); letter-spacing:.04em; }
.cs-display { font-family:var(--sans); font-size:44px; font-weight:500; letter-spacing:-0.02em; line-height:1.1; color:var(--text); margin:8px 0 16px; overflow-wrap:anywhere; }
.cs-body { font-size:15px; line-height:24px; color:var(--text); }
.cs-body.muted, .cs-muted { color:var(--muted); }
.cs-wordmark { display:flex; align-items:baseline; gap:16px; flex-wrap:wrap; margin:0 0 8px; }
.cs-mark { display:inline-block; width:12px; height:12px; background:var(--amber); margin-right:12px; transform:translateY(-2px); }
.cs-name { font-family:var(--sans); font-size:28px; font-weight:600; letter-spacing:-0.02em; color:var(--text); line-height:1.1; }
.cs-tag { font-family:var(--sans); font-size:11px; font-weight:500; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); }
.cs-rule { border:0; border-top:1px solid var(--line); margin:8px 0 24px; }
/* ---- cards */
.cs-card { background:var(--surface); border:1px solid var(--line); border-radius:6px; padding:24px; }
.cs-grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(280px, 1fr)); gap:16px; margin-bottom:32px; }
.cs-grid3 { display:grid; grid-template-columns:repeat(3, minmax(0, 1fr)); gap:16px; margin-bottom:32px; }
@media (max-width: 900px) { .cs-grid3 { grid-template-columns:1fr; } }
.u, .pm { color:var(--muted); }
.u { font-size:.8em; margin-left:.25em; }
.pm { margin-left:.35em; }
/* ---- tables: horizontal hairlines only, eyebrow headers, 36 px rows, numbers right-aligned in mono */
.cs-table { width:100%; border-collapse:collapse; font-size:14px; margin:0 0 8px; }
.cs-table th { font-family:var(--sans); font-size:11px; font-weight:500; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); text-align:left;
  padding:0 12px; height:32px; border-bottom:1px solid var(--line2); white-space:nowrap; }
.cs-table td { height:36px; padding:0 12px; border-bottom:1px solid var(--line); color:var(--text); vertical-align:middle; font-family:var(--sans); font-size:14px; }
.cs-table th.num, .cs-table td.num { text-align:right; font-family:var(--mono); font-variant-numeric:tabular-nums; white-space:nowrap; }
.cs-table td.muted { color:var(--muted); font-size:13px; }
.cs-table td.ok { color:var(--green); font-family:var(--mono); text-align:center; font-size:16px; }
.cs-table td.bad { color:var(--red); font-family:var(--mono); text-align:center; font-size:16px; }
.cs-table tr:last-child td { border-bottom:0; }
/* ---- pills, status, log */
.cs-pill { display:inline-block; padding:0 9px; border:1px solid var(--line2); border-radius:999px; font-size:12px; line-height:20px; color:var(--muted);
  margin:0 6px 4px 0; white-space:nowrap; font-family:var(--mono); font-variant-numeric:tabular-nums; }
.cs-pill.amber { color:var(--amber); border-color:rgba(250,178,25,.45); }
.cs-pill.green { color:var(--green); border-color:rgba(76,174,76,.45); }
.cs-pill.red { color:var(--red); border-color:rgba(208,59,59,.45); }
.cs-pill.blue { color:#86b6ef; border-color:rgba(134,182,239,.4); }
.cs-status { font-size:14px; color:var(--text); line-height:22px; margin:8px 0; display:flex; gap:8px; align-items:baseline; }
.cs-status .cs-pill { flex:none; margin:0; }
.cs-status .t { font-family:var(--mono); color:var(--muted); font-size:12px; flex:none; }
.cs-status .m { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.cs-log { font-family:var(--mono); font-size:12px; line-height:20px; color:var(--muted); white-space:pre-wrap; margin:0; }
.cs-caption { font-size:13px; line-height:20px; color:var(--muted); margin:0 0 8px; }
.cs-note { font-size:15px; line-height:24px; color:var(--muted); border-left:2px solid var(--amber); padding:2px 12px; margin:8px 0 16px; }
.cs-param { font-family:var(--mono); font-size:13px; line-height:1.15; color:var(--text); overflow-wrap:anywhere; }
/* ---- verdict band */
@keyframes cs-rise { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:none; } }
@keyframes cs-pulse { 0%, 100% { border-left-color:rgba(250,178,25,1); } 50% { border-left-color:rgba(250,178,25,.25); } }
@keyframes cs-blink { 0%, 100% { opacity:1; } 50% { opacity:.35; } }
.cs-hero { display:grid; grid-template-columns:minmax(0, 3fr) minmax(0, 2fr); gap:32px; background:var(--surface); border:1px solid var(--line); border-radius:6px;
  padding:24px; margin:0 0 24px; animation:cs-rise .4s ease-out both; }
@media (max-width: 1000px) { .cs-hero { grid-template-columns:1fr; } }
.cs-hero .cs-ham { font-family:var(--mono); font-size:14px; line-height:24px; color:var(--text); margin:0 0 24px; overflow-wrap:anywhere; }
.cs-hero .cs-ham b { font-weight:500; color:var(--amber); }
.cs-hero .cs-ham sub, .cs-hero .cs-ham sup { font-size:.72em; }
.cs-kpis { display:flex; gap:32px; flex-wrap:wrap; margin:0 0 24px; }
.cs-kpi .v { font-family:var(--mono); font-size:32px; font-weight:400; line-height:40px; color:var(--text); font-variant-numeric:tabular-nums; letter-spacing:-0.01em; }
.cs-kpi .v .u { font-size:14px; margin-left:4px; }
.cs-kpi .l { font-size:11px; font-weight:500; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); line-height:16px; margin-top:2px; }
.cs-kpi .l .cs-num { font-family:var(--mono); letter-spacing:.04em; }
.cs-bar { display:flex; height:6px; border-radius:3px; overflow:hidden; background:rgba(240,239,236,.08); }
.cs-bar .g { background:var(--green); height:100%; }
.cs-bar .r { background:var(--red); height:100%; }
.cs-bar-ends { display:flex; justify-content:space-between; font-family:var(--mono); font-size:12px; line-height:20px; margin-top:6px; color:var(--muted); }
.cs-bar-ends .g { color:var(--green); } .cs-bar-ends .r { color:var(--red); }
.cs-hero-r .cs-eyebrow { margin:16px 0 8px; }
.cs-hero-r svg { width:100%; max-width:440px; margin:0 auto; }
/* ---- hypothesis race */
.cs-race { display:flex; flex-direction:column; gap:4px; margin:8px 0 0; }
.cs-row { border-left:2px solid transparent; padding:6px 0 6px 12px; transition:opacity .4s ease; }
.cs-row.active { border-left-color:var(--amber); animation:cs-pulse 1.2s ease-in-out infinite; }
.cs-row.dim { opacity:.4; }
.cs-row .top { display:grid; grid-template-columns:minmax(0, 1fr) 140px 64px 72px; gap:12px; align-items:center; }
.cs-row .name { font-family:var(--sans); font-size:14px; line-height:20px; color:var(--text); overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
.cs-row .bar { height:6px; background:rgba(240,239,236,.08); border-radius:3px; overflow:hidden; }
.cs-row .bar div { height:100%; background:rgba(240,239,236,.45); border-radius:3px; transition:width .4s ease; }
.cs-row.winner .bar div { background:var(--amber); }
.cs-row .score { font-family:var(--mono); font-size:13px; text-align:right; color:var(--text); font-variant-numeric:tabular-nums; }
.cs-row .status { font-family:var(--sans); font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); text-align:right; }
.cs-row.active .status { color:var(--amber); animation:cs-blink 1.2s ease-in-out infinite; }
.cs-row.winner .status { color:var(--amber); }
.cs-row .status.rejected { color:var(--red); }
.cs-row .spark { display:flex; align-items:center; gap:12px; margin-top:6px; }
.cs-row .spark svg { height:40px; flex:1 1 auto; display:block; }
.cs-row .spark .sv { font-family:var(--mono); font-size:12px; color:var(--muted); white-space:nowrap; }
/* ---- stepper */
.cs-steps { display:flex; align-items:center; gap:0; margin:8px 0 16px; }
.cs-step { display:flex; align-items:center; gap:8px; font-size:12px; letter-spacing:.06em; text-transform:uppercase; color:var(--muted); white-space:nowrap; }
.cs-step .dot { width:10px; height:10px; border-radius:50%; border:1.5px solid var(--line2); background:transparent; flex:none; }
.cs-step.done { color:var(--text); } .cs-step.done .dot { background:var(--amber); border-color:var(--amber); }
.cs-step.active { color:var(--amber); } .cs-step.active .dot { border-color:var(--amber); animation:cs-blink 1.2s ease-in-out infinite; }
.cs-step-line { flex:1 1 12px; height:1px; background:var(--line2); margin:0 8px; min-width:8px; }
.cs-step-line.done { background:var(--amber); }
/* ---- spec sheet cards */
.cs-spec-row { display:flex; justify-content:space-between; align-items:baseline; gap:12px; padding:0; min-height:36px; border-bottom:1px solid var(--line); }
.cs-spec-row:last-child { border-bottom:0; }
.cs-spec-row .l { font-size:13px; line-height:18px; color:var(--muted); }
.cs-spec-row .v { font-family:var(--mono); font-size:14px; color:var(--text); white-space:nowrap; font-variant-numeric:tabular-nums; }
.cs-spec-empty { font-size:13px; color:var(--muted); padding:8px 0; }
/* ---- numbered cards */
.cs-badge { display:inline-flex; align-items:center; justify-content:center; width:24px; height:24px; border-radius:4px; background:var(--amber); color:#111214;
  font-family:var(--mono); font-size:13px; font-weight:500; flex:none; }
.cs-badge.muted { background:transparent; border:1px solid var(--line2); color:var(--muted); }
.cs-look { display:flex; flex-direction:column; gap:8px; }
.cs-look .head { display:flex; align-items:center; gap:12px; }
.cs-look .head .cs-eyebrow { flex:1 1 auto; }
.cs-look .f { font-family:var(--mono); font-size:24px; line-height:32px; color:var(--text); font-variant-numeric:tabular-nums; letter-spacing:-0.01em; }
.cs-look .f .u { font-size:13px; }
.cs-look .x { font-family:var(--mono); font-size:13px; line-height:20px; color:var(--muted); }
.cs-look .why { font-size:13px; line-height:20px; color:var(--muted); }
.cs-look .why b { color:var(--text); font-weight:500; }
/* ---- misc */
.cs-check { color:var(--green); font-family:var(--mono); } .cs-cross { color:var(--red); font-family:var(--mono); }
.cs-kv { display:flex; justify-content:space-between; gap:12px; font-size:13px; color:var(--muted); line-height:20px; }
.cs-kv .v { font-family:var(--mono); color:var(--text); }
.cs-inline-steps { margin:0; }
</style>"""

HEADER = """<div class="cs-wordmark"><span class="cs-name"><span class="cs-mark"></span>CircuitSleuth</span>
<span class="cs-tag">Identify the circuit behind a spectroscopy map</span></div><hr class="cs-rule">"""

UNITS = {"EJ": "GHz", "EC": "GHz", "EL": "GHz", "EJmax": "GHz", "fr": "GHz", "g": "GHz", "D": "GHz", "E": "GHz", "t_c": "GHz",
         "theta": "rad", "phi": "rad", "field_scale": "per unit", "lever_arm": "GHz / unit", "d": "", "period": "", "offset": "", "x0": "",
         "f_rad0": "MHz", "volt_scale": "MHz / √unit", "alpha1": "MHz/(V/cm)²", "alpha2": "MHz/(V/cm)²", "dfs": "MHz", "f0": "MHz"}


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header() -> None:
    st.markdown(HEADER, unsafe_allow_html=True)


def rule() -> None:
    st.markdown('<hr class="cs-rule">', unsafe_allow_html=True)


def md(s: str) -> None:
    """Render one HTML fragment (no blank lines inside, so Markdown never re-enters)."""
    st.markdown(s.replace("\n\n", "\n"), unsafe_allow_html=True)


def esc(s) -> str:
    return html.escape(str(s))


# ------------------------------------------------------------------ small pieces
def section(title: str, eyebrow: str = "", sub: str = "") -> str:
    eb = f'<div class="cs-eyebrow">{esc(eyebrow)}</div>' if eyebrow else ""
    sb = f'<div class="cs-sub">{esc(sub)}</div>' if sub else ""
    return f'<div class="cs-section">{eb}<h2 class="cs-h2">{esc(title)}</h2>{sb}</div>'


def eyebrow(text: str, amber: bool = False) -> str:
    return f'<div class="cs-eyebrow{" amber" if amber else ""}">{esc(text)}</div>'


def pill(text: str, kind: str = "") -> str:
    """A small status label. kind in {"", "amber", "green", "red", "blue"}."""
    return f'<span class="cs-pill {kind}">{esc(text)}</span>'


def pills(items, kind: str = "") -> str:
    return "".join(pill(t, kind) for t in items)


def status_html(label: str, text: str, kind: str = "", t: float | None = None) -> str:
    ts = f'<span class="t">{t:5.1f} s</span>' if t is not None else ""
    return f'<div class="cs-status">{pill(label, kind)}{ts}<span class="m" title="{esc(text)}">{esc(text)}</span></div>'


def log_html(lines, keep: int | None = None) -> str:
    body = "\n".join(esc(l) for l in (list(lines) if keep is None else list(lines)[-keep:]))
    return f'<div class="cs-log">{body}</div>'


def caption(text: str) -> str:
    return f'<div class="cs-caption">{esc(text)}</div>'


def note(text: str) -> str:
    return f'<div class="cs-note">{esc(text)}</div>'


def num(value, unit: str = "", sigma=None, sig: int = 4) -> str:
    """A number in mono with a muted ± σ and unit: 25.23 ± 0.17 GHz."""
    import numpy as np

    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return '<span class="cs-num">–</span>'
    s = f'<span class="cs-num">{float(value):.{sig}g}'.replace("-", "−")
    if sigma is not None and np.isfinite(sigma):
        s += f'<span class="pm">± {float(sigma):.2g}</span>'
    if unit:
        s += f'<span class="u">{esc(unit)}</span>'
    return s + "</span>"


def table_html(columns, rows, muted=(), numeric=(), classes=None) -> str:
    """Hairline table. `muted` / `numeric` list column indices; `classes` optionally gives a per-cell class via (row, col) -> str.
    Cells that already contain HTML (start with '<') are inserted as is; everything else is escaped."""
    # headers are set in uppercase by CSS; Greek letters (σ) must stay lowercase
    head = "".join(f'<th class="{"num" if i in numeric else ""}">{esc(c).replace("σ", "<span style=text-transform:none>σ</span>")}</th>' for i, c in enumerate(columns))
    body = []
    for ri, r in enumerate(rows):
        cells = []
        for i, c in enumerate(r):
            cls = " ".join(k for k, on in (("muted", i in muted), ("num", i in numeric)) if on)
            if classes:
                cls = (cls + " " + (classes(ri, i) or "")).strip()
            content = str(c) if isinstance(c, str) and c.startswith("<") else esc(c)
            cells.append(f'<td class="{cls}">{content}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="cs-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'


# ------------------------------------------------------------------ stepper (I)
STEPS = ["Extract", "Propose", "Fit", "Verify", "Report"]


def stepper_html(stage: int, running: bool = False) -> str:
    """`stage` = number of completed steps (0..5); when `running`, the next step is shown active."""
    parts = []
    for i, name in enumerate(STEPS):
        if i < stage:
            cls = "done"
        elif i == stage and running:
            cls = "active"
        else:
            cls = ""
        if i:
            parts.append(f'<div class="cs-step-line{" done" if i <= stage else ""}"></div>')  # lit once the previous step is done
        parts.append(f'<div class="cs-step {cls}"><span class="dot"></span>{name}</div>')
    return f'<div class="cs-steps">{"".join(parts)}</div>'


# ------------------------------------------------------------------ hypothesis race (C)
def _sparkline_svg(values, width: int = 600, height: int = 40) -> str:
    vals = [float(v) for v in values if v is not None]
    if len(vals) < 2:
        return f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none"></svg>'
    lo, hi = min(vals), max(vals)
    span = (hi - lo) or 1.0
    n = len(vals)
    pts = " ".join(f"{i * (width - 2) / (n - 1) + 1:.1f},{height - 3 - (v - lo) / span * (height - 6):.1f}" for i, v in enumerate(vals))
    return (f'<svg viewBox="0 0 {width} {height}" preserveAspectRatio="none" fill="none">'
            f'<polyline points="{pts}" stroke="#fab219" stroke-width="1.5" vector-effect="non-scaling-stroke" stroke-linejoin="round"/></svg>')


def race_html(rows: list[dict], active: str | None = None, finished: bool = False, winner: str | None = None) -> str:
    """One row per hypothesis in the order tried. Each row: name, status (queued / fitting / fitted / rejected / accepted / skipped),
    score (float or None), best (list of best-so-far losses for the sparkline)."""
    if not rows:
        return '<div class="cs-race"><div class="cs-spec-empty">Waiting for hypotheses…</div></div>'
    scores = [float(r["score"]) for r in rows if r.get("score") is not None]
    top = max([s for s in scores if s > 0] or [1.0])
    out = []
    for r in rows:
        name, status, score = r["name"], r.get("status", "queued"), r.get("score")
        cls = ["cs-row"]
        is_active = (not finished) and name == active and status == "fitting"
        is_winner = winner is not None and name == winner
        if is_active:
            cls.append("active")
        if is_winner:
            cls.append("winner")
        if finished and not is_winner and status in ("fitted", "rejected", "skipped"):
            cls.append("dim")
        if status == "accepted" or is_winner:
            status_txt = "accepted"
        else:
            status_txt = status
        width = 0.0 if score is None else max(0.0, min(100.0, 100.0 * max(float(score), 0.0) / top))
        sc = "" if score is None else f"{float(score):.3f}".replace("-", "−")
        top_html = (f'<div class="top"><span class="name" title="{esc(name)}">{esc(name)}</span><div class="bar"><div style="width:{width:.1f}%"></div></div>'
                    f'<span class="score">{sc}</span><span class="status {esc(status_txt)}">{esc(status_txt)}</span></div>')
        spark = ""
        if is_active:
            best = r.get("best") or []
            cur = f'{best[-1]:.4f}' if best else "…"
            spark = (f'<div class="spark">{_sparkline_svg(best)}<span class="sv" title="best-so-far search loss">best loss {cur}</span></div>')
        out.append(f'<div class="{" ".join(cls)}">{top_html}{spark}</div>')
    return f'<div class="cs-race">{"".join(out)}</div>'


# ------------------------------------------------------------------ verdict band (B)
def kpi_html(value: str, label: str) -> str:
    return f'<div class="cs-kpi"><div class="v">{value}</div><div class="l">{label}</div></div>'


def hero_html(eyebrow_txt: str, title: str, hamiltonian: str, kpis: list[tuple[str, str]], explained: float | None, right_html: str) -> str:
    """The verdict band: eyebrow, display title, Hamiltonian line, three KPIs and the explained/unexplained bar on the left;
    schematic and Hamiltonian table (right_html) on the right."""
    k = "".join(kpi_html(v, l) for v, l in kpis)
    bar = ""
    if explained is not None:
        g = max(0.0, min(100.0, 100.0 * float(explained)))
        r = 100.0 - g
        bar = (f'<div class="cs-bar"><div class="g" style="width:{g:.1f}%"></div><div class="r" style="width:{r:.1f}%"></div></div>'
               f'<div class="cs-bar-ends"><span class="g">{g:.0f}% explained</span><span class="r">{r:.0f}% unexplained</span></div>')
    return (f'<div class="cs-hero"><div class="cs-hero-l"><div class="cs-eyebrow amber">{eyebrow_txt}</div><div class="cs-display">{esc(title)}</div>'
            f'<div class="cs-ham">{hamiltonian}</div><div class="cs-kpis">{k}</div>{bar}</div><div class="cs-hero-r">{right_html}</div></div>')


def amber(v, sig: int = 4) -> str:
    """A fitted number substituted into the Hamiltonian line."""
    return f"<b>{float(v):.{sig}g}</b>".replace("-", "−")


def hamiltonian_html(family: str, params: dict, coupled: bool) -> str:
    p = params
    if family == "transmon":
        h = f"H = 4·{amber(p['EC'], 3)} n<sup>2</sup> − {amber(p['EJ'])} cos φ"
    elif family == "tunable_transmon":
        h = (f"H = 4·{amber(p['EC'], 3)} n<sup>2</sup> − E<sub>J</sub>(Φ) cos φ,&nbsp; E<sub>J</sub>(Φ) = {amber(p['EJmax'])} "
             f"√(cos<sup>2</sup>πΦ + {amber(p['d'], 3)}<sup>2</sup> sin<sup>2</sup>πΦ)")
    elif family == "fluxonium":
        h = f"H = 4·{amber(p['EC'], 3)} n<sup>2</sup> + ½·{amber(p['EL'], 3)} φ<sup>2</sup> − {amber(p['EJ'])} cos(φ − 2πΦ)"
    elif family == "nv_ensemble":
        h = f"H = {amber(p['D'])} S<sub>z</sub><sup>2</sup> + {amber(p['E'], 3)} (S<sub>x</sub><sup>2</sup> − S<sub>y</sub><sup>2</sup>) + γ<sub>e</sub> B·S"
    elif family == "charge_qubit":
        h = f"f = √(ε<sup>2</sup> + (2·{amber(p['t_c'])})<sup>2</sup>)"
    elif family == "trapped_ion":
        h = (f"ω<sub>z</sub> = {amber(p['volt_scale'])} √(V − V<sub>0</sub>),&nbsp; ω<sub>r</sub> = √({amber(p['f_rad0'])}<sup>2</sup> − ω<sub>z</sub><sup>2</sup>/2),"
             f"&nbsp; sidebands at ±ω<sub>z</sub>, ±ω<sub>r</sub>, ±2ω<sub>z</sub>")
    elif family == "rydberg_atom":
        h = (f"Δf = f<sub>0</sub> ± {amber(p['dfs'])}/2 − ½ α E<sup>2</sup>,&nbsp; α = {amber(p['alpha1'])} | {amber(p['alpha2'])},"
             f"&nbsp; E = {amber(p['field_scale'])} (V − V<sub>0</sub>)")
    else:
        h = esc(family)
    if coupled and "fr" in p and "g" in p:
        h += f" + {amber(p['fr'])} a<sup>†</sup>a + {amber(p['g'], 3)} n(a + a<sup>†</sup>)"
    try:
        from circuitsleuth import models as _m
        unit = _m.PLATFORMS[_m.platform_of(family)].get("freq_unit", "GHz")
    except Exception:
        unit = "GHz"
    return h + f'<span class="u">{unit}</span>'


# ------------------------------------------------------------------ spec cards (F)
def spec_rows_html(rows: list[tuple[str, str, str]]) -> str:
    """rows: (label, value_html, note) -> dim label left, mono value right, note as a tooltip."""
    if not rows:
        return '<div class="cs-spec-empty">nothing to show for this result</div>'
    out = []
    for label, value, note_txt in rows:
        title = f' title="{esc(note_txt)}"' if note_txt else ""
        v = value if (isinstance(value, str) and value.startswith("<")) else esc(value)
        out.append(f'<div class="cs-spec-row"{title}><span class="l">{esc(label)}</span><span class="v">{v}</span></div>')
    return "".join(out)


# ------------------------------------------------------------------ numbered cards (F) and unexplained cards
def card_html(badge: str | None, kicker: str, big: str, sub: str, why: str, muted_badge: bool = False) -> str:
    """A compact recommendation / finding card: badge + eyebrow kicker on top, a large mono figure (`big`, HTML allowed),
    a mono sub-line (control range, point count) and one line of why."""
    b = f'<span class="cs-badge{" muted" if muted_badge else ""}">{esc(badge)}</span>' if badge is not None else ""
    k = f'<div class="cs-eyebrow">{esc(kicker)}</div>' if kicker else ""
    big_html = f'<div class="f">{big}</div>' if big else ""
    sub_html = f'<div class="x">{esc(sub)}</div>' if sub else ""
    why_html = f'<div class="why">{esc(why)}</div>' if why else ""
    return f'<div class="cs-card cs-look"><div class="head">{b}{k}</div>{big_html}{sub_html}{why_html}</div>'


def frange_html(f_lo: float, f_hi: float, unit: str = "GHz") -> str:
    return f'{f_lo:.2f}–{f_hi:.2f}<span class="u">{esc(unit)}</span>'.replace("-", "−")


def grid_html(cards: list[str], three: bool = False) -> str:
    return f'<div class="{"cs-grid3" if three else "cs-grid"}">{"".join(cards)}</div>'


# ------------------------------------------------------------------ text helpers (H)
def two_sentences(text: str) -> str:
    """The first two sentences of a summary; never the raw text of an exception."""
    text = re.sub(r"\s+", " ", str(text or "")).strip()
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return " ".join(parts[:2]).strip()
