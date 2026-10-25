"""One CSS block (IBM Plex Sans / IBM Plex Mono, 8 px rhythm, flat cards) and the HTML builders for the pieces Streamlit has
no widget for: the verdict band, the hypothesis race, the pipeline stepper, spec-sheet cards, numbered recommendation
cards, the truth strip and compact hairline tables.

Palette: background #111214, surface #1a1a19, hairline rgba(240,239,236,.08), text #f0efec, muted #c3c2b7, accent #fab219.
Type: display 44/500/-0.02em, section 20/600, eyebrow 11 uppercase .12em, body 15/24, mono 14, KPI 32 mono.
Every number is set in Plex Mono with tabular figures; ± and units are muted.
"""
from __future__ import annotations

import html
import os
import re

import streamlit as st

SANS = '"IBM Plex Sans", "Helvetica Neue", Helvetica, Arial, sans-serif'
MONO = '"IBM Plex Mono", ui-monospace, Menlo, Consolas, monospace'
FONT = SANS  # kept for older imports

CSS = """<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&family=Playfair+Display:wght@700&display=swap');
:root { --bg:#111214; --surface:#1a1a19; --line:rgba(240,239,236,.08); --line2:rgba(240,239,236,.16); --text:#f0efec; --muted:#c3c2b7;
  --amber:#fab219; --green:#4cae4c; --red:#d03b3b; --sans:"IBM Plex Sans","Helvetica Neue",Helvetica,Arial,sans-serif;
  --mono:"IBM Plex Mono",ui-monospace,Menlo,Consolas,monospace; }
/* ---- chrome: no toolbar, no running indicator, no toasts, no footer, no shadows */
#MainMenu, footer, [data-testid="stDecoration"], .stAppDeployButton, [data-testid="stToolbarActions"], [data-testid="stMainMenu"],
[data-testid="stStatusWidget"], [data-testid="stToast"], [data-testid="stToastContainer"], [data-testid="stAppToastContainer"],
div[class*="stToast"], [data-testid="stHeaderActionElements"] { display:none !important; }
/* the toolbar stays: it is where Streamlit puts the "expand sidebar" button once the sidebar is collapsed */
[data-testid="stToolbar"] { background:transparent !important; }
[data-testid="stExpandSidebarButton"] { display:inline-flex !important; visibility:visible !important; opacity:1 !important; }
header[data-testid="stHeader"] { background:transparent; box-shadow:none; height:2.25rem; }
.block-container { max-width:1200px; padding:24px 32px 64px; }
section[data-testid="stSidebar"][data-collapsed="false"] { width:280px !important; min-width:280px !important; max-width:280px !important; }
section[data-testid="stSidebar"][data-collapsed="false"] > div:first-child { width:280px !important; }
section[data-testid="stSidebar"][data-collapsed="true"] { width:0 !important; min-width:0 !important; }
section[data-testid="stSidebar"] .block-container { padding-top:1.25rem; }
/* the sidebar has a fixed width: no resize handle (it showed a col-resize cursor that did nothing) */
[data-testid="stSidebarResizeHandle"] { display:none !important; }
section[data-testid="stSidebar"] { cursor:default; }
/* with the sidebar collapsed the main column fills the window */
section[data-testid="stSidebar"][data-collapsed="true"] ~ div .block-container { max-width:none; padding-left:48px; padding-right:48px; }
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
/* ---- tabs (diagnostics): quiet labels, amber highlight */
.stApp [data-baseweb="tab-list"] { gap:24px; border-bottom:1px solid var(--line); }
.stApp [data-baseweb="tab"] { padding:8px 0; background:transparent; }
.stApp [data-baseweb="tab"] p { font-family:var(--sans); font-size:13px; font-weight:500; letter-spacing:.02em; color:var(--muted); }
.stApp [aria-selected="true"][data-baseweb="tab"] p { color:var(--text); }
.stApp [data-baseweb="tab-highlight"] { background-color:var(--amber); height:2px; }
.stApp [data-baseweb="tab-border"] { background-color:var(--line); }
.stApp [data-testid="stTabs"] [data-testid="stVerticalBlock"] { padding-top:16px; }
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
.cs-brand { display:flex; align-items:center; gap:14px; flex-wrap:wrap; margin:0 0 10px; }
.cs-brand svg { height:46px; width:auto; display:block; flex:none; }
.cs-brand .cs-name { font-family:"Playfair Display", Georgia, "Times New Roman", serif; font-size:36px; font-weight:700; letter-spacing:0; line-height:1; color:var(--text); }
.cs-brand .cs-tag { margin-left:6px; }
/* ---- verdict band as a Streamlit container (so the Hamiltonian can be real KaTeX) */
.st-key-hero { animation:cs-rise .4s ease-out both; }
.st-key-hero [data-testid="stVerticalBlockBorderWrapper"] > div > [data-testid="stVerticalBlock"] { padding:24px !important; }
.st-key-hero .katex-display { text-align:left; margin:2px 0 14px; padding:2px 0 6px; overflow-x:auto; overflow-y:hidden; }
.st-key-hero .katex-display > .katex { text-align:left; white-space:nowrap; }
.st-key-hero .katex { font-size:1.22em; color:var(--text); }
.st-key-hero .cs-kpis { margin-top:4px; }
.stApp .katex { font-size:1.08em; }
section[data-testid="stSidebar"] .katex { font-size:1.0em; }
section[data-testid="stSidebar"] .cs-param-tex p { margin:0; line-height:1.2; }
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
.cs-row .top.with-p { grid-template-columns:minmax(0, 1fr) 140px 64px 64px 72px; }
.cs-row .score.p { color:var(--muted); }
.cs-race-head { display:grid; grid-template-columns:minmax(0, 1fr) 140px 64px 64px 72px; gap:12px; padding:0 0 4px 14px; }
.cs-race-head span { font-family:var(--sans); font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--muted); text-align:right; }
.cs-race-head span:first-child, .cs-race-head span:nth-child(2) { text-align:left; }
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

BRAND_NAME = "q\u0124olmes"  # qĤ: q for quantum, Ĥ (U+0124): the hat on H marks the Hamiltonian operator; the logo swaps it for a detective's fedora
TAGLINE = "Hamiltonians, deduced from spectra"
_BRAND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "static", "brand")
_MARK_FALLBACK = (
    '<svg xmlns="http://www.w3.org/2000/svg" viewBox="8 16 184 170" role="img" aria-label="qĤolmes logo">'
    '<g transform="translate(100 100) scale(1.16) translate(-100 -115)">'
    '<path d="M24 92 C56 80 144 80 176 92 C154 103 46 103 24 92 Z" fill="#D99A1E"/>'
    '<path d="M64 88 C62 60 70 44 86 46 C94 47 98 52 100 55 C102 52 106 47 114 46 C130 44 138 60 136 88 Z" fill="#D99A1E"/>'
    '<path d="M63 76 L137 76 L136 88 L64 88 Z" fill="#8A5A0C"/>'
    '<path d="M100 55 L100 68" stroke="#8A5A0C" stroke-width="3" stroke-linecap="round" fill="none"/>'
    '<path d="M54 100 H78 V132 H122 V100 H146 V186 H122 V154 H78 V186 H54 Z" fill="#FFFFFF"/></g></svg>')


def brand_mark_svg() -> str:
    """The logo mark for dark backgrounds (white H), inlined so the header needs no static-file route."""
    try:
        with open(os.path.join(_BRAND_DIR, "holmes-mark-reverse.svg"), encoding="utf-8") as fh:
            svg = fh.read()
        svg = re.sub(r"<\?xml[^>]*>", "", svg)
        return re.sub(r"\s+", " ", svg).strip()
    except OSError:
        return _MARK_FALLBACK


HEADER = (f'<div class="cs-brand" aria-label="{BRAND_NAME}">{brand_mark_svg()}<span class="cs-name">{BRAND_NAME}</span>'
          f'<span class="cs-tag">{TAGLINE}</span></div><hr class="cs-rule">')

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
    sb = f'<div class="cs-sub">{mathify(esc(sub))}</div>' if sub else ""
    return f'<div class="cs-section">{eb}<h2 class="cs-h2">{esc(title_case(title))}</h2>{sb}</div>'


def eyebrow(text: str, amber: bool = False) -> str:
    return f'<div class="cs-eyebrow{" amber" if amber else ""}">{esc(text)}</div>'


def pill(text: str, kind: str = "") -> str:
    """A small status label. kind in {"", "amber", "green", "red", "blue"}."""
    return f'<span class="cs-pill {kind}">{esc(cap_first(text))}</span>'


def pills(items, kind: str = "") -> str:
    return "".join(pill(t, kind) for t in items)


def status_html(label: str, text: str, kind: str = "", t: float | None = None) -> str:
    ts = f'<span class="t">{t:5.1f} s</span>' if t is not None else ""
    return f'<div class="cs-status">{pill(label, kind)}{ts}<span class="m" title="{esc(text)}">{mathify(esc(cap_first(text)))}</span></div>'


def log_html(lines, keep: int | None = None) -> str:
    body = "\n".join(esc(l) for l in (list(lines) if keep is None else list(lines)[-keep:]))
    return f'<div class="cs-log">{body}</div>'


def cap_first(text: str) -> str:
    """Sentence-start capital: upper-case the first letter when the text opens with an ordinary lower-case word (two or more
    letters, no digits); symbols such as g, d, f01 or φ, and strings that open with a tag, are left alone."""
    s = str(text or "")
    m = re.match(r"^([a-z][a-z'’-]+)\b", s)
    if not m or m.group(1) in _KEEP_WORDS:
        return s
    return s[0].upper() + s[1:]


def caption(text: str) -> str:
    return f'<div class="cs-caption">{mathify(esc(cap_first(text)))}</div>'


def note(text: str) -> str:
    return f'<div class="cs-note">{mathify(esc(cap_first(text)))}</div>'


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


def race_html(rows: list[dict], active: str | None = None, finished: bool = False, winner: str | None = None, posterior: dict | None = None) -> str:
    """One row per hypothesis in the order tried. Each row: name, status (queued / fitting / fitted / rejected / accepted / skipped),
    score (float or None), best (list of best-so-far losses for the sparkline).  `posterior` maps a hypothesis name to its Bayesian
    model probability (Laplace evidence); when given, a P column is shown after the score."""
    if not rows:
        return '<div class="cs-race"><div class="cs-spec-empty">Waiting for hypotheses…</div></div>'
    has_p = bool(posterior)
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
        p_html = ""
        if has_p:
            pv = (posterior or {}).get(name)
            p_txt = "" if pv is None else (f"{pv:.3f}" if pv >= 5e-4 else ("<10⁻³" if pv >= 1e-6 else "≈0"))
            p_html = f'<span class="score p" title="posterior model probability (Laplace evidence)">{p_txt}</span>'
        top_html = (f'<div class="top{" with-p" if has_p else ""}"><span class="name" title="{esc(name)}">{esc(title_case(name))}</span><div class="bar"><div style="width:{width:.1f}%"></div></div>'
                    f'<span class="score">{sc}</span>{p_html}<span class="status {esc(status_txt)}">{esc(status_txt)}</span></div>')
        spark = ""
        if is_active:
            best = r.get("best") or []
            cur = f'{best[-1]:.4f}' if best else "…"
            spark = (f'<div class="spark">{_sparkline_svg(best)}<span class="sv" title="best-so-far search loss">best loss {cur}</span></div>')
        out.append(f'<div class="{" ".join(cls)}">{top_html}{spark}</div>')
    head = ('<div class="cs-race-head"><span>hypothesis</span><span></span><span>score</span><span>P</span><span></span></div>' if has_p else "")
    return f'<div class="cs-race">{head}{"".join(out)}</div>'


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
        return '<div class="cs-spec-empty">Nothing to show for this result</div>'
    out = []
    for label, value, note_txt in rows:
        title = f' title="{esc(note_txt)}"' if note_txt else ""
        v = value if (isinstance(value, str) and value.startswith("<")) else esc(value)
        out.append(f'<div class="cs-spec-row"{title}><span class="l">{mathify(esc(title_case(label)))}</span><span class="v">{v}</span></div>')
    return "".join(out)


# ------------------------------------------------------------------ numbered cards (F) and unexplained cards
def card_html(badge: str | None, kicker: str, big: str, sub: str, why: str, muted_badge: bool = False) -> str:
    """A compact recommendation / finding card: badge + eyebrow kicker on top, a large mono figure (`big`, HTML allowed),
    a mono sub-line (control range, point count) and one line of why."""
    b = f'<span class="cs-badge{" muted" if muted_badge else ""}">{esc(badge)}</span>' if badge is not None else ""
    k = f'<div class="cs-eyebrow">{mathify(esc(title_case(kicker)))}</div>' if kicker else ""
    big_html = f'<div class="f">{big}</div>' if big else ""
    sub_html = f'<div class="x">{mathify(esc(cap_first(sub)))}</div>' if sub else ""
    why_html = f'<div class="why">{mathify(esc(cap_first(why)))}</div>' if why else ""
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


# ------------------------------------------------------------------ Title Case, typeset parameter names, LaTeX Hamiltonians
_SMALL_WORDS = {"a", "an", "the", "of", "in", "at", "for", "to", "on", "by", "vs", "per", "and", "or", "with", "from", "into", "over", "as",
                "is", "are", "its", "than", "off", "up", "via", "but", "nor", "so", "yet", "if"}
_KEEP_WORDS = {"scqubits", "arXiv", "npz", "csv", "json", "html", "h5", "hdf5", "nD", "et", "al", "e.g", "i.e"}


_ACRONYMS = {"nv": "NV", "cnn": "CNN", "lo": "LO", "squid": "SQUID", "rms": "RMS", "odmr": "ODMR", "dqd": "DQD", "tls": "TLS"}


def _cap_word(w: str) -> str:
    if not w or w in _KEEP_WORDS or any(ch.isdigit() for ch in w) or not w.isascii():
        return w
    if w.lower() in _ACRONYMS:
        return _ACRONYMS[w.lower()]
    if any(ch.isupper() for ch in w[1:]):  # CNN, GHz, EJmax, SQUID: already styled
        return w
    if not w[0].isalpha():
        return w
    if len(w) == 1 and w not in ("a", "i"):  # a lone symbol such as g or d (coupling, asymmetry) is a variable, not a word
        return w
    return w[0].upper() + w[1:]


def title_case(text: str) -> str:
    """Capitalise the first letter of every significant word (hyphenated parts too); short function words stay lower-case
    except at the start or after a separator; tokens with digits, Greek letters or internal capitals are left alone."""
    if not text:
        return text
    out = []
    start = True
    for tok in str(text).split(" "):
        if not tok:
            out.append(tok)
            continue
        m = re.match(r"^([\(\[\"'“‘«]*)(.*?)([\)\]\"'”’»:;,.!?…]*)$", tok)
        pre, core, post = m.groups() if m else ("", tok, "")
        if core in ("·", "—", "–", "-", "|", "/", "→", "↔", "vs.", ":"):
            out.append(tok)
            start = True
            continue
        if core and core.lower() in _SMALL_WORDS and not start and core.islower():
            out.append(pre + core.lower() + post)
        else:
            if not any(ch.isdigit() for ch in core) and re.search(r"[-–]", core):  # held-out, qubit–resonator: capitalise each part
                out.append(pre + "".join(x if x in "-–" else _cap_word(x) for x in re.split(r"([-–])", core)) + post)
            else:
                out.append(pre + _cap_word(core) + post)
        start = bool(post and post[-1] in ":;·—–!?") or tok.endswith(":")
    return " ".join(out)


PARAM_HTML = {"EJ": "E<sub>J</sub>", "EC": "E<sub>C</sub>", "EL": "E<sub>L</sub>", "EJmax": "E<sub>J,max</sub>", "fr": "f<sub>r</sub>", "g": "g", "d": "d",
              "period": "period", "offset": "offset", "D": "D", "E": "E", "theta": "θ", "phi": "φ", "field_scale": "κ", "x0": "x<sub>0</sub>",
              "t_c": "t<sub>c</sub>", "lever_arm": "lever arm", "f_rad0": "f<sub>r,0</sub>", "volt_scale": "κ<sub>V</sub>", "alpha1": "α<sub>1</sub>",
              "alpha2": "α<sub>2</sub>", "dfs": "Δ<sub>fs</sub>", "f0": "f<sub>0</sub>"}
PARAM_TEX = {"EJ": r"E_J", "EC": r"E_C", "EL": r"E_L", "EJmax": r"E_{J,\max}", "fr": r"f_r", "g": r"g", "d": r"d", "period": r"\text{period}",
             "offset": r"\text{offset}", "D": r"D", "E": r"E", "theta": r"\theta", "phi": r"\varphi", "field_scale": r"\kappa", "x0": r"x_0",
             "t_c": r"t_c", "lever_arm": r"\text{lever arm}", "f_rad0": r"f_{r,0}", "volt_scale": r"\kappa_V", "alpha1": r"\alpha_1",
             "alpha2": r"\alpha_2", "dfs": r"\Delta_{\mathrm{fs}}", "f0": r"f_0"}


def param_html(name: str) -> str:
    """Typeset parameter name for HTML tables (sub/superscripts); unknown names are escaped as they are."""
    return f'<span class="cs-pname">{PARAM_HTML.get(name, esc(name))}</span>'


def param_tex(name: str) -> str:
    """LaTeX for a parameter name (KaTeX, inline): $E_{J,\\max}$."""
    return PARAM_TEX.get(name, r"\text{" + str(name).replace("_", r"\_") + "}")


_MATHIFY = [
    (re.compile(r"\bE(J|C|L)max\b"), r"E<sub>\1,max</sub>"),
    (re.compile(r"\bE([JCL])\b"), r"E<sub>\1</sub>"),
    (re.compile(r"\bf(01|12|02|03)\b"), r"f<sub>\1</sub>"),
    (re.compile(r"\bfr\b"), "f<sub>r</sub>"),
    (re.compile(r"\bf_rad0\b"), "f<sub>r,0</sub>"),
    (re.compile(r"\bt_c\b"), "t<sub>c</sub>"),
    (re.compile(r"\bm_?j\b"), "m<sub>j</sub>"),
    (re.compile(r"\bx0\b"), "x<sub>0</sub>"),
    (re.compile(r"\bV0\b"), "V<sub>0</sub>"),
    (re.compile(r"\bT1\b"), "T<sub>1</sub>"),
    (re.compile(r"ω_([zr])\b"), r"ω<sub>\1</sub>"),
    (re.compile(r"\bTφ\b"), "T<sub>φ</sub>"),
    (re.compile(r"\balpha([12])\b"), r"α<sub>\1</sub>"),
    (re.compile(r"\bvolt_scale\b"), "κ<sub>V</sub>"),
    (re.compile(r"\bfield_scale\b"), "κ"),
    (re.compile(r"\blever_arm\b"), "lever arm"),
    (re.compile(r"\bdfs\b"), "Δ<sub>fs</sub>"),
    (re.compile(r"\bf0\b"), "f<sub>0</sub>"),
    (re.compile(r"\btheta\b"), "θ"),
    (re.compile(r"\bphi\b"), "φ"),
    (re.compile(r"Φ0\b"), "Φ<sub>0</sub>"),
    (re.compile(r"\bPhi0\b"), "Φ<sub>0</sub>"),
    (re.compile(r"(?<![\d.])(\d)-(\d)(?![\d.])"), "\\1→\\2"),  # transition labels 0-1 -> 0→1 (not decimal ranges)
]
_TEXIFY = [
    (re.compile(r"\bE(J|C|L)max\b"), r"$E_{\1,\\max}$"),
    (re.compile(r"\bE([JCL])\b"), r"$E_{\1}$"),
    (re.compile(r"\bf(01|12|02|03)\b"), r"$f_{\1}$"),
    (re.compile(r"\bfr\b"), r"$f_r$"),
    (re.compile(r"\bf_rad0\b"), r"$f_{r,0}$"),
    (re.compile(r"\bt_c\b"), r"$t_c$"),
    (re.compile(r"\bx0\b"), r"$x_0$"),
    (re.compile(r"\bf0\b"), r"$f_0$"),
    (re.compile(r"\bdfs\b"), r"$\\Delta_{\\mathrm{fs}}$"),
    (re.compile(r"\balpha([12])\b"), r"$\\alpha_{\1}$"),
    (re.compile(r"\btheta\b"), r"$\\theta$"),
    (re.compile(r"\bphi\b"), r"$\\varphi$"),
    (re.compile(r"\bvolt_scale\b"), r"$\\kappa_V$"),
    (re.compile(r"\bfield_scale\b"), r"$\\kappa$"),
    (re.compile(r"\blever_arm\b"), "lever arm"),
    (re.compile(r"Φ0\b|\bPhi0\b"), r"$\\Phi_0$"),
    (re.compile(r"(?<![\d.])(\d)-(\d)(?![\d.])"), "\\1→\\2"),
]


def texify_prose(text: str) -> str:
    """Markdown variant of mathify: parameter tokens in plain prose become inline KaTeX ($E_{J,\\max}$); code spans are left alone."""
    parts = re.split(r"(`[^`]*`|\$[^$]*\$)", str(text or ""))
    for i in range(0, len(parts), 2):
        for rx, repl in _TEXIFY:
            parts[i] = rx.sub(repl, parts[i])
    return "".join(parts)


def pills_html(items, kind: str = "") -> str:
    """Pills whose text is already HTML (typeset parameter names)."""
    return "".join(f'<span class="cs-pill {kind}">{t}</span>' for t in items)


def mathify(escaped_text: str) -> str:
    """Turn ASCII parameter tokens inside already-escaped prose into typeset HTML: EJ -> E<sub>J</sub>, f01 -> f<sub>01</sub>."""
    out = escaped_text
    for rx, repl in _MATHIFY:
        out = rx.sub(repl, out)
    return out


def _tex_num(v, sig: int = 4) -> str:
    """A fitted number set in amber inside a KaTeX formula."""
    txt = f"{float(v):.{sig}g}"
    if "e" in txt:  # scientific notation -> \times 10^{}
        m, e = txt.split("e")
        txt = f"{m}\\times 10^{{{int(e)}}}"
    return r"\textcolor{#fab219}{(" + txt + ")}"  # parentheses: the fitted numbers read as substituted values, not coefficients


def hamiltonian_tex(family: str, params: dict, coupled: bool) -> str:
    """The identified model as LaTeX (rendered by KaTeX via st.latex) with the fitted numbers substituted in amber."""
    p = params
    n = _tex_num
    unit_tex = ""
    try:
        from circuitsleuth import models as _m
        unit_tex = r"\quad\textcolor{#c3c2b7}{[\mathrm{" + _m.PLATFORMS[_m.platform_of(family)].get("freq_unit", "GHz") + "}]}"
    except Exception:  # noqa: BLE001
        unit_tex = r"\quad\textcolor{#c3c2b7}{[\mathrm{GHz}]}"
    res = (r" + " + n(p["fr"]) + r"\,\hat a^{\dagger}\hat a + " + n(p["g"], 3) + r"\,\hat n\,(\hat a + \hat a^{\dagger})") if (coupled and "fr" in p and "g" in p) else ""
    if family == "transmon":
        return r"\hat H = 4\," + n(p["EC"], 3) + r"\,\hat n^{2} - " + n(p["EJ"]) + r"\,\cos\hat\varphi" + res + unit_tex
    if family == "tunable_transmon":
        return (r"\begin{aligned}\hat H &= 4\," + n(p["EC"], 3) + r"\,\hat n^{2} - E_J(\Phi)\,\cos\hat\varphi" + res + unit_tex + r"\\[2pt]"
                r"E_J(\Phi) &= " + n(p["EJmax"]) + r"\,\sqrt{\cos^{2}(\pi\Phi) + " + n(p["d"], 3) + r"^{2}\,\sin^{2}(\pi\Phi)}\end{aligned}")
    if family == "fluxonium":
        return (r"\hat H = 4\," + n(p["EC"], 3) + r"\,\hat n^{2} + \tfrac{1}{2}\," + n(p["EL"], 3) + r"\,\hat\varphi^{2} - " + n(p["EJ"])
                + r"\,\cos(\hat\varphi - 2\pi\Phi)" + res + unit_tex)
    if family == "nv_ensemble":
        return (r"\hat H = " + n(p["D"]) + r"\,\hat S_z^{2} + " + n(p["E"], 3) + r"\,(\hat S_x^{2} - \hat S_y^{2}) + \gamma_e\,\mathbf{B}\cdot\hat{\mathbf{S}}" + unit_tex)
    if family == "charge_qubit":
        return r"f = \sqrt{\varepsilon^{2} + (2\," + n(p["t_c"]) + r")^{2}}" + unit_tex
    if family == "trapped_ion":
        return (r"\begin{aligned}\omega_z &= " + n(p["volt_scale"]) + r"\,\sqrt{V - V_0},\qquad \omega_r = \sqrt{" + n(p["f_rad0"]) + r"^{2} - \omega_z^{2}/2}" + unit_tex + r"\\[2pt]"
                r"&\text{sidebands at } \pm\omega_z,\ \pm\omega_r,\ \pm 2\omega_z\end{aligned}")
    if family == "rydberg_atom":
        return (r"\begin{aligned}\Delta f &= f_0 \pm \tfrac{1}{2}\," + n(p["dfs"]) + r" - \tfrac{1}{2}\,\alpha E^{2},\qquad \alpha = "
                + n(p["alpha1"]) + r"\ \big|\ " + n(p["alpha2"]) + unit_tex + r"\\[2pt]E &= " + n(p["field_scale"]) + r"\,(V - V_0)\end{aligned}")
    return r"\text{" + str(family).replace("_", r"\_") + "}"


def hero_top_html(eyebrow_txt: str, title: str) -> str:
    """Verdict band, upper left: amber eyebrow and the display title (the Hamiltonian follows as KaTeX)."""
    return f'<div class="cs-eyebrow amber">{eyebrow_txt}</div><div class="cs-display">{esc(title_case(title))}</div>'


def hero_bottom_html(kpis: list[tuple[str, str]], explained: float | None) -> str:
    """Verdict band, lower left: the KPIs and the explained / unexplained bar."""
    k = "".join(kpi_html(v, l) for v, l in kpis)
    bar = ""
    if explained is not None:
        g = max(0.0, min(100.0, 100.0 * float(explained)))
        r = 100.0 - g
        bar = (f'<div class="cs-bar"><div class="g" style="width:{g:.1f}%"></div><div class="r" style="width:{r:.1f}%"></div></div>'
               f'<div class="cs-bar-ends"><span class="g">{g:.0f}% explained</span><span class="r">{r:.0f}% unexplained</span></div>')
    return f'<div class="cs-kpis">{k}</div>{bar}'
