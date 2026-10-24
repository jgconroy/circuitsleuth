"""One CSS block, the wordmark header and a few small HTML helpers (pills, status line, run log).

Palette: background #111214, surface #1a1a19, border #2a2a28, text #f0efec, muted #c3c2b7, accent #fab219.
"""
from __future__ import annotations

import html

import streamlit as st

FONT = 'Inter, "Helvetica Neue", Helvetica, Arial, sans-serif'

CSS = """<style>
#MainMenu, footer, [data-testid="stDecoration"], .stAppDeployButton { display: none !important; }
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.1rem; padding-bottom: 2.5rem; max-width: 1500px; }
section[data-testid="stSidebar"] .block-container { padding-top: 1.4rem; }
.stApp h1, .stApp h2, .stApp h3, .stApp p, .stApp li, .stApp label, .stApp summary, .stApp input, .stApp table,
[data-testid="stMetricValue"], [data-testid="stMetricLabel"] { font-family: Inter, "Helvetica Neue", Helvetica, Arial, sans-serif; }
.stApp h3 { font-size: 1.1rem; font-weight: 500; letter-spacing: -0.01em; color: #f0efec; padding-top: .6rem; padding-bottom: .2rem; }
.cs-wordmark { display: flex; align-items: baseline; gap: .9rem; flex-wrap: wrap; margin-bottom: .1rem; }
.cs-mark { display: inline-block; width: 11px; height: 11px; background: #fab219; margin-right: .55rem; transform: translateY(-1px); }
.cs-name { font-size: 1.85rem; font-weight: 600; letter-spacing: -0.03em; color: #f0efec; line-height: 1.1; }
.cs-tag { font-variant: small-caps; letter-spacing: .09em; font-size: .95rem; color: #c3c2b7; }
.cs-rule { border: 0; border-top: 1px solid #2a2a28; margin: .55rem 0 1rem; }
[data-testid="stMetric"] { border: 1px solid #2a2a28; border-radius: 6px; padding: .7rem .9rem .6rem; background: #1a1a19; box-shadow: none; }
[data-testid="stMetricLabel"] p { color: #c3c2b7; font-size: .8rem; letter-spacing: .02em; }
[data-testid="stMetricValue"] { font-size: 1.45rem; font-weight: 500; }
[data-testid="stMetric"] .stMarkdownEllipsis, [data-testid="stMetric"] .stMarkdownEllipsis p { white-space: normal; overflow: visible; text-overflow: clip; line-height: 1.25; }
[data-testid="stTable"] table { font-size: .9rem; font-variant-numeric: tabular-nums; }
[data-testid="stTable"] th { color: #c3c2b7 !important; font-weight: 500 !important; }
[data-testid="stTable"] th, [data-testid="stTable"] td { border-color: #2a2a28 !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-color: #2a2a28 !important; border-radius: 6px; }
[data-testid="stExpander"] details, [data-testid="stExpander"] > div { border-color: #2a2a28 !important; border-radius: 6px; }
[data-testid="stExpander"] summary p { font-weight: 500; letter-spacing: .01em; }
.cs-pill { display: inline-block; padding: 0 9px; border: 1px solid #2a2a28; border-radius: 999px; font-size: 12px; line-height: 19px;
  color: #c3c2b7; margin: 0 6px 4px 0; white-space: nowrap; font-variant-numeric: tabular-nums; }
.cs-pill.amber { color: #fab219; border-color: rgba(250, 178, 25, .45); }
.cs-pill.green { color: #6ccf6c; border-color: rgba(108, 207, 108, .4); }
.cs-pill.red { color: #e06060; border-color: rgba(224, 96, 96, .4); }
.cs-pill.blue { color: #86b6ef; border-color: rgba(134, 182, 239, .4); }
.cs-status { font-size: .92rem; color: #f0efec; line-height: 1.45; margin: .3rem 0 .5rem; }
.cs-status .cs-pill { margin-right: 8px; vertical-align: 1px; }
.cs-log { font-family: ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-size: 11.5px; line-height: 1.5; color: #c3c2b7;
  border-top: 1px solid #2a2a28; padding-top: .45rem; margin-top: .5rem; white-space: pre-wrap; }
.cs-caption { font-size: .8rem; color: #c3c2b7; margin: -.3rem 0 .4rem; }
.cs-note { font-size: .9rem; color: #c3c2b7; border-left: 2px solid #fab219; padding: .15rem .7rem; margin: .2rem 0 .8rem; }
.cs-table { width: 100%; border-collapse: collapse; font-size: .9rem; font-variant-numeric: tabular-nums; margin: .2rem 0 .9rem; }
.cs-table th { text-align: left; color: #c3c2b7; font-weight: 500; padding: .35rem .55rem; border-bottom: 1px solid #2a2a28; }
.cs-table td { padding: .32rem .55rem; border-bottom: 1px solid #2a2a28; vertical-align: top; color: #f0efec; }
.cs-table td.muted { color: #c3c2b7; font-size: .82rem; }
.cs-table td.num { white-space: nowrap; }
.cs-param { font-size: .82rem; line-height: 1.15; color: #f0efec; overflow-wrap: anywhere; font-variant-numeric: tabular-nums; }
</style>"""

HEADER = """<div class="cs-wordmark"><span class="cs-name"><span class="cs-mark"></span>CircuitSleuth</span>
<span class="cs-tag">Identify the circuit behind a spectroscopy map</span></div><hr class="cs-rule">"""


def inject() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def header() -> None:
    st.markdown(HEADER, unsafe_allow_html=True)


def rule() -> None:
    st.markdown('<hr class="cs-rule">', unsafe_allow_html=True)


def pill(text: str, kind: str = "") -> str:
    """A small status label. kind in {"", "amber", "green", "red", "blue"}."""
    return f'<span class="cs-pill {kind}">{html.escape(str(text))}</span>'


def pills(items, kind: str = "") -> str:
    return "".join(pill(t, kind) for t in items)


def status_html(label: str, text: str, kind: str = "") -> str:
    return f'<div class="cs-status">{pill(label, kind)}{html.escape(text)}</div>'


def log_html(lines, keep: int = 7) -> str:
    body = "\n".join(html.escape(l) for l in list(lines)[-keep:])
    return f'<div class="cs-log">{body}</div>'


def caption(text: str) -> str:
    return f'<div class="cs-caption">{html.escape(text)}</div>'


def note(text: str) -> str:
    return f'<div class="cs-note">{html.escape(text)}</div>'


def table_html(columns, rows, muted=(), numeric=()) -> str:
    """Compact HTML table in the app's palette. `muted` and `numeric` list column indices drawn muted / kept on one line."""
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in columns)
    body = []
    for r in rows:
        cells = []
        for i, c in enumerate(r):
            cls = " ".join(k for k, on in (("muted", i in muted), ("num", i in numeric)) if on)
            cells.append(f'<td class="{cls}">{html.escape(str(c))}</td>')
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f'<table class="cs-table"><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table>'
