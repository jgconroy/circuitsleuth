"""Ask Ĥolmes: a chat section in which Claude answers questions about the identified device by calling the
physics engine as tools (circuitsleuth.ask). History lives in st.session_state["ask_history"], keyed to the best
hypothesis (reset when it changes). The panel never raises: a failure degrades to a muted caption."""
from __future__ import annotations

import json

import streamlit as st

from ui import theme

SUGGESTIONS = {
    "circuit": ["Bias point for 5 GHz with the least flux noise?", "Why was the resonator model accepted?", "What should I measure next and why?"],
    "default": ["How confident is the identification, and what limits it?", "Which parameters are poorly determined, and what would pin them down?", "What should I measure next and why?"],
}
ALT_RESONATOR_Q = "Which alternative hypotheses were tested, and why did they lose?"


def _report_key(report: dict) -> str:
    best = (report or {}).get("best") or {}
    return json.dumps([best.get("hypothesis"), best.get("params")], sort_keys=True, default=str)


def _suggestions(report: dict, platform_key: str) -> list[str]:
    qs = list(SUGGESTIONS.get(platform_key, SUGGESTIONS["default"]))
    if platform_key == "circuit" and not (report.get("best") or {}).get("coupled"):
        qs[1] = ALT_RESONATOR_Q
    return qs


def _tools_caption(tool_calls) -> str:
    if not tool_calls:
        return "no tools called"
    parts = []
    for c in tool_calls:
        args = ", ".join(f"{k}={v:g}" if isinstance(v, (int, float)) else f"{k}={v}" for k, v in (c.get("input") or {}).items())
        parts.append(f"{c['name']}({args})" + (" ✗" if c.get("is_error") else ""))
    return "tools called: " + " · ".join(parts)


def _show_turn(turn: dict) -> None:
    with st.chat_message(turn["role"]):
        st.markdown(turn.get("text") or "")
        if turn["role"] == "assistant":
            theme.md(theme.caption(turn.get("tools_caption") or _tools_caption(turn.get("tool_calls"))))


def render(report: dict, platform_key: str) -> None:
    """Streamlit section; add ``ask_panel.render(rep, platform)`` after the Posterior check and before Details & export."""
    try:
        _render(report, platform_key)
    except Exception as e:  # noqa: BLE001  — never let the assistant break the results page
        try:
            theme.md(theme.caption(f"Ask Ĥolmes is unavailable right now ({type(e).__name__})."))
        except Exception:  # noqa: BLE001
            pass


def _render(report: dict, platform_key: str) -> None:
    from circuitsleuth import ask

    with st.container():  # inside a container st.chat_input renders inline instead of pinned to the page bottom
        theme.md(theme.section("Ask Ĥolmes", "Claude with the physics engine as tools",
                               "Every number in an answer comes from a tool call on the fitted model, never from the language model; the tools used are listed under each answer."))
        if not isinstance(report, dict) or report.get("status") != "ok" or "best" not in report:
            theme.md(theme.caption("Identify a device first."))
            return
        key = _report_key(report)
        if st.session_state.get("ask_key") != key:
            st.session_state["ask_key"] = key
            st.session_state["ask_history"] = []
            st.session_state.pop("ask_pending", None)
        hist = st.session_state.setdefault("ask_history", [])
        if not ask.available():
            theme.md(theme.caption("Set ANTHROPIC_API_KEY in .env to enable the assistant; the tools below still describe what it can answer."))
        cols = st.columns(3)
        for i, (c, q) in enumerate(zip(cols, _suggestions(report, platform_key))):
            if c.button(q, key=f"ask_sugg_{i}", width="stretch"):
                st.session_state["ask_pending"] = q
        box = st.container()
        with box:
            for turn in hist:
                _show_turn(turn)
        typed = st.chat_input("Ask about this device: bias points, sensitivities, why a model won, what to measure next …", key="ask_input")
        pending = typed or st.session_state.pop("ask_pending", None)
        if not pending:
            return
        with box:
            _show_turn(dict(role="user", text=pending))
            with st.chat_message("assistant"):
                with st.spinner("consulting the physics engine …"):
                    res = ask.answer(pending, report, history=[dict(role=t["role"], text=t.get("text", "")) for t in hist])
                text = res.get("text") or "(no answer)"
                st.markdown(text)
                cap = _tools_caption(res.get("tool_calls"))
                theme.md(theme.caption(cap))
        hist.append(dict(role="user", text=pending))
        hist.append(dict(role="assistant", text=text, tool_calls=res.get("tool_calls", []), tools_caption=cap, error=res.get("error"), usage=res.get("usage")))
