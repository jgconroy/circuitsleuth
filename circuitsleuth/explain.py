"""Plain-English explanation of a report. Claude when a key is available, a template otherwise.

The LLM never produces numbers: it receives the fitted report and is instructed to restate it. The
template path returns the same JSON shape so the UI does not care which one ran.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

MODEL = "claude-opus-5-5"


def _load_env():
    p = Path(__file__).resolve().parent.parent / ".env"
    if p.exists():
        for line in p.read_text().splitlines():
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip())


def trimmed(report: dict) -> dict:
    drop = {"lines", "x_grid", "inlier_mask", "points", "track_ids"}
    return {k: v for k, v in report.items() if k not in drop}


def template(report: dict) -> dict:
    if report.get("status") != "ok":
        return dict(summary=report.get("error", "identification failed"), caveats=[], next_steps=[], source="template")
    b = report["best"]
    pp = b.get("params_pretty", {})
    circuit = ", ".join(f"{k} = {v}" for k, v in pp.items() if k not in ("period", "offset"))
    cal = ""
    if "period" in pp:
        cal = f" The flux axis calibrates to one flux quantum per {pp['period']} of the control unit (offset {pp['offset']})."
    ho = b.get("holdout") or {}
    s = (f"Best explanation: a {b['hypothesis']} with {circuit}.{cal} "
         f"It accounts for {100*b['track_score']:.0f}% of the detected features"
         + (f" and {100*ho['inlier_frac']:.0f}% of features in flux columns held out of the fit (RMS {ho['rms_MHz']:.1f} MHz)." if ho else "."))
    caveats = []
    for u in report.get("unexplained", [])[:4]:
        caveats.append(f"Unexplained {'flat' if u['flat'] else 'flux-dependent'} feature at {u['f_lo']:.2f}-{u['f_hi']:.2f} GHz: {u['interpretation']}.")
    for n in report.get("next_measurements", []):
        if n["kind"] == "constrain_parameter":
            caveats.append(f"{n['parameter']} is only determined to ±{100*n['rel_sigma']:.0f}%: {n['why']}.")
    others = [h for h in report.get("hypotheses", []) if h["hypothesis"] != b["hypothesis"]]
    if others:
        alt = max(others, key=lambda h: h["track_score"])
        caveats.append(f"Runner-up hypothesis: {alt['hypothesis']} (score {alt['track_score']:.2f} vs {b['track_score']:.2f}).")
    steps = []
    for n in report.get("next_measurements", []):
        if n["kind"] == "look_here":
            steps.append(f"Sweep {n['f_lo']:.2f}-{n['f_hi']:.2f} GHz over control values {n['x_lo']:.3g} to {n['x_hi']:.3g}: {n['why']}.")
        elif n["kind"] == "extend_band":
            steps.append(f"Extend the band to {n['f_lo']:.2f}-{n['f_hi']:.2f} GHz: {n['why']}.")
    return dict(summary=s, caveats=caveats[:5], next_steps=steps[:5], source="template")


SYSTEM = """You are CircuitSleuth's explainer. You receive a JSON report produced by a physics fit of a
superconducting-qubit two-tone spectroscopy map, plus a template draft. Write for an experimentalist at
the fridge. Rules: never invent or round numbers beyond what the report contains; every number you quote
must appear in the report. If the report says a parameter is poorly determined, say so plainly. Keep the
summary under 120 words; give at most 5 caveats and 5 concrete next steps."""


def _schema():
    from pydantic import BaseModel

    class Explanation(BaseModel):
        summary: str
        caveats: list[str]
        next_steps: list[str]

    return Explanation


def explain(report: dict, use_llm: bool = True, model: str = MODEL) -> dict:
    base = template(report)
    if not use_llm:
        return base
    _load_env()
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return base
    try:
        import anthropic

        client = anthropic.Anthropic()
        resp = client.messages.parse(
            model=model,
            max_tokens=2000,
            system=SYSTEM,
            messages=[{"role": "user", "content": "Report:\n" + json.dumps(trimmed(report), default=str) + "\n\nTemplate draft you may improve on:\n" + json.dumps(base)}],
            output_format=_schema(),
        )
        if resp.stop_reason == "refusal" or resp.parsed_output is None:
            base["llm_error"] = f"stop_reason={resp.stop_reason}"
            return base
        out = resp.parsed_output.model_dump()
        out["source"] = model
        return out
    except Exception as e:  # any failure -> template, never block the UI
        base["llm_error"] = f"{type(e).__name__}: {str(e)[:120]}"
        return base
