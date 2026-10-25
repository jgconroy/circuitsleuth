"""Ask Ĥolmes: Claude answers a lab user's questions about the identified device by calling the physics
engine as tools. The model never produces a number itself: every quantity comes from a tool that evaluates
the fitted report (derived quantities, operating points, predicted transitions, uncertainties, experimental
design). The tool functions below are pure and deterministic over the report dict, return JSON-serialisable
dicts with units, and are unit-tested without the API. ``answer()`` runs the SDK tool-runner loop.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any

import numpy as np

from . import derived, design, fit, models
from .explain import _load_env

MODEL = "claude-opus-5-5"
MAX_TOKENS = 8000  # adaptive thinking shares this budget; answers are a few sentences

# --------------------------------------------------------------------------- helpers


def _py(obj: Any, sig: int = 5) -> Any:
    """Recursively convert numpy types to plain Python, round floats to `sig` significant figures, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _py(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v, sig) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_py(v, sig) for v in obj.tolist()]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        if not math.isfinite(v):
            return None
        if v == 0.0:
            return 0.0
        return float(f"{v:.{sig}g}")
    return obj


def _ok(report: dict) -> dict:
    if not isinstance(report, dict) or report.get("status") != "ok" or "best" not in report:
        raise ValueError("no successful identification in this report")
    return report["best"]


def _hyp(best: dict) -> fit.Hypothesis:
    return fit.Hypothesis(best["family"], bool(best.get("coupled", False)))


def _x_range(report: dict) -> tuple[float, float]:
    xg = report.get("x_grid")
    if xg:
        return float(min(xg)), float(max(xg))
    pts = np.asarray(report.get("points", []), float)
    if pts.size:
        return float(pts[:, 0].min()), float(pts[:, 0].max())
    return (-1.0, 1.0)


def _platform(best: dict) -> str:
    return best.get("platform") or models.platform_of(best["family"])


def _freq_unit(best: dict) -> str:
    return models.PLATFORMS[_platform(best)].get("freq_unit", "GHz")


def _band(report: dict) -> tuple[float, float] | None:
    b = report.get("band_GHz")
    return (float(b[0]), float(b[1])) if b else None


def _flux_quanta(best: dict, x: float) -> float | None:
    p = best["params"]
    if "period" in p and "offset" in p and p["period"]:
        phi = (x - p["offset"]) / p["period"]
        return float(phi - round(phi))  # wrapped to [-0.5, 0.5]; Φ = 0 and Φ = ±0.5 are the sweet spots
    return None


_DESIGN_CACHE: dict[str, dict] = {}


def _design_for(report: dict) -> dict | None:
    """design.design_next_measurements once per report (it is ~1 s); None when the report lacks the points."""
    if not (report.get("points") and report.get("inlier_mask") and report.get("lines")):
        return None
    key = json.dumps([report["best"].get("hypothesis"), report["best"].get("params"), len(report["points"])], sort_keys=True, default=str)
    if key not in _DESIGN_CACHE:
        if len(_DESIGN_CACHE) > 8:
            _DESIGN_CACHE.clear()
        _DESIGN_CACHE[key] = design.design_next_measurements(report, band=_band(report))
    return _DESIGN_CACHE[key]


def _agent_defaults() -> dict:
    try:  # lazy: agent imports the CNN proposer (torch); the app has it loaded already
        from . import agent

        cfg = agent.AgentConfig()
        return dict(coupled_margin=float(cfg.coupled_margin), sigma_calibration=float(cfg.sigma_calibration), coupled_trials=int(cfg.coupled_trials))
    except Exception:  # noqa: BLE001
        return dict(coupled_margin=0.05, sigma_calibration=1.8, coupled_trials=2)


# --------------------------------------------------------------------------- pure tool functions


def report_summary(report: dict) -> dict:
    """Best hypothesis, parameters ± σ with units, hold-out check, confidence, measured band and control window, platform."""
    best = _ok(report)
    fam = best["family"]
    unit = _freq_unit(best)
    params = {k: dict(value=v, sigma=best.get("sigma", {}).get(k), unit=models.param_unit(fam, k) or "dimensionless") for k, v in best["params"].items()}
    xr = _x_range(report)
    pl = _platform(best)
    out = dict(
        hypothesis=best["hypothesis"],
        family=fam,
        coupled_to_readout_resonator=bool(best.get("coupled", False)),
        platform=pl,
        platform_label=models.PLATFORMS[pl]["label"],
        frequency_unit=unit,
        control_axis=models.PLATFORMS[pl]["control_label"],
        control_window=dict(lo=xr[0], hi=xr[1], unit="control units"),
        measured_band=dict(lo=report.get("band_GHz", [None, None])[0], hi=report.get("band_GHz", [None, None])[1], unit=unit),
        parameters=params,
        parameters_pretty=best.get("params_pretty", {}),
        explained_fraction_of_control_dependent_features=best.get("explained_dynamic"),
        inlier_fraction=best.get("inlier_frac"),
        rms_of_inliers_MHz=best.get("rms_inliers_MHz"),
        holdout=best.get("holdout"),
        confidence=best.get("confidence"),
        note=best.get("note") or None,
        n_features=report.get("n_points"),
        n_tracks=report.get("n_tracks"),
        flat_lines=dict(values=report.get("flat_lines_GHz", []), unit=unit, meaning="control-independent lines (spurs / tones / uncoupled modes)"),
        runtime_s=report.get("runtime_s"),
        flux_axis=(dict(period=best["params"]["period"], offset=best["params"]["offset"], unit="control units per flux quantum / control value of Φ = 0",
                        formula="Φ/Φ0 = (control − offset) / period") if "period" in best["params"] else None),
    )
    return _py(out)


def derived_quantities(report: dict) -> dict:
    """Operating points, anharmonicity, flux sensitivity, sweet spots, dispersive shift, Tφ estimates, ... for the best hypothesis."""
    best = _ok(report)
    items = report.get("derived")
    if not items:
        items = derived.derived_quantities(best, _x_range(report))
    rows = [dict(name=d["name"], value=d["value"], unit=d.get("unit") or "dimensionless", note=d.get("note", "")) for d in items]
    return _py(dict(quantities=rows, n=len(rows), frequency_unit=_freq_unit(best)))


def _slope_01(hyp: fit.Hypothesis, params: dict, x: float, h: float) -> float | None:
    try:
        L = fit.predict_lines(hyp, params, np.array([x - h, x + h]))["0-1"]
        s = (float(L[1]) - float(L[0])) / (2 * h)
        return s if math.isfinite(s) else None
    except Exception:  # noqa: BLE001
        return None


def control_for_frequency(report: dict, target_GHz: float) -> dict:
    """Control values inside the measured window where the fitted 0-1 transition equals the target, with the local
    slope |∂f01/∂control| (and per Φ0 when a flux axis exists) so the least-sensitive bias can be chosen."""
    best = _ok(report)
    unit = _freq_unit(best)
    xr = _x_range(report)
    hyp = _hyp(best)
    target = float(target_GHz)
    if not models.FAMILIES[best["family"]]["flux_dependent"]:
        return _py(dict(target=target, unit=unit, solutions=[], note=f"the {best['hypothesis']} model has no control dependence: f01 is fixed; see derived_quantities"))
    sols = derived.control_for_frequency(best, target, xr)
    h = 1e-3 * (xr[1] - xr[0])
    p = best["params"]
    rows = []
    for x in sols:
        s = _slope_01(hyp, p, x, h)
        row = dict(control=x, unit="control units", slope_GHz_per_control_unit=(abs(s) if s is not None else None))
        if row["slope_GHz_per_control_unit"] is not None and unit != "GHz":
            row = dict(control=x, unit="control units", **{f"slope_{unit}_per_control_unit": abs(s)})
        phi = _flux_quanta(best, x)
        if phi is not None:
            row["flux_quanta"] = phi
            if s is not None:
                row["slope_GHz_per_Phi0"] = abs(s) * p["period"]
        rows.append(row)
    best_i, tie_note = None, None
    slopes = [r.get("slope_GHz_per_control_unit", r.get(f"slope_{unit}_per_control_unit")) for r in rows]
    if rows and all(v is not None for v in slopes):
        if len(slopes) > 1 and (max(slopes) - min(slopes)) <= 0.01 * max(slopes):
            tie_note = "all solutions have the same flux sensitivity to within 1 % (flux-symmetric points of the same |Φ|); none is preferable on sensitivity grounds, choose by convenience"
        else:
            best_i = int(np.argmin(slopes))
    note = None
    if not rows:
        try:
            f = fit.predict_lines(hyp, p, np.linspace(xr[0], xr[1], 401))["0-1"]
            f = f[np.isfinite(f)]
            if f.size:
                note = f"f01 stays within {float(f.min()):.4g}–{float(f.max()):.4g} {unit} over the measured control window; {target:g} {unit} is not reachable there"
        except Exception:  # noqa: BLE001
            note = "no solution in the measured control window"
    return _py(dict(target=target, unit=unit, control_window=dict(lo=xr[0], hi=xr[1]), solutions=rows, lowest_sensitivity_index=best_i, sensitivity_tie=tie_note,
                    sensitivity_meaning="first-order flux-noise dephasing is proportional to |∂f01/∂Φ|; it vanishes at a sweet spot (Φ = 0 or ½)", note=note))


def predict_transitions(report: dict, control_value: float) -> dict:
    """Frequencies of every transition of the best hypothesis at one control value, flagged inside/outside the measured band."""
    best = _ok(report)
    unit = _freq_unit(best)
    band = _band(report)
    lines = fit.predict_lines(_hyp(best), best["params"], np.array([float(control_value)]))
    rows = []
    for name, arr in lines.items():
        f = float(arr[0])
        if not math.isfinite(f):
            rows.append(dict(line=name, frequency=None, unit=unit, in_band=False, note="not defined here"))
            continue
        rows.append(dict(line=name, frequency=f, unit=unit, in_band=(band[0] <= f <= band[1]) if band else None))
    xr = _x_range(report)
    out = dict(control_value=float(control_value), unit_of_control="control units", inside_measured_window=bool(xr[0] <= control_value <= xr[1]),
               measured_band=dict(lo=band[0], hi=band[1], unit=unit) if band else None, transitions=rows,
               lines_in_band=[r["line"] for r in rows if r.get("in_band")],
               legend="0-1 qubit; 0-2/2ph two-photon; res readout resonator (dressed); NVk± the two ODMR lines of orientation k")
    phi = _flux_quanta(best, float(control_value))
    if phi is not None:
        out["flux_quanta"] = phi
    return _py(out)


def next_measurements(report: dict) -> dict:
    """What to measure next: the fit's suggestions (extend band, look here, constrain parameter, discriminate families) plus
    the Fisher-information design candidates (expected information gain in bits per sweep, which parameter each pins)."""
    best = _ok(report)
    unit = _freq_unit(best)
    sugg = []
    for n in report.get("next_measurements", []) or []:
        n2 = dict(n)
        if "f_lo" in n2 or "f" in n2:
            n2["frequency_unit"] = unit
        if "x_lo" in n2 or "x" in n2:
            n2["control_unit"] = "control units"
        sugg.append(n2)
    cands, corr_note, sigma_f = [], None, None
    d = report.get("design") if isinstance(report.get("design"), dict) and report["design"].get("candidates") else None
    try:
        d = d or _design_for(report)
    except Exception as e:  # noqa: BLE001
        corr_note = f"design unavailable: {type(e).__name__}"
        d = None
    if d:
        sigma_f = d.get("sigma_f")
        for i, c in enumerate(d.get("candidates", [])[:5], start=1):
            cands.append(dict(rank=i, line=c["line"], control_lo=c["x_lo"], control_hi=c["x_hi"], frequency_lo=c["f_lo"], frequency_hi=c["f_hi"], frequency_unit=unit,
                              expected_gain_bits=c["gain_bits"], gain_per_column_bits=c.get("gain_per_column_bits"), parameter_helped_most=c.get("best_parameter"),
                              variance_reduction=c.get("variance_reduction")))
        if d.get("discriminate"):
            sugg.append(dict(d["discriminate"], frequency_unit=unit))
    return _py(dict(fit_suggestions=sugg, design_candidates=cands, design_note=corr_note,
                    design_meaning="expected information gain ½·log det(F0+F_S)/det(F0) of sweeping that control range looking for that line; assumed line precision σ_f",
                    sigma_f=dict(value=sigma_f, unit=unit) if sigma_f is not None else None))


def hypotheses(report: dict) -> dict:
    """Ranked table of every hypothesis tried (score, track score, inlier fraction, RMS, hold-out), the CNN prior, and the
    acceptance rules for a readout-resonator model."""
    best = _ok(report)
    rows = sorted(report.get("hypotheses", []), key=lambda h: -float(h.get("score", -9)))
    bare = {h["family"]: h for h in rows if not h.get("coupled")}
    table = []
    for i, h in enumerate(rows, start=1):
        r = dict(rank=i, hypothesis=h["hypothesis"], family=h["family"], coupled=bool(h.get("coupled")), score=h.get("score"), track_score=h.get("track_score"),
                 inlier_fraction=h.get("inlier_frac"), rms_of_inliers_MHz=h.get("rms_inliers_MHz"), holdout=h.get("holdout"), is_best=(h["hypothesis"] == best["hypothesis"]),
                 params=h.get("params"), sigma=h.get("sigma"))
        if h.get("coupled"):
            b = bare.get(h["family"])
            if b is not None and h.get("score") is not None and b.get("score") is not None:
                r["score_gain_over_bare_family"] = float(h["score"]) - float(b["score"])
            g, sg = (h.get("params") or {}).get("g"), (h.get("sigma") or {}).get("g")
            if g is not None and sg:
                r["g_over_sigma_g"] = float(g) / float(sg)
        table.append(r)
    runner_up = next((r for r in table if not r["is_best"]), None)
    cfg = _agent_defaults()
    return _py(dict(
        best=best["hypothesis"],
        best_note=best.get("note") or None,
        confidence=best.get("confidence"),
        runner_up=runner_up["hypothesis"] if runner_up else None,
        score_gap_best_minus_runner_up=(float(table[0]["score"]) - float(runner_up["score"])) if runner_up and table[0].get("score") is not None and runner_up.get("score") is not None else None,
        table=table,
        cnn_prior=report.get("proposals", []),
        scoring="score = explained track weight − missing-line penalty − RMS precision penalty; flat (control-independent) tracks count half; spurs removed before fitting control-dependent circuit models",
        resonator_acceptance_rules=[f"a readout-resonator model is tried only for the top-{cfg['coupled_trials']} bare families (other families never get a '+ readout resonator' row)",
                                    f"score gain over the bare circuit > {cfg['coupled_margin']:g}", "coupling g > 2 σ_g", "gain ≥ 0.02 on control-dependent features alone (a flat extra line is not evidence of coupling)",
                                    "resonator frequency seeded from the cluster of unexplained points at an avoided crossing"],
    ))


def unexplained_features(report: dict) -> dict:
    """Detected features the best model does not explain, with the fit's interpretation, plus the control-independent lines."""
    best = _ok(report)
    unit = _freq_unit(best)
    rows = []
    for u in report.get("unexplained", []) or []:
        rows.append(dict(frequency_lo=u.get("f_lo"), frequency_hi=u.get("f_hi"), frequency_mean=u.get("f_mean"), frequency_unit=unit, control_lo=u.get("x_lo"), control_hi=u.get("x_hi"),
                         control_independent=bool(u.get("flat")), n_points=u.get("n_points"), weight=u.get("weight"), unexplained_fraction=u.get("unexplained_frac"), interpretation=u.get("interpretation")))
    return _py(dict(unexplained=rows, n=len(rows), flat_lines=dict(values=report.get("flat_lines_GHz", []), unit=unit),
                    explained_fraction_of_control_dependent_features=best.get("explained_dynamic")))


def parameter_uncertainties(report: dict) -> dict:
    """σ, relative σ and hold-out shift per parameter, the σ calibration, and the parameter correlation matrix (degeneracies such as EJ–EC)."""
    best = _ok(report)
    fam = best["family"]
    rows = []
    shift = (best.get("holdout") or {}).get("param_shift_rel", {})
    for k, v in best["params"].items():
        s = best.get("sigma", {}).get(k)
        rows.append(dict(parameter=k, value=v, sigma=s, relative_sigma=(abs(float(s) / float(v)) if s is not None and v else None), unit=models.param_unit(fam, k) or "dimensionless",
                         holdout_relative_shift=shift.get(k)))
    names, corr = None, None
    d = report.get("design") if isinstance(report.get("design"), dict) and report["design"].get("correlation") else None
    try:
        d = d or _design_for(report)
    except Exception:  # noqa: BLE001
        d = None
    if d and d.get("correlation"):
        names, corr = d["correlation"]
    strong = []
    if names and corr is not None:
        C = np.asarray(corr, float)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                if abs(C[i, j]) > 0.8:
                    strong.append(dict(a=names[i], b=names[j], correlation=float(C[i, j])))
        strong.sort(key=lambda r: -abs(r["correlation"]))
    cfg = _agent_defaults()
    poorly = [r["parameter"] for r in rows if r["relative_sigma"] is not None and r["relative_sigma"] > 0.05]
    return _py(dict(parameters=rows, poorly_determined_over_5_percent=poorly,
                    sigma_meaning=f"1σ from the Laplace (Jacobian) covariance at the optimum × {cfg['sigma_calibration']:g} (synthetic coverage calibration); log-parameterised energies",
                    correlation=dict(names=names, matrix=corr) if names else None, strong_correlations=strong,
                    correlation_meaning="|ρ| near 1 means the data constrain a combination (e.g. the EJ–EC valley along constant f01), not each parameter separately"))


def scqubits_code(report: dict) -> dict:
    """A scqubits (or plain-formula) snippet reproducing the fitted Hamiltonian."""
    best = _ok(report)
    code = report.get("scqubits_code") or derived.scqubits_code(best)
    return dict(language="python", code=code, note="energies in GHz; flux in units of Φ0 where applicable")


TOOLS: dict[str, dict] = {
    "report_summary": dict(fn=report_summary, params=[], description="best hypothesis, parameters ± σ, hold-out, confidence, band, platform"),
    "derived_quantities": dict(fn=derived_quantities, params=[], description="operating points, anharmonicity, flux sensitivity, sweet spots, χ, Tφ estimates"),
    "control_for_frequency": dict(fn=control_for_frequency, params=["target_GHz"], description="control values where f01 equals a target, with local flux sensitivity"),
    "predict_transitions": dict(fn=predict_transitions, params=["control_value"], description="all transition frequencies at one control value"),
    "next_measurements": dict(fn=next_measurements, params=[], description="fit suggestions plus Fisher-information design candidates"),
    "hypotheses": dict(fn=hypotheses, params=[], description="ranked hypothesis table, CNN prior, resonator acceptance rules"),
    "unexplained_features": dict(fn=unexplained_features, params=[], description="features the model does not explain"),
    "parameter_uncertainties": dict(fn=parameter_uncertainties, params=[], description="σ, relative σ, hold-out shifts, correlation matrix"),
    "scqubits_code": dict(fn=scqubits_code, params=[], description="scqubits snippet for the fitted Hamiltonian"),
}


def run_tool(name: str, report: dict, **kwargs) -> dict:
    """Call a registered tool by name (used by the UI and tests)."""
    return TOOLS[name]["fn"](report, **kwargs)


# --------------------------------------------------------------------------- Claude tools (closures over the report)


def make_tools(report: dict) -> list:
    """``@beta_tool`` wrappers bound to one report. Each returns a JSON string; exceptions become is_error tool results."""
    from anthropic import beta_tool

    def dumps(d: dict) -> str:
        return json.dumps(d, ensure_ascii=False, default=str)

    @beta_tool
    def report_summary_tool() -> str:
        """Best hypothesis, every parameter with ± σ and units, hold-out check, confidence, measured band and control window, platform and frequency unit."""
        return dumps(report_summary(report))

    @beta_tool
    def derived_quantities_tool() -> str:
        """Derived quantities of the fitted model: f01 at the sweet spots, anharmonicity, EJ/EC, tuning range, max flux sensitivity, sweet-spot control values, 1/f flux-noise Tφ estimates, qubit–resonator detuning, g/Δ, dispersive shift χ, Purcell factor (circuits); platform-specific quantities otherwise."""
        return dumps(derived_quantities(report))

    @beta_tool
    def control_for_frequency_tool(target_GHz: float) -> str:
        """Control (bias) values inside the measured window where the fitted 0-1 transition equals a target frequency, each with its flux in Φ0 and local slope |∂f01/∂control| and |∂f01/∂Φ| (flux sensitivity). Use it for "where do I bias for X GHz" questions.

        Args:
            target_GHz: Target 0-1 frequency in the report's frequency unit (GHz for circuits, NV centres and quantum dots; MHz for ions and Rydberg atoms).
        """
        return dumps(control_for_frequency(report, target_GHz))

    @beta_tool
    def predict_transitions_tool(control_value: float) -> str:
        """Every transition frequency predicted by the best hypothesis at one control value, flagged inside/outside the measured band, with the flux in Φ0 when a flux axis exists.

        Args:
            control_value: Control-axis value in control units (the x axis of the map).
        """
        return dumps(predict_transitions(report, control_value))

    @beta_tool
    def next_measurements_tool() -> str:
        """What to measure next: the fit's suggestions (extend the band, look here for a predicted line, constrain a parameter, a single column that discriminates the families) and the Fisher-information design candidates ranked by expected information gain in bits, with the parameter each sweep pins down."""
        return dumps(next_measurements(report))

    @beta_tool
    def hypotheses_tool() -> str:
        """Ranked table of all hypotheses tried (score, track score, inlier fraction, RMS, hold-out, parameters), the CNN prior, the score gap to the runner-up, and the rules under which a readout-resonator model is accepted. Use it for "why did model X win / was accepted" questions."""
        return dumps(hypotheses(report))

    @beta_tool
    def unexplained_features_tool() -> str:
        """Detected spectral features the best model does not explain, with frequency and control ranges, whether they are control-independent, and the fit's interpretation (spur, pump tone, neighbouring mode, ...)."""
        return dumps(unexplained_features(report))

    @beta_tool
    def parameter_uncertainties_tool() -> str:
        """Per-parameter σ, relative σ, hold-out shift, which parameters are poorly determined, how σ was calibrated, and the parameter correlation matrix with the strongest correlations (degeneracies)."""
        return dumps(parameter_uncertainties(report))

    @beta_tool
    def scqubits_code_tool() -> str:
        """Python/scqubits snippet that reproduces the fitted Hamiltonian."""
        return dumps(scqubits_code(report))

    return [report_summary_tool, derived_quantities_tool, control_for_frequency_tool, predict_transitions_tool, next_measurements_tool, hypotheses_tool,
            unexplained_features_tool, parameter_uncertainties_tool, scqubits_code_tool]


def tool_base_name(name: str) -> str:
    """'control_for_frequency_tool' -> 'control_for_frequency' (the TOOLS registry key)."""
    return name[:-5] if name.endswith("_tool") else name


# --------------------------------------------------------------------------- the chat loop

SYSTEM = """You are the lab assistant of Ĥolmes, a tool that identified a quantum device's Hamiltonian from a spectroscopy map. One fitted report is loaded; you can only see it through the tools.

Rules:
- Every number you state must come from a tool result in this conversation. Never compute, extrapolate, convert or recall a value yourself. Quote values rounded to 3–4 significant figures and always with the unit the tool gave (GHz, MHz, control units, Φ0, bits, µs ...).
- Pick the tool that answers the question directly: control_for_frequency for "where do I bias for X", predict_transitions for "what is at control value X", derived_quantities for sweet spots / sensitivity / χ / anharmonicity, hypotheses for "why did model X win", parameter_uncertainties for error bars and degeneracies, next_measurements for what to measure next, unexplained_features for leftover lines, scqubits_code for code. Call report_summary when you need orientation. Several tools may be called in one turn.
- Be concise: answer in ONE paragraph of 3–6 plain sentences, including "why" and "explain" questions. Do not use bullet points, numbered lists, headings or tables unless the user explicitly asks for a list or a table. State the practical conclusion first, then the two or three numbers that support it; leave the rest out.
- Pass on flags from tool results (outside the band, NOT dispersive, poorly determined, no solution in the window, rejected: ...). When a result's note says something is unreachable, say so.
- If the question needs something the report does not contain (T1, T2 echo, temperature, gate fidelity, κ of the resonator, materials ...), say plainly that the spectroscopy model does not cover it and name what the tools do cover. Never invent values.
"""


def available() -> bool:
    """True when an API key is configured (same mechanism as explain.py: .env or environment)."""
    _load_env()
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _history_messages(history) -> list[dict]:
    msgs = []
    for t in history or []:
        role, text = t.get("role"), (t.get("text") or t.get("content") or "")
        if role in ("user", "assistant") and isinstance(text, str) and text.strip():
            msgs.append({"role": role, "content": text})
    while msgs and msgs[0]["role"] != "user":  # the first message must be a user turn
        msgs.pop(0)
    return msgs


def _unavailable(reason: str, error: str) -> dict:
    return dict(text=f"(assistant unavailable: {reason})", tool_calls=[], usage={}, error=error)


def answer(question: str, report: dict, history=None, model: str = MODEL, max_turns: int = 8, effort: str = "low") -> dict:
    """Answer one question with Claude + the physics tools. Returns dict(text, tool_calls=[{name, input, output_preview, is_error}], usage, ...).
    Never raises: API problems come back as dict(text="(assistant unavailable: ...)", error=...)."""
    if not available():
        return _unavailable("no ANTHROPIC_API_KEY in .env or the environment", "no_api_key")
    try:
        _ok(report)
    except ValueError as e:
        return _unavailable(str(e), "no_report")
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        return _unavailable("anthropic SDK not installed", f"ImportError: {e}")
    calls: list[dict] = []
    usage = dict(input_tokens=0, output_tokens=0, cache_read_input_tokens=0, cache_creation_input_tokens=0)
    try:
        client = anthropic.Anthropic()
        runner = client.beta.messages.tool_runner(
            model=model,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            tools=make_tools(report),
            messages=_history_messages(history) + [{"role": "user", "content": question}],
            max_iterations=max(1, int(max_turns)),
            thinking={"type": "adaptive"},
            output_config={"effort": effort},
            cache_control={"type": "ephemeral"},  # system + tool definitions are the stable prefix across turns
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",  # a safety-classifier decline is re-run on a fallback model inside the same call
        )
        last = None
        for message in runner:
            last = message
            u = getattr(message, "usage", None)
            for k in usage:
                usage[k] += int(getattr(u, k, 0) or 0) if u is not None else 0
            tool_uses = [b for b in message.content if b.type == "tool_use"]
            if not tool_uses:
                continue
            resp = runner.generate_tool_call_response()  # cached: the runner reuses it, tools run once
            results = {r.get("tool_use_id"): r for r in (resp or {}).get("content", []) if isinstance(r, dict)}
            for tu in tool_uses:
                r = results.get(tu.id, {})
                content = r.get("content")
                preview = content if isinstance(content, str) else json.dumps(content, default=str)
                calls.append(dict(name=tool_base_name(tu.name), input=dict(tu.input) if isinstance(tu.input, dict) else {}, output_preview=(preview or "")[:400], is_error=bool(r.get("is_error"))))
        if last is None:
            return dict(text="(assistant unavailable: empty response)", tool_calls=calls, usage=usage, error="empty")
        text = "\n".join(b.text for b in last.content if b.type == "text").strip()
        stop = getattr(last, "stop_reason", None)
        if stop == "refusal":
            det = getattr(last, "stop_details", None)
            text = text or f"(the assistant declined to answer{': ' + str(getattr(det, 'explanation', '')) if det is not None else ''})"
        elif stop == "max_tokens" and not text:
            text = "(answer truncated: the token budget ran out)"
        elif stop == "tool_use" and not text:
            text = f"(the assistant used all {max_turns} tool turns without reaching an answer; try a narrower question)"
        return dict(text=text or "(no answer)", tool_calls=calls, usage=usage, model=getattr(last, "model", model), stop_reason=stop)
    except anthropic.AuthenticationError as e:
        return _unavailable("the API key was rejected", f"AuthenticationError: {str(e)[:160]}")
    except anthropic.RateLimitError as e:
        return _unavailable("rate limited, try again in a moment", f"RateLimitError: {str(e)[:160]}")
    except anthropic.APIStatusError as e:
        return _unavailable(f"API error {e.status_code}", f"{type(e).__name__}: {str(e)[:200]}")
    except anthropic.APIConnectionError as e:
        return _unavailable("no network connection to the API", f"APIConnectionError: {str(e)[:160]}")
    except Exception as e:  # noqa: BLE001  — the UI must never break
        return _unavailable(f"{type(e).__name__}: {str(e)[:160]}", f"{type(e).__name__}: {str(e)[:300]}")
