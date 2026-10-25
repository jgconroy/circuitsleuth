"""Unit tests for the Ask Ĥolmes tool functions (circuitsleuth.ask) on the saved demo A report. No API calls."""
import json

import numpy as np
import pytest

from circuitsleuth import ask

REPORT = "artifacts/demo_A_report.json"


@pytest.fixture(scope="module")
def rep():
    return json.load(open(REPORT))


def _json_ok(d):
    s = json.dumps(d, ensure_ascii=False)
    assert "NaN" not in s and "Infinity" not in s
    return s


def test_control_for_frequency_six_GHz(rep):
    out = ask.control_for_frequency(rep, 6.0)
    _json_ok(out)
    assert out["unit"] == "GHz" and out["target"] == 6.0
    assert len(out["solutions"]) >= 2
    for s in out["solutions"]:
        assert -1.0 <= s["control"] <= 1.0
        assert s["slope_GHz_per_control_unit"] > 0 and s["slope_GHz_per_Phi0"] > s["slope_GHz_per_control_unit"]  # period 1.3 > 1
        assert -0.5 <= s["flux_quanta"] <= 0.5
    if out["lowest_sensitivity_index"] is None:  # mirror-image flux points: identical |∂f01/∂Φ|, reported as a tie, never a fake winner
        assert out["sensitivity_tie"] and len({round(s["slope_GHz_per_Phi0"], 3) for s in out["solutions"]}) == 1
    else:
        assert 0 <= out["lowest_sensitivity_index"] < len(out["solutions"]) and out["sensitivity_tie"] is None
    unreachable = ask.control_for_frequency(rep, 12.0)
    assert unreachable["solutions"] == [] and "not reachable" in unreachable["note"]


def test_predict_transitions_at_control_0p2(rep):
    out = ask.predict_transitions(rep, 0.2)
    _json_ok(out)
    f01 = next(r for r in out["transitions"] if r["line"] == "0-1")
    assert abs(f01["frequency"] - 7.43) < 0.06 and f01["in_band"] is True  # dressed 0-1 at Φ ≈ 0 (bare 7.431 GHz pushed by the 7 GHz resonator)
    xg = np.array(rep["x_grid"])
    i = int(np.argmin(np.abs(xg - 0.2)))
    assert abs(f01["frequency"] - rep["lines"]["0-1"][i]) < 2e-3  # same numbers as the report's own fitted curve
    assert abs(out["flux_quanta"]) < 1e-3  # control 0.2 is the fitted offset: Φ = 0
    names = {r["line"] for r in out["transitions"]}
    assert {"0-1", "0-2", "0-2/2ph", "1-2", "res"} <= names
    assert "0-2" not in out["lines_in_band"] and "res" in out["lines_in_band"]
    assert out["inside_measured_window"] is True


def test_hypotheses_lists_five_rows(rep):
    out = ask.hypotheses(rep)
    _json_ok(out)
    assert len(out["table"]) == 5
    assert out["table"][0]["rank"] == 1 and out["table"][0]["is_best"] is True
    assert out["table"][0]["hypothesis"] == "tunable transmon + readout resonator" == out["best"]
    assert out["table"][0]["score_gain_over_bare_family"] > 0.05  # the acceptance margin
    assert out["table"][0]["g_over_sigma_g"] > 2
    assert out["runner_up"] == "tunable transmon" and out["score_gap_best_minus_runner_up"] > 0.1
    assert any("0.05" in r for r in out["resonator_acceptance_rules"])
    assert [r["score"] for r in out["table"]] == sorted((r["score"] for r in out["table"]), reverse=True)


def test_derived_quantities_non_empty(rep):
    out = ask.derived_quantities(rep)
    _json_ok(out)
    assert out["n"] > 10 and out["frequency_unit"] == "GHz"
    d = {q["name"]: q for q in out["quantities"]}
    assert abs(d["f01 at Φ = 0 (upper sweet spot)"]["value"] - 7.43) < 0.05 and d["f01 at Φ = 0 (upper sweet spot)"]["unit"] == "GHz"
    assert all(q["unit"] for q in out["quantities"])  # dimensionless is spelled out, never empty
    # the same result when the report carries no precomputed table
    bare = {k: v for k, v in rep.items() if k != "derived"}
    assert ask.derived_quantities(bare)["n"] == out["n"]


def test_scqubits_code(rep):
    out = ask.scqubits_code(rep)
    assert "scq." in out["code"] and "TunableTransmon" in out["code"] and out["language"] == "python"


def test_summary_uncertainties_next_and_unexplained(rep):
    s = ask.report_summary(rep)
    _json_ok(s)
    assert s["hypothesis"] == "tunable transmon + readout resonator" and s["platform"] == "circuit" and s["frequency_unit"] == "GHz"
    assert s["parameters"]["EJmax"]["unit"] == "GHz" and s["parameters"]["d"]["unit"] == "dimensionless"
    assert s["holdout"]["n_test_points"] == 19 and s["confidence"]["level"] == "high"
    assert s["flux_axis"]["period"] == pytest.approx(1.2998, abs=1e-3)

    u = ask.parameter_uncertainties(rep)
    _json_ok(u)
    assert {r["parameter"] for r in u["parameters"]} == set(rep["best"]["params"])
    assert all(r["relative_sigma"] is not None and r["relative_sigma"] < 0.05 for r in u["parameters"])
    assert u["poorly_determined_over_5_percent"] == []
    strong = {(r["a"], r["b"]): r["correlation"] for r in u["strong_correlations"]}
    assert strong[("EJmax", "EC")] < -0.95  # the EJ–EC valley
    assert "1.8" in u["sigma_meaning"]

    n = ask.next_measurements(rep)
    _json_ok(n)
    assert len(n["fit_suggestions"]) == 3 and n["fit_suggestions"][0]["kind"] == "extend_band"
    assert n["design_candidates"] and n["design_candidates"][0]["expected_gain_bits"] > 0.5
    assert n["design_candidates"][0]["parameter_helped_most"] in rep["best"]["params"]

    x = ask.unexplained_features(rep)
    _json_ok(x)
    assert x["n"] == 1 and x["unexplained"][0]["control_independent"] is True
    assert abs(x["unexplained"][0]["frequency_mean"] - 5.10) < 0.02


def test_registry_and_claude_tools(rep):
    assert set(ask.TOOLS) == {"report_summary", "derived_quantities", "control_for_frequency", "predict_transitions", "next_measurements", "hypotheses",
                              "unexplained_features", "parameter_uncertainties", "scqubits_code"}
    tools = ask.make_tools(rep)
    names = [ask.tool_base_name(t.name) for t in tools]
    assert names == list(ask.TOOLS)
    by = {ask.tool_base_name(t.name): t for t in tools}
    schema = by["control_for_frequency"].to_dict()["input_schema"]
    assert schema["properties"]["target_GHz"]["type"] == "number" and schema["required"] == ["target_GHz"]
    assert by["report_summary"].to_dict()["input_schema"]["properties"] == {}
    res = json.loads(by["control_for_frequency"].call({"target_GHz": 6.0}))
    assert len(res["solutions"]) >= 2
    res = json.loads(by["predict_transitions"].call({"control_value": 0.2}))
    assert any(r["line"] == "0-1" for r in res["transitions"])
    for t in tools:
        assert t.to_dict()["description"]


def test_answer_never_raises_without_key(rep, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")  # _load_env uses setdefault, so the .env key does not override this
    out = ask.answer("Where do I bias for 5 GHz?", rep)
    assert out["text"].startswith("(assistant unavailable") and out["error"] == "no_api_key" and out["tool_calls"] == []
    bad = ask.answer("anything", {"status": "error", "error": "x"})
    assert bad["text"].startswith("(assistant unavailable") and bad["error"] in ("no_api_key", "no_report")
