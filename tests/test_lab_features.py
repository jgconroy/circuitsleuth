import json

import numpy as np

from circuitsleuth import agent, derived, fit, models, synth


def test_crossing_label_and_enrichment():
    cfg = synth.SynthConfig()
    rng = np.random.default_rng(0)
    n_cross = sum(synth.crossing_visible(synth.sample_truth(rng, cfg, coupled=True)) for _ in range(60))
    assert n_cross >= 20  # enrichment makes crossings common enough to learn
    t = dict(family="tunable_transmon", platform="circuit", coupled=True, params=dict(EJmax=25.0, EC=0.3, d=0.3, fr=7.0, g=0.1), period=1.3, offset=0.2, x_lo=-1, x_hi=1, f_lo=3, f_hi=9)
    assert synth.crossing_visible(t) is True
    t["params"]["fr"] = 8.5
    assert synth.crossing_visible(t) is False


def test_derived_quantities_and_operating_points():
    rep = json.load(open("artifacts/demo_A_report.json"))
    xr = (min(rep["x_grid"]), max(rep["x_grid"]))
    d = {q["name"]: q for q in derived.derived_quantities(rep["best"], xr)}
    assert abs(d["f01 at Φ = 0 (upper sweet spot)"]["value"] - 7.43) < 0.05
    assert d["anharmonicity at Φ = 0"]["value"] < -250  # MHz, transmon-like
    assert any(k.startswith("sweet spot") for k in d)
    sols = derived.control_for_frequency(rep["best"], 5.0, xr)
    assert len(sols) >= 2
    code = derived.scqubits_code(rep["best"])
    assert "TunableTransmon" in code and "Oscillator" in code
    csv = derived.curves_csv(rep)
    assert csv.startswith("control,") and csv.count("\n") > 50


def test_user_bounds_fix_a_parameter():
    cfg = agent.AgentConfig(param_bounds={"fr": 7.0, "EC": (0.25, 0.35)})
    b = agent._apply_user_bounds(fit.default_bounds(fit.Hypothesis("tunable_transmon", True), np.array([-1.0, 1.0])), cfg)
    assert abs(b["fr"][0] - 7.0) < 1e-3 and abs(b["fr"][1] - 7.0) < 1e-3 and b["EC"] == (0.25, 0.35)


def test_experimental_design_points_at_the_degeneracy_breaker():
    rep = json.load(open("artifacts/demo_A_report.json"))
    from circuitsleuth import design
    d = design.design_next_measurements(rep, band=tuple(rep["band_GHz"]))
    names, corr = d["correlation"]
    corr = np.array(corr)
    i, j = names.index("EJmax"), names.index("EC")
    assert abs(corr[i, j]) > 0.95  # the EJ-EC valley shows up as a near-perfect (anti)correlation in log space
    assert d["candidates"], "no candidate sweeps"
    assert d["candidates"][0]["gain_bits"] > 0.5
    assert all(c["x_hi"] >= c["x_lo"] and c["f_hi"] >= c["f_lo"] for c in d["candidates"])
