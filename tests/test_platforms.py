"""NV-centre and double-quantum-dot platforms: physics limits, synthetic rendering, recovery."""
import numpy as np

from circuitsleuth import agent, extract, fit, models, synth


def test_nv_known_limits():
    p = dict(D=2.87, E=0.005, theta=float(np.arccos(1 / np.sqrt(3))), phi=np.pi / 4)  # field along (1,1,1) = NV1
    L = models.nv_lines(p, np.array([0.0, 100.0]))
    assert np.allclose([L["NV1-"][0], L["NV1+"][0]], [2.865, 2.875], atol=1e-6)
    assert np.allclose([L["NV1-"][1], L["NV1+"][1]], [2.87 - 0.28025, 2.87 + 0.28025], atol=2e-4)
    # the other three orientations see the same axial projection (-1/3) and are degenerate
    assert np.allclose(L["NV2+"][1], L["NV3+"][1], atol=1e-9) and np.allclose(L["NV3+"][1], L["NV4+"][1], atol=1e-9)


def test_dqd_limits():
    L = models.dqd_lines(dict(t_c=3.0), np.array([0.0, 8.0]))
    assert np.allclose(L["0-1"], [6.0, 10.0]) and np.allclose(L["0-1/2ph"], [3.0, 5.0])


def test_synth_renders_other_platforms():
    cfg = synth.SynthConfig(nx=61, nf=300)
    for platform in ("nv", "dot"):
        rng = np.random.default_rng(3)
        t = synth.sample_truth(rng, cfg, platform=platform)
        r = synth.render(t, cfg, rng)
        assert r["image"].shape == (300, 61) and len(r["points"]) > 20, platform
        assert t["platform"] == platform


def _hidden(family, params, f_lo, f_hi, seed=0, noise=0.05):
    truth = dict(family=family, platform=models.platform_of(family), coupled=False, params=dict(params), period=1.0, offset=0.0, x_lo=-1.0, x_hi=1.0,
                 f_lo=f_lo, f_hi=f_hi, linewidth=0.006 if family == "nv_ensemble" else 0.03, noise=noise, two_photon=True, sign=1, spurs=[])
    cfg = synth.SynthConfig(nx=81, nf=500)
    r = synth.render(truth, cfg, np.random.default_rng(seed))
    return truth, r


def test_agent_recovers_nv_ensemble():
    t, r = _hidden("nv_ensemble", dict(D=2.870, E=0.004, theta=0.6, phi=0.3, field_scale=120.0, x0=0.1), 2.4, 3.35, seed=1)
    rep = agent.identify(r["image"], r["x"], r["f"], config=agent.AgentConfig(maxiter=40, use_classifier=False, families=("nv_ensemble",), try_coupled="never", holdout=True))
    assert rep["status"] == "ok" and rep["best"]["family"] == "nv_ensemble", rep.get("best")
    b = rep["best"]["params"]
    print("\nNV:", b, "truth", t["params"])
    assert abs(b["D"] - 2.870) < 0.003
    assert abs(b["E"] - 0.004) < 0.002
    # field direction up to the crystal's tetrahedral symmetry: compare the axial projections on the four NV axes
    assert np.allclose(b["field_scale"] * models.nv_projections(b["theta"], b["phi"]), 120.0 * models.nv_projections(0.6, 0.3), rtol=0.05)


def test_agent_recovers_charge_qubit():
    t, r = _hidden("charge_qubit", dict(t_c=2.5, lever_arm=9.0, x0=0.15), 1.0, 12.0, seed=2)
    rep = agent.identify(r["image"], r["x"], r["f"], config=agent.AgentConfig(maxiter=40, use_classifier=False, families=("charge_qubit",), try_coupled="never"))
    b = rep["best"]["params"]
    print("\nDQD:", b)
    assert abs(b["t_c"] - 2.5) / 2.5 < 0.03 and abs(b["lever_arm"] - 9.0) / 9.0 < 0.03 and abs(b["x0"] - 0.15) < 0.02
