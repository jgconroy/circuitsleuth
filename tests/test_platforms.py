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
                 f_lo=f_lo, f_hi=f_hi, linewidth={"nv_ensemble": 0.006, "trapped_ion": 0.03, "rydberg_atom": 0.5}.get(family, 0.03), noise=noise, two_photon=True, sign=1, spurs=[])
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


def test_ion_and_rydberg_limits():
    L = models.ion_lines(dict(f_rad0=3.0), np.array([0.0, 1.0]))
    assert L["axial +"][1] == 1.0 and abs(L["radial +"][1] - np.sqrt(9 - 0.5)) < 1e-12 and L["carrier"][1] == 0.0 and L["2nd axial -"][1] == -2.0
    wz = models.control_from_x("trapped_ion", dict(volt_scale=2.0, x0=-1.5), np.array([-1.5, -1.0, 0.5]))
    assert np.allclose(wz, [0.0, 2.0 * np.sqrt(0.5), 2.0 * np.sqrt(2.0)])
    R = models.rydberg_lines(dict(alpha1=2.0, alpha2=4.0, dfs=10.0, f0=1.0), np.array([0.0, 2.0]))
    assert np.allclose(R["|mj| = 1/2"], [6.0, 2.0]) and np.allclose(R["|mj| = 3/2"], [-4.0, -12.0])


def test_synth_renders_ion_and_atom():
    for platform in ("ion", "atom"):
        rng = np.random.default_rng(2)
        t = synth.sample_truth(rng, synth.SynthConfig(nx=61, nf=300), platform=platform)
        r = synth.render(t, synth.SynthConfig(nx=61, nf=300), rng)
        assert r["image"].shape == (300, 61) and len(r["points"]) > 30


def test_agent_recovers_trapped_ion():
    t, r = _hidden("trapped_ion", dict(f_rad0=3.2, volt_scale=1.8, x0=-1.3), -6.5, 6.5, seed=4, noise=0.05)
    rep = agent.identify(r["image"], r["x"], r["f"], config=agent.AgentConfig(maxiter=40, use_classifier=False, families=("trapped_ion",), try_coupled="never"))
    b = rep["best"]["params"]
    print("\nION:", b)
    assert abs(b["f_rad0"] - 3.2) / 3.2 < 0.03 and abs(b["volt_scale"] - 1.8) / 1.8 < 0.03 and abs(b["x0"] + 1.3) < 0.05


def test_agent_recovers_rydberg_atom():
    t, r = _hidden("rydberg_atom", dict(alpha1=2.5, alpha2=4.0, dfs=12.0, f0=0.0, field_scale=2.5, x0=0.12), -80.0, 20.0, seed=5, noise=0.05)
    rep = agent.identify(r["image"], r["x"], r["f"], config=agent.AgentConfig(maxiter=40, use_classifier=False, families=("rydberg_atom",), try_coupled="never"))
    b = rep["best"]["params"]
    print("\nATOM:", b)
    # alpha and field_scale are degenerate as alpha * field_scale^2: compare the curvature in control units
    assert abs(b["alpha1"] * b["field_scale"] ** 2 - 2.5 * 2.5**2) / (2.5 * 2.5**2) < 0.03
    assert abs(b["alpha2"] * b["field_scale"] ** 2 - 4.0 * 2.5**2) / (4.0 * 2.5**2) < 0.03
    assert abs(b["dfs"] - 12.0) < 0.5 and abs(b["x0"] - 0.12) < 0.02
