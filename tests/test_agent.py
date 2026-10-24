import json

import numpy as np

from circuitsleuth import agent, extract, fit, synth


def hidden(family, params, period, offset, f_lo, f_hi, spurs=(), two_photon=True, noise=0.06, seed=0, sign=1):
    truth = dict(family=family, coupled="fr" in params, params=dict(params), period=period, offset=offset, x_lo=-1.0, x_hi=1.0,
                 f_lo=f_lo, f_hi=f_hi, linewidth=0.025, noise=noise, two_photon=two_photon, sign=sign, spurs=list(spurs))
    r = synth.render(truth, synth.SynthConfig(nx=81, nf=400), np.random.default_rng(seed))
    return truth, r


FAST = agent.AgentConfig(maxiter=40, use_classifier=False, holdout=True)


def test_agent_identifies_tunable_transmon_with_resonator_and_spur():
    t, r = hidden("tunable_transmon", dict(EJmax=25.0, EC=0.3, d=0.3, fr=7.0, g=0.1), 1.3, 0.2, 3.0, 9.0, spurs=[dict(f=5.1, amp=0.9, sideband=0.0)], seed=1)
    events = []
    rep = agent.identify(r["image"], r["x"], r["f"], config=FAST, progress=lambda k, p: events.append(k))
    assert rep["status"] == "ok", rep
    assert rep["best"]["family"] == "tunable_transmon" and rep["best"]["coupled"] is True, rep["best"]
    for k in ("EJmax", "EC", "fr"):
        assert abs(rep["best"]["params"][k] - t["params"][k]) / t["params"][k] < 0.05, (k, rep["best"]["params"])
    assert any(abs(u["f_mean"] - 5.1) < 0.05 for u in rep["unexplained"]), rep["unexplained"]
    assert any(abs(v - 5.1) < 0.05 for v in rep["flat_lines_GHz"])
    assert "holdout" in rep["best"] and rep["best"]["holdout"]["inlier_frac"] > 0.7
    json.loads(agent.to_json(rep))  # serialisable
    assert {"features", "proposals", "fit_start", "fit_done", "done"} <= set(events)


def test_agent_identifies_fixed_transmon():
    t, r = hidden("transmon", dict(EJ=15.0, EC=0.28), 1.0, 0.0, 3.0, 8.0, seed=2)
    rep = agent.identify(r["image"], r["x"], r["f"], config=FAST)
    assert rep["best"]["family"] == "transmon", rep["best"]
    assert abs(rep["best"]["params"]["EC"] - 0.28) < 0.02


def test_agent_identifies_fluxonium():
    t, r = hidden("fluxonium", dict(EJ=6.0, EC=1.0, EL=0.6), 1.1, -0.3, 0.3, 9.0, seed=3)
    rep = agent.identify(r["image"], r["x"], r["f"], config=FAST)
    assert rep["best"]["family"] == "fluxonium", rep["best"]
    for k in ("EJ", "EC", "EL"):
        assert abs(rep["best"]["params"][k] - t["params"][k]) / t["params"][k] < 0.05, (k, rep["best"]["params"])


def test_agent_points_input_and_next_measurements():
    # narrow band hides the two-photon line -> the agent should ask for it
    t, r = hidden("tunable_transmon", dict(EJmax=25.0, EC=0.3, d=0.3), 1.3, 0.2, 5.5, 9.0, two_photon=False, seed=4)
    pts = extract.extract_points(r["image"], r["x"], r["f"])
    rep = agent.identify(points=pts, band=(5.5, 9.0), config=agent.AgentConfig(maxiter=30, use_classifier=False, holdout=False, try_coupled="never"))
    assert rep["best"]["family"] == "tunable_transmon"
    kinds = {(n["kind"], n.get("line")) for n in rep["next_measurements"]}
    assert ("look_here", "0-2/2ph") in kinds or ("constrain_parameter", None) in kinds, rep["next_measurements"]
