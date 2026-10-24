import time

import numpy as np
import pytest

from circuitsleuth import extract, fit, synth


def make_case(seed, family, coupled, **cfg_kw):
    cfg = synth.SynthConfig(nx=81, nf=400, **cfg_kw)
    rng = np.random.default_rng(seed)
    t = synth.sample_truth(rng, cfg, family=family, coupled=coupled)
    r = synth.render(t, cfg, rng)
    pts = extract.extract_points(r["image"], r["x"], r["f"])
    return t, r, pts


def rel_err(a, b):
    return abs(a - b) / abs(b)


def check_params(result, truth, tol=0.03):
    for k, v in truth["params"].items():
        got = result.params[k]
        if k == "d":
            assert abs(got - v) < 0.06, (k, got, v)
        elif k == "g":
            assert rel_err(got, v) < 0.25, (k, got, v)
        else:
            assert rel_err(got, v) < tol, (k, got, v)
    if "period" in result.params:
        assert rel_err(result.params["period"], truth["period"]) < tol


@pytest.mark.parametrize("family", ["tunable_transmon", "fluxonium", "transmon"])
def test_fit_recovers_bare_circuit(family):
    t, r, pts = make_case(11, family, coupled=False, max_spurs=0, noise_range=(0.03, 0.08))
    band = (r["f"][0], r["f"][-1])
    t0 = time.time()
    res = fit.fit_hypothesis(pts, fit.Hypothesis(family, False), band, seed=1)
    dt = time.time() - t0
    print(f"\n{family}: {dt:.1f}s evals={res.n_evals} inlier={res.inlier_frac:.3f} rms={1e3*res.rms_inliers:.1f}MHz")
    print("truth", t["params"], t["period"], "\nfit  ", res.params)
    assert res.inlier_frac > 0.85
    check_params(res, t)
    assert dt < 150


def test_fit_robust_to_spurs_and_dropout():
    t, r, pts = make_case(21, "tunable_transmon", coupled=False, max_spurs=2, noise_range=(0.05, 0.1))
    t["spurs"] = [dict(f=0.5 * (r["f"][0] + r["f"][-1]), amp=1.0, sideband=0.0), dict(f=r["f"][0] + 0.8 * (r["f"][-1] - r["f"][0]), amp=0.8, sideband=0.2)]
    rng = np.random.default_rng(22)
    r = synth.render(t, synth.SynthConfig(nx=81, nf=400), rng)
    pts = extract.extract_points(r["image"], r["x"], r["f"])
    band = (r["f"][0], r["f"][-1])
    res = fit.fit_hypothesis(pts, fit.Hypothesis("tunable_transmon", False), band, seed=2)
    print("\nspurs:", res.inlier_frac, res.params, t["params"], t["period"])
    check_params(res, t, tol=0.04)
    # spur points must be flagged as outliers
    out_f = pts[~res.inlier_mask, 1]
    assert np.any(np.abs(out_f - t["spurs"][0]["f"]) < 0.05)
    ho = fit.holdout_check(pts, res, band)
    assert ho["inlier_frac"] > 0.5


def test_fit_coupled_from_bare_init():
    t, r, pts = make_case(31, "tunable_transmon", coupled=True, max_spurs=0, noise_range=(0.03, 0.08))
    band = (r["f"][0], r["f"][-1])
    bare = fit.fit_hypothesis(pts, fit.Hypothesis("tunable_transmon", False), band, seed=3)
    init = dict(bare.params, fr=7.0, g=0.1)
    res = fit.fit_hypothesis(pts, fit.Hypothesis("tunable_transmon", True), band, init=init, seed=3, maxiter=40)
    print("\ncoupled:", res.inlier_frac, res.params, "\ntruth", t["params"], t["period"])
    assert res.inlier_frac >= bare.inlier_frac - 0.02
    check_params(res, t, tol=0.05)
