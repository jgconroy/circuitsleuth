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


def handpicked(family, params, period, offset, f_lo, f_hi, spurs=(), two_photon=True, noise=0.06, seed=0, nx=81, sign=1):
    truth = dict(family=family, coupled="fr" in params, params=dict(params), period=period, offset=offset, x_lo=-1.0, x_hi=1.0,
                 f_lo=f_lo, f_hi=f_hi, linewidth=0.025, noise=noise, two_photon=two_photon, sign=sign, spurs=list(spurs))
    cfg = synth.SynthConfig(nx=nx, nf=400)
    r = synth.render(truth, cfg, np.random.default_rng(seed))
    pts = extract.extract_points(r["image"], r["x"], r["f"])
    return truth, r, pts


def test_fit_robust_to_spurs_and_dropout():
    spurs = [dict(f=5.1, amp=0.9, sideband=0.0), dict(f=7.3, amp=0.8, sideband=0.2)]
    t, r, pts = handpicked("tunable_transmon", dict(EJmax=25.0, EC=0.3, d=0.3), period=1.3, offset=0.2, f_lo=2.0, f_hi=9.0, spurs=spurs, seed=5)
    band = (r["f"][0], r["f"][-1])
    ids = extract.link_tracks(pts)
    flat = extract.flat_track_mask(pts, ids)
    res = fit.fit_hypothesis(pts[~flat], fit.Hypothesis("tunable_transmon", False), band, seed=2)
    print("\nspurs:", res.inlier_frac, res.params)
    check_params(res, t, tol=0.04)
    # evaluate on ALL points: spur points must be outliers, and the track score must still be high
    obj = fit.Objective(res.hyp, pts, band, 0.012)
    rr = obj.residuals(res.theta)
    inl = rr < 0.036
    spur_pts = np.abs(pts[:, 1] - 5.1) < 0.03
    assert inl[spur_pts].mean() < 0.2
    assert fit.track_score(pts, ids, flat, inl) > 0.6
    ho = fit.holdout_check(pts[~flat], res, band)
    assert ho["inlier_frac"] > 0.7


def test_fit_coupled_from_bare_init():
    # qubit sweeps through the 7 GHz readout resonator: a visible avoided crossing
    t, r, pts = handpicked("tunable_transmon", dict(EJmax=25.0, EC=0.3, d=0.3, fr=7.0, g=0.1), period=1.3, offset=0.2, f_lo=3.0, f_hi=9.0, seed=6)
    band = (r["f"][0], r["f"][-1])
    bare = fit.fit_hypothesis(pts, fit.Hypothesis("tunable_transmon", False), band, seed=3)
    init = dict(bare.params, fr=7.0, g=0.1)
    res = fit.fit_hypothesis(pts, fit.Hypothesis("tunable_transmon", True), band, init=init, seed=3, maxiter=40)
    print("\ncoupled:", res.inlier_frac, res.params, "\nbare   :", bare.inlier_frac, bare.params)
    assert res.inlier_frac >= bare.inlier_frac - 0.02
    check_params(res, t, tol=0.05)


def test_degenerate_case_reports_large_uncertainty():
    # without the two-photon or 1-2 line, EC and EJmax are not separately identifiable: sigma must say so
    t, r, pts = handpicked("tunable_transmon", dict(EJmax=25.0, EC=0.3, d=0.3), period=1.3, offset=0.2, f_lo=5.5, f_hi=9.0, two_photon=False, seed=7)
    band = (r["f"][0], r["f"][-1])
    res = fit.fit_hypothesis(pts, fit.Hypothesis("tunable_transmon", False), band, seed=4, maxiter=30)
    print("\ndegenerate:", res.params, res.sigma)
    assert res.inlier_frac > 0.8
    assert res.sigma["EC"] / res.params["EC"] > 0.1 or abs(res.params["EC"] - 0.3) / 0.3 < 0.05
