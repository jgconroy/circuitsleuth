"""Posterior sampler on the stored demo-A result: the chain must move (acceptance 0.1-0.7) and its widths must agree
with the Laplace covariance of the same robust likelihood within a factor 3 for the EJmax-EC valley.  Under 30 s."""
import json
import time

import numpy as np

from circuitsleuth import fit, models


def test_posterior_matches_laplace_on_demo_A():
    rep = json.load(open("artifacts/demo_A_report.json"))
    best = rep["best"]
    hyp = fit.Hypothesis(best["family"], best["coupled"])
    points = np.asarray(rep["points"], float)
    theta = fit.to_theta(best["params"], hyp.names)
    result = fit.result_from_report(rep)
    assert result.hyp == hyp and np.allclose(result.theta, theta)
    c_local = models.platform_scales(hyp.platform)["c_local"]
    t0 = time.time()
    post = fit.sample_posterior(points, result, tuple(rep["band_GHz"]), c_local, n_steps=800, seed=0)
    assert time.time() - t0 < 30
    print("\nposterior:", {k: post[k] for k in ("acceptance_rate", "n_kept", "n_points_used", "seconds")}, post["sd"], post["laplace_sd"])
    assert post["samples"].shape == (post["n_kept"], len(hyp.names)) and post["names"] == hyp.names
    assert 0.1 <= post["acceptance_rate"] <= 0.7
    for n in ("EJmax", "EC"):
        ratio = post["sd"][n] / post["laplace_sd"][n]
        assert 0.3 <= ratio <= 3.0, (n, ratio)
        assert abs(post["mean"][n] - best["params"][n]) < 4 * post["sd"][n]  # the chain stays around the MAP
    # the spur points (flat 5.1 GHz line) are excluded, as in the fit itself
    assert post["n_points_used"] < len(points)


def test_posterior_figure_and_pair_choice():
    from ui import design_panel

    assert design_panel.posterior_pair_choice(["EJmax", "EC", "d"]) == ("EJmax", "EC")
    assert design_panel.posterior_pair_choice(["EJ", "EC", "EL"]) == ("EJ", "EC")
    assert design_panel.posterior_pair_choice(["D", "E", "theta"]) == ("D", "E")
    rng = np.random.default_rng(0)
    S = np.column_stack([25 + 0.2 * rng.standard_normal(200), 0.3 - 0.003 * rng.standard_normal(200), 0.3 + 0.001 * rng.standard_normal(200)])
    d = dict(samples=S, names=["EJmax", "EC", "d"], map={"EJmax": 25.0, "EC": 0.3, "d": 0.3}, acceptance_rate=0.3, family="tunable_transmon",
             laplace_cov_params=np.cov(S.T))
    fig = design_panel.posterior_figure(d)
    assert len(fig.data) == 3  # samples, ellipse, MAP
    assert "GHz" in fig.layout.xaxis.title.text and "GHz" in fig.layout.yaxis.title.text
    fig2 = design_panel.posterior_figure(d, pair=("EJmax", "d"), laplace_cov=None)
    assert len(fig2.data) == 3
