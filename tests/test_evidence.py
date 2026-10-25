"""Bayesian model comparison: Laplace evidence and posterior model probabilities on the saved demo reports (no fitting)."""
import json
from types import SimpleNamespace

import numpy as np

from circuitsleuth import extract, fit, models


def _evidence_for(path):
    rep = json.load(open(path))
    pts = np.asarray(rep["points"], float)
    platform = models.platform_of(rep["best"]["family"])
    c_local = models.platform_scales(platform)["c_local"]
    flat = extract.flat_feature_mask(pts, tol=models.platform_scales(platform)["flat_tol"])
    ev_pts = pts[~flat] if int((~flat).sum()) >= max(10, 0.2 * len(pts)) else pts
    ev = {}
    for h in rep["hypotheses"]:
        hyp = fit.Hypothesis(h["family"], bool(h.get("coupled")))
        theta = fit.to_theta({n: float(h["params"][n]) for n in hyp.names}, hyp.names)
        e = fit.log_evidence(ev_pts, SimpleNamespace(hyp=hyp, theta=theta), tuple(rep["band_GHz"]), c_local)
        if e:
            ev[hyp.label] = e
    return rep, ev


def test_evidence_ranks_demo_a_like_the_score_and_gives_huge_bayes_factor():
    rep, ev = _evidence_for("artifacts/demo_A_report.json")
    assert set(ev) >= {"tunable transmon + readout resonator", "tunable transmon", "fluxonium"}
    post = fit.model_posterior(ev)
    assert post["best"] == rep["best"]["hypothesis"]
    assert post["families"]["tunable_transmon"] > 0.999
    assert post["log10_bayes_factor_vs_other_family"] > 20
    for e in ev.values():  # the Occam factor is negative (prior volume > posterior volume) and finite
        assert np.isfinite(e["occam"]) and e["occam"] < 0


def test_occam_factor_prefers_bare_fluxonium_on_demo_b():
    rep, ev = _evidence_for("artifacts/demo_B_report.json")
    post = fit.model_posterior(ev)
    assert post["best"] == "fluxonium"
    assert post["posteriors"]["fluxonium"] > post["posteriors"]["fluxonium + readout resonator"]
    # the two fluxonium variants explain the same features: the resonator pays only its prior volume (a few nats), not hundreds
    assert 0 < ev["fluxonium"]["log_evidence"] - ev["fluxonium + readout resonator"]["log_evidence"] < 30


def test_model_posterior_handles_empty_and_single():
    assert fit.model_posterior({}) == {}
    one = fit.model_posterior({"x": dict(log_evidence=-5.0, family="transmon")})
    assert one["posteriors"]["x"] == 1.0 and one["log10_bayes_factor_vs_other_family"] is None
