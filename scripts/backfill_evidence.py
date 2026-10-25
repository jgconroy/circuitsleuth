"""Add Bayesian model comparison (Laplace evidence, posterior model probabilities) to saved reports that predate it.
python scripts/backfill_evidence.py artifacts/demo_A_report.json [more.json ...]"""
import json
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, ".")
from circuitsleuth import extract, fit, models  # noqa: E402


def main():
    for path in sys.argv[1:]:
        rep = json.load(open(path))
        pts = np.asarray(rep["points"], float)
        platform = models.platform_of(rep["best"]["family"])
        c_local = models.platform_scales(platform)["c_local"]
        flat = extract.flat_feature_mask(pts, tol=models.platform_scales(platform)["flat_tol"])
        ev_pts = pts[~flat] if int((~flat).sum()) >= max(10, 0.2 * len(pts)) else pts
        evidence = {}
        for h in rep["hypotheses"]:
            hyp = fit.Hypothesis(h["family"], bool(h.get("coupled")))
            theta = fit.to_theta({n: float(h["params"][n]) for n in hyp.names}, hyp.names)
            e = fit.log_evidence(ev_pts, SimpleNamespace(hyp=hyp, theta=theta), tuple(rep["band_GHz"]), c_local)
            if e:
                evidence[hyp.label] = e
        post = fit.model_posterior(evidence)
        for h in rep["hypotheses"]:
            lab = fit.Hypothesis(h["family"], bool(h.get("coupled"))).label
            h["log_evidence"] = round(evidence[lab]["log_evidence"], 2) if lab in evidence else None
            h["posterior"] = round(post["posteriors"][lab], 4) if post and lab in post["posteriors"] else None
        rep["model_posterior"] = post
        json.dump(rep, open(path, "w"))
        print(path)
        for h in rep["hypotheses"]:
            print(f"   {h['hypothesis']:40s} score {h['track_score']:+.3f}  logZ {h['log_evidence']}  P {h['posterior']}")
        print("   families:", post.get("families"), "| log10 BF vs other family:", post.get("log10_bayes_factor_vs_other_family"))


if __name__ == "__main__":
    main()
