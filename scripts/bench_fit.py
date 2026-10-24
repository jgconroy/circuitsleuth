"""Benchmark fitter success/time on random cases. Usage: python scripts/bench_fit.py fluxonium 11 12 13 --workers 4"""
import sys, time, argparse
import numpy as np
from circuitsleuth import synth, extract, fit

ap = argparse.ArgumentParser()
ap.add_argument("family")
ap.add_argument("seeds", nargs="+", type=int)
ap.add_argument("--workers", type=int, default=1)
ap.add_argument("--coupled", action="store_true")
a = ap.parse_args()
cfg = synth.SynthConfig(nx=81, nf=400, max_spurs=0, noise_range=(0.03, 0.08))
for seed in a.seeds:
    rng = np.random.default_rng(seed)
    t = synth.sample_truth(rng, cfg, family=a.family, coupled=a.coupled)
    r = synth.render(t, cfg, rng)
    pts = extract.extract_points(r["image"], r["x"], r["f"])
    band = (r["f"][0], r["f"][-1])
    t0 = time.time()
    res = fit.fit_hypothesis(pts, fit.Hypothesis(a.family, a.coupled), band, seed=1, workers=a.workers)
    dt = time.time() - t0
    errs = {k: abs(res.params[k] - v) / abs(v) for k, v in t["params"].items()}
    if "period" in res.params:
        errs["period"] = abs(res.params["period"] - t["period"]) / t["period"]
    ok = all(e < 0.05 for k, e in errs.items() if k not in ("d", "g")) and errs.get("d", 0) * 0.9 < 0.06
    print(f"seed {seed}: {dt:6.1f}s evals={res.n_evals:6d} npts={len(pts):4d} inlier={res.inlier_frac:.3f} ok={ok} maxerr={max(errs.values()):.3f}", flush=True)
    print("   truth", {k: round(v, 3) for k, v in t["params"].items()}, "P", round(t["period"], 3), "2ph", t["two_photon"], "noise", round(t["noise"], 3), flush=True)
    print("   fit  ", {k: round(v, 3) for k, v in res.params.items()}, flush=True)
