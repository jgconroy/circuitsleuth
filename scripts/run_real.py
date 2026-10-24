"""Run CircuitSleuth on a digitised real map.  python scripts/run_real.py data/demo/nguyen2019_devA.npz"""
import json, sys, time
import numpy as np
from circuitsleuth import agent, extract


def main():
    path = sys.argv[1]
    d = np.load(path)
    img, x, f = d["image"], d["x"], d["f"]
    pts = extract.extract_points(img, x, f)
    print("points", len(pts), "flat", extract.flat_lines(pts))
    calibrated = "--calibrated" in sys.argv
    cfg = agent.AgentConfig(maxiter=60, workers=4, use_classifier=False, families=("fluxonium",), try_coupled="never", flux_calibrated=calibrated)
    t0 = time.time()
    rep = agent.identify(img, x, f, band=(0.05, 5.0), config=cfg, progress=lambda k, p: print(k, {kk: vv for kk, vv in p.items() if kk not in ("lines", "params", "report")}) if k not in ("fit_progress",) else None)
    print(f"{time.time()-t0:.0f}s")
    b = rep["best"]
    print("best:", b["hypothesis"], b["params_pretty"], "score", b["track_score"], "holdout", b.get("holdout"))
    print("truth: EJ 3.0  EC 0.84  EL 1.0  (paper fit)")
    print("unexplained:", [(u["f_lo"], u["f_hi"], u["n_points"]) for u in rep["unexplained"]][:6])
    print("next:", [(n["kind"], n.get("line"), n.get("parameter")) for n in rep["next_measurements"]])
    json.dump(rep, open("artifacts/nguyen2019_report" + ("_calibrated" if calibrated else "") + ".json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
