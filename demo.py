"""Blind test: random hidden circuits -> identify -> recovery table.  python demo.py --n 12 --workers 4"""
import argparse, json, time
from pathlib import Path

import numpy as np

from circuitsleuth import agent, extract, synth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=12)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--no-classifier", action="store_true")
    ap.add_argument("--out", default="artifacts/blind_results.json")
    ap.add_argument("--platform", default="circuit", choices=list(synth.models.PLATFORMS))
    a = ap.parse_args()
    cfg = synth.SynthConfig(nx=81, nf=400, noise_range=(0.03, 0.1))
    fams = tuple(synth.models.PLATFORMS[a.platform]["families"])
    acfg = agent.AgentConfig(maxiter=a.maxiter, workers=a.workers, use_classifier=not a.no_classifier, families=fams if a.platform != "circuit" else (), try_coupled="auto" if a.platform == "circuit" else "never")
    rows = []
    for i in range(a.n):
        rng = np.random.default_rng(a.seed + i)
        t = synth.sample_truth(rng, cfg, platform=a.platform)
        r = synth.render(t, cfg, rng)
        t0 = time.time()
        rep = agent.identify(r["image"], r["x"], r["f"], config=acfg)
        dt = time.time() - t0
        row = dict(case=i, truth=f"{t['family']}{'+res' if t['coupled'] else ''}", n_points=rep.get("n_points"), runtime_s=round(dt, 1))
        if rep.get("status") == "ok":
            b = rep["best"]
            row["found"] = f"{b['family']}{'+res' if b['coupled'] else ''}"
            row["family_ok"] = b["family"] == t["family"]
            errs = {k: abs(b["params"][k] - v) / abs(v) for k, v in t["params"].items() if k in b["params"]}
            if "period" in b["params"] and synth.models.FAMILIES[t["family"]]["flux_dependent"]:
                errs["period"] = abs(b["params"]["period"] - t["period"]) / t["period"]
            circuit_errs = {k: v for k, v in errs.items() if k not in ("fr", "g", "x0", "phi")}  # resonator params need a crossing; x0/phi are offsets/angles, judged separately
            row["max_rel_err"] = round(max(circuit_errs.values()), 3) if circuit_errs else None
            row["max_rel_err_all"] = round(max(errs.values()), 3) if errs else None
            row["errs"] = {k: round(v, 3) for k, v in errs.items()}
            row["track_score"] = b["track_score"]
            row["holdout"] = (b.get("holdout") or {}).get("inlier_frac")
        else:
            row["found"] = rep.get("error")
            row["family_ok"] = False
        rows.append(row)
        print(f"[{i+1}/{a.n}] truth={row['truth']:22s} found={str(row['found']):22s} ok={row['family_ok']} maxerr={row.get('max_rel_err')} score={row.get('track_score')} {dt:.0f}s", flush=True)
    ok = [r for r in rows if r["family_ok"]]
    print(f"\nfamily identification: {len(ok)}/{len(rows)}")
    good = [r for r in ok if r.get("max_rel_err") is not None and r["max_rel_err"] < 0.05]
    print(f"family correct and all circuit parameters (EJ/EC/EL/d, flux period) within 5%: {len(good)}/{len(rows)}")
    print(f"mean runtime {np.mean([r['runtime_s'] for r in rows]):.0f} s per map")
    Path(a.out).parent.mkdir(exist_ok=True, parents=True)
    json.dump(rows, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
