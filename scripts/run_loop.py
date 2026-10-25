"""Precompute an autopilot run for a demo.  python scripts/run_loop.py A --rounds 3 [--fast]"""
import json, sys, time
import numpy as np
from circuitsleuth import loop, models
sys.path.insert(0, ".")


def main():
    from ui.demos import DEMOS, make_demo
    key = sys.argv[1]
    rounds = int(sys.argv[sys.argv.index("--rounds") + 1]) if "--rounds" in sys.argv else 3
    platform = next(pl for pl, d in DEMOS.items() if any(k.startswith(key) for k in d))
    name = [k for k in DEMOS[platform] if k.startswith(key)][0]
    truth, r = make_demo(platform, name)
    fams = tuple(models.PLATFORMS[platform]["families"])
    cfg = dict(maxiter=30, n_starts=1, popsize=12, top_k=2, coupled_trials=1, workers=4) if "--fast" in sys.argv else dict(workers=4)
    t0 = time.time()
    res = loop.run_loop(r["image"], r["x"], r["f"], fams, rounds=rounds, cfg_kwargs=cfg, progress=lambda k, p: print(k, {kk: vv for kk, vv in p.items() if kk in ("round", "n_points", "hypothesis", "runtime_s")}, flush=True))
    res["truth"] = dict(family=truth["family"], params=truth["params"], period=truth.get("period"), offset=truth.get("offset"))
    res["demo"] = key
    json.dump(res, open(f"artifacts/loop_{key}.json", "w"), indent=1, default=float)
    for row in loop.sigma_table(res):
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items() if k.endswith("rel_sigma") or k in ("round", "n_points")})
    print(f"done in {time.time()-t0:.0f}s -> artifacts/loop_{key}.json")


if __name__ == "__main__":
    main()
