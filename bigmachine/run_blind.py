"""Blind accuracy benchmark at scale.  python run_blind.py --n 200 --platform circuit --workers 28 [--effort normal]
Each map is identified in its own process; rows go to blind_<platform>.jsonl (resumable), summary printed at the end.
Run `--bench` first: it times 2 maps and projects the total."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, json, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circuitsleuth import agent, models, synth  # noqa: E402

PRESETS = {"fast": dict(maxiter=30, n_starts=1, popsize=12, top_k=2, coupled_trials=1), "normal": dict(maxiter=50, n_starts=2, popsize=15, top_k=3, coupled_trials=2)}
ABS = {"D": 0.002, "E": 0.002, "theta": 0.05, "phi": 0.05, "x0": 0.03, "d": 0.06}


def one(job):
    i, platform, effort, seed = job
    cfg = synth.SynthConfig(nx=81, nf=400, noise_range=(0.03, 0.1))
    rng = np.random.default_rng(seed)
    t = synth.sample_truth(rng, cfg, platform=platform)
    r = synth.render(t, cfg, rng)
    fams = tuple(models.PLATFORMS[platform]["families"])
    acfg = agent.AgentConfig(workers=1, families=fams if platform != "circuit" else (), try_coupled="auto" if platform == "circuit" else "never", **PRESETS[effort])
    t0 = time.time()
    try:
        rep = agent.identify(r["image"], r["x"], r["f"], config=acfg)
    except Exception as e:
        return dict(case=i, truth=t["family"], error=f"{type(e).__name__}: {e}"[:200])
    row = dict(case=i, truth=t["family"], truth_coupled=t["coupled"], runtime_s=round(time.time() - t0, 1))
    if rep.get("status") != "ok":
        row["error"] = rep.get("error"); return row
    b = rep["best"]
    row.update(found=b["family"], found_coupled=b["coupled"], family_ok=b["family"] == t["family"], score=b["track_score"], confidence=(b.get("confidence") or {}).get("level"))
    errs = {}
    for k, v in t["params"].items():
        if k in b["params"] and k not in ("fr", "g", "theta", "phi", "x0"):
            errs[k] = abs(b["params"][k] - v) / ABS[k] * 0.05 if k in ABS else abs(b["params"][k] - v) / abs(v)
    if "period" in b["params"] and models.FAMILIES[t["family"]]["flux_dependent"]:
        errs["period"] = abs(b["params"]["period"] - t["period"]) / t["period"]
    if t["family"] == "rydberg_atom":  # only alpha * field_scale^2 is observable
        for a_name in ("alpha1", "alpha2"):
            tv = t["params"][a_name] * t["params"]["field_scale"] ** 2
            fv = b["params"][a_name] * b["params"]["field_scale"] ** 2
            errs[a_name + "_curv"] = abs(fv - tv) / tv
            errs.pop(a_name, None)
        errs.pop("field_scale", None)
        errs["f0"] = abs(b["params"]["f0"] - t["params"]["f0"]) / 0.5 * 0.05
    if t["family"] == "nv_ensemble":  # field direction up to tetrahedral symmetry
        pt = models.nv_projections(t["params"]["theta"], t["params"]["phi"]) * t["params"]["field_scale"]
        pf = models.nv_projections(b["params"]["theta"], b["params"]["phi"]) * b["params"]["field_scale"]
        errs["field_vector"] = float(np.max(np.abs(pf - pt) / np.maximum(pt, 1.0)))
        errs.pop("field_scale", None)
    row["errs"] = {k: round(v, 4) for k, v in errs.items()}
    row["max_err"] = round(max(errs.values()), 4) if errs else None
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--platform", default="circuit", choices=list(models.PLATFORMS))
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--effort", default="normal", choices=list(PRESETS))
    ap.add_argument("--seed", type=int, default=5000)
    ap.add_argument("--bench", action="store_true")
    a = ap.parse_args()
    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"blind_{a.platform}.jsonl")
    done = set()
    if os.path.exists(out) and not a.bench:
        done = {json.loads(l)["case"] for l in open(out) if l.strip()}
    jobs = [(i, a.platform, a.effort, a.seed + i) for i in range(a.n) if i not in done]
    if a.bench:
        t0 = time.time(); rows = [one(j) for j in jobs[:2]]; dt = (time.time() - t0) / 2
        print(f"{dt:.0f} s per map in one process -> {a.n} maps on {a.workers} workers ≈ {a.n*dt/a.workers/60:.0f} min"); return
    t0 = time.time()
    with Pool(a.workers) as pool, open(out, "a") as fh:
        for k, row in enumerate(pool.imap_unordered(one, jobs), 1):
            fh.write(json.dumps(row) + "\n"); fh.flush()
            print(f"[{k}/{len(jobs)}] case {row['case']} {row.get('truth')} -> {row.get('found', row.get('error'))} ok={row.get('family_ok')} max_err={row.get('max_err')} ({(time.time()-t0)/60:.1f} min)", flush=True)
    rows = [json.loads(l) for l in open(out) if l.strip()]
    ok = [r for r in rows if r.get("family_ok")]
    good = [r for r in ok if r.get("max_err") is not None and r["max_err"] < 0.05]
    print(f"\n{a.platform}: family {len(ok)}/{len(rows)}; family + params within 5%: {len(good)}/{len(rows)}; mean runtime {np.mean([r.get('runtime_s', 0) for r in rows]):.0f} s")
    by = {}
    for r in rows:
        by.setdefault(r["truth"], []).append(r.get("family_ok", False))
    for f, v in by.items():
        print(f"  {f:18s} family ok {sum(v)}/{len(v)}")


if __name__ == "__main__":
    main()
