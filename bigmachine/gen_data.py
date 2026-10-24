"""Parallel synthetic-map generation for proposer training.  python gen_data.py --n 200000 --workers 30 --out data
Writes shards of 2000 maps (resumable: existing shards are skipped) plus manifest.json."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")  # one BLAS thread per worker process; set before numpy is imported

import argparse, json, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circuitsleuth import synth  # noqa: E402

SHARD = 2000


def encode(truths):
    from circuitsleuth import models
    fams = synth.FAMILY_NAMES
    fam = np.array([fams.index(t["family"]) for t in truths], np.int64)
    cross = np.array([int(t.get("crossing_visible", False)) for t in truths], np.float32)
    nper = np.array([(t["x_hi"] - t["x_lo"]) / t["period"] if models.FAMILIES[t["family"]]["flux_dependent"] else np.nan for t in truths], np.float32)
    nper = (np.log(nper) - np.log(0.4)) / (np.log(3.0) - np.log(0.4))
    P = {}
    for f in fams:
        slots = models.FAMILIES[f]["params"] + models.RESONATOR_PARAMS
        b = dict(models.FAMILIES[f]["bounds"]); b.update(models.RESONATOR_BOUNDS)
        arr = np.full((len(truths), len(slots)), np.nan, np.float32)
        for i, t in enumerate(truths):
            if t["family"] != f:
                continue
            for j, n in enumerate(slots):
                if n in t["params"]:
                    lo, hi = b[n]
                    arr[i, j] = (t["params"][n] - lo) / (hi - lo) if n == "d" else (np.log(t["params"][n]) - np.log(lo)) / (np.log(hi) - np.log(lo))
        P[f] = arr
    return fam, cross, nper, P


def make_shard(job):
    k, seed, out_dir = job
    path = os.path.join(out_dir, f"shard_{k:04d}.npz")
    if os.path.exists(path):
        return k, 0.0, True
    t0 = time.time()
    X, _, _, truths = synth.make_dataset(SHARD, seed=seed)
    fam, cross, nper, P = encode(truths)
    np.savez_compressed(path + ".tmp.npz", X=X, fam=fam, cross=cross, nper=nper, **{f"P_{f}": P[f] for f in P})
    os.replace(path + ".tmp.npz", path)
    return k, time.time() - t0, False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--out", default="data")
    ap.add_argument("--seed", type=int, default=12345)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    n_shards = int(np.ceil(a.n / SHARD))
    jobs = [(k, a.seed + 7919 * k, a.out) for k in range(n_shards)]
    t0 = time.time()
    done = 0
    with Pool(a.workers) as pool:
        for k, dt, skipped in pool.imap_unordered(make_shard, jobs):
            done += 1
            el = time.time() - t0
            print(f"[{time.strftime('%H:%M:%S')}] shard {k} {'skipped' if skipped else f'{dt:.0f}s'} ({done}/{n_shards}, {el/60:.1f} min, ETA {el/done*(n_shards-done)/60:.1f} min)", flush=True)
    json.dump(dict(n=a.n, shard=SHARD, n_shards=n_shards, seed=a.seed), open(os.path.join(a.out, "manifest.json"), "w"))
    print("done", flush=True)


if __name__ == "__main__":
    main()
