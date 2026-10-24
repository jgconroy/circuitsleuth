"""Timing benchmark: run this BEFORE a long job.  python bench.py --n 200000 --epochs 30 --workers 30 --width 2
Measures map generation (1 process and N processes), one training epoch per 1000 maps, and one blind
identification, then prints the projected wall time and memory for the requested run."""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, sys, time
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circuitsleuth import synth  # noqa: E402


def gen_some(k):
    synth.make_dataset(25, seed=1000 + k)
    return 25


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--skip-fit", action="store_true")
    a = ap.parse_args()
    print(f"cpu_count={os.cpu_count()} workers={a.workers}")
    t0 = time.time(); synth.make_dataset(60, seed=1); per_map = (time.time() - t0) / 60
    print(f"generation, 1 process: {per_map*1e3:.0f} ms/map")
    t0 = time.time()
    with Pool(a.workers) as pool:
        tot = sum(pool.map(gen_some, range(a.workers * 2)))
    rate = tot / (time.time() - t0)
    print(f"generation, {a.workers} processes: {rate:.0f} maps/s  -> {a.n} maps in {a.n/rate/60:.1f} min")
    import torch
    from circuitsleuth import classify
    torch.set_num_threads(a.workers)
    X, _, _, truths = synth.make_dataset(3000, seed=2)
    for t in truths:
        t.setdefault("crossing_visible", False)
    t0 = time.time()
    classify.train(epochs=1, out=__import__("pathlib").Path("bench_tmp.pt"), log=lambda s: None, width=a.width, data=(X, truths), batch=256)
    per_1000 = (time.time() - t0) / 3.0
    os.remove("bench_tmp.pt"); os.remove("bench_tmp.json")
    print(f"training (width {a.width}, {a.workers} threads): {per_1000:.1f} s per epoch per 1000 maps -> {a.epochs} epochs on {a.n} maps = {a.epochs*a.n/1000*per_1000/3600:.1f} h")
    print(f"memory for {a.n} thumbnails: {a.n*64*64*4/1e9:.1f} GB (+ about 2x during loading)")
    if not a.skip_fit:
        from circuitsleuth import agent
        cfg = synth.SynthConfig(nx=81, nf=400, noise_range=(0.03, 0.1))
        rng = np.random.default_rng(5); t = synth.sample_truth(rng, cfg); r = synth.render(t, cfg, rng)
        t0 = time.time(); agent.identify(r["image"], r["x"], r["f"], config=agent.AgentConfig(maxiter=40, workers=1, use_classifier=False)); dt = time.time() - t0
        print(f"blind identification, 1 process, normal-ish effort: {dt:.0f} s/map -> {a.workers} in parallel: {a.workers*3600/dt:.0f} maps/hour")
    print("\nIf the projected times are acceptable:  python run_train.py start --n", a.n, "--epochs", a.epochs, "--workers", a.workers, "--width", a.width)


if __name__ == "__main__":
    main()
