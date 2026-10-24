"""Batch mode: identify every map in a folder and write one row per device.

    python batch.py <folder> [--platform circuit|nv|dot] [--workers 4] [--effort fast|normal] [--out results.csv]

Accepts .npz (keys image, x, f), 2-D grid .csv (first row control values, first column frequencies in GHz),
and point-list .csv (columns x, f[, weight]).
"""
from __future__ import annotations

import argparse, csv, glob, json, os, time

import numpy as np

from circuitsleuth import agent, models

PRESETS = {"fast": dict(maxiter=30, n_starts=1, popsize=12, top_k=2, coupled_trials=1), "normal": dict(maxiter=50, n_starts=2, popsize=15, top_k=3, coupled_trials=2)}


def load_map(path: str):
    if path.endswith(".npz"):
        d = np.load(path)
        return dict(image=d["image"], x=d["x"], f=d["f"])
    arr = np.genfromtxt(path, delimiter=",")
    if arr.ndim == 2 and arr.shape[1] <= 3 and arr.shape[0] > 10:  # point list
        pts = arr[:, :3] if arr.shape[1] == 3 else np.column_stack([arr[:, :2], np.ones(len(arr))])
        return dict(points=pts)
    x = arr[0, 1:]
    f = arr[1:, 0]
    img = arr[1:, 1:]
    if np.nanmax(f) > 1e6:  # Hz -> GHz
        f = f / 1e9
    elif np.nanmax(f) > 1e3:  # MHz -> GHz
        f = f / 1e3
    return dict(image=img, x=x, f=f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("folder")
    ap.add_argument("--platform", default="circuit", choices=list(models.PLATFORMS))
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--effort", default="fast", choices=list(PRESETS))
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    files = sorted(glob.glob(os.path.join(a.folder, "*.npz")) + glob.glob(os.path.join(a.folder, "*.csv")))
    fams = tuple(models.PLATFORMS[a.platform]["families"])
    cfg = agent.AgentConfig(workers=a.workers, families=fams if a.platform != "circuit" else (), try_coupled="auto" if a.platform == "circuit" else "never", **PRESETS[a.effort])
    out = a.out or os.path.join(a.folder, "circuitsleuth_results.csv")
    rows = []
    for path in files:
        t0 = time.time()
        m = load_map(path)
        try:
            rep = agent.identify(m.get("image"), m.get("x"), m.get("f"), points=m.get("points"), config=cfg)
        except Exception as e:
            rows.append(dict(file=os.path.basename(path), status=f"error: {type(e).__name__}"))
            continue
        if rep.get("status") != "ok":
            rows.append(dict(file=os.path.basename(path), status=rep.get("error")))
            continue
        b = rep["best"]
        row = dict(file=os.path.basename(path), status="ok", hypothesis=b["hypothesis"], confidence=(b.get("confidence") or {}).get("level"), explained=b.get("explained_dynamic"),
                   holdout=(b.get("holdout") or {}).get("inlier_frac"), runtime_s=round(time.time() - t0, 1))
        for k, v in b["params"].items():
            row[k] = v
            row[k + "_sigma"] = b["sigma"].get(k)
        rows.append(row)
        json.dump(rep, open(os.path.splitext(path)[0] + "_report.json", "w"), indent=1, default=float)
        print(f"{row['file']}: {row['hypothesis']} ({row['runtime_s']} s)", flush=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
