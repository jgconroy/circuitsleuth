"""Render README figures from saved reports.  python scripts/make_figs.py artifacts/nguyen2019_report.json data/demo/nguyen2019_devA.npz artifacts/fig_real.png"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from circuitsleuth import extract

rep = json.load(open(sys.argv[1]))
d = np.load(sys.argv[2])
img, x, f = d["image"], d["x"], d["f"]
pts = np.array(rep["points"]); inl = np.array(rep["inlier_mask"], bool)
xg = np.array(rep["x_grid"])
fig, ax = plt.subplots(figsize=(8, 5.5), dpi=130)
ax.imshow(np.clip(extract.normalise(img), 0, 10), extent=[x[0], x[-1], f[0], f[-1]], origin="lower", aspect="auto", cmap="magma")
ax.scatter(pts[inl, 0], pts[inl, 1], s=8, c="#8ef58a", lw=0, label="explained features")
ax.scatter(pts[~inl, 0], pts[~inl, 1], s=12, c="#ff5c5c", marker="x", lw=0.8, label="unexplained")
styles = {"0-1": dict(lw=2.2), "0-2": dict(lw=1.2), "0-3": dict(lw=1.0), "0-2/2ph": dict(lw=1.2, ls="--"), "1-2": dict(lw=1.0, ls=":"), "res": dict(lw=1.0, ls="-.")}
for name, vals in rep["lines"].items():
    v = np.array([np.nan if u is None else u for u in vals], float)
    if np.isfinite(v).any():
        ax.plot(xg, v, color="white", alpha=0.9, label=name, **styles.get(name, {}))
boxes = [n for n in rep["next_measurements"] if n["kind"] == "look_here"]
lines = rep["lines"]; x_grid = xg
if boxes:
    for b in boxes:
        name = b.get("line")
        if lines is not None and x_grid is not None and name in lines:
            v = np.array([np.nan if u is None else u for u in lines[name]], float)
            m = (x_grid >= b["x_lo"]) & (x_grid <= b["x_hi"]) & np.isfinite(v)
            if m.any():
                ax.plot(x_grid[m], v[m], color="#ffd166", lw=4, alpha=0.55, solid_capstyle="round")
                ax.text(x_grid[m][0], v[m][0], f" look here: {name}", color="#ffd166", fontsize=8, va="bottom")
ax.set_ylim(f[0], f[-1]); ax.set_xlabel("control / flux"); ax.set_ylabel("frequency (GHz)")
b = rep["best"]
ax.set_title(f"{b['hypothesis']}: " + ", ".join(f"{k}={v}" for k, v in b["params_pretty"].items() if k not in ("offset",)), fontsize=8)
ax.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white", facecolor="black")
fig.tight_layout(); fig.savefig(sys.argv[3]); print("saved", sys.argv[3])
