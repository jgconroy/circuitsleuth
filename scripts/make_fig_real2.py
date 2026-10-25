"""Real-data figure for the README / Devpost: the published fluxonium map (Nguyen et al., PRX 2019, device A) cropped to the region that
has data, with qĤolmes's fitted 0-1 and two-photon lines, the paper's own parameters as dashed curves, and the numbers side by side.
    python scripts/make_fig_real2.py  ->  artifacts/fig_real_fit_vs_paper.png"""
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, ".")
from circuitsleuth import fit  # noqa: E402

d = np.load("data/demo/nguyen2019_devA.npz")
img, x, f = d["image"], d["x"], d["f"]
rep = json.load(open("artifacts/nguyen2019_report_calibrated.json"))
best = rep["best"]
hyp = fit.Hypothesis("fluxonium", False)
xg = np.linspace(0.0, 0.35, 300)
fitted = fit.predict_lines(hyp, best["params"], xg)
paper = fit.predict_lines(hyp, dict(EJ=3.0, EC=0.84, EL=1.0, period=1.0, offset=0.0), xg)
P = np.asarray(rep["points"], float)
inl = np.asarray(rep["inlier_mask"], bool)

BG, SURF, TEXT, MUTED, AMBER, GREEN = "#111214", "#1a1a19", "#f0efec", "#c3c2b7", "#fab219", "#4cae4c"
plt.rcParams.update({"font.family": "IBM Plex Sans, Helvetica, Arial", "text.color": TEXT, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": BG, "savefig.facecolor": BG})
fig, (ax, at) = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw=dict(width_ratios=[2.3, 1]))
mx = (x >= 0.0) & (x <= 0.35)
mf = (f >= 2.6) & (f <= 5.0)
ax.imshow(img[np.ix_(mf, mx)], origin="lower", aspect="auto", extent=[x[mx][0], x[mx][-1], f[mf][0], f[mf][-1]], cmap="magma", vmin=np.percentile(img[np.ix_(mf, mx)], 5), vmax=np.percentile(img[np.ix_(mf, mx)], 99.5))
for key, lab, lw in (("0-1", "0→1", 2.6), ("0-2/2ph", "0→2 two-photon", 1.8)):
    ax.plot(xg, fitted[key], color=AMBER, lw=lw, label=f"qĤolmes fit · {lab}")
    ax.plot(xg, paper[key], color="#86b6ef", lw=1.2, ls="--", label=f"paper's parameters · {lab}")
m = inl & (P[:, 0] >= 0) & (P[:, 0] <= 0.35) & (P[:, 1] >= 2.6) & (P[:, 1] <= 5)
ax.scatter(P[m, 0], P[m, 1], s=14, color=GREEN, edgecolor="none", zorder=5, label=f"features used by the fit ({int(m.sum())})")
ax.set_xlim(0.0, 0.35); ax.set_ylim(2.6, 5.0)
ax.set_xlabel("External flux Φ (Φ₀), axis as published"); ax.set_ylabel("Drive frequency (GHz)")
ax.set_title("Fluxonium device A, Nguyen et al. PRX 9, 041041 (2019): map digitised from the paper's figure", fontsize=11, color=MUTED, loc="left")
leg = ax.legend(loc="lower left", fontsize=9, facecolor=SURF, edgecolor="none", labelcolor=TEXT)
at.axis("off")
sig = best.get("sigma", {})
rows = [("E_J", best["params"]["EJ"], sig.get("EJ"), 3.0), ("E_C", best["params"]["EC"], sig.get("EC"), 0.84), ("E_L", best["params"]["EL"], sig.get("EL"), 1.0),
        ("flux period", best["params"].get("period", 1.0), sig.get("period"), 1.0)]
y = 0.92
at.text(0.0, y, "fit vs paper", fontsize=13, fontweight="bold", color=TEXT, transform=at.transAxes); y -= 0.1
at.text(0.0, y, "parameter (GHz)", fontsize=10, color=MUTED, transform=at.transAxes); at.text(0.40, y, "qĤolmes", fontsize=10, color=MUTED, transform=at.transAxes)
at.text(0.84, y, "paper", fontsize=10, color=MUTED, transform=at.transAxes); y -= 0.09
for name, v, s_, pv in rows:
    at.text(0.0, y, name, fontsize=11, color=TEXT, transform=at.transAxes)
    at.text(0.40, y, f"{v:.2f} ± {s_:.2f}" if s_ else f"{v:.2f}", fontsize=11, color=AMBER, family="monospace", transform=at.transAxes)
    at.text(0.84, y, f"{pv:.2f}", fontsize=11, color="#86b6ef", family="monospace", transform=at.transAxes); y -= 0.085
y -= 0.03
at.text(0.0, y, "Published map, no tuning.", fontsize=10, color=MUTED, transform=at.transAxes)
fig.tight_layout()
fig.savefig("artifacts/fig_real_fit_vs_paper.png", dpi=160)
print("wrote artifacts/fig_real_fit_vs_paper.png")
