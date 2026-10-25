"""Are the reported 1-sigma uncertainties calibrated?  Random hidden maps -> identify -> pulls (fit - truth) / sigma.

    python scripts/coverage.py --platform nv --n 40 --workers 4

For a calibrated estimator 68 % of the pulls lie within 1 sigma and 95 % within 2 sigma.  Maps run in parallel
(`--workers` processes, one map each, the fitter itself single-threaded).  Outputs: artifacts/coverage_<platform>.json
and artifacts/fig_pulls_<platform>.png (dark style matching the UI).  Not meant for circuits (minutes per map).

Symmetry-equivalent parameters are not scored: NV theta/phi/x0 (tetrahedral symmetry of the field direction),
Rydberg alpha1/alpha2/field_scale (only alpha * field_scale^2 is observable); a flux offset is compared modulo the period.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

SKIP = {"nv": {"theta", "phi", "x0"}, "atom": {"alpha1", "alpha2", "field_scale"}}
BG, SURFACE, TEXT, MUTED, GRID, BLUE, AMBER, RED = "#111214", "#1a1a19", "#f0efec", "#c3c2b7", "#2a2a28", "#3987e5", "#fab219", "#d03b3b"


def truth_values(t: dict) -> dict:
    from circuitsleuth import models

    vals = dict(t["params"])
    if models.control_params(t["family"]) == ("period", "offset"):
        vals["period"], vals["offset"] = t["period"], t["offset"]
    return vals


def pulls_for(t: dict, best: dict, platform: str) -> dict:
    """name -> (fit - truth) / sigma for every scored parameter (NaN when sigma is missing or zero)."""
    out = {}
    tv = truth_values(t)
    for n, v in tv.items():
        if n in SKIP.get(platform, set()) or n not in best["params"]:
            continue
        s = best["sigma"].get(n)
        fv = best["params"][n]
        dv = fv - v
        if n == "offset":  # defined modulo the period
            P = best["params"].get("period", tv.get("period", 1.0))
            dv = dv - P * np.round(dv / P)
        out[n] = float(dv / s) if (s is not None and np.isfinite(s) and s > 0) else float("nan")
    return out


def run_case(args):
    i, seed, platform, maxiter = args
    from circuitsleuth import agent, synth

    cfg = synth.SynthConfig(nx=81, nf=400, noise_range=(0.03, 0.1))
    fams = tuple(synth.models.PLATFORMS[platform]["families"])
    acfg = agent.AgentConfig(maxiter=maxiter, workers=1, use_classifier=False, families=fams if platform != "circuit" else (),
                             try_coupled="auto" if platform == "circuit" else "never")
    rng = np.random.default_rng(seed)
    t = synth.sample_truth(rng, cfg, platform=platform)
    r = synth.render(t, cfg, rng)
    t0 = time.time()
    try:
        rep = agent.identify(r["image"], r["x"], r["f"], config=acfg)
    except Exception as e:  # a crash is a miss, not a reason to lose the table
        rep = dict(error=f"{type(e).__name__}: {e}")
    row = dict(case=i, seed=seed, truth=t["family"], runtime_s=round(time.time() - t0, 1), truth_values={k: float(v) for k, v in truth_values(t).items()})
    if rep.get("status") == "ok":
        b = rep["best"]
        row.update(found=b["family"], family_ok=b["family"] == t["family"], params=b["params"], sigma=b["sigma"], n_points=rep.get("n_points"),
                   pulls=pulls_for(t, b, platform) if b["family"] == t["family"] else {})
    else:
        row.update(found=rep.get("error"), family_ok=False, pulls={})
    return row


def summarise(rows: list[dict]) -> dict:
    names = sorted({n for r in rows for n in r.get("pulls", {})})
    per = {}
    allp = []
    for n in names:
        p = np.array([r["pulls"][n] for r in rows if n in r.get("pulls", {})], float)
        p = p[np.isfinite(p)]
        allp.extend(p.tolist())
        if len(p):
            per[n] = dict(n=int(len(p)), within_1sigma=float(np.mean(np.abs(p) < 1)), within_2sigma=float(np.mean(np.abs(p) < 2)),
                          median_abs_pull=float(np.median(np.abs(p))), rms_pull=float(np.sqrt(np.mean(p**2))), frac_beyond_5sigma=float(np.mean(np.abs(p) > 5)))
    allp = np.array(allp)
    overall = dict(n=int(len(allp)), within_1sigma=float(np.mean(np.abs(allp) < 1)) if len(allp) else float("nan"),
                   within_2sigma=float(np.mean(np.abs(allp) < 2)) if len(allp) else float("nan"),
                   median_abs_pull=float(np.median(np.abs(allp))) if len(allp) else float("nan"),
                   rms_pull=float(np.sqrt(np.mean(allp**2))) if len(allp) else float("nan"))
    return dict(per_parameter=per, overall=overall, family_ok=int(sum(r["family_ok"] for r in rows)), n_maps=len(rows),
                mean_runtime_s=float(np.mean([r["runtime_s"] for r in rows])))


def pull_figure(rows: list[dict], summary: dict, platform: str, path: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fam = "IBM Plex Sans" if any("IBM Plex Sans" == f.name for f in font_manager.fontManager.ttflist) else "DejaVu Sans"
    plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": SURFACE, "savefig.facecolor": BG, "text.color": TEXT, "axes.labelcolor": MUTED,
                         "axes.edgecolor": GRID, "xtick.color": MUTED, "ytick.color": MUTED, "grid.color": GRID, "font.family": fam, "font.size": 10,
                         "axes.titlecolor": TEXT, "axes.spines.top": False, "axes.spines.right": False})
    names = list(summary["per_parameter"])
    n_pan = len(names) + 1
    ncol = min(n_pan, 4)
    nrow = int(np.ceil(n_pan / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol, 3.0 * nrow), squeeze=False)
    axes = axes.ravel()
    edges = np.linspace(-6, 6, 25)
    xs = np.linspace(-6, 6, 241)
    gauss = np.exp(-0.5 * xs**2) / np.sqrt(2 * np.pi)

    def panel(ax, pulls, title, stats):
        pulls = np.asarray(pulls, float)
        pulls = pulls[np.isfinite(pulls)]
        clipped = np.clip(pulls, -5.99, 5.99)
        ax.hist(clipped, bins=edges, density=True, color=BLUE, alpha=0.85, edgecolor=SURFACE, linewidth=0.6)
        ax.plot(xs, gauss, color=AMBER, lw=1.8, label="N(0, 1)")
        for v in (-1, 1):
            ax.axvline(v, color=MUTED, lw=0.8, ls=":")
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("pull = (fit − truth) / σ")
        ax.set_xlim(-6, 6)
        ax.grid(True, axis="y", lw=0.5, alpha=0.6)
        ax.text(0.98, 0.95, f"n = {stats['n']}\n|pull| < 1σ: {stats['within_1sigma']:.0%}\n|pull| < 2σ: {stats['within_2sigma']:.0%}\nrms {stats['rms_pull']:.2g}",
                transform=ax.transAxes, ha="right", va="top", fontsize=9, color=TEXT, family="monospace")

    allp = [r["pulls"][n] for r in rows for n in r.get("pulls", {})]
    panel(axes[0], allp, f"all scored parameters · {platform}", summary["overall"])
    for k, n in enumerate(names, start=1):
        panel(axes[k], [r["pulls"][n] for r in rows if n in r.get("pulls", {})], n, summary["per_parameter"][n])
    for ax in axes[n_pan:]:
        ax.set_visible(False)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(f"Uncertainty calibration · {summary['n_maps']} random {platform} maps · family right {summary['family_ok']}/{summary['n_maps']} · pulls clipped to ±6",
                 x=0.01, ha="left", fontsize=12, color=TEXT)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--platform", default="nv", choices=["nv", "dot", "ion", "atom", "circuit"])
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=500)
    ap.add_argument("--workers", type=int, default=4, help="maps fitted concurrently (one process each)")
    ap.add_argument("--maxiter", type=int, default=40)
    ap.add_argument("--out", default=None, help="JSON path (default artifacts/coverage_<platform>.json)")
    a = ap.parse_args()
    jobs = [(i, a.seed + i, a.platform, a.maxiter) for i in range(a.n)]
    t0 = time.time()
    rows = []
    if a.workers > 1:
        with ProcessPoolExecutor(max_workers=a.workers) as ex:
            for row in ex.map(run_case, jobs):
                rows.append(row)
                print(f"[{len(rows)}/{a.n}] truth={row['truth']:14s} found={str(row['found']):14s} ok={row['family_ok']} pulls=" + ", ".join(f"{k} {v:+.1f}" for k, v in row["pulls"].items()), flush=True)
    else:
        for job in jobs:
            row = run_case(job)
            rows.append(row)
            print(f"[{len(rows)}/{a.n}] truth={row['truth']:14s} found={str(row['found']):14s} ok={row['family_ok']} pulls=" + ", ".join(f"{k} {v:+.1f}" for k, v in row["pulls"].items()), flush=True)
    summary = summarise(rows)
    print(f"\nplatform {a.platform}: family right {summary['family_ok']}/{summary['n_maps']}, {summary['mean_runtime_s']:.1f} s/map, wall {time.time() - t0:.0f} s")
    print(f"{'parameter':14s} {'n':>4s} {'<1σ':>7s} {'<2σ':>7s} {'median|p|':>10s} {'rms':>8s} {'>5σ':>6s}")
    for n, s in summary["per_parameter"].items():
        print(f"{n:14s} {s['n']:4d} {s['within_1sigma']:7.2f} {s['within_2sigma']:7.2f} {s['median_abs_pull']:10.2f} {s['rms_pull']:8.2f} {s['frac_beyond_5sigma']:6.2f}")
    o = summary["overall"]
    print(f"{'overall':14s} {o['n']:4d} {o['within_1sigma']:7.2f} {o['within_2sigma']:7.2f} {o['median_abs_pull']:10.2f} {o['rms_pull']:8.2f}   (ideal 0.68 / 0.95)")
    out = Path(a.out or f"artifacts/coverage_{a.platform}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(dict(platform=a.platform, n=a.n, seed=a.seed, maxiter=a.maxiter, summary=summary, rows=rows), open(out, "w"), indent=1, default=float)
    fig_path = Path(f"artifacts/fig_pulls_{a.platform}.png")
    pull_figure(rows, summary, a.platform, fig_path)
    print(f"wrote {out} and {fig_path}")


if __name__ == "__main__":
    main()
