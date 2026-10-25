"""Build every ui.evidence figure for the three saved test cases, check them, print a one-line summary per figure
and write HTML previews to artifacts/evidence_preview_*.html.

    . .venv/bin/activate && python scripts/evidence_preview.py
"""
from __future__ import annotations

import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ui import evidence  # noqa: E402

assert "streamlit" not in sys.modules, "ui.evidence must not import streamlit"

CASES = [
    ("demo_A", ROOT / "artifacts/demo_A_report.json", ROOT / "artifacts/demo_A.npz",
     {"EJmax": 25.0, "EC": 0.3, "d": 0.3, "fr": 7.0, "g": 0.1, "period": 1.3, "offset": 0.2}),
    ("demo_B", ROOT / "artifacts/demo_B_report.json", ROOT / "artifacts/demo_B.npz",
     {"EJ": 6.0, "EC": 1.0, "EL": 0.6, "period": 1.1, "offset": -0.3}),
    ("nguyen2019", ROOT / "artifacts/nguyen2019_report_calibrated.json", ROOT / "data/demo/nguyen2019_devA.npz", None),
]
OUT = ROOT / "artifacts"


def n_marker_points(fig) -> int:
    n = 0
    for t in fig.data:
        if t.type == "scatter" and t.mode and "markers" in t.mode and t.x is not None:
            n += sum(1 for v in t.x if v is not None)
    return n


def save(fig, name: str) -> pathlib.Path:
    path = OUT / f"evidence_preview_{name}.html"
    fig.write_html(str(path), include_plotlyjs="cdn", config=dict(displaylogo=False))
    return path


def main() -> None:
    written = []
    for case, rep_path, npz_path, truth in CASES:
        report = json.load(open(rep_path))
        d = np.load(npz_path)
        image, x, f = d["image"], d["x"], d["f"]
        if truth is None and "truth_EJ" in d.files:  # the real data set ships the published values
            truth = {k: float(d[f"truth_{k}"]) for k in ("EJ", "EC", "EL") if f"truth_{k}" in d.files}
        if not report.get("derived"):  # the saved reports predate the derived block: build it the way agent.identify does
            from circuitsleuth import derived

            xg = report["x_grid"]
            report["derived"] = [dict(q, value=None if not np.isfinite(q["value"]) else float(q["value"]))
                                 for q in derived.derived_quantities(report["best"], (float(min(xg)), float(max(xg))))]
        best = report["best"]
        print(f"\n=== {case}: {best['hypothesis']} ({len(report['points'])} points, coupled={best['coupled']})")

        fig = evidence.residual_figure(report)
        assert fig is not None and len(fig.data) > 0, "residual_figure returned nothing"
        m = fig.layout.meta
        assert m["n_assigned"] == len(report["points"]), "every point must be assigned to a line"
        print(f"residual : {len(fig.data)} traces, {n_marker_points(fig)} markers drawn / {m['n_points']} points, {m['n_heldout']} on {m['n_heldout_columns']} held-out columns, "
              f"RMS {m['rms_MHz']} MHz, band ±{m['band_MHz']:.0f} MHz, {m['n_outside']} beyond ±{m['clip_MHz']:.0f} MHz, y range {list(fig.layout.yaxis.range)}")
        written.append(save(fig, f"{case}_residual"))

        fig = evidence.holdout_figure(report, image, x, f)
        assert fig is not None and len(fig.data) > 0 and fig.data[0].type == "heatmap"
        m = fig.layout.meta
        rep_n = (best.get("holdout") or {}).get("n_test_points")
        print(f"holdout  : {len(fig.data)} traces, {m['n_columns']} held-out columns, {m['n_test']} held-out points ({m['n_test_inlier']} within 3σ), "
              f"{m['n_rest']} others, report n_test_points={rep_n}, shapes={len(fig.layout.shapes)}; title '{m['title']}'")
        written.append(save(fig, f"{case}_holdout"))

        fig = evidence.crossing_zoom_figure(report, image, x, f)
        if best["coupled"]:
            assert fig is not None and len(fig.data) > 0, "coupled fit must produce a crossing zoom"
            m = fig.layout.meta
            heat = sum(1 for t in fig.data if t.type == "heatmap")
            print(f"crossing : {len(fig.data)} traces ({heat} heatmap crops), {m['n_crossings']} bare crossings at {m['crossings']}, shown {m['shown']}, "
                  f"2g = {m['two_g_MHz']} ± {m['sigma_two_g_MHz']} MHz, branch gap {m['branch_gap_MHz']} MHz, y range {list(fig.layout.yaxis.range)}")
            written.append(save(fig, f"{case}_crossing"))
        else:
            assert fig is None, "uncoupled fit must return None"
            print("crossing : None (best hypothesis has no resonator) - as documented")

        fig = evidence.pull_figure(report, truth)
        assert fig is not None and len(fig.data) > 0
        m = fig.layout.meta
        print("pull     : " + ", ".join(f"{k} {v:+.2f}σ" for k, v in m["pulls"].items()) + f"; {m['n_beyond']} beyond ±6; x range {list(fig.layout.xaxis.range)}")
        written.append(save(fig, f"{case}_pull"))

        cards = evidence.spec_card_data(report)
        assert set(cards) == {"Qubit", "Control axis", "Readout"} and sum(len(v) for v in cards.values()) == len(report["derived"])
        print("cards    : " + " · ".join(f"{k} {len(v)}" for k, v in cards.items()))
        for k, v in cards.items():
            for label, val in v:
                print(f"           [{k}] {label}: {val}")
    print("\nwrote:")
    for p in written:
        print(" ", p.relative_to(ROOT))


if __name__ == "__main__":
    main()
