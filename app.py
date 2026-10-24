"""CircuitSleuth - Streamlit UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import io
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from matplotlib.patches import Rectangle

from circuitsleuth import agent, classify, explain, extract, fit, synth

st.set_page_config(page_title="CircuitSleuth", page_icon="🔎", layout="wide")

DEMOS = {
    "A · tunable transmon through a 7 GHz readout resonator, plus a pump spur": dict(
        family="tunable_transmon", params=dict(EJmax=25.0, EC=0.3, d=0.3, fr=7.0, g=0.1), period=1.3, offset=0.2, f_lo=3.0, f_hi=9.0,
        spurs=[dict(f=5.1, amp=0.9, sideband=0.0)], seed=1),
    "B · fluxonium": dict(family="fluxonium", params=dict(EJ=6.0, EC=1.0, EL=0.6), period=1.1, offset=-0.3, f_lo=0.3, f_hi=9.0, spurs=[], seed=3),
    "C · fixed-frequency transmon with LO leakage": dict(family="transmon", params=dict(EJ=15.0, EC=0.28), period=1.0, offset=0.0, f_lo=3.0, f_hi=8.0,
                                                          spurs=[dict(f=6.2, amp=0.6, sideband=0.1)], seed=2),
    "D · surprise: random hidden circuit": None,
    "E · REAL DATA: fluxonium device A, Nguyen et al. PRX 2019 (arXiv:1810.11006)": "real",
}
LINE_STYLE = {"0-1": dict(lw=2.2), "0-2": dict(lw=1.2), "0-3": dict(lw=1.0), "0-2/2ph": dict(lw=1.2, ls="--"), "1-2": dict(lw=1.0, ls=":"), "res": dict(lw=1.0, ls="-.")}


def make_demo(name: str, seed: int | None = None):
    spec = DEMOS[name]
    cfg = synth.SynthConfig(nx=81, nf=400)
    if spec == "real":
        d = np.load("data/demo/nguyen2019_devA.npz")
        truth = dict(family="fluxonium", coupled=False, params=dict(EJ=float(d["truth_EJ"]), EC=float(d["truth_EC"]), EL=float(d["truth_EL"])), period=1.0, offset=0.0, spurs=[],
                     note="digitised from the paper's Fig. 2 (vector PDF); paper fit EJ=3, EC=0.84, EL=1 GHz; axis already in flux quanta so period should come out as 1")
        return truth, dict(image=d["image"], x=d["x"], f=d["f"])
    if spec is None:
        rng = np.random.default_rng(seed or 0)
        truth = synth.sample_truth(rng, synth.SynthConfig(nx=81, nf=400, noise_range=(0.03, 0.1)))
        r = synth.render(truth, cfg, rng)
        return truth, r
    truth = dict(family=spec["family"], coupled="fr" in spec["params"], params=dict(spec["params"]), period=spec["period"], offset=spec["offset"],
                 x_lo=-1.0, x_hi=1.0, f_lo=spec["f_lo"], f_hi=spec["f_hi"], linewidth=0.025, noise=0.06, two_photon=True, sign=1, spurs=list(spec["spurs"]))
    r = synth.render(truth, cfg, np.random.default_rng(spec["seed"]))
    return truth, r


def draw_map(image, x, f, points=None, inlier=None, lines=None, x_grid=None, boxes=None, title=None, flat_lines=None):
    fig, ax = plt.subplots(figsize=(7.2, 5.2), dpi=110)
    img = np.clip(extract.normalise(image), 0, 10)
    ax.imshow(img, extent=[x[0], x[-1], f[0], f[-1]], origin="lower", aspect="auto", cmap="magma", interpolation="nearest")
    if points is not None and len(points):
        if inlier is None:
            ax.scatter(points[:, 0], points[:, 1], s=6, c="#4fd1c5", alpha=0.7, lw=0, label="detected features")
        else:
            inl = np.asarray(inlier, bool)
            ax.scatter(points[inl, 0], points[inl, 1], s=7, c="#8ef58a", alpha=0.8, lw=0, label="explained")
            ax.scatter(points[~inl, 0], points[~inl, 1], s=10, c="#ff5c5c", alpha=0.9, lw=0, marker="x", label="unexplained")
    if lines is not None and x_grid is not None:
        for name, vals in lines.items():
            v = np.array([np.nan if u is None else u for u in vals], float)
            if np.all(~np.isfinite(v)):
                continue
            ax.plot(x_grid, v, color="white", alpha=0.9, label=name, **LINE_STYLE.get(name, {}))
    if boxes:
        for b in boxes:
            name = b.get("line")
            if lines is not None and x_grid is not None and name in lines:
                v = np.array([np.nan if u is None else u for u in lines[name]], float)
                m = (x_grid >= b["x_lo"]) & (x_grid <= b["x_hi"]) & np.isfinite(v)
                if m.any():
                    ax.plot(x_grid[m], v[m], color="#ffd166", lw=4, alpha=0.55, solid_capstyle="round")
                    ax.text(x_grid[m][0], v[m][0], f" look here: {name}", color="#ffd166", fontsize=8, va="bottom")
    if flat_lines:
        for fl in flat_lines:
            ax.axhline(fl, color="#ff5c5c", lw=0.6, alpha=0.5)
    ax.set_ylim(f[0], f[-1])
    ax.set_xlabel("control (flux bias, arbitrary units)")
    ax.set_ylabel("drive frequency (GHz)")
    if title:
        ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.3, labelcolor="white", facecolor="black")
    fig.tight_layout()
    return fig


# ------------------------------------------------------------------ sidebar
st.sidebar.title("CircuitSleuth")
st.sidebar.caption("Shazam for quantum chips: identify the circuit behind a spectroscopy map.")
source = st.sidebar.radio("Data", ["Demo circuit", "Upload points CSV (x, f[, weight])", "Upload map (.npz or image)"])
weights_available = classify.DEFAULT_PATH.exists()
speed = st.sidebar.select_slider("Search effort", ["fast", "normal", "thorough"], value="fast")
use_clf = st.sidebar.checkbox("Use CNN proposer to seed the search", value=weights_available, disabled=not weights_available,
                              help="Trained on synthetic maps; only proposes hypotheses and starting points. All numbers come from the physics fit.")
try_coupled = st.sidebar.selectbox("Try a readout-resonator model", ["auto", "always", "never"], index=0)
families = st.sidebar.multiselect("Candidate circuits", list(fit.models.FAMILIES), default=list(fit.models.FAMILIES))
workers = st.sidebar.slider("CPU workers", 1, 8, 1)
use_llm = st.sidebar.checkbox("Explain with Claude", value=True)
flux_calibrated = st.sidebar.checkbox("Flux axis is already in flux quanta (Φ/Φ₀)", value=False,
                                      help="Tick when the x axis is calibrated: the fit then holds period = 1 and offset = 0 instead of fitting them.")
maxiter = {"fast": 30, "normal": 60, "thorough": 100}[speed]

truth = None
image = x = f = points = None
band = None
if source == "Demo circuit":
    name = st.sidebar.selectbox("Hidden circuit", list(DEMOS))
    seed = st.sidebar.number_input("Random seed (surprise only)", 0, 10_000, 7) if DEMOS[name] is None else None
    truth, r = make_demo(name, seed)
    image, x, f = r["image"], r["x"], r["f"]
    if DEMOS[name] == "real":
        band = (0.05, float(f[-1]))
        flux_calibrated = st.sidebar.checkbox("Use the paper's flux axis (Φ/Φ₀) for this map", value=True)
        st.sidebar.caption("Real data: the right half of the published map is very noisy; the paper's own fit gives EJ=3, EC=0.84, EL=1 GHz.")
elif source.startswith("Upload points"):
    up = st.sidebar.file_uploader("CSV with columns x, f, (weight)", type=["csv", "txt"])
    if up is not None:
        arr = np.genfromtxt(io.StringIO(up.getvalue().decode()), delimiter=",", skip_header=1 if up.getvalue()[:1].isalpha() else 0)
        arr = np.atleast_2d(arr)
        if arr.shape[1] == 2:
            arr = np.column_stack([arr, np.ones(len(arr))])
        points = arr[:, :3]
        image, x, f = agent.rasterize(points)
else:
    up = st.sidebar.file_uploader("NPZ (keys image, x, f) or a screenshot PNG/JPG of a map", type=["npz", "png", "jpg", "jpeg"])
    if up is not None:
        if up.name.endswith(".npz"):
            d = np.load(io.BytesIO(up.getvalue()))
            image, x, f = d["image"], d["x"], d["f"]
        else:
            from PIL import Image

            im = np.asarray(Image.open(io.BytesIO(up.getvalue())).convert("L"), float)
            image = im[::-1]  # row 0 is the top of the picture -> highest frequency
            c1, c2 = st.sidebar.columns(2)
            x_lo = c1.number_input("x axis min", value=-1.0)
            x_hi = c2.number_input("x axis max", value=1.0)
            f_lo = c1.number_input("f axis min (GHz)", value=3.0)
            f_hi = c2.number_input("f axis max (GHz)", value=9.0)
            # thin to a sensible number of columns so that one column = one flux step
            ncols = min(image.shape[1], 121)
            idx = np.linspace(0, image.shape[1] - 1, ncols).round().astype(int)
            image = image[:, idx]
            x = np.linspace(x_lo, x_hi, image.shape[1])
            f = np.linspace(f_lo, f_hi, image.shape[0])

# ------------------------------------------------------------------ main
st.title("CircuitSleuth")
st.markdown("Upload a two-tone spectroscopy map. CircuitSleuth proposes circuit hypotheses, fits each against the detected "
            "features with a robust physics model (checked against scqubits), verifies on held-out flux columns, lists what it "
            "**cannot** explain, and tells you **where to measure next**.")

if image is None:
    st.info("Pick a demo circuit or upload data in the sidebar.")
    st.stop()

left, right = st.columns([1.15, 1])
with left:
    pts_preview = points if points is not None else extract.extract_points(image, x, f)
    st.pyplot(draw_map(image, x, f, points=pts_preview, title=f"input map · {len(pts_preview)} detected features"), clear_figure=True)
with right:
    st.subheader("Run")
    run = st.button("Identify circuit", type="primary", width="stretch")
    status = st.empty()
    live = st.empty()
    table = st.empty()

if run:
    cfg = agent.AgentConfig(maxiter=maxiter, use_classifier=use_clf, try_coupled=try_coupled, workers=workers, families=tuple(families), flux_calibrated=flux_calibrated)
    rows = []
    state = dict(last=0.0)
    with left:
        overlay = st.empty()

    def progress(kind, p):
        if kind == "features":
            status.info(f"{p['n_points']} features in {p['n_tracks']} tracks · flux-independent lines at {', '.join(f'{v:.2f}' for v in p['flat_lines']) or 'none'} GHz")
        elif kind == "proposals":
            status.info("Hypotheses to test: " + "; ".join(f"{h['h']}" + (f" (p={h['prob']:.2f})" if h.get("prob") else "") for h in p["hypotheses"]))
        elif kind == "fit_start":
            status.warning(f"Fitting **{p['hypothesis']}** on {p['n_points']} features …")
        elif kind == "fit_progress":
            now = time.time()
            if now - state["last"] > 0.7:
                state["last"] = now
                pp = ", ".join(f"{k}={v:.3g}" for k, v in p["params"].items())
                overlay.pyplot(draw_map(image, x, f, points=pts_preview, lines=p["lines"], x_grid=x, title=f"{p['hypothesis']} · searching · loss {p['loss']:.3f}\n{pp}"), clear_figure=True)
        elif kind == "fit_done":
            s = p["summary"]
            rows.append(dict(hypothesis=p["hypothesis"], score=round(p["score"], 3), explained=f"{100*s['inlier_frac']:.0f}%", rms_MHz=s["rms_inliers_MHz"]))
            table.dataframe(rows, width="stretch", hide_index=True)
            overlay.pyplot(draw_map(image, x, f, points=pts_preview, lines=p["lines"], x_grid=x, title=f"{p['hypothesis']} · fitted · score {p['score']:.3f}"), clear_figure=True)
        elif kind == "calibration":
            status.info(f"Symmetry analysis: flux period ≈ {p['period']:.3g} control units; sweet spots at " + ", ".join(f"{a['x']:.3g} ({a['type']})" for a in p['axes'][:3]))
        elif kind == "skip":
            status.info(f"Skipping {p['hypothesis']}: {p['reason']}")
        elif kind == "revise":
            status.warning("Revision: " + p["reason"])
        elif kind == "verify":
            ho = p["holdout"]
            if ho:
                status.info(f"Hold-out check: {100*ho['inlier_frac']:.0f}% of features in untouched flux columns explained, RMS {ho['rms_MHz']:.1f} MHz")

    t0 = time.time()
    rep = agent.identify(image, x, f, points=points, band=band, config=cfg, progress=progress)
    st.session_state["report"] = rep
    st.session_state["truth"] = truth
    status.success(f"Done in {time.time()-t0:.0f} s")

rep = st.session_state.get("report")
if rep and rep.get("status") == "ok":
    best = rep["best"]
    st.divider()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best hypothesis", best["hypothesis"])
    c2.metric("Features explained", f"{100*best['track_score']:.0f}%")
    ho = best.get("holdout") or {}
    c3.metric("Held-out columns explained", f"{100*ho['inlier_frac']:.0f}%" if ho else "n/a", help="Columns excluded from the fit, predicted by the model")
    c4.metric("Residual (RMS)", f"{best['rms_inliers_MHz']:.1f} MHz")
    a, b = st.columns([1.15, 1])
    with a:
        boxes = [n for n in rep["next_measurements"] if n["kind"] == "look_here"]
        st.pyplot(draw_map(image, x, f, points=np.array(rep["points"]), inlier=rep["inlier_mask"], lines=rep["lines"], x_grid=np.array(rep["x_grid"]), boxes=boxes,
                           flat_lines=rep["flat_lines_GHz"], title=f"{best['hypothesis']}: fitted transitions, explained/unexplained features, where to look next"), clear_figure=True)
    with b:
        st.subheader("Recovered Hamiltonian")
        st.table([{"parameter": k, "value": v} for k, v in best["params_pretty"].items()])
        truth = st.session_state.get("truth")
        if truth is not None and st.toggle("Reveal hidden truth (demo)"):
            rows = []
            tp = dict(truth["params"], period=truth["period"], offset=truth["offset"])
            for k, v in best["params"].items():
                tv = tp.get(k)
                rows.append({"parameter": k, "fit": f"{v:.4g}", "truth": f"{tv:.4g}" if tv is not None else "-", "error": f"{100*abs(v-tv)/abs(tv):.1f}%" if tv not in (None, 0) else "-"})
            st.table(rows)
            st.caption(f"truth: {truth['family']}{' + resonator' if truth['coupled'] else ''}; spurs at {[s['f'] for s in truth['spurs']]} GHz" + (f" · {truth['note']}" if truth.get("note") else ""))
        st.subheader("Hypotheses tested")
        st.dataframe([{"hypothesis": h["hypothesis"], "score": h["track_score"], "explained": f"{100*h['inlier_frac']:.0f}%", "RMS MHz": h["rms_inliers_MHz"]} for h in rep["hypotheses"]], hide_index=True, width="stretch")
    st.subheader("What the model cannot explain")
    if rep["unexplained"]:
        for u in rep["unexplained"]:
            st.markdown(f"- **{u['f_lo']:.2f}–{u['f_hi']:.2f} GHz** ({u['n_points']} points, {'flat' if u['flat'] else 'flux-dependent'}): {u['interpretation']}")
    else:
        st.markdown("Every detected feature is accounted for.")
    st.subheader("Where to look next")
    if rep["next_measurements"]:
        for n in rep["next_measurements"]:
            if n["kind"] == "look_here":
                st.markdown(f"- **Sweep {n['f_lo']:.2f}–{n['f_hi']:.2f} GHz** for control values {n['x_lo']:.3g} to {n['x_hi']:.3g}: {n['why']}")
            elif n["kind"] == "extend_band":
                st.markdown(f"- **Extend the band to {n['f_lo']:.2f}–{n['f_hi']:.2f} GHz**: {n['why']}")
            else:
                st.markdown(f"- **Constrain {n['parameter']}** (currently " + (f"±{100*n['rel_sigma']:.0f}%" if n['rel_sigma'] < 5 else "unconstrained") + f"): {n['why']}")
    else:
        st.markdown("No obvious gaps: every predicted line inside the band has data under it.")
    st.subheader("Plain-English summary")
    with st.spinner("writing summary …"):
        ex = explain.explain(rep, use_llm=use_llm)
    st.write(ex["summary"])
    if ex["caveats"]:
        st.markdown("**Caveats**\n" + "\n".join(f"- {c}" for c in ex["caveats"]))
    if ex["next_steps"]:
        st.markdown("**Next steps**\n" + "\n".join(f"- {c}" for c in ex["next_steps"]))
    st.caption(f"summary source: {ex.get('source')}" + (f" · {ex['llm_error']}" if ex.get("llm_error") else ""))
    st.download_button("Download report (JSON)", agent.to_json(rep), file_name="circuitsleuth_report.json", mime="application/json")
elif rep:
    st.error(rep.get("error"))
