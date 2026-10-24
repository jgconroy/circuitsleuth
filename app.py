"""CircuitSleuth - Streamlit UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import streamlit as st

from circuitsleuth import agent, classify, explain, extract, fit, synth
from ui import plots, schematic, theme

st.set_page_config(page_title="CircuitSleuth", layout="wide", initial_sidebar_state="expanded")
theme.inject()

from ui.demos import DEMOS, SAVED, make_demo  # noqa: E402
from circuitsleuth import models  # noqa: E402

PRESETS = {  # search effort -> agent settings
    "fast": dict(maxiter=30, n_starts=1, popsize=12, top_k=2, coupled_trials=1),
    "normal": dict(maxiter=50, n_starts=2, popsize=15, top_k=3, coupled_trials=2),
    "thorough": dict(maxiter=100, n_starts=3, popsize=20, top_k=3, coupled_trials=2),
}
PLOT = dict(width="stretch", theme=None, config=plots.CONFIG)


# ------------------------------------------------------------------ sidebar
# The three groups are created first so that the Search settings (which the Data group may override for the
# real-data demo) can be read before the Data widgets are drawn, while Data still appears on top.
data_box = st.sidebar.expander("Data", expanded=True)
search_box = st.sidebar.expander("Search", expanded=False)
explain_box = st.sidebar.expander("Explain", expanded=False)
st.sidebar.caption("Shazam for quantum chips. Every number comes from the physics fit: the CNN only proposes hypotheses, the language model only narrates.")

with data_box:
    platform_label = st.selectbox("Platform", [models.PLATFORMS[k]["label"] for k in models.PLATFORMS], index=0)
    platform = next(k for k, v in models.PLATFORMS.items() if v["label"] == platform_label)
    platform_families = list(models.PLATFORMS[platform]["families"])
    xlabel = "control (" + models.PLATFORMS[platform]["control_label"] + ")"
is_circuit = platform == "circuit"
with search_box:
    weights_available = classify.DEFAULT_PATH.exists() and is_circuit
    speed = st.select_slider("Search effort", ["fast", "normal", "thorough"], value="fast")
    use_clf = st.checkbox("Use CNN proposer to seed the search", value=weights_available, disabled=not weights_available,
                          help="Trained on synthetic circuit maps; only proposes hypotheses and starting points. All numbers come from the physics fit.")
    try_coupled = st.selectbox("Try a readout-resonator model", ["auto", "always", "never"], index=0, disabled=not is_circuit)
    families = st.multiselect("Candidate circuits", platform_families, default=platform_families) or platform_families
    workers = st.slider("CPU workers", 1, 8, 4)
    flux_calibrated = st.checkbox("Flux axis is already in flux quanta (Φ/Φ₀)", value=False, disabled=not is_circuit,
                                  help="Tick when the x axis is calibrated: the fit then holds period = 1 and offset = 0 instead of fitting them.")
preset = PRESETS[speed]
with explain_box:
    use_llm = st.checkbox("Explain with Claude", value=True)

truth = None
image = x = f = points = None
band = None
with data_box:
    source = st.radio("Data", ["Demo circuit", "Upload points CSV (x, f[, weight])", "Upload map (.npz or image)"], label_visibility="collapsed")
    if source == "Demo circuit":
        name = st.selectbox("Hidden circuit", list(DEMOS[platform]))
        seed = st.number_input("Random seed (surprise only)", 0, 10_000, 7) if DEMOS[platform][name] is None else None
        truth, r = make_demo(platform, name, seed)
        image, x, f = r["image"], r["x"], r["f"]
        if DEMOS[platform][name] == "real":
            band = (0.05, float(f[-1]))
            flux_calibrated = st.checkbox("Use the paper's flux axis (Φ/Φ₀) for this map", value=True)
            st.caption("Real data: the right half of the published map is very noisy; the paper's own fit gives EJ=3, EC=0.84, EL=1 GHz.")
    elif source.startswith("Upload points"):
        up = st.file_uploader("CSV with columns x, f, (weight)", type=["csv", "txt"])
        if up is not None:
            arr = np.genfromtxt(io.StringIO(up.getvalue().decode()), delimiter=",", skip_header=1 if up.getvalue()[:1].isalpha() else 0)
            arr = np.atleast_2d(arr)
            if arr.shape[1] == 2:
                arr = np.column_stack([arr, np.ones(len(arr))])
            points = arr[:, :3]
            image, x, f = agent.rasterize(points)
    else:
        up = st.file_uploader("NPZ (keys image, x, f) or a screenshot PNG/JPG of a map", type=["npz", "png", "jpg", "jpeg"])
        if up is not None:
            if up.name.endswith(".npz"):
                d = np.load(io.BytesIO(up.getvalue()))
                image, x, f = d["image"], d["x"], d["f"]
            else:
                from PIL import Image

                im = np.asarray(Image.open(io.BytesIO(up.getvalue())).convert("L"), float)
                image = im[::-1]  # row 0 is the top of the picture -> highest frequency
                c1, c2 = st.columns(2)
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
theme.header()
st.markdown("Upload a two-tone spectroscopy map. CircuitSleuth proposes circuit hypotheses, fits each against the detected "
            "features with a robust physics model (checked against scqubits), verifies on held-out flux columns, lists what it "
            "**cannot** explain, and tells you **where to measure next**.")

if image is None:
    st.info("Pick a demo circuit or upload data in the sidebar.")
    st.stop()

left, right = st.columns([1.15, 1])
with left:
    pts_preview = points if points is not None else extract.extract_points(image, x, f)
    map_slot = st.empty()
    map_slot.plotly_chart(plots.map_figure(image, x, f, points=pts_preview, title=f"input map · {len(pts_preview)} detected features", xlabel=xlabel), **PLOT)
    map_caption = st.empty()
with right:
    with st.container(border=True):
        st.subheader("Run")
        run = st.button("Identify circuit", type="primary", width="stretch")
        saved_path = None
        if source == "Demo circuit":
            key = name.split(" ")[0]
            if key in SAVED and os.path.exists(SAVED[key]):
                saved_path = SAVED[key]
        load_saved = st.button("Load precomputed result (normal preset)", width="stretch", disabled=saved_path is None,
                               help="The same pipeline run earlier with the 'normal' effort preset; identical output, no waiting.")
        status = st.empty()
        table = st.empty()
        spark_caption = st.empty()
        spark = st.empty()
        log_slot = st.empty()

if load_saved and saved_path:
    st.session_state["report"] = json.load(open(saved_path))
    st.session_state["truth"] = truth
    status.markdown(theme.status_html("loaded", f"{saved_path} (precomputed; runtime was {st.session_state['report'].get('runtime_s', '?')} s)", "green"),
                    unsafe_allow_html=True)

if run:
    cfg = agent.AgentConfig(use_classifier=use_clf, try_coupled=try_coupled if is_circuit else "never", workers=workers, families=tuple(families),
                            flux_calibrated=flux_calibrated and is_circuit, **preset)
    rows: list[dict] = []
    history: list[tuple[int, float]] = []
    log: list[str] = []
    state = dict(last=0.0, iter=0, t0=time.time())

    def say(label: str, text: str, kind: str = "") -> None:
        status.markdown(theme.status_html(label, text, kind), unsafe_allow_html=True)
        log.append(f"{time.time() - state['t0']:6.1f} s  {label:<10} {text}")
        log_slot.markdown(theme.log_html(log), unsafe_allow_html=True)

    def redraw(p: dict, title: str) -> None:
        map_slot.plotly_chart(plots.map_figure(image, x, f, points=pts_preview, lines=p["lines"], x_grid=x, title=title, xlabel=xlabel), **PLOT)
        if p.get("params"):
            map_caption.markdown(theme.pills(f"{k} = {v:.4g}" for k, v in p["params"].items()), unsafe_allow_html=True)
        if history:
            spark_caption.markdown(theme.caption(f"search loss · {p['hypothesis']} · iteration {history[-1][0]}"), unsafe_allow_html=True)
            spark.plotly_chart(plots.convergence_figure(history), **PLOT)

    def progress(kind, p):
        if kind == "features":
            say("features", f"{p['n_points']} features in {p['n_tracks']} tracks · flux-independent lines at {', '.join(f'{v:.2f}' for v in p['flat_lines']) or 'none'} GHz")
        elif kind == "proposals":
            say("proposals", "hypotheses to test: " + "; ".join(f"{h['h']}" + (f" (p={h['prob']:.2f})" if h.get("prob") else "") for h in p["hypotheses"]))
        elif kind == "fit_start":
            say("fitting", f"{p['hypothesis']} on {p['n_points']} features", "amber")
        elif kind == "fit_progress":
            state["iter"] += 1
            history.append((state["iter"], float(p["loss"])))
            now = time.time()
            if now - state["last"] > 0.7:  # throttle the live redraw
                state["last"] = now
                redraw(p, f"{p['hypothesis']} · searching · loss {p['loss']:.3f}")
        elif kind == "fit_done":
            s = p["summary"]
            rows.append(dict(hypothesis=p["hypothesis"], score=round(p["score"], 3), explained=f"{100*s['inlier_frac']:.0f}%", rms_MHz=s["rms_inliers_MHz"], note=p.get("note", "")))
            table.dataframe(rows, width="stretch", hide_index=True)
            redraw(dict(p, params=s.get("params")), f"{p['hypothesis']} · fitted · score {p['score']:.3f}")
            say("fitted", f"{p['hypothesis']}: score {p['score']:.3f}, {100*s['inlier_frac']:.0f}% explained, RMS {s['rms_inliers_MHz']:.1f} MHz" + (f" · {p['note']}" if p.get("note") else ""), "blue")
        elif kind == "calibration":
            say("symmetry", f"flux period ≈ {p['period']:.3g} control units; sweet spots at " + ", ".join(f"{a['x']:.3g} ({a['type']})" for a in p["axes"][:3]))
        elif kind == "skip":
            say("skip", f"{p['hypothesis']}: {p['reason']}")
        elif kind == "revise":
            say("revise", p["reason"], "amber")
        elif kind == "verify":
            ho = p["holdout"]
            if ho:
                say("hold-out", f"{100*ho['inlier_frac']:.0f}% of features in untouched flux columns explained, RMS {ho['rms_MHz']:.1f} MHz", "blue")
        elif kind == "warning":
            say("warning", p.get("message", ""), "red")

    t0 = time.time()
    rep = agent.identify(image, x, f, points=points, band=band, config=cfg, progress=progress)
    st.session_state["report"] = rep
    st.session_state["truth"] = truth
    say("done", f"{time.time()-t0:.0f} s", "green")

# ------------------------------------------------------------------ results
rep = st.session_state.get("report")
if rep and rep.get("status") == "ok":
    best = rep["best"]
    theme.rule()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Best hypothesis", best["hypothesis"])
    c2.metric("Flux-dependent features explained", f"{100*best.get('explained_dynamic', best['track_score']):.0f}%", help="Fraction of detected line segments (spurs excluded) that lie on a predicted transition")
    ho = best.get("holdout") or {}
    c3.metric("Held-out columns explained", f"{100*ho['inlier_frac']:.0f}%" if ho else "n/a", help="Columns excluded from the fit, predicted by the model")
    c4.metric("Residual (RMS)", f"{best['rms_inliers_MHz']:.1f} MHz")
    if best.get("note"):
        st.markdown(theme.note(best["note"]), unsafe_allow_html=True)
    conf = best.get("confidence")
    if isinstance(conf, dict) and conf.get("level"):
        kind = {"high": "green", "medium": "amber", "low": "red"}.get(conf["level"], "")
        st.markdown(theme.status_html(f"confidence: {conf['level']}", f"{conf.get('text', '')} (score gap {conf.get('gap', 0.0):+.3f})", kind), unsafe_allow_html=True)
    a, b = st.columns([1.15, 1])
    with a:
        boxes = [n for n in rep["next_measurements"] if n["kind"] == "look_here"]
        st.plotly_chart(plots.map_figure(image, x, f, points=np.array(rep["points"]), inlier=rep["inlier_mask"], lines=rep["lines"], x_grid=np.array(rep["x_grid"]), xlabel=xlabel, look_here=boxes,
                                         flat_lines=rep["flat_lines_GHz"], title=f"{best['hypothesis']} · fitted transitions, explained and unexplained features, where to look next"), **PLOT)
    with b:
        st.markdown(schematic.schematic_svg(best["family"], bool(best.get("coupled"))), unsafe_allow_html=True)
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
        st.plotly_chart(plots.hypothesis_bars(rep["hypotheses"]), **PLOT)
        with st.expander("Scores as a table"):
            st.dataframe([{"hypothesis": h["hypothesis"], "score": h["track_score"], "explained": f"{100*h['inlier_frac']:.0f}%", "RMS MHz": h["rms_inliers_MHz"]} for h in rep["hypotheses"]],
                         hide_index=True, width="stretch")
    physics = plots.potential_figure(best["family"], best["params"], bool(best.get("coupled")))
    if physics is None:  # non-circuit platforms: panels live in ui/platform_panels.py (optional)
        try:
            from ui import platform_panels

            physics = {"nv_ensemble": platform_panels.nv_levels_figure, "charge_qubit": platform_panels.dqd_figure}[best["family"]](best["params"])
        except Exception:
            physics = None
    if physics is not None:
        st.subheader("Circuit physics")
        st.plotly_chart(physics, **PLOT)
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
