"""CircuitSleuth - Streamlit UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import streamlit as st

from circuitsleuth import agent, classify, derived, explain, extract, fit, synth
from ui import importers, plots, schematic, theme

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
FREQ_UNITS = list(importers.FREQ_TO_GHZ)


# ------------------------------------------------------------------ helpers
def known_param_names(families: list[str], is_circuit: bool) -> list[str]:
    """Every parameter a user may constrain, de-duplicated across the platform's families:
    circuit parameters first, then the resonator, then the control-axis calibration."""
    names: list[str] = []

    def add(n: str) -> None:
        if n not in names:
            names.append(n)

    for fam in families:
        for n in models.FAMILIES[fam]["params"]:
            add(n)
    for fam in families:
        for n in models.hypothesis_params(fam, coupled=is_circuit):
            add(n)
    for fam in families:
        for n in models.control_params(fam):
            add(n)
    return names


def known_default(name: str) -> float:
    """A sensible starting value: the (geometric) middle of the prior bounds."""
    lo = hi = None
    for fam in models.FAMILIES.values():
        if name in fam["bounds"]:
            lo, hi = fam["bounds"][name]
            break
    else:
        lo, hi = models.RESONATOR_BOUNDS.get(name) or models.CONTROL_BOUNDS.get(name) or (None, None)
    if lo is None:
        return 1.0 if name == "period" else 0.0
    return float(np.sqrt(lo * hi)) if name in fit.LOG_PARAMS and lo > 0 else float(0.5 * (lo + hi))


def fmt_sig(v, unit: str = "") -> str:
    if v is None or not np.isfinite(v):
        return "–"
    return f"{float(v):.4g} {unit}".strip()


def remember_run(rep: dict | None, source_name: str, platform_label: str, note: str = "") -> None:
    """Append one row to the session history."""
    if not rep or rep.get("status") != "ok":
        return
    best = rep["best"]
    st.session_state.setdefault("history", []).append({
        "time": time.strftime("%H:%M:%S"), "data source": source_name, "platform": platform_label,
        "best hypothesis": best["hypothesis"] + (f" ({note})" if note else ""),
        "key parameters": ", ".join(f"{k} = {v:.4g}" for k, v in best["params"].items()),
        "score": round(float(best.get("track_score", best.get("score", 0.0))), 3),
    })


def show_history() -> None:
    with st.expander("This session"):
        slot = st.empty()
        if st.button("Clear"):
            st.session_state["history"] = []
        hist = st.session_state.get("history", [])
        if hist:
            slot.dataframe(hist, hide_index=True, width="stretch")
        else:
            slot.caption("No runs yet in this session.")


def index_axes(image: np.ndarray, key: str, need_x: bool = True, need_f: bool = True):
    """Axis ranges typed in by hand when a file carries none (screenshots, bare HDF5 arrays)."""
    c1, c2 = st.columns(2)
    x_lo = c1.number_input("x axis min", value=-1.0, key=f"{key}_xlo") if need_x else -1.0
    x_hi = c2.number_input("x axis max", value=1.0, key=f"{key}_xhi") if need_x else 1.0
    f_lo = c1.number_input("f axis min (GHz)", value=3.0, key=f"{key}_flo") if need_f else 3.0
    f_hi = c2.number_input("f axis max (GHz)", value=9.0, key=f"{key}_fhi") if need_f else 9.0
    return np.linspace(x_lo, x_hi, image.shape[1]), np.linspace(f_lo, f_hi, image.shape[0])


def finish_map(image, x, f, key: str, unit_select: bool = True):
    """Clean-up shared by every 2-D input: frequency unit, ascending axes, crop windows, column thinning."""
    image = np.asarray(image, float)
    x, f = np.asarray(x, float).ravel(), np.asarray(f, float).ravel()
    unit = "GHz"
    if unit_select:
        unit = st.selectbox("Frequency unit", FREQ_UNITS, index=0, key=f"{key}_unit", help="The frequency axis is converted to GHz before the fit.")
        f = f * importers.FREQ_TO_GHZ[unit]
    image, x, f = importers.orient(image, x, f)
    f_lo, f_hi, x_lo, x_hi = float(f[0]), float(f[-1]), float(x[0]), float(x[-1])
    tag = f"{key}_{unit}_{image.shape[0]}x{image.shape[1]}"  # a new slider whenever the data or the unit changes
    f_win = x_win = None
    if f_hi > f_lo:
        f_win = st.slider("Frequency window (GHz)", f_lo, f_hi, (f_lo, f_hi), step=(f_hi - f_lo) / 200, format="%.3f", key=f"{tag}_fwin")
    if x_hi > x_lo:
        x_win = st.slider("Control window", x_lo, x_hi, (x_lo, x_hi), step=(x_hi - x_lo) / 200, format="%.3g", key=f"{tag}_xwin")
    image, x, f, ok = importers.crop(image, x, f, f_win, x_win)
    if not ok:
        st.warning("Window too small: at least 10 frequency rows and 3 control columns are needed, so the full map is used.")
    image, x = importers.thin_columns(image, x)
    return image, x, f


# ------------------------------------------------------------------ sidebar
# The three groups are created first so that the Search settings (which the Data group may override for the
# real-data demo) can be read before the Data widgets are drawn, while Data still appears on top.
data_box = st.sidebar.expander("Data", expanded=True)
search_box = st.sidebar.expander("Search", expanded=False)
explain_box = st.sidebar.expander("Explain", expanded=False)
st.sidebar.caption("Shazam for quantum chips. Every number comes from the physics fit: the CNN only proposes hypotheses, the language model only narrates.")
st.sidebar.caption("Many devices? `python batch.py <folder>` writes one row per map.")

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
    # known parameters: design values or a single-tone resonator scan narrow the search to value ± tolerance
    param_bounds: dict[str, tuple[float, float]] = {}
    known_store: dict = st.session_state.setdefault("known_store", {})
    if st.toggle("Known parameters", key="known_toggle", help="Constrain parameters you already know (design values, a single-tone resonator scan). "
                                                               "The fit then searches only inside value ± tolerance; a tolerance of 0 % fixes the value."):
        KCOLS = [0.9, 0.4, 1.1, 0.8]  # name | tick | value | tolerance
        for c, t in zip(st.columns(KCOLS, gap="small"), ("known", "", "value", "± %")):
            c.markdown(theme.caption(t), unsafe_allow_html=True)
        for n in known_param_names(platform_families, is_circuit):
            prev = known_store.get(n, {})
            c0, c1, c2, c3 = st.columns(KCOLS, gap="small", vertical_alignment="center")
            c0.markdown(f'<div class="cs-param">{n}</div>', unsafe_allow_html=True)
            k = c1.checkbox("known", value=bool(prev.get("known", False)), key=f"known_{n}", label_visibility="collapsed")
            v = c2.number_input("value", value=float(prev.get("value", known_default(n))), format="%.4g", key=f"known_val_{n}", label_visibility="collapsed")
            t = c3.number_input("± tolerance (%)", min_value=0.0, max_value=100.0, value=float(prev.get("tol", 10.0)), step=1.0, format="%g", key=f"known_tol_{n}",
                                label_visibility="collapsed")
            known_store[n] = dict(known=k, value=float(v), tol=float(t))
            if k:
                param_bounds[n] = (float(v), float(v)) if t == 0 else (float(v) * (1 - t / 100), float(v) * (1 + t / 100))
preset = PRESETS[speed]
with explain_box:
    use_llm = st.checkbox("Explain with Claude", value=True)
    design_values: dict[str, float] = {}
    if st.toggle("Compare with design values", key="design_toggle", help="Enter the design values of the identified circuit; the result then shows the deviation of every fitted parameter."):
        prev_rep = st.session_state.get("report")
        if prev_rep and prev_rep.get("status") == "ok":
            pb = prev_rep["best"]
            st.caption(f"Design values for {pb['hypothesis']}; leave a field empty to skip it.")
            for n in pb["params"]:
                dv = st.number_input(n, value=None, placeholder="design value", format="%.4g", key=f"design_{pb['family']}_{int(bool(pb.get('coupled')))}_{n}")
                if dv is not None:
                    design_values[n] = float(dv)
        else:
            st.caption("Run an identification first; fields appear for every parameter of the best hypothesis.")

truth = None
image = x = f = points = None
band = None
source_name = "—"
with data_box:
    source = st.radio("Data", ["Demo circuit", "Upload points CSV (x, f[, weight])", "Upload map (.npz or image)",
                               "Upload 2-D grid CSV (first row = control values, first column = frequencies)"], label_visibility="collapsed")
    if source == "Demo circuit":
        name = st.selectbox("Hidden circuit", list(DEMOS[platform]))
        source_name = name
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
            source_name = up.name
            arr = np.genfromtxt(io.StringIO(up.getvalue().decode()), delimiter=",", skip_header=1 if up.getvalue()[:1].isalpha() else 0)
            arr = np.atleast_2d(arr)
            if arr.shape[1] == 2:
                arr = np.column_stack([arr, np.ones(len(arr))])
            points = arr[:, :3]
            image, x, f = agent.rasterize(points)
    elif source.startswith("Upload 2-D grid"):
        up = st.file_uploader("CSV grid: control values across the first row, frequencies down the first column", type=["csv", "txt"])
        if up is not None:
            source_name = up.name
            try:
                image, x0, f0 = importers.read_grid_csv(up.getvalue())
            except Exception as e:
                st.error(f"Could not parse the grid: {e}")
                image = None
            else:
                if st.checkbox("Transpose (frequency is along columns)", key="grid_T"):
                    image, x0, f0 = image.T, f0, x0
                image, x, f = finish_map(image, x0, f0, "grid")
    else:
        up = st.file_uploader("NPZ (keys image, x, f) or a screenshot PNG/JPG of a map", type=["npz", "png", "jpg", "jpeg", "h5", "hdf5"],
                              help="HDF5 (.h5/.hdf5) files are accepted too: pick the 2-D dataset that holds the map and, when present, the 1-D datasets of the two axes.")
        if up is not None:
            source_name = up.name
            lower = up.name.lower()
            if lower.endswith(".npz"):
                d = np.load(io.BytesIO(up.getvalue()))
                image, x0, f0 = np.asarray(d["image"], float), np.asarray(d["x"], float).ravel(), np.asarray(d["f"], float).ravel()
                if st.checkbox("Transpose (frequency is along columns)", key="npz_T"):
                    image = image.T
                if not importers.axes_match(image, x0, f0):
                    st.warning(f"The map is {image.shape[0]}×{image.shape[1]} but x has {len(x0)} and f has {len(f0)} entries; index ranges are used instead.")
                    x0, f0 = index_axes(image, "npz")
                image, x, f = finish_map(image, x0, f0, "npz")
            elif lower.endswith((".h5", ".hdf5")):
                data = up.getvalue()
                try:
                    two, one = importers.h5_catalog(data)
                except Exception as e:
                    st.error(f"Could not read the HDF5 file: {e}")
                    two, one = {}, {}
                if not two:
                    st.warning("No 2-D numeric dataset found in this file.")
                else:
                    opts = [f"{k}  ({v[0]} × {v[1]})" for k, v in two.items()]
                    pick = st.selectbox("Map dataset (2-D)", opts, key="h5_map")
                    path = list(two)[opts.index(pick)]
                    image = importers.h5_read(data, path)
                    if st.checkbox("Transpose (frequency is along columns)", key="h5_T"):
                        image = image.T
                    nf_, nx_ = image.shape
                    x_opts = ["index range"] + [k for k, v in one.items() if v[0] == nx_]
                    f_opts = ["index range"] + [k for k, v in one.items() if v[0] == nf_]
                    xk = st.selectbox("Control axis dataset (1-D)", x_opts, index=min(1, len(x_opts) - 1), key=f"h5_x_{path}_{nx_}")
                    fk = st.selectbox("Frequency axis dataset (1-D)", f_opts, index=min(1, len(f_opts) - 1), key=f"h5_f_{path}_{nf_}")
                    xi, fi = index_axes(image, "h5", need_x=xk == "index range", need_f=fk == "index range")
                    x0 = xi if xk == "index range" else importers.h5_read(data, xk)
                    f0 = fi if fk == "index range" else importers.h5_read(data, fk)
                    image, x, f = finish_map(image, x0, f0, "h5")
            else:
                from PIL import Image

                im = np.asarray(Image.open(io.BytesIO(up.getvalue())).convert("L"), float)
                image = im[::-1]  # row 0 is the top of the picture -> highest frequency
                if st.checkbox("Transpose (frequency is along columns)", key="img_T"):
                    image = image.T
                x0, f0 = index_axes(image, "img")  # the axis ranges read off the picture, in GHz
                image, x, f = finish_map(image, x0, f0, "img", unit_select=False)

# ------------------------------------------------------------------ main
theme.header()
st.markdown("Upload a two-tone spectroscopy map. CircuitSleuth proposes circuit hypotheses, fits each against the detected "
            "features with a robust physics model (checked against scqubits), verifies on held-out flux columns, lists what it "
            "**cannot** explain, and tells you **where to measure next**.")

if image is None:
    st.info("Pick a demo circuit or upload data in the sidebar.")
    show_history()
    st.stop()

left, right = st.columns([1.15, 1])
with left:
    pts_preview = points if points is not None else extract.extract_points(image, x, f)
    map_slot = st.empty()
    map_slot.plotly_chart(plots.map_figure(image, x, f, points=pts_preview, title=f"input map · {len(pts_preview)} detected features", xlabel=xlabel), **PLOT)
    map_caption = st.empty()
with right:
    if param_bounds:
        st.markdown(theme.pill("known") + theme.pills((f"{n} = {known_store[n]['value']:.4g}" + (" fixed" if known_store[n]["tol"] == 0 else f" ± {known_store[n]['tol']:g}%")
                                                       for n in param_bounds), "amber"), unsafe_allow_html=True)
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
    remember_run(st.session_state["report"], source_name, platform_label, note="precomputed")

if run:
    cfg = agent.AgentConfig(use_classifier=use_clf, try_coupled=try_coupled if is_circuit else "never", workers=workers, families=tuple(families),
                            flux_calibrated=flux_calibrated and is_circuit, param_bounds=dict(param_bounds), **preset)
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
    remember_run(rep, source_name, platform_label, note=f"{speed}" + (", constrained" if param_bounds else ""))

# ------------------------------------------------------------------ results
rep = st.session_state.get("report")
if rep and rep.get("status") == "ok":
    best = rep["best"]
    xg = np.array(rep["x_grid"], float)
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
        gap = conf.get("gap")
        gap_text = f" (score gap {gap:+.3f})" if isinstance(gap, (int, float)) else ""  # None when only one family was considered
        st.markdown(theme.status_html(f"confidence: {conf['level']}", f"{conf.get('text', '')}{gap_text}", kind), unsafe_allow_html=True)
    a, b = st.columns([1.15, 1])
    with a:
        final_slot = st.empty()  # drawn after the operating-point tool below has chosen its markers
    markers: list[tuple[float, float]] = []
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
        design_rows = [(k, f"{best['params'][k]:.4g}", f"{dv:.4g}", f"{100 * (best['params'][k] - dv) / abs(dv):+.1f}%" if dv else "–")
                       for k, dv in design_values.items() if k in best["params"]]
        if design_rows:
            st.subheader("Fit vs design")
            st.markdown(theme.table_html(("parameter", "fit", "design", "deviation"), design_rows, numeric=(1, 2, 3)), unsafe_allow_html=True)
        st.subheader("Derived quantities")
        dq = rep.get("derived")
        if dq is None:  # older saved reports: compute on the fly
            try:
                dq = derived.derived_quantities(best, (float(xg.min()), float(xg.max())))
            except Exception as e:
                dq = []
                st.caption(f"derived quantities unavailable: {type(e).__name__}: {e}")
        if dq:
            st.markdown(theme.table_html(("quantity", "value", "note"), [(d["name"], fmt_sig(d.get("value"), d.get("unit", "")), d.get("note", "")) for d in dq],
                                         muted=(2,), numeric=(1,)), unsafe_allow_html=True)
        else:
            st.caption("No derived quantities for this result.")
        fam = best["family"]
        if models.platform_of(fam) == "circuit" and models.FAMILIES[fam]["flux_dependent"]:
            band_rep = rep.get("band_GHz") or [float(f[0]), float(f[-1])]
            target = st.number_input("Target f01 (GHz)", value=float(round(0.5 * (band_rep[0] + band_rep[1]), 2)), step=0.05, format="%.3f",
                                     help="Where to bias the device: control values at which the fitted 0-1 transition equals this frequency.")
            try:
                sols = derived.control_for_frequency(best, float(target), (float(xg.min()), float(xg.max())))
            except Exception:
                sols = []
            if sols:
                st.markdown(f"f01 = {target:.3f} GHz at control = " + ", ".join(f"{s:.3f}" for s in sols).replace("-", "−"))
                markers = [(float(s), float(target)) for s in sols]
            else:
                st.markdown(theme.caption(f"the fitted 0-1 line does not reach {target:.3f} GHz inside the measured control range"), unsafe_allow_html=True)
        st.subheader("Hypotheses tested")
        st.plotly_chart(plots.hypothesis_bars(rep["hypotheses"]), **PLOT)
        with st.expander("Scores as a table"):
            st.dataframe([{"hypothesis": h["hypothesis"], "score": h["track_score"], "explained": f"{100*h['inlier_frac']:.0f}%", "RMS MHz": h["rms_inliers_MHz"]} for h in rep["hypotheses"]],
                         hide_index=True, width="stretch")
    boxes = [n for n in rep["next_measurements"] if n["kind"] == "look_here"]
    fig_final = plots.map_figure(image, x, f, points=np.array(rep["points"]), inlier=rep["inlier_mask"], lines=rep["lines"], x_grid=xg, xlabel=xlabel, look_here=boxes,
                                 flat_lines=rep["flat_lines_GHz"], title=f"{best['hypothesis']} · fitted transitions, explained and unexplained features, where to look next",
                                 markers=markers or None)
    final_slot.plotly_chart(fig_final, **PLOT)
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
    # exports
    try:
        csv_text = derived.curves_csv(rep)
    except Exception:
        csv_text = None
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download report (JSON)", agent.to_json(rep), file_name="circuitsleuth_report.json", mime="application/json", width="stretch")
    e2.download_button("Download fitted curves (CSV)", csv_text or "", file_name="circuitsleuth_curves.csv", mime="text/csv", disabled=csv_text is None, width="stretch")
    e3.download_button("Download annotated map (HTML)", fig_final.to_html(include_plotlyjs="cdn"), file_name="circuitsleuth_map.html", mime="text/html", width="stretch")
    with st.expander("Reproduce in scqubits"):
        st.markdown(theme.caption("Copy scqubits snippet: the copy button appears at the top right of the block when you hover over it."), unsafe_allow_html=True)
        code = rep.get("scqubits_code")
        if not code:
            try:
                code = derived.scqubits_code(best)
            except Exception as e:
                code = f"# unavailable: {type(e).__name__}: {e}"
        st.code(code, language="python")
elif rep:
    st.error(rep.get("error"))

show_history()
