"""qĤolmes (formerly CircuitSleuth) - Streamlit UI.  Run:  streamlit run app.py"""
from __future__ import annotations

import io
import json
import os
import time

import numpy as np
import streamlit as st

from circuitsleuth import agent, classify, derived, explain, extract, fit, synth
from ui import importers, plots, schematic, theme

try:  # the evidence figures are written by another engineer; the app works without them
    from ui import evidence
except Exception:  # noqa: BLE001
    evidence = None

_HERE = os.path.dirname(os.path.abspath(__file__))
try:  # brand icon in the browser tab (static/brand comes from the brand kit; falls back to an emoji)
    from PIL import Image as _PILImage

    _ICON = _PILImage.open(os.path.join(_HERE, "static", "brand", "icon-192.png"))
except Exception:  # noqa: BLE001
    _ICON = "🎩"
st.set_page_config(page_title="qĤolmes: Hamiltonians, Deduced From Spectra", page_icon=_ICON, layout="wide", initial_sidebar_state="expanded")
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
    with st.expander("This Session"):
        slot = st.empty()
        if st.button("Clear"):
            st.session_state["history"] = []
        hist = st.session_state.get("history", [])
        if hist:
            cols = ("time", "data source", "platform", "best hypothesis", "key parameters", "score")
            slot.markdown(theme.table_html(cols, [[h.get(c, "") for c in cols] for h in hist], muted=(1, 4), numeric=(0, 5)), unsafe_allow_html=True)
        else:
            slot.caption("No runs yet in this session.")


def index_axes(image: np.ndarray, key: str, need_x: bool = True, need_f: bool = True):
    """Axis ranges typed in by hand when a file carries none (screenshots, bare HDF5 arrays)."""
    c1, c2 = st.columns(2)
    x_lo = c1.number_input("Control Axis Min", value=-1.0, key=f"{key}_xlo") if need_x else -1.0
    x_hi = c2.number_input("Control Axis Max", value=1.0, key=f"{key}_xhi") if need_x else 1.0
    f_lo = c1.number_input("Frequency Axis Min (GHz)", value=3.0, key=f"{key}_flo") if need_f else 3.0
    f_hi = c2.number_input("Frequency Axis Max (GHz)", value=9.0, key=f"{key}_fhi") if need_f else 9.0
    return np.linspace(x_lo, x_hi, image.shape[1]), np.linspace(f_lo, f_hi, image.shape[0])


def finish_map(image, x, f, key: str, unit_select: bool = True):
    """Clean-up shared by every 2-D input: frequency unit, ascending axes, crop windows, column thinning."""
    image = np.asarray(image, float)
    x, f = np.asarray(x, float).ravel(), np.asarray(f, float).ravel()
    unit = "GHz"
    if unit_select:
        unit = st.selectbox("Frequency Unit", FREQ_UNITS, index=0, key=f"{key}_unit", help="The frequency axis is converted to GHz before the fit.")
        f = f * importers.FREQ_TO_GHZ[unit]
    image, x, f = importers.orient(image, x, f)
    f_lo, f_hi, x_lo, x_hi = float(f[0]), float(f[-1]), float(x[0]), float(x[-1])
    tag = f"{key}_{unit}_{image.shape[0]}x{image.shape[1]}"  # a new slider whenever the data or the unit changes
    f_win = x_win = None
    if f_hi > f_lo:
        f_win = st.slider("Frequency Window (GHz)", f_lo, f_hi, (f_lo, f_hi), step=(f_hi - f_lo) / 200, format="%.3f", key=f"{tag}_fwin")
    if x_hi > x_lo:
        x_win = st.slider("Control Window", x_lo, x_hi, (x_lo, x_hi), step=(x_hi - x_lo) / 200, format="%.3g", key=f"{tag}_xwin")
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
    _plat_keys = list(models.PLATFORMS)
    _want_pl = st.query_params.get("platform")  # ?platform=ion picks the platform (pitch links, screenshots)
    platform_label = st.selectbox("Platform", [models.PLATFORMS[k]["label"] for k in _plat_keys], index=_plat_keys.index(_want_pl) if _want_pl in _plat_keys else 0,
                                  format_func=theme.title_case)
    platform = next(k for k, v in models.PLATFORMS.items() if v["label"] == platform_label)
    platform_families = list(models.PLATFORMS[platform]["families"])
    xlabel = models.PLATFORMS[platform]["control_label"]  # e.g. "flux bias (arbitrary units)"
    ylabel = models.PLATFORMS[platform].get("freq_label", "drive frequency (GHz)")
is_circuit = platform == "circuit"
with search_box:
    weights_available = classify.DEFAULT_PATH.exists() and is_circuit
    speed = st.select_slider("Search Effort", ["fast", "normal", "thorough"], value="fast", format_func=str.capitalize)
    use_clf = st.checkbox("Use CNN Proposer to Seed the Search", value=weights_available, disabled=not weights_available,
                          help="Trained on synthetic circuit maps; only proposes hypotheses and starting points. All numbers come from the physics fit.")
    try_coupled = st.selectbox("Try a Readout-Resonator Model", ["auto", "always", "never"], index=0, disabled=not is_circuit, format_func=str.capitalize)
    families = st.multiselect("Candidate Circuits", platform_families, default=platform_families,
                              format_func=lambda k: theme.title_case(models.FAMILIES[k]["label"])) or platform_families
    workers = st.slider("CPU Workers", 1, 8, 4)
    flux_calibrated = st.checkbox(r"Flux Axis Is Already in Flux Quanta ($\Phi/\Phi_0$)", value=False, disabled=not is_circuit,
                                  help="Tick when the x axis is calibrated: the fit then holds period = 1 and offset = 0 instead of fitting them.")
    # known parameters: design values or a single-tone resonator scan narrow the search to value ± tolerance
    param_bounds: dict[str, tuple[float, float]] = {}
    known_store: dict = st.session_state.setdefault("known_store", {})
    if st.toggle("Known Parameters", key="known_toggle", help="Constrain parameters you already know (design values, a single-tone resonator scan). "
                                                               "The fit then searches only inside value ± tolerance; a tolerance of 0 % fixes the value."):
        KCOLS = [0.9, 0.4, 1.1, 0.8]  # name | tick | value | tolerance
        for c, t in zip(st.columns(KCOLS, gap="small"), ("Known", "", "Value", "± %")):
            c.markdown(theme.caption(t), unsafe_allow_html=True)
        for n in known_param_names(platform_families, is_circuit):
            prev = known_store.get(n, {})
            c0, c1, c2, c3 = st.columns(KCOLS, gap="small", vertical_alignment="center")
            c0.markdown(f"${theme.param_tex(n)}$")
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
    if st.toggle("Compare with Design Values", key="design_toggle", help="Enter the design values of the identified circuit; the result then shows the deviation of every fitted parameter."):
        prev_rep = st.session_state.get("report")
        if prev_rep and prev_rep.get("status") == "ok":
            pb = prev_rep["best"]
            st.caption(f"Design values for {pb['hypothesis']}; leave a field empty to skip it.")
            for n in pb["params"]:
                dv = st.number_input(f"${theme.param_tex(n)}$", value=None, placeholder="design value", format="%.4g", key=f"design_{pb['family']}_{int(bool(pb.get('coupled')))}_{n}")
                if dv is not None:
                    design_values[n] = float(dv)
        else:
            st.caption("Run an identification first; fields appear for every parameter of the best hypothesis.")

truth = None
image = x = f = points = None
band = None
source_name = "—"
with data_box:
    source = st.radio("Data", ["Demo Circuit", "Upload Points CSV (x, f[, weight])", "Upload Map (.npz or Image)",
                               "Upload 2-D Grid CSV (First Row = Control Values, First Column = Frequencies)"], label_visibility="collapsed")
    if source == "Demo Circuit":
        _opts = list(DEMOS[platform])
        _want = st.query_params.get("demo")
        _idx = next((i for i, o in enumerate(_opts) if _want and o.startswith(str(_want))), 0)
        name = st.selectbox("Hidden Circuit", _opts, index=_idx)
        source_name = name
        seed = st.number_input("Random Seed (Surprise Only)", 0, 10_000, 7) if DEMOS[platform][name] is None else None
        truth, r = make_demo(platform, name, seed)
        image, x, f = r["image"], r["x"], r["f"]
        if DEMOS[platform][name] == "real":
            band = (0.05, float(f[-1]))
            flux_calibrated = st.checkbox(r"Use the Paper's Flux Axis ($\Phi/\Phi_0$) for This Map", value=True)
            st.caption("Real data: the right half of the published map is very noisy; the paper's own fit gives EJ=3, EC=0.84, EL=1 GHz.")
    elif source.startswith("Upload Points"):
        up = st.file_uploader("CSV with Columns x, f (, weight)", type=["csv", "txt"])
        if up is not None:
            source_name = up.name
            arr = np.genfromtxt(io.StringIO(up.getvalue().decode()), delimiter=",", skip_header=1 if up.getvalue()[:1].isalpha() else 0)
            arr = np.atleast_2d(arr)
            if arr.shape[1] == 2:
                arr = np.column_stack([arr, np.ones(len(arr))])
            points = arr[:, :3]
            image, x, f = agent.rasterize(points)
    elif source.startswith("Upload 2-D Grid"):
        up = st.file_uploader("CSV Grid: Control Values Across the First Row, Frequencies Down the First Column", type=["csv", "txt"])
        if up is not None:
            source_name = up.name
            try:
                image, x0, f0 = importers.read_grid_csv(up.getvalue())
            except Exception as e:
                st.error(f"Could not parse the grid: {e}")
                image = None
            else:
                if st.checkbox("Transpose (Frequency Is Along Columns)", key="grid_T"):
                    image, x0, f0 = image.T, f0, x0
                image, x, f = finish_map(image, x0, f0, "grid")
    else:
        up = st.file_uploader("NPZ (Keys image, x, f) or a Screenshot PNG/JPG of a Map", type=["npz", "png", "jpg", "jpeg", "h5", "hdf5"],
                              help="HDF5 (.h5/.hdf5) files are accepted too: pick the 2-D dataset that holds the map and, when present, the 1-D datasets of the two axes.")
        if up is not None:
            source_name = up.name
            lower = up.name.lower()
            if lower.endswith(".npz"):
                d = np.load(io.BytesIO(up.getvalue()))
                image, x0, f0 = np.asarray(d["image"], float), np.asarray(d["x"], float).ravel(), np.asarray(d["f"], float).ravel()
                if st.checkbox("Transpose (Frequency Is Along Columns)", key="npz_T"):
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
                    pick = st.selectbox("Map Dataset (2-D)", opts, key="h5_map")
                    path = list(two)[opts.index(pick)]
                    image = importers.h5_read(data, path)
                    if st.checkbox("Transpose (Frequency Is Along Columns)", key="h5_T"):
                        image = image.T
                    nf_, nx_ = image.shape
                    x_opts = ["index range"] + [k for k, v in one.items() if v[0] == nx_]
                    f_opts = ["index range"] + [k for k, v in one.items() if v[0] == nf_]
                    xk = st.selectbox("Control Axis Dataset (1-D)", x_opts, index=min(1, len(x_opts) - 1), key=f"h5_x_{path}_{nx_}")
                    fk = st.selectbox("Frequency Axis Dataset (1-D)", f_opts, index=min(1, len(f_opts) - 1), key=f"h5_f_{path}_{nf_}")
                    xi, fi = index_axes(image, "h5", need_x=xk == "index range", need_f=fk == "index range")
                    x0 = xi if xk == "index range" else importers.h5_read(data, xk)
                    f0 = fi if fk == "index range" else importers.h5_read(data, fk)
                    image, x, f = finish_map(image, x0, f0, "h5")
            else:
                from PIL import Image

                im = np.asarray(Image.open(io.BytesIO(up.getvalue())).convert("L"), float)
                image = im[::-1]  # row 0 is the top of the picture -> highest frequency
                if st.checkbox("Transpose (Frequency Is Along Columns)", key="img_T"):
                    image = image.T
                x0, f0 = index_axes(image, "img")  # the axis ranges read off the picture, in GHz
                image, x, f = finish_map(image, x0, f0, "img", unit_select=False)

# ------------------------------------------------------------------ main
theme.header()

if image is None:
    st.info("Pick a demo circuit or upload data in the sidebar.")
    show_history()
    st.stop()


def _ev(fn_name: str, *args):
    """Call ui.evidence.<fn_name>(*args) if the module and function exist; None on any failure (never breaks the page)."""
    fn = getattr(evidence, fn_name, None) if evidence is not None else None
    if fn is None:
        return None
    try:
        return fn(*args)
    except Exception:  # noqa: BLE001
        return None


def sch_values(best: dict) -> dict[str, str]:
    """Display strings for the fitted values written next to the schematic symbols."""
    p = best["params"]
    out = {}
    for k in ("EJ", "EC", "EL", "EJmax", "fr", "g", "D", "E", "t_c"):
        if k in p:
            out[k] = f"{p[k]:.3g} GHz"  # three figures keep neighbouring labels apart; the table has the full values
    if "d" in p:
        out["d"] = f"d = {p['d']:.3g}"
    if "theta" in p and "phi" in p:
        out["theta"] = f"{p['theta']:.2f}, {p['phi']:.2f} rad"
    return out


def ham_table_html(best: dict) -> str:
    rows = [(theme.param_html(k), theme.num(v, theme.UNITS.get(k, ""), (best.get("sigma") or {}).get(k))) for k, v in best["params"].items()]
    return theme.table_html(("parameter", "value ± σ"), rows, numeric=(1,))


def spec_groups(rep: dict, dq: list[dict]) -> dict[str, list[tuple[str, str, str]]]:
    """Three spec-sheet groups. ui.evidence.spec_card_data(report) is used when present; otherwise the derived quantities
    are grouped by keyword and the control-axis calibration is appended from the fitted parameters."""
    best = rep["best"]
    groups: dict[str, list] = {"Qubit": [], "Control axis": [], "Readout": []}
    data = _ev("spec_card_data", rep)
    if isinstance(data, dict) and data:
        for g, items in data.items():
            key = next((k for k in groups if k.lower().startswith(str(g).lower()[:4])), "Qubit")
            for it in items or []:
                if isinstance(it, dict):
                    groups[key].append((str(it.get("name", it.get("label", ""))), theme.num(it.get("value"), it.get("unit", "")) if not isinstance(it.get("value"), str) else it["value"], str(it.get("note", ""))))
                elif isinstance(it, (list, tuple)) and len(it) >= 2:
                    groups[key].append((str(it[0]), str(it[1]), str(it[2]) if len(it) > 2 else ""))
        return groups

    def group_of(name: str) -> str:
        n = name.lower()
        if any(k in n for k in ("resonator", "detuning δ", "g / δ", "dispersive", "purcell", "χ")):
            return "Readout"
        if any(k in n for k in ("sweet spot", "control unit", "flux sensitivity", "field per control", "zero-field control", "lever arm", "zero-detuning", "axial field")):
            return "Control axis"
        return "Qubit"

    for d in dq or []:
        groups[group_of(d["name"])].append((d["name"], theme.num(d.get("value"), d.get("unit", "")), d.get("note", "")))
    p, sig = best["params"], best.get("sigma") or {}
    cal = [("flux period", "period", "control units", "one flux quantum in control units"), ("flux offset (Φ = 0 at)", "offset", "control units", "control value of the Φ = 0 sweet spot"),
           ("zero of the control axis", "x0", "control units", "")]
    for label, k, unit, note_txt in cal:
        if k in p:
            groups["Control axis"].insert(0, (label, theme.num(p[k], unit, sig.get(k)), note_txt))
    return groups


def truth_rows(best: dict, truth: dict) -> list[list]:
    """fitted ± σ | truth | within 2σ (offset modulo the period; NV field angles compared through their symmetry-equivalent projections)."""
    tp = dict(truth["params"], period=truth.get("period", 1.0), offset=truth.get("offset", 0.0))
    p, sig = best["params"], best.get("sigma") or {}
    rows = []
    for k, v in p.items():
        tv = tp.get(k)
        if tv is None:
            rows.append([theme.param_html(k), theme.num(v, theme.UNITS.get(k, ""), sig.get(k)), "–", ""])
            continue
        s = sig.get(k, float("nan"))
        tol = 2 * s if (s is not None and np.isfinite(s) and s > 0) else 0.01 * abs(tv)
        if k == "offset" and "period" in p and p["period"]:
            P = float(p["period"])
            diff = abs(((v - tv + P / 2) % P) - P / 2)
        elif k in ("theta", "phi") and best["family"] == "nv_ensemble" and {"theta", "phi"} <= set(tp):
            diff = float(np.max(np.abs(models.nv_projections(p["theta"], p["phi"]) - models.nv_projections(tp["theta"], tp["phi"]))))
            tol = max(tol, 0.02)
        else:
            diff = abs(v - tv)
        ok = diff <= tol
        rows.append([theme.param_html(k), theme.num(v, theme.UNITS.get(k, ""), sig.get(k)), theme.num(tv, theme.UNITS.get(k, "")), "✓" if ok else "✗"])
    return rows


# the precomputed result for the selected demo (if any); ?load=1 in the URL loads it on page open (screenshots, pitch links)
saved_path = None
if source == "Demo Circuit":
    key = name.split(" ")[0]
    if key in SAVED and os.path.exists(SAVED[key]):
        saved_path = SAVED[key]
auto_load = bool(saved_path and st.query_params.get("load") == "1" and "report" not in st.session_state)
prev_rep = st.session_state.get("report")
has_result = auto_load or bool(prev_rep and prev_rep.get("status") == "ok")
pts_preview = points if points is not None else extract.extract_points(image, x, f)

if param_bounds:
    theme.md(theme.pill("known") + theme.pills_html((f"{theme.param_html(n)} = {known_store[n]['value']:.4g}" + (" fixed" if known_store[n]["tol"] == 0 else f" ± {known_store[n]['tol']:g}%")
                                                     for n in param_bounds), "amber"))
if not has_result:
    # ---- landing state: the input map on the left, the run card with the pipeline stepper on the right
    st.markdown(f"Upload a two-tone spectroscopy map. {theme.BRAND_NAME} proposes circuit hypotheses, fits each against the detected "
                "features with a robust physics model (checked against scqubits), verifies on held-out flux columns, lists what it "
                "**cannot** explain, and tells you **where to measure next**.")
    left, right = st.columns([1.15, 1])
    with left:
        map_slot = st.empty()
        map_slot.plotly_chart(plots.map_figure(image, x, f, points=pts_preview, title=f"input map · {len(pts_preview)} detected features", xlabel=xlabel, ylabel=ylabel), key="map_initial", **PLOT)
        map_caption = st.empty()
    with right:
        with st.container(border=True):
            theme.md(theme.eyebrow("Run"))
            run = st.button("Identify Circuit", type="primary", width="stretch")
            load_saved = st.button("Load Precomputed Result (Normal Preset)", width="stretch", disabled=saved_path is None,
                                   help="The same pipeline run earlier with the 'normal' effort preset; identical output, no waiting.")
            theme.md(theme.eyebrow("What Happens When You Press Identify"))
            step_slot = st.empty()
            step_slot.markdown(theme.stepper_html(0), unsafe_allow_html=True)
            theme.md(theme.caption("Extract the spectral features, propose circuit hypotheses, fit each with the physics model, verify on held-out columns, report."))
        status = st.empty()
        race_slot = st.empty()
else:
    # ---- a result exists: a compact run strip; the evidence map below replaces the raw input map
    c_run, c_step = st.columns([1.5, 1.2], vertical_alignment="center")
    with c_run:
        b1, b2 = st.columns(2)
        run = b1.button("Identify Circuit", type="primary", width="stretch")
        load_saved = b2.button("Load Precomputed Result (Normal Preset)", width="stretch", disabled=saved_path is None,
                               help="The same pipeline run earlier with the 'normal' effort preset; identical output, no waiting.")
    with c_step:
        step_slot = st.empty()
        step_slot.markdown(theme.stepper_html(5), unsafe_allow_html=True)
    status = st.empty()
    last = st.session_state.get("last_status")
    if last:
        status.markdown(theme.status_html(*last), unsafe_allow_html=True)
    map_slot = st.empty()  # live map during a re-run
    map_caption = st.empty()
    race_slot = st.empty()

if auto_load:
    load_saved = True
if load_saved and saved_path:
    st.session_state["report"] = json.load(open(saved_path))
    st.session_state["truth"] = truth
    st.session_state["run_log"] = []
    st.session_state["race_rows"] = None
    st.session_state["last_status"] = ("loaded", "Loaded demo result (precomputed)", "green")
    remember_run(st.session_state["report"], source_name, platform_label, note="precomputed")
    st.rerun()

if run:
    cfg = agent.AgentConfig(use_classifier=use_clf, try_coupled=try_coupled if is_circuit else "never", workers=workers, families=tuple(families),
                            flux_calibrated=flux_calibrated and is_circuit, param_bounds=dict(param_bounds), **preset)
    rows: list[dict] = []  # hypothesis race, in the order tried
    log: list[str] = []
    state = dict(last=0.0, t0=time.time(), draw=0, active=None, stage=0)

    def row_for(h: str) -> dict:
        for r in rows:
            if r["name"] == h:
                return r
        r = dict(name=h, status="queued", score=None, best=[])
        rows.append(r)
        return r

    def say(label: str, text: str, kind: str = "") -> None:
        t = time.time() - state["t0"]
        status.markdown(theme.status_html(label, text, kind, t=t), unsafe_allow_html=True)  # the log collapses to its last line
        log.append(f"{t:6.1f} s  {label:<10} {text}")

    def step(stage: int) -> None:
        state["stage"] = stage
        step_slot.markdown(theme.stepper_html(stage, running=stage < 5), unsafe_allow_html=True)

    def draw_race() -> None:
        race_slot.markdown(theme.race_html(rows, active=state["active"]), unsafe_allow_html=True)

    def redraw(p: dict, title: str) -> None:
        state["draw"] += 1  # a fresh key per redraw: the placeholder replaces the element, keys never collide
        map_slot.plotly_chart(plots.map_figure(image, x, f, points=pts_preview, lines=p["lines"], x_grid=x, title=title, xlabel=xlabel, ylabel=ylabel), key=f"map_live_{state['draw']}", **PLOT)
        if p.get("params"):
            map_caption.markdown(theme.pills_html(f"{theme.param_html(k)} = {v:.4g}" for k, v in p["params"].items()), unsafe_allow_html=True)
        draw_race()

    def progress(kind, p):
        if kind == "features":
            step(1)
            say("features", f"{p['n_points']} features in {p['n_tracks']} tracks · flux-independent lines at {', '.join(f'{v:.2f}' for v in p['flat_lines']) or 'none'} GHz")
        elif kind == "proposals":
            step(2)
            for h in p["hypotheses"]:
                row_for(h["h"])
            draw_race()
            say("proposals", "hypotheses to test: " + "; ".join(f"{h['h']}" + (f" (p={h['prob']:.2f})" if h.get("prob") else "") for h in p["hypotheses"]))
        elif kind == "fit_start":
            step(2)
            r = row_for(p["hypothesis"])
            r.update(status="fitting", best=[])  # the best-so-far sparkline resets for every hypothesis
            state["active"] = p["hypothesis"]
            draw_race()
            say("fitting", f"{p['hypothesis']} on {p['n_points']} features", "amber")
        elif kind == "fit_progress":
            r = row_for(p["hypothesis"])
            loss = float(p["loss"])
            r["best"].append(min(loss, r["best"][-1]) if r["best"] else loss)
            now = time.time()
            if now - state["last"] > 0.7:  # throttle the live redraw
                state["last"] = now
                redraw(p, f"{p['hypothesis']} · searching · loss {loss:.3f}")
        elif kind == "fit_done":
            s = p["summary"]
            note = p.get("note", "") or ""
            r = row_for(p["hypothesis"])
            r.update(score=float(p["score"]), status="accepted" if note == "accepted" else ("rejected" if note.startswith("rejected") else "fitted"))
            state["active"] = None
            redraw(dict(p, params=s.get("params")), f"{p['hypothesis']} · fitted · score {p['score']:.3f}")
            say("fitted", f"{p['hypothesis']}: score {p['score']:.3f}, {100*s['inlier_frac']:.0f}% explained, RMS {s['rms_inliers_MHz']:.1f} MHz" + (f" · {note}" if note else ""), "blue")
        elif kind == "calibration":
            say("symmetry", f"flux period ≈ {p['period']:.3g} control units; sweet spots at " + ", ".join(f"{a['x']:.3g} ({a['type']})" for a in p["axes"][:3]))
        elif kind == "estimate":
            say("shape", "geometric prior from the track shapes: " + ", ".join(f"{k} ≈ {v:.3g}" for k, v in p["estimate"].items() if isinstance(v, float)))
        elif kind == "skip":
            row_for(p["hypothesis"]).update(status="skipped")
            draw_race()
            say("skip", f"{p['hypothesis']}: {p['reason']}")
        elif kind == "revise":
            say("revise", p["reason"], "amber")
        elif kind == "verify":
            step(4)
            ho = p["holdout"]
            if ho:
                say("hold-out", f"{100*ho['inlier_frac']:.0f}% of features in untouched flux columns explained, RMS {ho['rms_MHz']:.1f} MHz", "blue")
        elif kind == "warning":
            say("warning", p.get("message", ""), "red")
        elif kind == "done":
            step(5)

    t0 = time.time()
    rep = agent.identify(image, x, f, points=points, band=band, config=cfg, progress=progress)
    step(5)
    if rep.get("status") == "ok":
        win = rep["best"]["hypothesis"]
        for r in rows:
            if r["name"] == win:
                r["status"] = "accepted"
            elif r["status"] == "fitting":
                r["status"] = "fitted"
    say("done", f"{time.time()-t0:.0f} s · {speed} preset" + (" · constrained" if param_bounds else ""), "green")
    st.session_state["report"] = rep
    st.session_state["truth"] = truth
    st.session_state["run_log"] = list(log)
    st.session_state["race_rows"] = [dict(r, best=[]) for r in rows]
    st.session_state["last_status"] = ("done", f"identified in {time.time()-t0:.0f} s · {speed} preset" + (" · constrained" if param_bounds else ""), "green")
    remember_run(rep, source_name, platform_label, note=f"{speed}" + (", constrained" if param_bounds else ""))
    st.rerun()  # redraw in the result layout: verdict band + evidence map replace the input map

# ------------------------------------------------------------------ results
rep = st.session_state.get("report")
if rep and rep.get("status") == "ok":
    best = rep["best"]
    fam, coupled = best["family"], bool(best.get("coupled"))
    xg = np.array(rep["x_grid"], float)
    conf = best.get("confidence") if isinstance(best.get("confidence"), dict) else {}
    ho = best.get("holdout") or {}
    is_circ_fam = models.platform_of(fam) == "circuit"
    flux_dep = bool(models.FAMILIES[fam]["flux_dependent"])

    # ---- B. verdict band
    level = conf.get("level") or "not assessed"
    mp = rep.get("model_posterior") or {}
    _bf = mp.get("log10_bayes_factor_vs_other_family")
    eyebrow_txt = (f'IDENTIFIED · <span class="cs-num">{len(rep["hypotheses"])}</span> hypotheses · <span class="cs-num">{rep.get("runtime_s", "?")}</span> s · confidence {theme.esc(level)}')
    kpis: list[tuple[str, str]] = []
    if ho:
        n_test = ho.get("n_test_points")
        n_txt = f' · <span class="cs-num">{int(n_test)}</span> held-out points' if n_test is not None else ""
        kpis.append((f'{ho["rms_MHz"]:.1f}<span class="u">MHz</span>', f"held-out RMS{n_txt}"))
        kpis.append((f'{100*ho["inlier_frac"]:.0f}<span class="u">%</span>', "held-out explained"))
    else:
        kpis.append((f'{best["rms_inliers_MHz"]:.1f}<span class="u">MHz</span>', "in-sample RMS"))
        kpis.append((f'{100*best["inlier_frac"]:.0f}<span class="u">%</span>', "features explained"))
    gap = conf.get("gap")
    if isinstance(gap, (int, float)):
        kpis.append((f"{gap:+.3f}".replace("-", "−"), "margin to runner-up family"))
    else:
        kpis.append(("—", "margin · single family"))
    if isinstance(_bf, (int, float)) and np.isfinite(_bf):  # Bayesian model comparison: evidence ratio against the best other family
        kpis.append((f'10<sup>{_bf:.0f}</sup>', f"Bayes factor vs {theme.esc(str(mp.get('runner_up_family', '')).replace('_', ' '))}"))
    right = (schematic.schematic_svg(fam, coupled, values=sch_values(best)) + theme.eyebrow("Recovered Hamiltonian") + ham_table_html(best))
    with st.container(border=True, key="hero"):
        h_l, h_r = st.columns([3, 2], gap="large")
        with h_l:
            theme.md(theme.hero_top_html(eyebrow_txt, best["hypothesis"]))
            st.latex(theme.hamiltonian_tex(fam, best["params"], coupled))
            theme.md(theme.hero_bottom_html(kpis, best.get("explained_dynamic")))
        with h_r:
            theme.md(right)
    if best.get("note"):
        theme.md(theme.note(best["note"]))
    # ---- H. two sentences of plain English (template or Claude; exception text never reaches the page)
    ex_key = (best["hypothesis"], json.dumps(best["params"], sort_keys=True), bool(use_llm))
    cache = st.session_state.get("explain_cache")
    if not cache or cache[0] != ex_key:
        with st.spinner("writing summary …"):
            try:
                ex = explain.explain(rep, use_llm=use_llm)
            except Exception:  # noqa: BLE001
                ex = explain.template(rep)
        cache = (ex_key, ex)
        st.session_state["explain_cache"] = cache
    ex = cache[1]
    summary = theme.two_sentences(ex.get("summary", "")) if isinstance(ex, dict) else ""
    conf_txt = conf.get("text", "")
    if summary or conf_txt:
        theme.md(f'<div class="cs-block"><div class="cs-body">{theme.mathify(theme.esc(summary))}</div>' + (f'<div class="cs-body muted">{theme.mathify(theme.esc(conf_txt[:1].upper() + conf_txt[1:]))}.</div>' if conf_txt else "") + "</div>")

    # ---- C. hypotheses tested (the race in its final state)
    race_rows = st.session_state.get("race_rows") or [dict(name=h["hypothesis"], status="fitted", score=float(h["track_score"]), best=[]) for h in rep["hypotheses"]]
    _post = {h["hypothesis"]: h.get("posterior") for h in rep["hypotheses"] if h.get("posterior") is not None}
    theme.md(theme.section("Hypotheses tested", f"{len(rep['hypotheses'])} candidates · score = explained track weight − penalties"
                           + (" · P = Bayesian model probability (Laplace evidence, Occam factor included)" if _post else "")))
    theme.md(theme.race_html(race_rows, finished=True, winner=best["hypothesis"], posterior=_post or None))
    if mp.get("families") and len(mp["families"]) > 1:
        theme.md(theme.caption("Model posterior by family: " + " · ".join(f"{theme.title_case(k.replace('_', ' '))} {v:.3g}" if v >= 1e-3 else f"{theme.title_case(k.replace('_', ' '))} < 10⁻³" for k, v in mp["families"].items())
                               + ". Likelihood: each feature is on a predicted line (Gaussian) or background (uniform); evidence by Laplace approximation with log-uniform priors on energies."))

    # ---- D. evidence
    theme.md(theme.section("Evidence", "annotated map", "Explained features are filled green, unexplained ones hollow red; numbered amber boxes are where to look next."))
    show_ev = st.toggle("Show Held-Out and Residual Evidence", value=True)
    _look = agent.prune_look_here([n for n in rep["next_measurements"] if n["kind"] == "look_here"], (float(min(rep["x_grid"])), float(max(rep["x_grid"]))), tuple(rep["band_GHz"]),
                                  vis=models.line_weights(rep["best"]["family"]))
    rep["next_measurements"] = [n for n in rep["next_measurements"] if n["kind"] != "look_here"] + _look  # older saved reports predate the pruning
    boxes = _look[:5]
    ev_map_slot = st.empty()  # drawn after the operating-point tool below has chosen its markers
    if show_ev and evidence is not None:
        resid = _ev("residual_figure", rep)
        if resid is not None:
            st.plotly_chart(resid, key="ev_residual", **PLOT)
        hold = _ev("holdout_figure", rep, image, x, f)
        zoom = _ev("crossing_zoom_figure", rep, image, x, f)
        if hold is not None or zoom is not None:
            e1, e2 = st.columns(2)
            if hold is not None:
                e1.plotly_chart(hold, key="ev_holdout", **PLOT)
            if zoom is not None:
                e2.plotly_chart(zoom, key="ev_zoom", **PLOT)

    # ---- E. physics + spec sheet, side by side: the model on the left, what it implies for the lab on the right
    theme.md(theme.section("Physics and Spec Sheet", ("potential, levels and" if is_circ_fam else "energy levels and") + " derived quantities",
                           "Hover a spec row for the note behind it; the operating-point tool draws amber diamonds on the map."))
    dq = rep.get("derived")
    if dq is None:  # older saved reports: compute on the fly
        try:
            dq = derived.derived_quantities(best, (float(xg.min()), float(xg.max())))
        except Exception:  # noqa: BLE001
            dq = []
    groups = spec_groups(rep, dq)
    markers: list[tuple[float, float]] = []
    ph_col, sp_col = st.columns([1.25, 1], gap="large")
    with ph_col:
        physics = None
        if is_circ_fam:
            phi_ext = 0.0
            if flux_dep:
                phi_ext = st.slider(r"External Flux $\Phi$ ($\Phi_0$)", 0.0, 0.5, 0.0, 0.01, format="%.2f", help="Redraws the potential well and its levels at this flux.")
            physics = plots.potential_figure(fam, best["params"], coupled, phi_ext=phi_ext)
        else:
            try:
                from ui import platform_panels

                physics = {"nv_ensemble": platform_panels.nv_levels_figure, "charge_qubit": platform_panels.dqd_figure,
                           "trapped_ion": platform_panels.ion_modes_figure, "rydberg_atom": platform_panels.stark_figure}[fam](best["params"])
            except Exception:  # noqa: BLE001
                physics = None
        if physics is not None:
            st.plotly_chart(physics, key="physics", **PLOT)
        with st.container(border=True):
            theme.md(theme.eyebrow("Readout"))
            theme.md(theme.spec_rows_html(groups["Readout"]) if groups["Readout"] else '<div class="cs-spec-empty">no readout resonator in the accepted model</div>')
    with sp_col:
        with st.container(border=True):
            theme.md(theme.eyebrow("Qubit"))
            theme.md(theme.spec_rows_html(groups["Qubit"]))
            if is_circ_fam and flux_dep:
                band_rep = rep.get("band_GHz") or [float(f[0]), float(f[-1])]
                target = st.number_input(r"Target $f_{01}$ (GHz)", value=float(round(0.5 * (band_rep[0] + band_rep[1]), 2)), step=0.05, format="%.3f",
                                         help="Where to bias the device: control values at which the fitted 0→1 transition equals this frequency.")
                try:
                    sols = derived.control_for_frequency(best, float(target), (float(xg.min()), float(xg.max())))
                except Exception:  # noqa: BLE001
                    sols = []
                if sols:
                    theme.md(theme.spec_rows_html([("control for this f01", ", ".join(f"{s:.3f}" for s in sols).replace("-", "−"), "also drawn as amber diamonds on the map")]))
                    markers = [(float(s), float(target)) for s in sols]
                else:
                    theme.md(theme.caption(f"the fitted 0→1 line does not reach {target:.3f} GHz inside the measured control range"))
        with st.container(border=True):
            theme.md(theme.eyebrow("Control axis"))
            theme.md(theme.spec_rows_html(groups["Control axis"]))

    # the annotated map, now that the operating points are known
    fig_final = plots.map_figure(image, x, f, points=np.array(rep["points"]), inlier=rep["inlier_mask"], lines=rep["lines"], x_grid=xg, xlabel=xlabel, ylabel=ylabel, look_here=boxes,
                                 flat_lines=rep["flat_lines_GHz"], title=f"{best['hypothesis']} · fitted transitions over the measured map", markers=markers or None)
    ev_map_slot.plotly_chart(fig_final, key="map_final", **PLOT)

    # ---- F. next measurements: the gaps (numbered to match the map badges) on the left, the information-gain ranking on the right
    cards = []
    for i, b in enumerate(boxes, start=1):
        cards.append(theme.card_html(str(i), f"sweep · {plots.pretty(b['line'])} line", theme.frange_html(b["f_lo"], b["f_hi"]),
                                     f"control {b['x_lo']:.3g} → {b['x_hi']:.3g}".replace("-", "−"), b["why"]))
    for n in rep["next_measurements"]:
        if n["kind"] == "extend_band":
            cards.append(theme.card_html("↕", "extend the band", theme.frange_html(n["f_lo"], n["f_hi"]), f"{plots.pretty(n['line'])} line, outside the measured band", n["why"], muted_badge=True))
        elif n["kind"] == "discriminate":
            cards.append(theme.card_html("≠", "one column settles the family", f"{n['delta_sigma']:.0f} <span class='u'>σ apart</span>",
                                         f"control {n['x']:.3g} · {plots.pretty(n['line'])} line at {n['f']:.3g}" + ("" if n.get("inside_window") else " (outside the measured window)"), n["why"], muted_badge=True))
        elif n["kind"] == "constrain_parameter":
            rel = n["rel_sigma"]
            cards.append(theme.card_html("σ", f"constrain {n['parameter']}", f"{theme.param_html(n['parameter'])} <span class='u'>{f'±{100*rel:.0f}%' if rel < 5 else 'unconstrained'}</span>",
                                         "poorly determined by the current data", n["why"], muted_badge=True))
    dsg = rep.get("design")
    if not dsg or dsg.get("error") or "correlation" not in dsg:
        try:
            from circuitsleuth import design as _design
            dsg = _design.design_next_measurements(rep, band=tuple(rep["band_GHz"]))
        except Exception:
            dsg = None
    have_design = bool(dsg and dsg.get("candidates") is not None and "correlation" in dsg)
    theme.md(theme.section("Next Measurements", f"{len(cards)} gaps in the data · {'ranked by expected information gain' if have_design else 'from the predicted lines'}",
                           "Left: where a predicted line runs through measured columns with nothing under it (numbers match the badges on the map). "
                           "Right: the sweeps that would tighten the parameters most, from the Fisher information of the fit."))
    nx_l, nx_r = st.columns([1, 1.3], gap="large")
    with nx_l:
        theme.md(theme.eyebrow("Gaps and ambiguities"))
        if cards:
            theme.md(theme.grid_html(cards))
        else:
            theme.md('<div class="cs-body muted cs-block">No obvious gaps: every predicted line inside the band has data under it.</div>')
        if have_design:
            from ui import design_panel as _dp
            theme.md(theme.eyebrow("What the data cannot separate"))
            st.plotly_chart(_dp.correlation_figure(dsg), key="design_corr", **PLOT)
    with nx_r:
        if have_design:
            theme.md(theme.eyebrow("Expected information gain · best sweeps"))
            _unit = models.PLATFORMS[platform].get("freq_unit", "GHz")
            drows = []
            for i, c in enumerate(dsg["candidates"][:5], start=1):
                red = c["variance_reduction"].get(c["best_parameter"], 0.0)
                drows.append([f'<span class="cs-badge{"" if i <= 3 else " muted"}">{i}</span>', f"{plots.pretty(c['line'])} line",
                              f"{c['x_lo']:.3g} → {c['x_hi']:.3g}".replace("-", "−"), f"{c['f_lo']:.2f}–{c['f_hi']:.2f} {_unit}".replace("-", "−"),
                              f"{c['gain_bits']:.2f}", f"{theme.param_html(c['best_parameter'])} <span class='u'>−{100*red:.0f}% var</span>"])
            if drows:
                theme.md(theme.table_html(("", "sweep", "control", "frequency", "bits", "tightens"), drows, numeric=(2, 3, 4)))
            st.plotly_chart(_dp.gain_figure(dsg, xlabel=xlabel), key="design_gain", **PLOT)
            theme.md(theme.caption("Each candidate is a transition over a block of control values; its gain is ½ log of the determinant ratio of the Fisher "
                                   "information with and without it, discounted by the line's visibility and the sweep cost. Amber bands mark the three best."))
        else:
            theme.md(theme.caption("Information-gain design is unavailable for this result."))

    # ---- F3. autopilot: closed-loop spectroscopy on a digital twin (the full map stands in for the instrument)
    theme.md(theme.section("Autopilot", "closed-loop spectroscopy on a digital twin",
                           "Start from the first 40 % of the sweep, identify, let the information-gain design pick the next block of control values, "
                           "\"measure\" it from the full map, refit, repeat. The one function that reads the twin is the one a lab replaces with an instrument driver."))
    ap1, ap2 = st.columns([1, 1])
    _fams_ap = tuple(families) if families else tuple(models.PLATFORMS[platform]["families"])
    _loop_key = (name.split(" ")[0] if source == "Demo Circuit" else "")
    _saved_loop = f"artifacts/loop_{_loop_key}.json" if _loop_key else ""
    run_ap = ap1.button("Run Autopilot (3 Rounds, Live)", width="stretch", help="Seconds on NV / dots / ions / atoms; a few minutes for superconducting circuits")
    load_ap = ap2.button("Replay Saved Autopilot", width="stretch", disabled=not (_saved_loop and os.path.exists(_saved_loop)))
    if run_ap:
        from circuitsleuth import loop as _loop
        ap_status = st.empty()

        def _ap_progress(kind, p):
            if kind == "round_start":
                ap_status.markdown(theme.status_html("autopilot", f"round {p['round']}: fitting {p['n_points']} features from {p['n_columns']} columns", "amber"), unsafe_allow_html=True)
            elif kind == "round_done":
                blk = p.get("block")
                ap_status.markdown(theme.status_html("autopilot", f"round {p['round']} done · {p['hypothesis']}" + (f" · next: {plots.pretty(blk['line'])} line at control {blk['x_lo']:.3g} → {blk['x_hi']:.3g} ({blk['gain_bits']:.2f} bits)" if blk else " · nothing left worth measuring"), "blue"), unsafe_allow_html=True)

        _cfg_ap = dict(maxiter=preset["maxiter"], n_starts=1, popsize=preset["popsize"], top_k=preset["top_k"], coupled_trials=preset["coupled_trials"], workers=workers)
        st.session_state["autopilot"] = _loop.run_loop(image, x, f, _fams_ap, rounds=3, cfg_kwargs=_cfg_ap, progress=_ap_progress)
        st.session_state["autopilot"]["demo"] = _loop_key
        ap_status.markdown(theme.status_html("autopilot", "done", "green"), unsafe_allow_html=True)
    if load_ap and _saved_loop:
        st.session_state["autopilot"] = json.load(open(_saved_loop))
    apr = st.session_state.get("autopilot")
    if apr and apr.get("rounds") and ("demo" not in apr or apr.get("demo") == _loop_key):
        from ui import design_panel as _dp2
        rows_ap = [rr for rr in apr["rounds"] if "params" in rr]
        if rows_ap:
            r_idx = st.slider("Round", 0, len(rows_ap) - 1, len(rows_ap) - 1, key="ap_round") if len(rows_ap) > 1 else 0
            rr = rows_ap[r_idx]
            vis_pts = np.array(rr["points"])
            blk = rr.get("block")
            look = [dict(line=blk["line"], x_lo=blk["x_lo"], x_hi=blk["x_hi"], f_lo=blk["f_lo"], f_hi=blk["f_hi"])] if blk else None
            c_ap1, c_ap2 = st.columns([1.3, 1])
            c_ap1.plotly_chart(plots.map_figure(image, x, f, points=vis_pts, inlier=rr["inlier_mask"], lines=rr["lines"], x_grid=np.array(rr["x_grid"]), look_here=look,
                                                title=f"round {rr['round']} · {rr['n_points']} features from {rr['columns_measured']} columns · {rr['hypothesis']}" + (f" · next sweep: {plots.pretty(blk['line'])} ({blk['gain_bits']:.2f} bits)" if blk else ""),
                                                xlabel=xlabel, ylabel=ylabel), key=f"ap_map_{r_idx}", **PLOT)
            fig_s = _dp2.sigma_vs_round_figure(apr)
            if fig_s is not None:
                c_ap2.plotly_chart(fig_s, key="ap_sigma", **PLOT)
            tbl = []
            for k, v in rr["params"].items():
                sg = rr["sigma"].get(k, float("nan"))
                tbl.append([theme.param_html(k), theme.num(v, theme.UNITS.get(k, ""), sg), f"{abs(sg / v):.1%}" if v else "–"])
            with c_ap2:
                theme.md(theme.table_html(("parameter", "value ± σ", "σ / value"), tbl, numeric=(1, 2)))
            if apr.get("truth"):
                tp = apr["truth"]["params"]
                hits = [k for k in rr["params"] if k in tp]
                st.markdown("Hidden truth: " + ", ".join(f"${theme.param_tex(k)} = {tp[k]:.4g}$" for k in hits))

    # ---- G. diagnostics: one section, four tabs (unexplained features · posterior · truth and design · run log)
    truth_rep = st.session_state.get("truth")
    design_rows = [(theme.param_html(k), theme.num(best["params"][k], theme.UNITS.get(k, "")), theme.num(dv, theme.UNITS.get(k, "")), f"{100 * (best['params'][k] - dv) / abs(dv):+.1f}%".replace("-", "−") if dv else "–")
                   for k, dv in design_values.items() if k in best["params"]]
    theme.md(theme.section("Diagnostics", f"{len(rep['unexplained'])} unexplained features · error bars · checks",
                           "What the model leaves on the table, whether the stated uncertainties are honest, and how the fit compares with what you know."))
    tab_unexpl, tab_post, tab_truth, tab_log = st.tabs(["Unexplained Features", "Posterior Check", "Truth and Design", "Run Log"])
    with tab_unexpl:
        if rep["unexplained"]:
            theme.md(theme.grid_html([theme.card_html(None, "flux-independent line" if u["flat"] else "flux-dependent feature", theme.frange_html(u["f_lo"], u["f_hi"]),
                                                      f"{u['n_points']} points · {100*u.get('unexplained_frac', 1):.0f}% off any predicted line", u["interpretation"]) for u in rep["unexplained"]]))
        else:
            theme.md('<div class="cs-body muted cs-block">Every detected feature is accounted for.</div>')
    with tab_post:
        theme.md('<div class="cs-body muted cs-block">The reported σ come from the Jacobian at the optimum (Laplace), scaled by 1.8 because synthetic coverage runs '
                 'showed the raw value under-covers. Sampling the actual likelihood (Metropolis on the robust loss) shows the shape of the uncertainty, including the E<sub>J</sub>–E<sub>C</sub> valley.</div>')
        if st.button("Sample the Posterior (About 10 s)", width="stretch", key="btn_posterior"):
            try:
                from circuitsleuth import fit as _fit
                from ui import design_panel as _dp3
                _res = _fit.result_from_report(rep)
                _cl = models.platform_scales(platform)["c_local"]
                with st.spinner("sampling…"):
                    st.session_state["posterior"] = _fit.sample_posterior(np.array(rep["points"]), _res, tuple(rep["band_GHz"]), _cl, n_steps=3000)
            except Exception as _e:
                st.session_state["posterior"] = dict(error=f"{type(_e).__name__}: {_e}")
        _post = st.session_state.get("posterior")
        if _post and not _post.get("error") and _post.get("family") == best["family"]:
            from ui import design_panel as _dp3
            pair = _dp3.posterior_pair_choice(_post["names"])
            pc1, pc2 = st.columns([1.2, 1])
            pc1.plotly_chart(_dp3.posterior_figure(_post, pair=pair), key="posterior_fig", **PLOT)
            rows_p = []
            _sd, _lsd = _post["sd"], _post["laplace_sd"]
            for i, n in enumerate(_post["names"]):
                psd = _sd[n] if isinstance(_sd, dict) else _sd[i]
                lap = (_lsd.get(n, float("nan")) if isinstance(_lsd, dict) else _lsd[i]) if _lsd is not None else float("nan")
                rows_p.append([theme.param_html(n), f"{psd:.3g}", f"{lap:.3g}", f"{psd / lap:.2f}" if lap else "–"])
            with pc2:
                theme.md(theme.table_html(("parameter", "posterior σ", "Laplace σ", "ratio"), rows_p, numeric=(1, 2, 3)))
            pc2.caption(f"acceptance {_post['acceptance_rate']:.2f} · {_post['n_kept']} kept samples · {_post['seconds']:.1f} s")
        elif _post and _post.get("error"):
            st.caption("posterior sampling unavailable for this result")
    with tab_truth:
        if truth_rep is None and not design_rows:
            theme.md('<div class="cs-body muted cs-block">Nothing to compare against: this is not a demo with a hidden truth, and no design values were entered (sidebar → Explain → Compare with Design Values).</div>')
        if truth_rep is not None and st.toggle("Reveal Hidden Truth (Demo)"):
            rows_t = truth_rows(best, truth_rep)
            theme.md(theme.table_html(("parameter", "fitted ± σ", "truth", "within 2σ"), rows_t, numeric=(1, 2), classes=lambda r, c: ("ok" if rows_t[r][3] == "✓" else "bad") if c == 3 else ""))
            theme.md(theme.caption(f"truth: {truth_rep['family']}{' + resonator' if truth_rep.get('coupled') else ''}; spurs at {[s['f'] for s in truth_rep.get('spurs', [])]} GHz"
                                   + (f" · {truth_rep['note']}" if truth_rep.get("note") else "")))
            pull = _ev("pull_figure", rep, dict(truth_rep["params"], period=truth_rep.get("period", 1.0), offset=truth_rep.get("offset", 0.0)))
            if pull is not None:
                st.plotly_chart(pull, key="ev_pull", **PLOT)
        if design_rows:
            theme.md(theme.eyebrow("Fit vs design"))
            theme.md(theme.table_html(("parameter", "fit", "design", "deviation"), design_rows, numeric=(1, 2, 3)))
    with tab_log:
        run_log = st.session_state.get("run_log") or []
        if run_log:
            theme.md(theme.log_html(run_log))
        else:
            st.caption("No live log: this result was loaded from a precomputed run.")

    # ---- G3. Ask qĤolmes: Claude answers questions about this device by calling the physics engine as tools (never guesses numbers)
    try:
        from ui import ask_panel as _ask

        _ask.render(rep, platform)
    except Exception:  # noqa: BLE001  (optional panel; never breaks the page)
        pass

    # ---- H. details & export
    theme.md(theme.section("Details & export", "take it with you", "The full report as JSON, the fitted curves as CSV, the annotated map as a standalone HTML file, and the scqubits code to reproduce the model."))
    try:
        csv_text = derived.curves_csv(rep)
    except Exception:  # noqa: BLE001
        csv_text = None
    e1, e2, e3 = st.columns(3)
    e1.download_button("Download Report (JSON)", agent.to_json(rep), file_name="holmes_report.json", mime="application/json", width="stretch")
    e2.download_button("Download Fitted Curves (CSV)", csv_text or "", file_name="holmes_curves.csv", mime="text/csv", disabled=csv_text is None, width="stretch")
    e3.download_button("Download Annotated Map (HTML)", fig_final.to_html(include_plotlyjs="cdn"), file_name="holmes_map.html", mime="text/html", width="stretch")
    with st.expander("Reproduce in scqubits"):
        theme.md(theme.caption("Copy scqubits snippet: the copy button appears at the top right of the block when you hover over it."))
        code = rep.get("scqubits_code")
        if not code:
            try:
                code = derived.scqubits_code(best)
            except Exception as e:  # noqa: BLE001
                code = f"# unavailable: {type(e).__name__}: {e}"
        st.code(code, language="python")
    show_history()
elif rep:
    st.error(rep.get("error"))
    show_history()
else:
    show_history()
