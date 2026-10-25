"""Quantities a lab actually wants after a fit: operating points in control units, anharmonicity, charge
dispersion, flux sensitivity, dispersive shift, and a scqubits snippet / CSV to carry the result elsewhere."""
from __future__ import annotations

import io

import numpy as np

from . import fit, models


def _f01(hyp: fit.Hypothesis, params: dict, phi: np.ndarray) -> np.ndarray:
    circuit, _, _ = fit.split_params(params, hyp)
    kw = dict(fit.MODEL_KW)
    return models.transitions(hyp.family, circuit, phi, coupled=hyp.coupled, **kw)["0-1"]


def _x_of_phi(params: dict, phi: float) -> float:
    return params["offset"] + phi * params["period"]


def derived_quantities(best: dict, x_range: tuple[float, float] | None = None) -> list[dict]:
    """List of dict(name, value, unit, note) for the report's best hypothesis."""
    fam, coupled, p = best["family"], best.get("coupled", False), best["params"]
    hyp = fit.Hypothesis(fam, coupled)
    out: list[dict] = []
    pl = models.platform_of(fam)
    if pl == "circuit":
        circuit, _, _ = fit.split_params(p, hyp)
        kw = dict(fit.MODEL_KW)
        if fam == "transmon":
            phis = np.array([0.0])
        else:
            phis = np.array([0.0, 0.5])
        E = models.qubit_energies(fam, {k: v for k, v in circuit.items() if k not in ("fr", "g")}, phis, nlev=4, **{k: v for k, v in kw.items() if k in ("ncut", "cutoff")})
        f01 = E[:, 1] - E[:, 0]
        f12 = E[:, 2] - E[:, 1]
        if fam == "transmon":
            out.append(dict(name="f01", value=f01[0], unit="GHz", note="qubit transition"))
            out.append(dict(name="anharmonicity α = f12 − f01", value=1e3 * (f12[0] - f01[0]), unit="MHz", note="negative for a transmon"))
            out.append(dict(name="EJ / EC", value=circuit["EJ"] / circuit["EC"], unit="", note="≥ 50 keeps charge dispersion small"))
            # exact charge dispersion of the 0-1 transition from the charge-basis model
            Hs = [models.transmon_hamiltonians(circuit["EJ"], circuit["EC"], ng=ng, ncut=30)[0] for ng in (0.0, 0.5)]
            ev = [np.linalg.eigvalsh(H)[0] for H in Hs]
            eps = abs((ev[0][1] - ev[0][0]) - (ev[1][1] - ev[1][0]))
            out.append(dict(name="charge dispersion of f01", value=1e6 * eps, unit="kHz", note="peak-to-peak over the offset charge"))
        else:
            out.append(dict(name="f01 at Φ = 0 (upper sweet spot)", value=f01[0], unit="GHz", note=""))
            out.append(dict(name="f01 at Φ = ½ (lower sweet spot)", value=f01[1], unit="GHz", note=""))
            out.append(dict(name="anharmonicity at Φ = 0", value=1e3 * (f12[0] - f01[0]), unit="MHz", note=""))
            out.append(dict(name="anharmonicity at Φ = ½", value=1e3 * (f12[1] - f01[1]), unit="MHz", note=""))
            if fam == "tunable_transmon":
                out.append(dict(name="EJmax / EC", value=circuit["EJmax"] / circuit["EC"], unit="", note=""))
                out.append(dict(name="tuning range f01(0) − f01(½)", value=f01[0] - f01[1], unit="GHz", note="set by the junction asymmetry d"))
            else:
                out.append(dict(name="EJ / EC", value=circuit["EJ"] / circuit["EC"], unit="", note=""))
                out.append(dict(name="EL / EJ", value=circuit["EL"] / circuit["EJ"], unit="", note="≪ 1 for a heavy fluxonium"))
                out.append(dict(name="plasma frequency √(8 EL EC)", value=np.sqrt(8 * circuit["EL"] * circuit["EC"]), unit="GHz", note="inductive-well oscillation scale"))
            # flux sensitivity: max |df01/dPhi| in the window, converted to control units
            phi_grid = np.linspace(0.0, 0.5, 101)
            f = _f01(fit.Hypothesis(fam, False), {**circuit, "period": 1.0, "offset": 0.0}, phi_grid)
            df = np.gradient(f, phi_grid)
            k = int(np.argmax(np.abs(df)))
            out.append(dict(name="max flux sensitivity |∂f01/∂Φ|", value=abs(df[k]), unit="GHz / Φ₀", note=f"at Φ ≈ {phi_grid[k]:.2f}; flux-noise dephasing scales with this"))
            if "period" in p:
                out.append(dict(name="same, in control units", value=abs(df[k]) / p["period"], unit="GHz per control unit", note=""))
                # sweet spots in control units inside the measured range
                if x_range is not None:
                    xs = []
                    for kk in range(-20, 21):
                        for phi0, lab in ((0.0, "Φ = 0"), (0.5, "Φ = ½")):
                            xv = _x_of_phi(p, kk + phi0)
                            if x_range[0] <= xv <= x_range[1]:
                                xs.append((xv, lab))
                    xs.sort()
                    for xv, lab in xs[:6]:
                        out.append(dict(name=f"sweet spot {lab}", value=xv, unit="control units", note="first-order flux-insensitive bias point"))
        if fam != "transmon":
            # first-order 1/f flux-noise dephasing at the steepest point (Ithier et al. 2005 form, f_ir = 1 Hz, t = 10 µs)
            slope_rad = 2 * np.pi * abs(df[k]) * 1e9  # rad/s per Phi0
            fac = np.sqrt(2 * abs(np.log(2 * np.pi * 1.0 * 10e-6)))
            for A in (1e-6, 5e-6):
                tphi = 1.0 / (fac * A * slope_rad)
                out.append(dict(name=f"Tφ at the steepest flux point for 1/f flux noise of {A*1e6:.0f} µΦ₀", value=1e6 * tphi, unit="µs",
                                note="first-order estimate; at a sweet spot the first-order term vanishes"))
        if coupled:
            fr, g = p["fr"], p["g"]
            qp = {k: v for k, v in circuit.items() if k not in ("fr", "g")}
            spots = [(0.0, "Φ = 0")] if fam == "transmon" else [(0.0, "Φ = 0"), (0.5, "Φ = ½")]
            for phi0, lab in spots:
                E, labels = models.dressed_energies(fam, {**qp, "fr": fr, "g": g}, np.array([phi0]), kq=6, kr=4, **{kk: vv for kk, vv in kw.items() if kk in ("ncut", "cutoff")})
                e00 = models.dressed_level(E, labels, 0, 0)[0]; e01 = models.dressed_level(E, labels, 0, 1)[0]
                e10 = models.dressed_level(E, labels, 1, 0)[0]; e11 = models.dressed_level(E, labels, 1, 1)[0]
                chi_exact = (e11 - e10) - (e01 - e00)
                f01_here = f01[0] if phi0 == 0.0 else f01[-1]
                delta = f01_here - fr
                ratio = abs(g / delta) if delta else float("inf")
                out.append(dict(name=f"qubit–resonator detuning Δ at {lab}", value=delta, unit="GHz", note="f01 − fr"))
                out.append(dict(name=f"g / Δ at {lab}", value=ratio, unit="", note="dispersive regime needs ≪ 1" + ("; NOT dispersive here" if ratio > 0.2 else "")))
                out.append(dict(name=f"dispersive shift χ at {lab} (exact diagonalisation)", value=1e3 * chi_exact, unit="MHz",
                                note="(E11 − E10) − (E01 − E00) from the dressed spectrum; valid even near the crossing"))
                if ratio < 0.2:
                    out.append(dict(name=f"Purcell factor (Δ/g)² at {lab}", value=(delta / g) ** 2 if g else float("nan"), unit="", note="T1,Purcell = (Δ/g)² / κ; multiply by 1/κ of your resonator"))
    elif pl == "nv":
        b = np.array([np.sin(p["theta"]) * np.cos(p["phi"]), np.sin(p["theta"]) * np.sin(p["phi"]), np.cos(p["theta"])])
        proj = models.NV_AXES @ b
        out.append(dict(name="zero-field splitting D", value=1e3 * p["D"], unit="MHz", note="2870 MHz at room temperature; shifts with temperature and strain"))
        out.append(dict(name="strain / electric splitting 2E", value=2e3 * p["E"], unit="MHz", note="line separation at zero field"))
        out.append(dict(name="field per control unit", value=p["field_scale"], unit="G per unit", note="coil calibration"))
        for k, pr in enumerate(proj, start=1):
            out.append(dict(name=f"axial field fraction, NV orientation {k}", value=pr, unit="", note="B·n̂ / |B|; ±1 means aligned"))
        out.append(dict(name="zero-field control value", value=p["x0"], unit="control units", note=""))
    elif pl == "dot":
        out.append(dict(name="tunnel coupling gap 2 t_c", value=2 * p["t_c"], unit="GHz", note="minimum of the hyperbola"))
        out.append(dict(name="2 t_c in µeV", value=2 * p["t_c"] * 4.1357, unit="µeV", note="h × 1 GHz = 4.1357 µeV"))
        out.append(dict(name="lever arm", value=p["lever_arm"], unit="GHz per control unit", note=f"= {p['lever_arm'] * 4.1357:.3g} µeV per unit"))
        out.append(dict(name="zero-detuning control value", value=p["x0"], unit="control units", note="charge degeneracy point"))
    return out


def control_for_frequency(best: dict, target_GHz: float, x_range: tuple[float, float]) -> list[float]:
    """Control values inside x_range where the fitted 0-1 transition equals target_GHz."""
    fam, p = best["family"], best["params"]
    if not models.FAMILIES[fam]["flux_dependent"]:
        return []
    hyp = fit.Hypothesis(fam, best.get("coupled", False))
    xs = np.linspace(x_range[0], x_range[1], 801)
    f = fit.predict_lines(hyp, p, xs)["0-1"]
    f = np.where(np.isfinite(f), f, np.nan)
    d = f - target_GHz
    sols = []
    for i in range(len(xs) - 1):
        if np.isfinite(d[i]) and np.isfinite(d[i + 1]) and d[i] * d[i + 1] <= 0 and d[i] != d[i + 1]:
            sols.append(float(xs[i] - d[i] * (xs[i + 1] - xs[i]) / (d[i + 1] - d[i])))
    return sols


def scqubits_code(best: dict) -> str:
    fam, p, coupled = best["family"], best["params"], best.get("coupled", False)
    lines = ["import scqubits as scq", ""]
    if fam == "transmon":
        lines.append(f"qubit = scq.Transmon(EJ={p['EJ']:.5g}, EC={p['EC']:.5g}, ng=0.0, ncut=30)")
    elif fam == "tunable_transmon":
        lines.append(f"qubit = scq.TunableTransmon(EJmax={p['EJmax']:.5g}, EC={p['EC']:.5g}, d={p['d']:.5g}, flux=0.0, ng=0.0, ncut=30)")
        lines.append(f"# flux = (control - {p['offset']:.5g}) / {p['period']:.5g}")
    elif fam == "fluxonium":
        lines.append(f"qubit = scq.Fluxonium(EJ={p['EJ']:.5g}, EC={p['EC']:.5g}, EL={p['EL']:.5g}, flux=0.0, cutoff=110)")
        lines.append(f"# flux = (control - {p['offset']:.5g}) / {p['period']:.5g}")
    elif fam == "nv_ensemble":
        return (f"# NV ensemble: D = {p['D']:.6g} GHz, E = {p['E']:.4g} GHz, field direction theta = {p['theta']:.4g}, phi = {p['phi']:.4g} rad (crystal frame)\n"
                f"# B[G] = {p['field_scale']:.5g} * (control - {p['x0']:.5g})\n"
                "# H = D Sz^2 + E (Sx^2 - Sy^2) + 2.8025e-3 GHz/G * B.S for each <111> orientation")
    elif fam == "charge_qubit":
        return (f"# Double quantum dot: t_c = {p['t_c']:.5g} GHz, eps[GHz] = {p['lever_arm']:.5g} * (control - {p['x0']:.5g})\n"
                "# f01 = sqrt(eps**2 + (2*t_c)**2)")
    if coupled:
        lines += ["", f"resonator = scq.Oscillator(E_osc={p['fr']:.5g}, truncated_dim=4)",
                  "hilbertspace = scq.HilbertSpace([qubit, resonator])",
                  f"hilbertspace.add_interaction(g_strength={p['g']:.5g}, op1=qubit.n_operator, op2=resonator.creation_operator, add_hc=True)"]
    lines += ["", "print(qubit.eigenvals(evals_count=4))"]
    return "\n".join(lines)


def curves_csv(report: dict) -> str:
    """Fitted transition curves on the data grid as CSV text."""
    xg = report["x_grid"]
    names = list(report["lines"])
    buf = io.StringIO()
    buf.write("control," + ",".join(f"{n}_GHz" for n in names) + "\n")
    for i, xv in enumerate(xg):
        vals = [report["lines"][n][i] for n in names]
        buf.write(f"{xv:.6g}," + ",".join("" if v is None else f"{v:.6f}" for v in vals) + "\n")
    return buf.getvalue()
