"""Fast numpy forward models for superconducting circuits (units: GHz, h = 1).

Conventions follow scqubits exactly (verified in tests/test_models.py):
  transmon:          H = 4 EC (n - ng)^2 - EJ cos(phi)
  tunable_transmon:  EJ(Phi) = EJmax * sqrt(cos^2(pi Phi) + d^2 sin^2(pi Phi))
  fluxonium:         H = 4 EC n^2 - EJ cos(phi - 2 pi Phi) + EL phi^2 / 2
  + resonator:       H = H_q + fr a^dag a + g n_q (a + a^dag)

Everything is vectorised over an array of flux values `phi` (in units of Phi0) so that a
whole flux sweep is one batched eigensolve.
"""
from __future__ import annotations

import numpy as np

# ----------------------------------------------------------------------------- registry
# Each family: ordered parameter names, log-uniform prior bounds (GHz; d is dimensionless),
# and which labelled transitions a two-tone experiment typically shows, with a nominal
# visibility weight used by the synthetic generator.
FAMILIES: dict[str, dict] = {
    # ---- superconducting circuits (control axis = flux, mapped by period/offset)
    "transmon": {
        "platform": "circuit", "label": "fixed-frequency transmon",
        "params": ["EJ", "EC"],
        "bounds": {"EJ": (3.0, 40.0), "EC": (0.15, 0.6)},
        "flux_dependent": False, "control": (),
    },
    "tunable_transmon": {
        "platform": "circuit", "label": "flux-tunable transmon (SQUID)",
        "params": ["EJmax", "EC", "d"],
        "bounds": {"EJmax": (5.0, 50.0), "EC": (0.15, 0.6), "d": (0.0, 0.9)},
        "flux_dependent": True, "control": ("period", "offset"),
    },
    "fluxonium": {
        "platform": "circuit", "label": "fluxonium",
        "params": ["EJ", "EC", "EL"],
        "bounds": {"EJ": (2.0, 12.0), "EC": (0.4, 2.5), "EL": (0.1, 2.0)},
        "flux_dependent": True, "control": ("period", "offset"),
    },
    # ---- NV centre ensemble in diamond: ODMR vs magnetic field (control axis = coil current)
    #      H = D Sz^2 + E (Sx^2 - Sy^2) + gamma_e B.S  for each of the four <111> orientations
    "nv_ensemble": {
        "platform": "nv", "label": "NV-centre ensemble (ODMR)",
        "params": ["D", "E", "theta", "phi"],
        "bounds": {"D": (2.84, 2.90), "E": (0.0003, 0.02), "theta": (0.0, 1.5707963), "phi": (0.0, 0.7853982)},
        "flux_dependent": True, "control": ("field_scale", "x0"),  # B[G] = field_scale * (x - x0)
    },
    # ---- semiconductor double quantum dot charge qubit: f = sqrt(eps^2 + (2 t_c)^2) vs detuning
    "charge_qubit": {
        "platform": "dot", "label": "double-quantum-dot charge qubit",
        "params": ["t_c"],
        "bounds": {"t_c": (0.5, 6.0)},
        "flux_dependent": True, "control": ("lever_arm", "x0"),  # eps[GHz] = lever_arm * (x - x0)
    },
}
CONTROL_BOUNDS = {"field_scale": (20.0, 400.0), "lever_arm": (2.0, 60.0)}
PLATFORMS = {
    "circuit": dict(label="Superconducting circuits", families=["transmon", "tunable_transmon", "fluxonium"], control_label="flux bias (arbitrary units)"),
    "nv": dict(label="NV centres in diamond", families=["nv_ensemble"], control_label="coil current (arbitrary units)"),
    "dot": dict(label="Semiconductor quantum dots", families=["charge_qubit"], control_label="detuning gate voltage (arbitrary units)"),
}
GAMMA_E = 2.8025e-3  # GHz per gauss
NV_AXES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float) / np.sqrt(3.0)


def platform_of(family: str) -> str:
    return FAMILIES[family]["platform"]


def control_params(family: str) -> tuple:
    return tuple(FAMILIES[family]["control"])


def control_from_x(family: str, params: dict, x: np.ndarray) -> np.ndarray:
    """Map the measured control axis to the model's control variable (flux, field or detuning)."""
    x = np.asarray(x, float)
    c = FAMILIES[family]["control"]
    if not c:
        return np.zeros(len(x))
    if c == ("period", "offset"):
        return (x - params["offset"]) / params["period"]
    if c == ("field_scale", "x0"):
        return params["field_scale"] * (x - params["x0"])
    if c == ("lever_arm", "x0"):
        return params["lever_arm"] * (x - params["x0"])
    raise ValueError(c)
RESONATOR_PARAMS = ["fr", "g"]
RESONATOR_BOUNDS = {"fr": (4.5, 9.5), "g": (0.03, 0.35)}

# Lines reported from the ground state. (label, (i, j), n_photons, nominal visibility)
QUBIT_LINES = [
    ("0-1", (0, 1), 1, 1.0),
    ("0-2", (0, 2), 1, 0.35),
    ("0-3", (0, 3), 1, 0.15),
    ("0-2/2ph", (0, 2), 2, 0.45),
    ("1-2", (1, 2), 1, 0.12),
]


NV_LINES = [(f"NV{k}{sgn}", (0, j), 1, 1.0) for k in range(1, 5) for sgn, j in (("-", 1), ("+", 2))]
DQD_LINES = [("0-1", (0, 1), 1, 1.0), ("0-1/2ph", (0, 1), 2, 0.4)]


def family_lines(family: str) -> list:
    pl = platform_of(family)
    return QUBIT_LINES if pl == "circuit" else (NV_LINES if pl == "nv" else DQD_LINES)


def hypothesis_params(family: str, coupled: bool) -> list[str]:
    names = list(FAMILIES[family]["params"])
    if coupled:
        if platform_of(family) != "circuit":
            raise ValueError("readout-resonator coupling is only modelled for superconducting circuits")
        names += RESONATOR_PARAMS
    return names


def hypothesis_bounds(family: str, coupled: bool) -> dict[str, tuple[float, float]]:
    b = dict(FAMILIES[family]["bounds"])
    if coupled:
        b.update(RESONATOR_BOUNDS)
    return b


# ----------------------------------------------------------------------------- operators
def _charge_ops(ncut: int):
    n = np.arange(-ncut, ncut + 1, dtype=float)
    n_op = np.diag(n)
    cos_phi = 0.5 * (np.eye(2 * ncut + 1, k=1) + np.eye(2 * ncut + 1, k=-1))
    return n_op, cos_phi


def _fluxonium_ops(EC: float, EL: float, cutoff: int):
    """Oscillator-basis phi and n operators and e^{i phi} for the LC with (EC, EL)."""
    phi_osc = (8.0 * EC / EL) ** 0.25
    sq = np.sqrt(np.arange(1, cutoff, dtype=float))
    a = np.diag(sq, k=1)
    ad = a.T
    phi_op = (phi_osc / np.sqrt(2.0)) * (a + ad)
    n_op = 1j / (np.sqrt(2.0) * phi_osc) * (ad - a)
    lam, U = np.linalg.eigh(phi_op)
    exp_i_phi = (U * np.exp(1j * lam)) @ U.conj().T
    return phi_op, n_op, exp_i_phi


# ----------------------------------------------------------------------------- bare qubits
def transmon_hamiltonians(EJ: np.ndarray | float, EC: float, ng: float = 0.0, ncut: int = 30):
    """Batched transmon Hamiltonians, EJ may be an array (one per flux point)."""
    EJ = np.atleast_1d(np.asarray(EJ, dtype=float))
    n_op, cos_phi = _charge_ops(ncut)
    kin = 4.0 * EC * (n_op - ng * np.eye(n_op.shape[0])) @ (n_op - ng * np.eye(n_op.shape[0]))
    H = kin[None, :, :] - EJ[:, None, None] * cos_phi[None, :, :]
    return H, n_op


def tunable_EJ(EJmax: float, d: float, phi: np.ndarray) -> np.ndarray:
    phi = np.asarray(phi, dtype=float)
    return EJmax * np.sqrt(np.cos(np.pi * phi) ** 2 + d**2 * np.sin(np.pi * phi) ** 2)


def fluxonium_hamiltonians(EJ: float, EC: float, EL: float, phi: np.ndarray, cutoff: int = 60):
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    phi_op, n_op, exp_i_phi = _fluxonium_ops(EC, EL, cutoff)
    H_lc = 4.0 * EC * (n_op @ n_op) + 0.5 * EL * (phi_op @ phi_op)
    H_lc = H_lc.real if np.allclose(H_lc.imag, 0) else H_lc
    # -EJ cos(phi - 2 pi Phi) = -EJ/2 (e^{i phi} e^{-2 pi i Phi} + h.c.)
    ph = np.exp(-2j * np.pi * phi)[:, None, None]
    cos_term = 0.5 * (exp_i_phi[None] * ph + np.conj(exp_i_phi[None] * ph).transpose(0, 2, 1))
    H = H_lc[None, :, :] - EJ * cos_term
    return H, n_op


def qubit_hamiltonians(family: str, params: dict, phi: np.ndarray, **kw):
    """Return (H batched over phi, charge operator n in the same basis)."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    if family == "transmon":
        EJ = np.full(phi.shape, params["EJ"])
        return transmon_hamiltonians(EJ, params["EC"], ng=params.get("ng", 0.0), ncut=kw.get("ncut", 30))
    if family == "tunable_transmon":
        EJ = tunable_EJ(params["EJmax"], params["d"], phi)
        return transmon_hamiltonians(EJ, params["EC"], ng=params.get("ng", 0.0), ncut=kw.get("ncut", 30))
    if family == "fluxonium":
        return fluxonium_hamiltonians(params["EJ"], params["EC"], params["EL"], phi, cutoff=kw.get("cutoff", 60))
    raise ValueError(f"unknown family {family}")


def qubit_energies(family: str, params: dict, phi: np.ndarray, nlev: int = 5, **kw) -> np.ndarray:
    """Lowest `nlev` eigenenergies, shape (len(phi), nlev), referenced to E0 = 0 is NOT applied."""
    H, _ = qubit_hamiltonians(family, params, phi, **kw)
    ev = np.linalg.eigvalsh(H)
    return ev[:, :nlev]


# ----------------------------------------------------------------------------- resonator
def dressed_energies(
    family: str,
    params: dict,
    phi: np.ndarray,
    kq: int = 6,
    kr: int = 4,
    **kw,
):
    """Qubit + resonator. Returns (E_dressed (nphi, kq*kr), labels (nphi, kq*kr, 2)) where
    labels[p, s] = (qubit level, photon number) of the bare state with maximum overlap."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    H, n_op = qubit_hamiltonians(family, params, phi, **kw)
    evals, evecs = np.linalg.eigh(H)
    Eq = evals[:, :kq]
    V = evecs[:, :, :kq]
    # charge operator in truncated qubit eigenbasis
    n_eig = np.einsum("pai,ab,pbj->pij", V.conj(), n_op, V)
    fr, g = params["fr"], params["g"]
    sq = np.sqrt(np.arange(1, kr, dtype=float))
    a = np.diag(sq, k=1)
    x = a + a.T
    Hr = fr * np.diag(np.arange(kr, dtype=float))
    Iq = np.eye(kq)
    Ir = np.eye(kr)
    Hq = np.einsum("pi,ij->pij", Eq, Iq)  # diagonal per phi
    Hfull = (
        np.einsum("pij,kl->pikjl", Hq, Ir).reshape(len(phi), kq * kr, kq * kr)
        + np.einsum("ij,kl->ikjl", Iq, Hr).reshape(kq * kr, kq * kr)[None]
        + g * np.einsum("pij,kl->pikjl", n_eig, x).reshape(len(phi), kq * kr, kq * kr)
    )
    Hfull = 0.5 * (Hfull + np.conj(Hfull).transpose(0, 2, 1))
    E, W = np.linalg.eigh(Hfull)
    # bare state index b = i*kr + k ; assign each dressed state to the bare state of max overlap
    overlap = np.abs(W) ** 2  # (p, bare, dressed)
    bare_of_dressed = np.argmax(overlap, axis=1)  # (p, dressed)
    labels = np.stack([bare_of_dressed // kr, bare_of_dressed % kr], axis=-1)
    return E, labels


def dressed_level(E, labels, i: int, k: int) -> np.ndarray:
    """Energy of the dressed state labelled (qubit i, photons k) at each phi (nan if absent)."""
    out = np.full(E.shape[0], np.nan)
    mask = (labels[..., 0] == i) & (labels[..., 1] == k)
    for p in range(E.shape[0]):
        idx = np.flatnonzero(mask[p])
        if idx.size:
            # if several dressed states claim the same label, take the one with the largest overlap
            out[p] = E[p, idx[0]]
    return out


# ----------------------------------------------------------------------------- other platforms
_SX = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], complex) / np.sqrt(2)
_SY = np.array([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]], complex) / np.sqrt(2)
_SZ = np.diag([1.0, 0.0, -1.0]).astype(complex)


def nv_lines(params: dict, B: np.ndarray) -> dict[str, np.ndarray]:
    """ODMR transition frequencies (GHz) of the four NV orientations vs field magnitude B (gauss).

    The field direction (theta, phi) is given in the diamond crystal frame; for each NV axis the field
    is decomposed into axial and transverse parts and the S = 1 Hamiltonian is diagonalised."""
    B = np.atleast_1d(np.asarray(B, float))
    D, E, th, ph = params["D"], params["E"], params["theta"], params["phi"]
    bdir = np.array([np.sin(th) * np.cos(ph), np.sin(th) * np.sin(ph), np.cos(th)])
    H0 = D * (_SZ @ _SZ) + E * (_SX @ _SX - _SY @ _SY)
    out = {}
    for k, n in enumerate(NV_AXES, start=1):
        bz = float(bdir @ n)
        bperp = float(np.sqrt(max(1.0 - bz * bz, 0.0)))
        Hz = GAMMA_E * (bz * _SZ + bperp * _SX)
        H = H0[None] + B[:, None, None] * Hz[None]
        ev = np.linalg.eigvalsh(H)
        out[f"NV{k}-"] = ev[:, 1] - ev[:, 0]
        out[f"NV{k}+"] = ev[:, 2] - ev[:, 0]
    return out


def dqd_lines(params: dict, eps: np.ndarray) -> dict[str, np.ndarray]:
    eps = np.atleast_1d(np.asarray(eps, float))
    f = np.sqrt(eps**2 + (2.0 * params["t_c"]) ** 2)
    return {"0-1": f, "0-1/2ph": f / 2.0}


# ----------------------------------------------------------------------------- transitions
def transitions(family: str, params: dict, phi: np.ndarray, coupled: bool = False, **kw) -> dict[str, np.ndarray]:
    """Spectroscopy lines (GHz) vs the control variable (flux / field / detuning) for a hypothesis."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
    pl = platform_of(family)
    if pl == "nv":
        return nv_lines(params, phi)
    if pl == "dot":
        return dqd_lines(params, phi)
    lines: dict[str, np.ndarray] = {}
    if not coupled:
        E = qubit_energies(family, params, phi, nlev=5, **kw)
        for label, (i, j), nph, _w in QUBIT_LINES:
            lines[label] = (E[:, j] - E[:, i]) / nph
        return lines
    E, labels = dressed_energies(family, params, phi, **kw)
    E0 = dressed_level(E, labels, 0, 0)
    for label, (i, j), nph, _w in QUBIT_LINES:
        Ei = E0 if i == 0 else dressed_level(E, labels, i, 0)
        Ej = dressed_level(E, labels, j, 0)
        lines[label] = (Ej - Ei) / nph
    lines["res"] = dressed_level(E, labels, 0, 1) - E0
    return lines


def line_weights(family: str = "transmon") -> dict[str, float]:
    w = {label: vis for label, _, _, vis in family_lines(family)}
    if platform_of(family) == "circuit":
        w["res"] = 0.5
    return w
