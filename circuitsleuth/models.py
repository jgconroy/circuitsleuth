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
    "transmon": {
        "params": ["EJ", "EC"],
        "bounds": {"EJ": (3.0, 40.0), "EC": (0.15, 0.6)},
        "flux_dependent": False,
    },
    "tunable_transmon": {
        "params": ["EJmax", "EC", "d"],
        "bounds": {"EJmax": (5.0, 50.0), "EC": (0.15, 0.6), "d": (0.0, 0.9)},
        "flux_dependent": True,
    },
    "fluxonium": {
        "params": ["EJ", "EC", "EL"],
        "bounds": {"EJ": (2.0, 12.0), "EC": (0.4, 2.5), "EL": (0.1, 2.0)},
        "flux_dependent": True,
    },
}
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


def hypothesis_params(family: str, coupled: bool) -> list[str]:
    names = list(FAMILIES[family]["params"])
    if coupled:
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


# ----------------------------------------------------------------------------- transitions
def transitions(family: str, params: dict, phi: np.ndarray, coupled: bool = False, **kw) -> dict[str, np.ndarray]:
    """Spectroscopy lines (GHz) vs phi for a hypothesis.  Keys are line labels."""
    phi = np.atleast_1d(np.asarray(phi, dtype=float))
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


def line_weights() -> dict[str, float]:
    w = {label: vis for label, _, _, vis in QUBIT_LINES}
    w["res"] = 0.5
    return w
