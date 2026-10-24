import numpy as np
import pytest
import scipy.sparse as sp
import scqubits as scq


def dense(op):
    return np.asarray(op.todense()) if sp.issparse(op) else np.asarray(op)

from circuitsleuth import models

scq.settings.PROGRESSBAR_DISABLED = True


def test_transmon_matches_scqubits():
    EJ, EC, ng = 20.0, 0.3, 0.0
    ours = models.qubit_energies("transmon", {"EJ": EJ, "EC": EC}, np.array([0.0]), nlev=5, ncut=30)[0]
    ref = scq.Transmon(EJ=EJ, EC=EC, ng=ng, ncut=30).eigenvals(evals_count=5)
    assert np.allclose(ours, ref, atol=1e-6)


def test_tunable_transmon_matches_scqubits():
    EJmax, EC, d = 30.0, 0.25, 0.3
    phis = np.linspace(-0.5, 0.5, 7)
    ours = models.qubit_energies("tunable_transmon", {"EJmax": EJmax, "EC": EC, "d": d}, phis, nlev=4, ncut=30)
    for p, phi in enumerate(phis):
        ref = scq.TunableTransmon(EJmax=EJmax, EC=EC, d=d, flux=phi, ng=0.0, ncut=30).eigenvals(evals_count=4)
        assert np.allclose(ours[p], ref, atol=1e-6), phi


def test_fluxonium_matches_scqubits():
    EJ, EC, EL = 8.9, 2.5, 0.5
    phis = np.linspace(0.0, 1.0, 9)
    ours = models.qubit_energies("fluxonium", {"EJ": EJ, "EC": EC, "EL": EL}, phis, nlev=5, cutoff=110)
    for p, phi in enumerate(phis):
        ref = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, flux=phi, cutoff=110).eigenvals(evals_count=5)
        assert np.allclose(ours[p], ref, atol=1e-6), phi


def test_fluxonium_cutoff_60_is_converged_for_low_levels():
    EJ, EC, EL = 8.9, 2.5, 0.5
    phis = np.linspace(0.0, 1.0, 5)
    lo = models.qubit_energies("fluxonium", {"EJ": EJ, "EC": EC, "EL": EL}, phis, nlev=4, cutoff=60)
    hi = models.qubit_energies("fluxonium", {"EJ": EJ, "EC": EC, "EL": EL}, phis, nlev=4, cutoff=120)
    assert np.allclose(lo, hi, atol=1e-5)


def scq_coupled_reference(qubit, kq, kr, fr, g):
    """Dressed energies built from scqubits eigenstates: H = diag(E_q) + fr a^dag a + g n_q (a + a^dag)."""
    evals, evecs = qubit.eigensys(evals_count=kq)
    n_eig = evecs.conj().T @ dense(qubit.n_operator()) @ evecs
    a = np.diag(np.sqrt(np.arange(1, kr, dtype=float)), k=1)
    H = (np.kron(np.diag(evals), np.eye(kr)) + np.kron(np.eye(kq), fr * np.diag(np.arange(kr, dtype=float)))
         + g * np.kron(n_eig, a + a.T))
    return np.linalg.eigvalsh(H)


def test_coupled_transmon_matches_scqubits_reference():
    EJ, EC, fr, g = 20.0, 0.3, 7.0, 0.1
    kq, kr = 6, 4
    tmon = scq.Transmon(EJ=EJ, EC=EC, ng=0.0, ncut=30)
    ref = scq_coupled_reference(tmon, kq, kr, fr, g)
    E, labels = models.dressed_energies("transmon", {"EJ": EJ, "EC": EC, "fr": fr, "g": g}, np.array([0.0]), kq=kq, kr=kr, ncut=30)
    assert np.allclose(np.sort(E[0]), np.sort(ref), atol=1e-6)


def test_coupled_fluxonium_matches_scqubits_reference():
    EJ, EC, EL, fr, g = 6.0, 1.0, 0.6, 7.2, 0.15
    kq, kr = 6, 4
    phi = 0.37
    fl = scq.Fluxonium(EJ=EJ, EC=EC, EL=EL, flux=phi, cutoff=110)
    ref = scq_coupled_reference(fl, kq, kr, fr, g)
    E, labels = models.dressed_energies("fluxonium", {"EJ": EJ, "EC": EC, "EL": EL, "fr": fr, "g": g}, np.array([phi]), kq=kq, kr=kr, cutoff=110)
    assert np.allclose(np.sort(E[0]), np.sort(ref), atol=1e-6)


def test_avoided_crossing_shows_in_dressed_01_line():
    # tunable transmon sweeping through a 7 GHz resonator: dressed 0-1 line must deviate from bare near crossing
    params = {"EJmax": 40.0, "EC": 0.25, "d": 0.0, "fr": 7.0, "g": 0.12}
    phi = np.linspace(0.0, 0.45, 91)
    bare = models.transitions("tunable_transmon", params, phi, coupled=False)["0-1"]
    dressed = models.transitions("tunable_transmon", params, phi, coupled=True)["0-1"]
    assert np.nanmax(np.abs(dressed - bare)) > 0.05  # pulls by more than 50 MHz near crossing
    far = np.abs(bare - 7.0) > 1.5
    assert np.nanmax(np.abs(dressed[far] - bare[far])) < 0.03  # tiny dispersive shift far away


def test_transitions_keys_and_shapes():
    phi = np.linspace(-0.5, 0.5, 11)
    for fam, params in [
        ("transmon", {"EJ": 15.0, "EC": 0.3}),
        ("tunable_transmon", {"EJmax": 25.0, "EC": 0.3, "d": 0.2}),
        ("fluxonium", {"EJ": 5.0, "EC": 1.0, "EL": 0.5}),
    ]:
        lines = models.transitions(fam, params, phi)
        assert set(lines) == {l for l, *_ in models.QUBIT_LINES}
        for v in lines.values():
            assert v.shape == phi.shape
        assert np.all(lines["0-1"] > 0)
