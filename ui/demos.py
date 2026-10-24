"""Demo circuits for every platform. Each entry renders a hidden truth deterministically."""
from __future__ import annotations

import numpy as np

from circuitsleuth import models, synth

SAVED = {  # precomputed reports (normal preset, four workers) for the slow circuit demos
    "A": "artifacts/demo_A_report.json",
    "B": "artifacts/demo_B_report.json",
    "E": "artifacts/nguyen2019_report_calibrated.json",
}

DEMOS = {
    "circuit": {
        "A · tunable transmon through a 7 GHz readout resonator, plus a pump spur": dict(
            family="tunable_transmon", params=dict(EJmax=25.0, EC=0.3, d=0.3, fr=7.0, g=0.1), period=1.3, offset=0.2, f_lo=3.0, f_hi=9.0,
            spurs=[dict(f=5.1, amp=0.9, sideband=0.0)], seed=1),
        "B · fluxonium": dict(family="fluxonium", params=dict(EJ=6.0, EC=1.0, EL=0.6), period=1.1, offset=-0.3, f_lo=0.3, f_hi=9.0, spurs=[], seed=3),
        "C · fixed-frequency transmon with LO leakage": dict(family="transmon", params=dict(EJ=15.0, EC=0.28), period=1.0, offset=0.0, f_lo=3.0, f_hi=8.0,
                                                              spurs=[dict(f=6.2, amp=0.6, sideband=0.1)], seed=2),
        "D · surprise: random hidden circuit": None,
        "E · REAL DATA: fluxonium device A, Nguyen et al. PRX 2019 (arXiv:1810.11006)": "real",
    },
    "nv": {
        "F · NV-centre ensemble, field 34° off a <111> axis": dict(family="nv_ensemble", params=dict(D=2.870, E=0.004, theta=0.6, phi=0.3, field_scale=120.0, x0=0.1),
                                                               f_lo=2.4, f_hi=3.35, spurs=[], seed=1, linewidth=0.006),
        "G · NV-centre ensemble, surprise field direction": None,
    },
    "dot": {
        "H · double quantum dot charge qubit, t_c = 2.5 GHz": dict(family="charge_qubit", params=dict(t_c=2.5, lever_arm=9.0, x0=0.15), f_lo=1.0, f_hi=12.0, spurs=[], seed=2, linewidth=0.03),
        "I · double quantum dot, surprise": None,
    },
}


def make_demo(platform: str, name: str, seed: int | None = None):
    """Return (truth, dict(image, x, f)) for a demo entry."""
    spec = DEMOS[platform][name]
    cfg = synth.SynthConfig(nx=81, nf=400 if platform != "nv" else 500)
    if spec == "real":
        d = np.load("data/demo/nguyen2019_devA.npz")
        truth = dict(family="fluxonium", platform="circuit", coupled=False, params=dict(EJ=float(d["truth_EJ"]), EC=float(d["truth_EC"]), EL=float(d["truth_EL"])), period=1.0, offset=0.0, spurs=[],
                     note="digitised from the paper's Fig. 2 (vector PDF); paper fit EJ=3, EC=0.84, EL=1 GHz; axis already in flux quanta so period should come out as 1")
        return truth, dict(image=d["image"], x=d["x"], f=d["f"])
    if spec is None:
        rng = np.random.default_rng(seed or 0)
        truth = synth.sample_truth(rng, synth.SynthConfig(nx=81, nf=400, noise_range=(0.03, 0.1), max_spurs=1 if platform == "circuit" else 0), platform=platform)
        r = synth.render(truth, cfg, rng)
        return truth, r
    fam = spec["family"]
    truth = dict(family=fam, platform=models.platform_of(fam), coupled="fr" in spec["params"], params=dict(spec["params"]), period=spec.get("period", 1.0), offset=spec.get("offset", 0.0),
                 x_lo=-1.0, x_hi=1.0, f_lo=spec["f_lo"], f_hi=spec["f_hi"], linewidth=spec.get("linewidth", 0.025), noise=0.06, two_photon=True, sign=1, spurs=list(spec["spurs"]))
    r = synth.render(truth, cfg, np.random.default_rng(spec["seed"]))
    return truth, r
