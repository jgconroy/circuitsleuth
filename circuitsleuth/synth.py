"""Synthetic two-tone spectroscopy maps with realistic nuisances.

A sample = hidden truth (family, params, flux calibration) + rendered map + ground-truth points.
Nuisances: uncalibrated flux axis (period/offset in 'volts'), finite band, per-line visibility,
dropout segments, Lorentzian linewidth, background noise/drift, flux-independent spurious lines
(pumps, LO leakage, mixer spurs) with optional sidebands.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import models

FAMILY_NAMES = list(models.PLATFORMS["circuit"]["families"])  # the CNN proposer is trained on circuits
SPUR_LABEL = -1
BANDS = {  # per platform: ranges for the lower and upper edge of the measured band (GHz)
    "circuit": ((0.05, 1.5), (7.5, 13.0)),
    "nv": ((2.3, 2.6), (3.1, 3.45)),
    "dot": ((0.5, 1.5), (8.0, 14.0)),
}


@dataclass
class SynthConfig:
    nx: int = 101
    nf: int = 400
    f_lo_range: tuple[float, float] = (0.05, 1.5)
    f_hi_range: tuple[float, float] = (7.5, 13.0)
    periods_in_window: tuple[float, float] = (0.6, 2.5)
    linewidth_range: tuple[float, float] = (0.01, 0.08)  # GHz
    noise_range: tuple[float, float] = (0.03, 0.30)
    max_spurs: int = 3
    p_coupled: float = 0.5
    p_two_photon: float = 0.6
    dropout_max_segments: int = 3


def _loguniform(rng, lo, hi):
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


LINEAR_PARAMS = {"d", "D", "theta", "phi", "x0", "offset"}


def sample_truth(rng: np.random.Generator, cfg: SynthConfig | None = None, family: str | None = None, coupled: bool | None = None, platform: str = "circuit") -> dict:
    cfg = cfg or SynthConfig()
    fams = models.PLATFORMS[platform]["families"]
    family = family or fams[rng.integers(len(fams))]
    platform = models.platform_of(family)
    coupled = (bool(rng.random() < cfg.p_coupled) if coupled is None else coupled) if platform == "circuit" else False
    bounds = models.hypothesis_bounds(family, coupled)
    params = {}
    for name, (lo, hi) in bounds.items():
        params[name] = float(rng.uniform(lo, hi)) if name in LINEAR_PARAMS else _loguniform(rng, lo, hi)
    if platform == "circuit":
        f_lo = float(rng.uniform(*cfg.f_lo_range))
        f_hi = float(rng.uniform(*cfg.f_hi_range))
    else:
        (a, b), (c, d) = BANDS[platform]
        f_lo, f_hi = float(rng.uniform(a, b)), float(rng.uniform(c, d))
    x_lo, x_hi = -1.0, 1.0
    ctrl = models.control_params(family)
    period, offset = 1.0, 0.0
    if ctrl == ("period", "offset"):
        n_per = float(rng.uniform(*cfg.periods_in_window))
        period = (x_hi - x_lo) / n_per
        offset = float(rng.uniform(x_lo, x_hi))
    elif ctrl == ("field_scale", "x0"):
        params["field_scale"] = _loguniform(rng, 30.0, 200.0)
        params["x0"] = float(rng.uniform(-0.6, 0.6))
    elif ctrl == ("lever_arm", "x0"):
        params["lever_arm"] = _loguniform(rng, 4.0, 30.0)
        params["x0"] = float(rng.uniform(-0.5, 0.5))
    truth = dict(
        family=family,
        platform=platform,
        coupled=coupled,
        params=params,
        period=period,
        offset=offset,
        x_lo=x_lo,
        x_hi=x_hi,
        f_lo=f_lo,
        f_hi=f_hi,
        linewidth=float(rng.uniform(*(cfg.linewidth_range if platform == "circuit" else models.PLATFORMS[platform]["linewidth"]))),
        noise=float(rng.uniform(*cfg.noise_range)),
        two_photon=bool(rng.random() < cfg.p_two_photon),
        sign=int(rng.choice([1, -1], p=[0.7, 0.3])),
    )
    # spurious flux-independent lines
    nsp = int(rng.integers(0, cfg.max_spurs + 1))
    spurs = []
    for _ in range(nsp):
        f0 = float(rng.uniform(f_lo, f_hi))
        amp = float(rng.uniform(0.2, 1.0))
        side = float(rng.choice([0.0, rng.uniform(0.05, 0.5)], p=[0.6, 0.4]))
        spurs.append(dict(f=f0, amp=amp, sideband=side))
    truth["spurs"] = spurs
    return truth


def flux_from_x(x: np.ndarray, period: float, offset: float) -> np.ndarray:
    return (np.asarray(x) - offset) / period


def true_lines(truth: dict, x: np.ndarray) -> dict[str, np.ndarray]:
    fam = truth["family"]
    ctrl = models.control_params(fam)
    if ctrl == ("period", "offset") or not ctrl:
        phi = flux_from_x(x, truth["period"], truth["offset"])
        circuit = {k: v for k, v in truth["params"].items() if k not in ctrl}
        return models.transitions(fam, circuit, phi, coupled=truth["coupled"])
    circuit = {k: v for k, v in truth["params"].items() if k not in ctrl}
    return models.transitions(fam, circuit, models.control_from_x(fam, truth["params"], x), coupled=False)


def render(truth: dict, cfg: SynthConfig | None = None, rng: np.random.Generator | None = None) -> dict:
    """Render a map. Returns dict(image (nf,nx), x (nx,), f (nf,), points (N,3), labels (N,), line_names)."""
    cfg = cfg or SynthConfig()
    rng = rng or np.random.default_rng(0)
    x = np.linspace(truth["x_lo"], truth["x_hi"], cfg.nx)
    f = np.linspace(truth["f_lo"], truth["f_hi"], cfg.nf)
    lines = true_lines(truth, x)
    weights = models.line_weights(truth["family"])
    lw = truth["linewidth"]
    img = np.zeros((cfg.nf, cfg.nx))
    pts, labs = [], []
    line_names = list(lines)
    for li, name in enumerate(line_names):
        freq = lines[name]
        if name.endswith("2ph") and not truth["two_photon"]:
            continue
        amp = weights[name] * float(rng.uniform(0.5, 1.2))
        if name == "res":
            amp *= float(rng.choice([0.0, 1.0], p=[0.5, 0.5]))  # readout resonator not always in two-tone band
        if amp <= 0:
            continue
        visible = np.ones(cfg.nx, bool)
        for _ in range(int(rng.integers(0, cfg.dropout_max_segments + 1))):
            a = int(rng.integers(0, cfg.nx))
            b = min(cfg.nx, a + int(rng.integers(3, cfg.nx // 3 + 1)))
            visible[a:b] = False
        # higher lines fade with a smooth random envelope
        env = 0.6 + 0.4 * np.cos(2 * np.pi * rng.uniform() + np.linspace(0, 2 * np.pi * rng.uniform(0.3, 2), cfg.nx))
        for c in range(cfg.nx):
            fc = freq[c]
            if not visible[c] or not np.isfinite(fc) or fc < f[0] or fc > f[-1]:
                continue
            a_c = amp * env[c]
            img[:, c] += a_c * lw**2 / ((f - fc) ** 2 + lw**2)
            pts.append((x[c], fc, a_c))
            labs.append(li)
    for sp in truth["spurs"]:
        for fc, a_c in [(sp["f"], sp["amp"])] + ([(sp["f"] - sp["sideband"], 0.4 * sp["amp"]), (sp["f"] + sp["sideband"], 0.4 * sp["amp"])] if sp["sideband"] > 0 else []):
            if fc < f[0] or fc > f[-1]:
                continue
            w_sp = lw * float(rng.uniform(0.5, 1.5))
            img += a_c * w_sp**2 / ((f[:, None] - fc) ** 2 + w_sp**2)
            for c in range(cfg.nx):
                pts.append((x[c], fc, a_c))
                labs.append(SPUR_LABEL)
    # background drift + noise, then sign
    drift = 0.1 * np.sin(np.linspace(0, 2 * np.pi * rng.uniform(0.2, 1.5), cfg.nx))[None, :]
    img = img + drift + truth["noise"] * rng.standard_normal(img.shape)
    img *= truth["sign"]
    points = np.array(pts, dtype=float).reshape(-1, 3)
    labels = np.array(labs, dtype=int)
    return dict(image=img, x=x, f=f, points=points, labels=labels, line_names=line_names, lines=lines)


def downsample(image: np.ndarray, size: int = 64) -> np.ndarray:
    """Robustly normalised, sign-corrected, (size,size) thumbnail for the CNN (block max-pool)."""
    img = image - np.median(image, axis=0, keepdims=True)
    mad = np.median(np.abs(img)) + 1e-9
    img = img / mad
    if np.mean(np.clip(img, -50, 50) ** 3) < 0:  # dips -> peaks
        img = -img
    img = np.clip(img, 0, 20) / 20.0
    # upsample by repetition if smaller than the target so every block is non-empty
    rep_f = int(np.ceil(size / img.shape[0]))
    rep_x = int(np.ceil(size / img.shape[1]))
    if rep_f > 1:
        img = np.repeat(img, rep_f, axis=0)
    if rep_x > 1:
        img = np.repeat(img, rep_x, axis=1)
    nf, nx = img.shape
    fe = (np.arange(size) * nf) // size
    xe = (np.arange(size) * nx) // size
    out = np.maximum.reduceat(np.maximum.reduceat(img, fe, axis=0), xe, axis=1)
    return out.astype(np.float32)


def make_dataset(n: int, seed: int = 0, cfg: SynthConfig | None = None, size: int = 64):
    """Arrays for classifier training: thumbnails, family index, coupled flag, normalised log-params."""
    cfg = cfg or SynthConfig()
    rng = np.random.default_rng(seed)
    X = np.zeros((n, size, size), np.float32)
    fam = np.zeros(n, np.int64)
    cpl = np.zeros(n, np.int64)
    truths = []
    for i in range(n):
        t = sample_truth(rng, cfg)
        r = render(t, cfg, rng)
        X[i] = downsample(r["image"], size)
        fam[i] = FAMILY_NAMES.index(t["family"])
        cpl[i] = int(t["coupled"])
        truths.append(t)
    return X, fam, cpl, truths
