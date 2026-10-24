"""Turn a 2D spectroscopy map into a weighted point cloud of candidate spectral features."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def normalise(image: np.ndarray) -> np.ndarray:
    """Per-column median removal, global MAD scaling, sign so that lines are positive peaks."""
    img = image - np.median(image, axis=0, keepdims=True)
    mad = np.median(np.abs(img)) + 1e-12
    img = img / (1.4826 * mad)
    if np.mean(np.clip(img, -50, 50) ** 3) < 0:
        img = -img
    return img


def extract_points(image: np.ndarray, x: np.ndarray, f: np.ndarray, snr: float = 4.0, max_per_column: int = 8, min_sep_px: int = 2) -> np.ndarray:
    """Return (N,3) array of (x, f, weight) for peaks found column by column.

    Peaks are detected against a robust noise estimate, refined to sub-pixel precision with a
    parabola through the three points around the maximum, and weighted by prominence.
    """
    img = normalise(image)
    df = f[1] - f[0]
    out = []
    for c in range(img.shape[1]):
        col = img[:, c]
        # height is measured from the per-column baseline in robust-sigma units, so Gaussian
        # noise almost never clears it; prominence alone lets ~4-sigma noise wiggles through.
        peaks, props = find_peaks(col, height=snr, prominence=0.5 * snr, distance=min_sep_px)
        if peaks.size == 0:
            continue
        prom = props["peak_heights"]
        order = np.argsort(prom)[::-1][:max_per_column]
        for k in order:
            p = peaks[k]
            if 0 < p < len(col) - 1:
                y0, y1, y2 = col[p - 1], col[p], col[p + 1]
                denom = y0 - 2 * y1 + y2
                shift = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
                shift = float(np.clip(shift, -0.5, 0.5))
            else:
                shift = 0.0
            out.append((x[c], f[p] + shift * df, prom[k]))
    pts = np.array(out, dtype=float).reshape(-1, 3)
    if pts.size:
        pts[:, 2] = pts[:, 2] / pts[:, 2].max()
    return pts


def match_points(found: np.ndarray, truth: np.ndarray, tol_f: float) -> float:
    """Fraction of truth points that have a found point in the same column within tol_f (GHz)."""
    if truth.size == 0:
        return 1.0
    hit = 0
    for xt, ft, _ in truth:
        same_col = found[np.isclose(found[:, 0], xt)]
        if same_col.size and np.min(np.abs(same_col[:, 1] - ft)) <= tol_f:
            hit += 1
    return hit / len(truth)
