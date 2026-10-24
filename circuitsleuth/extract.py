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


# ----------------------------------------------------------------------------- tracks
def link_tracks(points: np.ndarray, df_tol: float = 0.08) -> np.ndarray:
    """Greedy column-to-column linking of points into line segments ("tracks").

    Returns an integer track id per point. Two points in adjacent columns join the same track when
    their frequencies differ by less than `df_tol` GHz; each point links to at most one predecessor.
    """
    n = len(points)
    ids = -np.ones(n, int)
    if n == 0:
        return ids
    cols = np.unique(points[:, 0])
    col_of = np.searchsorted(cols, points[:, 0])
    next_id = 0
    prev_idx: np.ndarray = np.array([], int)
    for c in range(len(cols)):
        cur_idx = np.flatnonzero(col_of == c)
        cur_idx = cur_idx[np.argsort(points[cur_idx, 1])]
        taken = set()
        for i in cur_idx:
            best, best_d = -1, df_tol
            for j in prev_idx:
                if j in taken:
                    continue
                d = abs(points[i, 1] - points[j, 1])
                if d < best_d:
                    best, best_d = j, d
            if best >= 0:
                ids[i] = ids[best]
                taken.add(best)
            else:
                ids[i] = next_id
                next_id += 1
        prev_idx = cur_idx
    return ids


def flat_lines(points: np.ndarray, tol: float = 0.02, min_frac: float = 0.6) -> list[float]:
    """Frequencies at which a feature is present in >= min_frac of all flux columns (flux-independent lines)."""
    cols = np.unique(points[:, 0])
    ncol = len(cols)
    if ncol == 0:
        return []
    f = points[:, 1]
    order = np.argsort(f)
    fs, xs = f[order], points[order, 0]
    found: list[float] = []
    i = 0
    while i < len(fs):
        j = i
        while j + 1 < len(fs) and fs[j + 1] - fs[i] < 2 * tol:
            j += 1
        n_cols = len(np.unique(xs[i : j + 1]))
        if n_cols >= min_frac * ncol:
            fc = float(np.median(fs[i : j + 1]))
            if not found or abs(fc - found[-1]) > 2 * tol:
                found.append(fc)
            i = j + 1
        else:
            i += 1
    return found


def flat_feature_mask(points: np.ndarray, tol: float = 0.02, min_frac: float = 0.6) -> np.ndarray:
    """Per-point boolean: lies on a flux-independent line (pump/LO/mixer spur, or a fixed-frequency line)."""
    mask = np.zeros(len(points), bool)
    for fc in flat_lines(points, tol, min_frac):
        mask |= np.abs(points[:, 1] - fc) < 1.5 * tol
    return mask


def flat_track_mask(points: np.ndarray, track_ids: np.ndarray, span_tol: float = 0.045, min_frac: float = 0.5) -> np.ndarray:
    """Backward-compatible alias: flat detection by frequency occupancy (robust to track swaps at crossings)."""
    return flat_feature_mask(points, tol=span_tol / 2, min_frac=min_frac)


def track_weights(points: np.ndarray, track_ids: np.ndarray, flat: np.ndarray) -> dict[int, float]:
    """Weight per track: sqrt(length) * mean point weight, halved for flat tracks (prior: often spurious)."""
    out = {}
    for t in np.unique(track_ids):
        m = track_ids == t
        w = np.sqrt(m.sum()) * float(points[m, 2].mean())
        if flat[m].any():
            w *= 0.5
        out[int(t)] = w
    return out


def summarize_tracks(points: np.ndarray, track_ids: np.ndarray, mask: np.ndarray | None = None) -> list[dict]:
    """Human-readable description of tracks (optionally only those selected by `mask`)."""
    out = []
    for t in np.unique(track_ids):
        m = track_ids == t
        if mask is not None and not (m & mask).sum() >= 0.5 * m.sum():
            continue
        f = points[m, 1]
        x = points[m, 0]
        out.append(dict(track=int(t), n_points=int(m.sum()), f_lo=float(f.min()), f_hi=float(f.max()), f_mean=float(f.mean()),
                        x_lo=float(x.min()), x_hi=float(x.max()), flat=bool((f.max() - f.min()) < 0.045), weight=float(points[m, 2].mean())))
    return sorted(out, key=lambda d: -d["n_points"])
