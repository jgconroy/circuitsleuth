"""Turn a 2D spectroscopy map into a weighted point cloud of candidate spectral features."""
from __future__ import annotations

import numpy as np
from scipy.signal import find_peaks


def normalise(image: np.ndarray, per_column: bool = True) -> np.ndarray:
    """Per-column median removal, robust (MAD) noise scaling, sign so that lines are positive peaks.

    With per_column=True each flux column is scaled by its own noise level, so columns with a noisy
    drive/readout (common in real data: vertical streaks) do not flood the point cloud with false peaks."""
    img = image - np.median(image, axis=0, keepdims=True)
    mad_g = np.median(np.abs(img)) + 1e-12
    if per_column:
        mad_c = np.median(np.abs(img), axis=0, keepdims=True)
        mad = np.maximum(mad_c, mad_g) + 1e-12  # only ever raise the noise estimate for noisy columns
    else:
        mad = mad_g
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


# ----------------------------------------------------------------------------- flux calibration
def brightest_per_column(points: np.ndarray):
    cols = np.unique(points[:, 0])
    fb = np.full(len(cols), np.nan)
    for i, c in enumerate(cols):
        m = points[:, 0] == c
        if m.any():
            fb[i] = points[m, 1][np.argmax(points[m, 2])]
    return cols, fb


def _column_matrix(points: np.ndarray):
    cols = np.unique(points[:, 0])
    idx = np.searchsorted(cols, points[:, 0])
    kmax = int(np.bincount(idx).max())
    F = np.full((len(cols), kmax), np.nan)
    fill = np.zeros(len(cols), int)
    for i, fi in zip(idx, points[:, 1]):
        F[i, fill[i]] = fi
        fill[i] += 1
    return cols, idx, F


def estimate_flux_calibration(points: np.ndarray, max_axes: int = 4) -> dict:
    """Physics prior: every flux-tunable circuit's spectrum is mirror-symmetric about Phi = 0 and Phi = 1/2.

    Symmetry score of the whole point cloud (Chamfer distance to its mirror image) as a function of the
    mirror position x0; its minima are the symmetry axes, the autocorrelation of the score curve gives the
    axis spacing (= period / 2), and the lowest line's behaviour at an axis says whether it is Phi = 0
    (frequency maximum) or Phi = 1/2 (minimum).  Only used to narrow the search bounds.
    """
    if len(points) < 12:
        return dict(axes=[], period=None)
    cols, idx, F = _column_matrix(points)
    n = len(cols)
    f_low = np.nanmin(F, axis=1)
    if n < 8 or np.nanstd(f_low) < 0.05:
        return dict(axes=[], period=None)
    dx = float(np.median(np.diff(cols)))
    span = float(cols[-1] - cols[0])
    cands = np.arange(cols[0] - span / 2, cols[-1] + span / 2 + 1e-9, dx / 2)
    res = np.full(len(cands), np.nan)
    fpts = points[:, 1]
    for k, x0 in enumerate(cands):
        j = np.round((2 * x0 - points[:, 0] - cols[0]) / dx).astype(int)
        ok = (j >= 0) & (j < n)
        if ok.sum() < max(8, 0.25 * len(points)):
            continue
        d = np.nanmin(np.abs(F[j[ok]] - fpts[ok, None]), axis=1)
        d = d[np.isfinite(d)]
        if d.size:
            res[k] = np.median(d)
    if np.isfinite(res).sum() < 5:
        return dict(axes=[], period=None)
    thresh = max(0.03, 0.3 * float(np.nanmedian(res)))
    r = np.where(np.isfinite(res), res, np.inf)
    axes = [(float(cands[k]), float(r[k])) for k in range(1, len(cands) - 1) if r[k] <= thresh and r[k] <= r[k - 1] and r[k] <= r[k + 1]]
    merged: list[tuple[float, float]] = []
    for x0, rr in sorted(axes):
        if merged and abs(x0 - merged[-1][0]) < 2 * dx:
            if rr < merged[-1][1]:
                merged[-1] = (x0, rr)
        else:
            merged.append((x0, rr))
    # period from the autocorrelation of the (negated) symmetry score: periodic with period/2
    period = None
    sig = np.where(np.isfinite(res), res, np.nanmedian(res))
    sig = -(sig - sig.mean())
    step = dx / 2
    max_lag = int(len(sig) * 0.6)
    ac = np.array([np.dot(sig[: len(sig) - L], sig[L:]) / max(len(sig) - L, 1) for L in range(max_lag)])
    if ac.size > 6 and ac[0] > 0:
        ac = ac / ac[0]
        min_lag = max(3, int(round(0.08 * span / step)))
        for L in range(min_lag, len(ac) - 1):
            if ac[L] > 0.25 and ac[L] >= ac[L - 1] and ac[L] >= ac[L + 1]:
                period = 2 * L * step
                break
    out_axes = []
    for x0, rr in merged:
        near = np.abs(cols - x0) < max(span / 12, 2 * dx)
        far = (np.abs(cols - x0) >= max(span / 12, 2 * dx)) & (np.abs(cols - x0) < span / 4)
        if np.isfinite(f_low[near]).sum() == 0 or np.isfinite(f_low[far]).sum() == 0:
            kind = "unknown"
        else:
            kind = "max" if np.nanmean(f_low[near]) > np.nanmean(f_low[far]) else "min"
        out_axes.append(dict(x=x0, type=kind, resid=rr))
    if period is None and len(out_axes) >= 2:
        sp = np.diff(np.sort([a["x"] for a in out_axes]))
        if sp.max() / max(sp.min(), 1e-9) < 1.35:
            period = 2 * float(np.median(sp))
    out_axes.sort(key=lambda a: a["resid"])
    return dict(axes=out_axes[:max_axes], period=period)
