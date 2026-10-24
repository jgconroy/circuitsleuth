"""Readers and clean-up steps for uploaded maps (HDF5, 2-D grid CSV, NPZ) shared by the Streamlit importer.

Everything here is plain numpy: the widgets that drive these functions live in app.py.
"""
from __future__ import annotations

import io

import numpy as np

MAX_COLUMNS = 121  # thin the control axis so that one column is one control step
FREQ_TO_GHZ = {"GHz": 1.0, "MHz": 1e-3, "Hz": 1e-9}


# ------------------------------------------------------------------ HDF5
def h5_catalog(data: bytes) -> tuple[dict[str, tuple], dict[str, tuple]]:
    """Walk every group of an HDF5 file; return ({path: shape} of 2-D numeric datasets, same for 1-D)."""
    import h5py

    two, one = {}, {}

    def visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.dtype.kind in "iufc":
            if obj.ndim == 2:
                two[name] = tuple(obj.shape)
            elif obj.ndim == 1:
                one[name] = tuple(obj.shape)

    with h5py.File(io.BytesIO(data), "r") as h:
        h.visititems(visit)
    return two, one


def h5_read(data: bytes, path: str) -> np.ndarray:
    """One dataset as a float array (complex data -> magnitude)."""
    import h5py

    with h5py.File(io.BytesIO(data), "r") as h:
        arr = np.asarray(h[path][()])
    if np.iscomplexobj(arr):
        arr = np.abs(arr)
    return np.asarray(arr, float)


# ------------------------------------------------------------------ CSV grid
def _sniff_delimiter(text: str) -> str:
    head = "\n".join(text.splitlines()[:5])
    counts = {d: head.count(d) for d in (",", "\t", ";")}
    best = max(counts, key=counts.get)
    return best if counts[best] else " "


def read_grid_csv(data: bytes) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """2-D grid CSV: first row = control values, first column = frequencies, body = signal.
    The top-left cell may hold a label; it is ignored."""
    text = data.decode("utf-8", errors="replace")
    delim = _sniff_delimiter(text)
    arr = np.genfromtxt(io.StringIO(text), delimiter=None if delim == " " else delim, dtype=float, invalid_raise=False)
    arr = np.atleast_2d(arr)
    if arr.shape[0] < 3 or arr.shape[1] < 3:
        raise ValueError("a grid CSV needs at least two control columns and two frequency rows plus the axis row and column")
    x = arr[0, 1:]
    f = arr[1:, 0]
    image = arr[1:, 1:]
    if not (np.isfinite(x).all() and np.isfinite(f).all()):
        raise ValueError("the first row / first column must be numeric control values and frequencies")
    return np.nan_to_num(image), x, f


# ------------------------------------------------------------------ shared 2-D clean-up
def axes_match(image: np.ndarray, x, f) -> bool:
    return x is not None and f is not None and image.shape == (len(f), len(x))


def orient(image: np.ndarray, x: np.ndarray, f: np.ndarray):
    """Sort both axes ascending (the feature extractor assumes an increasing frequency axis)."""
    x, f = np.asarray(x, float), np.asarray(f, float)
    ox, of = np.argsort(x), np.argsort(f)
    return np.asarray(image, float)[np.ix_(of, ox)], x[ox], f[of]


def crop(image: np.ndarray, x: np.ndarray, f: np.ndarray, f_window=None, x_window=None, min_rows: int = 10, min_cols: int = 3):
    """Keep the rows / columns inside the windows; refuse (return the input) when too little would remain."""
    mf = np.ones(len(f), bool) if f_window is None else (f >= f_window[0]) & (f <= f_window[1])
    mx = np.ones(len(x), bool) if x_window is None else (x >= x_window[0]) & (x <= x_window[1])
    if mf.sum() < min_rows or mx.sum() < min_cols:
        return image, x, f, False
    return image[np.ix_(mf, mx)], x[mx], f[mf], True


def thin_columns(image: np.ndarray, x: np.ndarray, max_cols: int = MAX_COLUMNS):
    """At most `max_cols` evenly spaced control columns, as the screenshot path does."""
    if image.shape[1] <= max_cols:
        return image, x
    idx = np.linspace(0, image.shape[1] - 1, max_cols).round().astype(int)
    return image[:, idx], np.asarray(x, float)[idx]
