"""Digitise Fig. 2 of Nguyen et al., PRX 9, 041041 (2019) [arXiv:1810.11006]: two-tone spectroscopy of
fluxonium device A (paper fit: EJ = 3, EC = 0.84, EL = 1 GHz).  Produces data/demo/nguyen2019_devA.npz.

Steps: render the vector PDF, locate the axes frame, crop, mask the inset, convert colour to a
'blue signal' channel (data is dark blue; theory overlays are grey/black and are suppressed)."""
import numpy as np
import pymupdf

SRC = "data/raw/nguyen2019_fig2.pdf"
OUT = "data/demo/nguyen2019_devA.npz"
X_RANGE = (0.0, 0.55)   # phi_ext / 2pi
F_RANGE = (0.0, 5.0)    # GHz

doc = pymupdf.open(SRC)
pix = doc[0].get_pixmap(dpi=220)
rgb = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, pix.n)[:, :, :3].astype(float)
H, W, _ = rgb.shape
dark = rgb.mean(axis=2) < 80
# axes frame: the longest dark rows/cols
row_frac = dark.mean(axis=1)
col_frac = dark.mean(axis=0)
rows = np.flatnonzero(row_frac > 0.5)
cols = np.flatnonzero(col_frac > 0.5)
top, bottom = rows.min(), rows.max()
left, right = cols.min(), cols.max()
print("frame px:", top, bottom, left, right, "of", H, W)
plot = rgb[top + 2 : bottom - 1, left + 2 : right - 1]
R, G, B = plot[..., 0], plot[..., 1], plot[..., 2]
signal = (B - R)  # dark blue data -> large; light background -> small; grey/black overlays -> ~0
signal = np.clip(signal, 0, None)
# flip so that row 0 is the lowest frequency
signal = signal[::-1]
h, w = signal.shape
f = np.linspace(F_RANGE[0], F_RANGE[1], h)
x = np.linspace(X_RANGE[0], X_RANGE[1], w)
# mask the inset (lower-left box) and the white region (no data) lower-right
inset = (x[None, :] > 0.015) & (x[None, :] < 0.30) & (f[:, None] > 0.05) & (f[:, None] < 2.65)
signal[inset] = 0.0
# the inset frame itself is black -> already ~0 in B-R; mask a margin too
# thin columns to ~1 per flux step of the original measurement (~110 steps)
ncol = 110
idx = np.linspace(0, w - 1, ncol).round().astype(int)
img = np.stack([signal[:, max(i - 2, 0) : i + 3].max(axis=1) for i in idx], axis=1)
xs = x[idx]
# thin rows to ~400
nrow = 400
ridx = np.linspace(0, h - 1, nrow).round().astype(int)
img = np.stack([img[max(i - 2, 0) : i + 3].max(axis=0) for i in ridx], axis=0)
fs = f[ridx]
np.savez(OUT, image=img, x=xs, f=fs, source="Nguyen et al. PRX 9, 041041 (2019), Fig. 2, device A; arXiv:1810.11006", truth_EJ=3.0, truth_EC=0.84, truth_EL=1.0)
print("saved", OUT, img.shape, "signal range", img.min(), img.max())
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
fig, ax = plt.subplots(figsize=(6, 5), dpi=100)
ax.imshow(img, extent=[xs[0], xs[-1], fs[0], fs[-1]], origin="lower", aspect="auto", cmap="magma")
ax.set_xlabel("phi_ext/2pi"); ax.set_ylabel("GHz")
fig.savefig("artifacts/nguyen2019_preview.png", bbox_inches="tight")
