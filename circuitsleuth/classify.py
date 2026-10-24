"""Tiny CNN proposer: a 64x64 thumbnail of the map -> family probabilities, resonator flag and a coarse
parameter seed (normalised log-parameters) plus the number of flux periods in the window.

The proposer only *seeds* the physics fit (narrower bounds, good starting point); every number that is
reported comes from the fit, never from the network.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import models, synth

FAMILIES = synth.FAMILY_NAMES
PARAM_SLOTS = {fam: models.FAMILIES[fam]["params"] + models.RESONATOR_PARAMS for fam in FAMILIES}
N_PER_RANGE = (0.4, 3.0)  # periods in window, log-normalised
DEFAULT_PATH = Path(__file__).resolve().parent.parent / "artifacts" / "classifier.pt"


def _bounds(fam):
    b = dict(models.FAMILIES[fam]["bounds"])
    b.update(models.RESONATOR_BOUNDS)
    return b


def normalise(fam: str, params: dict) -> np.ndarray:
    b = _bounds(fam)
    out = np.full(len(PARAM_SLOTS[fam]), np.nan, np.float32)
    for i, n in enumerate(PARAM_SLOTS[fam]):
        if n not in params:
            continue
        lo, hi = b[n]
        if n == "d":
            out[i] = (params[n] - lo) / (hi - lo)
        else:
            out[i] = (np.log(params[n]) - np.log(lo)) / (np.log(hi) - np.log(lo))
    return out


def denormalise(fam: str, u: np.ndarray, coupled: bool) -> dict:
    b = _bounds(fam)
    out = {}
    for i, n in enumerate(PARAM_SLOTS[fam]):
        if n in models.RESONATOR_PARAMS and not coupled:
            continue
        lo, hi = b[n]
        v = float(np.clip(u[i], 0, 1))
        out[n] = lo + v * (hi - lo) if n == "d" else float(np.exp(np.log(lo) + v * (np.log(hi) - np.log(lo))))
    return out


class Net(nn.Module):
    def __init__(self, n_fam: int = 3, slots: dict | None = None):
        super().__init__()
        slots = slots or PARAM_SLOTS
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(64 * 16, 256), nn.ReLU(), nn.Dropout(0.1))
        self.fam = nn.Linear(256, n_fam)
        self.coupled = nn.Linear(256, 1)
        self.nper = nn.Linear(256, 1)
        self.param_heads = nn.ModuleDict({f: nn.Linear(256, len(slots[f])) for f in slots})

    def forward(self, x):
        h = self.fc(self.features(x))
        return dict(fam=self.fam(h), coupled=self.coupled(h).squeeze(-1), nper=self.nper(h).squeeze(-1),
                    params={f: torch.sigmoid(head(h)) for f, head in self.param_heads.items()})


def build_targets(truths: list[dict]):
    fam = np.array([FAMILIES.index(t["family"]) for t in truths], np.int64)
    cpl = np.array([int(t["coupled"]) for t in truths], np.float32)
    nper = np.array([(t["x_hi"] - t["x_lo"]) / t["period"] if models.FAMILIES[t["family"]]["flux_dependent"] else np.nan for t in truths], np.float32)
    nper = (np.log(nper) - np.log(N_PER_RANGE[0])) / (np.log(N_PER_RANGE[1]) - np.log(N_PER_RANGE[0]))
    P = {f: np.full((len(truths), len(PARAM_SLOTS[f])), np.nan, np.float32) for f in FAMILIES}
    for i, t in enumerate(truths):
        P[t["family"]][i] = normalise(t["family"], t["params"])
    return fam, cpl, nper, P


def _masked_mse(pred, target):
    m = ~torch.isnan(target)
    if m.sum() == 0:
        return pred.sum() * 0
    return F.mse_loss(pred[m], target[m])


def train(n_train: int = 20000, n_val: int = 2000, epochs: int = 12, batch: int = 128, seed: int = 0, out: Path = DEFAULT_PATH, log=print):
    torch.manual_seed(seed)
    t0 = time.time()
    X, _, _, truths = synth.make_dataset(n_train + n_val, seed=seed)
    log(f"generated {len(X)} maps in {time.time()-t0:.0f}s")
    fam, cpl, nper, P = build_targets(truths)
    Xt = torch.from_numpy(X[:, None])
    idx_tr, idx_va = np.arange(n_train), np.arange(n_train, n_train + n_val)
    net = Net()
    opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=3e-3, total_steps=epochs * (n_train // batch + 1))
    T = dict(fam=torch.from_numpy(fam), cpl=torch.from_numpy(cpl), nper=torch.from_numpy(nper), P={f: torch.from_numpy(P[f]) for f in FAMILIES})
    for ep in range(epochs):
        net.train()
        perm = np.random.default_rng(seed + ep).permutation(idx_tr)
        tot = 0.0
        for b0 in range(0, n_train, batch):
            bi = torch.from_numpy(perm[b0 : b0 + batch])
            xb = Xt[bi]
            # augmentation: random flux-axis flip and roll (period/offset invariance), small gain jitter
            if np.random.rand() < 0.5:
                xb = torch.flip(xb, dims=[3])
            xb = torch.roll(xb, shifts=int(np.random.randint(0, 64)), dims=3) if np.random.rand() < 0.3 else xb
            xb = xb * float(np.exp(np.random.randn() * 0.1))
            out_ = net(xb)
            loss = F.cross_entropy(out_["fam"], T["fam"][bi]) + F.binary_cross_entropy_with_logits(out_["coupled"], T["cpl"][bi]) + _masked_mse(out_["nper"], T["nper"][bi])
            for f in FAMILIES:
                loss = loss + 2.0 * _masked_mse(out_["params"][f], T["P"][f][bi])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item() * len(bi)
        metrics = evaluate(net, Xt[idx_va], fam[idx_va], cpl[idx_va], nper[idx_va], {f: P[f][idx_va] for f in FAMILIES})
        log(f"epoch {ep+1}/{epochs} loss={tot/n_train:.4f} val_fam_acc={metrics['fam_acc']:.3f} val_coupled_acc={metrics['coupled_acc']:.3f} param_rmse={metrics['param_rmse']:.3f} nper_rmse={metrics['nper_rmse']:.3f} ({time.time()-t0:.0f}s)")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state=net.state_dict(), families=FAMILIES, slots=PARAM_SLOTS, metrics=metrics, n_train=n_train, epochs=epochs), out)
    json.dump(metrics, open(out.with_suffix(".json"), "w"), indent=1)
    return net, metrics


@torch.no_grad()
def evaluate(net, X, fam, cpl, nper, P):
    net.eval()
    out = net(X)
    pf = out["fam"].argmax(1).numpy()
    pc = (out["coupled"] > 0).numpy().astype(int)
    m = ~np.isnan(nper)
    nper_rmse = float(np.sqrt(np.mean((out["nper"].numpy()[m] - nper[m]) ** 2))) if m.any() else float("nan")
    errs = []
    for i, f in enumerate(FAMILIES):
        sel = fam == i
        if sel.any():
            pred = out["params"][f].numpy()[sel]
            tgt = P[f][sel]
            mm = ~np.isnan(tgt)
            errs.append((pred[mm] - tgt[mm]) ** 2)
    conf = np.zeros((len(FAMILIES), len(FAMILIES)), int)
    for a, b in zip(fam, pf):
        conf[a, b] += 1
    return dict(fam_acc=float(np.mean(pf == fam)), coupled_acc=float(np.mean(pc == cpl.astype(int))), param_rmse=float(np.sqrt(np.mean(np.concatenate(errs)))), nper_rmse=nper_rmse, confusion=conf.tolist())


def load(path: Path = DEFAULT_PATH) -> Net | None:
    path = Path(path)
    if not path.exists():
        return None
    ck = torch.load(path, map_location="cpu", weights_only=False)
    net = Net(len(ck["families"]), ck["slots"])
    net.load_state_dict(ck["state"])
    net.eval()
    return net


@torch.no_grad()
def propose(net: Net, image: np.ndarray, x: np.ndarray, top_k: int = 3) -> list[dict]:
    """Ranked hypotheses with seeds. Each: family, coupled, prob, seed (params incl. period guess)."""
    thumb = synth.downsample(image, 64)
    out = net(torch.from_numpy(thumb)[None, None])
    pfam = torch.softmax(out["fam"][0], 0).numpy()
    pcpl = float(torch.sigmoid(out["coupled"][0]))
    nper = float(np.exp(np.log(N_PER_RANGE[0]) + float(out["nper"][0]) * (np.log(N_PER_RANGE[1]) - np.log(N_PER_RANGE[0]))))
    span = float(np.max(x) - np.min(x))
    props = []
    for i, fam in enumerate(FAMILIES):
        for coupled in (False, True):
            p = float(pfam[i]) * (pcpl if coupled else 1 - pcpl)
            seed = denormalise(fam, out["params"][fam][0].numpy(), coupled)
            if models.FAMILIES[fam]["flux_dependent"]:
                seed["period"] = span / np.clip(nper, *N_PER_RANGE)
                seed["offset"] = float(np.min(x))
            props.append(dict(family=fam, coupled=coupled, prob=p, seed=seed, p_coupled=pcpl, nper=nper))
    props.sort(key=lambda d: -d["prob"])
    return props[:top_k]
