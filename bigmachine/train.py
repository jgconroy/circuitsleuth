"""Train the proposer on generated shards.  python train.py --data data --epochs 30 --threads 30 --width 2
Checkpoints every epoch to ckpt/last.pt (resume with --resume); writes classifier.pt + classifier.json."""
import argparse, glob, json, os, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circuitsleuth import classify  # noqa: E402

FAMS = classify.FAMILIES


def load_shards(data_dir, limit=None):
    files = sorted(glob.glob(os.path.join(data_dir, "shard_*.npz")))
    if limit:
        files = files[:limit]
    Xs, fams, crosses, npers, Ps = [], [], [], [], {f: [] for f in FAMS}
    for fn in files:
        d = np.load(fn)
        Xs.append(d["X"]); fams.append(d["fam"]); crosses.append(d["cross"]); npers.append(d["nper"])
        for f in FAMS:
            Ps[f].append(d[f"P_{f}"])
    X = np.concatenate(Xs); fam = np.concatenate(fams); cross = np.concatenate(crosses); nper = np.concatenate(npers)
    P = {f: np.concatenate(Ps[f]) for f in FAMS}
    return X, fam, cross, nper, P


def masked_mse(pred, target):
    m = ~torch.isnan(target)
    return F.mse_loss(pred[m], target[m]) if m.any() else pred.sum() * 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--width", type=int, default=2)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--limit-shards", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--out", default="classifier.pt")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    log = open("train_log.txt", "a")

    def say(s):
        line = f"[{time.strftime('%H:%M:%S')}] {s}"
        print(line, flush=True); log.write(line + "\n"); log.flush()

    t0 = time.time()
    X, fam, cross, nper, P = load_shards(a.data, a.limit_shards)
    n = len(X); n_val = max(1000, n // 20); n_tr = n - n_val
    say(f"loaded {n} maps from {a.data} in {time.time()-t0:.0f}s; train {n_tr} / val {n_val}; width {a.width}; threads {a.threads}")
    Xt = torch.from_numpy(X[:, None])
    T = dict(fam=torch.from_numpy(fam), cpl=torch.from_numpy(cross), nper=torch.from_numpy(nper), P={f: torch.from_numpy(P[f]) for f in FAMS})
    net = classify.Net(width=a.width)
    opt = torch.optim.AdamW(net.parameters(), lr=a.lr, weight_decay=1e-4)
    steps = a.epochs * (n_tr // a.batch + 1)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=a.lr, total_steps=steps)
    start_epoch = 0
    os.makedirs("ckpt", exist_ok=True)
    if a.resume and os.path.exists("ckpt/last.pt"):
        ck = torch.load("ckpt/last.pt", map_location="cpu", weights_only=False)
        net.load_state_dict(ck["state"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); start_epoch = ck["epoch"] + 1
        say(f"resumed from epoch {start_epoch}")
    rng = np.random.default_rng(0)
    idx_va = np.arange(n_tr, n)
    for ep in range(start_epoch, a.epochs):
        net.train(); perm = rng.permutation(n_tr); tot = 0.0; te = time.time()
        for b0 in range(0, n_tr, a.batch):
            bi = torch.from_numpy(perm[b0:b0 + a.batch]); xb = Xt[bi]
            if rng.random() < 0.5:
                xb = torch.flip(xb, dims=[3])
            if rng.random() < 0.3:
                xb = torch.roll(xb, shifts=int(rng.integers(0, 64)), dims=3)
            xb = xb * float(np.exp(rng.standard_normal() * 0.1))
            out = net(xb)
            loss = F.cross_entropy(out["fam"], T["fam"][bi]) + F.binary_cross_entropy_with_logits(out["coupled"], T["cpl"][bi]) + masked_mse(out["nper"], T["nper"][bi])
            for f in FAMS:
                loss = loss + 2.0 * masked_mse(out["params"][f], T["P"][f][bi])
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tot += loss.item() * len(bi)
        m = classify.evaluate(net, Xt[idx_va], fam[idx_va], cross[idx_va], nper[idx_va], {f: P[f][idx_va] for f in FAMS})
        say(f"epoch {ep+1}/{a.epochs} loss={tot/n_tr:.4f} val_fam_acc={m['fam_acc']:.3f} val_crossing_acc={m['coupled_acc']:.3f} param_rmse={m['param_rmse']:.3f} nper_rmse={m['nper_rmse']:.3f} ({time.time()-te:.0f}s/epoch, {(time.time()-t0)/60:.0f} min total)")
        torch.save(dict(state=net.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), epoch=ep), "ckpt/last.pt")
        torch.save(dict(state=net.state_dict(), families=FAMS, slots=classify.PARAM_SLOTS, metrics=m, n_train=n_tr, epochs=ep + 1, width=a.width), a.out)
        json.dump(m, open(os.path.splitext(a.out)[0] + ".json", "w"), indent=1)
    say("done")


if __name__ == "__main__":
    main()
