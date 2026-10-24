import argparse, sys, time
from pathlib import Path
import torch
from circuitsleuth import classify

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=20000)
ap.add_argument("--val", type=int, default=2000)
ap.add_argument("--epochs", type=int, default=12)
ap.add_argument("--threads", type=int, default=4)
ap.add_argument("--out", default=str(classify.DEFAULT_PATH))
a = ap.parse_args()
torch.set_num_threads(a.threads)
log = lambda s: print(f"[{time.strftime('%H:%M:%S')}] {s}", flush=True)
classify.train(n_train=a.n, n_val=a.val, epochs=a.epochs, out=Path(a.out), log=log)
log("done")
