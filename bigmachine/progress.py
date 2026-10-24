"""Show generation / training progress."""
import glob, json, os

here = os.path.dirname(os.path.abspath(__file__))
shards = len(glob.glob(os.path.join(here, "data", "shard_*.npz")))
man = os.path.join(here, "data", "manifest.json")
total = json.load(open(man))["n_shards"] if os.path.exists(man) else "?"
print(f"shards: {shards} / {total}")
log = os.path.join(here, "train_log.txt")
if os.path.exists(log):
    lines = [l for l in open(log).read().splitlines() if "epoch" in l or "done" in l]
    print("\n".join(lines[-3:]) if lines else "training: not started")
else:
    print("training: not started")
print("running:", os.path.exists(os.path.join(here, "run.pid")))
