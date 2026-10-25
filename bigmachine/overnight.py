"""Self-regulating overnight training of the proposer CNN.

    python overnight.py start --hours 6.5 --workers 30 --width 3 [--max-maps 600000]
    python overnight.py status
    python overnight.py stop

The detached worker is aware of its deadline:
  1. measures generation and training speed on this machine (about 2 minutes),
  2. spends ~40 % of the budget generating maps (2000-map shards, resumable; an existing data/ folder is reused),
  3. picks the number of epochs that fits the remaining time and trains with a proper one-cycle schedule,
  4. after every epoch keeps the best model by held-out family accuracy as classifier_best.pt / classifier_best.json,
  5. kills the training 10 minutes before the deadline if it is still running; the best checkpoint stays.
In the morning copy classifier_best.pt and classifier_best.json back into the repo's artifacts/ as classifier.pt / classifier.json
if their held-out accuracy beats the current ones (artifacts/classifier.json).  overnight_log.txt tells the whole story."""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import argparse, glob, json, shutil, signal, subprocess, sys, time

here = os.path.dirname(os.path.abspath(__file__))
PID = os.path.join(here, "overnight.pid")
LOG = os.path.join(here, "overnight_log.txt")
SHARD = 2000


def log(msg: str) -> None:
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as fh:
        fh.write(line + "\n")


def _gen_some(k):
    from circuitsleuth import synth

    synth.make_dataset(25, seed=5000 + k)
    return 25


def measure(workers: int, width: int) -> tuple[float, float]:
    """(maps per second with `workers` processes, seconds per epoch per 1000 maps at this width)."""
    from multiprocessing import Pool

    sys.path.insert(0, here)
    from circuitsleuth import classify, synth

    t0 = time.time()
    with Pool(workers) as pool:
        tot = sum(pool.map(_gen_some, range(workers * 2)))
    rate = tot / max(time.time() - t0, 1e-6)
    import torch

    torch.set_num_threads(workers)
    X, _, _, truths = synth.make_dataset(3000, seed=2)
    for t in truths:
        t.setdefault("crossing_visible", False)
    t0 = time.time()
    classify.train(epochs=1, out=__import__("pathlib").Path(os.path.join(here, "bench_tmp.pt")), log=lambda s: None, width=width, data=(X, truths), batch=256)
    per_1000 = (time.time() - t0) / 3.0
    for f in ("bench_tmp.pt", "bench_tmp.json"):
        try:
            os.remove(os.path.join(here, f))
        except OSError:
            pass
    return rate, per_1000


def _kill(p: subprocess.Popen) -> None:
    try:
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGTERM)
    except Exception:  # noqa: BLE001
        try:
            p.kill()
        except Exception:  # noqa: BLE001
            pass


def worker(a) -> None:
    os.chdir(here)
    t_start = time.time()
    deadline = t_start + a.hours * 3600 - a.margin_min * 60
    log(f"overnight start: budget {a.hours:.2f} h (stop by {time.strftime('%H:%M', time.localtime(deadline))}), workers {a.workers}, width {a.width}, max maps {a.max_maps}")
    py = sys.executable
    rate, per_1000 = measure(a.workers, a.width)
    log(f"measured: generation {rate:.0f} maps/s with {a.workers} processes; training {per_1000:.2f} s per epoch per 1000 maps at width {a.width}")
    n_have = len(glob.glob(os.path.join(here, "data", "shard_*.npz"))) * SHARD
    gen_budget = a.gen_frac * (deadline - time.time())
    n_target = int(min(a.max_maps, n_have + rate * gen_budget) // SHARD * SHARD)
    n_target = max(n_target, min(a.max_maps, max(n_have, 4 * SHARD)))
    if n_target > n_have:
        log(f"generating {n_target} maps ({n_have} already on disk; budget {gen_budget/60:.0f} min)")
        r = subprocess.run([py, os.path.join(here, "gen_data.py"), "--n", str(n_target), "--workers", str(a.workers), "--out", "data"], stdout=open(os.path.join(here, "gen_log.txt"), "a"), stderr=subprocess.STDOUT)
        if r.returncode != 0:
            log("generation failed (see gen_log.txt); training on what exists")
    n_maps = len(glob.glob(os.path.join(here, "data", "shard_*.npz"))) * SHARD
    # never skip training: if measuring/generating ate the budget, still train for `min_train_min` past the deadline
    deadline = max(deadline, time.time() + a.min_train_min * 60)
    remaining = deadline - time.time()
    load_s = n_maps / 50000 * 60  # loading the shards into memory, roughly
    epoch_s = per_1000 * n_maps / 1000 + 5
    epochs = int(max(3, min(a.max_epochs, (remaining - load_s) // epoch_s)))
    log(f"training on {n_maps} maps: {remaining/60:.0f} min left, ~{epoch_s:.0f} s per epoch -> {epochs} epochs")
    plan = dict(n_maps=n_maps, epochs=epochs, width=a.width)
    plan_path = os.path.join(here, "plan.json")
    resume = os.path.exists(plan_path) and json.load(open(plan_path)) == plan and os.path.exists(os.path.join(here, "ckpt", "last.pt"))
    if not resume:
        shutil.rmtree(os.path.join(here, "ckpt"), ignore_errors=True)
    json.dump(plan, open(plan_path, "w"))
    kw = dict(cwd=here, stdout=open(os.path.join(here, "train_log.txt"), "a"), stderr=subprocess.STDOUT)
    if os.name != "nt":
        kw["start_new_session"] = True
    p = subprocess.Popen([py, os.path.join(here, "train.py"), "--data", "data", "--epochs", str(epochs), "--threads", str(a.workers), "--width", str(a.width), "--resume"], **kw)
    best = None
    seen_mtime = 0.0
    cls_pt, cls_json = os.path.join(here, "classifier.pt"), os.path.join(here, "classifier.json")
    while True:
        time.sleep(20)
        try:
            mt = os.path.getmtime(cls_json)
        except OSError:
            mt = 0.0
        if mt > seen_mtime:
            seen_mtime = mt
            try:
                m = json.load(open(cls_json))
                key = (round(float(m["fam_acc"]), 4), round(float(m.get("coupled_acc", 0.0)), 4))
                if best is None or key > best[0]:
                    shutil.copy(cls_pt, os.path.join(here, "classifier_best.pt"))
                    shutil.copy(cls_json, os.path.join(here, "classifier_best.json"))
                    best = (key, time.strftime("%H:%M"))
                    log(f"new best: fam_acc {key[0]:.4f} crossing_acc {key[1]:.4f} -> classifier_best.pt")
            except (OSError, ValueError, KeyError):
                pass
        if p.poll() is not None:
            log(f"training finished on its own (exit {p.returncode})")
            break
        if time.time() >= deadline:
            log("deadline reached: stopping the training; the best checkpoint is kept")
            _kill(p)
            break
    summary = dict(best_fam_acc=(best[0][0] if best else None), best_crossing_acc=(best[0][1] if best else None), best_at=(best[1] if best else None),
                   n_maps=n_maps, epochs_planned=epochs, width=a.width, hours_used=round((time.time() - t_start) / 3600, 2))
    json.dump(summary, open(os.path.join(here, "overnight_summary.json"), "w"), indent=1)
    log(f"summary: {summary}")
    try:
        os.remove(PID)
    except OSError:
        pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["start", "status", "stop"])
    ap.add_argument("--hours", type=float, default=6.0)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--width", type=int, default=3)
    ap.add_argument("--max-maps", type=int, default=600000)
    ap.add_argument("--max-epochs", type=int, default=60)
    ap.add_argument("--gen-frac", type=float, default=0.4)
    ap.add_argument("--margin-min", type=float, default=10.0)
    ap.add_argument("--min-train-min", type=float, default=20.0)
    ap.add_argument("--_worker", action="store_true")
    ap.add_argument("--foreground", action="store_true", help="run the worker in this terminal (for tests)")
    a = ap.parse_args()
    if a._worker or (a.cmd == "start" and a.foreground):
        worker(a)
        return
    py = sys.executable
    if a.cmd == "start":
        if os.path.exists(PID):
            print("already running (overnight.pid exists); `stop` first or `status`")
            return
        out = open(LOG, "a")
        kw = dict(stdout=out, stderr=subprocess.STDOUT, cwd=here)
        if os.name == "nt":
            kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        args = [py, os.path.join(here, "overnight.py"), "start", "--_worker", "--hours", str(a.hours), "--workers", str(a.workers), "--width", str(a.width),
                "--max-maps", str(a.max_maps), "--max-epochs", str(a.max_epochs), "--gen-frac", str(a.gen_frac), "--margin-min", str(a.margin_min),
                "--min-train-min", str(a.min_train_min)]
        p = subprocess.Popen(args, **kw)
        open(PID, "w").write(str(p.pid))
        print(f"started pid {p.pid}; it will stop itself by {time.strftime('%H:%M', time.localtime(time.time() + a.hours * 3600))}. Follow with `python overnight.py status`.")
    elif a.cmd == "status":
        shards = len(glob.glob(os.path.join(here, "data", "shard_*.npz")))
        print(f"shards on disk: {shards} ({shards * SHARD} maps); running: {os.path.exists(PID)}")
        if os.path.exists(LOG):
            print("\n".join(open(LOG).read().splitlines()[-6:]))
        tl = os.path.join(here, "train_log.txt")
        if os.path.exists(tl):
            lines = [l for l in open(tl).read().splitlines() if "epoch" in l]
            if lines:
                print(lines[-1])
    elif a.cmd == "stop":
        if not os.path.exists(PID):
            print("not running")
            return
        pid = int(open(PID).read())
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"])
            else:
                os.killpg(pid, signal.SIGTERM)
        finally:
            try:
                os.remove(PID)
            except OSError:
                pass
        print("stopped (classifier_best.pt is kept); `start` again resumes if the plan is unchanged")


if __name__ == "__main__":
    main()
