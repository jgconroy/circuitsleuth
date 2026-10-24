"""Detached, resumable launcher.
    python run_train.py bench  [--n N --epochs E --workers W --width 2]   measure first (takes ~2 min)
    python run_train.py start  --n 200000 --epochs 30 --workers 30 --width 2
    python run_train.py status
    python run_train.py stop
Re-running `start` after a stop resumes: finished shards are kept and training continues from ckpt/last.pt."""
import argparse, os, signal, subprocess, sys, time

here = os.path.dirname(os.path.abspath(__file__))
PID = os.path.join(here, "run.pid")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["bench", "start", "status", "stop"])
    ap.add_argument("--n", type=int, default=200000)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 2))
    ap.add_argument("--width", type=int, default=2)
    a = ap.parse_args()
    py = sys.executable
    if a.cmd == "bench":
        subprocess.run([py, os.path.join(here, "bench.py"), "--n", str(a.n), "--epochs", str(a.epochs), "--workers", str(a.workers), "--width", str(a.width)])
    elif a.cmd == "start":
        if os.path.exists(PID):
            print("already running (run.pid exists); use `stop` first or `status`"); return
        out = open(os.path.join(here, "run_log.txt"), "a")
        kw = dict(stdout=out, stderr=subprocess.STDOUT, cwd=here)
        if os.name == "nt":
            kw["creationflags"] = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kw["start_new_session"] = True
        p = subprocess.Popen([py, os.path.join(here, "pipeline.py"), str(a.n), str(a.epochs), str(a.workers), str(a.width)], **kw)
        open(PID, "w").write(str(p.pid))
        print(f"started pid {p.pid}; follow with `python progress.py`; log: run_log.txt")
    elif a.cmd == "status":
        subprocess.run([py, os.path.join(here, "progress.py")])
    elif a.cmd == "stop":
        if not os.path.exists(PID):
            print("not running"); return
        pid = int(open(PID).read())
        try:
            if os.name == "nt":
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"])
            else:
                os.killpg(pid, signal.SIGTERM)
        finally:
            os.remove(PID)
        print("stopped; `start` again to resume")


if __name__ == "__main__":
    main()
