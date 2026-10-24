"""Run a built-in demo circuit headlessly and save its report for figures.  python scripts/run_demo.py A"""
import json, sys, time
import numpy as np
from circuitsleuth import agent
sys.path.insert(0, ".")


def main():
    import app  # noqa: E402  (pulls in the DEMOS table; Streamlit calls are no-ops outside a server)
    key = [k for k in app.DEMOS if k.startswith(sys.argv[1])][0]
    truth, r = app.make_demo(key)
    fast = "--fast" in sys.argv
    cfg = agent.AgentConfig(maxiter=30, n_starts=1, popsize=12, top_k=2, coupled_trials=1, workers=4) if fast else agent.AgentConfig(maxiter=60, workers=4)
    t0 = time.time()
    rep = agent.identify(r["image"], r["x"], r["f"], config=cfg)
    print(f"{time.time()-t0:.0f}s best:", rep["best"]["hypothesis"], rep["best"]["params_pretty"], "score", rep["best"]["track_score"], "holdout", rep["best"].get("holdout", {}).get("inlier_frac"))
    print("truth:", truth["params"], truth.get("period"), truth.get("offset"))
    np.savez(f"artifacts/demo_{sys.argv[1]}.npz", image=r["image"], x=r["x"], f=r["f"])
    json.dump(rep, open(f"artifacts/demo_{sys.argv[1]}_report.json", "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
