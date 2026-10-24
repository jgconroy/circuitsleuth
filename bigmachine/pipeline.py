"""gen_data then train, in one detached process (used by run_train.py)."""
import os, subprocess, sys

sys.argv  # noqa
here = os.path.dirname(os.path.abspath(__file__))
n, epochs, workers, width = sys.argv[1:5]
py = sys.executable
subprocess.run([py, os.path.join(here, "gen_data.py"), "--n", n, "--workers", workers, "--out", "data"], check=True)
subprocess.run([py, os.path.join(here, "train.py"), "--data", "data", "--epochs", epochs, "--threads", workers, "--width", width, "--resume"], check=True)
