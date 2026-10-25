# CircuitSleuth — big-machine package (proposer training + blind benchmark)

Self-contained: this folder has everything (the `circuitsleuth/` package is a copy; nothing else is imported).
No API keys, no scqubits, no Streamlit needed. CPU only.

## Set up once (Windows, Anaconda Prompt)

Make a FRESH conda env (an existing env named `sleuth` on lanthanum turned out to carry a NumPy < 1.17 and failed;
use `sleuth311`):
```
cd <this folder>
conda create -n sleuth311 python=3.11 -y
conda activate sleuth311
python -m pip install -r requirements.txt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```
Check: `python -c "import numpy, sys; print(sys.executable, numpy.__version__)"` must show the `envs\sleuth311` python and NumPy >= 1.26.
Use `conda activate sleuth311` before every command below (on both machines).

## Always benchmark first (about 2–3 minutes)

```
python run_train.py bench --n 200000 --epochs 30 --workers 30 --width 2
```
It measures generation speed (1 process and 30 processes), one training epoch per 1000 maps, one blind
identification, and prints the projected hours. On an 8-core M3 laptop: 40 ms/map single process, about 150 maps/s
with 6 workers, and ~1.0 s per epoch per 1000 maps at width 2 — so 200k maps / 30 epochs is roughly 25 min of
generation and 1.7 h of training on the laptop; expect the 32-core machine to be 3–5× faster.
Rule of thumb: on lanthanum use `--workers 30`; on QEC1 (20 cores, shared) use `--workers 12`.

## Train (detached, resumable)

```
python run_train.py start --n 200000 --epochs 30 --workers 30 --width 2
python run_train.py status            # shards done, last epochs
python run_train.py stop              # safe at any time; `start` again resumes
```
Outputs: `data/shard_*.npz` (generation, resumable), `ckpt/last.pt` (training, resumable), `classifier.pt` +
`classifier.json` (the result, updated after every epoch, so even a stopped run yields a usable model), `run_log.txt`
and `train_log.txt`.

Copy `classifier.pt` and `classifier.json` back into the repo's `artifacts/` folder (keep the old ones as a backup);
the app and the agent pick them up automatically. `classifier.json` holds the held-out accuracies, compare before
replacing. Bigger `--width` (3) and more maps help family accuracy; `--epochs 40` is sensible for 400k maps.

## Blind accuracy benchmark (optional, measures the whole pipeline, not just the proposer)

```
python run_blind.py --bench --platform circuit --workers 28         # times 2 maps, projects the total
python run_blind.py --n 200 --platform circuit --workers 28         # ~ (s per map) x 200 / 28
python run_blind.py --n 100 --platform nv --workers 28
```
Rows accumulate in `blind_<platform>.jsonl` (resumable); copy the file back for analysis. For the circuit platform
each map takes several minutes in one process, so 200 maps on 28 workers is about 1 hour.

## Notes
- All BLAS thread pools are pinned to 1 inside worker processes; do not set OMP_NUM_THREADS yourself.
- Memory: thumbnails are 16 KB each (200k maps = 3.3 GB in RAM during training, 2× while loading). Fine on lanthanum;
  on QEC1 keep `--n` at or below 150000.
- If a run dies, just run `start` again with the same arguments.

## Overnight, self-regulating training (use this when you have a fixed number of hours)

```
python overnight.py start --hours 6.5 --workers 30 --width 3      # lanthanum
python overnight.py start --hours 6.0 --workers 16 --width 2      # QEC1 (a second candidate with the current architecture)
python overnight.py status
python overnight.py stop
```
The worker measures the machine, spends ~40 % of the budget generating maps (resumable shards), trains for the epochs that fit
with a proper one-cycle schedule, keeps the best held-out model as `classifier_best.pt` + `classifier_best.json`, and stops itself
10 minutes before the deadline.  `overnight_log.txt` and `overnight_summary.json` tell the story.  In the morning copy
`classifier_best.pt` and `classifier_best.json` back as `artifacts/classifier.pt` / `classifier.json` if `fam_acc` beats the current
`artifacts/classifier.json`.
