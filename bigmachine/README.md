# CircuitSleuth — big-machine package (proposer training + blind benchmark)

Self-contained: this folder has everything (the `circuitsleuth/` package is a copy; nothing else is imported).
No API keys, no scqubits, no Streamlit needed. CPU only.

## Set up once (Windows)

lanthanum (Python 3.14):
```
cd <this folder>
py -3.14 -m venv venv
venv\Scripts\pip install --upgrade pip
venv\Scripts\pip install numpy scipy
venv\Scripts\pip install torch --index-url https://download.pytorch.org/whl/cpu
```
If `torch` has no wheel for 3.14 yet, make the venv with `py -3.12` (or 3.11) instead; everything else is identical.

QEC1 (the `(dtc)` env, Python 3.11): activate it, then `pip install torch --index-url https://download.pytorch.org/whl/cpu` (numpy/scipy are already there).

Use `venv\Scripts\python` (or the activated env's `python`) for every command below.

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
