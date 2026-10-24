# CircuitSleuth — project state and results log

Purpose of this file: a fresh Claude session (or a human) must be able to continue from here WITHOUT
re-running anything listed as done. Update it whenever a result, decision or long run finishes.
Companion files: `README.md` (public story), `docs/devpost.md` (submission text), `docs/pitch.md` (3-min script),
`PRD.md` (original spec). Memory note for Claude: `~/.claude/projects/-Users-grantconroy/memory/calhacks-2026-hackathon.md`.

## 0. Hard facts
- Hackathon: Cal Hacks 13.0, Palace of Fine Arts SF, Oct 23–25 2026. **Devpost hard deadline Sun Oct 25, 9:00 AM PDT**
  (repo + image + video; 3-min pitch + 2-min Q&A; judging: Application, Intentionality, Functionality, Creativity,
  Technical Complexity). Rules: all code written during the event (true: repo started Fri Oct 23 evening); no posted ban on AI agents.
- Repo: `~/calhacks13/circuitsleuth`, GitHub `jgconroy/circuitsleuth` (main). Python 3.12 venv at `.venv`
  (system python3 is 3.14 and lacks wheels; always use `.venv`). `pip install -e .` done.
- Machine: Apple M3, 8 cores, 8 GB RAM. Use `workers=4` for fits. Never run two heavy jobs + tests at once (everything slows 3x).
- Other machines (user copy-pastes a folder; no remote access from Claude): lanthanum = 32 cores, Windows 10, `py -3.14`;
  QEC1 = 20 cores (shared), Windows 11, `(dtc)` env py3.11. Package for them: `dist/circuitsleuth_bigmachine(.zip)`
  built by `python scripts/make_bigmachine_pkg.py` from `bigmachine/`. The user said these machines are NOT usable for the
  final build, only for side computations (training / benchmarks).
- Anthropic API key is in `.env` (git-ignored) but the account had **no credits** as of Sat Oct 24 ~13:00; `explain.py` falls back
  to a template automatically. GitHub CLI is logged in (`gh auth status` ok).
- The user prefers short writeups; deliver deep work with brief messages.

## 1. What the product does (one paragraph)
Upload a two-tone spectroscopy map (control axis x frequency). Extract features (per-column robust peaks, sub-pixel),
link them into tracks, split off flux-independent lines (spurs). Propose hypotheses (families x optional readout resonator;
CNN proposer for circuits). Fit each with batched NumPy Hamiltonians (validated vs scqubits 1e-6) using a Geman-McClure
nearest-line loss, annealed differential evolution + staged 0-1-first search + multi-start + soft-L1 polish; the control
axis calibration (period/offset, or gauss-per-unit, or lever arm) is fitted. Score by explained track weight with
missing-line and precision penalties, verify on held-out columns, try resonator models for the top-2 families (accepted
only with a margin and significant g), report params +- sigma, confidence (family score gap), unexplained features,
"look here" suggestions, derived quantities, operating points, scqubits snippet. Platforms: superconducting circuits
(transmon, tunable transmon, fluxonium), NV-centre ensemble ODMR, double-quantum-dot charge qubit.

## 2. File map
```
circuitsleuth/models.py    FAMILIES/PLATFORMS registry, Hamiltonians, transitions(), NV + DQD models, platform_scales()
circuitsleuth/synth.py     synthetic maps (validity re-draw: primary line in band >=30%), crossing_visible(), make_dataset()
circuitsleuth/extract.py   extract_points (height>=4 sigma per-column noise, smooth_cols), link_tracks (slope-aware),
                           flat_lines/flat_feature_mask, estimate_flux_calibration (symmetry axes), track_weights
circuitsleuth/fit.py       Hypothesis, Objective (line_filter), _fit_once (3 anneal stages, EJ-EC valley sweep, 2-stage polish),
                           fit_hypothesis (staged + multi-start), holdout_check, track_score
circuitsleuth/classify.py  CNN proposer (width param); target = crossing_visible (not raw coupled); propose()
circuitsleuth/agent.py     identify(): orchestration, scoring, resonator trials, confidence, next_measurements, report
circuitsleuth/derived.py   derived_quantities, control_for_frequency, scqubits_code, curves_csv
circuitsleuth/explain.py   Claude Opus 5.5 structured-output summary; template fallback
app.py + ui/               Streamlit UI (theme, plots (Plotly), schematic (SVG), platform_panels, demos registry)
demo.py                    blind test (random hidden maps) per platform;  batch.py  folder -> CSV
scripts/                   prep_nguyen2019.py (real data), run_real.py, run_demo.py, make_figs.py, train_classifier.py,
                           bench_fit.py, make_bigmachine_pkg.py
bigmachine/                gen_data.py, train.py, bench.py, run_train.py, progress.py, run_blind.py, pipeline.py, README.md
tests/                     test_models (scqubits), test_synth_extract, test_fit, test_agent (4 end-to-end), test_platforms, test_lab_features
artifacts/                 classifier.pt(+json) [v2], demo_A/B reports, nguyen2019_report_calibrated.json, fig_*.png, blind_*.json
data/demo/nguyen2019_devA.npz   digitised Fig. 2 of Nguyen et al. PRX 9, 041041 (2019); data/raw has the PDF
```

## 3. Results already obtained (do not repeat)
### Tests
- `pytest tests/test_models.py tests/test_synth_extract.py tests/test_platforms.py tests/test_lab_features.py` -> all pass (~5 s).
- `tests/test_fit.py` (6 tests, ~5 min) pass; `tests/test_agent.py` (4 end-to-end, **~28 min**) -> **4/4 pass** as of Sat ~13:00
  (after the expansion changes). Only rerun test_agent if agent/fit/extract/scoring logic changes.
### Real data (Nguyen 2019 fluxonium device A, paper fit EJ=3, EC=0.84, EL=1 GHz)
- With the paper's flux axis (period held at 1): **EJ 3.23+-0.20, EC 0.80+-0.05, EL 0.98+-0.04**; explained 58 % of coherent
  segments, hold-out 78 %, RMS 16 MHz. Without the axis prior the period is unconstrained (right half of the map has no
  extractable features; `smooth_cols` does not recover it). Report: `artifacts/nguyen2019_report_calibrated.json`, figure `artifacts/fig_real.png`.
### Hand-picked demos (precomputed with the "normal" preset, 4 workers)
- A tunable transmon + 7 GHz resonator + 5.1 GHz spur: EJmax 25.23+-0.17 (25), EC 0.2969 (0.3), d 0.3002 (0.3), fr 7.001 (7),
  g 0.0985 (0.1), period 1.300 (1.3). 426 s. `artifacts/demo_A_report.json`.
- B fluxonium: EJ 5.992 (6), EC 1.002 (1), EL 0.6001 (0.6), period 1.1, hold-out 1.0. 683 s. `artifacts/demo_B_report.json`.
- C fixed transmon + LO spur: family right, EC within 1 % (live, ~1-2 min fast preset).
- NV demo F: D 2.86996 (2.870), E 0.0041 (0.004), theta/phi symmetry-equivalent, field_scale 119.9 (120), x0 0.1007 (0.1), seconds.
- DQD demo H: t_c 2.50006, lever 8.99996, x0 0.15001, seconds.
### Blind tests (random hidden maps, harsh generator: spurs, dropouts, noise <=10 %)
- Circuits (10 maps, before the generator validity fix): family 6/10, all params <5 % in 2/10; clean tunable circuits <=1.2 %;
  fixed transmons family-right but EJ/EC degenerate when the 2-photon line is absent; misses = tunable+resonator <-> fluxonium
  confusion (2), one faint fluxonium. Runtime 5-9 min/map when the laptop is shared. A rerun with the validity fix was
  started Sat ~12:30 (log in the session scratchpad; writes `artifacts/blind_results.json`) - first 3 rows identical to before.
- NV (8 maps): family **8/8**, params within tolerance **6/8** (misses: E off by ~3 MHz, field_scale 12 %). ~4 s/map.
- DQD (8 maps): family **8/8**, params within tolerance **5/8** (misses are maps with <20 usable features / spur-dominated). ~5 s/map.
### Proposer (CNN)
- v1: 20k maps, 12 epochs: fam acc 0.857, "coupled" acc 0.552 (unlearnable target), param rmse 0.241 (`artifacts/classifier_v1.*`).
- v2 (adopted, `artifacts/classifier.pt`): 40k maps, 20 epochs: fam acc **0.888**, coupled acc 0.587, param rmse 0.230.
  Both call demo A (tunable+resonator) "fluxonium" with high prob -> prior weight reduced to 0.05.
- Since v2 the training target changed to **crossing_visible** (resonator only learnable when the 0-1 line crosses it; the generator
  now enriches crossings to ~16 % of maps). **Not yet retrained with this target** - that is what the big-machine run is for.
  Smoke run (2k maps, 1 epoch) gave crossing acc 0.85 immediately (vs 0.55 for the old target).
### Timings (M3, 4 workers)
- Fast preset ~2 min/map (may miss a resonator), normal 5-7 min, thorough longer. NV/DQD seconds. Generation 40-55 ms/map
  per process; training width 2 ~ 1-2.5 s per epoch per 1000 maps on 4 threads.

## 4. Decisions and pitfalls (why things are the way they are)
- Peak extraction uses **height above the per-column baseline**, not scipy prominence (prominence let 4-sigma noise through).
  Per-column noise = max(column MAD, global MAD) (never lower than global, else quiet columns explode).
- Scoring counts **tracks** (line segments >=3 points), flat tracks at half weight, plus a visibility-weighted
  **missing-line penalty** (0.3) and an RMS precision penalty; spurs are removed before fitting flux-dependent circuit
  hypotheses; flux-dependent circuit hypotheses are skipped when <20 % of features vary (fixed transmon data).
- Track linker is **slope-aware** (steep lines were fragmenting into <3-point tracks, so only the spur counted).
- EJ-EC degeneracy along constant f01 (transmons): a valley sweep at coarse scale + 2-stage polish; the 2-photon line pins EC;
  stage B of the staged search keeps EJ/EC bounds open for transmons.
- Flux calibration prior from mirror symmetry (period = 2x axis spacing); only for circuits; the wide search always runs too.
- Fluxonium global search is luck-dependent -> multi-start (2) + popsize 20 + 1.5x iterations; "staged" 0-1-first search.
- Resonator model accepted only if score gain > 0.05 **and** g > 2 sigma_g; tried for the top-2 bare families; fr seeded from the
  histogram of unexplained points (an avoided crossing leaves a cluster), then proposer, then flat lines, then 7 GHz.
  Coupled refinement bounds 0.5-2x around the bare solution (0.8-1.25x was too tight).
- NV/DQD need platform-specific scales (NV lines are MHz apart): `models.platform_scales()`; NV field direction compared
  up to tetrahedral symmetry (`models.nv_projections`).
- Don't launch multiprocessing fits from a `python -` stdin script (spawn re-imports stdin); use files + a main guard.
- Streamlit: `use_container_width` is deprecated -> `width="stretch"`; AppTest works headlessly (see section 6).
- `kill`/`pkill` may be blocked by the sandbox classifier; background jobs via `nohup ... &` plus `until grep` waiters work.
- Shell heredocs: never nest an `EOF` heredoc inside another `EOF` heredoc (this file was first written that way and the shell choked).

## 5. Pending / in progress (check before starting new work)
- Circuit blind rerun (background, started Sat ~12:30): results -> `artifacts/blind_results.json`; update README Results if different.
- UI lab-features pass by a subagent (known-parameter constraints, derived table, operating-point tool, exports, HDF5/2-D CSV
  import with transpose/unit/crop, design comparison, session history). Verify with the AppTest snippet in section 6 after it lands,
  then commit.
- Big-machine run: user copies `dist/circuitsleuth_bigmachine` to lanthanum, runs `python run_train.py bench ...` then `start`,
  copies `classifier.pt/json` back into `artifacts/` (compare `fam_acc` with 0.888 first). Optional `run_blind.py` for a 200-map table.
- User to-dos: API credits (optional), record video per `docs/pitch.md`, Devpost submission before Sun 9 AM PDT.

## 6. How to verify quickly
```
. .venv/bin/activate
python -m pytest tests/test_models.py tests/test_synth_extract.py tests/test_platforms.py tests/test_lab_features.py -q   # 5 s
python scripts/apptest_quick.py        # headless Streamlit check: default run + precomputed load
streamlit run app.py                   # demos A/B/E: use "Load precomputed result"; C live (fast); NV/DQD live (seconds)
```

## 7. Ideas not done (ranked)
1. Retrain proposer with crossing_visible on 200k+ maps (big machines) and regress fr/g only when crossing visible.
2. Instrument drivers ("look here" -> take the sweep): out of hackathon scope; mention as future work.
3. Add 0-pi qubit / flux qubit / coupled-transmon families; trapped-ion sideband spectroscopy.
4. Ridge-following extractor for very noisy real maps.
5. GPU/JAX forward models for speed (not needed for the demo).
