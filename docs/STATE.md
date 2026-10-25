# Ĥolmes (formerly CircuitSleuth) — project state and results log

Purpose of this file: a fresh Claude session (or a human) must be able to continue from here WITHOUT
re-running anything listed as done. Update it whenever a result, decision or long run finishes.
Companion files: `README.md` (public story), `docs/devpost.md` (submission text), `docs/pitch.md` (3-min script),
`PRD.md` (original spec). Memory note for Claude: `~/.claude/projects/-Users-grantconroy/memory/calhacks-2026-hackathon.md`.

## 0. Hard facts
- **Clock warning:** the section labels "Sun ~00:45" … "Sun ~04:10" below were written in UTC. In PDT they are Sat ~17:45 … ~21:10.
  Check `date` before budgeting time; the deadline is **Sun 9:00 AM PDT = 16:00 UTC**.
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
- Anthropic API key is in `.env` (git-ignored); credits were added Sun ~02:30 and the Claude Opus 5.5 structured-output explainer works
  (~12 s per report, prompt ~6.5k chars; max_tokens 8000 with effort low because thinking shares the budget). Template fallback remains. GitHub CLI is logged in (`gh auth status` ok).
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
- With the paper's flux axis (period held at 1): **EJ 3.03+-0.06, EC 0.86+-0.02, EL 0.96+-0.02** with the final extractor (faint peaks kept only
  next to a confident neighbour); earlier settings gave 3.23/0.80/0.98 and 2.83/0.93/0.93 (marginal data; all within 10 % of the paper); explained 58 % of coherent
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
- Circuits (10 maps): family 6/10 before and **7/10 after** the generator validity fix (same seeds), all params <5 % in 2/10; clean tunable circuits <=1.2 %;
  fixed transmons family-right but EJ/EC degenerate when the 2-photon line is absent; misses = tunable+resonator <-> fluxonium
  confusion (2), one faint fluxonium. Runtime 5-9 min/map when the laptop is shared. A rerun with the validity fix was
  started Sat ~12:30 (log in the session scratchpad; writes `artifacts/blind_results.json`) - first 3 rows identical to before.
- NV (8 maps): family **8/8**, params within tolerance **6/8** (misses: E off by ~3 MHz, field_scale 12 %). ~4 s/map.
- DQD (8 maps): family **8/8**, params within tolerance **6/8** after the generalised relabel step (Sun ~00:50; was 5/8; the two misses have 10–25 usable features). ~4 s/map.
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

### Sat evening: judge critiques and fixes (results)
- Two fresh-eyes critiques (design lead, SC-qubit PI) rated the UI 4/10 and credibility 6/10. Their specs are being implemented
  (verdict band hero, hypothesis race, evidence map with legend outside + numbered look-here badges, residual strip, held-out overlay,
  avoided-crossing zoom, pull plot, spec-sheet cards, IBM Plex typography, hide toast/log/internals). New module `ui/evidence.py`.
- Live-run bug found and fixed: Plotly charts redrawn inside `st.empty()` need a UNIQUE key per redraw (`key=f"map_live_{n}"`), otherwise
  StreamlitDuplicateElementId/Key is raised inside the progress callback. `agent._emit` now swallows callback exceptions so a UI error can
  never abort a fit. AppTest does NOT catch this class of bug: verify live runs with `scripts/screenshot.py ... --click "Identify circuit"`
  against a server started on a spare port (the user's own server on 8501 does not reload library modules; restart it after backend edits).
- Extractor: faint peaks (3–4σ) are now kept when an adjacent column has a peak within ~60 MHz (continuity). Demo A 0-1 detection went from
  ~50 to 64 of 78 rendered columns. Root cause of the "look here where data exists" complaint was partly real dropouts in the synthetic map
  and partly faint-edge misses; hand-picked demos now render with at most one dropout segment (`ui/demos.py`).
- Demo C spur changed to a single weak 7.1 GHz tone (the old 6.2 GHz tone with sidebands mimicked a transmon ladder and the fast preset
  locked onto it; a "resonator" then explained the real line). Fixed transmon maps are inherently ambiguous against spur families.
- Physics: offset now reported modulo period near the data centre; χ computed by exact diagonalisation at Φ = 0 and Φ = ½ with a
  "NOT dispersive" flag when g/Δ > 0.2; first-order 1/f flux-noise Tφ at the steepest point for 1 and 5 µΦ₀; flat lines the qubit crosses
  without splitting are labelled "uncoupled mode / pump tone"; report carries `holdout_columns`; summary text uses the same numbers as tiles.
- `?demo=C&load=1` URL parameters select a demo and auto-load its saved result (pitch links, screenshots).

### Sat night: redesign landed
- New UI (app.py + ui/theme.py, plots.py, schematic.py, evidence.py): IBM Plex, verdict band hero, hypothesis race, stepper, evidence section
  (map with legend below, residual strip, held-out overlay, avoided-crossing zoom with 2g bracket), physics panel with Φ slider and transition
  arrows, spec-sheet cards, numbered look-here cards, truth strip + pull plot, "Details & export" strip. Verified: apptest_quick, NV/DQD AppTest,
  and REAL browser runs on port 8502 (`scripts/screenshot.py ... --click "Identify circuit"`): demo C live in ~80 s, demo A loaded.
- Resonator acceptance now additionally requires a gain >= 0.02 on FLUX-DEPENDENT features (a flat extra line is not evidence). Demo C live:
  resonator correctly rejected; answer transmon EJ 14.2±0.3 / EC 0.300±0.007 (truth 15 / 0.28; fixed transmons stay EJ–EC degenerate).
- `agent.prune_look_here` drops suggestion boxes covering >20 % of the map and the thermal 1-2 line unless tiny; applied in the app too.
- Demo A regenerated: EJmax 25.01±0.10, EC 0.2998, d 0.300, fr 7.002, g 0.0996, period 1.300, offset 0.200 (offset now reported mod period).
- README screenshot `artifacts/ui_results.png` is the new loaded-A page.
- Resonator scenario (tests/test_agent.py -k resonator_and_spur) re-passed under the flux-dependent-evidence rule (12 min): EJmax 25.3, EC 0.296, fr 7.003, g 0.099.
- Demo B regenerated: EJ 5.998, EC 1.000, EL 0.600, offset −0.300. A full `tests/test_agent.py` run with the FINAL code has not completed yet (one with the new extractor but the old acceptance rule was running Sat ~22:00); rerun it when the laptop is idle and record the result here.

### Sat late: two more platforms
- `trapped_ion` (platform "ion"): sideband detunings in MHz vs endcap voltage; ω_z = volt_scale·√(V − x0), ω_r² = f_rad0² − ω_z²/2; lines carrier,
  ±ω_z, ±ω_r, ±2ω_z. Hand-picked recovery exact (f_rad0 3.2009, volt_scale 1.7995, x0 −1.2996), seconds.
- `rydberg_atom` (platform "atom"): two |mj| components with quadratic Stark shifts vs plate voltage; E = field_scale·(V − x0). α and field_scale are
  degenerate (only α·field_scale² is observable): the derived table reports the Stark curvature per unit² and says to fix field_scale under
  Known parameters to get α. Recovery: curvatures within 1 %, dfs 11.95 (12), x0 0.1207 (0.12).
- Frequency units are per platform (`PLATFORMS[...]["freq_unit"/"freq_label"]`, MHz for ion/atom); all numeric code is unit-agnostic.
- URL params now: `?platform=ion&demo=J`, `?demo=A&load=1`.
- Blind (8 maps each): ion family 8/8, params **7/8** after the generalised relabel step (Sun ~00:50; was 5/8: volt_scale halved); atom family 8/8, curvatures/dfs/x0 within tolerance 5/8. ~4 s/map.
- Unit bug fixed: RMS-to-MHz conversion now uses `models.to_MHz(family)` (1 for MHz platforms); the ion/atom hero had shown 1000× values.

### Sat ~midnight status
- Five platforms live in the app (circuits, NV, quantum dots, trapped ions, Rydberg atoms); all headless checks pass; ion/atom runs verified
  in a real browser (6–7 s). Per-parameter units everywhere (`models.param_unit`). Big-machine package rebuilt with the final code.
- A full `tests/test_agent.py` run on the FINAL code was started Sat ~23:30 (`scratchpad/test_agent_final.log`); the previous run (new extractor,
  old acceptance rule) had `test_agent_identifies_fixed_transmon` failing on EC (0.13 relative vs 0.02 allowed: EJ–EC degeneracy on a fixed transmon
  with a faint two-photon line). If it fails again, either relax that test's EC tolerance to 0.15 (physically honest: EC is weakly determined
  there) or make the continuity rule keep 3σ peaks next to 3.5σ neighbours. Do not block the submission on it.

### Sun ~00:45: experimental design
- `circuitsleuth/design.py`: Fisher information F0 from explained points (finite-difference Jacobian of line frequencies w.r.t. theta), expected
  information gain per candidate sweep = ½ log det(F0+F_S)/det(F0) in bits × line visibility, greedy non-overlapping top-5, per-column gain map,
  Laplace correlation matrix. Attached to every report as `report["design"]`; UI section "Most informative next sweep" (`ui/design_panel.py`).
  Demo A: EJmax–EC correlation −0.997, top sweep = 0-2/2ph near the crossing (3.3 bits). Test in tests/test_lab_features.py.

### Sun ~01:00: Lanthanum classifier adopted
- 190k maps / 30 epochs / width 2, crossing_visible target: fam_acc 0.936, crossing acc 0.874, param rmse 0.216 (was 0.888 / 0.587 / 0.230). Now `artifacts/classifier.pt`.

### Sun ~02:00: autopilot + discrimination
- `circuitsleuth/loop.py` run_loop(): digital twin = all extracted points of the full map; start with the first 40 % of columns; each round
  identify(points, x=full grid) → design candidates on the full grid → add the best block's columns → refit. `scripts/run_loop.py A --fast`
  precomputes `artifacts/loop_A.json` for the circuit demo (replay in the app); NV/dot/ion/atom run live in seconds. tests/test_loop.py (NV) passes.
- `design.discriminate()`: single control value separating the best and runner-up families (σ units), preferring in-window/in-band; appended
  to next_measurements as kind "discriminate" and folded into the confidence text.
- Autopilot precompute for demo A (`artifacts/loop_A.json`, 237 s): rounds 0→3 add 23 features; σ(EJmax) 3.4 %→0.7 %, σ(EC) 3.5 %→0.8 %, σ(period) 0.73 %→0.03 %; family stable.
- Posterior sampling + coverage script are being built by a subagent (fit.sample_posterior, design_panel.posterior_figure, scripts/coverage.py).

### Sun ~03:15: posterior + coverage + calibration (DONE)
- `fit.sample_posterior` (random-walk Metropolis on the soft-L1 robust likelihood, 31-column subsample, GN recentre, 10 % burn-in; helpers
  `result_from_report`, `laplace_covariance`, `gauss_newton_recentre`), `design_panel.posterior_figure/posterior_pair_choice`,
  `scripts/coverage.py`, `tests/test_posterior.py` (2 pass, 6 s). Demo A: acceptance 0.49, posterior sd / Laplace sd ≈ 1.0–1.14.
- Coverage (40 maps each): 1σ/2σ fractions nv 0.42/0.60, dot 0.34/0.53, ion 0.62/0.72 → raw σ ~1.5–2× too small + wrong-optimum tails.
  Decision: `AgentConfig.sigma_calibration = 1.8` multiplies reported σ (noted in README and in the app's Posterior check section).
- App: "Posterior check" section with a sampling button (state in `st.session_state["posterior"]`); verified in AppTest on loaded demo A:
  acceptance 0.49, 900 kept samples, ~20 s under load. `sample_posterior` returns dicts (sd, laplace_sd, map, mode keyed by parameter).
- Big machines: the user's `sleuth` env was broken (old NumPy); a side session created `sleuth311` (python 3.11 + requirements) and the
  bench runs there. bigmachine/README.md updated accordingly. Lanthanum is running the final-code blind tables (5 platforms) in
  `circuitsleuth_bigmachine_v2`; QEC1 was at 109/200 on the OLD-code circuit table.
- Claude explainer verified live (credits added Sun ~02:30).

### Test status at Sun ~04:10
- All quick suites pass (models, synth/extract, platforms, lab_features incl. design, loop, posterior): 32 tests, 18 s, re-run Sun ~04:15
  after the relabel change. `tests/test_fit.py` 7/7 pass (6 min, Sun ~04:20) incl. the new relabel test.
- `tests/test_agent.py`: **4/4 pass** on the final code. `test_agent_identifies_fixed_transmon` re-passed Sun ~04:10 after the relabel fix
  (5.5 min; transmon EJ 15.023 / EC 0.280, truth 15 / 0.28; transmon+resonator second at 0.810). The other three passed Sun ~03:00 on
  code that differs only by the relabel step, which runs for control-independent circuit hypotheses only.

### Sun ~04:00: fixed-transmon label ambiguity fixed (relabel alternatives)
- Diagnosis of `test_agent_identifies_fixed_transmon`: NOT the EJ–EC valley. The map has two flat lines (0-1 at 5.50, two-photon 0-2/2
  at 5.34). Reading the second as the thermal 1-2 line (f01 − EC) instead of 0-2/2 (f01 − EC/2) fits the same points with EC/2
  (EC pinned at its 0.15 bound, EJ 26.7); the robust loss cannot tell the two readings apart, so the search landed on either by seed.
- Fix: `fit.relabel_alternatives` (called from `fit_hypothesis` for control-independent circuit hypotheses only): for each
  (used weaker line, unused brighter line) pair, solve for the parameters that put the brighter line on that track with the 0-1 line
  held, polish locally, attach as `FitResult.alternatives`. `agent.run()` scores alternatives with `_score` (its visibility-weighted
  missing-line penalty prefers the reading without a bright empty line) and emits a `relabel` progress event. Unused = ≤ max(2, 10 %)
  points under an in-band line (two stray noise points must not block it).
- Unit test `tests/test_fit.py::test_fixed_transmon_relabel_alternative_breaks_two_photon_vs_12_ambiguity` (4 s): seed 0 now offers
  EC 0.2796 as the alternative; seed 1 finds it directly. Physics prior encoded: at usual two-tone powers the 2-photon line is more
  common than the thermal 1-2 line, so an isolated second flat line is read as 0-2/2.

### Sat ~22:45 PDT: rebrand to Ĥolmes, Title Case, KaTeX math, sidebar fix (DONE, verified in the browser)
- Brand kit from the user (`~/Downloads/holmes-brand.zip`) -> `docs/BRAND.md` + `static/brand/` (favicons, icons, marks, og-image). Name is
  **Ĥolmes** (U+0124) in UI text, `holmes` in file names; the Python package / repo stay `circuitsleuth`. Header lockup = inlined
  `holmes-mark-reverse.svg` + Playfair Display 700 wordmark (Google Fonts @import, has the Ĥ glyph) + tagline; tab icon = icon-192.png via
  `page_icon`. Streamlit cannot add OG / manifest head tags, so those parts of the kit are not applied.
- Title Case everywhere a title/label/bullet-like item appears: `theme.title_case()` (acronym map NV/CNN/LO/SQUID…, lone symbols such as
  g or d untouched, hyphen and en-dash parts capitalised, tokens with digits/Greek left alone). Applied inside `section()`, `card_html`
  kickers, `spec_rows_html` labels, race names, hero title and chart titles (not formula titles containing "="); widget labels rewritten
  literally in app.py; demo names in ui/demos.py renamed (keys still `name.split(" ")[0]`); radio option checks updated ("Demo Circuit",
  "Upload Points", "Upload 2-D Grid").
- Math: the verdict band is now a `st.container(border=True, key="hero")` with two columns so the Hamiltonian is real KaTeX
  (`theme.hamiltonian_tex`, fitted numbers in amber via \textcolor); parameter names typeset in tables (`theme.param_html`), KaTeX in the
  sidebar known-parameter list and design-value inputs (`theme.param_tex`), `theme.mathify()` turns EJ/f01/… tokens in spec labels, card
  text and section subtitles into sub-scripted HTML; Plotly titles/axes use <sub> (potential plots, correlation matrix, σ-vs-round legend).
- Sidebar bug (user report: could not get the sidebar back after collapsing it): the CSS hid `[data-testid="stToolbar"]`, which is where
  Streamlit 1.66 puts the "expand sidebar" button. Now only the toolbar actions / main menu are hidden; verified the » button is 28×28 px.
- Verified: `scripts/apptest_quick.py` (no exceptions), headless autopilot replay + posterior sampling (no exceptions), Playwright
  screenshots of landing, loaded demo A (full page) and a live NV run. `scripts/screenshot.py` waits for "Identify Circuit" now.

### Sun ~00:40 PDT: final-stretch additions (in progress; see "Pending")
- **Bayesian model comparison** (`fit.mixture_loglik`, `fit.log_evidence`, `fit.model_posterior`): Laplace evidence per hypothesis under a
  Gaussian + uniform-background likelihood (σ = c_local, ε = 0.1), Gauss-Newton Hessian with inlier responsibilities, σ capped at half the
  prior box, log-uniform priors on energies.  Computed in `agent.identify` (step 6) on ONE common point set (dynamic features when ≥ 20 %,
  as the fits use) → `report["model_posterior"]` and per-hypothesis `log_evidence` / `posterior`.  Decision stays score-based.  Saved demo
  reports backfilled with `scripts/backfill_evidence.py` (A: P(tunable+res) = 1.0, log10 BF vs fluxonium 306; B: bare fluxonium 0.993 vs
  +resonator 0.007 = Occam at work).  UI: P column + header in the race, Bayes factor in the verdict eyebrow, caption with family posteriors.
  Tests: `tests/test_evidence.py` (3, 4 s).
- **Relabel alternatives generalised to every platform** (`fit.relabel_alternatives`): for each (used weaker line, unused brighter line) pair,
  a plain least-squares solve moves the points of the weaker line onto the brighter one with the family's primary line as anchor, then a local
  polish; the agent scores alternatives with `_score`.  Flat predicted lines (ion carrier) are skipped for control-dependent hypotheses.
  Fixes the ×2 failures: ions reading first-order axial sidebands as second-order (volt_scale/2), dots reading the 0-1 hyperbola as its
  two-photon copy (t_c, lever arm ×2).  Root cause of the ion cases is upstream: at noise ≈ 0.08 the 4σ extractor misses most faint sidebands,
  so the loss barely separates the two readings; the scorer's missing-line penalty does.
- **REST API + Docker + CI** (subagent): `api.py` (FastAPI: /health, /demos, /identify, /identify/demo), `tests/test_api.py` (5 tests),
  `requirements.txt` (new), `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml` (quick suites + API tests), README "Run it" + badge.
- **Ask Ĥolmes** (DONE, verified live): `circuitsleuth/ask.py` = 9 pure tools over the report (summary, derived, control_for_frequency with
  sensitivities, predict_transitions, next_measurements + design, hypotheses with acceptance rules, unexplained, uncertainties + correlation,
  scqubits_code) + `answer()` running the Anthropic beta tool runner (claude-opus-5-5, adaptive thinking, effort low, prompt caching,
  never raises).  `ui/ask_panel.py` chat section (3 suggested questions, tools-called caption), wired into app.py after the Posterior
  check (section G3).  Tests `tests/test_ask.py` (8, no API).  Live answers cite tool numbers only (~6 s, < 2 ¢ each).
- Old-code QEC table (200 maps, `artifacts/qec_oldcode_blind_circuit.jsonl`): family 70 %, params 37 %; fluxonium 34/65 with most misses
  read as a FIXED transmon.  Checked the suspected cause on the current code (flat-feature mask swallowing a heavy fluxonium's plateau):
  on 60 synthetic fluxonium maps the mask removes a median 3 % of the circuit's own points, > 30 % in 4/60, and the skip rule
  ("< 20 % dynamic features") wrongly trips in 3/60 (tunable: 2/60).  So that mechanism explains ~5–10 % of maps, not 46 %; wait for
  the Lanthanum new-code table before chasing it further (`scripts/flat_stats2.py` in the session scratchpad).
- Big machines tonight: Lanthanum circuit table (normal preset, n = 120, 32 min/map/core, started Sat 23:20, ETA ~01:30) → copy back as
  `artifacts/lanthanum_blind_circuit.jsonl`; QEC1 finishing the OLD-code 200-map table, then new env + atom/nv/ion/dot tables (n = 200 each,
  minutes); optional Lanthanum retrain `run_train.py start --n 400000 --epochs 40 --workers 30 --width 3` after 01:30 (adopt only if better).

## 5. Pending / in progress (check before starting new work)
- (done Sat ~19:30) Circuit blind rerun with the validity-fixed generator: family **7/10**, params <5 % 2/10 (unchanged causes: fixed-transmon
  EJ/EC degeneracy without a 2-photon line; tunable+resonator mistaken for fluxonium in 2 maps; one faint fluxonium -> transmon+res).
  Mean 2120 s/map only because the laptop was shared with training; `artifacts/blind_results.json`. README updated.
- (done Sat ~18:00) UI lab features landed and verified headlessly on circuit/NV/DQD (`scripts/apptest_quick.py` + platform runs):
  known-parameter constraints (toggle inside Search), derived table, operating-point tool, exports (JSON/CSV/HTML/scqubits),
  HDF5 + 2-D grid CSV import (`ui/importers.py`), design comparison (toggle inside Explain), session history. AppTest cannot
  drive file uploads; those paths were unit-tested by the subagent only.
- Big-machine: training DONE and adopted (fam 0.936). Blind tables for 5 platforms running on lanthanum (`blind_*.jsonl` to be copied into
  `artifacts/`; summarise with a small script into README "Results"). QEC1 old-code circuit table may also come back (baseline only).
- User to-dos: API credits (optional), record video per `docs/pitch.md`, Devpost submission before Sun 9 AM PDT.

## 6. How to verify quickly
```
. .venv/bin/activate
python -m pytest tests/test_models.py tests/test_synth_extract.py tests/test_platforms.py tests/test_lab_features.py -q   # 5 s
python scripts/apptest_quick.py        # headless Streamlit check: default run + precomputed load
streamlit run app.py                   # demos A/B/E: use "Load precomputed result"; C live (fast); NV/DQD live (seconds)
```

## 7. Ideas not done (ranked)
0. A fresh screenshot of the lab-feature UI for README/Devpost (current image predates them).
1. Retrain proposer with crossing_visible on 200k+ maps (big machines) and regress fr/g only when crossing visible.
2. Instrument drivers ("look here" -> take the sweep): out of hackathon scope; mention as future work.
3. Add 0-pi qubit / flux qubit / coupled-transmon families; trapped-ion sideband spectroscopy.
4. Ridge-following extractor for very noisy real maps.
5. GPU/JAX forward models for speed (not needed for the demo).
