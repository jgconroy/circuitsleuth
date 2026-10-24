# CircuitSleuth — spec

**One line:** Upload a two-tone spectroscopy map (flux-axis vs frequency); CircuitSleuth identifies the
superconducting circuit, recovers its Hamiltonian parameters and flux calibration, overlays the fitted
transitions live, lists features it cannot explain, and tells you where to measure next.

Pitch: "Shazam for quantum chips."  Judges: Most Technically Advanced (overall), optional Fetch.ai track.
Deadline: Sun Oct 25 2026, 9:00 AM PDT (Devpost: repo + image + video).

## Non-negotiables
1. Every forward model is numerically checked against scqubits in the test suite (tolerance 1e-6 GHz).
2. The fitter is robust: spurious flux-independent lines (pumps/LO/mixer spurs), missing segments and
   noise must not break identification. Unexplained features are reported, never forced into the fit.
3. The flux axis is NOT assumed calibrated: period and offset are fitted (x in volts -> Phi/Phi0).
4. Verification is built in: hold-out flux columns are predicted, not fitted; report hold-out residual.
5. Works with no API key (deterministic proposer + template explanation). Claude layer is additive.
6. One command (`python demo.py`) runs a blind test set and prints a recovery table.

## Modules (package `circuitsleuth/`)
- `models.py`   fast numpy forward models: transmon, tunable_transmon, fluxonium, each ±readout resonator.
                `transitions(family, params, phi) -> {label: freq_array}` (one- and two-photon lines).
- `synth.py`    synthetic datasets: sample family+params from priors, flux calibration, visibility
                dropouts, linewidth, noise, spurious lines, sidebands -> image + point cloud + truth.
- `extract.py`  2D map -> peak points (x, f, weight) via per-column peak finding.
- `fit.py`      robust (Geman-McClure) global (differential evolution) + local (least_squares) fit for a
                hypothesis; returns params, uncertainties, inlier mask, residuals, hold-out score.
- `classify.py` tiny CNN (torch) on 64x64 maps -> family probabilities + coarse log-params seed.
- `agent.py`    orchestration loop: extract -> propose (CNN top-k) -> fit -> verify -> select ->
                unexplained features -> next-measurement suggestions; yields progress events.
- `explain.py`  Claude explanation + hypothesis revision when score is poor; template fallback.
- `app.py`      Streamlit UI with live overlay animation and "where to look next" boxes.
- `demo.py`     blind test table.

## Tasks (check off as done; each has a pytest)
- [ ] T1 models: transmon / tunable_transmon / fluxonium energies match scqubits (tests/test_models.py)
- [ ] T2 models: resonator-coupled dressed transitions, label by overlap; avoided crossing sanity test
- [ ] T3 synth: generator produces image + points + truth; deterministic under seed; shapes/ranges tests
- [ ] T4 extract: on clean synthetic maps, >=90% of truth points recovered within 1 linewidth
- [ ] T5 fit: recovers params within 3% on 5 seeds per family (no spurs), <2 min each on laptop
- [ ] T6 fit: still recovers with 2 spurious lines + 30% dropout; outliers flagged as unexplained
- [ ] T7 agent: end-to-end on 3 hidden circuits -> correct family, report JSON schema validated
- [ ] T8 classify: train on >=20k synthetic maps; >=90% family accuracy on held-out set; saved weights
- [ ] T9 next-measurement: predicted-but-unobserved lines -> (f_lo, f_hi, x_lo, x_hi) boxes
- [ ] T10 explain: Claude call behind a flag; template fallback has identical JSON shape
- [ ] T11 app: Streamlit upload/demo-pick, live overlay, report panel, download JSON
- [ ] T12 demo.py blind table + README with results and screenshots
- [ ] T13 real-data check: digitized published spectroscopy figure recovered within paper's values
- [ ] T14 reviewer pass (independent agent) on correctness of physics conventions + robustness
