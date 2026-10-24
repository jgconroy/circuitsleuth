# CircuitSleuth

**Shazam for quantum chips.** Give it a two-tone spectroscopy map of a superconducting qubit and it tells you
*what circuit you built*, recovers the Hamiltonian parameters and the flux calibration, overlays the fitted
transitions live, lists every feature it **cannot** explain, and tells you **where to measure next**.

Built in 36 hours at Cal Hacks 13.0 (Oct 2026).

![real data](artifacts/fig_real.png)

*Above: a published fluxonium spectroscopy map (Nguyen et al., PRX 9, 041041, 2019, device A), digitised
from the paper's vector PDF. CircuitSleuth was told nothing except "this is a map between 0 and 5 GHz".*

## Why

Every superconducting-qubit lab does this by hand: stare at a spectroscopy map, guess the circuit, open
scqubits or a home-made script, tweak EJ/EC/EL until the lines land on the data, argue about which
features are pump leakage, then decide what to sweep next. It takes hours to days per device and it is the
first thing you do with every new chip. CircuitSleuth automates the whole loop, and keeps the parts a
physicist needs: honest uncertainties, explicit unexplained features, and a verification step that never
sees the data it is tested on.

## What it does

1. **Feature extraction.** Robust per-column peak finding (noise estimated per flux column, so streaky
   real data does not flood the point cloud), sub-pixel refinement, linking of points into line
   segments ("tracks"), and detection of flux-independent lines (pump / LO / mixer spurs).
2. **Hypotheses.** Three circuit families (fixed transmon, flux-tunable transmon, fluxonium), each with or
   without a readout resonator. A tiny CNN trained on synthetic maps *proposes* families and seeds; it
   never reports a number.
3. **Physics fit.** Fast batched NumPy Hamiltonians (checked against scqubits to 1e-6 GHz in the test
   suite). The loss is Geman–McClure on the distance of every observed feature to the nearest predicted
   transition line (0-1, 0-2, 0-3, two-photon 0-2/2, thermal 1-2, dressed by the resonator when present),
   so spurious lines cost a bounded amount instead of bending the fit. Global search = differential
   evolution with an annealed robust scale; local polish = soft-L1 least squares; uncertainties from the
   Jacobian. The flux axis is **not** assumed calibrated: period and offset (in volts / mA) are fitted,
   seeded by a symmetry analysis (every flux-tunable spectrum is mirror-symmetric about Φ = 0 and Φ = ½).
4. **Verification.** Every candidate is scored by the fraction of *features* (not pixels) it explains, then
   re-fitted on 70 % of the flux columns and scored on the untouched 30 %.
5. **Revision.** If the best bare circuit leaves a cluster of unexplained flux-dependent points, the agent
   seeds a resonator hypothesis at that frequency (that is what an avoided crossing looks like) and tries
   again.
6. **Report.** Parameters ± σ, hold-out score, every hypothesis tried, unexplained features with a
   physical interpretation, and "where to look next": predicted lines with no data under them, lines just
   outside the band, and which transition would pin down a poorly determined parameter. Claude (Opus 5.5,
   structured output) turns the JSON into a plain-English summary; a template does the same without a key.

## Run it

```bash
python3.12 -m venv .venv && . .venv/bin/activate
pip install -e .
pytest -q                       # physics vs scqubits, extraction, fitting, end-to-end agent
streamlit run app.py            # UI with demo circuits, CSV / NPZ / screenshot upload
python demo.py --n 12           # blind test on random hidden circuits
python scripts/run_real.py data/demo/nguyen2019_devA.npz   # the published fluxonium map
```

Optional: `ANTHROPIC_API_KEY=...` in `.env` enables the Claude explanation layer;
`python scripts/train_classifier.py` (re)trains the CNN proposer on synthetic maps.

## Results

See `artifacts/` after running `demo.py` and `scripts/run_real.py`. (Filled in below before submission.)

RESULTS_PLACEHOLDER

## Honest limitations

- Three circuit families. Adding one means writing its Hamiltonian in `models.py` and a scqubits test.
- No signal, no answer: the fitter needs visible lines. Given partial lines it reports where to look.
- Very noisy maps with few features can be genuinely ambiguous between a tunable transmon and a
  high-EL fluxonium; the report shows both scores instead of hiding it.
- Trained proposer and all tests use synthetic data; the only real map is the digitised figure above.

## Layout

```
circuitsleuth/models.py    Hamiltonians, transitions (validated against scqubits)
circuitsleuth/synth.py     synthetic maps with realistic nuisances; CNN training data
circuitsleuth/extract.py   peaks, tracks, flat lines, symmetry-based flux calibration
circuitsleuth/fit.py       robust global+local fit, uncertainties, hold-out check
circuitsleuth/classify.py  CNN proposer (torch)
circuitsleuth/agent.py     orchestration: propose -> fit -> verify -> revise -> report
circuitsleuth/explain.py   Claude explanation (structured output) with template fallback
app.py                     Streamlit UI;  demo.py  blind test;  tests/  pytest suite
```

Figure 2 of Nguyen et al. is reproduced under the arXiv non-exclusive license for demonstration.
