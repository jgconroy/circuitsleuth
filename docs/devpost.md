# CircuitSleuth — Devpost submission draft

**Tagline:** Shazam for quantum chips: point it at a spectroscopy map and it tells you what circuit you built, how well it knows, and what to measure next.

## Inspiration
Every superconducting-qubit lab starts a new chip the same way: a two-tone spectroscopy sweep, then hours of staring at it, guessing the circuit, hand-tuning EJ/EC/EL in a script until the theory lines land on the data, arguing about which features are pump leakage, and deciding what to sweep next. It is the most repeated, least automated step in the field, and the step where a wrong guess silently poisons every calibration after it.

## What it does
Upload a map (image, NPZ, or a CSV of features). CircuitSleuth
- finds the spectral features and separates flux-dependent lines from flux-independent spurs,
- proposes circuit hypotheses (fixed transmon, tunable transmon, fluxonium, each with or without a readout resonator) using a small CNN trained on synthetic maps,
- fits each hypothesis with a robust physics model (batched NumPy Hamiltonians checked against scqubits to 1e-6 GHz), including the flux-axis calibration, with the fit overlaid live on the data,
- verifies on flux columns it never fitted,
- revises itself: when a bare circuit leaves an unexplained cluster, it seeds a resonator hypothesis where the data bends,
- reports parameters ± uncertainty, every hypothesis it tried and why it lost, every feature it could not explain (with a physical interpretation), and concrete "look here" boxes: frequency and flux ranges where the next measurement should go, plus which transition would pin down any poorly determined parameter,
- and explains all of it in plain English via Claude (structured output), falling back to a template without a key.

## How we built it
Python 3.12, NumPy/SciPy, scqubits (ground truth for the physics), PyTorch (proposer), Streamlit (UI), Claude Opus 5.5 (explanations). The build itself was agent-driven: a spec with pytest acceptance criteria, an implementation loop that only moves on when the tests are green, and an independent reviewer pass. RESULTS_NUMBERS

## Challenges
- Flux-independent spurs with more pixels than the qubit line hijacked the fit until we scored *features* (line segments) instead of pixels and split off flat lines before fitting flux-tunable circuits.
- EJ and EC are degenerate along constant f01; only the two-photon or 1-2 line breaks it. The fitter now reports that degeneracy as a large uncertainty and asks for exactly that line.
- A fluxonium with large EL can mimic a tunable transmon's 0-1 line. Precision (RMS residual) became part of the score, and the report shows both candidates.
- Real data: the published map has streaky columns and an overlaid theory fit. Per-column noise estimation and a colour-channel trick (data is blue, overlays are grey) made it usable.

## Accomplishments
RESULTS_ACCOMPLISHMENTS

## What we learned
Robust statistics beat clever optimisers: once the loss stopped rewarding "explain the most pixels", everything else got easier. And the most useful output of a fitter is the list of things it could not fit.

## What's next
Close the loop with the instrument: the same agent that says "look here" can drive the VNA/AWG to take that sweep, then resonator punch-out and coherence measurements, i.e. an autonomous bring-up pipeline for new chips. More families (0-π, flux qubits, coupled transmons), and uploading a lab's existing fits as priors.

## Built with
python, numpy, scipy, scqubits, pytorch, streamlit, anthropic claude opus 5.5, matplotlib
