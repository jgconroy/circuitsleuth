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
- labels its own confidence from the score gap between circuit families, so an ambiguous map says so,
- and explains all of it in plain English via Claude (structured output), falling back to a template without a key.

The interface is a dark, single-accent Streamlit app: interactive map with hover, live fit overlay with a loss sparkline, a schematic of the identified circuit, the Hamiltonian table with uncertainties, hypothesis scores, and a physics panel (potential well and levels; spin-1 levels versus field for NV centres; the charge anticrossing for quantum dots).

## How we built it
Python 3.12, NumPy/SciPy, scqubits (ground truth for the physics), PyTorch (proposer), Streamlit (UI), Claude Opus 5.5 (explanations). The build itself was agent-driven: a spec with pytest acceptance criteria, an implementation loop that only moves on when the tests are green, and an independent reviewer pass. On a published 2019 PRX fluxonium map digitised from the paper, with no tuning, it returns EJ 3.23 ± 0.20, EC 0.80 ± 0.05, EL 0.98 ± 0.04 GHz against the paper's 3.0 / 0.84 / 1.0. On hand-built hidden circuits it recovers every parameter, including resonator frequency and coupling and the flux period, to about 1 %.

## Beyond circuit QED
The method is "spectrum versus a control knob, explained by a small Hamiltonian", so other platforms are plug-in families. We added NV-centre ensembles in diamond (ODMR versus coil current: zero-field splitting, strain, field direction and the gauss-per-amp calibration all come out), semiconductor double-quantum-dot charge qubits (tunnel coupling and lever arm from spectroscopy versus detuning), trapped ions (axial and radial secular frequencies from a sideband spectrum versus endcap voltage) and neutral atoms (a Rydberg Stark map whose parabola vertex is the stray-field null every tweezer lab hunts for). Same extraction, same robust fit, same verification and "look here" logic.

## It decides what to measure next, optimally
"Where to look next" is not a heuristic: from every explained feature we build the Fisher information of the fit and rank candidate sweeps by expected information gain (bits, from the log-determinant ratio of the information matrix, discounted by visibility and cost). On the transmon demo it immediately finds the EJ–EC degeneracy (correlation −1) and asks for the two-photon sweep that breaks it. This is Bayesian experimental design on top of a physics fit, computed in a tenth of a second.

## It drives the experiment
Autopilot closes the loop on a digital twin: start from a partial sweep, identify, let the information-gain design pick the next block of control values, measure it, refit, repeat, and watch the parameter uncertainties fall round by round. The function that reads the twin is the one a lab replaces with an instrument driver. When two circuit families both fit, a discrimination design names the single control value where their predictions differ by hundreds of sigma.

## Error bars you can check
A coverage study on random hidden maps showed the raw Jacobian uncertainties were 1.5–2× too small, so the app scales them and says so, and a Metropolis sampler of the actual robust likelihood draws the posterior (the EJ–EC valley appears as a banana) in ten seconds.

## Built for the lab bench
Known-parameter constraints (fix EC from design, fr from a single-tone scan), derived quantities a physicist actually uses (anharmonicity, charge dispersion, sweet-spot bias values in your own control units, flux sensitivity, dispersive shift, Purcell factor), an operating-point solver ("which bias gives me 5 GHz?"), exports (CSV curves, HTML map, a scqubits snippet that rebuilds the fit), HDF5/CSV instrument-file import with cropping and unit handling, and a batch mode for fab lots. This is the part a Jupyter notebook never gets around to.

## Challenges
- Flux-independent spurs with more pixels than the qubit line hijacked the fit until we scored *features* (line segments) instead of pixels and split off flat lines before fitting flux-tunable circuits.
- EJ and EC are degenerate along constant f01; only the two-photon or 1-2 line breaks it. The fitter now reports that degeneracy as a large uncertainty and asks for exactly that line.
- A fluxonium with large EL can mimic a tunable transmon's 0-1 line. Precision (RMS residual) became part of the score, and the report shows both candidates.
- Real data: the published map has streaky columns and an overlaid theory fit. Per-column noise estimation and a colour-channel trick (data is blue, overlays are grey) made it usable.

## Accomplishments
- Recovered a real device's Hamiltonian from a figure in a paper, with uncertainties that bracket the authors' own fit.
- Every Hamiltonian in the code is checked against scqubits to a millionth of a gigahertz in the test suite, including the resonator-dressed spectrum.
- Resonator discovery: when the bare circuit cannot explain the bend in the data, the agent hypothesises a resonator at that frequency and finds fr and g to 1 %.
- Honest reporting: a blind test on ten random, deliberately nasty synthetic maps identifies the family in six and recovers all parameters in the clean tunable-circuit cases; the misses are listed in the README with why.

## What we learned
Robust statistics beat clever optimisers: once the loss stopped rewarding "explain the most pixels", everything else got easier. And the most useful output of a fitter is the list of things it could not fit.

## What's next
Close the loop with the instrument: the same agent that says "look here" can drive the VNA/AWG to take that sweep, then resonator punch-out and coherence measurements, i.e. an autonomous bring-up pipeline for new chips. More families (0-π, flux qubits, coupled transmons), and uploading a lab's existing fits as priors.

## Built with
python, numpy, scipy, scqubits, pytorch, streamlit, anthropic claude opus 5.5, matplotlib
