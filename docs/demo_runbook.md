# Demo runbook (screen share, 3 minutes + Q&A)

## Pre-flight (10 minutes before)

1. Terminal: `cd ~/calhacks13/circuitsleuth && source .venv/bin/activate && streamlit run app.py` (a fresh restart picks up every library change).
2. Browser at 90 % zoom, a window with only these tabs, in this order:
   - Tab 1  `http://localhost:8501/?demo=C`            (live run, fixed transmon with LO leakage; fast preset; ~80 s)
   - Tab 2  `http://localhost:8501/?demo=A&load=1`     (precomputed: tunable transmon + resonator)
   - Tab 3  `http://localhost:8501/?platform=nv`       (live NV run; seconds)
   - Tab 4  `http://localhost:8501/?demo=E&load=1`     (real data, PRX 2019 fluxonium)
3. In Tab 2 click **Replay Saved Autopilot** once so the autopilot result is already on the page.
4. Collapse the sidebar (» at the top-left) in Tabs 2 and 4: the page then uses the full width.
5. Check `.env` has credits: the summary paragraph under the verdict should be two sentences, not the template.
6. Press **Identify Circuit** in Tab 1 about 20 seconds before you start talking, so the race is running when you switch to it.

## One-minute script

Start on Tab 2 (demo A, precomputed, sidebar collapsed, scrolled to the verdict band). Tab 3 and Tab 4 ready behind it.

**0:00 Verdict (Tab 2).** "A raw spectroscopy map goes in. This comes out: the circuit, its Hamiltonian with the fitted numbers in place, the
held-out error, and a Bayes factor against the next family. Every number is from a physics fit checked against scqubits."

**0:12 Evidence (scroll to the annotated map and the crossing zoom).** "It found the readout resonator itself, from the bend the bare circuit
left at 7 GHz; here is the splitting it measured. The one line it cannot explain, it calls a pump tone, because the qubit crosses it
without splitting."

**0:27 Next Measurements and Autopilot (scroll on; drag the Round slider).** "It ranks the next sweep by information gain in bits, and the
autopilot runs that loop on a digital twin: 40 % of the sweep, choose, measure, refit; watch the error bars fall."

**0:42 Not just circuits (Tab 3, press Identify Circuit).** "Same engine on an NV-centre map: zero-field splitting, strain, field direction, in
seconds. Dots, trapped ions and Rydberg atoms work the same way."

**0:52 Real data and close (Tab 4, Diagnostics → Truth and Design → Reveal Hidden Truth).** "A published fluxonium map, no tuning: the
parameters match the paper. It knows when it is wrong, says what to measure next, and the engine is a CLI and a REST API. github.com/jgconroy/qholmes."

Timing notes: do not run the live circuit demo in the one-minute version (80 s); the NV run is the only live fit, and it takes under ten
seconds. If asked for more, the three-minute script below has the extra beats.

## Three-minute script (if time allows)

**0:00 Hook (Tab 4, scrolled to the map).** "This is what every superconducting-qubit lab sees on day one with a new chip: a map, a screen
full of lines, and a person guessing which circuit made them. Get it wrong and every calibration afterwards is wrong. qĤolmes reads the
Hamiltonian off the map."

**0:20 Live run (Tab 1).** Point at the stepper (Extract → Propose → Fit → Verify → Report), the hypothesis race, the loss sparkline, the
lines converging. "Every line is a physics model checked against scqubits to a millionth of a gigahertz. It also tried a resonator and
rejected it: a flat line alone is not evidence of coupling."

**0:50 The reveal (Tab 2).** Verdict band: the Hamiltonian typeset with the fitted numbers in parentheses, held-out RMS, the Bayes factor
against the runner-up family, the schematic. Race: "the P column is a Bayesian model probability; the score and the posterior agree."
Scroll to Evidence: the avoided-crossing zoom ("it hypothesised a resonator because the bare circuit left a bend at 7 GHz; here is the
2g splitting it measured") and the residual strip ("flat at ±6 MHz: that is what a correct model looks like").

**1:30 What it cannot explain, and what to measure (Tab 2).** Diagnostics → Unexplained Features: "the qubit crosses this 5.1 GHz line
without splitting, so it is a tone, not a coupled mode." Next Measurements: the numbered gap cards match the badges on the map; the
ranked table is the information-gain design in bits. Physics and Spec Sheet: sweet spots, dispersive shift by exact diagonalisation,
flux-noise $T_\varphi$, the operating-point tool.

**1:55 Autopilot (Tab 2, already replayed).** Drag the Round slider: "it started with 40 % of the sweep, chose the block of control values
with the most information, measured it from the digital twin, refit; watch σ fall. Swap one function for an instrument driver and this is
self-driving spectroscopy."

**2:15 Not just circuit QED (Tab 3).** Press **Identify Circuit**; it finishes in seconds. "Same engine, different Hamiltonian: zero-field
splitting, strain, field direction and the gauss-per-amp calibration from an ODMR map. Quantum dots, trapped-ion sidebands and Rydberg
Stark maps work the same way."

**2:35 Real data (Tab 4).** Diagnostics → Truth and Design → Reveal Hidden Truth: "a 2019 PRX fluxonium map digitised from the paper, no
tuning: E_J, E_C, E_L next to the paper's values, and the flux period comes out as one flux quantum."

**2:50 Close.** "It knows when it is wrong, it says what to measure next, and the same engine is a CLI and a REST API. Repo, tests and the
blind-test tables are on GitHub."

## If a judge asks about the LLM
Tab 2, bottom: **Ask qĤolmes** → click "Bias point for 5 GHz with the least flux noise?" (about 6 s). Point at the "tools called" line:
"every number comes from a tool call on the fitted model; the language model never invents a value."

## Fallbacks
- Live run slow or wrong: switch to Tab 2 (precomputed) and say the fast preset is what the audience is watching; the normal preset is
  what the saved result used.
- Ask panel errors: skip it; the explainer paragraph already shows the LLM path.
- Anything breaks: Ctrl-C the server and restart; the precomputed demos load in two seconds.
