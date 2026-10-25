# 3-minute pitch script (2 min Q&A after)

**0:00 – 0:20  Hook.**  "This is what every superconducting-qubit lab sees on day one with a new chip." *(show the real map)* "Somewhere in here is a circuit. Finding its parameters by hand takes hours to days, and if you get it wrong every calibration afterwards is wrong. We built Shazam for quantum chips."

**0:20 – 0:50  Live demo, circuit C (fast, ~80 s; start it before you begin talking).**  Click *Identify circuit*. Narrate the stepper (Extract → Propose → Fit → Verify → Report), the hypothesis race eliminating candidates, the loss sparkline, the fit lines converging on the data. "Those lines come from a physics model we check against scqubits to a millionth of a gigahertz." Point at the rejected row: "it tried a resonator and rejected it: a flat line alone is not evidence of coupling."

**0:50 – 1:30  The reveal (circuit A, "Load precomputed result").**  The verdict band: circuit name, the Hamiltonian with fitted numbers, held-out RMS 6 MHz, margin to the runner-up, the schematic with values. Toggle *Reveal hidden truth*: every parameter within 0.4 %, including the resonator frequency and coupling it found by itself. Scroll to the avoided-crossing zoom: "the bare circuit left a bend at 7 GHz, so it hypothesised a resonator there; here is the 2g splitting it measured." The residual strip: "flat at ±6 MHz; that is what a correct model looks like."

**1:30 – 2:05  What it cannot explain and where to look next.**  The 5.1 GHz line card: "the qubit crosses it with no splitting, so it is a tone, not a coupled mode." The numbered 'look here' cards matching the badges on the map. The spec sheet: sweet-spot bias values, dispersive shift by exact diagonalisation, flux-noise Tφ. "A fitter's most useful output is the list of things it could not fit, and the measurement that would settle them."

**1:50 – 2:05  Autopilot.**  On the NV demo press *Run autopilot (3 rounds, live)*: "it starts with 40 % of the sweep, computes which block of control values carries the most information in bits, measures it, refits; watch σ fall round by round. Swap one function for an instrument driver and this is self-driving spectroscopy."

**2:05 – 2:25  Not just circuit QED.**  Switch platform to "NV centres in diamond", pick the demo, click Identify (it finishes in seconds): "Same engine, different Hamiltonian: zero-field splitting, strain, the field direction and the gauss-per-amp calibration, from an ODMR map. Quantum dots, trapped-ion sidebands and Rydberg Stark maps work the same way; five platforms, and a new one is one function."

**2:25 – 2:45  Real data.**  Switch to demo E: a 2019 PRX fluxonium map digitised from the paper. Show the recovered EJ, EC, EL next to the paper's values, and the flux period coming out as one flux quantum because the paper's axis was already calibrated. "No tuning for this dataset. Same code."

**2:45 – 3:00  Close.**  "Next step is obvious: the agent that says 'look here' drives the instrument and takes that sweep. Bring-up of a new chip becomes a loop that runs itself. Repo, tests and the blind-test table are on GitHub."

**Q&A ammunition**
- Why not just least squares? Spurs and missing lines break it; Geman–McClure bounds the cost of any outlier; features are scored, not pixels.
- Does the neural net decide anything? No. It proposes families and starting points; every number is from the physics fit, with Jacobian uncertainties.
- How do you know the physics is right? `tests/test_models.py` compares every Hamiltonian to scqubits at 1e-6 GHz, including the resonator-dressed spectrum.
- Degeneracies? EJ–EC along constant f01: the report flags it and asks for the two-photon line.
- Flux axis in volts? Period and offset are fitted; a symmetry analysis seeds them.
