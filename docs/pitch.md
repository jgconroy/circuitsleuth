# 3-minute pitch script (2 min Q&A after)

**0:00 – 0:20  Hook.**  "This is what every superconducting-qubit lab sees on day one with a new chip." *(show the real map)* "Somewhere in here is a circuit. Finding its parameters by hand takes hours to days, and if you get it wrong every calibration afterwards is wrong. We built Shazam for quantum chips."

**0:20 – 0:50  Live demo, synthetic circuit A.**  Click *Identify*. Narrate what appears: features found, spur at 5.1 GHz flagged as flux-independent, hypotheses proposed, fit lines converging live on the data. "Those lines come from a physics model we check against scqubits to a millionth of a gigahertz."

**0:50 – 1:30  The reveal.**  Toggle *Reveal hidden truth*: parameters within a percent, including the resonator frequency and coupling it found by itself. Point at the revision message: "the bare circuit left a cluster of unexplained points at 7 GHz, so it hypothesised a resonator there and tried again." Point at the hold-out score: "it is graded on flux columns it never fitted."

**1:30 – 2:05  What it cannot explain and where to look next.**  Scroll: the unexplained 5.1 GHz line labelled as pump/LO leakage; the yellow 'look here' boxes; the Claude summary. "A fitter's most useful output is the list of things it could not fit, and the measurement that would settle them."

**2:05 – 2:40  Real data.**  Switch to demo E: a 2019 PRX fluxonium map digitised from the paper. Show the recovered EJ, EC, EL next to the paper's values, and the flux period coming out as one flux quantum because the paper's axis was already calibrated. "No tuning for this dataset. Same code."

**2:40 – 3:00  Close.**  "Next step is obvious: the agent that says 'look here' drives the instrument and takes that sweep. Bring-up of a new chip becomes a loop that runs itself. Repo, tests and the blind-test table are on GitHub."

**Q&A ammunition**
- Why not just least squares? Spurs and missing lines break it; Geman–McClure bounds the cost of any outlier; features are scored, not pixels.
- Does the neural net decide anything? No. It proposes families and starting points; every number is from the physics fit, with Jacobian uncertainties.
- How do you know the physics is right? `tests/test_models.py` compares every Hamiltonian to scqubits at 1e-6 GHz, including the resonator-dressed spectrum.
- Degeneracies? EJ–EC along constant f01: the report flags it and asks for the two-photon line.
- Flux axis in volts? Period and offset are fitted; a symmetry analysis seeds them.
