# Demo runbook: exactly what to do and say

Works for the recorded video and for a live screen share. Total on-camera time about 90 seconds; preparation about 10 minutes.

## A. Preparation (do all of this before recording or sharing)

1. **Start the app fresh.** In a terminal:
   ```
   cd ~/calhacks13/circuitsleuth && source .venv/bin/activate && streamlit run app.py
   ```
   Wait until the browser opens on `http://localhost:8501`.
2. **Browser setup.** One window, zoom 90 % (Cmd/Ctrl + minus once), bookmarks bar hidden, no other tabs. Open these four tabs in this order:
   - Tab 1: `http://localhost:8501/?demo=A&load=1`  (the main one)
   - Tab 2: `http://localhost:8501/?platform=nv`
   - Tab 3: `http://localhost:8501/?demo=E&load=1`
   - Tab 4: `http://localhost:8501/?demo=C`  (only needed if someone asks for a live circuit run)
3. **Tab 1 warm-up.** Wait until the big "Tunable Transmon + Readout Resonator" verdict band is on screen and the two-sentence summary
   under it has appeared (it is written by Claude; it takes about 5 s). Then:
   - click the **»** at the top-left to collapse the sidebar (the page widens),
   - scroll down to **Autopilot** and click **Replay Saved Autopilot** once; a map with a Round slider and a σ-versus-round chart appears,
   - scroll back to the top.
4. **Tab 2 warm-up.** Collapse the sidebar the same way. Do NOT press Identify yet.
5. **Tab 3 warm-up.** Wait for the verdict band ("Fluxonium"), collapse the sidebar, scroll down to **Diagnostics**, click the
   **Truth and Design** tab, and leave the **Reveal Hidden Truth (Demo)** toggle OFF for now. Scroll back to the top.
6. **Check the Claude text exists.** In Tab 1 the paragraph under the verdict band should read like a physicist wrote it ("Best fit: ...").
   If it is a bare template or missing, the API key or credits are the problem; the demo still works, just skip any mention of Claude.
7. Go back to Tab 1, top of the page. Start recording / sharing.

## B. The 90-second demo, click by click

Each step: what you do, what is on screen, what you say. The lines are written to be spoken; keep a relaxed pace and let the screen carry the numbers.

**Step 1 (0:00–0:15) Tab 1, top of page.**
Do: nothing, just hover over the verdict band.
Screen: the big title, the Hamiltonian with amber numbers in parentheses, the KPI tiles, the schematic.
Say: "This is qĤolmes. Every new quantum chip starts as a spectroscopy map full of lines, and someone spends hours working out which circuit made them and what its energies are. qĤolmes reads that off the map. This is the result for one device: the circuit it identified, the Hamiltonian with the fitted energies written in, how well it predicts the columns it never saw, and a Bayes factor against the next-best family, all from a physics fit checked against scqubits."

**Step 2 (0:20–0:40) Tab 1, scroll down past "Hypotheses Tested" to "Evidence".**
Do: scroll until the large annotated map is in view, pause two seconds, then on to the two small panels (held-out columns; avoided crossings).
Screen: the map with coloured fitted lines over the data and numbered amber boxes; then the crossing zoom with the orange and white branches.
Say: "The map behind it is the raw data with the fitted transitions drawn on top. Notice that it found the readout resonator on its own: the bare circuit could not explain the bend at seven gigahertz, so it hypothesised a coupled resonator and then measured the splitting there. Just as important is what it refuses to explain. That flat line at five gigahertz is a pump tone, and it says so, because the qubit crosses it without any splitting."

**Step 3 (0:40–1:00) Tab 1, scroll to "Next Measurements", then "Autopilot".**
Do: pause on the ranked table for a beat, then scroll to the Autopilot result and drag the **Round** slider from 0 to 3.
Screen: the ranked sweeps in bits; then the autopilot map changing round by round and the σ-versus-round chart.
Say: "Because it knows what it has not explained, it can also say what to measure next. These sweeps are ranked by expected information gain, in bits, from the Fisher information of the fit. And in autopilot mode it closes the loop on a digital twin: it starts from forty percent of the sweep, picks the most informative block, measures it, refits, and you can watch the error bars collapse round by round. Swap one function for an instrument driver and this is self-driving spectroscopy."

**Step 4 (1:00–1:15) Tab 2.**
Do: switch to Tab 2 and press **Identify Circuit**; let the stepper run while you talk (under ten seconds).
Screen: the pipeline stepper lighting up, then the NV verdict band with the spin Hamiltonian.
Say: "None of this is specific to superconducting circuits. This is the same engine on an NV-centre map, and in a few seconds it returns the zero-field splitting, the strain, the field direction and the coil calibration. Quantum dots, trapped-ion sidebands and Rydberg Stark maps work the same way, each with its own Hamiltonian."

**Step 5 (1:15–1:30) Tab 3.**
Do: switch to Tab 3, scroll to Diagnostics (the Truth and Design tab is already selected) and switch the **Reveal Hidden Truth (Demo)** toggle ON.
Screen: a table of fitted values against the paper's values, with green ticks.
Say: "Finally, real data. This is a fluxonium map digitised from a 2019 PRX paper, run with no tuning at all, and the three energies land on the published values within their error bars. It knows when it is wrong, it tells you what to measure next, and the same engine is available as a command line and a REST API. The code is at github.com/jgconroy/qholmes."

Stop recording.

## C. If a judge asks (live only)

- **"Where is the AI?"** Tab 1, scroll to the bottom, **Ask qĤolmes**, click the first suggested question ("Bias point for 5 GHz with the least flux noise?"). About 6 s. Point at the grey "tools called" line: "every number comes from a tool call on the fitted model; the language model never invents a value."
- **"Show it live on a circuit."** Tab 4, press **Identify Circuit**; about 80 s with the fast preset. Narrate the hypothesis race and the loss sparkline; it ends on a fixed transmon and the 7.1 GHz spur appears under Unexplained Features.
- **"How accurate is it?"** README section "Blind test at scale": 200 random devices per platform, family 100 % on ions, NV, dots, atoms; parameters within 5 % in 86 / 72 / 78 / 63 % of them; circuits 72 % family with the failure mode named.
- **"Does the neural net decide?"** No: it proposes families and starting points; the fit decides; the Bayes factor and the score are computed independently and agree.

## D. If something goes wrong

- A page shows an error or spinner forever: reload the tab (the `?demo=A&load=1` URL reloads the precomputed result in two seconds).
- The live NV run looks odd: say "fast preset" and move on; the precomputed tabs carry the demo.
- Claude text missing: skip Step C's first item; nothing else depends on the API.
- Whole app dead: Ctrl-C in the terminal, run the streamlit command again, reload Tab 1.
