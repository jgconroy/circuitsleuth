# Demo runbook: exactly what to do and say

Works for the recorded video and for a live screen share. Total on-camera time about 60 seconds; preparation about 10 minutes.

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

## B. The 60-second demo, click by click

Each step: what you do, what is on screen, what you say. Keep the voice relaxed; the numbers do the work.

**Step 1 (0:00–0:14) Tab 1, top of page.**
Do: nothing, just hover over the verdict band.
Screen: the big title, the Hamiltonian with amber numbers in parentheses, the KPI tiles (held-out RMS, % explained, margin, Bayes factor), the schematic.
Say: "This is qĤolmes: Shazam for quantum chips. A raw spectroscopy map goes in. This comes out: the circuit, its Hamiltonian with the fitted numbers in place, the held-out error, and a Bayes factor against the next-best family. Every number comes from a physics fit checked against scqubits."

**Step 2 (0:12–0:27) Tab 1, scroll down past "Hypotheses Tested" to "Evidence".**
Do: scroll until the large annotated map is in view, pause two seconds, then scroll a little further to the two small panels (held-out columns; avoided crossings).
Screen: the map with coloured fitted lines over the data, numbered amber boxes; then the crossing zoom with the orange and white branches.
Say: "It found the readout resonator by itself, from the bend the bare circuit left at 7 GHz; here is the splitting it measured. The one line it cannot explain, it calls a pump tone, because the qubit crosses it without splitting."

**Step 3 (0:27–0:42) Tab 1, scroll to "Next Measurements", then "Autopilot".**
Do: pause on the ranked table (right side) for a beat, then scroll to the Autopilot result and drag the **Round** slider from 0 to 2.
Screen: the ranked sweeps with bits; then the autopilot map changing per round and the σ-versus-round chart.
Say: "It ranks the next sweep by information gain in bits, and the autopilot runs that loop on a digital twin: 40 % of the sweep, choose, measure, refit. Watch the error bars fall."

**Step 4 (0:42–0:52) Tab 2.**
Do: switch to Tab 2 and press **Identify Circuit**. It finishes in under ten seconds; let the stepper run while you talk.
Screen: the pipeline stepper lighting up, then the NV verdict band with the spin Hamiltonian.
Say: "Same engine on an NV-centre map: zero-field splitting, strain, field direction, in seconds. Quantum dots, trapped ions and Rydberg atoms work the same way."

**Step 5 (0:52–1:00) Tab 3.**
Do: switch to Tab 3, scroll to Diagnostics (the Truth and Design tab is already selected) and switch the **Reveal Hidden Truth (Demo)** toggle ON.
Screen: a table "fitted ± σ | truth | within 2σ" with green ticks; the paper's E_J, E_C, E_L beside the fit.
Say: "A published fluxonium map, no tuning: the parameters match the paper. It knows when it is wrong, it says what to measure next, and the same engine is a CLI and a REST API. github.com/jgconroy/qholmes."

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
