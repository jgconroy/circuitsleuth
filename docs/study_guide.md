# qĤolmes study guide: how it works, what every number means, and how to answer the judges

*For the author, before Q&A. Everything below is taken from the code in `circuitsleuth/`, `app.py`, `ui/` and the results in `README.md` / `docs/STATE.md`. Function names are given so you can point at the code if a judge asks. Numbers are quoted only where the repo records them.*

---

## 1. One-page summary

**What it does, in plain words.** A spectroscopy map is a picture: along the horizontal axis a control knob (flux bias, coil current, gate voltage, endcap voltage, plate voltage), along the vertical axis a probe frequency, and bright lines wherever the device absorbs. qĤolmes takes that picture and returns the *equation* that drew the lines: which Hamiltonian family the device is, the numerical value of every parameter in it (with error bars), how the knob axis maps onto the physical control variable, which lines in the picture the model cannot explain, and which measurement to take next to learn the most. It does this for five kinds of quantum hardware (superconducting circuits, NV centres in diamond, semiconductor double quantum dots, trapped ions, Rydberg atoms) with one shared engine, and for superconducting circuits it also decides on its own whether a readout resonator is coupled to the qubit.

**The one-sentence pitch.** "The map already contains the Hamiltonian, and qĤolmes reads it out with a physics fit that knows when it is wrong."

**Why it is technically hard (three sentences to say out loud).**

1. The data are not a clean curve: a real map contains pump tones and mixer spurs (perfectly straight lines that any flexible model loves to "explain"), dropouts, noise, an uncalibrated control axis, and lines that overlap or swap labels at avoided crossings; so the problem is simultaneously *which model*, *which parameters*, *which calibration*, and *which features are not even physics*.
2. The forward model is a batched eigensolve, not an analytic formula, and the loss landscape has exact degeneracies (for example $E_J$ and $E_C$ trade off along constant $f_{01}$ for a transmon), so a naive least-squares fit lands in the wrong valley; qĤolmes combines a bounded robust loss, annealed differential evolution, explicit degeneracy sweeps and relabelling alternatives to get out.
3. Every reported quantity has to be *honest*: the fit is verified on flux columns it never saw, uncertainties were calibrated by coverage runs on synthetic data and are backed by an MCMC posterior, model choice is reported both as a feature-level score and as a Bayes factor, and the next measurement is chosen by Fisher information, not by a heuristic.

**Three things to remember about the division of labour.** The CNN (convolutional neural network) only *proposes* families and starting points; it never reports a number. Claude only *narrates* and *answers questions by calling the physics engine as tools*; it never computes a value. Every number on the page comes from `fit.py` / `models.py` / `derived.py` / `design.py`.

---

## 2. The pipeline, step by step

Think of one map going through. The Streamlit stepper shows the same five stages (`ui.theme.STEPS`): **Extract → Propose → Fit → Verify → Report**. The orchestration is `agent.identify()` in `circuitsleuth/agent.py`.

### 2.1 Input formats

**Simple version.** You can give qĤolmes an image (a 2-D array plus the two axis vectors) or a list of already-extracted points.

**What the code accepts.**
- `.npz` with keys `image`, `x`, `f`; HDF5 (`.h5`, `.hdf5`) where you pick the 2-D dataset and optional 1-D axis datasets (`ui.importers.h5_catalog`, `h5_read`; complex data is turned into magnitude); a 2-D grid CSV whose first row is the control values and first column the frequencies (`importers.read_grid_csv`, delimiter sniffed); a screenshot PNG/JPG with axis ranges typed in; or a point list CSV `x, f[, weight]`.
- Frequency units are converted to GHz on import (`importers.FREQ_TO_GHZ`: GHz/MHz/Hz) for GHz platforms; ion and atom platforms work natively in MHz (`models.PLATFORMS[...]["freq_unit"]`) and all numeric code is unit-agnostic.
- `importers.orient` sorts both axes ascending (the extractor assumes an increasing frequency axis), `crop` applies the frequency/control windows from the sidebar (refusing to leave fewer than 10 rows or 3 columns), and `thin_columns` caps the control axis at 121 columns.
- The REST API (`api.py`) takes the same two shapes: `{"image","x","f"}` or `{"points","band"}`, validated by pydantic before any fitting starts (422 on malformed input).
- When only points are given, `agent.rasterize` builds a Lorentzian image from them so the CNN proposer (which looks at pictures) can still run.

### 2.2 Feature extraction (`circuitsleuth/extract.py`)

**Simple version.** Go column by column (one control value at a time), find the bright peaks, locate each to better than a pixel, link peaks in neighbouring columns into line segments ("tracks"), and separate lines that do not move with the control knob (spurs) from lines that do (physics).

**Normalisation (`extract.normalise`).** Each column has its median subtracted (removes the background level per control value). The noise scale is the MAD (median absolute deviation) multiplied by 1.4826 so that it equals the standard deviation for Gaussian noise. The per-column noise estimate is `max(column MAD, global MAD)`: it is only ever *raised* for noisy columns, never lowered for quiet ones (a quiet column with a tiny MAD would otherwise explode into false peaks). If the skewness of the clipped image is negative the sign is flipped so that lines are positive peaks (two-tone spectroscopy often shows dips).

**The 4σ rule (`extract.extract_points`).** `scipy.signal.find_peaks` is called per column with `height = 3` (that is $0.75 \times$ `snr`, `snr = 4`) in robust-σ units, `prominence = 1.5`, and a minimum separation of 2 pixels. Peaks with height $\ge 4\sigma$ are kept outright. Peaks between 3σ and 4σ are kept **only** when an adjacent column has a confident ($\ge 4\sigma$) peak within ±4 pixels of the same frequency (a *continuity* rule: a faint piece of a real line has bright neighbours, a noise wiggle does not). Height above the per-column baseline is used rather than scipy's prominence alone because prominence let 4σ noise through (STATE §4). At most 8 peaks per column survive, the brightest first. For Gaussian noise the chance that a single pixel clears 4σ is about $3\times10^{-5}$, so a 400-pixel column yields on average about 0.01 false peaks.

**Sub-pixel parabola.** For a peak at pixel $p$ with neighbours $y_0, y_1, y_2$, the maximum of the parabola through the three points sits at

$$\delta = \frac{1}{2}\,\frac{y_0 - y_2}{y_0 - 2y_1 + y_2},$$

clipped to $[-\tfrac12, \tfrac12]$ pixels. The feature frequency is $f_p + \delta\,\Delta f$. The weight of a feature is its peak height, normalised so the brightest feature in the map has weight 1.

**Optional column smoothing.** `smooth_cols = k > 1` averages each row over $k$ neighbouring columns and multiplies by $\sqrt{k}$ so σ units keep their meaning; a continuous line gains $\sqrt{k}$ in signal-to-noise, column noise does not. Default is 1.

**Track linking (`extract.link_tracks`).** Greedy column-to-column linking. A point joins the track of a point in the previous column if it lies within a tolerance of that track's linear extrapolation (last frequency step added to the last frequency). The tolerance is *slope-aware*: `tol = min(max_tol, max(df_tol, 1.5 × |last step|))`, so a steep line (which moves far between columns) gets a wider window than a flat one. `df_tol` is the platform scale ×20/3 (0.08 GHz for circuits), `max_tol = max(0.45, 5.6 × df_tol)`. Each previous point links to at most one successor. Slope-awareness was added because steep lines fragmented into sub-3-point tracks and then only the spur counted in the score (STATE §4).

**Flat lines (`extract.flat_lines`, `flat_feature_mask`).** Sort all feature frequencies; any cluster of features within $2\times$`tol` in frequency that spans at least 60 % of all control columns is a flux-independent line; points within $1.5\times$`tol` of it are masked as "flat". `tol` is the platform `flat_tol` (0.02 GHz for circuits). This is done by frequency occupancy rather than by track, so it is robust to tracks swapping identity at crossings.

**Why flat lines are removed before flux-dependent fits.** A pump tone, LO (local oscillator) leakage, or mixer spur is a straight horizontal line. A flux-tunable Hamiltonian has no straight horizontal transition, so any such line is noise *for that hypothesis*; left in, the robust loss would still pay a bounded price for every one of its 80 points and the optimiser would try to bend a predicted line through it. `agent._fit_points_for` therefore fits flux-dependent circuit hypotheses on the non-flat points only, provided enough points are dynamic (see the `DYN_MIN_ABS` rule in 2.5). A *fixed-frequency* transmon's own lines are flat, so it is fitted on everything, as are the other platforms (a charge qubit far from zero detuning is nearly flat).

**Track weights (`extract.track_weights`).** A track of $n \ge 3$ points gets weight $\sqrt{n}\times$ (mean point weight); shorter tracks are isolated noise and get zero weight, so a flexible model cannot score by sweeping lines through noise. Flat tracks count half: "a flat feature matched by a flat line is one number of evidence, a curved track matched by a curved line is a shape".

**Artefact columns (`extract.artefact_columns_mask`).** A column with more than `max(6, 3 × median peaks per column)` peaks is a burst (a glitch, a saturated sweep, a vertical streak in a digitised figure). Those points *stay in the fit* (the robust loss ignores them, and dropping them shifted the real-data $E_J$ to 3.8) but are excluded from the explained fraction and the unexplained list. On the Nguyen 2019 real map, 72 of 169 extracted features sat in 9 streak columns.

**Flux-calibration prior (`extract.estimate_flux_calibration`), circuits only.** Every flux-tunable spectrum is mirror-symmetric about $\Phi = 0$ and $\Phi = \tfrac12$. For each candidate mirror position $x_0$ the code computes the median nearest-neighbour distance between the point cloud and its mirror image (a Chamfer-type distance); minima of that curve are symmetry axes; the autocorrelation of the curve gives the axis spacing, which is half a period; and whether the lowest line is a maximum or minimum at an axis tells $\Phi = 0$ from $\Phi = \tfrac12$. This only *narrows* the search (period ±15 %, offset ±8 % of a period) as a second candidate; the wide search always runs too. It can fail: on the "surprise seed 7" heavy fluxonium, whose plateaus look flat, it returned period 1.68 instead of 0.93.

**Geometric priors for dots and atoms (`extract.estimate_dqd_init`, `estimate_stark_init`).** For a charge qubit, $f^2 = \text{lever}^2 (x-x_0)^2 + (2t_c)^2$ is *linear* in $(x-x_0)^2$, so for each trial $x_0$ a two-parameter linear least squares on the longest dynamic tracks gives lever arm and $t_c$; the $x_0$ with the smallest median residual wins. For a Stark map, $f = f_v - k\,(x-x_0)^2$ is linear in $u = (x-x_0)^2$ per track; tracks with matching vertex and curvature are merged into components; the plate calibration $\kappa$ is unobservable so it is set to put the geometric-mean polarisability at 2. These priors raised the 200-map tables from 58 % to 78 % (dots) and 57 % to 63 % (atoms).

**What can go wrong.** Very faint lines below 3σ are lost; the real-data right half had no extractable features. A line wider than the inlier band (the digitised figure had ~100 MHz line width at 12.5 MHz pixels against a ±36 MHz band) looks "unexplained" unless `c_local` is widened (the app uses 0.03 GHz for the real demo). The flat-feature mask can swallow the plateau of a heavy fluxonium (on 60 synthetic fluxoniums: median 3 % of the circuit's own points, > 30 % in 4 of 60).

### 2.3 Hypothesis proposal: the CNN (`circuitsleuth/classify.py`)

**Simple version.** A tiny image classifier looks at a 64×64 thumbnail of the map and says "this looks like a fluxonium, probably no visible avoided crossing, roughly 1.3 flux periods in the window, energies around here". That ranking decides which families are fitted first and where the search starts. It never decides the answer.

**Architecture (`classify.Net`).** Input: one channel, 64×64. Four 3×3 convolutions with channel widths $16w, 32w, 64w, 64w$ (the adopted model has width $w = 2$), each followed by ReLU; max-pooling by 2 after the first three; an adaptive average pool to 4×4; a fully connected layer to 256 units with ReLU and 10 % dropout. Heads: a 3-way family softmax, a 1-logit "visible avoided crossing" flag, a 1-output "number of flux periods in the window" regressor (log-normalised to the range 0.4–3.0), and one parameter head per family predicting normalised log-parameters in $[0,1]$ (with the resonator slots $f_r, g$ appended).

**Thumbnail (`synth.downsample`).** Per-column median removal, global MAD scaling, sign flip if dips, clip to $[0, 20]$ and divide by 20, then block max-pooling to 64×64. Max-pooling preserves thin lines that mean-pooling would wash out.

**Training (`classify.train`).** AdamW (lr $2\times10^{-3}$, weight decay $10^{-4}$) with a one-cycle schedule (max lr $3\times10^{-3}$), batch 128. Loss = cross-entropy (family) + binary cross-entropy (crossing flag) + masked MSE (periods) + 2 × masked MSE (parameters; masked because each map only has targets for its own family). Augmentation per batch: horizontal flip with probability 0.5, circular roll along the flux axis with probability 0.3 (period/offset invariance), and a multiplicative gain jitter $\exp(0.1\,\mathcal N(0,1))$.

**What it is trained on.** Only synthetic maps from `synth.py` (see 2.10). The adopted model (`artifacts/classifier.pt`) was trained on 190 000 maps for 30 epochs on a 32-core machine: family accuracy 0.936, visible-crossing accuracy 0.874, normalised parameter RMSE 0.216. Earlier laptop runs: 20k maps → 0.857, 40k maps → 0.888. The validation confusion matrix (rows = truth transmon, tunable, fluxonium) is `[[3171, 43, 53], [85, 3066, 163], [79, 214, 3126]]`: the dominant confusion is tunable transmon ↔ fluxonium, exactly the physically ambiguous pair.

**The target change that mattered.** The raw label "resonator present" sat near chance (0.55–0.59) because a far-detuned resonator leaves no trace in a two-tone map. The target was changed to `synth.crossing_visible`: resonator present *and* the bare 0-1 line crosses $f_r$ inside the window and band. A 2 000-map, 1-epoch smoke run went from 0.55 to 0.85 immediately. Lesson to quote: "pick a label the data can actually carry".

**How proposals are used (`classify.propose`, `agent.identify` step 2).** For each (family, coupled) pair the probability is $p_\text{fam} \times p_\text{cpl}$ or $p_\text{fam}\times(1-p_\text{cpl})$; the top 6 are taken and de-duplicated; the top `top_k` *bare* families (2 for the fast preset, 3 for normal) are fitted. The seed parameters are de-normalised; the flux period seed is window span divided by the predicted number of periods; the offset seed is the window start. The family probability also enters the score as a mild prior: `prior_weight × (p_family − 1/3)` with `prior_weight = 0.05` (raising it to 0.25 changed nothing on the hard cases, STATE "Sun ~03:40"). If the CNN is unavailable, every family of the platform is fitted.

**Why this way.** A pure neural regressor would have to be right about numbers on data it has never seen (real maps); a proposer only has to be *useful*, and a wrong proposal costs search time, not correctness. The CNN is also confidently wrong sometimes: both v1 and v2 called demo A (tunable transmon + resonator) a fluxonium with high probability, which is why the prior weight is small and why there is an escalation rule (2.5).

### 2.4 Fitting (`circuitsleuth/fit.py`)

**What a hypothesis is.** `fit.Hypothesis(family, coupled)`: a family name from `models.FAMILIES` and a flag for "plus readout resonator" (circuits only). Its parameter names are the family's physical parameters, plus `fr, g` when coupled, plus the family's **control-calibration parameters**: `period, offset` for circuits ($\Phi/\Phi_0 = (x-\text{offset})/\text{period}$), `field_scale, x0` for NV and Rydberg ($B = \text{field\_scale}\,(x-x_0)$ in gauss, or $E$ in V/cm), `lever_arm, x0` for dots ($\varepsilon = \text{lever\_arm}\,(x-x_0)$), `volt_scale, x0` for ions ($\omega_z = \text{volt\_scale}\sqrt{x-x_0}$). The mapping lives in `models.control_from_x`. So the control axis is *not* assumed calibrated: calibrating it is part of the fit.

**Log-parameterisation (`fit.to_theta`, `LOG_PARAMS`).** Energies and scales ($E_J, E_C, E_L, E_{J,\max}, f_r, g$, period, field_scale, lever_arm, $t_c$, $E$, $f_{rad0}$, volt_scale, $\alpha_{1,2}$, dfs) are searched in $\theta = \ln(\text{value})$; $d$, angles, offsets and $f_0$ stay linear. Reasons: positivity for free; a step of 0.1 in $\theta$ is 10 % whatever the magnitude, which makes the optimiser's step sizes meaningful across parameters that differ by factors of 100; and the Jacobian σ in θ-space is a *relative* uncertainty (converted back by $\sigma_\text{value} = \text{value}\times\sigma_\theta$). Bounds come from `models.FAMILIES[...]["bounds"]` (e.g. transmon $E_J \in [3, 40]$, $E_C \in [0.15, 0.6]$ GHz; fluxonium $E_J \in [2, 12]$, $E_C \in [0.4, 2.5]$, $E_L \in [0.1, 2]$; resonator $f_r \in [4.5, 9.5]$, $g \in [0.03, 0.35]$), the period bounds from the window span (`default_bounds`: span/3 to span/0.4), and user-known parameters narrow or pin them (`agent._apply_user_bounds`).

**The forward model.** `fit.predict_lines(hyp, params, x)` maps $x$ to the control variable and calls `models.transitions`, which returns a dictionary of predicted line frequencies per column: for circuits the five lines 0-1, 0-2, 0-3, 0-2/2ph (two-photon, at half the 0-2 frequency) and 1-2 (thermal), plus "res" when coupled; for NV the eight lines NV1± … NV4±; for dots 0-1 and its two-photon copy; for ions carrier, ±axial, ±radial, ±2×axial; for Rydberg the two $|m_j|$ components. These are batched eigensolves: one `np.linalg.eigvalsh` call over all columns at once.

**The nearest-line Geman–McClure loss (`fit.Objective`).** For feature $i$ at $(x_i, f_i)$ with weight $w_i$ (normalised to sum to 1), let $d_i = \min_\ell |f_i - L_\ell(x_i;\theta)|$ be the distance to the nearest predicted line in its column. The loss is

$$\mathcal L(\theta) = \sum_i w_i\,\rho_c(d_i),\qquad \rho_c(d) = \frac{d^2}{d^2 + c^2},$$

plus a weak coverage term $0.15\times$ (fraction of in-band 0-1 columns with no feature within $3c$). Geman–McClure is quadratic for $d \ll c$ and saturates at 1 for $d \gg c$. **Why bounded losses matter:** with ordinary least squares, one spur line 1 GHz away contributes $(1\,\text{GHz})^2$ per point, thousands of times more than a real line 10 MHz off, so the optimiser bends the model toward the spur. With a bounded loss a spur costs at most its weight, so the best the model can do is ignore it, and it is then reported as unexplained. The nearest-line construction means no explicit assignment of points to lines is needed (the assignment is implicit and can change during the search). The robust scale $c$ is the "width" of the loss: large $c$ gives a smooth, wide landscape (good for global search), small $c$ gives precision (good for the final answer).

**Differential evolution and the annealing of $c$ (`fit._fit_once`).** The global stage uses `scipy.optimize.differential_evolution` (DE): a population of parameter vectors (popsize 15 per dimension; 20 for fluxonium) is initialised with a Sobol sequence; at each generation every member is replaced by a mutant combining three others (`DE/rand/1/bin` style) if the mutant has a lower loss. DE needs no gradients, respects bounds, and handles multimodal landscapes. qĤolmes runs **three stages with the robust scale annealed** from coarse to fine: $c = 4c_g$ for `maxiter/2` generations, then $c = c_g$ for `maxiter`, then $c = \max(c_g/4, 1.5\,c_\ell)$ for `maxiter/2`, with $c_g = 0.08$ GHz and $c_\ell = 0.012$ GHz for circuits (`models.platform_scales`). The population is **carried between stages** (`init = res.population`), so the search narrows rather than restarting; the first stage sees a smooth landscape where a line 300 MHz off still gets partial credit, the last stage sharpens on the real lines. Stages stop early when the population has converged (`tol = 1e-3`). The global stage works on a thinned copy of the data (`fit.subsample`: at most 31 columns, 3 000 points; faint points are kept because they pin $E_C$ and $E_L$) and with a reduced oscillator cutoff (40 instead of 60 for fluxonium). Multiprocessing is used for the population evaluation (`workers`, `updating="deferred"`); `Objective` is picklable for that reason.

**The staged 0-1-first search (`fit.fit_hypothesis`, `staged=True`).** For flux-dependent circuit hypotheses with at least 12 columns, the first pass fits *only the 0-1 line* to the *brightest feature per column* (`fit.brightest_points`, `line_filter=["0-1"]`): one line, no assignment ambiguity, hence a much smoother landscape. The bounds are then narrowed around that solution (period ±10 %, offset ±6 % of a period, log parameters within a factor 1.6, but $E_J$/$E_C$ left open for transmons because the 0-1 line alone cannot separate them, and $f_r, g$ left open) and the full fit runs. If the inlier fraction exceeds 0.97 the result is accepted immediately.

**The EJ–EC valley.** For a transmon $f_{01} \approx \sqrt{8E_JE_C} - E_C$: many $(E_J, E_C)$ pairs give the same 0-1 line, and only the anharmonicity (the 0-2/2ph line at $f_{01} - E_C/2$ or the 1-2 line at $f_{01} - E_C$) separates them. After the global search, `_fit_once` sweeps 25 values of $E_C$ from $0.5\times$ to $2\times$ the current value, solving $E_J = (f_{01} + E_C)^2/(8E_C)$ to keep $f_{01}$ fixed, evaluates the loss at scale $3c_\ell$, and restarts the polish from the best point. Fluxonium has an analogous valley at fixed half-flux $f_{01}$: 13 $E_C$ values, $E_J$ solved by `brentq` on the exact half-flux splitting (kept because it is cheap; it did not fix the fast-preset fluxonium cases).

**Multi-start.** `n_starts` (1 fast, 2 normal, 3 thorough) complete global+local runs from different DE seeds; the lowest fine-scale loss wins. Fluxonium's global search was "luck-dependent", hence the extra starts and the larger population.

**Soft-L1 polish.** `scipy.optimize.least_squares` with `loss="soft_l1"`, $\rho(u) = 2(\sqrt{1+u^2}-1)$, on the residuals $\sqrt{w_i N}\,d_i$. Two stages: `f_scale = 3c_\ell` first (pulls in lines 50–100 MHz off), then `f_scale = c_\ell`. Soft-L1 is quadratic near zero and linear far away, so it is still robust but has a non-vanishing gradient far from the lines (Geman–McClure's gradient dies at large $d$). This is a trust-region reflective Gauss–Newton solver with bounds.

**Inliers, RMS, Jacobian σ.** After the polish, inliers are points with $d_i < 3c_\ell$ (36 MHz for circuits, 4.5 MHz for NV, 30 kHz for ions, 0.45 MHz for atoms). `inlier_frac` is the weight of inliers; `rms_inliers` their RMS residual. The covariance is $s^2 (J^\top J)^{+}$ with $J$ the least-squares Jacobian restricted to inliers and $s^2 = \sum r_i^2/(n - p)$; `sigma[name] = value × sd` for log parameters.

**Trimmed local polish for geometric priors.** When a dot/atom geometric prior exists, `fit_hypothesis(do_global=False, trim=10 c_ℓ)` fits only points within $4\times$trim of the prior's lines, then within trim of the refined lines: a crowd of spur points cannot drag the soft-L1 polish out of the prior's basin. The result is scored on all points like any other candidate.

**Relabel alternatives (`fit.relabel_alternatives`).** Some label ambiguities are invisible to the loss: a fixed transmon's second flat line fits equally well as the two-photon 0-2/2ph line ($f_{01} - E_C/2$) or the thermal 1-2 line ($f_{01} - E_C$), with $E_C$ differing by exactly 2×; an ion's first-order axial sidebands fit as second-order ones with volt_scale halved; a dot's 0-1 hyperbola fits as its two-photon copy with $t_c$ and lever arm doubled. After the fit, for each pair (used weaker line $u$, unused brighter line $m$ with nominal visibility $\text{vis}(m) > \text{vis}(u)$), the points on $u$ are reassigned to $m$, a *plain* least squares (robustness would flatten the 1 MHz pull this step needs) solves for the parameters with the family's primary line as anchor, and a local polish follows. Alternatives are attached to the result and scored by the agent; the visibility-weighted missing-line penalty then prefers the reading that does not leave a bright predicted line with nothing under it. "Unused" means at most `max(2, 10 %)` of the in-band columns have a point under the line (two stray noise points must not block it). Physics prior encoded: at usual two-tone powers the two-photon line is more common than the thermal line. This step raised trapped ions from 74 % to 86 % and (with the geometric prior) quantum dots from 58 % to 78 % on the 200-map tables.

**Offset normalisation.** The flux offset is defined modulo the period; `fit._normalise_offset` reports the equivalent value nearest the data centre.

### 2.5 Scoring and model choice (`agent._score`, `agent.identify` steps 3–4b)

**Simple version.** A hypothesis is rewarded for the *features* it explains, not the pixels, and punished for predicting bright lines where nothing is seen, for needing many parameters, and for sloppiness.

**The decision score** (`agent._score`, stored as `track_score`):

$$S = \underbrace{\frac{\sum_t w_t\,(\text{inlier fraction of track } t)}{\sum_t w_t}}_{\texttt{fit.track\_score}} \;-\; 0.3\,P_\text{miss} \;-\; 0.02\,n_\text{params} \;-\; \frac{\min(\text{RMS}_\text{MHz}, 30)}{300} \;+\; \text{prior}.$$

- **Track score:** fraction of track *weight* ($\sqrt{n}\times$ brightness, flat tracks at half) that lies within $3c_\ell$ of a predicted line. Counting tracks instead of points stops a model from winning by matching one 80-point spur.
- **Missing-line penalty** (`agent.missing_line_penalty`): for every predicted line that runs in-band through at least 3 measured columns, the fraction of those columns with no feature within $3c_\ell$, weighted by the line's nominal visibility (`models.line_weights`: 0-1 → 1.0, 0-2/2ph → 0.45, 0-2 → 0.35, 0-3 → 0.15, 1-2 → 0.12, res → 0.5). A model that predicts a strong 0-1 line through empty space is implausible; an unseen faint 0-3 line costs little. Weight 0.3.
- **Parameter-count penalty:** 0.02 per parameter (a resonator adds two).
- **RMS precision penalty:** a model that explains the same features with 4 MHz residuals beats a look-alike that needs 15 MHz.
- **Prior:** `0.05 × (p_CNN(family) − 1/3)`.

Note there are two "score"-like numbers in a report: `track_score` (the one above, which decides) and `score` from `FitResult.score` (hold-out inlier fraction minus 0.02 per parameter). The hypothesis race shows `track_score`.

**Candidates per hypothesis.** `agent.run()` tries several bound/seed combinations for each family: the wide search; for circuits the symmetry-calibrated narrow search; for dots/atoms the narrowed search around the geometric prior and the trimmed local polish; plus every relabel alternative. Each candidate is scored on **all** points (including the spurs that were removed from the fit) and the best wins.

**The `DYN_MIN_ABS` skip rule.** Flux-tunable circuit families are skipped when fewer than `max(10, min(0.2 n, 40))` features vary with flux. The old rule was a plain 20 % fraction, and three pump lines of 81 points each could hide a 50-point fluxonium (the run then ended in "no hypothesis could be fitted" because the CNN proposed only tunable families); with the absolute cap of 40 a 10-map fluxonium table went from 8/10 to 10/10 families. If every proposed family was skipped, the fixed transmon is fitted as a fallback.

**Resonator acceptance rules** (step 4, circuits only). A coupled model is tried when `try_coupled="always"`, or in `"auto"` mode when the CNN's $p_\text{crossing} > 0.25$ or the best bare score is below 0.85, for the top `coupled_trials` bare families (1 fast, 2 normal) that are within 0.35 of the best. Seeds for $f_r$, in order: the peak of a 0.2 GHz histogram of *unexplained flux-dependent* points (an avoided crossing leaves a cluster of unexplained points at $f_r$), the CNN's seed, any flat line between 4 and 10 GHz, then 7 GHz. The circuit and calibration parameters are kept within 0.5–2× of the bare solution (0.8–1.25× was too tight), $f_r$ within ±0.6 GHz of the seed, $g$ started at 0.1. The coupled model is **accepted only if all three hold**: (i) score gain over the bare circuit > `coupled_margin = 0.05`; (ii) $g > 2\sigma_g$; (iii) the track score on *flux-dependent features alone* improves by at least 0.02 ("a resonator must buy its place with flux-dependent evidence: a bend, a crossing, a dispersive pull; explaining one more flat line is not evidence, any tone does that"). A rejected coupled model is kept in the report with its score capped just below the bare one, so the race shows it as "rejected".

**Escalation on CNN disagreement** (step 4b). When the CNN puts ≥ 0.8 on a family whose fit lost and ≤ 0.2 on the winner's family, that family is refitted with double effort (2× iterations, popsize ≥ 16, two starts) and re-ranked. If it still loses, confidence becomes "low" with `cnn_disagrees = {family, prob}` and an explicit sentence. Surprise seeds 4 and 7 (fluxoniums read as transmons) now carry that warning.

**Confidence.** From the score gap to the best *other family* (same-family resonator variants do not count): gap > 0.15 "high", > 0.05 "medium", else "low". Two overrides: `poorly_explained = True` and level "low" when the winning track score is below 0.1 (on the 200-map tables the maps with wrong parameters averaged −0.18 for dots and 0.17 for atoms, against 0.36 / 0.45 for correct ones; all demos stay above 0.46); and `cnn_disagrees` as above. If the confidence is medium or low and a discriminating column exists, its sentence is appended.

**Why two criteria: score versus Bayesian evidence.** The decision stays with the feature-level score because it is robust to pump tones that no likelihood models and because it includes physics-motivated terms (visibility-weighted missing lines). The Laplace evidence (2.7) is computed *in addition* for every fitted hypothesis and shown as the P column and Bayes factor. When they agree (they do on every demo) that is a strong statement; when they disagree, you have learned something about the data.

### 2.6 Verification: hold-out columns (`fit.holdout_check`)

**Simple version.** Hide every third control column, refit the model on the remaining 70 %, then ask how many of the hidden features land within $3c_\ell$ of the predicted lines. A model that merely memorised noise fails this; a correct Hamiltonian predicts columns it never saw.

**Details.** Test columns are `cols[::3]` (≈ 33 %, the docs say "30 %"). The refit is *local only* (`do_global=False`, starting from the full-data solution), on the fit points restricted to coherent segments (tracks with ≥ 5 points) when there are at least 20 such points, so noise fragments do not dilute the score. The report carries `holdout = {n_test_points, inlier_frac, rms_MHz, param_shift_rel}` and `holdout_columns`. The verdict band's two left KPI tiles are the held-out RMS and the held-out explained fraction. Demo B scored a hold-out of 1.0; the real map 73 %.

### 2.7 Uncertainties: Jacobian σ, the 1.8 calibration, Laplace, Metropolis

**Jacobian σ (reported everywhere).** From the soft-L1 polish Jacobian on inliers (2.4). Reported as `value ± σ` after multiplication by `AgentConfig.sigma_calibration = 1.8`.

**Why 1.8 (coverage calibration, `scripts/coverage.py`).** On 40 random hidden maps per platform, the pulls $(\text{fit} - \text{truth})/\sigma$ should fall within ±1σ 68 % of the time and within ±2σ 95 %. Raw σ gave 1σ/2σ coverage of 0.42/0.60 (NV), 0.34/0.53 (dots), 0.62/0.72 (ions): about 1.5–2× too small, partly because the likelihood scale is set by the feature tolerance $c_\ell$ rather than by the actual scatter, partly because some fits sit on a wrong optimum (tails that no local σ can describe). The app multiplies σ by 1.8 and says so in the Posterior Check tab. Symmetry-equivalent parameters (NV field direction, Rydberg $\alpha$ and field_scale separately) are not scored.

**Laplace covariance (`fit.laplace_covariance`).** The Gauss–Newton inverse Hessian of $\sum_i w_i\,\rho(r_i/c)$ with soft-L1 $\rho$: $H = J^\top \text{diag}(w_i\,\rho''(u_i)/c^2)\,J$ with $\rho''(u) = 2(1+u^2)^{-3/2}$, so far-away features contribute no curvature, exactly as they contribute no information. $J$ is the central-difference Jacobian of the *signed* residual to the line each point is assigned to at θ (`fit.signed_residual_jacobian`; the assignment is frozen during differentiation so the kink of the nearest-line distance does not bite).

**Metropolis posterior (`fit.sample_posterior`).** Random-walk Metropolis in θ-space over the robust likelihood $\log L = -\sum_i w_i\,\rho(r_i/c_\ell)$ with a flat prior inside the bounds. Columns are thinned to 31 and the weights rescaled to stand in for the full cloud (so thinning does not widen the posterior). The chain starts at the mode of the subsampled likelihood (a few damped Gauss–Newton steps from the MAP, `fit.gauss_newton_recentre`), proposal covariance = 0.3 × Laplace covariance, capped at half the prior width per axis. 3 000 steps, 10 % burn-in, thinned to about 750 kept samples; flat features are dropped first for control-dependent hypotheses (the same rule as the agent). Output: samples in parameter units, sample σ, Laplace σ, MAP, mode, 16/50/84 % quantiles, acceptance rate. Demo A: acceptance 0.49, posterior σ / Laplace σ between 1.0 and 1.14, about 10 s (20 s under load).

### 2.8 Experimental design (`circuitsleuth/design.py`)

**Simple version.** Which sweep would shrink the error bars most? Pretend you measured a block of columns on one transition, compute how much the uncertainty ellipsoid would shrink, rank the candidates in bits.

**Fisher information (`design.fisher_information`).** For each explained point (inlier within $5\sigma_f$ of its nearest line) with Jacobian row $J_i = \partial L_\ell/\partial\theta$ at its column (finite differences in θ, `design._theta_jacobian`), $F_0 = \sum_i J_i^\top J_i/\sigma_f^2$, with $\sigma_f = c_\ell$ (the assumed line precision). $F_0^{-1}$ is the Laplace covariance; its normalised form is the correlation matrix (`design.correlation_matrix`), where the EJ–EC degeneracy shows up as an entry near ±1 (demo A: −0.997).

**Expected information gain.** A candidate is "sweep the columns $[x_\text{lo}, x_\text{hi}]$ looking for line $\ell$". With $F_S = \sum_{c\in S} J_{\ell c}^\top J_{\ell c}/\sigma_f^2$,

$$\Delta I = \tfrac12\left[\log\det(F_0 + F_S) - \log\det F_0\right]\times 1.4427\ \text{bits} \times \text{vis}(\ell).$$

This is the expected reduction in the entropy of a Gaussian posterior (Bayesian experimental design under the Laplace approximation), discounted by the line's nominal visibility (the chance it shows up). Blocks are 6 columns with 3-column steps, need at least 3 in-band unmeasured columns, must gain > 0.05 bits, and are selected greedily without overlap, top 5, by gain per column (cost). Each candidate also reports which free parameter's variance it reduces most. A per-column gain map per line feeds the gain chart. Demo A's top sweep is the 0-2/2ph line near the crossing, 3.3 bits.

**Discrimination (`design.discriminate`).** When a runner-up family exists: on a grid of 161 control values over the window extended by 25 % either side, compute for every line of the best model the distance to the nearest line of the rival, in units of $\sigma_f$; pick the column with the largest such distance, preferring a column inside the measured window with the line inside the band when that already gives ≥ 5σ. Reported as kind `"discriminate"` ("one column settles the family") and folded into the confidence text.

### 2.9 Autopilot (`circuitsleuth/loop.py`)

**Simple version.** Closed-loop spectroscopy on a digital twin: start with 40 % of the columns, identify, let the design pick the best block, "measure" it by copying those columns from the full map, refit, repeat.

**Details.** `loop.run_loop`: the twin is the full extracted point cloud; round 0 uses the first `max(8, 40 %)` columns; each later round restricts the family to the previous winner, narrows the bounds around the previous solution (`loop._narrow_bounds`: ±50 % for positive parameters, ±10 % for the period, ±0.15 for $d$), and uses one start. The design is computed with `measured_x` so only unmeasured columns are candidates; the first candidate with at least 3 new columns is taken. `loop.measure` is the one function a lab replaces with an instrument driver. Three rounds for demo A (precomputed, 237 s) added 23 features and took σ($E_{J,\max}$) from 3.4 % to 0.7 %, σ($E_C$) from 3.5 % to 0.8 %, σ(period) from 0.73 % to 0.03 %, family stable. NV/dot/ion/atom autopilots run live in seconds.

### 2.10 Synthetic data (`circuitsleuth/synth.py`)

**What is randomised** (`synth.sample_truth`): family (uniform over the platform), resonator with probability 0.5 (circuits), every parameter log-uniform in its bounds (linear for $d$, $D$, angles, offsets, $f_0$), band edges ($f_\text{lo}$ 0.05–1.5 GHz, $f_\text{hi}$ 7.5–13 GHz for circuits; per-platform `BANDS` otherwise), 0.6–2.5 flux periods in the window with a random offset, linewidth 0.01–0.08 GHz, noise 0.03–0.30, two-photon line present 60 % of the time, sign flipped 30 % of the time, up to 3 spurs (random frequency, amplitude 0.2–1, with ±sidebands 40 % of the time), and when a resonator is present its $f_r$ is placed inside the 0-1 line's range 60 % of the time (so avoided crossings are enriched to ~16 % of maps). **Validity re-draw:** up to 25 draws until the primary line is inside the band for at least 30 % of the sweep. **Rendering** (`synth.render`): Lorentzian lines of amplitude visibility × U(0.5, 1.2) with a smooth random envelope, up to 3 dropout segments per line, the "res" line visible half the time, a sinusoidal background drift, Gaussian noise. The blind tables use `nx = 81`, noise 0.03–0.1 ("harsh" tests up to 0.30).

### 2.11 Report

`agent.identify` returns one JSON-serialisable dictionary: `best` (hypothesis, params, σ, pretty strings, inlier_frac, RMS, hold-out, track_score, explained_dynamic, note, confidence), `hypotheses` (every candidate with score, track_score, log_evidence, posterior), `model_posterior`, `unexplained`, `next_measurements`, `lines`, `x_grid`, `inlier_mask`, `points`, `track_ids`, `holdout_columns`, `artefact_columns`, `design`, `derived`, `scqubits_code`, `runtime_s`. `explain.explain` turns it into a summary (Claude Opus 5.5 with structured output, template fallback), `derived.curves_csv` / `scqubits_code` export it, `ask.py` exposes it as tools.

---

## 3. Every output on the results page

The page order (app.py): verdict band → summary sentences → Hypotheses Tested → Evidence (map, residual strip, held-out panel, crossing zoom) → Physics and Spec Sheet → Next Measurements → Autopilot → Diagnostics (four tabs) → Ask qĤolmes → Details & Export.

### 3.1 Verdict band (hero)

- **Eyebrow:** `IDENTIFIED · N hypotheses · T s · confidence <level>`; N is the number of hypotheses in the report, T the runtime.
- **Title:** the winning hypothesis label (e.g. "Tunable Transmon + Readout Resonator").
- **Hamiltonian in KaTeX** (`theme.hamiltonian_tex`) with the fitted numbers substituted in amber: transmon $\hat H = 4E_C\hat n^2 - E_J\cos\hat\varphi$; tunable transmon the same with $E_J(\Phi) = E_{J,\max}\sqrt{\cos^2\pi\Phi + d^2\sin^2\pi\Phi}$ on a second line; fluxonium $4E_C\hat n^2 + \tfrac12E_L\hat\varphi^2 - E_J\cos(\hat\varphi - 2\pi\Phi)$; with a resonator $+ f_r\hat a^\dagger\hat a + g\,\hat n(\hat a + \hat a^\dagger)$; NV $D\hat S_z^2 + E(\hat S_x^2 - \hat S_y^2) + \gamma_e\mathbf B\cdot\hat{\mathbf S}$; dot $f = \sqrt{\varepsilon^2 + (2t_c)^2}$; ion $\omega_z = a\sqrt{V-V_0}$, $\omega_r = \sqrt{f_{r0}^2 - \omega_z^2/2}$; Rydberg $\Delta f = f_0 \pm \tfrac12 d_\text{fs} - \tfrac12\alpha E^2$.
- **KPI tiles** (`hero_bottom_html`): *held-out RMS* in MHz with the number of held-out points (the RMS of inlier residuals on the test columns; in-sample RMS if no hold-out ran); *held-out explained* (weighted inlier fraction on test columns); *margin to runner-up family* (the confidence gap, "—" if a single family); *Bayes factor* shown as $10^{N}$ "vs <runner-up family>", where $N$ is `log10_bayes_factor_vs_other_family` rounded to an integer.
- **Explained / unexplained bar:** `explained_dynamic`: of the points that belong to coherent segments (≥ 5 points), are not flat and not in artefact columns, the weighted fraction within $3c_\ell$ of a predicted line. This is the number the summary quotes ("accounts for X % of the coherent spectral features").
- **Right side:** an inline SVG schematic of the identified circuit (`ui.schematic`) with fitted values next to the components, and the Hamiltonian table `parameter | value ± σ`.
- **Note line** when $d > 0.95$ for a tunable transmon: effectively a fixed-frequency transmon.
- **Two sentences** (`theme.two_sentences` of the explainer summary) followed by the confidence text.

### 3.2 Hypotheses Tested (the race, `theme.race_html`)

One row per hypothesis in the order tried: name, a bar proportional to the score (relative to the best positive score), the **score** (= `track_score`), the **P** column (Bayesian posterior model probability from the Laplace evidences; shown as "<10⁻³" or "≈0" when tiny), and a status pill: queued / fitting / fitted / rejected / accepted / skipped. The winner is "accepted"; a resonator model that failed the acceptance rules is "rejected"; a flux-tunable family dropped by the `DYN_MIN_ABS` rule is "skipped". During a live run the active row shows a sparkline of the best-so-far search loss. A caption under the race gives the posterior aggregated by family and one sentence on the likelihood used.

**How P is computed (`fit.log_evidence`, `fit.model_posterior`).** For each fitted hypothesis, on one common point set (the flux-dependent features when there are enough, as the fits use):

$$\log Z \approx \log L(\theta^*) + \tfrac{k}{2}\log 2\pi + \tfrac12\log\det\Sigma - \log V_\text{prior},$$

with the **mixture likelihood** $L_i = (1-\epsilon)\,\mathcal N(r_i; 0, \sigma) + \epsilon/W$ ($\sigma = c_\ell$, $\epsilon = 0.1$, $W$ the band width): every feature is either on a predicted line (Gaussian) or background (uniform over the band). Unlike soft-L1 this is a proper, normalised likelihood, so an unexplained feature costs the same finite amount under every hypothesis and the Bayes factor is driven by what each model *explains*. $\theta^*$ is the fit refined by a few Gauss–Newton steps, $\Sigma$ the Gauss–Newton covariance weighted by inlier responsibilities $\gamma_i = (1-\epsilon)\mathcal N_i/L_i$, each σ capped at half the prior box (an unidentified direction cannot claim more volume than the prior), $V_\text{prior}$ the volume of the parameter box in θ (log-uniform priors on energies). The last three terms are the **Occam factor**: a model pays for prior volume it does not use. Posteriors are $P_m \propto \exp(\log Z_m)$ with equal prior odds; the family posterior sums bare and coupled variants; the Bayes factor is $\log_{10} Z_\text{best} - \log_{10} Z_\text{other}$ against the best hypothesis of a *different* family. Demo A: $P(\text{tunable + res}) = 1.0$, $\log_{10}\text{BF}$ vs fluxonium = 306 ("a model that explains one more line wins by hundreds"). Demo B: bare fluxonium 0.993 vs fluxonium + resonator 0.007: the Occam factor at work ("a needless resonator loses a few nats").

### 3.3 Evidence map (`plots.map_figure`)

The normalised map (single-hue blue heat scale, clipped at 10σ) with: **explained features as filled green discs, unexplained as hollow red rings** (by `inlier_mask`); the fitted transitions with fixed colours per label so a line never changes colour between hypotheses (0→1 orange and thick, 0→2 amber, 0→3 purple, 0→2 two-photon pink dashed, 1→2 green dotted, resonator white dash-dot); **flat lines as faint red horizontals**; **numbered amber hatched "look here" boxes** whose numbers match the recommendation cards; **amber diamonds** for operating points from the target-$f_{01}$ tool. The legend sits below the plot. Hover gives control value and frequency.

### 3.4 Residual strip (`evidence.residual_figure`)

Data minus model in MHz for every detected feature against its nearest predicted line, on the map's x range. Points are coloured by the line they were assigned to; **hollow = held-out column**. The shaded blue band is the inlier threshold $\pm 3c_\ell$ (±36 MHz for circuits); dashed lines are ±RMS over inliers with an annotation "RMS x MHz (n inliers)"; the y axis is clipped to $\pm\max(5\,\text{RMS}, 60\,\text{MHz})$ and points beyond are clamped to the edge as triangles and counted in the top-right annotation ("n unexplained points beyond ±clip MHz" or "all points within ±clip MHz"). How to read it: a good fit shows a flat cloud centred on zero inside the band with structure-free scatter; a systematic bow would mean a wrong model; the triangles are the spurs.

### 3.5 Held-out panel (`evidence.holdout_figure`)

The map with the never-fitted columns shaded, the fitted lines, and the held-out features as **hollow squares: green within 3σ of a predicted line, red outside**; the rest as dim dots. Title: "held-out columns: X % of features within 3σ, RMS Y MHz". The columns are `report["holdout_columns"]` (every third column of the coherent fit set).

### 3.6 Avoided-crossing zoom (`evidence.crossing_zoom_figure`)

Only when the winner is resonator-coupled and the bare 0-1 line crosses $f_r$ inside the window. For up to two crossings (ranked by how many features sit in the window): a data crop ±0.12 control units wide and $\pm 4g$ in frequency; the **bare 0→1 line (grey dashed)**, **$f_r$ (white dashed)**, the **dressed 0→1 and dressed resonator branches** (NaN inserted where the overlap-labelled branches swap so no vertical segment is drawn), explained features green, unexplained red, and an **amber bracket at the crossing with "2g ≈ … ± … MHz"** (the ± is $2\sigma_g$). If the actual gap between the branches at that flux differs from $2g$ by more than 5 % a second line reads "gap here … MHz": the exact splitting is $2g\,|\langle 0|\hat n|1\rangle|$, which is not exactly $2g$ because the charge matrix element of a transmon is not 1. Say: "the two branches repel by twice the coupling; that bracket is the measurement of $g$".

### 3.7 Physics panel

- **Circuits (`plots.potential_figure`):** the potential $U(\varphi) = -E_J\cos\varphi$ (transmon; $E_J(\Phi)$ for a tunable one) or $U = \tfrac12E_L\varphi^2 - E_J\cos(\varphi - 2\pi\Phi)$ (fluxonium) with the lowest four levels drawn between their classical turning points (where $U \le E_n$), transition arrows 0→1, 1→2 (and 0→2 when it fits) labelled in GHz, and an **external-flux slider** 0–0.5 $\Phi_0$ that redraws it. For a fluxonium at $\Phi = \tfrac12$ the double well and the small 0→1 splitting are visible.
- **NV (`platform_panels.nv_levels_figure`):** spin-1 levels of the orientation most aligned with the field versus field magnitude, ODMR transitions marked.
- **Dot (`dqd_figure`):** the two hybridised charge states versus detuning with the $2t_c$ anticrossing.
- **Ion (`ion_modes_figure`):** axial ($\propto\sqrt{V-V_0}$) and radial secular frequencies versus endcap voltage, $V_0$ marked.
- **Atom (`stark_figure`):** the two Stark parabolas with their common vertex at the stray-field null.

### 3.8 Spec sheet (`derived.derived_quantities`, cards Qubit / Control axis / Readout)

Circuits:
- $f_{01}$ at $\Phi = 0$ (upper sweet spot) and $\Phi = \tfrac12$ (lower sweet spot); **anharmonicity** $\alpha = f_{12} - f_{01}$ in MHz (negative for a transmon, roughly $-E_C$); $E_J/E_C$ (≥ 50 keeps charge dispersion small); for fixed transmons the **exact charge dispersion** of $f_{01}$ (difference between $n_g = 0$ and $n_g = \tfrac12$, from the charge-basis Hamiltonian, in kHz); for fluxonium $E_L/E_J$ and the plasma frequency $\sqrt{8E_LE_C}$; tuning range $f_{01}(0) - f_{01}(\tfrac12)$ (set by $d$).
- **Max flux sensitivity** $|\partial f_{01}/\partial\Phi|$ over $\Phi\in[0, \tfrac12]$ in GHz/$\Phi_0$ and in GHz per control unit; **sweet spots** in control units inside the measured window ($x = \text{offset} + (k + \{0, \tfrac12\})\times\text{period}$).
- **Flux-noise $T_\varphi$** for 1/f flux noise of amplitude $A = 1$ and $5\,\mu\Phi_0$ at the steepest point (first-order Ithier et al. 2005 form):
  $$T_\varphi = \frac{1}{\sqrt{2|\ln(2\pi f_\text{ir} t)|}\;A\;2\pi\,|\partial f_{01}/\partial\Phi|},\quad f_\text{ir} = 1\,\text{Hz},\ t = 10\,\mu\text{s},$$
  with the note that at a sweet spot the first-order term vanishes.
- With a resonator: detuning $\Delta = f_{01} - f_r$ at each sweet spot, $g/\Delta$ with a **"NOT dispersive here" flag when $g/\Delta > 0.2$**, the **dispersive shift $\chi$ by exact diagonalisation**, $\chi = (E_{11} - E_{10}) - (E_{01} - E_{00})$ from the dressed spectrum (valid even near the crossing, unlike the perturbative $g^2/\Delta$), and the **Purcell factor** $(\Delta/g)^2$ when dispersive ($T_{1,\text{Purcell}} = (\Delta/g)^2/\kappa$).
- A **target-$f_{01}$ tool** (`derived.control_for_frequency`): type a frequency, get the control values where the fitted 0-1 line crosses it (drawn as amber diamonds).

NV: $D$ in MHz (2870 MHz at room temperature), $2E$ (line separation at zero field), gauss per control unit, the axial field fraction $\mathbf B\cdot\hat n_k/|\mathbf B|$ for each of the four orientations, zero-field control value. Dot: $2t_c$ in GHz and µeV (4.1357 µeV per GHz), lever arm (GHz and µeV per unit), zero-detuning control value. Ion: axial and radial secular frequencies at the window centre, axial range over the sweep, anisotropy $\omega_r/\omega_z$ (> $\sqrt3$ keeps the axial mode lowest), $V_0$ ("extrapolated; do not go there"), MHz per $\sqrt{V}$. Atom: stray-field null, V/cm per unit, $\alpha$ for each component, tensor part $\alpha_{3/2} - \alpha_{1/2}$, zero-field splitting, and the **Stark curvatures $\alpha\kappa^2$ in control units, which is what the map actually determines**.

### 3.9 Next Measurements

Left, "Gaps and ambiguities": cards numbered to match the map badges: **sweep** cards (`look_here`: a predicted line runs through ≥ 50 % of its in-band columns, at least 3, with nothing within $3c_\ell$ under it; contiguous runs reported as boxes, pruned by `agent.prune_look_here` if they cover > 20 % of the map or are weak lines with large boxes; 0-1 first), **extend the band** (↕; a line predicted entirely outside the band but within 3 GHz of it), **one column settles the family** (≠; the discrimination result with its σ separation), **constrain parameter** (σ; any parameter with relative σ > 10 %, with a physics hint: "EJ is degenerate with EC along constant f01; the 0-2/2ph line breaks the degeneracy", etc.). Below them, **"What the data cannot separate": the correlation matrix** heat map (blue −1 to red +1, numbers in cells) from the Laplace covariance.

Right, **"Expected information gain · best sweeps"**: the ranked table (rank badge, line, control range, frequency range, **bits**, "tightens <parameter> −x % var"), then the **gain chart**: expected information gain of one more column, per transition, versus control value, with amber bands for the three best blocks. Caption: "½ log of the determinant ratio of the Fisher information with and without it, discounted by the line's visibility and the sweep cost".

### 3.10 Autopilot

Two buttons: **Run Autopilot (3 rounds, live)** and **Replay Saved Autopilot** (`artifacts/loop_A.json` for demo A). A **Round slider**; the map for that round shows only the features measured so far, the fit lines, and the next block as a look-here box, titled "round r · n features from m columns · hypothesis · next sweep: line (bits)". Right: the **σ-versus-round chart** (relative 1σ per parameter, log scale, round 0 = initial partial sweep), the parameter table with σ/value, and the hidden truth for demos.

### 3.11 Diagnostics tabs

- **Unexplained Features:** one card per unexplained track (≥ 60 % of its points off any predicted line, ≥ 3 points): frequency range, point count, and the interpretation: flat → "likely pump/LO leakage, a mixer spur, or a fixed-frequency neighbour"; flat *and crossed by the qubit line with no visible splitting* → "a pump/LO/mixer tone or an uncoupled mode (g below the linewidth), not a coupled resonator"; flux-dependent → "candidate neighbouring tunable mode, higher transition, or sideband of a qubit line".
- **Posterior Check:** the explanatory paragraph (σ from the Jacobian × 1.8), the **Sample the Posterior** button, then the **posterior figure** (`design_panel.posterior_figure`: faint blue MCMC samples of the most interesting pair, EJ(max)–EC when it exists; the **1σ Laplace ellipse** in dashed amber centred on the sampled mode; the MAP as an amber diamond) and a table `parameter | posterior σ | Laplace σ | ratio`, with acceptance rate, kept samples and seconds. The ratio near 1 says the Laplace (Gaussian) approximation is adequate; the ellipse's tilt is the EJ–EC valley.
- **Truth and Design:** for demos, the **Reveal Hidden Truth** toggle gives `parameter | fitted ± σ | truth | within 2σ` (offset compared modulo the period, NV angles through their symmetry-equivalent projections), the truth caption (family, spurs), and the **pull plot** (`evidence.pull_figure`: $(\text{fit} - \text{truth})/\sigma$ per parameter, green ±1σ band, amber ±2σ, red beyond 3σ, clamped at ±6 and listed). With design values entered in the sidebar, a fit-vs-design deviation table.
- **Run Log:** the timestamped live log of a run (features, proposals, symmetry calibration, fits, relabel events, escalation, revise, hold-out, warnings).

### 3.12 Ask qĤolmes (`ui/ask_panel.py`, `circuitsleuth/ask.py`)

Three suggested questions (circuits: "Bias point for 5 GHz with the least flux noise?", "Why was the resonator model accepted?" or "Which alternative hypotheses were tested, and why did they lose?", "What should I measure next and why?") and a chat input. Claude Opus 5.5 answers via the Anthropic beta tool runner with nine **pure tools over the report** (`ask.TOOLS`): `report_summary`, `derived_quantities`, `control_for_frequency(target_GHz)` (with local slopes $|\partial f_{01}/\partial x|$ and per $\Phi_0$ so the least-sensitive bias can be chosen), `predict_transitions(control_value)`, `next_measurements`, `hypotheses` (with the acceptance rules spelled out), `unexplained_features`, `parameter_uncertainties` (with the correlation matrix and strong correlations), `scqubits_code`. The system prompt forbids computing or recalling any number; every value must come from a tool result, rounded to 3–4 significant figures with its unit; questions outside the model (T1, temperature, κ) get "the spectroscopy model does not cover it". The grey caption under each answer lists the tools called. Adaptive thinking, effort low, prompt caching; 3–7 s and under 2 ¢ per answer; a 10-question × 3-report stress test had 0 errors. The tools are unit-tested without the API (`tests/test_ask.py`).

### 3.13 Details & Export

Download the JSON report, the fitted curves as CSV (`derived.curves_csv`: control value and every line's frequency on the data grid), the annotated map as standalone HTML, and a **scqubits snippet** (`derived.scqubits_code`) that rebuilds the fitted circuit (`scq.Transmon(...)`, `scq.TunableTransmon(...)`, `scq.Fluxonium(...)`, plus `scq.Oscillator` and `HilbertSpace.add_interaction` when coupled; plain-formula comments for the other platforms). "This Session" lists every run.

---

## 4. The physics platforms, from scratch

### 4.1 Superconducting circuits

**Transmon.** A Josephson junction (energy $-E_J\cos\hat\varphi$) shunted by a large capacitor (charging energy $E_C = e^2/2C$, kinetic term $4E_C\hat n^2$). The code (`models.transmon_hamiltonians`) builds $\hat H = 4E_C(\hat n - n_g)^2 - E_J\cos\hat\varphi$ in the **charge basis** (states $|n\rangle$, $n = -n_\text{cut}\ldots n_\text{cut}$ with $n_\text{cut} = 20$ in the fit; $\cos\hat\varphi$ is the hopping matrix $\tfrac12(|n\rangle\langle n+1| + \text{h.c.})$) and diagonalises it. For $E_J/E_C \gg 1$ it is a weakly anharmonic oscillator: $f_{01} \approx \sqrt{8E_JE_C} - E_C$, anharmonicity $f_{12} - f_{01} \approx -E_C$, and the sensitivity to the offset charge $n_g$ (charge dispersion) is exponentially small, which is the whole point of the transmon. Visible lines in two-tone spectroscopy: 0-1 (bright), 0-2 via two photons at $f_{02}/2 = f_{01} - E_C/2$ (needs higher drive power), 1-2 if the first excited state is thermally populated, and 0-2/0-3 directly if the drive is strong. **qĤolmes fits $E_J, E_C$.** Degeneracy: with the 0-1 line alone only $f_{01}$ is observable, so $E_J$ and $E_C$ are unidentifiable along the valley $\sqrt{8E_JE_C} - E_C = \text{const}$; the two-photon (or 1-2) line fixes $E_C$. The report says so and asks for the 0-2/2ph line.

**Tunable transmon (SQUID).** Replace the junction by two junctions in a loop (a SQUID, superconducting quantum interference device). The effective Josephson energy depends on the flux $\Phi$ threading the loop, in units of the flux quantum $\Phi_0$:

$$E_J(\Phi) = E_{J,\max}\sqrt{\cos^2(\pi\Phi) + d^2\sin^2(\pi\Phi)},\qquad d = \frac{E_{J1} - E_{J2}}{E_{J1} + E_{J2}}.$$

$f_{01}$ is maximal at $\Phi = 0$ and minimal at $\Phi = \tfrac12$ (where $E_J = d\,E_{J,\max}$); both are **sweet spots**, where $\partial f_{01}/\partial\Phi = 0$ and first-order flux-noise dephasing vanishes. The spectrum is periodic in $\Phi$ with period 1 and mirror-symmetric about 0 and ½, which is what the symmetry calibration exploits. **qĤolmes fits $E_{J,\max}, E_C, d$ plus period and offset** (control units per $\Phi_0$, and the control value of $\Phi = 0$). The flux noise $T_\varphi$ in the spec sheet uses the fitted slope.

**Fluxonium.** A small junction shunted by a large inductance (a chain of junctions), $\hat H = 4E_C\hat n^2 + \tfrac12E_L\hat\varphi^2 - E_J\cos(\hat\varphi - 2\pi\Phi)$ with inductive energy $E_L$. Now the phase is not periodic; the potential is a parabola with cosine corrugations, and at $\Phi = \tfrac12$ it is a double well whose two lowest states are split by a small tunnelling energy (hundreds of MHz down to tens of MHz for "heavy" fluxonium with $E_L \ll E_J$), which is why fluxonium 0-1 frequencies can be below 1 GHz at half flux while 0-2 sits several GHz up. The code (`models.fluxonium_hamiltonians`) uses the **harmonic-oscillator basis** of the LC part (cutoff 60 states, 40 in the global search; $\varphi_\text{osc} = (8E_C/E_L)^{1/4}$) and builds $e^{i\hat\varphi}$ by diagonalising $\hat\varphi$; the flux enters as a phase factor on $e^{i\hat\varphi}$. **qĤolmes fits $E_J, E_C, E_L$ plus period and offset.** The characteristic spectral signature versus a tunable transmon: deep minima at half flux, strong 0-2 lines, and plateaus near $\Phi = 0$ for heavy devices; when the band hides the flux-dependent part, a heavy fluxonium's plateaus look like a fixed transmon, which is the main circuit failure mode. Validation: all three circuit Hamiltonians agree with scqubits to $10^{-6}$ GHz (`tests/test_models.py`, `atol=1e-6`).

**Readout resonator and dispersive readout.** A linear resonator (frequency $f_r$, operators $\hat a, \hat a^\dagger$) coupled capacitively to the qubit's charge: $\hat H = \hat H_q + f_r\hat a^\dagger\hat a + g\,\hat n(\hat a + \hat a^\dagger)$. The code (`models.dressed_energies`) takes the lowest $k_q = 5$ qubit eigenstates, expresses $\hat n$ in that basis, builds the $k_q\times k_r$ ($k_r = 3$ photon states) product Hamiltonian per flux value and diagonalises it. Each dressed eigenstate is **labelled by the bare state it overlaps most** (`labels`), so "0-1" means the dressed transition between the states that are mostly $|g, 0\rangle$ and $|e, 0\rangle$, and "res" is the dressed $|g, 1\rangle - |g, 0\rangle$ transition. When the bare $f_{01}$ crosses $f_r$ the two states hybridise and the lines repel: an **avoided crossing** with minimum splitting $2g|\langle 0|\hat n|1\rangle|$; far from it, in the **dispersive regime** $g \ll |\Delta| = |f_{01} - f_r|$, the resonator frequency depends on the qubit state by $\pm\chi$ (approximately $g^2/\Delta$ for a two-level system, with transmon corrections; qĤolmes computes it exactly from the dressed levels) and the qubit frequency is Lamb-shifted. That state-dependent shift is how superconducting qubits are read out. The Purcell effect is qubit decay through the resonator, suppressed by $(\Delta/g)^2$. **qĤolmes fits $f_r, g$** when a coupled model is accepted, seeded at the cluster of unexplained points that an avoided crossing leaves behind.

### 4.2 NV centres in diamond

A nitrogen–vacancy (NV) centre is a nitrogen atom next to a missing carbon in the diamond lattice; its ground state is an electron **spin-1** ($m_s = 0, \pm1$). The Hamiltonian per orientation is

$$\hat H = D\hat S_z^2 + E(\hat S_x^2 - \hat S_y^2) + \gamma_e\,\mathbf B\cdot\hat{\mathbf S},$$

with the **zero-field splitting** $D \approx 2.870$ GHz (temperature dependent), the **strain/electric-field splitting** $E$ (mixes $m_s = \pm1$, splits the zero-field line by $2E$), and the **Zeeman** term with $\gamma_e = 2.8025$ MHz/G. The NV axis can point along any of the **four ⟨111⟩ directions** of the diamond lattice (`models.NV_AXES`); in an ensemble all four are present, each sees the same field at a different angle, so a field of a given direction produces up to **eight lines** (NV$k\pm$, $k = 1..4$). **ODMR** (optically detected magnetic resonance) is the measurement: shine green light, sweep a microwave, watch the red fluorescence dip when the microwave is resonant with $m_s = 0 \to \pm1$. The map is fluorescence versus microwave frequency and coil current. `models.nv_lines` decomposes the field into axial and transverse components per orientation and diagonalises the 3×3 Hamiltonian. **qĤolmes fits $D, E, \theta, \phi$ (field direction in the crystal frame) plus field_scale (gauss per control unit) and $x_0$ (zero-field current).** Degeneracy: the diamond's **tetrahedral symmetry** means several $(\theta, \phi)$ give identical spectra (permuting which orientation is which); the right invariant is the sorted set of $|\mathbf B\cdot\hat n_k|$ (`models.nv_projections`), which is what the truth table and the blind metric compare. Residual failure: field direction 5–8 % off when two orientations are nearly degenerate.

### 4.3 Semiconductor double quantum dots

Two electrostatically defined puddles of electrons in a semiconductor, with one excess electron that can sit on the left or the right dot: a **charge qubit**. The two charge states are detuned by $\varepsilon$ (set by a gate voltage) and coupled by **tunnel coupling** $t_c$. Diagonalising the 2×2 Hamiltonian gives a hyperbola,

$$f = \sqrt{\varepsilon^2 + (2t_c)^2},\qquad \varepsilon = \text{lever\_arm}\times(V - V_0),$$

with minimum $2t_c$ at zero detuning (the charge degeneracy point, $V_0$) and asymptotic slope equal to the **lever arm** (how much energy a volt on the gate buys; the gate couples capacitively so the conversion is a fraction of $e$ per volt, here in GHz per control unit, or µeV per unit using $h\times 1\,\text{GHz} = 4.1357\,\mu$eV). A two-photon copy of the hyperbola at $f/2$ may appear. **qĤolmes fits $t_c$ plus lever_arm and $x_0$.** Failure mode: steep V's (lever arm 14–30) with few visible points among spurs; fixed mostly by the geometric prior (`estimate_dqd_init`) and the trimmed polish. Label ambiguity: the 0-1 hyperbola read as its two-photon copy doubles $t_c$ and the lever arm; the relabel step catches it.

### 4.4 Trapped ions

A single ion held in a linear Paul trap oscillates harmonically along the trap axis ($\omega_z$, the **axial secular frequency**, set by the DC endcap voltage) and transversely ($\omega_r$, the **radial** frequency, set mainly by the RF pseudopotential). A narrow laser transition acquires **motional sidebands** at the carrier ± the mode frequencies (and second-order sidebands at ±2ω). Sideband spectroscopy maps the laser detuning (MHz) versus endcap voltage. The code's model (`models.ion_lines`): $\omega_z = \text{volt\_scale}\sqrt{V - V_0}$ (harmonic axial confinement from a DC potential, so $\omega_z^2 \propto V$; $V_0$ is where axial confinement vanishes), $\omega_r^2 = f_{rad0}^2 - \omega_z^2/2$ (the DC field that confines axially anti-confines radially), lines at 0, ±ω_z, ±ω_r, ±2ω_z with visibilities 1, 0.7, 0.5, 0.25. **qĤolmes fits $f_{rad0}$ plus volt_scale and $x_0$.** Spec sheet: secular frequencies at the window centre, anisotropy $\omega_r/\omega_z$ (> √3 keeps the axial mode the lowest, relevant for laser cooling and for the ion-chain geometry), "$V_0$: do not go there". Failure: radial frequency when the radial sidebands are faint; the first-order/second-order axial ambiguity (volt_scale halved) is fixed by the relabel step.

### 4.5 Rydberg atoms

An atom excited to a high principal quantum number $n$ (a Rydberg state) is huge and extremely polarisable. In an electric field $E$ its energy shifts quadratically (the **quadratic Stark shift**) $\Delta f = -\tfrac12\alpha E^2$, with **polarisability** $\alpha$ scaling as $n^7$. An $nD$ state has two **fine-structure / $|m_j|$ components** ($|m_j| = \tfrac12, \tfrac32$) with different $\alpha$ (the difference is the tensor polarisability), split at zero field by $d_\text{fs}$. A Stark map is laser detuning (MHz) versus the voltage on a field plate; the field is $E = \kappa(V - V_0)$ where $V_0$ is the **stray-field null** (the plate voltage that cancels the residual field: the common vertex of the parabolas, and the right place to bias). The code: `models.rydberg_lines`, $f_{1/2} = f_0 + d_\text{fs}/2 - \tfrac12\alpha_1E^2$, $f_{3/2} = f_0 - d_\text{fs}/2 - \tfrac12\alpha_2E^2$. **qĤolmes fits $\alpha_1, \alpha_2, d_\text{fs}, f_0$ plus field_scale $\kappa$ and $x_0$.** Degeneracy: only the **curvatures $\alpha\kappa^2$** are observable from the map (a stiffer plate with a softer atom gives the same parabola), so the spec sheet reports the curvatures in control units and says to fix field_scale under Known Parameters to get $\alpha$ itself; the blind metric judges the curvatures. Failure: the second component lost among spurs.

### 4.6 One table to memorise

| platform | fitted physics | fitted calibration | what is degenerate / unobservable |
|---|---|---|---|
| transmon | $E_J, E_C$ | none | $E_J$–$E_C$ along constant $f_{01}$ without the two-photon (or 1-2) line |
| tunable transmon | $E_{J,\max}, E_C, d$ (+ $f_r, g$) | period, offset | same valley, broken by 0-2/2ph; offset modulo period |
| fluxonium | $E_J, E_C, E_L$ (+ $f_r, g$) | period, offset | $E_J$–$E_C$ at fixed half-flux splitting when only 0-1 is seen; heavy fluxonium out of band looks like a fixed transmon |
| NV ensemble | $D, E, \theta, \phi$ | field_scale, $x_0$ | field direction up to tetrahedral symmetry |
| double dot | $t_c$ | lever_arm, $x_0$ | hyperbola vs its two-photon copy (×2) |
| trapped ion | $f_{rad0}$ | volt_scale, $x_0$ | first- vs second-order axial sidebands (×2) |
| Rydberg atom | $\alpha_1, \alpha_2, d_\text{fs}, f_0$ | field_scale, $x_0$ | only $\alpha\kappa^2$ observable |

---

## 5. The CS / statistics toolbox (glossary)

**Robust statistics, M-estimators.** Least squares minimises $\sum r_i^2$ and is destroyed by one far outlier. An M-estimator minimises $\sum\rho(r_i)$ with a $\rho$ that grows slower than $r^2$. Three used here: **Huber** (quadratic inside, linear outside; not used directly but the conceptual parent), **soft-L1** $\rho(u) = 2(\sqrt{1+u^2}-1)$ (smooth Huber-like; the polish, the posterior likelihood, the Laplace covariance), and **Geman–McClure** $\rho(d) = d^2/(d^2+c^2)$ (redescending and *bounded*: an outlier's influence goes to zero; the global search). Bounded losses make the landscape multimodal, which is why a global optimiser is needed in that stage and why the scale $c$ is annealed.

**Differential evolution (DE).** A population-based, gradient-free global optimiser (Storn & Price 1997): mutate by adding a scaled difference of two population members to a third, crossover with the current member, keep the better. Strengths: bounds, multimodality, parallel evaluation, no derivatives of an eigensolve needed. Weaknesses: slow convergence near the optimum (hence the local polish), and cost ∝ popsize × dimension × generations × (one batched eigensolve). Used in `fit._fit_once` with a Sobol initialisation, population carried between annealing stages, `tol = 1e-3` for early stopping, `workers` processes.

**Least squares, Jacobians, Gauss–Newton.** With residual vector $r(\theta)$ and Jacobian $J = \partial r/\partial\theta$, Gauss–Newton approximates the Hessian of $\tfrac12\|r\|^2$ by $J^\top J$ and steps $\delta\theta = -(J^\top J)^{-1}J^\top r$. `scipy.optimize.least_squares` (trust-region reflective, with a robust loss re-weighting) is the polish; `fit.gauss_newton_recentre` is a hand-rolled damped Gauss–Newton on the robust loss; the covariance $s^2(J^\top J)^+$ is the parameter uncertainty. Finite-difference Jacobians are used in `design.py` and for the signed residuals because the forward model is an eigensolve.

**Laplace approximation.** Approximate a peaked distribution by a Gaussian centred on its mode with covariance = inverse Hessian of the negative log. Used three times: parameter uncertainties (σ, correlation matrix, the ellipse in the posterior plot), the model evidence (`fit.log_evidence`), and the MCMC proposal. It is exact for a quadratic log-likelihood and fails on multimodal or banana-shaped posteriors, which is why the Metropolis sampler exists as a check.

**Fisher information and the Cramér–Rao bound.** For Gaussian measurement noise σ_f on observables $L(\theta)$, the Fisher information is $F = \sum_i \nabla L_i\nabla L_i^\top/\sigma_f^2$; the Cramér–Rao bound says no unbiased estimator has covariance smaller than $F^{-1}$. qĤolmes uses $F_0^{-1}$ as the current covariance (`design.fisher_information`) and $\det F$ (the inverse volume of the uncertainty ellipsoid) as the quantity a new measurement should increase.

**Expected information gain, Bayesian experimental design.** The entropy of a Gaussian is $\tfrac12\log\det\Sigma$ + const, so adding data with information $F_S$ reduces entropy by $\tfrac12[\log\det(F_0+F_S) - \log\det F_0]$ (the D-optimality criterion; Lindley's expected information gain under the Laplace approximation). In bits after dividing by $\ln 2$ (×1.4427). Visibility weighting turns "information if the line appears" into an expectation. The discrimination column uses a Box–Hill-style criterion: separate rival models where their predictions differ most relative to σ_f.

**Bayesian model comparison.** The marginal likelihood (evidence) $Z = \int L(\theta)\pi(\theta)\,d\theta$ averages the fit quality over the prior; it automatically penalises models with more or less-constrained parameters (the **Occam factor** $\tfrac{k}{2}\log 2\pi + \tfrac12\log\det\Sigma - \log V_\text{prior}$ in the Laplace form: the ratio of posterior to prior volume). The **Bayes factor** $Z_1/Z_2$ is the evidence ratio; $\log_{10}\text{BF} > 2$ is "decisive" on the Jeffreys scale; the **posterior model probability** with equal prior odds is $Z_m/\sum Z$. In qĤolmes the likelihood is the Gaussian-plus-uniform-background mixture so that outliers are modelled, not just down-weighted.

**Metropolis MCMC.** Markov chain Monte Carlo: propose $\theta' = \theta + \text{noise}$, accept with probability $\min(1, L(\theta')/L(\theta))$ (flat prior); the chain's visited points are samples from the posterior. **Acceptance rate**: fraction of proposals accepted; around 0.2–0.5 is healthy for a random walk (demo A: 0.49). **Burn-in**: the first 10 % of steps discarded while the chain forgets its start. **Thinning**: keep every $k$-th step to reduce autocorrelation and memory (to ~750 samples). The proposal covariance is 0.3 × the Laplace covariance, so steps are the right size and orientation (along the valley).

**Coverage calibration.** Fit many synthetic maps with known truth, compute pulls, check that 68 % lie within 1σ. Under-coverage means σ is too small; qĤolmes measured 1.5–2× and corrects by 1.8. This is an *empirical* calibration honest about model misspecification and wrong optima, which no analytic covariance captures.

**Hold-out validation.** Train on part of the data, test on the rest (cross-validation with a single fold). Here on *columns* (control values) rather than random points, because the goal is to predict unseen control values, and because random point hold-out would leak information through neighbouring points on the same column.

**CNN basics.** A convolutional layer slides small filters (3×3 here) over the image and produces feature maps; stacking layers with pooling builds translation-tolerant detectors of edges → line segments → shapes (a bend, a periodic wave, an avoided crossing). A 64×64 thumbnail classifier learns coarse morphology: "sinusoid with deep minima" (fluxonium), "shallow sinusoid" (tunable transmon), "horizontal lines" (fixed transmon), "a kink at one frequency" (visible crossing). It cannot learn what leaves no trace, which is why the raw resonator label failed. ReLU = rectified linear unit; max-pooling keeps the strongest activation in each block; adaptive average pooling makes the head size independent of the input size; dropout randomly zeroes units during training to regularise.

**Data augmentation.** Flipping and rolling along the flux axis teaches the network the period/offset invariance of the physics without generating more maps; gain jitter teaches brightness invariance.

**Multiprocessing.** DE evaluates the population in parallel (`workers` processes, `updating="deferred"`, which is why `Objective` must be picklable). The blind tables and coverage runs use `multiprocessing.Pool` / `ProcessPoolExecutor` with one map per process and BLAS threads pinned to 1 inside workers (otherwise 28 processes × 20 BLAS threads thrash). Known pitfall recorded in STATE: never launch multiprocessing from a `python -` stdin script.

**REST API, CI, Docker.** `api.py` (FastAPI) exposes `/health`, `/demos`, `/identify`, `/identify/demo` with pydantic validation (422 before any fitting); `tests/test_api.py` has 5 tests; the Dockerfile ships CPU-only torch with ports 8501 (UI) and 8000 (API); `.github/workflows/ci.yml` runs the quick suites (physics vs scqubits, extraction, platforms, lab features, evidence, posterior, API) on every push. The slow suites (`tests/test_fit.py` ≈ 6 min, `tests/test_agent.py` ≈ 30 min, 4 end-to-end scenarios) run locally.

**Synthetic data and why blind tables matter.** The hand-picked demos were chosen; the blind tables draw hundreds of random devices from the generator and run the *whole* pipeline (`bigmachine/run_blind.py`), judging family and "every parameter within 5 %" with symmetry-aware metrics (NV field vector up to tetrahedral symmetry, Rydberg curvatures, offsets modulo period, absolute tolerances for $D, E$, angles, $x_0$, $d$). They found the ×2 label ambiguities, the DYN skip-rule failure, the steep-V dot failure. "Blind tables are EVALUATIONS of the physics fitter (nothing accumulates); only the CNN proposer is trained."

**Big-machine training (`bigmachine/overnight.py`).** A detached, resumable, deadline-aware worker: measure generation and training speed (~2 min), spend ~40 % of the budget generating 2 000-map shards in parallel, choose the number of epochs that fits the remaining time, train with a one-cycle schedule, after every epoch keep the best held-out model as `classifier_best.pt/json`, kill training 10 minutes before the deadline. `run_train.py bench/start/status/stop` is the manual version; `run_blind.py` the benchmark. Thumbnails are 16 KB each (200k maps ≈ 3.3 GB in RAM).

---

## 6. Results and honest limitations

### 6.1 The numbers (quote these, nothing else)

**Real data: Nguyen et al., PRX 9, 041041 (2019), fluxonium device A**, Fig. 2 digitised from the vector PDF (`scripts/prep_nguyen2019.py`). The right half of the published map is too noisy for any feature to survive extraction; only the 0-1 line from $\Phi = 0$ to about 0.3 and fragments of 0-2 are available. With the paper's flux axis (period held at 1) and `c_local = 0.03` GHz (the figure has 12.5 MHz pixels and ~100 MHz line width):

| parameter | qĤolmes | paper |
|---|---|---|
| $E_J$ | 3.05 ± 0.21 GHz | 3.0 |
| $E_C$ | 0.86 ± 0.07 GHz | 0.84 |
| $E_L$ | 0.95 ± 0.07 GHz | 1.0 |

All within 1σ; explained 80 % of coherent features (42 % before the artefact-column handling and the wider band), hold-out 73 %. Earlier extractor settings gave 3.23/0.80/0.98 and 2.83/0.93/0.93; every run landed within 10 % of the paper. Without the axis prior the period is unconstrained by this half-map and the report asks for a sweep near $\Phi = \tfrac12$ below 1.2 GHz, which is where the paper's inset lives. How to say it: "marginal data, no tuning, within the paper's values; the error bars reflect the figure's pixel size, and we say that the digitised half-map moves the answer at the 5–10 % level with extraction choices."

**Hand-picked demos (normal preset, 4 workers).** A: tunable transmon + 7 GHz resonator + 5.1 GHz spur: $E_{J,\max}$ 25.01 ± 0.10 (truth 25), $E_C$ 0.2998 (0.3), $d$ 0.300 (0.3), $f_r$ 7.002 (7), $g$ 0.0996 (0.1), period 1.300 (1.3), offset 0.200; the spur reported as an unexplained flat line the qubit crosses without splitting; 426 s. B: fluxonium $E_J$ 5.998 (6), $E_C$ 1.000 (1), $E_L$ 0.600 (0.6), hold-out 1.0, a trial resonator rejected; 683 s. C: fixed transmon + 7.1 GHz spur, live in ~80 s (fast): family right, $E_J$ 14.2 ± 0.3 / $E_C$ 0.300 ± 0.007 against 15 / 0.28 (the end-to-end test gives 15.023 / 0.280). NV F: $D$ 2.86996 (2.870), $E$ 0.0041 (0.004), field_scale 119.9 (120), $x_0$ 0.1007 (0.1), seconds. Dot H: $t_c$ 2.50006, lever 8.99996, $x_0$ 0.15001. Ion J: $f_{rad0}$ 3.2009, volt_scale 1.7995, $x_0$ −1.2996. Atom L: curvatures within 1 %, $d_\text{fs}$ 11.95 (12), $x_0$ 0.1207 (0.12).

**Blind tables, 200 random hidden devices per platform, 20-core machine (`bigmachine/run_blind.py`, within-5 % metric):**

| platform | family | family + every parameter within 5 % | mean runtime | what still fails |
|---|---|---|---|---|
| trapped ion | 200/200 | 86 % (74 % before relabel) | 1 s | radial frequency when radial sidebands are faint |
| NV centre | 200/200 | 72 % | 4 s | field direction 5–8 % off (near-degenerate orientations) |
| quantum dot | 200/200 | 78 % (58 % before the geometric prior) | 1 s | steep V with < ~20 visible points among spurs |
| Rydberg atom | 200/200 | 63 % (57 % before) | 1 s | second Stark component lost among spurs; curvatures 5–8 % off |

**Superconducting circuits:** normal preset, 120 maps on a 32-core machine: family 87/120 (72 %), all parameters within 5 % 35 %, 27 min per map per core; fast preset (the live-demo preset), 100 maps: 64 % / 26 %, 8 min. Old code, 200 maps: 70 % / 37 %. Confusions are almost entirely one-directional: fluxonium → transmon (+ resonator) 12/36, tunable → fixed transmon 8/38 ("X + res → X" counts as family-correct). Fixed-transmon parameters are within 5 % in only 11/43 because $E_J$–$E_C$ is unidentifiable without the two-photon line (the metric counts that as a miss although $f_{01}$ is right). Fluxonium parameters with the fast preset are the weak spot (3/10 within 5 % on a fluxonium-only table; the normal preset recovers 16/20 on the 120-map table). A 10-map fluxonium table went from 8/10 to 10/10 families with the `DYN_MIN_ABS` fix.

**Small laptop blind runs (`demo.py`, 8–10 maps, harsh generator):** circuits family 7/10 (all parameters < 5 % in 2/10; clean tunable circuits to ≈ 1 % including the period); NV 7/8, dot 8/8, ion 7/8, atom 6/8 within tolerance, the misses having fewer than about twenty usable features and flagged low confidence.

**Proposer:** 600 000 maps (570 000 train), 28 epochs, trained overnight on a 20-core machine: family accuracy 94.3 %, visible-crossing accuracy 88.5 % (the previous model, 190 000 maps / 30 epochs, had 93.6 % / 87.4 %) (the raw "resonator present" label sat at chance).

**Uncertainty calibration:** raw 1σ coverage 0.34–0.62 instead of 0.68 → σ × 1.8; posterior/Laplace σ ratio 1.0–1.14 on demo A.

**Design and autopilot (demo A):** $E_{J,\max}$–$E_C$ correlation −0.997; top sweep 0-2/2ph near the crossing, 3.3 bits; three autopilot rounds (23 extra features): σ($E_{J,\max}$) 3.4 % → 0.7 %, σ(period) 0.73 % → 0.03 %.

**Runtime (M3 laptop, 4 workers):** circuits ~2 min per map with the fast preset, 5–7 min normal; NV/dot/ion/atom seconds (ion/atom 6–7 s in the browser). Design computation a fraction of a second to about a second; posterior sampling ~10 s; Claude summary ~12 s (template instantly); Ask answers 3–7 s.

### 6.2 What fails and why (say these before being asked)

1. **Heavy fluxonium read as a fixed transmon (+ resonator)** when the band hides the flux-dependent part: the plateaus look flat, the symmetry calibration can return a wrong period (1.68 for a true 0.93 on surprise seed 7), and the bogus solution genuinely scores higher (0.36 vs 0.23 for the polished true solution, because the spur at 4.08 GHz gets credited to the resonator). The CNN escalation now flags these with "cnn disagrees"; it does not fix them.
2. **Fixed-transmon $E_J$/$E_C$ without the two-photon line:** only $f_{01}$ is observable; the report says so (large stated σ, the "constrain EC" card, the −1 correlation entry).
3. **Fast-preset fluxonium parameters:** family right, $E_J$/$E_C$ 25–55 % off while $E_L$ is right on some surprise seeds; the normal preset is much better. Say "the live demo uses the fast preset for time; the saved results use normal".
4. **Few features:** below about 20 usable features the fit is unreliable on every platform; the `poorly_explained` flag catches most such cases.
5. **Stated σ are optimistic on real data** even after the 1.8 factor, because the likelihood scale is a tolerance, not the measured scatter.
6. **Only three circuit families**; no 0–π qubit, flux qubit, coupled transmons; no TLS (two-level-system defect) census.
7. **One real map.** Everything else is synthetic, and the generator was written by the same people who wrote the fitter.
8. **Fixed-transmon maps are inherently ambiguous against spur families**: a spur with sidebands can mimic a transmon ladder (the demo C spur was changed from 6.2 GHz with sidebands to a single weak 7.1 GHz tone for that reason).

---

## 7. Likely judge questions with model answers

**1. Why not just train a neural net end to end?**
Because the numbers have to be right on data the network has never seen, and a real map contains things no training set anticipates: a pump at an arbitrary frequency, a streak from a glitchy column, a line width three times the pixel. A regressor would silently absorb those into its answer. We use the network for what it is good at, recognising morphology, and the physics fit for what it is good at, numbers with uncertainties. The proposer's label change proves the point: the "resonator present" label was unlearnable (chance) because a far-detuned resonator leaves no trace; a network trained on it would have hallucinated resonators. Also, on a published map we had never fitted, the physics fit landed within the paper's values with no retraining.

**2. How do you know the physics is right?**
The three circuit Hamiltonians are checked against scqubits to $10^{-6}$ GHz in `tests/test_models.py` (transmon, tunable transmon, fluxonium, and the dressed qubit–resonator levels), and the recovered scqubits snippet lets anyone rebuild the fit. For the other platforms the models are the textbook ones (spin-1 NV Hamiltonian, the charge-qubit hyperbola, Paul-trap secular frequencies, quadratic Stark shift) and hand-picked maps come back to better than 1 %. The real test is the Nguyen 2019 fluxonium: $E_J$, $E_C$, $E_L$ within 1σ of the paper.

**3. What happens with real noisy data?**
We have one real map and it was hard: the right half of the published figure had no extractable features, 72 of 169 features sat in nine vertical streak columns, and the line width was ~100 MHz. Three things made it work: per-column noise estimation (noisy columns do not flood the point cloud), the artefact-column rule (burst columns stay in the fit but do not count as unexplained), and a wider inlier band for the known pixel size. The result moves at the 5–10 % level with extraction choices, and we say so. The next step is collecting raw maps from labs on all five platforms.

**4. How is this different from curve fitting?**
Curve fitting assumes you know the model, the assignment of points to curves, and that all points are signal. Here we do not know the family (model selection), we do not know which point belongs to which transition (the nearest-line loss makes the assignment implicit and lets it change during the search), we do not know which points are physics at all (bounded loss, flat-line removal, unexplained-feature list), and we do not know the control-axis calibration (period/offset or gauss-per-unit are fitted). Then we verify on held-out columns and tell you what to measure next. A curve fitter does none of that.

**5. What is the Bayes factor telling me?**
It is the ratio of marginal likelihoods of the best hypothesis and the best hypothesis of a different family: how much more probable the data are under one model than the other, after integrating over each model's parameters. Integrating over parameters is what penalises a needless resonator (the Occam factor: it pays for prior volume it does not use; on demo B the bare fluxonium gets 0.993 and the resonator variant 0.007), while a model that explains one more line wins by hundreds of orders of magnitude (demo A: $10^{306}$ versus fluxonium). We compute it with a Laplace approximation under a Gaussian-plus-uniform-background likelihood, so an unexplained feature costs the same under every model. The decision itself uses the feature-level score; the Bayes factor is an independent second opinion, and they agree on every demo.

**6. Why differential evolution?**
The forward model is an eigensolve, so gradients are expensive, and with a bounded robust loss the landscape is multimodal: every spur and every wrong line assignment is a local minimum. DE is gradient-free, respects bounds, handles multimodality with a population, and parallelises trivially. We make it work harder by annealing the robust scale (coarse to fine) while carrying the population between stages, by fitting the 0-1 line first on the brightest point per column, and by multi-start. Then a Gauss–Newton polish gives the precision DE lacks.

**7. What is the time complexity / runtime?**
Per loss evaluation: one batched eigensolve, $O(n_\text{cols}\times N^3)$ with $N = 41$ (charge basis) or 40–60 (fluxonium) or $5\times3 = 15$ with a resonator, on at most 31 subsampled columns. DE does popsize × dimension × generations such evaluations per stage (15 × 5 × 60-ish), times three stages, times the number of candidates, starts, and hypotheses. On the laptop that is about 2 minutes per circuit map with the fast preset and 5–7 minutes with normal, in 4 processes; NV/dot/ion/atom have closed-form or 3×3 models and run in seconds. Everything after the fit (hold-out, evidence, design, derived quantities) is under a few seconds; posterior sampling is ~10 s on demand.

**8. What did the LLM do?**
Two things, both grounded. First, `explain.py`: Claude Opus 5.5 with structured output rewrites the JSON report into a short summary with caveats and next steps under a system prompt that forbids inventing numbers; a template produces the same fields without a key. Second, "Ask qĤolmes": Claude answers lab questions by calling nine deterministic tools over the report (bias points with sensitivities, predicted transitions, uncertainties, why a model won, what to measure next); every number in an answer comes from a tool call and the tools used are listed. Claude Code was the development environment. No model decides the physics and no model produces a parameter value.

**9. What would you do with more time?**
Real data from labs on all five platforms with their own fits as ground truth, and tune the extractor and priors on that instead of our generator. Connect the "look here" suggestion to an instrument driver so the autopilot runs on a real fridge. Add 0–π and coupled-transmon families, trapped-ion multi-ion spectra. A ridge-following extractor for very noisy maps. A defect census for small avoided crossings along the qubit line. Retrain the proposer on more maps with the crossing target.

**10. What is the biggest weakness?**
Heavy fluxoniums whose tuning sits outside the measured band get read as fixed transmons, sometimes with a spurious resonator: 12 of 36 fluxoniums in the 120-map table. The data genuinely look like flat lines, the bogus solution scores higher, and only more band or a sweep at half flux settles it. We flag it (the CNN disagreement warning and the discriminating column) but we do not fix it. Second: everything except one map is synthetic.

**11. How do you handle spurious lines?**
Four layers. Extraction detects flux-independent lines by frequency occupancy (present in ≥ 60 % of columns) and removes them before fitting any flux-dependent hypothesis. The Geman–McClure loss is bounded, so a spur left in costs a fixed amount instead of bending the fit. The score counts tracks, not pixels, and flat tracks at half weight, so an 80-point spur cannot buy a model. And a resonator is only accepted if it improves the fit on flux-dependent features, because explaining one more flat line is something any tone does. What is left is reported as unexplained with a physical interpretation: a flat line the qubit crosses with no splitting is a pump or an uncoupled mode, not a resonator.

**12. How does the autopilot pick the next sweep?**
From the Fisher information of the current fit. Each explained feature contributes the outer product of the derivatives of its line frequency with respect to the parameters, divided by the line precision squared. A candidate sweep is a block of six control columns on one transition; its expected information gain is half the log of the determinant ratio of the Fisher matrix with and without those columns, in bits, discounted by the line's nominal visibility and ranked by gain per column. The top non-overlapping block becomes the next measurement; the twin hands over those columns, the fit is redone with the previous solution as a prior, and the loop repeats. On demo A three rounds took the relative uncertainty of $E_{J,\max}$ from 3.4 % to 0.7 %.

**13. What's a 1/f flux-noise $T_\varphi$?**
Flux noise in SQUID loops has a 1/f spectrum with amplitude of order microflux quanta at 1 Hz. To first order the qubit frequency fluctuates by the slope $\partial f_{01}/\partial\Phi$ times the flux noise, and for 1/f noise the Gaussian decay time is $T_\varphi = 1/(\sqrt{2|\ln(2\pi f_\text{ir}t)|}\,A\,2\pi|\partial f_{01}/\partial\Phi|)$ with an infrared cutoff $f_\text{ir}$ and the measurement time $t$ (the Ithier 2005 form). We evaluate it at the steepest flux point for $A = 1$ and $5\,\mu\Phi_0$ with $f_\text{ir} = 1$ Hz and $t = 10\,\mu$s; at a sweet spot the first-order term vanishes, so the number is a worst case that tells you how much you lose by biasing off the sweet spot. It is a derived quantity of the fitted slope, not a measurement of coherence.

**14. Why is the uncertainty multiplied by 1.8? Is that a fudge factor?**
It is an empirical calibration. On 40 random synthetic maps per platform, the Jacobian σ covered the truth at 1σ only 34–62 % of the time instead of 68 %, for two identifiable reasons: the likelihood scale is a feature tolerance rather than the measured scatter, and some fits sit on a wrong optimum. A single factor cannot fix the second, so we also offer the MCMC posterior on demand, which samples the actual robust likelihood and shows the EJ–EC valley directly, and whose σ agrees with the Laplace σ to within 14 % on demo A. We tell the user both numbers.

**15. The CNN is trained on synthetic data. Does it transfer?**
It only needs to transfer well enough to rank families and seed the search, and its mistakes cost time, not correctness: the physics fit runs on every proposed family and the score decides. Its prior weight in the score is 0.05 because it was confidently wrong on resonator-crossing maps. When it disagrees strongly with the fit we refit its favourite with double effort and, if that still loses, lower the confidence and say so.

**16. Why Geman–McClure in the search but soft-L1 in the polish?**
Geman–McClure is bounded, so spurs cannot pull the global search, but its gradient dies far from the lines, so a local solver would stall. Soft-L1 is robust but still has a linear tail, so a Gauss–Newton solver can walk a line that is 50–100 MHz off onto the data. Hence the two-stage polish with f_scale $3c$ then $c$.

**17. How do you handle the EJ–EC degeneracy?**
Four ways. The staged search keeps $E_J$ and $E_C$ bounds open after the 0-1-only fit, because the 0-1 line alone cannot separate them. After the global search we sweep 25 $E_C$ values along the constant-$f_{01}$ valley and polish from the best. The relabel step tries the two-photon versus thermal reading, which differs by exactly a factor two in $E_C$. And if the data still do not pin it, we report a large σ, a −1 correlation entry, a "constrain EC" card and an information-gain sweep on the two-photon line; the design module on demo A ranks exactly that sweep first.

**18. What is the P column versus the score?**
The score is a physics-motivated feature-level criterion: fraction of track weight explained, minus a penalty for predicted lines with nothing under them, minus 0.02 per parameter, minus a precision term. P is the Bayesian posterior model probability from Laplace evidences under a proper mixture likelihood. The score decides because it is robust to things no likelihood models; P is shown next to it as an independent check and as the Bayes factor in the verdict band.

**19. How do you avoid overfitting with a flexible model and a resonator?**
Parameter-count penalty in the score, the Occam factor in the evidence, three acceptance rules for a resonator (score gain > 0.05, $g > 2\sigma_g$, gain on flux-dependent features ≥ 0.02), and hold-out verification on columns the local refit never saw.

**20. Why hold out columns rather than random points?**
Because the thing we want to predict is the spectrum at control values we have not measured, and because random points on a column would leak information through their neighbours on the same line. Every third column is hidden, the model is refitted locally on the rest, and the hidden features are scored.

**21. How does the symmetry calibration work?**
Every flux-tunable spectrum is mirror-symmetric about $\Phi = 0$ and $\Phi = \tfrac12$. We slide a mirror across the point cloud and measure the median distance between the cloud and its reflection; the minima are symmetry axes; their spacing (from the autocorrelation) is half a period; and whether the lowest line is a maximum or a minimum at an axis tells which sweet spot it is. It only narrows the bounds for a second candidate; the wide search always runs too. It can fail on heavy fluxoniums whose plateaus look flat.

**22. How many parameters does the search have?**
Two to seven per hypothesis: physics parameters (2 for a fixed transmon, 3 for a tunable transmon or fluxonium, 4 for NV or Rydberg, 1 for a dot or ion) plus 2 for the resonator when coupled plus 2 for the control calibration. Energies are searched in log space.

**23. What does "explained 80 %" mean exactly?**
Of the detected features that belong to coherent line segments of at least five points, are not flux-independent spurs and are not in artefact columns, the brightness-weighted fraction lying within three times the tolerance (36 MHz for circuits) of a predicted transition. The remainder is listed under Unexplained Features with an interpretation.

**24. Why five platforms, and how hard was adding one?**
Because the problem is the same everywhere: a spectrum versus a control knob explained by a small Hamiltonian, with spurs and uncalibrated axes. Adding a platform means writing its transition frequencies in `models.py` and declaring its control axis and scales; extraction, robust fitting, hold-out, unexplained features, design and the UI are shared. Trapped ions and Rydberg atoms were added late on the second night, and each came with a label ambiguity we found through blind tables and fixed with the generalised relabel step.

**25. What does the 2g bracket on the crossing zoom mean?**
At the flux where the bare qubit frequency equals the resonator frequency, the two dressed states repel by twice the coupling, $2g$ times the charge matrix element. The bracket is drawn between the two fitted dressed branches at that flux and labelled $2g \pm 2\sigma_g$; when the actual gap differs from $2g$ by more than 5 % (because the matrix element of a transmon is not exactly one) we also print the actual gap. It is the one place where $g$ is literally visible in the data.

**26. What would break it in a real lab?**
Very noisy maps where no ridge survives the 4σ rule (we would need ridge following); maps with many TLS avoided crossings along the qubit line (not modelled); circuits outside the three families; a drifting control axis within one map; maps where the frequency axis is not monotonic (we sort it). Also any case with fewer than about 20 usable features; the report flags those as low confidence.

**27. How do you test it?**
Quick suites run in CI on every push: Hamiltonians against scqubits, synthetic generation and extraction, the four non-circuit platforms, lab features including the design module, the autopilot, the posterior sampler, the evidence, the Ask tools (without the API), and the REST API. Slow suites run locally: `tests/test_fit.py` (7 tests, 6 min) and `tests/test_agent.py` (4 end-to-end scenarios, ~30 min: tunable transmon through a resonator plus a spur, fluxonium, fixed transmon plus LO leakage, a band too narrow for the two-photon line). Then the blind tables at scale, which are the real test of the whole pipeline.

**28. How do you make the fit robust to the UI or the LLM failing?**
`agent._emit` swallows every exception from progress callbacks so a UI error can never abort a fit; the explainer and Ask panel fall back to a template or a muted caption; the design, derived and evidence steps are wrapped so they cannot fail a run; the API returns 422 for bad input and JSON for engine errors.

**29. Where is the data-set bias?**
The generator and the fitter were written by the same team, so the blind tables measure robustness to the nuisances we thought of (spurs with sidebands, dropouts, drift, sign flips, noise up to 30 %). They do not measure robustness to the nuisances real labs have and we did not think of. That is why the honest claim is "validated on synthetic data at scale and on one published map", and why the next step is real maps.

**30. Why Streamlit and Plotly?**
Speed of development in 36 hours and interactivity for free (hover for frequency and control value, zoom). The engine is independent of the UI: the same `agent.identify` runs from the CLI batch mode (`batch.py`, one CSV row per device), from the REST API, and in the tests.

---

### Appendix: numbers and names to have on the tip of the tongue

- Peak threshold 4σ (3σ with a confident neighbour); inlier band $3c_\ell$ = 36 MHz (circuits), 4.5 MHz (NV), 30 kHz (ions), 0.45 MHz (atoms).
- Loss: Geman–McClure $d^2/(d^2+c^2)$; annealed $c$: 0.32 → 0.08 → max(0.02, 0.018) GHz for circuits; polish soft-L1 at $3c_\ell$ then $c_\ell = 0.012$ GHz.
- Score: track fraction − 0.3 × missing-line − 0.02 × params − RMS/300 (MHz, capped at 30) + 0.05 × (p_CNN − 1/3).
- Resonator acceptance: gain > 0.05, $g > 2\sigma_g$, flux-dependent gain ≥ 0.02; $f_r$ seeded from the histogram of unexplained dynamic points.
- Confidence: gap > 0.15 high, > 0.05 medium; low if score < 0.1 (poorly explained) or the CNN disagrees after escalation.
- Hold-out: every third column; σ × 1.8; posterior 3 000 Metropolis steps, 10 % burn-in, ~750 kept.
- Evidence: Gaussian (σ = $c_\ell$) + uniform background (ε = 0.1); log-uniform priors; Laplace.
- Design: $\tfrac12\log\det$ ratio × 1.4427 × visibility, blocks of 6 columns, top 5.
- CNN: 4 conv layers, width 2, 64×64, 600k maps, 28 epochs, family 94.3 %, crossing 88.5 %.
- Real data: 3.05 ± 0.21 / 0.86 ± 0.07 / 0.95 ± 0.07 GHz vs 3.0 / 0.84 / 1.0.
- Blind: 200/200 families on ions, NV, dots, atoms; 86 / 72 / 78 / 63 % within 5 %; circuits 72 % family (normal), 35 % within 5 %.
