# LPO lifecycle figure review and revision plan

## Scope and overall assessment

This review covers the 28 lifecycle PNGs in:

`/home/cwp29/tmp/mpas/crf_on/final/organisation_lpt/publication/2026-08-03`

The figures contain a coherent scientific story, but they are not yet publication-ready. The largest limitations are analytical rather than cosmetic:

1. Figure 5 mixes incompatible lifecycle definitions and samples.
2. The cloud-depth/UTLS coupling central to the proposed mechanism is largely absent from the figures.
3. Most two-dimensional experiment differences lack uncertainty information.
4. The current cloud maps do not cleanly separate LPO-related cloud from ambient tropical high cloud.
5. The filtered phase diagnostics identify wave phase but do not yet provide phase-conditioned dynamical composites.

The overall purpose of the analysis is to determine how independently imposed zonal-wind and temperature changes affect the lifecycle and organization of tracked LPOs, including:

- The transition between active convection, stratiform cloud, and persistent nonprecipitating high cloud.
- Convective depth and interaction with the UTLS thermal environment.
- The role of organized equatorial-wave variability.
- Whether u-nudging and T-nudging produce distinguishable responses.

The LPO envelope should be retained as contextual guidance and for appropriate local scalar sampling, but it should not be treated as a hard spatial boundary for the larger cloud system.

## Finalized conventions

- Use one TTL reference layer from 150 to 70 hPa, approximately 14 to 18.5 km.
- Define TTL-reaching cloud as cloud-top pressure at or below 150 hPa.
- Use native MPAS levels for vertical composites and sample pressure only for plotting.
- Cache absolute surface and vertical values without subtracting either simulation climatology.
- Plot absolute composites separately from each nudged experiment minus control.
- Sum precipitation over the trailing 24 hours and spatially integrate it inside the LPO envelope.
- Call this quantity integrated rain volume and display it in km3.
- Label lifecycle axes as lag from precipitation peak.
- Use concise figure titles and avoid repeating subplot labels or annotations.
- Label absolute and experiment-minus-control panels as Composite and Difference.
- Retain cloud-class contours in Figure 6.
- Show the matched random phase baseline in the third column of Figures 25–28.
- Move phase coherence, amplitude, and sample support to Figure 40.

## What the current figures show

### Convective and cloud lifecycle

- The clearest lifecycle signal is a transition from active convection near the precipitation peak to stratiform cloud and then persistent nonprecipitating high cloud.
- OLR broadly follows the same evolution.
- Because lag zero uses integrated rain volume, its maximum at lag zero is a coordinate definition rather than an independent result.
- The term **nonprecipitating high cloud** should remain the diagnostic name, with **anvil/cirriform proxy** added where useful. It should not be relabelled unconditionally as cirrus or anvil.

### UTLS environment

- The u-nudged simulation has a substantially colder, less statically stable, and more easterly 100-hPa environment than control.
- T-nudge is generally intermediate.
- This establishes the intended imposed environmental separation.
- The figures do not yet demonstrate that convection or cloud penetrates more frequently into the affected layer, so the central thermal-pathway mechanism remains incomplete.

### Cloud spatial context

- Stratiform cloud dominates much of the central LPO region.
- Nonprecipitating high cloud occupies a broader region and persists later in the lifecycle.
- Active deep cloud is comparatively patchy.
- Background tropical high cloud contributes strongly outside the LPO. The existing maps therefore cannot distinguish system-related cloud from ambient occurrence.

### Unfiltered composites

- The surface composites show a wet, high-PW, low-OLR environment around tracked systems.
- Vertical composites show coherent tropospheric warm, moist, and ascending structures, together with UTLS temperature changes.
- Some T-nudge anomalies appear visually stronger, but this should not be interpreted as an attributable response without experiment-minus-control maps and bootstrap information.

### Filtered wave composites

- Kelvin, equatorial Rossby, and eastward intraseasonal bands show the clearest lifecycle coherence.
- TD-type results are substantially noisier, particularly in the vertical and phase diagnostics.
- Raw filtered amplitudes are not directly comparable between bands because the filters have different frequency-wavenumber bandwidths.

### Group velocity and averaging sensitivity

- Presenting \(C_{gx}\), \(C_{gz}\), and \(C_{gz}/C_{gx}\) together is appropriate and follows Lim and Son.
- The ratio is presently dominated by speckling where \(C_{gx}\) is small, which obscures the physically relevant UTLS signal.
- The longitude-width sensitivity figure shows that point sampling is noisy and that the 2.5-degree, 3.75-degree, and 5-degree averages are fairly similar in control.
- This supports finite-area averaging, but the sensitivity test currently does not establish that the same choice is appropriate for both nudged responses.

## Clean-up

These items should not require changing the scientific definition of the diagnostics.

- [ ] **Assign every output an unambiguous figure identifier.** Several independent files currently share Figure 7 or Figure 10. Use unique numbers or explicit subfigure/supplement labels. Ensure that references to current figure numbers in these instructions do not confuse you after this change.

- [ ] Remove the "event-equal" wording used throughout the figures: it is scientifically obvious that LPOs are given equal weighting. It is only be necessary to state exceptions to this standard weighting choice. Similarly, all events should be used for composites and statistics rather than restricting to "complete" events. Some later instructions may use this wording; overrule them.

- [x] **Repair units and scaling in Figure 4.** Use explicit scaled units for precipitation, cloud area, and OLR. Name the precipitation quantity integrated rain volume.

- [ ] Composites should use all available LPO tracks and extend +-72h from the precipitation maximum. Do not restrict to using only LPOs that cover the full 72h window. 

- [x] Use native levels for all vertical composites with a sampled log-pressure plotting coordinate.

- [ ] **Standardize vertical-section contours.** Use common contour levels across simulations, solid positive contours, dashed negative contours, and a heavier zero contour. State contour values and the pressure-velocity sign convention.

- [ ] **Use shared colour bars where the numerical scale is shared.** In particular, Figures 13 and 14 should use one colour bar per shaded variable rather than repeating it for every simulation row.

- [ ] Ensure that colour bars are consistent with those used in other diagnostic scripts.

- [ ] **Correct the cloud-occurrence scales in Figure 10.** Total-high-cloud occurrence should use a 0–1 scale. Individual-class panels can retain a narrower shared scale. 

- [ ] **Clarify the modal-cloud encoding in Figure 9.** Add a colourbar showing the three class colours, with three different hues for each tercile of modal agreement. 

- [ ] **Mask or fade poorly supported modal classifications.** Avoid presenting a confident-looking class where event support or class agreement is weak.

- [x] Restrict Figure 6 to the LPO region and retain the 10% cloud-class contours.

- [ ] **Repair phase-figure axes.** Plot phase probability over its full 0–1 range. Normalized amplitude should be on a separate axis to probabilities and fractions, with appropriate units and a brief comment on the normalisation calculation.

- [x] **Use numeric phase labels only.** Do not label phases as active or suppressed.

- [ ] Change the existing spatial averaging from  a fixed 10 degree S - 10 degrees N average to a +/- 10 degree average centred on the LPO centroid latitude.
These items require recomputation, revised sampling, or additional diagnostics. Each item below now includes a recommended implementation, bounded alternatives, and the decision needed to narrow the work.

## Refinements
### 1. Redesign the lifecycle-timing analysis

**Problem:** Figure 5 extrema use full tracks, while adjacent composites use the complete +/-72-h sample. The full-track OLR minimum lies outside +/-72 h for 26/43 control systems and 26/37 u-nudged systems; cloud-top and TTL extrema have the same problem.

- [ ] Produce a peak-centred +/-72-h timing analysis from events complete over that interval.
- [x] Reject extrema at either window boundary.
- [ ] Produce a separate normalized whole-lifetime analysis for developing, mature, and dissipating stages.
- [ ] Plot event distributions with counts, medians, interquartile ranges, and bootstrap intervals.

Lifecycle staging using system size and vertical extent has precedent in [Futyan and Del Genio (2007)](https://www.giss.nasa.gov/pubs/abs/fu09100s.html). LPO duration, maximum size, and propagation distributions are presented explicitly by [Kerns and Chen (2020)](https://orca.atmos.washington.edu/publications_files/Kerns_Chen_2020_JGR_LPT.pdf).

**Recommended implementation**

Create two products:

1. **Peak-centred timing figure:** use all events. Apply the same three-time-step centred smoothing to every event variable, find its event-level extremum inside the window, and reject boundary extrema. Plot one horizontal violin/box distribution per simulation for active-cloud maximum, stratiform-cloud maximum, nonprecipitating-high-cloud maximum, OLR minimum, deepest cloud top, and maximum TTL-reaching fraction. Add the median, interquartile range, 95% event-bootstrap interval, event count and number of rejected extrema.
2. **Normalized-lifetime figure:** map every full track from 0% at genesis to 100% at termination, interpolate each event to 20 equal lifetime bins, and average events equally. Plot precipitation and the three cloud-class areas or fractions against normalized lifetime. Mark the diagnosed composite precipitation maximum.

Replace the current single “post-rain persistence” number with a quantity with an explicit interpretation:
- **Anvil delay:** lag from the precipitation peak to the post-peak maximum in nonprecipitating high-cloud area.

### 2. Add the missing cloud-depth/UTLS pathway

- [ ] **Plot the cached median cloud-top pressure.**
- [ ] **Plot a deep-tail cloud-top metric**, such as the existing low-percentile cloud-top pressure.
- [ ] **Plot the cached fraction of cloud/events reaching the TTL.**
- [x] **Plot native-level lag-pressure cloud-fraction composites.**
- [ ] **Pair cloud-depth measures with 100-hPa temperature and lower-stratospheric static-stability anomalies.**
- [ ] **Show experiment-minus-control responses and their event-bootstrap uncertainty.**

This is the most important missing mechanism figure. It would test whether the nudged UTLS environment changes the frequency or depth of convective penetration, as motivated by [Sakaeda et al. (2020)](https://agupubs.onlinelibrary.wiley.com/doi/full/10.1029/2020JD033196) and the TTL temperature–ascent–cirrus relationship documented by [Virts and Wallace (2014)](https://journals.ametsoc.org/view/journals/atsc/71/3/jas-d-13-0178.1.xml).

Radiative-heating composites are out of scope because the required variables are unavailable.

**Recommended implementation**

Add two figures:

1. **Cloud-depth lifecycle metrics:** use median CTP, P10 CTP, and the CTP ≤150 hPa fraction. Show absolute means above. Show experiment-minus-control responses below.
2. **Lag–pressure mechanism figure:** use absolute control fields above. Show raw experiment-minus-control responses below. Shade the shared 150–70 hPa TTL.

For cloud fraction, calculate the event composite first and then bootstrap complete events. Use TTL-reaching fraction as the primary penetration metric. Use median and deep-tail cloud top to distinguish population-wide changes from changes confined to the deepest tail.
### 3. Add attributable experiment contrasts

- [ ] **Add u-nudge minus control and T-nudge minus control maps** for surface, vertical, cloud-occurrence, and group-velocity diagnostics.
- [ ] **Overlay or mask using event-bootstrap significance/robustness.**
- [ ] **Use independent-sample event-equal contrasts.** Treat the simulation track sets as unpaired samples.
- [ ] **Retain absolute composites as context**.

**Recommended implementation**

For every retained two-dimensional diagnostic, create one companion response figure with two rows: u-nudge minus control and T-nudge minus control. Keep the existing variables as columns. Use a common symmetric colour range for the two rows, chosen from the rounded 98th percentile of the absolute values across both contrasts so that neither experiment controls the scale.

Reuse the cached 1,000 equal-event bootstrap sequences. At each cell, add sparse black stippling where the two-sided 95% percentile interval excludes zero. Report the fraction of the displayed domain meeting this criterion in the caption. Use the pointwise result in the main figure and a Benjamini–Hochberg false-discovery-rate mask as a supplementary sensitivity.

For line composites, show the response line, its 95% bootstrap ribbon, and a thin zero line. For contour variables, use the same numerical contour levels in both experiment rows.

**Decision to narrow:** first produce response figures for cloud depth, cloud occurrence, unfiltered vertical structure, and group velocity. Review those results before commissioning surface and filtered-wave response figures.

### 4. Separate LPO-related cloud from ambient high cloud

- [ ] **Calculate excess cloud-class occurrence** relative to a time- and latitude-matched climatology.
- [ ] **Show raw occurrence and excess occurrence as complementary products.**
- [ ] **Show u-minus-control and T-minus-control excess-occurrence maps with event-bootstrap uncertainty.**
- [ ] **Assess whether spatial smoothing or coarser aggregation improves the modal-class stability.**

**Recommended implementation**

Use a time- and latitude-matched climatology. For every event, lag, cloud class, and relative-latitude row:

1. Form the binary class-occurrence field.
2. Define background occurrence as the mean over both far-field sectors where 60 degrees <= |relative longitude| <= 150 degrees at the same model time and absolute latitude.
3. Subtract that background from the occurrence field before averaging events equally.

The default plotted cloud domain should be +/-30 degrees relative longitude and +/-20 degrees relative latitude. Test 45–90-degree and 90–150-degree far-field sectors to determine whether the conclusion depends on the background definition.

Retain the raw modal-class figure for physical interpretation. Calculate the modal class event by event at each grid cell and display cells where total high-cloud occurrence is at least 0.20 and modal agreement is at least 0.50. Show agreement continuously through saturation. 

Add a separate occurrence-response figure for total, active, stratiform, and nonprecipitating high cloud. It should show u-nudge minus control and T-nudge minus control in background-relative occurrence, with the event-bootstrap stippling defined above.

### 5. Implement phase-conditioned filtered composites

- [ ] **Retain LPO lag as the coordinate for filtered lifecycle evolution.**
- [ ] **Add Rios-Berrios-style phase–pressure composites** of filtered temperature, humidity, pressure velocity, and winds for sufficiently strong wave events.
- [ ] **Restrict wave-phase conditioning to filtered composites.**
- [ ] **Use consistent phase conventions and reverse the plotted phase direction where required for westward waves.**
- [ ] **Apply standard positive/negative/zero contour styling.**

The phase–pressure presentation should follow [Rios-Berrios et al. (2023)](https://journals.ametsoc.org/view/journals/clim/36/9/JCLI-D-22-0422.1.xml).

**Recommended implementation**

Use the existing phase definition based on normalized filtered precipitation and its tendency. Retain normalized phase-space amplitude > 1 as the strong-wave threshold.

For each wave band, create a phase–pressure figure with eight wave phases on the horizontal axis and pressure on the vertical axis. Use three columns:

1. Temperature shading with pressure-velocity contours.
2. Specific-humidity shading with relative-humidity contours.
3. Zonal-wind shading with the zero-wind contour emphasized.

Use rows for control, u-nudge minus control, and T-nudge minus control. Pool strong-wave LPO samples over the full -72 to +72-h lifecycle and give every tracked event equal total weight within each phase. 

The existing LPO-lag filtered composites remain the primary lifecycle view. The phase–pressure figure diagnoses filtered three-dimensional wave structure. Generate and cache all four wave bands.

### 6. Make the phase statistics internally consistent

- [ ] **Show strong-event and all-event measures separately.**
- [ ] **Add bootstrap or phase-randomized uncertainty.**
- [ ] **Show the number of strong-wave events contributing at every lag.**
- [ ] Calculate a reference for the phase distribution: select a set of random points with the same size as the LPO events in each simulation, then record the phase in the same way as the actual LPO events. Ensure the random points are in the same latitude range as the acceptance criterion for the LPO tracks. For each lag, track the random points forward/backward in time with the mean speed of the LPO tracks. This allows comparison of the phase/lag relationship for each wave between LPOs and a random set of points, to distinguish noise from a genuine connection.

**Recommended implementation**

At each LPO lag, condition the phase histogram and circular concentration on the same amplitude > 1 strong-wave sample. Normalize the eight phase probabilities so that they sum to one at every lag. Use the identical calculation for matched random tropical trajectories. Advect each random trajectory with the simulation mean LPO speed. Calculate circular concentration as

\[
R = \left|\sum_j w_j e^{i\theta_j}\right| / \sum_j w_j,
\]

with equal total weight per event. Figures 25–28 show all LPOs, strong-wave LPOs, and the matched random baseline. Figure 40 shows coherence, amplitude, and sample support.

Bootstrap complete event IDs 1,000 times for confidence intervals. Use the random phase distribution to note where the phase/lag relationship is significant.

### 7. Refine the group-velocity comparison

- [ ] **Keep \(C_{gx}\), \(C_{gz}\), and \(C_{gz}/C_{gx}\) in one figure.**
- [ ] **Focus visual attention on the upper troposphere and lower stratosphere.**
- [ ] **Mask or grey regions where \(|C_{gx}|\) is too small for a meaningful ratio.**
- [ ] **Use sequential colour scales for quantities that are effectively one-signed and diverging scales only where sign changes are meaningful.**
- [ ] **Add u-minus-control and T-minus-control panels with bootstrap robustness.**
- [ ] **Summarize the cached sensitivity to zonal wavenumber and vertical wavelength.**
- [ ] **State the dry-reference assumptions and masking threshold explicitly.**

The combined comparison follows [Lim and Son (2022)](https://doi.org/10.1175/JAS-D-21-0296.1).

**Recommended implementation**

Use a three-column main figure for \(C_{gx}\), \(C_{gz}\), and \(C_{gz}/C_{gx}\), with rows for control, u-nudge minus control, and T-nudge minus control. Put the three absolute simulations in a supplementary version. Show 10–25 km in the main figure and retain the full-depth version in the supplement.

For the ratio, apply the Lim–Son mask where \(|C_{gx}|\) <= \(10^{-3}\) m s\(^{-1}\).

State both masks explicitly. Use sequential colour maps when the displayed \(C_{gx}\) or \(C_{gz}\) values are one-signed and zero-centred diverging maps when both signs are present.

Summarize the existing wavenumber/vertical-wavelength sensitivity in one supplementary 3-by-3 heat map for zonal wavenumbers 1–3 and vertical wavelengths 5, 10, and 15 km. Each cell should contain the event-mean value over 14–20 km and -24 to +24 h for \(C_{gx}\), \(C_{gz}\), and their ratio. Use wavenumber 1 and 10-km vertical wavelength as the literature-comparable default.

### 8. Work out why figure 8 was not produced by the script.
