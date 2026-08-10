# Comments on high-cloud analysis figures

Located in `~/tmp/mpas/crf_on/final/high_cloud_refined/2026-08-01/`

## Figure 1: (a) high cloud occurrence and composition and (b) regime attribution.

### Figure 1a
- High-cloud composition is just the fraction of high-clouds within each class for each simulation. It does not add anything scientifically useful beyond panels (a) and (b). 
- The literature values that were inteneded to act as validation on the efficacy of our high-cloud classification have lost meaning, and their formatting is inconsistent. Re-explore this idea, using any relevant values in the literature to show whether the classification is consistent with real-Earth observations and other modelling studies (particularly aquaplanets). If there are none, say so: there are already literature validation values in figure 1b. Remove the annotations in panel (a) showing the values; these are redundant.
- Change panel (b) horizontal axis label to "Difference (% pts)"

### Figure 1b
- The second figure does not add any scientific value at present. Move panel (a) of figure 1b to figure 1a
along with the literature values (which support our work). Remove panels (b) and (c) permanently.

## Figure 2: OLR distributions

- Change the part of the title in brackets to "(subtropics, 20-40$^\circ$)" and "(deep tropics, 20$^\circ$S-20$^\circ$N)". 
- Are these distributions PDFs or histograms? If the first, use a number between 0 and 1. If the latter, use "relative frequency" as the vertical axis. The existing vertical label is non-standard.

## Figure 2: vertical structure

- Remove the height labels. Use a vertical log-pressure axis only.
- Correct the TTL shading (that may already be done!)
- Mark the deep cloud boundary.
- Same point as above regarding probability axis. Probabilities should be between 0 and 1; but should this be relative frequency?
- Remove the summary TTL-reaching share annotation, and instead place it as a small aligned subplot below each panel with a bar chart.
- Use one shared figure legend rather than repeating in each subplot.
- Same change to the figure titles as above.

## Figure 3: validation maps

- Review and fix the formatting. For example, you should ensure that colourbars do not intersect with axis labels.
- Label the subplots.
- Remove the cyan tint in the bottom left panel, and use cloud-top pressure instead of height.
- Remove the cloud class tint in the top left panel (which is not visible anyway).
- Use a legend (1: warm shield, 2: cold shield, 3: cold core) with the appropriate T_b values instead of a colourbar in the top right panel. Label the subplot as "PyFLEXTRKR category".

## Figure 3: vertical slices

- Use a vertical log-pressure axis only.
- The bottom ticks are not readable and it is not clear why they are there? Why mark 12mm/day?
- Use a shared figure legend for the high-cloud boundary.
- Use a better range for the cldfrac colourbar; it is mostly saturated in the centre of clouds. Also: can you design a colourmap (or identify one in the literature, or see other scripts that make a movie from the simulation output) that looks a bit more snazzy, maybe with white clouds and a blue background?

## Figure 4: vertical profiles

- Label subplots. 
- Change bottom row horizontal axis label to "Difference (% pts)"
- Use a vertical log-pressure axis only, shared between subplots.
- Correct the TTL shading (that may already be done!)
- Remove the subtitle.

## Figure 5: spatial response

- Fix the aspect ratio for latitude and longitude to 1, and adjust the plot size to compensate. It may have to be split into multiple figures.
- Clarify what occurence is: area-time coverage as in earlier plots? Ensure consistency.

## Figure 6:
- Remove the "negative = ..." axis labels.

## General comments

- Renumber the figures: too many figures have the same number at the moment. If it is the same figure but for different simulations, use a/b/c to distinguish.
- Figures should be 8 inches wide by default. Only large or wide figures should be 12 inches wide.
- Ensure consistent in terminology, colouring and other conventions with other diagnostic scripts.
