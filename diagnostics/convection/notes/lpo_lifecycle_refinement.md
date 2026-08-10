# LPO lifecycle refinements and extensions

Note that "height" means the vertical axis, which should be vertical log-pressure by default.

## Matsuno-Gill response test

- Add a figure that shows an LPO-centred relative longitude vs. relative latitude composite plot of relative vorticity
(shaded) at 850 hPa and 200 hPa with horizontal velocity vectors overlaid.
- Anomalies should be calculated relative to the simulation's own climatology.
- Use the geometric centroid, not precipitation-weighted.
- Only use eastward propagating LPOs. Indicate in the title how many of the total number of LPOs are included
in the composite.

## Compact UTLS stability pathway analysis

The aim is to use the separate u-nudged T-nudged simulations to show how
changes in the background wind and static stability lead to changes in Kelvin-wave energy dispersion, hence
the cold-cap response to tropical convection and the associated enhancement of static stability reduction that
could enhance convection. Plot N^2 for the static-stability profile but use N = sqrt(N^2) for the group
velocity calculation below. We will produce height vs. relative longitude plots to show the Kelvin-wave group
velocity changes identified by Lim & Son 2022 and the associated temperature and static stability response. By
comparing simulations with and without a background wind change, and comparing an LPO-centred composite with a
generic composite, we can identify the role of the wind/thermal pathways and the important of convective
feedback on the response. The existing height vs. lag plots show, in addition, whether these responses
persist through the LPO lifecycle.

1. Add unconditional tropical-mean profiles of background zonal wind and static stability to figure 13. Change
   the figure to use a vertical log-pressure axis. Remove the row with differences vs. control. Indicate the
   latitude range for the mean in the title (i.e. just add 10^\circS-10^\circN or whatever is appropriate).
   Remove the "Lim-Son" word from the title but keep the wave parameters. Repeat the plot for k=2. 
2. Add three LPO-centred relative longitude vs. height composites to the lifecycle analysis script, showing
   Kelvin-wave-filtered temperature at lag 0 (peak precipitation), unfiltered static stability and unfiltered
   high-cloud occurrence. Overlay non-precipitating cloud occurence as contours. Each plot should show the
   control simulation, then u-nudged minus control and T-nudged minus control. Each plot should have a subplot
   below it with the LPO precipitation vs. longitude. The composites should show the predicted rays from the
   LPO-centred background wind U(z) and static stability N(z) computed in the vicinity of the LPO centroid
   (ensure this is consistent with existing calculations in the script). The calculation of the ray is
   described below.
3. To isolate LPO-associated amplification, add an additional plot focused on static stability changes. Compute teh
   difference between the LPO-conditioned longitude vs. height static stability and the background static
   stability computed for figure 13 of the equatorial wave diagnostics script. Then plot this for control,
   then u-nudged minus control and T_nudged minus control. That will isolate static stability changes
   associated with LPOs from the changes throughout the tropical atmosphere.
4. Add static stability to the existing unfiltered and wave-filtered height vs. lag plots, and include
   LPO-envelope precipitation vs. lag. 

### Theoretical ray calculation

To calculate the ray path, use the group-velocity expressions for Kelvin waves but with the horizontal
component adjusted based on the LPO propagation speed c_{LPO}. The LPO propagation speed should be calculated
using the geometric centroid, averaged over a 48 hour pre-peak period. Compute U(z) and N(z) for each event,
compute the rays, then composite. The rays should be computed using geometric height then converted to
pressure. The relative x-position should be converted to relative longitude for plotting. Launch rays every 20
degrees of relative longitude to help with visualisation. Start the rays at 200 hPa.

## Extension to mixed Rossby-gravity (MRG) waves

There is a strong mixed Rossby-gravity wave signal at upper levels which was not clear from the precipitation
spectrum. This wave should be added to the analysis and propagated through the equatorial wave diagnostics
script as well as the lifecycle analysis. Repeat the wave-response longituide-height analysis for MRG-filtered
temperature, with the corresponding MRG theoretical ray. Use the appropriate antisymmetric latitude
projection. Use the standard MRG dispersion relation with k=-2. For now, use a 10km vertical wavelength but we
may refine that once the RRB-style wave composites are calculated to show the wave structures.
