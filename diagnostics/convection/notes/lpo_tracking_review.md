# Updates to LPO tracking methodology and validation figures

## 1. Change tracking parameters
- The linking between objects needs to be more strict, given that they were designed for satellite observations that are temporally incomplete. Set:
	bare_minimum_overlap_fraction = 0.10
	fall_below_threshold_max_hours = 24
	break_up_merge_split = True

## 2. Change sensitivity checks
- Sensitivity checks should be run on 100 days of data from control only. 
- Remove the running-mean and Gaussian-width checks (these parameters are based on Kerns & Chen and should not be altered).
- Remove the half-record checks.
- Check precipitation thresholds of 11, 12, 13 mm/day, gap allowance of 0, 24, 72 hours, mutual-overlap floor of 0.0, 0.1, 0.2, and merge/split handling True/False.
- Update the validation figures to reflect the modified sensitivity checks.

## 3. Correct centroids in track-overview figures
- Change `_read_plot_track_steps()` to read the upstream track centroid and area from `lpt_track_steps.csv`: centroid_lon, centroid_lat, area_km2. Do not use the precipitation-weighted centroid of the combined event maskf or tracking lines. These can jump between simultaneous precipitation lobes and visually exagerate track swtiching.
- Keep the precipitation-weighted centroid in the saved diagnostics for lifecycle analysis; change only the tracking-overview presentation. 

## 4. Refine the compact sensitivity summary
- Keep the existing six-panel layout, but use:
	- event count;
	- median event duration with interquartile range;
	- median zonal propagation speed with interquartile range;
	- fraction of primary events recovered;
	- median integrated precipitation with interquartile range;
	- fraction of events involving a merger or split.
- Remove the first-half/second-half count panel and avoid plotting every individual duration and precipitation value.
- Group results by parameter family

## 5. Update the methodology
- Update the module documentation and methodology metadata to state that the method follows the Kerns–Chen object definition but makes two dataset-motivated linking adaptations:
	- below-threshold reconnection is limited to 24 hours because the model record is complete and the rainfall field is already averaged over 72 hours;
 	- ambiguous merger/split paths are reduced to a largest-area lineage so that retained events do not duplicate shared branches.
- Increment `PRODUCT_VERSION` and `TRACKING_CACHE_VERSION` so that stale tracks cannot be reused.

## 6. Expand the LPO figure products
- Generate one envelope-evolution plot for every retained LPO in every simulation.
- Draw concurrent LPO boundaries with faded, dashed purple lines.
- Use one tracking-overview figure for each simulation.
- Place days 1–100 and 101–200 in side-by-side panels.
- Draw every instantaneous 12 mm/day candidate envelope in each overview.
- Draw centroid tracks only for candidates that become retained LPOs.
