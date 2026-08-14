# Diagnostic calculation contracts

This inventory records the maintained workflow after the architecture migration.

| Owner | Authoritative inputs | Permanent products | Uncertainty unit |
|---|---|---|---|
| Mean state | Latitude-longitude and zonal-mean archives | Five field families and CAPE/PW block distributions | Independent ten-day blocks |
| Energy budget | Native archived scalars and MPAS moisture mixing ratios | Global budget time series and closure summaries | Independent ten-day blocks |
| Vapour budget | Native fields, archived `relhum`, and archived `precipw` | Mean budget fields, native mean states, and attribution | No new interval changes |
| Thermal wind | Zonal means and compact latitude-longitude block states | Mean fields, realised blocks, and response intervals | Independent ten-day blocks |
| Circulation | Zonal-mean circulation fields | Compact means, block metrics, and response tables | Independent ten-day blocks |
| Jet variability | Zonal wind archives | Jet tracks, EOF products, and response intervals | Independent ten-day blocks |
| High cloud | Cloud fraction, OLR, precipitation, and pressure | Classifications and compact figure products | Independent ten-day blocks |
| LPO tracking | Precipitation and current cloud classifications | Standard events, masks, summaries, and uncertainty | Complete events |
| Equatorial waves | Archived fields, LPO products, and classifications | Filtered fields and compact figure products | Independent blocks by simulation |
| LPO lifecycle | LPO events and wave products | Final event bundles and plot-ready sections | Complete events |

## Shared definitions

`analysis_config.py` owns simulations, archives, periods, colours, and block settings.

`cache_support.py` owns cache identity, source signatures, and atomic metadata writes.

`uncertainty.py` owns independent simulation resampling and field-significance mechanics.

Owner scripts retain thresholds, equations, masks, and diagnostic transformations.

## Scientific invariants

Use archived `relhum` for native relative humidity.

Use archived `precipw` and `cape` instead of rebuilding those native variables.

Treat each nudged simulation as an independent statistical climate.

Use the same control-selected block length across simulations.

Retain nonpositive native N-squared samples in lifecycle composites.

Apply cosine latitude weights to physical latitude sampling.

Use MPAS dry density with all dry-air moisture mixing ratios for moist density.

The vapour decomposition remains the documented phase-aware saturation exception.

## Execution contract

Run every owner directly or through `run_all_diagnostics.py`.

Default mode validates and reuses current products.

Figure-only mode reads calculation products without opening raw histories.

Rebuild mode replaces only the selected scientific cache families.
