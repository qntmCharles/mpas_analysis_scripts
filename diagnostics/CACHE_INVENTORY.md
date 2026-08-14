# Diagnostic cache inventory

The permanent cache root is:

`/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/cache`

The central layout contains these cache families:

| Directory | Owner |
|---|---|
| `block_resampling` | Control autocorrelation and block-length review |
| `classification` | Shared high-cloud classifications |
| `circulation` | Circulation diagnostics |
| `energy_budget` | Global atmospheric energy budget |
| `high_cloud` | Compact high-cloud figure products |
| `jet_variability` | Jet variability diagnostics |
| `lpt` | LPO tracking and lifecycle products |
| `mean_state` | Shared mean-state products |
| `thermal_wind` | Thermal-wind diagnostics |
| `vapour_transport` | Vapour-budget calculation products |
| `vapour_transport_data` | Retained vapour-budget summaries |
| `waves` | Filtered waves and compact wave products |

`analysis_config.py` defines all directory names.

Diagnostic scripts do not construct cache paths directly.

Large lifecycle and wave intermediates support restart after interruption.

Successful final validation removes only intermediates used by that run.

The user controls every permanent-cache removal.
