# Diagnostic cache policy

This policy separates scientific products from presentation outputs.

All reusable products use one central cache root:

`/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/cache`

`analysis_config.py` owns every cache-family directory name.

Diagnostic scripts request a logical cache family from the shared helper.

Diagnostic scripts must not construct cache paths from the simulation root.

## Required modes

Expensive diagnostics should expose these modes.

- Default mode reuses every current cache and rebuilds each stale cache.
- `--figures-only` reads current calculation products and never opens raw histories.
- `--rebuild-cache` rebuilds calculation products before plotting.
- `--figures` selects presentation outputs without changing cache identity.

Figure-only mode must fail when a required cache is missing or stale.

## Cache identity

Each cache producer owns one local schema identifier.

Each cache manifest records the producer, schema, scientific configuration, dependencies, and source-file signature.

The source signature contains canonical paths, sizes, and nanosecond modification times.

The schema identifier changes only when the cached calculation or storage format changes.

Figure layout changes do not change cache schemas.

## Figure releases

`FigureSaver` writes figures beneath a release directory.

The default release is `v1`.

Set `MPAS_DIAGNOSTIC_FIGURE_VERSION` to select another release.

Use `run_all_diagnostics.py --figure-version NAME` for managed runs.

Figure release names do not control scientific cache reuse.

## Current support

| Diagnostic | Current cache control |
|---|---|
| Mean state | Five source-signed compact families support figure-only and rebuild modes. |
| Energy budget | Source-signed time-series caches supply closure uncertainty and figures. |
| Vapour budget | Source-signed compact products and attribution caches support all figures. |
| Thermal wind | Signed mean states and block summaries support figure-only rendering. |
| Circulation | Signed compact products include independent block-sign evidence. |
| Jet variability | Signed index products supply response intervals and figures. |
| High cloud | Classification grids and compact figure products have separate identities. |
| LPO tracking | Source-signed standard products support validated default reuse. |
| Equatorial waves | Filtered products support restart and compact figure products support rendering. |
| LPO lifecycle | Final bundles use source signatures and resumable event checkpoints. |

The significance and symmetry scripts are absent from managed workflows.

The standalone significance script is retired.

The LPO owner now produces retained event-property uncertainty.

The mean-state owner now produces retained CAPE and precipitable-water distributions.

`run_all_diagnostics.py --figures-only` passes strict reuse modes to supported scripts.

`run_all_diagnostics.py --rebuild-caches` rebuilds supported scientific products.

Every managed diagnostic now supports strict figure-only execution.

## Implementation rule

Cache compact scientific arrays and tables that fully determine each figure.

Do not cache rendered images as calculation products.

Do not open raw histories before cache validation.

Write caches atomically through a temporary file.

Keep test caches separate from publication caches.

Delete restart intermediates only after final-product validation succeeds.

Never delete permanent products automatically.
