# `run_all_diagnostics.py` options

Run this command from the analysis directory:

```bash
python run_all_diagnostics.py [OPTIONS]
```

## Simulation profiles

Use `--profile` to select simulations for every child diagnostic.

| Profile | Simulations | Aggregate cache location |
| --- | --- | --- |
| `legacy` | control, nudged-u, existing nudged-T | Existing cache paths |
| `from-u` | control, nudged-u, nudged-T-from-u | `cache/<family>/profiles/from-u/` |
| `both` | control, nudged-u, existing nudged-T, nudged-T-from-u | `cache/<family>/profiles/both/` |

The default profile is `legacy`.

Per-simulation caches use stable simulation names and remain shared.
The new simulation uses `nudged T from u` in cache filenames.
Aggregate products use profile-specific paths when simulations change.
This layout keeps existing nudged-T products available.

Run without a rebuild option first.
Each diagnostic then reuses current products and builds missing products.
Use `--rebuild-caches` only when every selected cache requires replacement.

## Cache reuse between profiles

The runner does not rebuild a current simulation cache after a profile change.
Control and nudged-u use the same stable cache locations in every profile.
Existing nudged-T also remains available for `legacy` and `both`.
Nudged-T-from-u receives separate stable cache locations.

The first run can migrate current aggregate cache slices into stable simulation caches.
Migration reads the old cache and leaves it unchanged.
The diagnostic then calculates only missing simulation products.

| Diagnostic product | Reuse behavior |
| --- | --- |
| Mean-state fields and distributions | Reuse each simulation and variable family |
| Circulation and eddy transport | Reuse each simulation and product family |
| Fixed-level and annular-mode jet products | Reuse each simulation and product family |
| Thermal-wind zonal means and realised state | Reuse each simulation |
| Energy and vapour budgets | Reuse each simulation |
| High-cloud classification | Reuse each simulation |
| LPT tracking and lifecycle inputs | Reuse each simulation |
| Filtered wave products | Reuse each simulation |

Profile contrasts and uncertainty tables remain profile-specific.
These products combine several simulations and rebuild from the reusable inputs.
They remain under `profiles/<profile>/`.

Do not add `--rebuild-caches` for a normal `from-u` run.
That option intentionally invalidates all selected simulation caches.

Build the replacement profile:

```bash
python run_all_diagnostics.py --profile from-u
```

Build figures containing both temperature-nudged simulations:

```bash
python run_all_diagnostics.py \
    --profile both \
    --figure-version both-T-nudging
```

Rebuild only one diagnostic when required:

```bash
python run_all_diagnostics.py \
    --profile from-u \
    --only mean_state/energy_budget.py \
    --rebuild-caches
```

## Selection options

- `--list` lists the configured diagnostics.
- `--dry-run` prints commands without running them.
- `--only NAME ...` runs only the named diagnostics.
- `--skip NAME ...` excludes named diagnostics.
- `--continue-on-error` or `-k` continues after a failure.
- `--test` uses bounded actual-data subsets and isolated test products.
- `--qc` requests supported validation figures.

Names accept a stem, filename, or family-relative path.
For example, use `energy_budget` or `mean_state/energy_budget.py`.

## Cache options

Use only one general cache mode.

- `--figures-only` requires current products and only renders figures.
- `--rebuild-caches` forces supported diagnostics to rebuild their products.

The convection workflow provides separate cache controls.

- `--convection-workflow` runs classification, tracking, waves, and lifecycle analysis.
- `--rebuild-convection-caches` rebuilds all convection workflow products.

`--rebuild-convection-caches` requires `--convection-workflow`.

## Figure options

- `--figure-root PATH` changes the common figure root.
- `--figure-dir PATH` is an alias for `--figure-root`.
- `--figure-version NAME` places diagnostic directories beneath one release parent.

The default figure root is `/home/cwp29/tmp/mpas/crf_on`.
For example, `--figure-version from-u` writes beneath `final/from-u/`.
Each diagnostic keeps its normal subset beneath that parent.
The LPT figures use `final/<release>/organisation_lpt/` directly.

Use separate figure versions when comparing profiles.
This prevents one profile from replacing another profile's rendered figures.

## Execution and logging options

- `--python PATH` selects the child Python executable.
- `--log-dir PATH` selects the per-diagnostic log directory.
- `--no-log` disables per-diagnostic logs.

The runner executes diagnostics serially.
It prints memory heartbeat messages during silent, long-running calculations.

## Email options

- `--notify-email ADDRESS` sends workflow notifications.
- `--notify-on always` sends progress and final notifications.
- `--notify-on failure` sends only failure notifications.

Email requires a local `mail` or `mailx` command.

## Common examples

Inspect the replacement-profile commands:

```bash
python run_all_diagnostics.py --profile from-u --dry-run
```

Render the existing nudged-T figures from current caches:

```bash
python run_all_diagnostics.py \
    --profile legacy \
    --figures-only \
    --figure-version legacy-T
```

Run selected replacement-profile diagnostics:

```bash
python run_all_diagnostics.py \
    --profile from-u \
    --only energy_budget thermal_wind equatorial_wave_diagnostics
```

Run the full convection workflow for all four simulations:

```bash
python run_all_diagnostics.py \
    --profile both \
    --convection-workflow \
    --figure-version both-T-nudging
```
