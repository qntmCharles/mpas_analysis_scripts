# Convection publication workflow

Run the complete convection/LPO publication analysis from the repository root:

```bash
python run_all_diagnostics.py --convection-workflow
```

The runner enforces this dependency order:

1. `refined_high_cloud_classification_diagnostics.py`
2. `lpo_tracking.py`
3. `equatorial_wave_diagnostics.py --figures 11 13 14`
4. `lpo_lifecycle_analysis.py`

Figure 13 supplies the Kelvin reference-dispersion product and action 14 writes
the compact filtered event sections consumed by the lifecycle analysis.

## Corrected T-nudge omega rebuild

After changing source `omg_isobaric`, rebuild every affected cache owner once:

```bash
python run_all_diagnostics.py \
  --convection-workflow \
  --rebuild-convection-caches
```

This rebuilds high-cloud, LPO, wave, and lifecycle products. Use this mode only
after a relevant scientific calculation changes.

Use `--dry-run` to inspect the exact commands and `--list` to inspect the order.
Normal workflow runs reuse current caches, which is the intended mode for rapid
figure refinement after the one-time rebuild.
