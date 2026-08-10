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
5. `significance_bootstrap_diagnostics.py`

Wave Figure 11 supplies the response table consumed by the bootstrap analysis.
Figure 13 supplies the Kelvin reference-dispersion product and action 14 writes
the compact filtered event sections consumed by the lifecycle analysis.

## Corrected T-nudge omega rebuild

After changing source `omg_isobaric`, rebuild every affected cache owner once:

```bash
python run_all_diagnostics.py \
  --convection-workflow \
  --rebuild-convection-caches
```

This forces wave, lifecycle-section, and bootstrap regeneration. High-cloud
classification and LPO tracking are still run in the required order, but their
caches are not forcibly invalidated because they do not derive products from
`omg_isobaric`. Cache versions also encode the corrected T-nudge omega revision,
so stale pre-correction products are rejected by ordinary subsequent runs.

Use `--dry-run` to inspect the exact commands and `--list` to inspect the order.
Normal workflow runs reuse current caches, which is the intended mode for rapid
figure refinement after the one-time rebuild.
