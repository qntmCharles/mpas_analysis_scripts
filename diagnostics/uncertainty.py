"""Approved uncertainty mechanics for independent simulation climates."""

from __future__ import annotations

import numpy as np

from diagnostics.analysis_config import CONFIDENCE_PERCENTILES, FDR_ALPHA


def independent_bootstrap_difference(
    simulation_values,
    control_values,
    *,
    rng,
    n_resamples,
    percentiles=CONFIDENCE_PERCENTILES,
    statistic=np.nanmean,
):
    """Resample two simulation climates independently and return differences."""

    simulation = np.asarray(simulation_values, dtype=float)
    control = np.asarray(control_values, dtype=float)
    simulation = simulation[np.isfinite(simulation)]
    control = control[np.isfinite(control)]
    if simulation.size == 0 or control.size == 0:
        raise ValueError("Both simulations require finite samples")

    replicates = np.empty(int(n_resamples), dtype=float)
    for index in range(replicates.size):
        sim_indices = rng.integers(0, simulation.size, simulation.size)
        control_indices = rng.integers(0, control.size, control.size)
        replicates[index] = statistic(simulation[sim_indices]) - statistic(
            control[control_indices]
        )
    estimate = float(statistic(simulation) - statistic(control))
    low, high = np.nanpercentile(replicates, percentiles)
    sign = np.sign(estimate)
    sign_probability = np.nan if sign == 0 else np.mean(np.sign(replicates) == sign)
    return {
        "estimate": estimate,
        "interval_low": float(low),
        "interval_high": float(high),
        "sign_probability": float(sign_probability),
        "replicates": replicates,
        "n_simulation": int(simulation.size),
        "n_control": int(control.size),
    }


def independent_weighted_ratio_difference(
    simulation_sums,
    simulation_weights,
    control_sums,
    control_weights,
    *,
    rng,
    n_resamples,
    percentiles=CONFIDENCE_PERCENTILES,
):
    """Bootstrap weighted means with independent indices for each climate."""

    sim_sum, sim_weight = _finite_weighted_samples(simulation_sums, simulation_weights)
    ctl_sum, ctl_weight = _finite_weighted_samples(control_sums, control_weights)
    replicates = np.empty(int(n_resamples), dtype=float)
    for index in range(replicates.size):
        sim_indices = rng.integers(0, sim_sum.size, sim_sum.size)
        ctl_indices = rng.integers(0, ctl_sum.size, ctl_sum.size)
        replicates[index] = (
            sim_sum[sim_indices].sum() / sim_weight[sim_indices].sum()
            - ctl_sum[ctl_indices].sum() / ctl_weight[ctl_indices].sum()
        )
    estimate = sim_sum.sum() / sim_weight.sum() - ctl_sum.sum() / ctl_weight.sum()
    low, high = np.nanpercentile(replicates, percentiles)
    return float(estimate), float(low), float(high), replicates


def independent_bootstrap_field_difference(
    simulation_blocks,
    control_blocks,
    *,
    rng,
    n_resamples,
    percentiles=CONFIDENCE_PERCENTILES,
):
    """Bootstrap block fields with independent cross-simulation indices."""

    simulation = np.asarray(simulation_blocks, dtype=float)
    control = np.asarray(control_blocks, dtype=float)
    if simulation.ndim < 1 or control.ndim < 1:
        raise ValueError("Block fields require a leading block dimension")
    if simulation.shape[1:] != control.shape[1:]:
        raise ValueError("Block fields require matching spatial shapes")
    replicates = np.empty((int(n_resamples),) + simulation.shape[1:], dtype=np.float32)
    for index in range(replicates.shape[0]):
        sim_indices = rng.integers(0, simulation.shape[0], simulation.shape[0])
        control_indices = rng.integers(0, control.shape[0], control.shape[0])
        replicates[index] = np.nanmean(simulation[sim_indices], axis=0) - np.nanmean(
            control[control_indices], axis=0
        )
    estimate = np.nanmean(simulation, axis=0) - np.nanmean(control, axis=0)
    low, high = np.nanpercentile(replicates, percentiles, axis=0)
    return estimate, low, high


def _finite_weighted_samples(sums, weights):
    sums = np.asarray(sums, dtype=float)
    weights = np.asarray(weights, dtype=float)
    valid = np.isfinite(sums) & np.isfinite(weights) & (weights > 0.0)
    if not np.any(valid):
        raise ValueError("Weighted bootstrap requires positive finite weights")
    return sums[valid], weights[valid]


def bootstrap_complete_events(values, *, rng, n_resamples, statistic=np.nanmean):
    """Resample complete event rows without splitting an event into steps."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        raise ValueError("Event bootstrap requires finite events")
    replicates = np.empty(int(n_resamples), dtype=float)
    for index in range(replicates.size):
        sample = rng.integers(0, values.size, values.size)
        replicates[index] = statistic(values[sample])
    low, high = np.nanpercentile(replicates, CONFIDENCE_PERCENTILES)
    return float(statistic(values)), float(low), float(high), replicates


def benjamini_hochberg(p_values, alpha=FDR_ALPHA):
    """Return a Benjamini-Hochberg field-significance mask."""

    values = np.asarray(p_values, dtype=float)
    result = np.zeros(values.shape, dtype=bool)
    finite = np.isfinite(values)
    flat = values[finite]
    if flat.size == 0:
        return result
    order = np.argsort(flat)
    sorted_values = flat[order]
    thresholds = float(alpha) * np.arange(1, flat.size + 1) / flat.size
    accepted = np.flatnonzero(sorted_values <= thresholds)
    if accepted.size:
        cutoff = sorted_values[accepted[-1]]
        result[finite] = flat <= cutoff
    return result
