"""Resample circulation blocks independently by simulation and hemisphere."""

from __future__ import annotations

import hashlib
import warnings
import weakref

import numpy as np
import xarray as xr


def hemisphere_samples(field: xr.DataArray, parity: str) -> xr.DataArray:
    """Reflect SH onto NH coordinates without averaging the observations."""

    latitude = np.asarray(field.latitude.values)
    if parity not in {"even", "odd"}:
        raise ValueError(f"Unknown hemispheric parity: {parity!r}")
    if not np.allclose(latitude, -latitude[::-1]):
        raise ValueError("Hemisphere samples need symmetric latitudes")
    north = field.where(field.latitude >= 0.0, drop=True)
    south = field.isel(latitude=slice(None, None, -1)).assign_coords(
        latitude=field.latitude
    ).where(field.latitude >= 0.0, drop=True)
    samples = xr.concat(
        [south if parity == "even" else -south, north],
        dim=xr.IndexVariable("hemisphere", ["SH", "NH"]),
    ).rename(latitude="absolute_latitude")
    if "block" in samples.dims:
        samples = samples.transpose("block", "hemisphere", ...)
    samples.attrs.update(field.attrs, hemisphere_parity=parity)
    samples.absolute_latitude.attrs.update(
        units="degrees_north_absolute", long_name="absolute latitude"
    )
    return samples


def finite_mean(values, axis):
    """Average finite observations without warnings for empty boundary cells."""

    finite = np.isfinite(values)
    count = finite.sum(axis=axis)
    total = np.where(finite, values, 0.0).sum(axis=axis)
    return np.divide(total, count, out=np.full(total.shape, np.nan), where=count > 0)


def response_distribution(
    blocks, variables, run_index, reference_index, statistic, *,
    n_resamples, minimum_blocks, seed,
):
    """Evaluate a statistic using joint variable draws and independent hemispheres.

    The statistic receives arrays with axes (variable, hemisphere, spatial...).
    Each draw preserves complete spatial fields and the pairing of variables.
    Both hemispheres share the equator's draws: it is one observation per time.
    Input block products remain immutable while plotting.
    """

    def climate(index):
        selected = blocks[list(variables)].isel(simulation=index, drop=True)
        if "hemisphere" in selected.dims:
            selected = selected.sel(hemisphere=["SH", "NH"])
        else:
            selected = selected.expand_dims(hemisphere=["single"])
        fields = [selected[name].transpose("hemisphere", "block", ...) for name in variables]
        if any(field.dims != fields[0].dims for field in fields):
            raise ValueError("Related bootstrap variables need matching dimensions")
        values = np.stack([field.values for field in fields]).astype(float)
        finite = np.isfinite(values).all(axis=0)
        values = np.where(finite[None, ...], values, np.nan)
        keep = finite.any(axis=tuple(i for i in range(finite.ndim) if i != 1))
        values = values[:, :, keep]
        count = values.shape[2]
        if count < minimum_blocks:
            raise ValueError(f"Insufficient finite blocks: {count}; need {minimum_blocks}")
        enough = np.isfinite(values[0]).sum(axis=1) >= minimum_blocks
        template = fields[0].isel(hemisphere=0, block=0, drop=True)
        equator = None
        if "absolute_latitude" in template.dims:
            equator = np.asarray(template.absolute_latitude.values) == 0.0
            shape = [1] * template.ndim
            shape[template.get_axis_num("absolute_latitude")] = equator.size
            equator = equator.reshape(shape)
        label = str(blocks.simulation.values[index]) if "simulation" in blocks.coords else str(index)
        identity = int.from_bytes(hashlib.blake2s(label.encode(), digest_size=4).digest(), "little")
        rng = np.random.default_rng(np.random.SeedSequence([seed, identity]))

        def mean(resample):
            if not resample:
                return np.where(enough[None, ...], finite_mean(values, axis=2), np.nan)
            indices = rng.integers(count, size=(values.shape[1], count))
            means = []
            for hemisphere, selection in enumerate(indices):
                result = finite_mean(values[:, hemisphere, selection], axis=1)
                if hemisphere and equator is not None and equator.any():
                    shared = finite_mean(values[:, hemisphere, indices[0]], axis=1)
                    result = np.where(equator, shared, result)
                means.append(np.where(enough[hemisphere], result, np.nan))
            return np.stack(means, axis=1)

        return mean, template, count, values.shape[1]

    run, template, n_run, h_run = climate(run_index)
    reference, reference_template, n_reference, h_reference = climate(reference_index)
    xr.align(template, reference_template, join="exact")
    estimate = statistic(run(False), reference(False))
    samples = np.stack([
        statistic(run(True), reference(True)) for _ in range(n_resamples)
    ]).astype(np.float32)
    metadata = dict(
        method="independent complete-block bootstrap by simulation and hemisphere",
        bootstrap_replicates=n_resamples,
        n_blocks_run=n_run,
        n_blocks_reference=n_reference,
        n_hemispheres_run=h_run,
        n_hemispheres_reference=h_reference,
        equator_handling="shared block draws across hemispheres",
    )
    return estimate, samples, template, metadata


def response_interval(estimate, samples, template, metadata, percentiles):
    """Return pointwise intervals only where every replicate is defined."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        low, high = np.nanpercentile(samples, percentiles, axis=0)
    valid = np.isfinite(estimate) & np.isfinite(samples).all(axis=0)
    return xr.Dataset(
        {
            "block_mean_response": template.copy(data=estimate),
            "ci_low": template.copy(data=np.where(valid, low, np.nan)),
            "ci_high": template.copy(data=np.where(valid, high, np.nan)),
        },
        attrs={**metadata, "confidence_percentiles": list(percentiles)},
    )


_INTERVAL_CACHE = {}


def cached_interval(blocks, key, calculate):
    """Reuse intervals across figures for the same immutable block dataset."""

    identity = id(blocks)
    if identity not in _INTERVAL_CACHE:
        reference = weakref.ref(blocks, lambda _: _INTERVAL_CACHE.pop(identity, None))
        _INTERVAL_CACHE[identity] = (reference, {})
    results = _INTERVAL_CACHE[identity][1]
    if key not in results:
        results[key] = calculate()
    return results[key]
