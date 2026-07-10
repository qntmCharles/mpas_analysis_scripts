"""Memory-bounded helpers for the precipitation mean-state diagnostics."""

import numbers

import dask
import dask.array as dsa
import numpy as np


DEFAULT_TROPICAL_LAT_BOUNDS = (-10.0, 10.0)


def accumulated_precipitation_components(ds):
    """Return total, convective, and resolved accumulated rain as float32."""
    hundred = np.float32(100.0)
    convective = (
        ds["rainc"].astype(np.float32)
        + ds["i_rainc"].astype(np.float32) * hundred
    )
    resolved = (
        ds["rainnc"].astype(np.float32)
        + ds["i_rainnc"].astype(np.float32) * hundred
    )
    return (convective + resolved).astype(np.float32), convective, resolved


def precipitation_rate(accumulated, dt_hours=6):
    """Convert accumulated precipitation to a float32 rate in mm/day."""
    if issubclass(accumulated.Time.dtype.type, numbers.Integral):
        rate = accumulated.diff("Time") / np.float32(dt_hours / 24.0)
    else:
        dt_days = accumulated.Time.diff("Time") / np.timedelta64(1, "D")
        rate = accumulated.diff("Time") / dt_days
    return rate.astype(np.float32)


def domain_fields(field, tropical_lat_bounds=DEFAULT_TROPICAL_LAT_BOUNDS):
    """Return lazy global and tropical views without loading either view."""
    lat_name = "latitude" if "latitude" in field.dims else "lat"
    lo, hi = tropical_lat_bounds
    latitude = field[lat_name]
    tropical = field.where((latitude >= lo) & (latitude <= hi), drop=True)
    return field, tropical


def _histogram_task(field, bins, value_range):
    values = field.data
    if not isinstance(values, dsa.Array):
        return np.histogram(np.asarray(values), bins=bins, range=value_range)

    counts, edges = dsa.histogram(values.ravel(), bins=bins, range=value_range)
    return counts, edges


def domain_histograms(
    field,
    bins,
    value_range=None,
    tropical_lat_bounds=DEFAULT_TROPICAL_LAT_BOUNDS,
):
    """Compute global and tropical histograms without materialising the field."""
    tasks = [
        _histogram_task(domain, bins=bins, value_range=value_range)
        for domain in domain_fields(field, tropical_lat_bounds)
    ]
    count_tasks = [task[0] for task in tasks]
    if any(isinstance(task, dsa.Array) for task in count_tasks):
        counts = dask.compute(*count_tasks)
    else:
        counts = tuple(np.asarray(task) for task in count_tasks)

    # Both domains use identical edges. Dask returns these eagerly.
    edges = np.asarray(tasks[0][1])
    return tuple((np.asarray(count), edges) for count in counts)


def relative_frequency_from_histogram(counts, edges, smooth):
    """Apply the diagnostic's existing smoothing and zero-bin normalisation."""
    from scipy.ndimage import uniform_filter1d

    smoothed = uniform_filter1d(np.asarray(counts, dtype=float), size=smooth)
    scale = smoothed[0] if smoothed[0] > 0 else np.nanmax(smoothed)
    if not np.isfinite(scale) or scale <= 0:
        scale = 1.0
    centres = 0.5 * (edges[1:] + edges[:-1])
    return centres, smoothed / scale


def duration_domain_histograms(
    field,
    threshold,
    bins,
    value_range,
    tropical_lat_bounds=DEFAULT_TROPICAL_LAT_BOUNDS,
    time_block=8,
):
    """Return global/tropical wet-spell histograms with bounded peak memory.

    A few global timesteps are loaded at once, while an integer run-length
    state is retained for each grid point. This matches the files' global
    horizontal precipitation chunks and reads every timestep only once.
    Terminal spells are omitted to match the historical routine.
    """
    lat_name = "latitude" if "latitude" in field.dims else "lat"
    lon_name = "longitude" if "longitude" in field.dims else "lon"
    time_name = "Time" if "Time" in field.dims else "time"
    ordered = field.transpose(time_name, lat_name, lon_name)

    edges = np.histogram_bin_edges([], bins=bins, range=value_range)
    global_counts = np.zeros(edges.size - 1, dtype=np.int64)
    tropical_counts = np.zeros_like(global_counts)
    latitudes = np.asarray(ordered[lat_name].values)
    lo, hi = tropical_lat_bounds
    tropical_mask = (latitudes >= lo) & (latitudes <= hi)
    run_dtype = (
        np.int16
        if ordered.sizes[time_name] <= np.iinfo(np.int16).max
        else np.int32
    )
    run_lengths = np.zeros(
        (ordered.sizes[lat_name], ordered.sizes[lon_name]), dtype=run_dtype
    )

    for time_start in range(0, ordered.sizes[time_name], time_block):
        time_stop = min(time_start + time_block, ordered.sizes[time_name])
        values = ordered.isel({time_name: slice(time_start, time_stop)}).values
        wet_block = np.isfinite(values) & (values > threshold)
        for wet in wet_block:
            run_lengths[wet] += 1
            ended = (~wet) & (run_lengths > 0)
            if not ended.any():
                continue
            global_counts += np.histogram(run_lengths[ended], bins=edges)[0]
            tropical_ended = ended & tropical_mask[:, None]
            if tropical_ended.any():
                tropical_counts += np.histogram(
                    run_lengths[tropical_ended], bins=edges
                )[0]
            run_lengths[ended] = 0

    return (global_counts, edges), (tropical_counts, edges)
