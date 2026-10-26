"""Compact mean-state inputs for the convection publication summary.

This owner reduces surface histories before loading them. It retains complete
20-day blocks for uncertainty and the full record for each plotted estimate.
The existing mean-state N2 definition operates on time-mean zonal theta.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from diagnostics.analysis_config import (
    ANALYSIS_END, ANALYSIS_START, BLOCK_LENGTH_DAYS,
    DEEP_TROPICAL_LAT_BOUNDS, TROPICAL_LAT_BOUNDS, LATLON_PATTERN, ZONAL_MEAN_PATTERN,
    cache_directory, simulation_directory,
)
from diagnostics.cache_support import (
    cache_manifest, cache_manifest_matches, load_netcdf, materialize,
    read_json_file, write_json_file, write_netcdf_atomic,
)
from diagnostics.diagnostic_helpers import (
    N2_latheight, hemispheric_fold, load_latlon_dataset,
    standardize_latlon_dataset, subset_between, time_block_slices,
    total_precip_accum, vertical_velocity_from_zm,
)
from diagnostics.mean_state.vapour_transport_budget import mixing_ratio_to_specific_humidity


LATITUDE_LIMIT = 40.0
BLOCK_DAYS = BLOCK_LENGTH_DAYS["mean_state_distributions"]
SCHEMAS = {
    "surface": "convection-summary-surface-v2-positive-latitude-endpoints",
    "environment": "convection-summary-environment-v4-native-profiles",
    "humidity_ascent": "convection-summary-humidity-ascent-v1-specific-humidity",
    "ascent": "convection-summary-ascent-v3-native-vertical-velocity",
}
SURFACE_FIELDS = ("precipitation", "olr", "precipitable_water")
ENVIRONMENT_FIELDS = (
    "temperature", "static_stability", "zonal_wind",
)
MAP_FIELDS = ("specific_humidity", "upward_pressure_velocity")


def product_paths(simulation, family, *, time_count=None):
    root = (cache_directory("mean_state", "testing", f"profiles_{time_count:03d}times", simulation)
            if time_count is not None else cache_directory("mean_state", "simulations", simulation))
    stem = f"convection_summary_{family}"
    return root / f"{stem}.nc", root / f"{stem}.manifest.json"


def product_manifest(simulation, family, *, time_count=None):
    patterns = ([LATLON_PATTERN, ZONAL_MEAN_PATTERN] if family == "humidity_ascent"
                else [LATLON_PATTERN if family == "surface" else ZONAL_MEAN_PATTERN])
    sources = [path for pattern in patterns
               for path in sorted(simulation_directory(simulation).glob(pattern))]
    return cache_manifest(
        "diagnostics/mean_state/convection_summary_cache.py",
        schema=SCHEMAS[family],
        configuration={
            "simulation": simulation,
            "analysis_start": ANALYSIS_START,
            "analysis_end": ANALYSIS_END,
            "block_days": BLOCK_DAYS,
            **({"test_time_count": time_count,
                "test_block_days": min(BLOCK_DAYS, (time_count // 2) * 0.25)}
               if time_count is not None else {}),
            "latitude_bounds": (
                [-LATITUDE_LIMIT, LATITUDE_LIMIT] if family == "surface"
                else list(DEEP_TROPICAL_LAT_BOUNDS) if family == "ascent"
                else list(TROPICAL_LAT_BOUNDS)
            ),
            **({"profile_latitude_bounds": list(DEEP_TROPICAL_LAT_BOUNDS)}
               if family == "environment" else {}),
            **({"profile_fields": "native t, uReconstructZonal, theta, and pressure; no isobaric interpolation"}
               if family == "environment" else {}),
            **({"specific_humidity": "qv_isobaric / (1 + qv_isobaric), before all averaging",
                "upward_pressure_velocity": "-omg_isobaric, Pa s-1"}
               if family == "humidity_ascent" else {}),
            **({"vertical_velocity": "cosine-weighted native w; align bounding interfaces to pressure midlevels only"}
               if family == "ascent" else {}),
            "rainfall_intervals": "six hours ending at the timestamp; preceding endpoint included",
            "n2": "g / time-mean zonal theta * d(time-mean zonal theta)/dz",
            "hemispheres": "paired even fold; cosine weights for latitude means",
        },
        source_paths=sources,
    )


def surface_profiles(reduced, *, block_days=BLOCK_DAYS):
    """Difference bucket-corrected zonal accumulation before time averaging.

The first timestamp supplies the preceding endpoint. Surface state means use
the interval-ending times, so precipitation, OLR, and PW cover the same record.
"""
    times = reduced.Time.values
    if times.size < 2 or not np.all(np.diff(times) == np.timedelta64(6, "h")):
        raise ValueError("Surface profiles require consecutive six-hour endpoints")
    rates = reduced.accumulation.diff("Time") / (
        reduced.Time.diff("Time") / np.timedelta64(1, "D")
    )
    samples = xr.concat(
        [rates, reduced.olr.sel(Time=rates.Time),
         reduced.precipitable_water.sel(Time=rates.Time)],
        dim=xr.IndexVariable("field", list(SURFACE_FIELDS)),
    )
    if not np.isfinite(samples.values).all():
        raise ValueError("Nonfinite samples in publication surface profiles")
    if float(rates.min()) < -1e-5:
        raise ValueError("Negative accumulated-rainfall increment")
    samples = subset_between(
        hemispheric_fold(samples, parity="even"), "latitude", (0.0, LATITUDE_LIMIT)
    ).rename(latitude="absolute_latitude")
    slices = time_block_slices(samples, block_days, complete=True, min_blocks=2)
    blocks = xr.concat(
        [samples.isel(Time=s).mean("Time") for s in slices],
        dim=xr.IndexVariable("block", np.arange(len(slices))),
    )
    return xr.Dataset({
        "surface_mean": samples.mean("Time"),
        "surface_blocks": blocks,
        "block_start": ("block", [samples.Time.values[s.start] for s in slices]),
        "block_end": ("block", [samples.Time.values[s.stop - 1] for s in slices]),
    })


def _latitude_mean(field):
    weights = np.cos(np.deg2rad(field.latitude))
    return field.weighted(weights).mean("latitude")


def environment_mean(zonal):
    """Return the established deep-tropical mean-state profiles."""
    deep = subset_between(zonal, "latitude", DEEP_TROPICAL_LAT_BOUNDS)
    n2 = N2_latheight(deep)

    output = xr.Dataset({
        "temperature": _latitude_mean(deep.t.astype("float64").mean("Time")),
        "zonal_wind": _latitude_mean(deep.uReconstructZonal.astype("float64").mean("Time")),
        "static_stability": _latitude_mean(n2),
        "pressure_hpa": _latitude_mean(deep.pressure.mean("Time")) / 100.0,
    })
    return output


def environment_profiles(zonal, *, block_days=BLOCK_DAYS):
    slices = time_block_slices(zonal.theta, block_days, complete=True, min_blocks=2)
    output = environment_mean(zonal)
    blocks = xr.concat(
        [environment_mean(zonal.isel(Time=s)) for s in slices],
        dim=xr.IndexVariable("block", np.arange(len(slices))),
    )
    for name in ENVIRONMENT_FIELDS:
        output[f"{name}_blocks"] = blocks[name]
    output["block_start"] = ("block", [zonal.Time.values[s.start] for s in slices])
    output["block_end"] = ("block", [zonal.Time.values[s.stop - 1] for s in slices])
    return output


def build_surface(simulation):
    start = str(pd.Timestamp(ANALYSIS_START) - pd.Timedelta(hours=6))
    source = load_latlon_dataset(
        str(simulation_directory(simulation)), LATLON_PATTERN,
        chunks={"Time": 8}, start=start,
    )
    try:
        selected = standardize_latlon_dataset(
            source, ["rainnc", "i_rainnc", "rainc", "i_rainc", "olrtoa", "precipw"]
        )
        selected = subset_between(selected, "latitude", (-LATITUDE_LIMIT, LATITUDE_LIMIT))
        # Spatial reductions precede materialisation. Each load returns only
        # Time x latitude profiles, rather than complete longitude histories.
        reduced = xr.Dataset({
            "accumulation": total_precip_accum(selected).astype("float64").mean("longitude"),
            "olr": selected.olrtoa.astype("float64").mean("longitude"),
            "precipitable_water": selected.precipw.astype("float64").mean("longitude"),
        })
        batches = []
        for start_index in range(0, reduced.sizes["Time"], 32):
            batches.append(materialize(
                reduced.isel(Time=slice(start_index, start_index + 32)),
                scheduler="synchronous",
            ))
        return surface_profiles(xr.concat(batches, dim="Time"))
    finally:
        source.close()


def build_environment(simulation, *, time_count=None):
    source = load_latlon_dataset(
        str(simulation_directory(simulation)), ZONAL_MEAN_PATTERN,
        chunks={"Time": 32},
    )
    try:
        selected = standardize_latlon_dataset(
            source, ["theta", "pressure", "t", "uReconstructZonal"]
        )
        selected = subset_between(selected, "latitude", DEEP_TROPICAL_LAT_BOUNDS)
        if time_count is not None:
            selected = selected.isel(Time=slice(0, time_count))
        selected = materialize(selected, scheduler="synchronous")
        days = BLOCK_DAYS if time_count is None else min(BLOCK_DAYS, (time_count // 2) * 0.25)
        return environment_profiles(selected, block_days=days)
    finally:
        source.close()


def humidity_ascent_profiles(reduced, *, block_days=BLOCK_DAYS):
    """Fold both hemispheres and retain complete blocks on physical pressure levels."""
    samples = subset_between(
        xr.Dataset({name: hemispheric_fold(reduced[name], parity="even") for name in MAP_FIELDS}),
        "latitude", (0.0, TROPICAL_LAT_BOUNDS[1])
    ).rename(latitude="absolute_latitude")
    slices = time_block_slices(samples.specific_humidity, block_days, complete=True, min_blocks=2)
    output = samples.mean("Time")
    blocks = xr.concat(
        [samples.isel(Time=s).mean("Time") for s in slices],
        dim=xr.IndexVariable("block", np.arange(len(slices))),
    )
    for name in MAP_FIELDS:
        output[f"{name}_blocks"] = blocks[name]
    output["block_start"] = ("block", [samples.Time.values[s.start] for s in slices])
    output["block_end"] = ("block", [samples.Time.values[s.stop - 1] for s in slices])
    output.specific_humidity.attrs.update(
        units="kg kg-1", source_field="qv_isobaric",
        conversion="r/(1+r) before longitude, time, and hemisphere averaging",
    )
    output.upward_pressure_velocity.attrs.update(
        units="Pa s-1", source_field="omg_isobaric", sign_convention="positive represents ascent (-omega)"
    )
    return output


def specific_humidity_zonal_mean(source):
    """Convert local dry-air mixing ratios before the longitude reduction."""
    return mixing_ratio_to_specific_humidity(source.qv_isobaric.astype("float64")).mean("longitude")


def build_humidity_ascent(simulation):
    sources = []
    try:
        for pattern in (LATLON_PATTERN, ZONAL_MEAN_PATTERN):
            sources.append(load_latlon_dataset(
                str(simulation_directory(simulation)), pattern, chunks={"Time": 8}
            ))
        vapour = subset_between(
            standardize_latlon_dataset(sources[0], ["qv_isobaric"]), "latitude", TROPICAL_LAT_BOUNDS
        )
        omega = subset_between(
            standardize_latlon_dataset(sources[1], ["omg_isobaric"]), "latitude", TROPICAL_LAT_BOUNDS
        )
        humidity, ascent = xr.align(specific_humidity_zonal_mean(vapour), -omega.omg_isobaric, join="exact")
        reduced = xr.Dataset({"specific_humidity": humidity, "upward_pressure_velocity": ascent})
        batches = [materialize(reduced.isel(Time=slice(i, i + 32)), scheduler="synchronous")
                   for i in range(0, reduced.sizes["Time"], 32)]
        return humidity_ascent_profiles(xr.concat(batches, dim="Time"))
    finally:
        for source in sources:
            source.close()


def ascent_profiles(zonal, *, block_days=BLOCK_DAYS):
    """Average native vertical velocity before resampling complete time blocks."""
    deep = subset_between(zonal, "latitude", DEEP_TROPICAL_LAT_BOUNDS)
    velocity = vertical_velocity_from_zm(deep).astype("float64")
    slices = time_block_slices(velocity, block_days, complete=True, min_blocks=2)
    output = xr.Dataset({
        "vertical_velocity": _latitude_mean(velocity.mean("Time")),
        "vertical_velocity_blocks": xr.concat(
            [_latitude_mean(velocity.isel(Time=s).mean("Time")) for s in slices],
            dim=xr.IndexVariable("block", np.arange(len(slices))),
        ),
        "pressure_hpa": _latitude_mean(deep.pressure.astype("float64").mean("Time")) / 100.0,
        "block_start": ("block", [velocity.Time.values[s.start] for s in slices]),
        "block_end": ("block", [velocity.Time.values[s.stop - 1] for s in slices]),
    })
    for name in ("vertical_velocity", "vertical_velocity_blocks"):
        output[name].attrs.update(
            units="m s-1", source_field="w", sign_convention="positive represents upward motion",
            latitude_average="cosine-weighted mean over 10 degrees S–10 degrees N",
        )
    return output


def build_ascent(simulation, *, time_count=None):
    source = load_latlon_dataset(
        str(simulation_directory(simulation)), ZONAL_MEAN_PATTERN, chunks={"Time": 32}
    )
    try:
        selected = subset_between(
            standardize_latlon_dataset(source, ["w", "pressure"]), "latitude", DEEP_TROPICAL_LAT_BOUNDS
        )
        if time_count is not None:
            selected = selected.isel(Time=slice(0, time_count))
        days = BLOCK_DAYS if time_count is None else min(BLOCK_DAYS, (time_count // 2) * 0.25)
        return ascent_profiles(materialize(selected, scheduler="synchronous"), block_days=days)
    finally:
        source.close()


def prepare_product(simulation, family, *, figures_only=False, rebuild=False, time_count=None):
    """Validate and reuse one product without rebuilding any upstream owner."""
    if time_count is not None and family not in {"environment", "ascent"}:
        raise ValueError("Bounded test inputs apply only to mean-state profiles")
    extra = {} if time_count is None else {"time_count": time_count}
    path, metadata_path = product_paths(simulation, family, **extra)
    manifest = product_manifest(simulation, family, **extra)
    if not rebuild and path.is_file() and cache_manifest_matches(
        read_json_file(metadata_path), manifest
    ):
        return load_netcdf(path)
    if figures_only:
        command = ("mean_state_diagnostics.py --figures 2" if family == "environment"
                   else "convection_summary.py")
        raise RuntimeError(
            f"Missing or stale {family} summary for {simulation}: {path}. "
            f"Run {command} with the selected experiment profile, without --figures-only."
        )
    builder = {"surface": build_surface, "environment": build_environment,
               "humidity_ascent": build_humidity_ascent, "ascent": build_ascent}[family]
    product = builder(simulation, **extra)
    product.attrs.update(
        cache_schema=SCHEMAS[family], simulation=simulation,
        analysis_start=ANALYSIS_START, analysis_end=ANALYSIS_END,
        block_days=BLOCK_DAYS,
    )
    if time_count is not None:
        product.attrs.update(test_time_count=time_count, block_days=min(BLOCK_DAYS, (time_count // 2) * 0.25))
    write_netcdf_atomic(product, path)
    write_json_file(metadata_path, manifest)
    return product
