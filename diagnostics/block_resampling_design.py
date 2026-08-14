#!/usr/bin/env python3
"""Review control-series block lengths for every active uncertainty family.

The project workflow builds compact daily influence series from the control
simulation.  It reports autocorrelation scales, the corrected Politis--White
automatic selector, and sensitivity across practical candidate lengths.
It never edits the production configuration.

Method references
-----------------
Kunsch (1989), Annals of Statistics, doi:10.1214/aos/1176347265.
Wilks (1997), Journal of Climate,
doi:10.1175/1520-0442(1997)010<0065:RHTFAF>2.0.CO;2.
Politis and White (2004), Econometric Reviews,
doi:10.1081/ETC-120028836.
Patton, Politis, and White (2009), Econometric Reviews,
doi:10.1080/07474930802459016.
Buhlmann and Kunsch (1999), Computational Statistics and Data Analysis,
doi:10.1016/S0167-9473(99)00014-6.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

WORKFLOW_ROOT = Path(__file__).resolve().parents[1]
if str(WORKFLOW_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKFLOW_ROOT))

from diagnostics.analysis_config import (
    ANALYSIS_END,
    ANALYSIS_START,
    BLOCK_LENGTH_DAYS,
    BOOTSTRAP_SEED,
    CONFIDENCE_PERCENTILES,
    LATLON_PATTERN,
    SIMULATION_ARCHIVES,
    SIMULATION_ROOT,
    TROPICAL_LAT_BOUNDS,
    WAVE_RESPONSE_BLOCK_LENGTH_DAYS,
    ZONAL_MEAN_PATTERN,
    cache_directory,
)
from diagnostics.ttl import TTL_HEIGHT_BOUNDS_M
from diagnostics.waves.equatorial_wave_config import (
    FILTERED_WAVE_PRODUCT_VERSION,
    WAVE_BOXES,
    WAVE_LATITUDE_PROJECTION,
    WAVE_RESPONSE_FILTERS,
)


CANDIDATE_DAYS = (2.5, 5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0)
CONTROL_DIRECTORY = SIMULATION_ROOT / SIMULATION_ARCHIVES["control"]
PROJECT_OUTPUT = cache_directory("block_resampling")
CONTROL_SERIES_FILE = "control_influence_series.nc"
CONTROL_SERIES_MANIFEST = "control_influence_series.manifest.json"
SECONDS_PER_DAY = 86400.0
EARTH_RADIUS_M = 6_371_000.0
GRAVITY = 9.80665
REFERENCE_PRESSURE_PA = 100_000.0
RD_OVER_CP = 287.0 / 1004.0

HIGH_CLOUD_VARIABLES = (
    "high_cloud_mask",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Review production block lengths from control data."
    )
    parser.add_argument(
        "input",
        nargs="?",
        type=Path,
        help="Optional compact NetCDF or CSV scalar series.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Review every active project uncertainty family.",
    )
    parser.add_argument("--variable", help="Variable or CSV column for one input.")
    parser.add_argument("--time", default="Time", help="Time coordinate or CSV column.")
    parser.add_argument("--family", help="Owning family for one input.")
    parser.add_argument(
        "--candidate-days",
        nargs="+",
        type=float,
        default=list(CANDIDATE_DAYS),
    )
    parser.add_argument(
        "--resamples",
        type=int,
        default=2000,
        help="Resamples used for candidate sensitivity.",
    )
    parser.add_argument("--seed", type=int, default=BOOTSTRAP_SEED)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild the compact control-series product.",
    )
    args = parser.parse_args(argv)
    if args.all:
        if args.input or args.variable or args.family:
            parser.error("--all does not accept input, --variable, or --family")
    elif not (args.input and args.variable and args.family):
        parser.error("use --all or supply input, --variable, and --family")
    return args


def load_series(path: Path, variable: str, time_name: str):
    """Load one compact scalar time series and its time coordinate."""

    if path.suffix.lower() == ".csv":
        table = pd.read_csv(path, usecols=[time_name, variable])
        return pd.to_datetime(table[time_name]).to_numpy(), table[variable].to_numpy(float)
    with xr.open_dataset(path) as dataset:
        data = dataset[variable]
        other_dims = [dimension for dimension in data.dims if dimension != time_name]
        if other_dims:
            raise ValueError(f"{variable} must be scalar over time; found {data.dims}")
        return data[time_name].values, np.asarray(data.values, dtype=float)


def autocorrelation(values, max_lag):
    """Calculate the finite-sample autocorrelation through one maximum lag."""

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    anomaly = values - values.mean()
    denominator = np.dot(anomaly, anomaly)
    if values.size < 3 or denominator == 0.0:
        raise ValueError("The series requires at least three variable samples")
    maximum = min(int(max_lag), values.size - 2)
    result = np.empty(maximum + 1, dtype=float)
    result[0] = 1.0
    for lag in range(1, maximum + 1):
        result[lag] = np.dot(anomaly[:-lag], anomaly[lag:]) / denominator
    return result


def decorrelation_summary(correlation):
    """Return first e-folding and first nonpositive lags plus an integral scale."""

    below = np.flatnonzero(correlation <= np.exp(-1.0))
    nonpositive = np.flatnonzero(correlation <= 0.0)
    stop = int(nonpositive[0]) if nonpositive.size else correlation.size
    integral = 1.0 + 2.0 * np.sum(correlation[1:stop])
    return {
        "e_folding_lag": int(below[0]) if below.size else np.nan,
        "first_nonpositive_lag": int(nonpositive[0]) if nonpositive.size else np.nan,
        "integral_scale_samples": float(max(1.0, integral)),
    }


def politis_white_block_lengths(values):
    """Return corrected stationary and circular bootstrap block lengths.

    This implements Politis and White (2004), with the correction from Patton,
    Politis, and White (2009).  The input is the influence series for the
    statistic under review.
    """

    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    sample_count = values.size
    if sample_count < 10 or np.nanstd(values) == 0.0:
        return {"stationary_samples": np.nan, "circular_samples": np.nan}
    eps = values - values.mean()
    b_max = int(np.ceil(min(3.0 * np.sqrt(sample_count), sample_count / 3.0)))
    consecutive = max(5, int(np.log10(sample_count)))
    m_max = int(np.ceil(np.sqrt(sample_count))) + consecutive
    threshold = 2.0 * np.sqrt(np.log10(sample_count) / sample_count)
    autocovariance = np.zeros(m_max + 1)
    absolute_correlation = np.zeros(m_max + 1)
    variance_sum = float(np.dot(eps, eps))
    optimum_m = None
    for lag in range(m_max + 1):
        cross = float(np.dot(eps[lag:], eps[: sample_count - lag]))
        autocovariance[lag] = cross / sample_count
        absolute_correlation[lag] = abs(cross) / variance_sum
        if lag >= consecutive and optimum_m is None:
            window = absolute_correlation[lag - consecutive : lag]
            if np.all(window < threshold):
                optimum_m = lag - consecutive
    m = 2 * max(optimum_m, 1) if optimum_m is not None else m_max
    m = min(m, m_max)
    long_run_covariance = autocovariance[0]
    weighted_moment = 0.0
    for lag in range(1, m + 1):
        ratio = lag / m
        weight = 1.0 if ratio <= 0.5 else 2.0 * (1.0 - ratio)
        long_run_covariance += 2.0 * weight * autocovariance[lag]
        weighted_moment += 2.0 * weight * lag * autocovariance[lag]
    if long_run_covariance == 0.0:
        return {"stationary_samples": 1.0, "circular_samples": 1.0}
    numerator = 2.0 * weighted_moment**2
    stationary_denominator = 2.0 * long_run_covariance**2
    circular_denominator = (4.0 / 3.0) * long_run_covariance**2
    scale = sample_count ** (1.0 / 3.0)
    stationary = max(
        1.0,
        (numerator / stationary_denominator) ** (1.0 / 3.0) * scale,
    )
    circular = max(
        1.0,
        (numerator / circular_denominator) ** (1.0 / 3.0) * scale,
    )
    return {
        "stationary_samples": float(min(stationary, b_max)),
        "circular_samples": float(min(circular, b_max)),
    }


def moving_block_mean_replicates(values, block_samples, n_resamples, rng):
    """Return memory-bounded moving-block bootstrap replicates of the mean."""

    values = np.asarray(values, dtype=float)
    block_samples = int(block_samples)
    if block_samples < 1 or block_samples > values.size:
        raise ValueError("Block length must lie within the available record")
    start_count = values.size - block_samples + 1
    block_sums = np.convolve(values, np.ones(block_samples), mode="valid")
    complete = values.size // block_samples
    remainder = values.size % block_samples
    remainder_sums = None
    if remainder:
        remainder_sums = np.convolve(values, np.ones(remainder), mode="valid")
    replicates = np.empty(int(n_resamples), dtype=float)
    for index in range(replicates.size):
        starts = rng.integers(0, start_count, size=complete)
        total = float(block_sums[starts].sum())
        if remainder:
            remainder_start = int(rng.integers(0, values.size - remainder + 1))
            total += float(remainder_sums[remainder_start])
        replicates[index] = total / values.size
    return replicates


def candidate_table(
    candidate_days,
    step_days,
    values,
    summary,
    *,
    n_resamples=2000,
    seed=BOOTSTRAP_SEED,
):
    """Describe decorrelation coverage and mean sensitivity for each candidate."""

    values = np.asarray(values, dtype=float)
    sample_count = values.size
    e_folding_days = summary["e_folding_lag"] * step_days
    nonpositive_days = summary["first_nonpositive_lag"] * step_days
    integral_days = summary["integral_scale_samples"] * step_days
    independent_se = np.std(values, ddof=1) / np.sqrt(values.size)
    rows = []
    for candidate_index, days in enumerate(candidate_days):
        samples = max(1, int(round(days / step_days)))
        if samples > sample_count:
            continue
        actual_days = samples * step_days
        rng = np.random.default_rng(int(seed) + candidate_index)
        replicates = moving_block_mean_replicates(values, samples, n_resamples, rng)
        low, high = np.percentile(replicates, CONFIDENCE_PERCENTILES)
        bootstrap_se = float(np.std(replicates, ddof=1))
        rows.append(
            {
                "candidate_days": float(days),
                "actual_block_days": float(actual_days),
                "samples_per_block": samples,
                "complete_blocks": int(sample_count // samples),
                "e_folding_days": e_folding_days,
                "first_nonpositive_days": nonpositive_days,
                "integral_scale_days": integral_days,
                "candidate_to_e_folding": (
                    float(actual_days / e_folding_days)
                    if e_folding_days > 0
                    else np.nan
                ),
                "candidate_to_integral_scale": float(actual_days / integral_days),
                "bootstrap_mean_standard_error": bootstrap_se,
                "iid_mean_standard_error": float(independent_se),
                "standard_error_inflation": (
                    float(bootstrap_se / independent_se)
                    if independent_se > 0
                    else np.nan
                ),
                "interval_low": float(low),
                "interval_high": float(high),
                "interval_width": float(high - low),
            }
        )
    if not rows:
        raise ValueError("Every candidate block exceeds the available record")
    return pd.DataFrame(rows)


def _latitude_mean(field: xr.DataArray, bounds=None) -> xr.DataArray:
    if bounds is not None:
        field = field.sel(latitude=slice(*sorted(bounds)))
    weights = np.cos(np.deg2rad(field.latitude)).clip(min=0.0)
    dimensions = [name for name in ("latitude", "longitude") if name in field.dims]
    return field.weighted(weights).mean(dimensions, skipna=True)


def _daily(field: xr.DataArray, method="mean") -> xr.DataArray:
    if method == "rms":
        return np.sqrt((field * field).resample(Time="1D").mean(skipna=True))
    return field.resample(Time="1D").mean(skipna=True)


def _series(
    field: xr.DataArray,
    name: str,
    family: str,
    statistic: str,
    configured_days: float,
    source: str,
    source_status: str = "current",
) -> xr.DataArray:
    output = field.astype(np.float64).rename(name)
    output.attrs.update(
        family=family,
        statistic=statistic,
        configured_days=float(configured_days),
        source=source,
        source_status=source_status,
    )
    return output


def _surface_reductions(dataset: xr.Dataset) -> xr.Dataset:
    """Return the numerical surface influence-series reductions."""

    net_toa = (dataset.swdnt - dataset.swupt) + (dataset.lwdnt - dataset.lwupt)
    surface_radiation = (
        (dataset.swdnb - dataset.swupb) + (dataset.lwdnb - dataset.lwupb)
    )
    forcing = net_toa - surface_radiation + dataset.lh + dataset.hfx
    fields = {
        "energy_budget__atmospheric_forcing": _latitude_mean(forcing),
        "energy_budget__net_toa": _latitude_mean(net_toa),
        "thermal_wind__olr": _latitude_mean(dataset.olrtoa, (-10.0, 10.0)),
        "thermal_wind__pw": _latitude_mean(dataset.precipw, (-10.0, 10.0)),
    }
    for threshold in (500.0, 1000.0, 1500.0):
        key = f"mean_state_distributions__cape_gt_{threshold:g}_fraction"
        fields[key] = _latitude_mean(
            (dataset.cape >= threshold).astype(float), TROPICAL_LAT_BOUNDS
        )
    for threshold in (30.0, 40.0, 50.0):
        key = f"mean_state_distributions__pw_gt_{threshold:g}_fraction"
        fields[key] = _latitude_mean(
            (dataset.precipw >= threshold).astype(float), TROPICAL_LAT_BOUNDS
        )
    return xr.Dataset(fields)


def _surface_control_series() -> list[xr.DataArray]:
    variables = (
        "swdnt", "swupt", "lwdnt", "lwupt", "swdnb", "swupb",
        "lwdnb", "lwupb", "lh", "hfx", "cape", "precipw", "olrtoa",
    )
    monthly = []
    paths = sorted(CONTROL_DIRECTORY.glob(LATLON_PATTERN))
    for path in paths:
        print(f"Reducing surface control data: {path.name}")
        with xr.open_dataset(
            path,
            chunks={"Time": 8},
        ) as dataset:
            dataset = dataset[list(variables)].sel(Time=slice(ANALYSIS_START, ANALYSIS_END))
            reduced = _surface_reductions(dataset).compute()
            monthly.append(reduced)
    if not monthly:
        raise FileNotFoundError(f"No control files match {LATLON_PATTERN}")
    combined = xr.concat(monthly, dim="Time").sortby("Time")
    combined = combined.isel(Time=~combined.get_index("Time").duplicated())
    combined = _daily(combined)
    definitions = {
        "energy_budget__atmospheric_forcing": (
            "energy_budget", "mean atmospheric forcing", BLOCK_LENGTH_DAYS["energy_budget"]
        ),
        "energy_budget__net_toa": (
            "energy_budget", "mean net TOA input", BLOCK_LENGTH_DAYS["energy_budget"]
        ),
        "thermal_wind__olr": (
            "thermal_wind", "deep-tropical mean OLR", BLOCK_LENGTH_DAYS["thermal_wind"]
        ),
        "thermal_wind__pw": (
            "thermal_wind", "deep-tropical mean precipitable water", BLOCK_LENGTH_DAYS["thermal_wind"]
        ),
    }
    for threshold in (500.0, 1000.0, 1500.0):
        definitions[f"mean_state_distributions__cape_gt_{threshold:g}_fraction"] = (
            "mean_state_distributions",
            f"CAPE PDF probability above {threshold:g} J kg-1",
            BLOCK_LENGTH_DAYS["mean_state_distributions"],
        )
    for threshold in (30.0, 40.0, 50.0):
        definitions[f"mean_state_distributions__pw_gt_{threshold:g}_fraction"] = (
            "mean_state_distributions",
            f"PW PDF probability above {threshold:g} kg m-2",
            BLOCK_LENGTH_DAYS["mean_state_distributions"],
        )
    return [
        _series(combined[name], name, *definitions[name], source="control latlon archive")
        for name in combined.data_vars
    ]


def _hadley_strength_series(v_wind: xr.DataArray) -> dict[str, xr.DataArray]:
    daily = _daily(v_wind).transpose("Time", "pres", "latitude").load()
    pressure = np.asarray(daily.pres.values, dtype=float) * 100.0
    values = np.asarray(daily.values, dtype=float)
    integral = np.zeros_like(values)
    increments = 0.5 * (values[:, 1:] + values[:, :-1])
    increments *= np.diff(pressure)[None, :, None]
    integral[:, 1:] = np.cumsum(increments, axis=1)
    factor = (
        2.0 * np.pi * EARTH_RADIUS_M
        * np.cos(np.deg2rad(np.asarray(daily.latitude.values)))[None, None, :]
        / GRAVITY
    )
    streamfunction = xr.DataArray(
        integral * factor,
        dims=daily.dims,
        coords=daily.coords,
    )
    layer = streamfunction.sel(pres=slice(300.0, 700.0)).integrate("pres") / 400.0
    output = {}
    for hemisphere, bounds in {"sh": (-40.0, -0.5), "nh": (0.5, 40.0)}.items():
        cell = layer.sel(latitude=slice(*bounds))
        output[hemisphere] = np.abs(cell).max("latitude", skipna=True)
    return output


def _zonal_reductions(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Return the numerical zonal-mean influence-series reductions."""

    daily_u250 = _daily(dataset.uzonal_isobaric.sel(pres=250.0)).load()
    hadley = _hadley_strength_series(dataset.umeridional_isobaric)
    jet = {}
    for hemisphere, bounds in {"sh": (-85.0, -5.0), "nh": (5.0, 85.0)}.items():
        wind = daily_u250.sel(latitude=slice(*bounds))
        jet[hemisphere] = np.abs(wind.idxmax("latitude", skipna=True))

    theta = _daily(dataset.theta).load()
    pressure = _daily(dataset.pressure).load()
    temperature = theta * (pressure / REFERENCE_PRESSURE_PA) ** RD_OVER_CP
    ttl = temperature.sel(z=slice(*TTL_HEIGHT_BOUNDS_M), latitude=slice(-10.0, 10.0))
    lat_weights = np.cos(np.deg2rad(ttl.latitude))
    height_weights = xr.DataArray(
        np.gradient(np.asarray(ttl.z.values, dtype=float)),
        dims="z",
        coords={"z": ttl.z},
    )
    ttl_temperature = ttl.weighted(lat_weights * height_weights).mean(
        ("z", "latitude"), skipna=True
    )
    temperature_16km = temperature.interp(z=16_000.0).sel(latitude=slice(-10.0, 10.0))
    temperature_16km = temperature_16km.weighted(
        np.cos(np.deg2rad(temperature_16km.latitude))
    ).mean("latitude", skipna=True)
    theta_values = np.asarray(theta.values, dtype=float)
    n2_values = (
        GRAVITY / theta_values
        * np.gradient(theta_values, np.asarray(theta.z.values), axis=1)
    )
    n2 = xr.DataArray(n2_values, dims=theta.dims, coords=theta.coords)
    ttl_n2 = n2.sel(z=slice(*TTL_HEIGHT_BOUNDS_M), latitude=slice(-10.0, 10.0))
    ttl_n2 = ttl_n2.weighted(
        np.cos(np.deg2rad(ttl_n2.latitude)) * height_weights
    ).mean(("z", "latitude"), skipna=True)
    return {
        "jet_sh": jet["sh"],
        "jet_nh": jet["nh"],
        "hadley_sh": hadley["sh"],
        "hadley_nh": hadley["nh"],
        "temperature_16km": temperature_16km,
        "ttl_temperature": ttl_temperature,
        "ttl_n2": ttl_n2,
    }


def _zonal_control_series() -> list[xr.DataArray]:
    paths = sorted(CONTROL_DIRECTORY.glob(ZONAL_MEAN_PATTERN))
    if not paths:
        raise FileNotFoundError(f"No control files match {ZONAL_MEAN_PATTERN}")
    print("Reducing zonal-mean control data")
    dataset = xr.open_mfdataset(
        paths,
        combine="by_coords",
        chunks={"Time": 40},
        data_vars="minimal",
        coords="minimal",
        compat="override",
    ).sel(Time=slice(ANALYSIS_START, ANALYSIS_END))
    try:
        reduced = _zonal_reductions(dataset)
    finally:
        dataset.close()

    output = []
    for hemisphere in ("sh", "nh"):
        output.append(
            _series(
                reduced[f"jet_{hemisphere}"],
                f"jet_variability__{hemisphere}_jet_latitude",
                "jet_variability",
                f"{hemisphere.upper()} 250-hPa jet latitude",
                BLOCK_LENGTH_DAYS["jet_variability"],
                "control zonal-mean archive",
            )
        )
        output.append(
            _series(
                reduced[f"hadley_{hemisphere}"],
                f"circulation__{hemisphere}_hadley_strength",
                "circulation",
                f"{hemisphere.upper()} Hadley-cell strength",
                BLOCK_LENGTH_DAYS["circulation"],
                "control zonal-mean archive",
            )
        )
    for field, name, statistic in (
        (reduced["temperature_16km"], "thermal_wind__temperature_16km", "deep-tropical 16-km temperature"),
        (reduced["ttl_temperature"], "thermal_wind__ttl_temperature", "deep-tropical TTL temperature"),
        (reduced["ttl_n2"], "thermal_wind__ttl_n2", "deep-tropical TTL static stability"),
    ):
        output.append(
            _series(
                field,
                name,
                "thermal_wind",
                statistic,
                BLOCK_LENGTH_DAYS["thermal_wind"],
                "control zonal-mean archive",
            )
        )
    return output


def _high_cloud_reductions(dataset: xr.Dataset) -> dict[str, xr.DataArray]:
    """Return numerical daily high-cloud occurrence series."""

    return {
        variable: _daily(
            _latitude_mean(dataset[variable].astype(float), TROPICAL_LAT_BOUNDS)
        ).compute()
        for variable in HIGH_CLOUD_VARIABLES
    }


def _high_cloud_control_series() -> list[xr.DataArray]:
    path = cache_directory(
        "classification", "control_classified_high_cloud.nc"
    )
    if not path.is_file():
        raise FileNotFoundError(f"Missing high-cloud classification cache: {path}")
    print("Reducing control high-cloud classifications")
    output = []
    with xr.open_dataset(
        path,
        chunks={"Time": 8, "latitude": 90, "longitude": 180},
    ) as dataset:
        reduced = _high_cloud_reductions(dataset)
        for variable in HIGH_CLOUD_VARIABLES:
            name = f"high_cloud__{variable}"
            output.append(
                _series(
                    reduced[variable],
                    name,
                    "high_cloud",
                    f"tropical domain fraction for {variable}",
                    BLOCK_LENGTH_DAYS["high_cloud"],
                    f"{path.name}; {dataset.attrs.get('cache_version', 'unknown cache')}",
                )
            )
    return output


def _wave_cache_path(wave_name: str) -> Path:
    projection = WAVE_LATITUDE_PROJECTION[wave_name]
    return cache_directory(
        "waves",
        f"control_Precip_{wave_name}_m10_10_avgLat_{projection}.nc",
    )


def _wave_amplitude(wave: xr.DataArray) -> xr.DataArray:
    """Return the daily longitudinal RMS filtered-wave amplitude."""

    return _daily(np.sqrt((wave * wave).mean("longitude", skipna=True))).compute()


def _wave_control_series() -> list[xr.DataArray]:
    output = []
    for wave_name in WAVE_RESPONSE_FILTERS:
        path = _wave_cache_path(wave_name)
        if not path.is_file():
            raise FileNotFoundError(f"Missing filtered control wave cache: {path}")
        with xr.open_dataset(path, chunks={"Time": 80}) as dataset:
            wave = dataset.wave
            amplitude = _wave_amplitude(wave)
            block_days = WAVE_RESPONSE_BLOCK_LENGTH_DAYS[wave_name]
            saved_version = str(wave.attrs.get("cache_version", "unknown"))
            source = f"{path.name}; {saved_version}"
            if saved_version != FILTERED_WAVE_PRODUCT_VERSION:
                source += f"; current version is {FILTERED_WAVE_PRODUCT_VERSION}"
            source_status = (
                "current"
                if saved_version == FILTERED_WAVE_PRODUCT_VERSION
                else "stale filtered cache"
            )
            output.append(
                _series(
                    amplitude,
                    f"wave_response__{wave_name.lower()}_precip_amplitude",
                    "wave_response",
                    f"{wave_name} filtered precipitation RMS amplitude",
                    block_days,
                    source,
                    source_status,
                )
            )
            output[-1].attrs["wave_name"] = wave_name
    return output


def _source_paths() -> list[Path]:
    paths = sorted(CONTROL_DIRECTORY.glob(LATLON_PATTERN))
    paths += sorted(CONTROL_DIRECTORY.glob(ZONAL_MEAN_PATTERN))
    paths.append(
        cache_directory("classification", "control_classified_high_cloud.nc")
    )
    paths += [_wave_cache_path(name) for name in WAVE_RESPONSE_FILTERS]
    return paths


def _calculation_fingerprint() -> str:
    functions = (
        _latitude_mean,
        _daily,
        _surface_reductions,
        _hadley_strength_series,
        _zonal_reductions,
        _high_cloud_reductions,
        _wave_amplitude,
    )
    payload = "\n".join(inspect.getsource(function) for function in functions)
    payload += json.dumps(
        {
            "filtered_wave_product": FILTERED_WAVE_PRODUCT_VERSION,
            "wave_periods": {
                name: WAVE_BOXES[name]["period"] for name in WAVE_RESPONSE_FILTERS
            },
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _control_manifest() -> dict:
    sources = []
    for path in _source_paths():
        if not path.is_file():
            sources.append({"path": str(path), "missing": True})
            continue
        status = path.stat()
        sources.append(
            {"path": str(path), "size": status.st_size, "mtime_ns": status.st_mtime_ns}
        )
    return {
        "calculation_fingerprint": _calculation_fingerprint(),
        "analysis_start": ANALYSIS_START,
        "analysis_end": ANALYSIS_END,
        "simulation": "control",
        "sources": sources,
    }


def _atomic_json(payload: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _atomic_dataset(dataset: xr.Dataset, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    dataset.to_netcdf(temporary)
    os.replace(temporary, path)


def build_control_series(output_dir: Path, rebuild=False) -> xr.Dataset:
    """Build or reuse the compact control influence-series product."""

    output_dir.mkdir(parents=True, exist_ok=True)
    product_path = output_dir / CONTROL_SERIES_FILE
    manifest_path = output_dir / CONTROL_SERIES_MANIFEST
    manifest = _control_manifest()
    if product_path.is_file() and manifest_path.is_file() and not rebuild:
        saved = json.loads(manifest_path.read_text())
        if saved == manifest:
            print(f"Using current compact control series: {product_path}")
            dataset = xr.open_dataset(product_path).load()
            for name, field in dataset.data_vars.items():
                family = field.attrs["family"]
                if family in BLOCK_LENGTH_DAYS:
                    field.attrs["configured_days"] = BLOCK_LENGTH_DAYS[family]
                elif family == "wave_response":
                    wave_name = field.attrs.get("wave_name")
                    if wave_name is None:
                        matches = [
                            candidate
                            for candidate in WAVE_RESPONSE_FILTERS
                            if name.startswith(
                                f"wave_response__{candidate.lower()}_"
                            )
                        ]
                        if len(matches) != 1:
                            raise ValueError(
                                f"Cannot identify the wave band for {name!r}"
                            )
                        wave_name = matches[0]
                        field.attrs["wave_name"] = wave_name
                    field.attrs["configured_days"] = (
                        WAVE_RESPONSE_BLOCK_LENGTH_DAYS[wave_name]
                    )
                field.attrs["source_status"] = (
                    "stale filtered cache"
                    if "current version is" in str(field.attrs.get("source", ""))
                    else "current"
                )
            return dataset
    fields = []
    fields.extend(_surface_control_series())
    fields.extend(_zonal_control_series())
    fields.extend(_high_cloud_control_series())
    fields.extend(_wave_control_series())
    dataset = xr.merge(fields, join="outer", compat="no_conflicts").sortby("Time")
    dataset.attrs.update(
        description="Daily control influence series for block-length review",
        calculation_fingerprint=manifest["calculation_fingerprint"],
        note="This compact review product does not replace archive variables.",
    )
    _atomic_dataset(dataset, product_path)
    _atomic_json(manifest, manifest_path)
    return dataset


def analyse_one_series(
    times,
    values,
    candidate_days,
    n_resamples,
    seed,
):
    finite = np.isfinite(values)
    times = pd.DatetimeIndex(np.asarray(times)[finite])
    values = np.asarray(values, dtype=float)[finite]
    if len(times) < 10:
        raise ValueError("The series requires at least ten finite time samples")
    steps = np.diff(times.values).astype("timedelta64[s]").astype(float) / SECONDS_PER_DAY
    step_days = float(np.median(steps))
    if not np.allclose(steps, step_days):
        raise ValueError("The compact time series must use a regular time step")
    maximum_days = max(max(candidate_days), 60.0)
    correlation = autocorrelation(values, np.ceil(maximum_days / step_days))
    summary = decorrelation_summary(correlation)
    automatic = politis_white_block_lengths(values)
    table = candidate_table(
        candidate_days,
        step_days,
        values,
        summary,
        n_resamples=n_resamples,
        seed=seed,
    )
    details = {
        "sample_count": values.size,
        "time_step_days": step_days,
        "record_days": values.size * step_days,
        "sample_mean": float(np.mean(values)),
        "e_folding_days": summary["e_folding_lag"] * step_days,
        "first_nonpositive_days": summary["first_nonpositive_lag"] * step_days,
        "integral_scale_days": summary["integral_scale_samples"] * step_days,
        "politis_white_stationary_days": automatic["stationary_samples"] * step_days,
        "politis_white_circular_days": automatic["circular_samples"] * step_days,
    }
    return values, correlation, table, details


def _write_series_figure(
    family: str,
    rows: list[dict],
    correlations: list[tuple[str, np.ndarray, float]],
    output: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(7.2, 7.0), constrained_layout=True)
    for label, correlation, step_days in correlations:
        axes[0].plot(np.arange(correlation.size) * step_days, correlation, label=label)
    axes[0].axhline(0.0, color="0.6", linewidth=0.8)
    axes[0].axhline(np.exp(-1.0), color="0.6", linewidth=0.8, linestyle="--")
    axes[0].set(
        xlabel="Lag (days)",
        ylabel="Autocorrelation",
        title=family.replace("_", " "),
    )
    axes[0].legend(frameon=False, fontsize=7)
    positions = np.arange(len(rows))
    configured = [row["configured_days"] for row in rows]
    automatic = [row["politis_white_circular_days"] for row in rows]
    axes[1].scatter(configured, positions, marker="s", label="Current production")
    axes[1].scatter(automatic, positions, marker="o", label="Corrected Politis–White")
    axes[1].set(
        xlabel="Block length (days)",
        yticks=positions,
        yticklabels=[row["statistic"] for row in rows],
    )
    axes[1].legend(frameon=False)
    figure.savefig(output, dpi=200)
    plt.close(figure)


def _write_method_report(summary: pd.DataFrame, output_dir: Path) -> None:
    lines = [
        "# Choosing time-block lengths for uncertainty calculations",
        "",
        "Values on neighbouring days are often similar because weather systems persist.",
        "An uncertainty calculation must preserve part of this connection between days.",
        "The workflow therefore resamples groups of consecutive days instead of single days.",
        "",
        "The control simulation determines one block length for each group of related analyses.",
        "The same length then applies to the control and both nudged simulations.",
        "The script does not edit `analysis_config.py`.",
        "",
        "## What the script measures",
        "",
        "The script first calculates one control value for each day and each reported quantity.",
        "Examples include daily jet latitude, tropical cloud fraction, and mean atmospheric energy input.",
        "For distribution plots, it calculates the daily fraction above several CAPE or water-vapour thresholds.",
        "These daily values represent the actual quantities whose uncertainty the publication figures report.",
        "",
        "Correlation measures the similarity between values separated by a chosen number of days.",
        "The script then measures how correlation decreases as the separation between days increases.",
        "It records when this similarity first falls below 37 percent of its same-day value.",
        "It also records when the similarity first reaches zero or becomes negative.",
        "Finally, it sums the positive correlations to describe the total persistence in each daily record.",
        "",
        "A corrected Politis–White calculation provides an automatic block-length estimate.",
        "This calculation combines the correlations across lags and accounts for the available record length.",
        "The method follows [Politis and White (2004)](https://doi.org/10.1081/ETC-120028836).",
        "It uses the correction from [Patton et al. (2009)](https://doi.org/10.1080/07474930802459016).",
        "",
        "The script also tests practical lengths between 2.5 and 60 days.",
        "For each length, it reports the available block count and the resulting uncertainty in the mean.",
        "No single number gives a uniquely correct answer.",
        "The final choice balances preserved persistence against the number of blocks available for resampling.",
        "",
        "## Results and suggested choices",
        "",
        "The control record contains 201 daily values.",
        "A 20-day choice supplies ten complete blocks, while a 25-day choice supplies only eight.",
        "",
        "| Analysis group | Present length | Length calculated from correlations | Suggested length |",
        "|---|---:|---:|---:|",
    ]
    suggestions = {
        "circulation": 10,
        "energy_budget": 10,
        "high_cloud": 20,
        "jet_variability": 20,
        "mean_state_distributions": 20,
        "thermal_wind": 20,
        "wave_response": "10 or 20 days by wave band",
    }
    family_labels = {
        "circulation": "Large-scale circulation",
        "energy_budget": "Atmospheric energy budget",
        "high_cloud": "High-cloud occurrence",
        "jet_variability": "Jet latitude",
        "mean_state_distributions": "CAPE and atmospheric water distributions",
        "thermal_wind": "Temperature, water, radiation, and static stability",
        "wave_response": "Equatorial-wave relationships",
    }
    for family, table in summary.groupby("family", sort=True):
        current = sorted(set(np.round(table.configured_days, 6)))
        current_text = ", ".join(f"{value:g}" for value in current)
        finite_pw = table.politis_white_circular_days.dropna()
        pw_text = "unavailable" if finite_pw.empty else f"{finite_pw.min():.2f}–{finite_pw.max():.2f}"
        suggestion = suggestions[family]
        suggestion_text = (
            str(suggestion)
            if isinstance(suggestion, str)
            else f"{suggestion:g} days"
        )
        lines.append(
            f"| {family_labels[family]} | {current_text} days | {pw_text} days | "
            f"{suggestion_text} |"
        )
    lines.extend(
        [
            "",
            "Ten days remain adequate for circulation and energy-budget calculations.",
            "The other four non-wave groups show substantially longer persistence.",
            "Twenty days closely matches their automatic estimates while retaining ten complete blocks.",
            "",
            "## Wave calculations",
            "",
            "The wave analysis first isolates variations within each chosen range of periods and spatial scales.",
            "Its figures compare the isolated precipitation changes with matching cloud or outgoing-radiation changes.",
            "The daily strength of each isolated precipitation variation supplies the control persistence measure.",
            "Kelvin, mixed Rossby-gravity, and equatorial Rossby waves use 10-day blocks.",
            "Easterly disturbances and eastward intraseasonal variability use 20-day blocks.",
            "The ER correlation changes sign, but it crosses the dependence threshold near the other 10-day bands.",
            "",
            "## Literature basis",
            "",
            "[Künsch (1989)](https://doi.org/10.1214/aos/1176347265) develops block resampling for dependent observations.",
            "[Wilks (1997)](https://doi.org/10.1175/1520-0442(1997)010%3C0065:RHTFAF%3E2.0.CO;2) applies this approach to autocorrelated climate fields.",
            "[Bühlmann and Künsch (1999)](https://doi.org/10.1016/S0167-9473(99)00014-6) show that block choice depends on the reported calculation.",
            "",
            "See `series_summary.csv` for every persistence measure and calculated block length.",
            "See each family figure for the complete correlation curves.",
        ]
    )
    stale = summary.loc[summary.source_status != "current", ["series", "source"]]
    if not stale.empty:
        lines.extend(
            [
                "",
                "## Stale source products",
                "",
                "These series remain useful for provisional persistence evidence.",
                "Rebuild their owner cache before the final block-length decision.",
                "The compact review cache invalidates after those source files change.",
                "",
            ]
        )
        for row in stale.itertuples(index=False):
            lines.append(f"- `{row.series}`: {row.source}")
    (output_dir / "README.md").write_text("\n".join(lines) + "\n")


def review_project(args) -> int:
    output_dir = args.output_dir or PROJECT_OUTPUT
    dataset = build_control_series(output_dir, rebuild=args.rebuild_cache)
    summary_rows = []
    candidate_tables = []
    correlation_tables = []
    family_correlations: dict[str, list] = {}
    for index, name in enumerate(dataset.data_vars):
        field = dataset[name]
        values, correlation, candidates, details = analyse_one_series(
            field.Time.values,
            field.values,
            args.candidate_days,
            args.resamples,
            args.seed + 100 * index,
        )
        row = {
            "series": name,
            "family": field.attrs["family"],
            "statistic": field.attrs["statistic"],
            "configured_days": float(field.attrs["configured_days"]),
            "source": field.attrs["source"],
            "source_status": field.attrs.get("source_status", "current"),
            **details,
        }
        summary_rows.append(row)
        candidates.insert(0, "series", name)
        candidates.insert(1, "family", row["family"])
        candidate_tables.append(candidates)
        correlation_tables.append(
            pd.DataFrame(
                {
                    "series": name,
                    "family": row["family"],
                    "lag_days": np.arange(correlation.size) * details["time_step_days"],
                    "autocorrelation": correlation,
                }
            )
        )
        family_correlations.setdefault(row["family"], []).append(
            (row["statistic"], correlation, details["time_step_days"])
        )
    summary = pd.DataFrame(summary_rows).sort_values(["family", "series"])
    summary.to_csv(output_dir / "series_summary.csv", index=False)
    pd.concat(candidate_tables, ignore_index=True).to_csv(
        output_dir / "candidate_sensitivity.csv", index=False
    )
    pd.concat(correlation_tables, ignore_index=True).to_csv(
        output_dir / "autocorrelation.csv", index=False
    )
    for family, correlations in family_correlations.items():
        rows = [row for row in summary_rows if row["family"] == family]
        _write_series_figure(
            family,
            rows,
            correlations,
            output_dir / f"{family}_block_length_review.png",
        )
    _write_method_report(summary, output_dir)
    dataset.close()
    print(summary.to_string(index=False))
    print(f"Wrote the complete review package to {output_dir}")
    return 0


def review_one(args) -> int:
    times, values = load_series(args.input, args.variable, args.time)
    values, correlation, table, details = analyse_one_series(
        times,
        values,
        args.candidate_days,
        args.resamples,
        args.seed,
    )
    output = (args.output_dir or PROJECT_OUTPUT) / args.family
    output.mkdir(parents=True, exist_ok=True)
    table.to_csv(output / "candidate_block_lengths.csv", index=False)
    pd.DataFrame([details]).to_csv(output / "decorrelation_summary.csv", index=False)
    pd.DataFrame(
        {"lag_days": np.arange(correlation.size) * details["time_step_days"], "autocorrelation": correlation}
    ).to_csv(output / "autocorrelation.csv", index=False)
    print(table.to_string(index=False))
    print(f"Wrote review products to {output}")
    return 0


def main(argv=None):
    args = parse_args(argv)
    if args.resamples < 2:
        raise ValueError("--resamples must be at least 2")
    return review_project(args) if args.all else review_one(args)


if __name__ == "__main__":
    raise SystemExit(main())
