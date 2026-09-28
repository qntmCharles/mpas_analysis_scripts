"""Import-safe mathematical helpers shared by MPAS wave diagnostics.

This module contains no file discovery, plotting, or top-level execution.
Definitions follow Inoue et al. (2020), Sakaeda et al. (2020), and the Kelvin
reference-ray calculation used by Lim and Son (2022).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import xarray as xr
from scipy import signal


METRICS_VERSION = "wave-metrics-v6-response-block-bootstrap"
PREDICTOR_VARIANCE_EPS = 1.0e-12


def wave_filter_field_units(field_name: str) -> str:
    """Return the canonical units for a named wave-filter input field."""

    surface_units = {
        "Precip": "mm day-1",
        "OLR": "W m-2",
        "PW": "kg m-2",
        "CAPE": "J kg-1",
    }
    if field_name in surface_units:
        return surface_units[field_name]
    for prefix, units in (
        ("RH", "%"),
        ("qv", "kg kg-1"),
        ("T", "K"),
        ("u", "m s-1"),
        ("v", "m s-1"),
        ("w", "Pa s-1"),
    ):
        level = field_name.removeprefix(prefix)
        if level == field_name or not level:
            continue
        try:
            float(level.replace("p", "."))
        except ValueError:
            continue
        return units
    raise ValueError(f"No canonical wave-filter units for {field_name!r}")


def project_latitude_for_wave_filter(
    field: xr.DataArray,
    projection: str,
    latitude_bounds: tuple[float, float],
) -> xr.DataArray:
    """Project latitude while retaining the source field metadata."""

    if "latitude" not in field.dims:
        raise ValueError("Wave-filter latitude projection requires latitude")
    if projection not in {"symmetric", "antisymmetric", "direct"}:
        raise ValueError(f"Unsupported wave latitude projection {projection}")
    lower, upper = sorted(map(float, latitude_bounds))

    def subset(lo: float, hi: float) -> xr.DataArray:
        return field.where(
            (field.latitude >= lo) & (field.latitude <= hi),
            drop=True,
        )

    def cosine_mean(value: xr.DataArray) -> xr.DataArray:
        weights = np.cos(np.deg2rad(value.latitude))
        return value.weighted(weights).mean("latitude", skipna=True)

    if projection in {"direct", "symmetric"}:
        projected = cosine_mean(subset(lower, upper))
    else:
        half_width = max(abs(lower), abs(upper))
        north = cosine_mean(subset(0.0, half_width))
        south = cosine_mean(subset(-half_width, 0.0))
        north, south = xr.align(north, south, join="inner")
        projected = 0.5 * (north - south)

    projected = projected.transpose("Time", "longitude")
    projected.name = field.name
    projected.attrs = dict(field.attrs)
    return projected


def filtered_wave_phase_diagnostics(
    filtered,
    time_indices,
    longitude,
    centroid_longitude,
    *,
    observations_per_day: float,
) -> dict[str, np.ndarray]:
    """Return eight-phase precipitation-wave metadata at moving centroids.

    Filtered precipitation and its time derivative are independently
    standardised at each longitude, following the phase-plane construction
    used by Rios-Berrios et al. Phase 5 is the positive precipitation maximum;
    phases 4 and 6 are its growing and decaying flanks.
    """

    filtered = np.asarray(filtered, dtype=float)
    if filtered.ndim != 2:
        raise ValueError("Filtered wave phase requires time by longitude data")
    if observations_per_day <= 0:
        raise ValueError("observations_per_day must be positive")
    precipitation_scale = np.nanstd(filtered, axis=0)
    precipitation_scale[precipitation_scale <= 0.0] = np.nan
    tendency = np.gradient(
        filtered, 1.0 / float(observations_per_day), axis=0
    )
    tendency_scale = np.nanstd(tendency, axis=0)
    tendency_scale[tendency_scale <= 0.0] = np.nan
    longitude = np.asarray(longitude, dtype=float)
    longitude_indices = np.asarray(
        [
            np.argmin(
                np.abs(((longitude - value + 180.0) % 360.0) - 180.0)
            )
            for value in np.asarray(centroid_longitude, dtype=float)
        ],
        dtype=int,
    )
    time_indices = np.asarray(time_indices, dtype=int)
    if len(time_indices) != len(longitude_indices):
        raise ValueError(
            "Wave phase time indices and centroid longitudes must align"
        )
    precipitation = (
        filtered[time_indices, longitude_indices]
        / precipitation_scale[longitude_indices]
    )
    normalized_tendency = (
        tendency[time_indices, longitude_indices]
        / tendency_scale[longitude_indices]
    )
    angle = np.mod(
        np.degrees(np.arctan2(-normalized_tendency, precipitation)),
        360.0,
    )
    amplitude = np.hypot(precipitation, normalized_tendency)
    valid = np.isfinite(angle) & np.isfinite(amplitude)
    phase = np.full(angle.shape, np.nan)
    phase[valid] = (
        (
            np.floor((angle[valid] + 22.5) / 45.0).astype(int)
            + 4
        )
        % 8
    ) + 1
    return {
        "phase_angle_degrees": angle,
        "phase_number": phase,
        "normalized_amplitude": amplitude,
        "strong_amplitude": valid & (amplitude > 1.0),
    }


def upward_pressure_velocity(
    dataset: xr.Dataset,
    pressure_hpa: float,
    *,
    times=None,
) -> xr.DataArray:
    """Return native isobaric ``-omega`` with ascent defined as positive."""

    if "omg_isobaric" not in dataset:
        raise RuntimeError(
            "Pressure-level vertical motion requires native MPAS "
            "omg_isobaric; geometric vertical velocity is not a fallback"
        )
    omega = dataset.omg_isobaric
    if "pres" not in omega.dims:
        raise RuntimeError(
            "Native MPAS omg_isobaric has no 'pres' pressure dimension"
        )
    try:
        selected = omega.sel(pres=float(pressure_hpa))
    except KeyError as exc:
        raise RuntimeError(
            f"Native MPAS omg_isobaric has no {pressure_hpa:g} hPa level"
        ) from exc
    if times is not None:
        selected = selected.sel(Time=times)
    out = (-selected.where(np.isfinite(selected))).rename(
        f"w{pressure_hpa:g}"
    )
    out.attrs.update(
        units="Pa s-1",
        long_name=(
            f"upward-positive pressure velocity at {pressure_hpa:g} hPa"
        ),
        source_field="omg_isobaric",
        sign_convention="positive values represent ascent (-omega)",
    )
    return out


def weighted_mean(values, weights=None) -> float:
    values = np.asarray(values, dtype=float)
    valid = np.isfinite(values)
    if weights is None:
        return float(np.mean(values[valid])) if valid.any() else np.nan
    weights = np.broadcast_to(np.asarray(weights, dtype=float), values.shape)
    valid &= np.isfinite(weights) & (weights > 0)
    return (
        float(np.sum(values[valid] * weights[valid]) / np.sum(weights[valid]))
        if valid.any()
        else np.nan
    )


def weighted_covariance(x, y, weights=None) -> float:
    """Population covariance after jointly masking invalid samples."""

    x, y = np.broadcast_arrays(
        np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    )
    valid = np.isfinite(x) & np.isfinite(y)
    if weights is None:
        weights = np.ones(x.shape, dtype=float)
    else:
        weights = np.broadcast_to(np.asarray(weights, dtype=float), x.shape)
    valid &= np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return np.nan
    w = weights[valid]
    xv = x[valid]
    yv = y[valid]
    x_mean = np.sum(w * xv) / np.sum(w)
    y_mean = np.sum(w * yv) / np.sum(w)
    return float(np.sum(w * (xv - x_mean) * (yv - y_mean)) / np.sum(w))


def filtered_wave_metrics(
    precipitation,
    response,
    *,
    response_kind: str,
    weights=None,
    variance_epsilon: float = PREDICTOR_VARIANCE_EPS,
) -> dict[str, float]:
    """Return wave amplitude and response per unit filtered precipitation."""

    precip, response = np.broadcast_arrays(
        np.asarray(precipitation, dtype=float),
        np.asarray(response, dtype=float),
    )
    if weights is None:
        sample_weights = np.ones(precip.shape, dtype=float)
    else:
        sample_weights = np.broadcast_to(
            np.asarray(weights, dtype=float), precip.shape
        )
    valid = (
        np.isfinite(precip)
        & np.isfinite(response)
        & np.isfinite(sample_weights)
        & (sample_weights > 0)
    )
    if not valid.any():
        raise ValueError(
            "Filtered precipitation and response have no joint finite samples"
        )
    # Both fields are already band-pass filtered. Preserve the published
    # zero-lag spectral-regression definition <Pf Rf>/<Pf^2> rather than
    # silently re-centring each finite sample subset. The same joint finite
    # sample and weight mask is used in numerator and denominator.
    variance = weighted_mean(
        precip[valid] * precip[valid], sample_weights[valid]
    )
    if not np.isfinite(variance) or variance <= variance_epsilon:
        raise ValueError(
            "Filtered-precipitation variance is zero or numerically negligible"
        )
    covariance = weighted_mean(
        precip[valid] * response[valid], sample_weights[valid]
    )
    sign = -1.0 if response_kind == "toa_longwave" else 1.0
    return {
        "precipitation_rms_mm_day": float(np.sqrt(variance)),
        "precipitation_variance_mm2_day2": float(variance),
        "covariance": float(covariance),
        "regression_slope": float(sign * covariance / variance),
    }


def circular_time_block_indices(
    sample_count: int,
    block_length: int,
    n_resamples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return circular moving-block indices with the original record length."""

    sample_count = int(sample_count)
    block_length = int(block_length)
    n_resamples = int(n_resamples)
    if sample_count < 1:
        raise ValueError("A block bootstrap requires at least one time sample")
    if block_length < 1:
        raise ValueError("Block length must be positive")
    if n_resamples < 1:
        raise ValueError("Bootstrap resample count must be positive")
    block_length = min(block_length, sample_count)
    blocks_per_resample = int(np.ceil(sample_count / block_length))
    starts = rng.integers(
        0,
        sample_count,
        size=(n_resamples, blocks_per_resample),
    )
    offsets = np.arange(block_length, dtype=np.int64)
    indices = (starts[..., None] + offsets) % sample_count
    return indices.reshape(n_resamples, -1)[:, :sample_count]


def block_bootstrap_filtered_wave_metrics(
    precipitation,
    response,
    *,
    response_kind: str,
    time_indices,
    variance_epsilon: float = PREDICTOR_VARIANCE_EPS,
) -> dict[str, np.ndarray]:
    """Recalculate filtered-wave metrics for supplied time resamples."""

    precip, response = np.broadcast_arrays(
        np.asarray(precipitation, dtype=float),
        np.asarray(response, dtype=float),
    )
    if precip.ndim < 1:
        raise ValueError("Block-bootstrap inputs require a time dimension")
    indices = np.asarray(time_indices, dtype=np.int64)
    if indices.ndim != 2:
        raise ValueError("Bootstrap time indices must have two dimensions")
    if np.any(indices < 0) or np.any(indices >= precip.shape[0]):
        raise IndexError("Bootstrap time indices exceed the input record")

    valid = np.isfinite(precip) & np.isfinite(response)
    spatial_axes = tuple(range(1, precip.ndim))
    time_count = np.sum(valid, axis=spatial_axes, dtype=float)
    time_x2 = np.sum(
        np.where(valid, precip * precip, 0.0),
        axis=spatial_axes,
    )
    time_xy = np.sum(
        np.where(valid, precip * response, 0.0),
        axis=spatial_axes,
    )

    count = np.sum(time_count[indices], axis=1)
    sum_x2 = np.sum(time_x2[indices], axis=1)
    sum_xy = np.sum(time_xy[indices], axis=1)
    usable = (count > 0.0) & (sum_x2 > variance_epsilon * count)
    variance = np.full(indices.shape[0], np.nan, dtype=float)
    covariance = np.full(indices.shape[0], np.nan, dtype=float)
    slope = np.full(indices.shape[0], np.nan, dtype=float)
    variance[usable] = sum_x2[usable] / count[usable]
    covariance[usable] = sum_xy[usable] / count[usable]
    sign = -1.0 if response_kind == "toa_longwave" else 1.0
    slope[usable] = sign * sum_xy[usable] / sum_x2[usable]
    return {
        "precipitation_rms_mm_day": np.sqrt(variance),
        "precipitation_variance_mm2_day2": variance,
        "covariance": covariance,
        "regression_slope": slope,
    }


def record_subsets(time: Sequence, include_halves: bool) -> dict[str, np.ndarray]:
    """Return full-record and optional contiguous half-record index masks."""

    ntime = len(time)
    masks = {"full_record": np.ones(ntime, dtype=bool)}
    if include_halves:
        midpoint = ntime // 2
        first = np.zeros(ntime, dtype=bool)
        second = np.zeros(ntime, dtype=bool)
        first[:midpoint] = True
        second[midpoint:] = True
        masks.update(first_half=first, second_half=second)
    return masks


def resolved_frequency_count(
    sample_count: int,
    interval_hours: float,
    period_bounds_days: Sequence[float],
) -> int:
    """Count positive discrete Fourier frequencies inside a period band."""

    if sample_count < 2:
        return 0
    frequencies = np.fft.rfftfreq(
        sample_count, d=float(interval_hours) / 24.0
    )
    low = 1.0 / max(period_bounds_days)
    high = 1.0 / min(period_bounds_days)
    return int(np.count_nonzero((frequencies >= low) & (frequencies <= high)))


def pressure_layer_weights(pressure_hpa: Sequence[float]) -> np.ndarray:
    """Positive pressure-layer thickness weights, normalized to sum to one."""

    pressure = np.asarray(pressure_hpa, dtype=float)
    if pressure.ndim != 1 or pressure.size < 2:
        raise ValueError("pressure_hpa must be a one-dimensional pressure grid")
    order = np.argsort(pressure)
    sorted_pressure = pressure[order]
    edges = np.empty(sorted_pressure.size + 1)
    edges[1:-1] = 0.5 * (
        sorted_pressure[:-1] + sorted_pressure[1:]
    )
    edges[0] = sorted_pressure[0] - 0.5 * (
        sorted_pressure[1] - sorted_pressure[0]
    )
    edges[-1] = sorted_pressure[-1] + 0.5 * (
        sorted_pressure[-1] - sorted_pressure[-2]
    )
    sorted_weights = np.diff(edges)
    weights = np.empty_like(sorted_weights)
    weights[order] = sorted_weights
    return weights / np.sum(weights)


def orient_vertical_motion_eofs(
    eofs: np.ndarray,
    pressure_hpa: Sequence[float],
    weights: Sequence[float],
) -> tuple[np.ndarray, tuple[int, int]]:
    """Orient EOF1 as deep ascent and EOF2 as a top-minus-bottom mode."""

    eofs = np.asarray(eofs, dtype=float).copy()
    pressure = np.asarray(pressure_hpa, dtype=float)
    weights = np.asarray(weights, dtype=float)
    if eofs.shape[0] < 2 or eofs.shape[1] != pressure.size:
        raise ValueError("Two EOFs on the supplied pressure grid are required")
    signs = [1, 1]
    if np.sum(eofs[0] * weights) < 0:
        eofs[0] *= -1
        signs[0] = -1
    log_pressure = np.log(pressure)
    midpoint = np.median(log_pressure)
    upper = log_pressure < midpoint
    lower = log_pressure >= midpoint
    contrast = weighted_mean(eofs[1][upper], weights[upper]) - weighted_mean(
        eofs[1][lower], weights[lower]
    )
    if contrast < 0:
        eofs[1] *= -1
        signs[1] = -1
    return eofs, (signs[0], signs[1])


def common_vertical_motion_eofs(
    upward_motion_profiles: np.ndarray,
    pressure_hpa: Sequence[float],
    *,
    weights: Sequence[float] | None = None,
) -> dict:
    """Derive a pressure-weighted, physically oriented two-EOF basis."""

    profiles = np.asarray(upward_motion_profiles, dtype=float)
    if profiles.ndim != 2:
        raise ValueError("profiles must have shape (sample, pressure)")
    if not np.isfinite(profiles).all():
        raise ValueError("EOF input contains missing values")
    weights = (
        pressure_layer_weights(pressure_hpa)
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    weights = weights / np.sum(weights)
    anomaly = profiles - np.mean(profiles, axis=0, keepdims=True)
    _u, singular, vt = np.linalg.svd(
        anomaly * np.sqrt(weights)[None, :], full_matrices=False
    )
    eofs = vt[:2] / np.sqrt(weights)[None, :]
    for index in range(2):
        norm = np.sqrt(np.sum(eofs[index] ** 2 * weights))
        eofs[index] /= norm
    eofs, signs = orient_vertical_motion_eofs(eofs, pressure_hpa, weights)
    pcs = anomaly @ (eofs * weights[None, :]).T
    explained = singular**2 / np.sum(singular**2)
    if explained.size < 2 or explained[0] <= 0:
        raise ValueError("EOF1 does not represent a coherent vertical-motion mode")
    return {
        "eofs": eofs,
        "pcs": pcs,
        "weights": weights,
        "explained_variance": explained[:2],
        "orientation_signs": signs,
        "convention": (
            "input is upward-positive; EOF1 has positive pressure-weighted "
            "column mean; EOF2 is positive aloft and negative below"
        ),
    }


def project_onto_eofs(
    upward_motion_profiles: np.ndarray,
    eofs: np.ndarray,
    weights: Sequence[float],
) -> np.ndarray:
    profiles = np.asarray(upward_motion_profiles, dtype=float)
    anomaly = profiles - np.mean(profiles, axis=0, keepdims=True)
    return anomaly @ (
        np.asarray(eofs, dtype=float) * np.asarray(weights)[None, :]
    ).T


def spectral_coefficients_for_box(
    field: np.ndarray,
    *,
    interval_hours: float,
    k_bounds: Sequence[int],
    period_bounds_days: Sequence[float],
    taper_fraction: float = 0.1,
) -> tuple[np.ndarray, int]:
    """Return coefficients in an east/west k-frequency rectangle.

    The input dimensions are time and longitude. Positive configured
    wavenumber denotes eastward propagation, matching the repository filter
    convention. Linear temporal detrending and a Tukey taper match the
    spectrum-family preprocessing convention.
    """

    values = np.asarray(field, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("spectral input must be finite (time, longitude)")
    values = signal.detrend(values, axis=0, type="linear")
    values *= signal.windows.tukey(values.shape[0], alpha=taper_fraction)[:, None]
    coefficients = np.fft.fft2(values)
    frequency = np.fft.fftfreq(
        values.shape[0], d=float(interval_hours) / 24.0
    )
    zonal_wavenumber = np.fft.fftfreq(values.shape[1]) * values.shape[1]
    low_frequency = 1.0 / max(period_bounds_days)
    high_frequency = 1.0 / min(period_bounds_days)
    k0, k1 = sorted(k_bounds)
    if k0 >= 0:
        primary = (
            (zonal_wavenumber[None, :] >= k0)
            & (zonal_wavenumber[None, :] <= k1)
            & (frequency[:, None] <= -low_frequency)
            & (frequency[:, None] >= -high_frequency)
        )
    else:
        primary = (
            (zonal_wavenumber[None, :] >= k0)
            & (zonal_wavenumber[None, :] <= k1)
            & (frequency[:, None] >= low_frequency)
            & (frequency[:, None] <= high_frequency)
        )
    # Retain one independent propagation half-plane. Adding its conjugate
    # partner would cancel the quadrature spectrum and force tau_i toward zero.
    return coefficients[primary], int(np.count_nonzero(primary))


def complex_top_heaviness(
    pc1_coefficients: np.ndarray,
    pc2_coefficients: np.ndarray,
) -> dict[str, float]:
    """Inoue/Sakaeda PC2--PC1 cospectral and quadrature ratios."""

    o1, o2 = np.broadcast_arrays(
        np.asarray(pc1_coefficients, dtype=complex),
        np.asarray(pc2_coefficients, dtype=complex),
    )
    valid = (
        np.isfinite(o1.real)
        & np.isfinite(o1.imag)
        & np.isfinite(o2.real)
        & np.isfinite(o2.imag)
    )
    denominator = np.sum(o1[valid] * np.conj(o1[valid])).real
    if denominator <= PREDICTOR_VARIANCE_EPS:
        raise ValueError("PC1 spectral power is zero or negligible")
    ratio = np.sum(o2[valid] * np.conj(o1[valid])) / denominator
    return {
        "tau_r": float(np.real(ratio)),
        "tau_i": float(np.imag(ratio)),
        "pc1_power": float(denominator),
        "pc2_power": float(np.sum(o2[valid] * np.conj(o2[valid])).real),
        "spectral_coefficient_count": int(np.count_nonzero(valid)),
    }


def buoyancy_frequency_squared(
    theta: np.ndarray,
    height_m: Sequence[float],
    *,
    axis: int = -1,
    gravity_m_s2: float = 9.80665,
    mask_nonpositive: bool = True,
) -> np.ndarray:
    """Return N², with optional masking of statically unstable values."""

    theta = np.asarray(theta, dtype=float)
    height = np.asarray(height_m, dtype=float)
    axis = np.core.numeric.normalize_axis_index(axis, theta.ndim)
    if height.ndim == 1:
        if height.size != theta.shape[axis]:
            raise ValueError(
                "One-dimensional height must match the selected theta axis"
            )
        shape = [1] * theta.ndim
        shape[axis] = height.size
        height = height.reshape(shape)
    theta, height = np.broadcast_arrays(theta, height)
    theta = np.moveaxis(theta, axis, -1)
    height = np.moveaxis(height, axis, -1)
    if theta.shape[-1] < 2:
        raise ValueError("N² requires at least two vertical levels")

    gradient = np.empty_like(theta, dtype=float)
    lower_spacing = height[..., 1:-1] - height[..., :-2]
    upper_spacing = height[..., 2:] - height[..., 1:-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        lower_weight = -upper_spacing / (
            lower_spacing * (lower_spacing + upper_spacing)
        )
        centre_weight = (upper_spacing - lower_spacing) / (
            lower_spacing * upper_spacing
        )
        upper_weight = lower_spacing / (
            upper_spacing * (lower_spacing + upper_spacing)
        )
        gradient[..., 1:-1] = (
            lower_weight * theta[..., :-2]
            + centre_weight * theta[..., 1:-1]
            + upper_weight * theta[..., 2:]
        )
        gradient[..., 0] = (
            theta[..., 1] - theta[..., 0]
        ) / (height[..., 1] - height[..., 0])
        gradient[..., -1] = (
            theta[..., -1] - theta[..., -2]
        ) / (height[..., -1] - height[..., -2])
        n_squared = gravity_m_s2 * gradient / theta
    n_squared = np.moveaxis(n_squared, -1, axis)
    valid = np.isfinite(n_squared)
    if mask_nonpositive:
        valid &= n_squared > 0.0
    return np.where(valid, n_squared, np.nan)


def blockwise_buoyancy_frequency_squared(
    theta: xr.DataArray,
    height_m: xr.DataArray,
    vertical_dimension: str,
    *,
    block_size: int = 5,
    mask_nonpositive: bool = True,
) -> xr.DataArray:
    """Return exact N² using small vertical blocks with one-level halos.

    The halos preserve the centred finite differences at block boundaries.
    They also avoid rechunking every model column into one large Dask task.
    """

    theta, height_m = xr.align(theta, height_m, join="exact")
    if vertical_dimension not in theta.dims:
        raise ValueError(
            f"Missing vertical dimension {vertical_dimension!r} in theta"
        )
    if vertical_dimension not in height_m.dims:
        raise ValueError(
            f"Missing vertical dimension {vertical_dimension!r} in height"
        )
    if block_size < 1:
        raise ValueError("block_size must be positive")
    level_count = theta.sizes[vertical_dimension]
    if level_count < 2:
        raise ValueError("N² requires at least two vertical levels")

    blocks: list[xr.DataArray] = []
    for start in range(0, level_count, block_size):
        stop = min(start + block_size, level_count)
        halo_start = max(0, start - 1)
        halo_stop = min(level_count, stop + 1)
        selector = {vertical_dimension: slice(halo_start, halo_stop)}
        halo_n2 = xr.apply_ufunc(
            buoyancy_frequency_squared,
            theta.isel(selector),
            height_m.isel(selector),
            kwargs={"mask_nonpositive": mask_nonpositive},
            input_core_dims=[
                [vertical_dimension],
                [vertical_dimension],
            ],
            output_core_dims=[[vertical_dimension]],
            vectorize=False,
            dask="parallelized",
            output_dtypes=[float],
            dask_gufunc_kwargs={"allow_rechunk": True},
        )
        blocks.append(
            halo_n2.isel(
                {
                    vertical_dimension: slice(
                        start - halo_start,
                        stop - halo_start,
                    )
                }
            )
        )
    output = xr.concat(blocks, dim=vertical_dimension)
    output = output.assign_coords(
        {vertical_dimension: theta[vertical_dimension]}
    )
    output.name = "buoyancy_frequency_squared"
    output.attrs.update(units="s-2")
    return output


def kelvin_group_velocities(
    zonal_wind_m_s: np.ndarray,
    buoyancy_frequency_s1: np.ndarray,
    *,
    zonal_wavenumber_m1: float = 1.57e-7,
    vertical_wavenumber_m1: float = -6.28e-4,
    minimum_abs_cgx_m_s: float = 1.0e-3,
) -> dict[str, np.ndarray]:
    """Lim--Son reference Kelvin-wave group velocities and ray slope."""

    u, n = np.broadcast_arrays(
        np.asarray(zonal_wind_m_s, dtype=float),
        np.asarray(buoyancy_frequency_s1, dtype=float),
    )
    if vertical_wavenumber_m1 == 0:
        raise ValueError("vertical_wavenumber_m1 must be non-zero")
    cgx = u - n / vertical_wavenumber_m1
    cgz = n * zonal_wavenumber_m1 / vertical_wavenumber_m1**2
    slope = np.full(cgx.shape, np.nan, dtype=float)
    np.divide(
        cgz,
        cgx,
        out=slope,
        where=np.abs(cgx) > minimum_abs_cgx_m_s,
    )
    return {
        "cgx_m_s": cgx,
        "cgz_m_s": cgz,
        "ray_slope": slope,
        "near_zero_cgx": np.abs(cgx) <= minimum_abs_cgx_m_s,
    }


def mrg_group_velocities(
    zonal_wind_m_s: np.ndarray,
    buoyancy_frequency_s1: np.ndarray,
    *,
    zonal_wavenumber_m1: float = -2.0 / 6.371e6,
    vertical_wavenumber_m1: float = -2.0 * np.pi / 10_000.0,
    beta_m1_s1: float = 2.28e-11,
    minimum_abs_cgx_m_s: float = 1.0e-3,
) -> dict[str, np.ndarray]:
    """Return standard mixed Rossby-gravity group velocities.

    The positive-frequency MRG branch uses
    ``omega = k*c/2 + sqrt((k*c/2)^2 + beta*c)``.
    Hydrostatic phase speed is ``c = -N/m`` for the configured negative
    vertical wavenumber. This sign gives upward group propagation.
    """

    u, n = np.broadcast_arrays(
        np.asarray(zonal_wind_m_s, dtype=float),
        np.asarray(buoyancy_frequency_s1, dtype=float),
    )
    if vertical_wavenumber_m1 >= 0.0:
        raise ValueError(
            "vertical_wavenumber_m1 must be negative for upward MRG energy"
        )
    if beta_m1_s1 <= 0.0:
        raise ValueError("beta_m1_s1 must be positive")
    phase_speed = -n / vertical_wavenumber_m1
    discriminant = np.sqrt(
        (0.5 * zonal_wavenumber_m1 * phase_speed) ** 2
        + beta_m1_s1 * phase_speed
    )
    intrinsic_cgx = (
        zonal_wavenumber_m1 * phase_speed**2 / (4.0 * discriminant)
        + 0.5 * phase_speed
    )
    domega_dc = (
        0.5 * zonal_wavenumber_m1
        + (
            zonal_wavenumber_m1**2 * phase_speed
            + 2.0 * beta_m1_s1
        )
        / (4.0 * discriminant)
    )
    dc_dm = n / vertical_wavenumber_m1**2
    cgx = u + intrinsic_cgx
    cgz = domega_dc * dc_dm
    slope = np.full(cgx.shape, np.nan, dtype=float)
    np.divide(
        cgz,
        cgx,
        out=slope,
        where=np.abs(cgx) > minimum_abs_cgx_m_s,
    )
    return {
        "cgx_m_s": cgx,
        "cgz_m_s": cgz,
        "ray_slope": slope,
        "near_zero_cgx": np.abs(cgx) <= minimum_abs_cgx_m_s,
    }


def sample_masked_field(
    field: xr.DataArray,
    mask: xr.DataArray,
    area_km2: xr.DataArray,
    *,
    quantiles: Sequence[float] = (),
) -> Mapping[str, float]:
    """Area-weighted event-mask statistics for a two-dimensional field."""

    field, mask, area_km2 = xr.align(
        field, mask.astype(bool), area_km2, join="exact"
    )
    selected = field.where(mask)
    weights = area_km2.where(mask & np.isfinite(field))
    total_weight = weights.sum(skipna=True)
    if not np.isfinite(float(total_weight)) or float(total_weight) <= 0:
        return {
            "mean": np.nan,
            "area_integral": np.nan,
            **{
                f"q{int(round(100 * quantile)):02d}": np.nan
                for quantile in quantiles
            },
        }
    result = {
        "mean": float((selected * weights).sum(skipna=True) / total_weight),
        "area_integral": float((selected * weights).sum(skipna=True)),
    }
    selected_values = np.asarray(selected.values, dtype=float).ravel()
    selected_weights = np.asarray(weights.values, dtype=float).ravel()
    finite = (
        np.isfinite(selected_values)
        & np.isfinite(selected_weights)
        & (selected_weights > 0)
    )
    order = np.argsort(selected_values[finite])
    ordered_values = selected_values[finite][order]
    ordered_weights = selected_weights[finite][order]
    cumulative = (
        np.cumsum(ordered_weights) - 0.5 * ordered_weights
    ) / np.sum(ordered_weights)
    for quantile in quantiles:
        result[f"q{int(round(100 * quantile)):02d}"] = float(
            np.interp(
                float(quantile),
                cumulative,
                ordered_values,
                left=ordered_values[0],
                right=ordered_values[-1],
            )
        )
    return result
