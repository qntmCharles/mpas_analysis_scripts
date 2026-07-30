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


METRICS_VERSION = "wave-metrics-v4-native-omega"
PREDICTOR_VARIANCE_EPS = 1.0e-12


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
) -> np.ndarray:
    """N² = g/theta dtheta/dz, with invalid/non-positive values masked."""

    theta = np.asarray(theta, dtype=float)
    height = np.asarray(height_m, dtype=float)
    gradient = np.gradient(theta, height, axis=axis)
    n_squared = gravity_m_s2 * gradient / theta
    return np.where(np.isfinite(n_squared) & (n_squared > 0), n_squared, np.nan)


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
