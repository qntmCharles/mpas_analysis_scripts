"""Daily production-period circulation/moisture timing checks.

Form lagged pairs before resampling dates. This avoids creating artificial
time neighbours where resampled blocks join.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from diagnostics.mean_state import circulation_mechanism as mechanism
from diagnostics.waves.wave_metrics import circular_time_block_indices

MAX_LAG_DAYS = 5
MOISTURE_PRESSURE_HPA = (850.0, 950.0)
TROPICAL_LATITUDE = (-10.0, 10.0)
HC_LATITUDE = (5.0, 15.0)
HC_PRESSURE_HPA = (150.0, 850.0)
HC_LATITUDES = np.linspace(*HC_LATITUDE, 61)
HC_PRESSURES = np.linspace(*HC_PRESSURE_HPA, 81)
TEM_PRIMITIVES = ("a_ep", "vres", "eta", "mass_weight")
INDICES = ("v_ep", "hc_streamfunction", "moisture_convergence", "pw", "precipitation")
PAIRS = (
    ("v_ep", "hc_streamfunction"),
    ("hc_streamfunction", "moisture_convergence"),
    ("hc_streamfunction", "pw"),
    ("hc_streamfunction", "precipitation"),
    ("moisture_convergence", "precipitation"),
    ("pw", "precipitation"),
)
PREDICTIVE_MODELS = (
    ("precipitation_to_hc", "hc_streamfunction", "precipitation"),
    ("hc_to_precipitation", "precipitation", "hc_streamfunction"),
    ("ep_to_hc", "hc_streamfunction", "v_ep"),
)
JOINT_PREDICTIVE_MODEL = ("ep_and_precipitation_to_hc", "hc_streamfunction", ("v_ep", "precipitation"))
UNITS = dict(v_ep="m s-1", hc_streamfunction="kg s-1", moisture_convergence="mm day-1",
             pw="kg m-2", precipitation="mm day-1")


def daily_slices(times):
    """Retain four right-ended six-hourly samples for each exact 24-hour day."""
    times = np.asarray(times, dtype="datetime64[ns]")
    if len(times) < 4 or len(times) % 4 or not np.all(np.diff(times) == np.timedelta64(6, "h")):
        raise ValueError("Daily timing diagnostics require complete, contiguous four-sample days")
    return [slice(first, first + 4) for first in range(0, len(times), 4)]


def daily_tem_state(regional, diagnostics, archive, *, average_time=True):
    """Reduce fixed spatial indices; optionally retain native time samples."""
    mean = regional.mean("Time", skipna=False, keep_attrs=True) if average_time else regional
    result = mean[list(TEM_PRIMITIVES)].copy()
    psi = mechanism.hemisphere_samples(diagnostics.residual_streamfunction, "odd")
    pressure = mechanism.hemisphere_samples(archive.pressure, "even")
    latitudes = np.asarray(psi.absolute_latitude)
    window = slice(max(0, np.searchsorted(latitudes, HC_LATITUDE[0]) - 1),
                   min(len(latitudes), np.searchsorted(latitudes, HC_LATITUDE[1], side="right") + 1))
    psi = mechanism._on_pressure(psi.isel(absolute_latitude=window),
                                 pressure.isel(absolute_latitude=window), HC_PRESSURES, "hc_pres").interp(
                                     absolute_latitude=HC_LATITUDES)
    if average_time:
        psi = psi.mean("Time", skipna=False)
    weights = (mechanism._quadrature(HC_PRESSURES * 100)[:, None]
               * mechanism._quadrature(np.deg2rad(HC_LATITUDES))[None, :]
               * np.cos(np.deg2rad(HC_LATITUDES))[None, :])
    weights = xr.DataArray(weights, dims=("hc_pres", "absolute_latitude"),
                           coords=dict(hc_pres=HC_PRESSURES, absolute_latitude=HC_LATITUDES))
    result["hc_streamfunction"] = (psi * weights).sum(("hc_pres", "absolute_latitude"), skipna=False) / weights.sum()
    result.hc_streamfunction.attrs.update(units="kg s-1", definition="fixed signed cos(latitude)*dp mean of residual streamfunction; 5-15 degrees, 150-850 hPa; SH sign reversed")
    return result


def low_level_convergence(mean_v, mean_q, *, radius_m, gravity_m_s2):
    """Use tropical boundary inflow to area-average mean-circulation convergence.

F = integral_850^950 mean(v)*mean(q) dp/g. Retain cross-equatorial
transport in each hemisphere before averaging their equal-area means.
"""
    if not np.allclose(mean_v.pres.values[[0, -1]], MOISTURE_PRESSURE_HPA):
        raise ValueError("Low-level flux must cover exactly 850-950 hPa")
    flux = (mean_v * mean_q).assign_coords(pres=mean_v.pres * 100).integrate("pres") / gravity_m_s2
    boundary = flux.interp(latitude=[TROPICAL_LATITUDE[0], 0.0, TROPICAL_LATITUDE[1]])
    angle = np.deg2rad(TROPICAL_LATITUDE[1])
    south, equator, north = (boundary.sel(latitude=value, drop=True) for value in (-10.0, 0.0, 10.0))
    result = xr.concat([(np.cos(angle) * south - equator) / (radius_m * np.sin(angle)),
                        (equator - np.cos(angle) * north) / (radius_m * np.sin(angle))],
                       dim=xr.IndexVariable("hemisphere", ["SH", "NH"])) * 86400.0
    result.attrs.update(units="mm day-1", sign_convention="positive convergence",
                        definition="850-950 hPa mean-circulation boundary inflow / tropical hemisphere area")
    return result


def daily_anomalies(daily):
    """Centre each simulation on its own production climatology, including EP prediction."""
    times = np.asarray(daily.Time, dtype="datetime64[ns]")
    if len(times) < 2 or not np.all(np.diff(times) == np.timedelta64(1, "D")):
        raise ValueError("Daily indices require identical contiguous dates across simulations")
    starts = daily.day_start.transpose("simulation", "Time").values
    ends = daily.day_end.transpose("simulation", "Time").values
    if (not np.all(starts == starts[0]) or not np.all(ends == ends[0])
            or not np.all(ends - starts == np.timedelta64(1, "D"))
            or not np.all(starts + np.timedelta64(12, "h") == times)):
        raise ValueError("Daily indices require matched exact 24-hour intervals and midpoint dates")
    for name in (*TEM_PRIMITIVES, "hc_streamfunction", "moisture_convergence", "pw", "precipitation"):
        if not np.isfinite(daily[name].values).all():
            raise ValueError(f"Nonfinite daily {name}; check archive coverage before interpreting lead-lag relationships")
    q = xr.DataArray(
        mechanism._quadrature(daily.region_pres * 100)[:, None]
        * mechanism._quadrature(np.deg2rad(daily.region_latitude))[None, :],
        dims=("region_pres", "region_latitude"),
        coords=dict(region_pres=daily.region_pres, region_latitude=daily.region_latitude),
    )
    anomalies = []
    for run_index, label in enumerate(daily.simulation.values):
        run = daily.isel(simulation=run_index, drop=True)
        climatology = run.mean("Time", skipna=False, keep_attrs=True)
        fields = {name: (run[name] - climatology[name]).mean("hemisphere", skipna=False)
                  for name in ("hc_streamfunction", "moisture_convergence", "pw", "precipitation")}
        numerator = -(run.a_ep - climatology.a_ep) - climatology.vres * (run.eta - climatology.eta)
        valid = (abs(run.eta) > 1e-8) & (abs(climatology.eta) > 1e-8)
        weights = (climatology.mass_weight * q).where(valid & (climatology.mass_weight > 0), 0.0)
        denominator = weights.sum(("region_pres", "region_latitude"), skipna=False)
        if not np.all(denominator.values > 0):
            raise ValueError("Daily EP prediction has no nonsingular vorticity samples")
        predicted = numerator / run.eta.where(valid)
        fields["v_ep"] = ((predicted.where(weights > 0, 0.0) * weights).sum(
            ("region_pres", "region_latitude"), skipna=False) / denominator).mean("hemisphere", skipna=False)
        # A time-varying denominator need not preserve a zero temporal mean.
        fields["v_ep"] = fields["v_ep"] - fields["v_ep"].mean("Time", skipna=False)
        possible = (climatology.mass_weight * q).sum(("region_pres", "region_latitude"), skipna=False)
        fields["v_ep_valid_mass_fraction"] = (denominator / possible).mean("hemisphere", skipna=False)
        result = xr.Dataset(fields)
        for name, units in UNITS.items():
            result[name].attrs.update(units=units, anomaly_reference="own full-record simulation time mean")
        result.v_ep.attrs["definition"] = "centred regional mean (-EPFd_prime-vres_climatology*eta_prime)/eta_daily; own climatological mass weights"
        result.v_ep_valid_mass_fraction.attrs.update(units="1")
        anomalies.append(result.expand_dims(simulation=[label]))
    result = xr.concat(anomalies, dim="simulation", join="exact")
    result.attrs["anomaly_reference"] = "own full-record simulation time mean"
    return result


def _correlation(x, y):
    """Pearson correlation along the last axis, retaining undefined constants."""
    valid = np.isfinite(x) & np.isfinite(y)
    count = valid.sum(axis=-1)
    x = np.where(valid, x, 0.0)
    y = np.where(valid, y, 0.0)
    x = np.where(valid, x - x.sum(axis=-1, keepdims=True) / np.maximum(count[..., None], 1), 0.0)
    y = np.where(valid, y - y.sum(axis=-1, keepdims=True) / np.maximum(count[..., None], 1), 0.0)
    denominator = np.sqrt((x*x).sum(axis=-1) * (y*y).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((count >= 3) & (denominator > 0), (x*y).sum(axis=-1) / denominator, np.nan)


def _shift_for_lag(y, lag):
    """Align y(t+lag) with x(t), with no wrapping at production boundaries."""
    shifted = np.full_like(y, np.nan)
    if lag > 0:
        shifted[:-lag] = y[lag:]
    elif lag < 0:
        shifted[-lag:] = y[:lag]
    else:
        shifted[:] = y
    return shifted


def lag_analysis(responses, *, block_days, resamples, minimum_blocks, seed, percentiles, bootstrap=True, pairs=PAIRS):
    """Jointly resample aligned lag pairs with the codebase's circular blocks.

Retain all available production pairs at each lag. Use identical sampled dates
for every variable, comparison, lag, and trend sensitivity. No lag is computed
across a join between resampled blocks. Intervals are pointwise, not a
multiple-lag correction or causal estimate.
"""
    count = responses.sizes["Time"]
    if count <= 2 * MAX_LAG_DAYS + 2:
        raise ValueError("Lead-lag curves require at least 13 complete daily samples")
    if bootstrap and count < block_days * minimum_blocks:
        raise ValueError("Lead-lag uncertainty requires at least six complete ten-day production blocks")
    indices = (circular_time_block_indices(count, int(block_days), resamples, np.random.default_rng(seed))
               if bootstrap else None)
    lags = np.arange(-MAX_LAG_DAYS, MAX_LAG_DAYS + 1)
    time = np.arange(count, dtype=float)
    rows, summary = [], []
    for comparison in responses.comparison.values:
        source = responses.sel(comparison=comparison)
        for first, second in pairs:
            for treatment in ("daily_anomaly", "linear_detrend_sensitivity"):
                x, y = (np.asarray(source[name], dtype=float) for name in (first, second))
                if not np.isfinite(x).all() or not np.isfinite(y).all():
                    raise ValueError("Lead-lag indices contain missing daily samples")
                if treatment == "linear_detrend_sensitivity":
                    x = x - np.polyval(np.polyfit(time, x, 1), time)
                    y = y - np.polyval(np.polyfit(time, y, 1), time)
                correlations, draws = [], []
                for lag in lags:
                    shifted = _shift_for_lag(y, int(lag))
                    value = float(_correlation(x, shifted))
                    correlations.append(value)
                    ci, finite_draws = [np.nan, np.nan], 0
                    if bootstrap:
                        sample = _correlation(x[indices], shifted[indices])
                        draws.append(sample)
                        finite = np.isfinite(sample)
                        finite_draws = int(finite.sum())
                        if finite.all():
                            ci = np.percentile(sample, percentiles)
                    rows.append(dict(comparison=str(comparison), first=first, second=second,
                                     treatment=treatment, lag_days=int(lag), n_pairs=count-abs(int(lag)),
                                     correlation=value, ci_low=ci[0], ci_high=ci[1], finite_bootstrap_draws=finite_draws))
                correlations = np.asarray(correlations)
                valid = np.isfinite(correlations).all()
                peak = lags[np.argmax(correlations)] if valid else np.nan
                contrast = (correlations[lags > 0].max() - correlations[lags < 0].max()) if valid else np.nan
                contrast_ci = [np.nan, np.nan]
                if bootstrap:
                    curves = np.asarray(draws)
                    contrast_draws = curves[lags > 0].max(axis=0) - curves[lags < 0].max(axis=0)
                    if np.isfinite(contrast_draws).all():
                        contrast_ci = np.percentile(contrast_draws, percentiles)
                summary.append(dict(comparison=str(comparison), first=first, second=second, treatment=treatment,
                                    peak_positive_correlation_lag_days=peak, zero_lag_correlation=correlations[MAX_LAG_DAYS],
                                    leading_minus_following_peak_correlation=contrast,
                                    contrast_ci_low=contrast_ci[0], contrast_ci_high=contrast_ci[1],
                                    interpretation="positive lag: first index leads; association, not causality"))
    return pd.DataFrame(rows), pd.DataFrame(summary)


def _lagged_fit(persistence, predictor, future):
    """Fit intercept plus two predictors along the last axis, including bootstrap batches."""
    valid = np.isfinite(persistence) & np.isfinite(predictor) & np.isfinite(future)
    count = valid.sum(axis=-1)

    def centre(values):
        mean = np.where(valid, values, 0.0).sum(axis=-1) / np.maximum(count, 1)
        return np.where(valid, values - mean[..., None], 0.0), mean

    z, zm = centre(persistence)
    x, xm = centre(predictor)
    y, ym = centre(future)
    zz, xx, yy = ((value * value).sum(axis=-1) for value in (z, x, y))
    zx, zy, xy = ((first * second).sum(axis=-1) for first, second in ((z, x), (z, y), (x, y)))
    determinant = zz * xx - zx * zx
    # Reject absent variance and numerically collinear predictors explicitly.
    defined = (count >= 4) & (zz > 0) & (xx > 0) & (yy > 0) & (determinant > 1e-12 * zz * xx)
    with np.errstate(invalid="ignore", divide="ignore"):
        a = np.where(defined, (zy * xx - xy * zx) / determinant, np.nan)
        b = np.where(defined, (xy * zz - zy * zx) / determinant, np.nan)
        intercept = ym - a * zm - b * xm
        r2 = (a * zy + b * xy) / yy
        delta_r2 = r2 - zy * zy / (zz * yy)
    return np.stack((a, b, intercept, r2, delta_r2), axis=-1)


def _joint_lagged_fit(persistence, predictors, future):
    """Fit persistence plus EP and precipitation, rejecting singular designs."""
    design = np.stack((persistence, *predictors), axis=-1)
    valid = np.isfinite(design).all(axis=-1) & np.isfinite(future)
    count = valid.sum(axis=-1)
    x_mean = np.where(valid[..., None], design, 0.0).sum(axis=-2) / np.maximum(count[..., None], 1)
    y_mean = np.where(valid, future, 0.0).sum(axis=-1) / np.maximum(count, 1)
    x = np.where(valid[..., None], design - x_mean[..., None, :], 0.0)
    y = np.where(valid, future - y_mean[..., None], 0.0)
    gram = np.einsum("...ti,...tj->...ij", x, x)
    cross = np.einsum("...ti,...t->...i", x, y)
    yy = (y*y).sum(axis=-1)
    eigenvalues = np.linalg.eigvalsh(gram)
    defined = (count >= 5) & (yy > 0) & (eigenvalues[..., 0] > 1e-12 * eigenvalues[..., -1])
    coefficients = np.full(cross.shape, np.nan)
    coefficients[defined] = np.linalg.solve(gram[defined], cross[defined][..., None])[..., 0]
    intercept = y_mean - (coefficients * x_mean).sum(axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = (coefficients * cross).sum(axis=-1) / yy
        delta_r2 = r2 - cross[..., 0]**2 / (gram[..., 0, 0] * yy)
    return np.concatenate([coefficients, np.stack((intercept, r2, delta_r2), axis=-1)], axis=-1)


def predictive_analysis(responses, *, block_days, resamples, minimum_blocks, seed, percentiles,
                        bootstrap=True, models=(*PREDICTIVE_MODELS, JOINT_PREDICTIVE_MODEL), pairs=PAIRS):
    """Test separate/joint predictors beyond persistence and integrated lag asymmetry.

Form (target[t+1], target[t], predictor[t]) before resampling. Keep the final
unavailable future value missing. Sample dates jointly across models and lag
directions. Fixed full-record scales make standardized slopes comparable to
their bootstrap draws; retain physical-unit coefficients as well.
"""
    count = responses.sizes["Time"]
    times = np.asarray(responses.Time, dtype="datetime64[ns]")
    if count < 13 or not np.all(np.diff(times) == np.timedelta64(1, "D")):
        raise ValueError("Predictive timing checks require at least 13 contiguous complete daily samples")
    if bootstrap and count < block_days * minimum_blocks:
        raise ValueError("Predictive uncertainty requires at least six complete ten-day production blocks")
    indices = (circular_time_block_indices(count, int(block_days), resamples, np.random.default_rng(seed))
               if bootstrap else None)
    regressions, asymmetries = [], []
    time = np.arange(count, dtype=float)

    def interval(draws):
        finite = np.isfinite(draws)
        ci = np.percentile(draws, percentiles) if bootstrap and finite.all() else [np.nan, np.nan]
        return ci, int(finite.sum())

    for comparison in responses.comparison.values:
        source = responses.sel(comparison=comparison)
        for treatment in ("daily_anomaly", "linear_detrend_sensitivity"):
            names = {target for _, target, _ in models}
            names.update(name for _, _, predictors in models
                         for name in ((predictors,) if isinstance(predictors, str) else predictors))
            names.update(name for pair in pairs for name in pair)
            values = {name: np.asarray(source[name].transpose("Time"), dtype=float) for name in names}
            if not all(np.isfinite(value).all() for value in values.values()):
                raise ValueError("Predictive indices contain missing daily samples")
            if treatment == "linear_detrend_sensitivity":
                values = {name: (value - np.polyval(np.polyfit(time, value, 1), time)
                                 if np.ptp(value) > 0 else np.zeros_like(value))
                          for name, value in values.items()}
            for model, target, predictors in models:
                predictors = (predictors,) if isinstance(predictors, str) else predictors
                y = values[target]
                y_mean, y_scale = y.mean(), y.std()
                x_means = np.array([values[name].mean() for name in predictors])
                x_scales = np.array([values[name].std() for name in predictors])
                n_predictors = len(predictors)
                estimates = np.full(n_predictors + 4, np.nan)
                samples = np.full((resamples if bootstrap else 0, n_predictors + 4), np.nan)
                if y_scale > 0 and (x_scales > 0).all():
                    z = (y - y_mean) / y_scale
                    x_standard = [(values[name] - mean) / scale
                                  for name, mean, scale in zip(predictors, x_means, x_scales)]
                    future = _shift_for_lag(z, 1)
                    estimates = (_lagged_fit(z, x_standard[0], future) if n_predictors == 1
                                 else _joint_lagged_fit(z, x_standard, future))
                    if bootstrap:
                        samples = (_lagged_fit(z[indices], x_standard[0][indices], future[indices]) if n_predictors == 1
                                   else _joint_lagged_fit(z[indices], [x[indices] for x in x_standard], future[indices]))
                persistence_ci, _ = interval(samples[:, 0])
                ratios = np.divide(y_scale, x_scales, out=np.full(n_predictors, np.nan), where=x_scales > 0)
                intercept = (estimates[n_predictors+1] * y_scale + y_mean * (1-estimates[0])
                             - (estimates[1:n_predictors+1] * ratios * x_means).sum())
                for index, predictor in enumerate(predictors, start=1):
                    cross_ci, finite_draws = interval(samples[:, index])
                    ratio, x_scale = ratios[index-1], x_scales[index-1]
                    regressions.append(dict(
                        comparison=str(comparison), treatment=treatment, model=model, target=target, predictor=predictor,
                        conditioning_predictors=",".join(name for name in predictors if name != predictor),
                        lag_days=1, n_pairs=count-1, persistence_coefficient=estimates[0],
                        persistence_ci_low=persistence_ci[0], persistence_ci_high=persistence_ci[1],
                        standardized_cross_coefficient=estimates[index], standardized_ci_low=cross_ci[0], standardized_ci_high=cross_ci[1],
                        cross_coefficient=estimates[index] * ratio, ci_low=cross_ci[0] * ratio, ci_high=cross_ci[1] * ratio,
                        cross_coefficient_units=f"({UNITS[target]}) / ({UNITS[predictor]})",
                        intercept=intercept,
                        intercept_units=UNITS[target], target_scale=y_scale, predictor_scale=x_scale,
                        r_squared=estimates[-2], added_in_sample_r_squared=estimates[-1],
                        finite_bootstrap_draws=finite_draws,
                        significantly_positive=bool(np.isfinite(cross_ci[0]) and cross_ci[0] > 0),
                        status="defined" if np.isfinite(estimates).all() else "undefined: constant series or collinear predictors",
                    ))
            for first, second in pairs:
                x, y = values[first], values[second]
                differences, joint_draws, positives, negatives = [], [], [], []
                for tau in range(1, MAX_LAG_DAYS + 1):
                    plus, minus = _shift_for_lag(y, tau), _shift_for_lag(y, -tau)
                    positive, negative = float(_correlation(x, plus)), float(_correlation(x, minus))
                    draws = (_correlation(x[indices], plus[indices]) - _correlation(x[indices], minus[indices])
                             if bootstrap else np.empty(0))
                    positives.append(positive)
                    negatives.append(negative)
                    differences.append(positive-negative)
                    joint_draws.append(draws)
                    integrated = float(np.mean(differences))
                    integrated_draws = np.mean(joint_draws, axis=0)
                    ci, finite_draws = interval(integrated_draws)
                    asymmetries.append(dict(
                        comparison=str(comparison), treatment=treatment, first=first, second=second,
                        L_days=tau, minimum_n_pairs_each_direction=count-tau,
                        mean_correlation_positive=float(np.mean(positives)), mean_correlation_negative=float(np.mean(negatives)),
                        integrated_asymmetry=integrated,
                        ci_low=ci[0], ci_high=ci[1], finite_bootstrap_draws=finite_draws,
                        significantly_positive=bool(np.isfinite(ci[0]) and ci[0] > 0),
                        interpretation="A(L)=mean_tau=1..L[C(+tau)-C(-tau)]; positive: first index leads second",
                    ))
    return pd.DataFrame(regressions), pd.DataFrame(asymmetries)
