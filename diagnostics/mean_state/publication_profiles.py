"""Shared publication rendering for mean-state Figure 2."""

from __future__ import annotations

import numpy as np
import xarray as xr
from matplotlib import pyplot as plt, ticker
from matplotlib.lines import Line2D

from diagnostics.analysis_config import (
    ALL_SIMULATION_PUBLICATION_LABELS, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
    CONFIDENCE_PERCENTILES, simulation_colour,
)
from diagnostics.mean_state.convection_summary_cache import (
    ENVIRONMENT_FIELDS, prepare_product, product_paths,
)
from diagnostics.convection.lpo_environment_profiles import (
    prepare_profiles as prepare_lpo_profiles, profile_paths as lpo_profile_paths,
    prepare_ascent_profiles as prepare_lpo_ascent_profiles,
)
from diagnostics.ttl import TTL_BASE_PRESSURE_HPA, TTL_TOP_PRESSURE_HPA
from diagnostics.uncertainty import independent_bootstrap_field_difference


ENVIRONMENT_SPECS = (
    ("temperature", "Temperature", "K", 1.0),
    ("static_stability", "Static stability", r"$10^{-4}$ s$^{-2}$", 1.0e4),
    ("zonal_wind", "Zonal wind", r"m s$^{-1}$", 1.0),
)
ASCENT_SPEC = ("vertical_velocity", "Ascent", r"mm s$^{-1}$", 1000.0)
PROFILE_SPECS = (*ENVIRONMENT_SPECS, ASCENT_SPEC)
PROFILE_CAPTION = """Absolute deep-tropical vertical profiles.
Solid profiles cover 10 degrees S–10 degrees N and 5 May–21 November 2000 without event conditioning.
The four panels show temperature, static stability, zonal wind, and ascent for control and both nudged simulations.
Temperature uses native t; zonal wind uses native uReconstructZonal.
Solid curves use cosine-weighted latitude means on their native model levels.
N2 uses the established mean-state definition, g/theta_bar times d(theta_bar)/dz on native height levels.
Each simulation uses its own time-mean, cosine-weighted native pressure on the logarithmic display axis.
All panels cover 850–50 hPa without interpolation onto isobaric surfaces.
Gray shading marks the 150–70 hPa TTL reference.
Envelopes show marginal 95% mean intervals from complete 20-day blocks resampled within each simulation.
Temperature, stability, and wind share bootstrap draws across native levels within each climate.
Ascent resamples complete 20-day native profiles after latitude averaging.
Pressure coordinates use the full-record means while envelopes show field uncertainty at each native level.
Ascent uses native vertical velocity, w, with the existing adjacent-interface averaging onto native pressure midpoints.
Signed absolute velocity is displayed in mm/s; positive values indicate upward motion.
These intervals do not provide field-wide significance tests.
Realised thermal perturbations differ in magnitude between experiments.
Their response difference does not exactly separate thermal and mechanical mechanisms.
"""
LPO_PROFILE_CAPTION = PROFILE_CAPTION.replace(
    "Absolute deep-tropical vertical profiles.", "Absolute deep-tropical and LPO-conditioned vertical profiles."
).replace("850–50 hPa", "400–50 hPa") + """
Subdued dashed lines show absolute LPO-conditioned profiles near the precipitation peak.
These use five six-hour samples at lags -12, -6, 0, +6, and +12 hours, then equal complete-event weights.
Only primary tropical events with every near-peak sample inside the analysis record contribute.
Sampling uses saved LPO-envelope columns with concurrent unsmoothed precipitation above 1 mm/day.
The existing lifecycle geometry retains +/-10 degrees around centroid latitude and +/-5 degrees in longitude.
Cosine-weighted raining-column means precede time and event means.
Conditioned temperature uses native theta and pressure; conditioned N2 uses instantaneous native-height theta gradients.
Conditioned wind uses native uReconstructZonal with the same saved rain mask.
Conditioned ascent uses exactly the environment event pool, rain mask, and centroid geometry.
Dashed curves use their own event-mean native pressure without isobaric interpolation.
Lighter envelopes show marginal 95% independent-event intervals around the conditioned means.
Conditioned intervals share whole-event draws across native levels; their draws also preserve cross-field covariance in panels (a–c).
"""
PROFILE_VARIANTS = (
    ("", False, 850.0, PROFILE_CAPTION),
    ("_with_lpos", True, 400.0, LPO_PROFILE_CAPTION),
)


def field_response(simulation_mean, control_mean, simulation_blocks, control_blocks, seed):
    """Contrast climates while resampling complete spatial fields within each climate."""
    simulation_mean, control_mean = xr.align(simulation_mean, control_mean, join="exact")
    simulation_blocks, control_blocks = xr.align(
        simulation_blocks, control_blocks, join="exact", exclude={"block"}
    )
    trailing = list(simulation_mean.dims)
    _, low, high = independent_bootstrap_field_difference(
        simulation_blocks.transpose("block", *trailing).values,
        control_blocks.transpose("block", *trailing).values,
        rng=np.random.default_rng(seed), n_resamples=BOOTSTRAP_REPLICATES,
    )
    return xr.Dataset({
        "response": simulation_mean - control_mean,
        "ci_low": xr.DataArray(low, dims=trailing, coords=simulation_mean.coords),
        "ci_high": xr.DataArray(high, dims=trailing, coords=simulation_mean.coords),
    })


def absolute_profile_intervals(climate, *, seed, n_resamples=BOOTSTRAP_REPLICATES, fields=ENVIRONMENT_FIELDS):
    """Resample whole native profiles jointly within one simulation climate."""
    blocks = xr.concat(
        [climate[f"{name}_blocks"] for name in fields],
        dim=xr.IndexVariable("field", list(fields)),
    ).transpose("block", "field", "z").values
    if blocks.shape[0] < 2:
        raise ValueError("Profile uncertainty requires at least two complete blocks")
    indices = np.random.default_rng(seed).integers(0, blocks.shape[0], (n_resamples, blocks.shape[0]))
    replicates = np.nanmean(blocks[indices], axis=1)
    low, high = np.nanpercentile(replicates, CONFIDENCE_PERCENTILES, axis=0)
    return {
        name: xr.Dataset({"mean": climate[name], "ci_low": ("z", low[index]), "ci_high": ("z", high[index])})
        for index, name in enumerate(fields)
    }


def prepare_profile_inputs(simulations, *, figures_only=False, rebuild=False, time_count=None, include_lpo=True):
    extra = {} if time_count is None else {"time_count": time_count}
    absolute = {sim: prepare_product(sim, "environment", figures_only=figures_only, rebuild=rebuild, **extra)
                for sim in simulations}
    intervals = {sim: absolute_profile_intervals(absolute[sim], seed=BOOTSTRAP_SEED + 20 + 10 * index)
                 for index, sim in enumerate(simulations)}
    sources = [path for sim in simulations for path in product_paths(sim, "environment", **extra)]
    products = {"environment_absolute": absolute, "environment": intervals}
    ascent = {sim: prepare_product(sim, "ascent", figures_only=figures_only, rebuild=rebuild, **extra)
              for sim in simulations}
    products["ascent_absolute"] = ascent
    products["ascent_intervals"] = {
        sim: absolute_profile_intervals(climate, seed=BOOTSTRAP_SEED + 61 + 10 * index,
                                       fields=("vertical_velocity",))["vertical_velocity"]
        for index, (sim, climate) in enumerate(ascent.items())
    }
    sources.extend(path for sim in simulations for path in product_paths(sim, "ascent", **extra))
    if include_lpo and time_count is None:
        products["environment_lpo"] = {
            sim: prepare_lpo_profiles(sim, figures_only=figures_only, rebuild=rebuild) for sim in simulations
        }
        sources.extend(path for sim in simulations for path in lpo_profile_paths(sim))
        products["ascent_lpo_absolute"] = {
            sim: prepare_lpo_ascent_profiles(sim, figures_only=figures_only, rebuild=rebuild) for sim in simulations
        }
        sources.extend(path for sim in simulations for path in lpo_profile_paths(sim, "ascent"))
    return products, sources


def pressure_axis(axis, *, ylabel=False, zero=False, lower=850.0):
    axis.axhspan(TTL_TOP_PRESSURE_HPA, TTL_BASE_PRESSURE_HPA,
                color="0.8", alpha=0.25, zorder=0)
    ticks = [value for value in [850, 700, 500, 400, 300, 200, 150, 100, 70, 50] if value <= lower]
    axis.set_yscale("log")
    axis.set_ylim(lower, 50.0)
    axis.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    axis.yaxis.set_major_formatter(ticker.FixedFormatter([str(v) for v in ticks]))
    axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    axis.grid(alpha=0.17)
    if ylabel:
        axis.set_ylabel("Pressure (hPa)")
    if zero:
        axis.axvline(0, color="0.55", lw=0.7, zorder=1)


def environment_profile_figure(products, *, include_lpo=False, lower=850.0):
    """Show native climate means, uncertainty envelopes, and subdued LPO profiles."""
    absolute = products["environment_absolute"]
    simulations = tuple(absolute)
    conditioned = products.get("environment_lpo", {}) if include_lpo else {}
    figure, axes = plt.subplots(1, 4, figsize=(8.0, 3.8 if conditioned else 3.5),
                                sharey=True, layout="constrained")
    for column, (name, title, units, scale) in enumerate(PROFILE_SPECS):
        axis = axes[column]
        is_ascent = name == "vertical_velocity"
        climates = products["ascent_absolute"] if is_ascent else absolute
        local_profiles = (products.get("ascent_lpo_absolute", {}) if include_lpo else {}) if is_ascent else conditioned
        for sim in simulations:
            pressure = climates[sim].pressure_hpa
            visible = (pressure >= 50.0) & (pressure <= lower)
            displayed_pressure = pressure.where(visible, drop=True)
            interval = products["ascent_intervals"][sim] if is_ascent else products["environment"][sim][name]
            interval = interval.where(visible, drop=True)
            colour = simulation_colour(sim)
            axis.fill_betweenx(
                displayed_pressure, scale * interval.ci_low, scale * interval.ci_high,
                color=colour, alpha=0.15, lw=0,
            )
            axis.plot(scale * interval["mean"], displayed_pressure, color=colour, lw=1.7)
        for sim in simulations:
            if sim not in local_profiles:
                continue
            profile = local_profiles[sim]
            visible = (profile.pressure_hpa >= 50.0) & (profile.pressure_hpa <= lower)
            profile = profile.where(visible, drop=True)
            axis.fill_betweenx(profile.pressure_hpa, scale * profile[f"{name}_ci_low"],
                               scale * profile[f"{name}_ci_high"], color=simulation_colour(sim),
                               alpha=0.045, lw=0)
            axis.plot(scale * profile[name], profile.pressure_hpa,
                      color=simulation_colour(sim), lw=1.15, ls="--", alpha=0.6)
        pressure_axis(axis, ylabel=column == 0, lower=lower)
        axis.set_title(f"({chr(97 + column)}) {title}", loc="left")
        label = "Vertical velocity" if is_ascent else title
        axis.set_xlabel(f"{label} ({units})")
    handles = []
    styles = [Line2D([0], [0], color="0.3", lw=1.7, label="Tropical mean"),
              Line2D([0], [0], color="0.3", lw=1.15, ls="--", alpha=0.6, label="LPO near peak")]
    for index, sim in enumerate(simulations):
        handles.append(Line2D([0], [0], color=simulation_colour(sim), lw=1.7,
                              label=ALL_SIMULATION_PUBLICATION_LABELS[sim]))
        if conditioned and index < len(styles):
            handles.append(styles[index])
    figure.legend(handles=handles, loc="outside lower center", ncol=len(simulations),
        frameon=False, fontsize=8)
    return figure
