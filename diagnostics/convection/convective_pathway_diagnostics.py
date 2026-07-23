#!/usr/bin/env python
"""Explicit cloud, convective-system, temperature, and wind comparisons.

The filtered predictors are the published Kelvin, easterly-wave, and ER bands
already produced by equatorial_wave_diagnostics.py. Cloud responses are never
filtered. The figures report only simulated quantities and experimental
contrasts: no intraseasonal mode or phase state is inferred.
"""

import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from diagnostics.diagnostic_helpers import (
    FigureSaver, ZONAL_MEAN_PATTERN, diagnostic_cache_dir,
    convective_organisation_cache_dir, diagnostic_output_dir,
    load_latlon_dataset, PRIMARY_TRACK_SOURCE, progress,
    PYFLEXTRKR_PRODUCT_VERSION,
    TRACK_POPULATION_LABELS, TRACK_POPULATIONS, table_boolean_mask,
    standardize_latlon_dataset, subset_latitude,
)
from diagnostics.convection.high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION, HighCloudThresholds,
    cached_classification_is_current,
)
from diagnostics.waves.cached_wave_products import load_existing_filtered_wave
from diagnostics.waves.equatorial_wave_config import CORE_WAVE_NAMES, WAVE_BOXES


save = True
save_loc = diagnostic_output_dir(
    "/home/cwp29/tmp/mpas/crf_on/final/convective_pathways/"
)
figures_to_run = [1, 2]
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]
WAVE_NAMES = tuple(CORE_WAVE_NAMES)

TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
TTL_Z_BOUNDS_M = (14_000.0, 18_000.0)
TTL_LAT_BOUNDS = (-10.0, 10.0)
CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN_MM_DAY = 12.0

classification_cache_dir = diagnostic_cache_dir(
    "classification_cache", root_dir=root_dir
)
organisation_output_dir = convective_organisation_cache_dir(root_dir=root_dir)
os.makedirs(save_loc, exist_ok=True)

CLASSIFICATION_VARIABLES = (
    "cloud_top_pressure",
    "high_cloud_mask",
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
)

CLOUD_RESPONSES = (
    (
        "convective_core",
        "PyFLEXTRKR convective core",
        "convective_precipitating_high_cloud_mask",
    ),
    (
        "stratiform",
        "Stratiform precipitating",
        "stratiform_precipitating_high_cloud_mask",
    ),
    (
        "nonprecipitating",
        "Non-precipitating",
        "nonprecipitating_high_cloud_mask",
    ),
    ("ttl_top", "TTL-reaching high cloud\n(CTP <= 180 hPa)", None),
)

TRACK_METRICS = (
    ("event_rate", "MCS event rate", "events per 100 days"),
    ("lifetime_hours", "Median MCS lifetime", "h"),
    ("max_area_km2", "Median maximum cold-shield area", "km2"),
)


def classification_thresholds():
    return HighCloudThresholds(
        cf_thresh=CF_THRESH,
        high_cloud_top_pressure_hpa=P_HIGH_HPA,
        surface_precip_min_mm_day=P_PRECIP_MIN_MM_DAY,
    )


def load_classification(simulation):
    path = os.path.join(
        classification_cache_dir,
        f"{simulation.replace(' ', '_')}_classified_high_cloud.nc",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{simulation}: missing shared high-cloud product {path}. Run "
            "refined_high_cloud_classification_diagnostics.py first."
        )
    dataset = xr.open_dataset(path, chunks={})
    if not cached_classification_is_current(dataset, classification_thresholds()):
        dataset.close()
        raise RuntimeError(f"{simulation}: stale high-cloud product {path}")
    missing = [name for name in CLASSIFICATION_VARIABLES if name not in dataset]
    if missing:
        dataset.close()
        raise RuntimeError(f"{simulation}: high-cloud product lacks {missing}: {path}")
    progress(f"{simulation}: using {path}")
    return dataset


def response_masks(classification):
    high = classification.high_cloud_mask.astype(bool)
    masks = {}
    for key, _label, variable in CLOUD_RESPONSES:
        masks[key] = (
            (high & (classification.cloud_top_pressure <= 180.0)).fillna(False)
            if variable is None else classification[variable].astype(bool)
        )
    return masks


def tropical_zonal_occurrence(mask):
    mask = subset_latitude(mask.astype(float), lat_bounds=TROPICAL_LAT_BOUNDS)
    weights = xr.DataArray(
        np.cos(np.deg2rad(mask.latitude.values)),
        dims=("latitude",),
        coords={"latitude": mask.latitude},
    )
    return mask.weighted(weights).mean("latitude", skipna=True)


def standardised_regression_slope(response, predictor):
    response, predictor = xr.align(response, predictor, join="inner")
    predictor_sigma = predictor.std(("Time", "longitude"), skipna=True)
    predictor_standard = predictor / predictor_sigma
    x_mean = predictor_standard.mean(("Time", "longitude"), skipna=True)
    y_mean = response.mean(("Time", "longitude"), skipna=True)
    x_anomaly = predictor_standard - x_mean
    y_anomaly = response - y_mean
    denominator = (x_anomaly**2).mean(("Time", "longitude"), skipna=True)
    slope = (x_anomaly * y_anomaly).mean(
        ("Time", "longitude"), skipna=True
    ) / denominator
    slope_value = slope.compute()
    sigma_value = predictor_sigma.compute()
    return float(slope_value.values), float(sigma_value.values)


def wave_cloud_regression_table(classifications):
    rows = []
    for simulation, classification in zip(sim_names, classifications):
        responses = {
            key: tropical_zonal_occurrence(mask).load()
            for key, mask in response_masks(classification).items()
        }
        for wave_name in WAVE_NAMES:
            predictor = load_existing_filtered_wave(
                root_dir, simulation, "Precip", wave_name
            )
            for response_key, response_label, _variable in CLOUD_RESPONSES:
                slope, predictor_sigma = standardised_regression_slope(
                    responses[response_key], predictor
                )
                rows.append(
                    dict(
                        simulation=simulation,
                        wave=wave_name,
                        response_key=response_key,
                        response_label=response_label.replace("\n", " "),
                        occurrence_slope_per_predictor_std=slope,
                        predictor_std_mm_day=predictor_sigma,
                        predictor=f"{wave_name}-filtered 10S-10N precipitation",
                        response="unfiltered tropical cloud occurrence",
                        units="occurrence fraction per predictor standard deviation",
                    )
                )
    return pd.DataFrame(rows)


def realised_ttl_temperature_mean(zonal_dataset):
    theta = zonal_dataset.theta
    pressure = zonal_dataset.pressure
    pressure_pa = xr.where(pressure < 2_000.0, pressure * 100.0, pressure)
    temperature = theta * (pressure_pa / 100_000.0) ** (287.0 / 1004.0)
    temperature = temperature.where(
        (temperature.z >= TTL_Z_BOUNDS_M[0])
        & (temperature.z <= TTL_Z_BOUNDS_M[1]),
        drop=True,
    ).sel(latitude=slice(*TTL_LAT_BOUNDS))
    weights = np.cos(np.deg2rad(temperature.latitude))
    return float(
        temperature.weighted(weights).mean(
            [dim for dim in ("Time", "z", "latitude") if dim in temperature.dims],
            skipna=True,
        ).load().values
    )


def realised_temperature_scaling(zonal_datasets):
    means = np.asarray(
        [realised_ttl_temperature_mean(dataset) for dataset in zonal_datasets]
    )
    delta_u = means[1] - means[0]
    delta_t = means[2] - means[0]
    alpha = delta_u / delta_t if np.isfinite(delta_t) and abs(delta_t) > 1e-12 else np.nan
    table = pd.DataFrame(
        {
            "simulation": sim_names,
            "mean_temperature_14_18km_10S_10N_K": means,
            "temperature_change_from_control_K": means - means[0],
            "thermal_scaling_alpha_deltaT_u_over_deltaT_T": alpha,
        }
    )
    table.to_csv(os.path.join(save_loc, "pathway_temperature_scaling.csv"), index=False)
    return float(alpha)


def scaled_contrasts(table, index_columns, value_column, alpha):
    pivot = table.pivot(
        index=index_columns, columns="simulation", values=value_column
    )
    rows = []
    for index, values in pivot.iterrows():
        index = index if isinstance(index, tuple) else (index,)
        base = dict(zip(index_columns, index))
        control = values[sim_names[0]]
        delta_u = values[sim_names[1]] - control
        delta_t = values[sim_names[2]] - control
        for contrast_key, contrast_label, value in (
            ("T_minus_control", "T-nudged - control", delta_t),
            ("U_minus_control", "U-nudged - control", delta_u),
            (
                "U_residual_after_thermal_scaling",
                "U response after scaled temperature response",
                delta_u - alpha * delta_t if np.isfinite(alpha) else np.nan,
            ),
        ):
            rows.append(
                {
                    **base,
                    "contrast_key": contrast_key,
                    "contrast_label": contrast_label,
                    "response_change": value,
                    "thermal_scaling_alpha": alpha,
                }
            )
    return pd.DataFrame(rows)


def load_convective_system_tracks():
    path = os.path.join(
        organisation_output_dir, "organisation_track_lifetimes.csv"
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing dual-population track product {path}. Run "
            "convective_organisation_diagnostics.py first."
        )
    metadata_path = os.path.join(
        organisation_output_dir,
        "organisation_track_persistence_cache_metadata.json",
    )
    if not os.path.exists(metadata_path):
        raise FileNotFoundError(
            f"Missing dual-population track metadata {metadata_path}. Rebuild the "
            "organisation diagnostic."
        )
    with open(metadata_path, "r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    expected = {
        "cloud_classification_cache_version": CLASSIFICATION_CACHE_VERSION,
        "product_version": PYFLEXTRKR_PRODUCT_VERSION,
        "primary_track_source": PRIMARY_TRACK_SOURCE,
    }
    changed = [
        key for key, value in expected.items() if metadata.get(key) != value
    ]
    if changed:
        raise RuntimeError(
            f"Track metadata is stale ({changed}): {metadata_path}. "
            "Rebuild convective_organisation_diagnostics.py."
        )
    tracks = pd.read_csv(path)
    required = {
        "simulation", "track_id", "lifetime_hours", "max_area_km2",
        *TRACK_POPULATIONS,
    }
    missing = sorted(required - set(tracks.columns))
    if missing:
        raise RuntimeError(f"Dual-population track table lacks {missing}: {path}")
    return tracks


def analysed_duration_days(classification):
    time = classification.Time.values
    if len(time) < 2:
        return np.nan
    differences = np.diff(time)
    if np.issubdtype(time.dtype, np.datetime64):
        step = float(np.nanmedian(differences / np.timedelta64(1, "D")))
        span = float((time[-1] - time[0]) / np.timedelta64(1, "D"))
    else:
        step = float(np.nanmedian(differences))
        span = float(time[-1] - time[0])
    return span + step


def convective_system_population_table(classifications, tracks):
    rows = []
    for population in TRACK_POPULATIONS:
        population_tracks = tracks[table_boolean_mask(tracks, population)]
        population_label = TRACK_POPULATION_LABELS[population]
        for simulation, classification in zip(sim_names, classifications):
            subset = population_tracks[population_tracks.simulation == simulation]
            duration = analysed_duration_days(classification)
            for metric_key, metric_label, units in TRACK_METRICS:
                display_label = metric_label.replace("MCS", "system")
                if metric_key == "event_rate":
                    value = 100.0 * len(subset) / duration if duration > 0.0 else np.nan
                    denominator = "population events divided by analysed duration"
                else:
                    value = float(subset[metric_key].median()) if len(subset) else np.nan
                    denominator = "median across population events"
                rows.append(
                    dict(
                        track_population=population,
                        track_population_label=population_label,
                        simulation=simulation,
                        metric_key=metric_key,
                        metric_label=display_label,
                        units=units,
                        value=value,
                        event_count=int(len(subset)),
                        analysed_duration_days=duration,
                        denominator=denominator,
                    )
                )
    return pd.DataFrame(rows)


def plot_wave_cloud_pathways(regressions, contrasts, alpha):
    response_keys = [key for key, _label, _variable in CLOUD_RESPONSES]
    response_labels = [label for _key, label, _variable in CLOUD_RESPONSES]
    contrast_specs = (
        ("T_minus_control", "T-nudged - control", "tab:red"),
        ("U_minus_control", "U-nudged - control", "tab:blue"),
        (
            "U_residual_after_thermal_scaling",
            "U residual after\nthermal scaling",
            "tab:purple",
        ),
    )
    fig, axes = plt.subplots(
        len(WAVE_NAMES), 4, figsize=(14.8, 9.2), constrained_layout=True,
        sharey=True,
    )
    x = np.arange(len(response_keys))
    width = 0.8 / len(sim_names)
    for row, wave_name in enumerate(WAVE_NAMES):
        wave_rows = regressions[regressions.wave == wave_name]
        for simulation_index, (simulation, color) in enumerate(zip(sim_names, cols)):
            values = wave_rows[wave_rows.simulation == simulation].set_index(
                "response_key"
            )
            axes[row, 0].bar(
                x + (simulation_index - 1) * width,
                [100.0 * values.loc[key, "occurrence_slope_per_predictor_std"] for key in response_keys],
                width=width,
                color=color,
                label=simulation,
            )
        axes[row, 0].set_ylabel(
            f"{wave_name}\nOccurrence slope\n(percentage points per +1 std)"
        )
        wave_contrasts = contrasts[contrasts.wave == wave_name]
        for column, (contrast_key, title, color) in enumerate(contrast_specs, start=1):
            values = wave_contrasts[
                wave_contrasts.contrast_key == contrast_key
            ].set_index("response_key")
            axes[row, column].bar(
                x,
                [100.0 * values.loc[key, "response_change"] for key in response_keys],
                color=color,
                alpha=0.82,
            )
            if row == 0:
                axes[row, column].set_title(title)
        for axis in axes[row]:
            axis.axhline(0.0, color="0.45", lw=0.8)
            axis.set_xticks(x)
            axis.set_xticklabels(response_labels, rotation=27, ha="right")
            axis.grid(axis="y", alpha=0.2)
    axes[0, 0].set_title("Regression slope by simulation")
    axes[0, 0].legend(frameon=False, fontsize=7)
    alpha_text = f"alpha={alpha:.2f}" if np.isfinite(alpha) else "alpha unavailable"
    fig.suptitle(
        "Unfiltered high-cloud occurrence regressed on filtered CCEW precipitation\n"
        "Rows identify the filtered predictor; cloud response is unfiltered | "
        + alpha_text,
        fontsize=11,
    )
    figsave.save(fig, "ccew_cloud_utls_nudging_contrasts", number=1)
    plt.close(fig)


def plot_convective_system_pathways(population, contrasts, alpha):
    metric_keys = [key for key, _label, _units in TRACK_METRICS]
    contrast_specs = (
        ("T_minus_control", "T-control", "tab:red"),
        ("U_minus_control", "U-control", "tab:blue"),
        (
            "U_residual_after_thermal_scaling",
            "U residual after\nthermal scaling",
            "tab:purple",
        ),
    )
    for track_population in TRACK_POPULATIONS:
        population_rows = population[
            population.track_population == track_population
        ]
        contrast_population = contrasts[
            contrasts.track_population == track_population
        ]
        if len(population_rows) == 0:
            continue
        fig, axes = plt.subplots(2, 3, figsize=(13.0, 7.6), constrained_layout=True)
        for column, metric_key in enumerate(metric_keys):
            rows = population_rows[population_rows.metric_key == metric_key].set_index("simulation")
            metadata = population_rows[population_rows.metric_key == metric_key].iloc[0]
            axes[0, column].bar(np.arange(len(sim_names)), [rows.loc[name, "value"] for name in sim_names], color=cols)
            axes[0, column].set_xticks(np.arange(len(sim_names)))
            axes[0, column].set_xticklabels(sim_names, rotation=20, ha="right")
            axes[0, column].set(title=metadata.metric_label, ylabel=metadata.units)
            contrast_rows = contrast_population[contrast_population.metric_key == metric_key].set_index("contrast_key")
            axes[1, column].bar(np.arange(3), [contrast_rows.loc[key, "response_change"] for key, _label, _color in contrast_specs], color=[color for _key, _label, color in contrast_specs])
            axes[1, column].set_xticks(np.arange(3))
            axes[1, column].set_xticklabels([label for _key, label, _color in contrast_specs], rotation=20, ha="right")
            axes[1, column].set_ylabel(f"Change in {metadata.units}")
            axes[1, column].axhline(0.0, color="0.45", lw=0.8)
            for row in range(2):
                axes[row, column].grid(axis="y", alpha=0.2)
        alpha_text = f"alpha={alpha:.2f}" if np.isfinite(alpha) else "alpha unavailable"
        fig.suptitle(
            f"{TRACK_POPULATION_LABELS[track_population]} under UTLS nudging\n"
            "Top: simulated values; bottom: experimental nudging contrasts | " + alpha_text,
            fontsize=11,
        )
        figsave.save(fig, f"{track_population}_utls_nudging_contrasts", number=2)
        plt.close(fig)


def write_method_table():
    rows = []
    for wave_name in WAVE_NAMES:
        box = WAVE_BOXES[wave_name]
        rows.append(
            dict(
                component="cloud predictor",
                predictor=wave_name,
                method=(
                    f"existing 10S-10N filtered precipitation cache; "
                    f"k={box['k']}, period={box['period']} days"
                ),
                response="unfiltered 20S-20N cloud-class occurrence",
                statistic="linear occurrence slope per predictor standard deviation",
                inference="none; descriptive association in the simulated CCEW band",
                reference=(
                    "Rios-Berrios et al. (2023), J. Climate"
                    if wave_name == "Easterly"
                    else "Wheeler-Kiladis filter convention"
                ),
            )
        )
    rows.extend(
        [
            dict(
                component="temperature experiment",
                predictor="not applicable",
                method="T-nudged minus control",
                response="cloud-regression slope or convective-system population metric",
                statistic="experimental difference",
                inference="temperature-associated response",
                reference="experiment design",
            ),
            dict(
                component="U residual",
                predictor="not applicable",
                method="U-control minus alpha times T-control; alpha from realised 14-18-km temperature changes",
                response="cloud-regression slope or convective-system population metric",
                statistic="scaled experimental contrast",
                inference="response remaining after realised-temperature scaling",
                reference="experiment design",
            ),
        ]
    )
    pd.DataFrame(rows).to_csv(
        os.path.join(save_loc, "convective_pathway_method.csv"), index=False
    )


def main():
    progress("Loading shared classification and zonal-mean temperature products")
    classifications = []
    zonal_datasets = []
    for data_loc, simulation in zip(data_locs, sim_names):
        classifications.append(load_classification(simulation))
        dataset = load_latlon_dataset(
            os.path.join(root_dir, data_loc),
            latlon_pattern=ZONAL_MEAN_PATTERN,
            chunks={"Time": -1},
        )
        zonal_datasets.append(
            standardize_latlon_dataset(dataset, ["theta", "pressure"])
        )

    alpha = realised_temperature_scaling(zonal_datasets)
    regressions = wave_cloud_regression_table(classifications)
    regressions.to_csv(
        os.path.join(save_loc, "ccew_unfiltered_cloud_occurrence_regressions.csv"),
        index=False,
    )
    cloud_contrasts = scaled_contrasts(
        regressions,
        ["wave", "response_key"],
        "occurrence_slope_per_predictor_std",
        alpha,
    )
    cloud_contrasts.to_csv(
        os.path.join(save_loc, "ccew_cloud_pathway_contrasts.csv"), index=False
    )

    population = convective_system_population_table(
        classifications, load_convective_system_tracks()
    )
    population.to_csv(
        os.path.join(save_loc, "convective_system_population_pathway_metrics.csv"), index=False
    )
    track_contrasts = scaled_contrasts(
        population, ["track_population", "metric_key"], "value", alpha
    )
    track_contrasts.to_csv(
        os.path.join(save_loc, "convective_system_population_pathway_contrasts.csv"), index=False
    )

    write_method_table()
    plot_wave_cloud_pathways(regressions, cloud_contrasts, alpha)
    plot_convective_system_pathways(population, track_contrasts, alpha)
    print("\nFinished convective_pathway_diagnostics.py")


if __name__ == "__main__":
    main()
