"""Scientific products for the two convection publication figures.

Mean-state, high-cloud, and lifecycle owners retain their calculation products.
This module combines climate summaries and native near-peak LPO profiles.
It never rebuilds classifications or circulation diagnostics.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from diagnostics.analysis_config import (
    ANALYSIS_END, ANALYSIS_START, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, LATLON_PATTERN,
    SIMULATION_PROFILES, canonical_simulation_name,
    profile_cache_directory, simulation_directory,
)
from diagnostics.cache_support import (
    load_netcdf, read_json_file, source_file_signature,
)
from diagnostics.convection.refined_high_cloud_classification_diagnostics import (
    DERIVED_CACHE_VERSION, MIN_RAIN_CONDITIONED_BLOCKS, VERTICAL_PROFILE_BLOCK_SCHEMA,
)
from diagnostics.mean_state.convection_summary_cache import prepare_product
from diagnostics.mean_state.publication_profiles import field_response, prepare_profile_inputs


SIMULATIONS = SIMULATION_PROFILES["from-u"]
EXPERIMENTS = SIMULATIONS[1:]
CLOUD_CLASSES = (
    ("high_cloud_mask", "Total high cloud"),
    ("convective_precipitating_high_cloud_mask", "Deep convection"),
    ("stratiform_precipitating_high_cloud_mask", "Stratiform"),
    ("nonprecipitating_high_cloud_mask", "Nonprecipitating"),
)


def canonical_dataset(dataset):
    if "simulation" in dataset.coords:
        names = [canonical_simulation_name(str(v)) for v in dataset.simulation.values]
        if len(set(names)) != len(names) or set(names) != set(SIMULATIONS):
            raise ValueError("Publication products require exactly the from-u experiment set")
        dataset = dataset.assign_coords(simulation=names).sel(simulation=list(SIMULATIONS))
    return dataset


def read_cloud_product(stem, suffix):
    root = profile_cache_directory("high_cloud", profile="from-u")
    path = root / f"{stem}.{suffix}"
    metadata_path = root / f"{stem}.metadata.json"
    metadata = read_json_file(metadata_path)
    if not path.is_file() or not isinstance(metadata, dict):
        raise FileNotFoundError(f"Missing high-cloud product {path}; run its owning diagnostic")
    names = [canonical_simulation_name(v) for v in metadata.get("simulations", [])]
    if (metadata.get("derived_cache_version") != DERIVED_CACHE_VERSION
            or metadata.get("test_mode") or tuple(names) != SIMULATIONS):
        raise RuntimeError(f"Stale or test high-cloud product: {path}")
    for axis in metadata.get("time_axes", {}).values():
        if (pd.Timestamp(axis["first"]) != pd.Timestamp(ANALYSIS_START)
                or pd.Timestamp(axis["last"]) != pd.Timestamp(ANALYSIS_END)):
            raise RuntimeError(f"High-cloud analysis period differs: {path}")
    for label, signature in metadata.get("source_archives", {}).items():
        simulation = canonical_simulation_name(label)
        current = source_file_signature(sorted(simulation_directory(simulation).glob(LATLON_PATTERN)))
        if current["sha256"] != signature:
            raise RuntimeError(f"High-cloud source archives change: {path}; rebuild its owning diagnostic")
    if suffix == "nc":
        dataset = canonical_dataset(load_netcdf(path))
        if stem == "high_cloud_vertical_contribution_profiles" and (
            dataset.attrs.get("vertical_profile_block_schema") != VERTICAL_PROFILE_BLOCK_SCHEMA
            or not {"cloud_fraction_numerator_blocks", "cloud_fraction_denominator_blocks"}.issubset(dataset)
        ):
            raise RuntimeError(f"Missing cloud-profile uncertainty blocks: {path}; run high-cloud Figure 6 without --figures-only")
        return dataset, [path, metadata_path]
    table = pd.read_csv(path)
    table["simulation"] = table.simulation.map(canonical_simulation_name)
    return table, [path, metadata_path]


def cloud_profile_response(cloud, simulation, *, seed, n_resamples=BOOTSTRAP_REPLICATES):
    """Resample class sufficient statistics jointly, then form total-cloud replicates."""
    cloud = cloud.sel(latitude_domain="tropical").reset_coords(drop=True)
    means = cloud.domain_mean_cldfrac_contribution
    rng = np.random.default_rng(seed)
    replicates = []
    for sim in (simulation, "control"):
        numerator = cloud.cloud_fraction_numerator_blocks.sel(simulation=sim).transpose("block", "class_key", "z").values
        denominator = cloud.cloud_fraction_denominator_blocks.sel(simulation=sim).transpose("block", "class_key", "z").values
        if numerator.shape[0] < 2 or np.any(denominator <= 0):
            raise ValueError("Cloud uncertainty requires at least two complete blocks with positive weights")
        indices = rng.integers(0, numerator.shape[0], (n_resamples, numerator.shape[0]))
        replicates.append(numerator[indices].sum(axis=1) / denominator[indices].sum(axis=1))
    class_replicates = replicates[0] - replicates[1]
    difference = means.sel(simulation=simulation) - means.sel(simulation="control")
    profiles = {}
    for name, key in (
        ("total", None), ("nonprecipitating", "nonprecipitating_high_cloud_mask"),
        ("deep_convective", "convective_precipitating_high_cloud_mask"),
    ):
        if key is None:
            values, central = class_replicates.sum(axis=1), difference.sum("class_key")
        else:
            index = list(cloud.class_key.values).index(key)
            values, central = class_replicates[:, index], difference.sel(class_key=key, drop=True)
        low, high = np.percentile(values, [2.5, 97.5], axis=0)
        profiles[name] = xr.Dataset({
            "response": central, "ci_low": ("z", low), "ci_high": ("z", high),
        })
    return xr.concat(list(profiles.values()), dim=xr.IndexVariable("contribution", list(profiles)))


def prepare_inputs(figures, *, figures_only=False, rebuild=False):
    """Read shared surface products and figure-specific climate and LPO profiles."""
    from diagnostics.mean_state.convection_summary_cache import product_paths

    products = {}
    sources = []
    if 1 in figures:
        surface = {
            sim: prepare_product(sim, "surface", figures_only=figures_only, rebuild=rebuild)
            for sim in SIMULATIONS
        }
        products["surface_absolute"] = surface
        products["surface"] = {
            sim: field_response(
                surface[sim].surface_mean, surface["control"].surface_mean,
                surface[sim].surface_blocks, surface["control"].surface_blocks, BOOTSTRAP_SEED + index,
            ) for index, sim in enumerate(EXPERIMENTS)
        }
        for sim in SIMULATIONS:
            sources.extend(product_paths(sim, "surface"))
        for stem, suffix, key in (
            ("high_cloud_class_budget", "csv", "cloud_budget"),
            ("high_cloud_vertical_contribution_profiles", "nc", "cloud_profiles"),
            ("rain_matched_high_cloud_response", "csv", "rain_matched"),
        ):
            products[key], paths = read_cloud_product(stem, suffix)
            sources.extend(paths)
        products["cloud_response"] = {
            sim: cloud_profile_response(products["cloud_profiles"], sim, seed=BOOTSTRAP_SEED + 80 + index)
            for index, sim in enumerate(EXPERIMENTS)
        }
    if 2 in figures:
        profiles, paths = prepare_profile_inputs(SIMULATIONS, figures_only=figures_only, rebuild=rebuild)
        products.update(profiles)
        sources.extend(paths)
    products["provenance"] = source_file_signature(sources)
    return products


def selected_rain_matched(table, simulation):
    """Show populated precipitating high-cloud bins without dry-column changes."""
    return table[
        (table.simulation == simulation)
        & (table.statistic == "difference_from_control")
        & (table.class_key == "high_cloud_mask")
        & (table.property_key == "cloud_top_pressure")
        & (table.rain_bin_lower_mm_h >= 0.5)
        & (table.n_simulation_blocks >= MIN_RAIN_CONDITIONED_BLOCKS)
        & (table.n_control_blocks >= MIN_RAIN_CONDITIONED_BLOCKS)
    ].sort_values("rain_bin_index")
