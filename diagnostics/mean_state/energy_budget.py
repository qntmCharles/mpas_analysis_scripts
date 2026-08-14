#!/usr/bin/env python
# coding: utf-8

"""Global atmospheric energy-budget closure diagnostics.

This diagnostic keeps the fixed-SST/surface uptake separate from the
atmospheric column closure.  Positive atmospheric forcing and storage tendency
mean energy gain by the atmosphere.  The plotted residual is

    dE_atm/dt - diagnosed atmospheric forcing

where diagnosed atmospheric forcing is atmospheric radiative heating plus
surface sensible and latent heat fluxes.  Precipitation latent heating is shown
only as internal latent-to-dry-static bookkeeping, not added again to the total
atmospheric forcing.
"""

import argparse
import glob
import json
import os
import sys
from functools import partial

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr

from diagnostics.diagnostic_helpers import (
    ANALYSIS_END, ANALYSIS_START, FigureSaver, LATLON_PATTERN,
    annotate_panel_label, cache_manifest, cache_manifest_matches,
    diagnostic_cache_dir,
    diagnostic_output_dir, figure_should_run, load_latlon_dataset,
    mpas_moist_air_density, netcdf_safe_dataset, precip_rate_from_accum,
    standardize_latlon_dataset, time_block_slices, total_precip_accum, vertical_dim,
)
from diagnostics.analysis_config import (
    BLOCK_LENGTH_DAYS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SIMULATION_ARCHIVES,
    SIMULATION_LABELS as PROJECT_SIMULATION_LABELS,
    SIMULATION_ROOT,
)
from diagnostics.uncertainty import independent_bootstrap_difference



save = True
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/global_atmospheric_energy_budget/")

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: atmospheric_closure_timeseries
#   2: atmospheric_closure_cumulative
#   3: atmospheric_budget_decomposition
#
# WHAT TO LOOK FOR:
#   * The atmospheric residual is the headline quantity: residual =
#     atmospheric storage tendency minus atmospheric forcing.
#   * Fixed-SST/surface uptake is shown separately from atmospheric forcing.
#     A large surface uptake does not by itself imply atmospheric energy drift.
#   * If the dry+latent-only residual is much larger than the full residual,
#     missing KE storage was contributing materially to the apparent imbalance.
figures_to_run = []


should_run_figure = partial(figure_should_run, figures_to_run)


RD = 287.0
CP = 1004.0
P0_PA = 100000.0
G = 9.80665
LV = 2.5e6
DT_HOURS = 6
ENERGY_CACHE_SCHEMA = "energy-budget-v2-mpas-moist-density"
ENERGY_CACHE_ROOT = diagnostic_cache_dir("energy_budget")


root_dir = str(SIMULATION_ROOT)
data_locs = list(SIMULATION_ARCHIVES.values())
sim_names = list(PROJECT_SIMULATION_LABELS.values())

latlon_chunks = {
    "Time": 8,
    "nVertLevels": 25,
    "nVertLevelsP1": 26,
    "latitude": 60,
    "longitude": 120,
}

MOISTURE_MIXING_RATIO_VARIABLES = ("qv", "qc", "qr", "qi", "qs", "qg")

needed_vars = [
    "theta", "pressure", "rho", *MOISTURE_MIXING_RATIO_VARIABLES, "zgrid",
    "ke", "uReconstructZonal", "uReconstructMeridional",
    "rainnc", "i_rainnc", "rainc", "i_rainc",
    "lwdnt", "lwdnb", "lwupb", "lwupt",
    "swdnt", "swdnb", "swupb", "swupt",
    "lh", "hfx",
]


TERM_LABELS = {
    "net_toa": "Net TOA/system input",
    "surface_radiation": "Net surface radiation, down",
    "fixed_sst_surface_uptake": "Fixed-SST/surface energy uptake",
    "atmospheric_radiation": "Atmospheric radiative heating",
    "surface_sensible": "Surface sensible heat into atmosphere",
    "surface_latent": "Surface latent heat into atmosphere",
    "atmospheric_forcing": "Atmospheric forcing: rad + SH + LH",
    "precip_latent_release": "Precip latent release, internal only",
    "dry_static_storage_tendency": "Dry/static storage tendency",
    "latent_storage_tendency": "Latent storage tendency",
    "kinetic_storage_tendency": "KE storage tendency",
    "atmospheric_storage_tendency": "Atmospheric storage tendency",
    "residual_full": "Unresolved residual: storage - forcing",
    "residual_dry_latent_only": "Residual if KE storage omitted",
    "residual_no_storage": "Residual if all storage omitted",
}

SERIES_NAMES = tuple(TERM_LABELS)
STORAGE_NAMES = (
    "dry_static_energy",
    "latent_energy",
    "kinetic_energy",
)


def global_area_mean(obj, lat_dim="latitude", lon_dim="longitude"):
    dims = [dim for dim in (lat_dim, lon_dim) if dim in obj.dims]
    if lat_dim not in obj.dims:
        name = getattr(obj, "name", None)
        raise ValueError(f"Cannot area-average {name}: missing {lat_dim!r} dimension")
    weights = np.cos(np.deg2rad(obj[lat_dim]))
    return obj.weighted(weights).mean(dim=dims, skipna=True)


def time_seconds_since_start(da, origin=None):
    times = da.Time.values
    if not np.issubdtype(times.dtype, np.datetime64):
        raise TypeError(f"Expected decoded datetime Time coordinate, got {times.dtype}")
    origin_time = origin if origin is not None else times[0]
    return np.asarray((times - origin_time) / np.timedelta64(1, "s"), dtype=float)


def time_days_since_start(da, origin=None):
    return time_seconds_since_start(da, origin=origin) / 86400.0


def cumulative_energy_mj_m2(ts, origin=None):
    t_seconds = time_seconds_since_start(ts, origin=origin)
    flux = np.asarray(ts.values, dtype=float)
    cumulative = np.zeros_like(flux, dtype=float)
    if flux.size > 1:
        increments = 0.5 * (flux[1:] + flux[:-1]) * np.diff(t_seconds) / 1.0e6
        cumulative[1:] = np.cumsum(increments)
    return t_seconds / 86400.0, cumulative


def storage_tendency(energy_ts, name):
    seconds = time_seconds_since_start(energy_ts)
    values = np.asarray(energy_ts.values, dtype=float)
    tendency = np.full_like(values, np.nan, dtype=float)
    if values.size >= 2 and np.all(np.diff(seconds) > 0.0):
        tendency = np.gradient(values, seconds)
    out = xr.DataArray(
        tendency,
        dims=("Time",),
        coords={"Time": energy_ts.Time},
        name=name,
        attrs={"units": "W m-2"},
    )
    return out


def layer_geometry(ds, template):
    layer_coord = np.arange(template.sizes["nVertLevels"])
    lower = ds.zgrid.isel(nVertLevelsP1=slice(0, -1)).rename(
        {"nVertLevelsP1": "nVertLevels"}
    ).assign_coords(nVertLevels=layer_coord)
    upper = ds.zgrid.isel(nVertLevelsP1=slice(1, None)).rename(
        {"nVertLevelsP1": "nVertLevels"}
    ).assign_coords(nVertLevels=layer_coord)

    dz = np.abs(upper - lower)
    zmid = 0.5 * (upper + lower)
    dz.name = "layer_thickness"
    zmid.name = "layer_midpoint_height"
    dz.attrs["units"] = "m"
    zmid.attrs["units"] = "m"
    return dz, zmid


def air_temperature(ds):
    temperature = ds.theta * (ds.pressure / P0_PA) ** (RD / CP)
    temperature.name = "temperature"
    temperature.attrs["units"] = "K"
    return temperature


def total_moisture_mixing_ratio(ds):
    """Return total moist-species mass per unit dry-air mass."""

    total = xr.zeros_like(ds.qv)
    for name in MOISTURE_MIXING_RATIO_VARIABLES:
        total = total + ds[name]
    total.name = "total_moisture_mixing_ratio"
    total.attrs.update(
        units="kg kg-1",
        basis="moist-species mass per unit dry-air mass",
    )
    return total


def moist_air_density(ds, total_moisture=None):
    """Apply the shared MPAS-A moist-density definition."""

    return mpas_moist_air_density(ds, total_moisture=total_moisture)


def water_vapour_mass_fraction(ds, total_moisture=None):
    """Return water-vapour mass per unit total moist-air mass."""

    if total_moisture is None:
        total_moisture = total_moisture_mixing_ratio(ds)
    fraction = ds.qv / (1.0 + total_moisture)
    fraction.name = "water_vapour_mass_fraction"
    fraction.attrs.update(
        units="kg kg-1",
        basis="water-vapour mass per unit total moist-air mass",
    )
    return fraction


def layer_mass_per_area(ds, temperature, total_moisture=None):
    dz, zmid = layer_geometry(ds, temperature)
    layer_mass = moist_air_density(ds, total_moisture) * dz
    layer_mass.name = "layer_mass_per_area"
    layer_mass.attrs.update(
        units="kg m-2",
        mass_basis="total moist air, including all MPAS moisture scalars",
    )
    return layer_mass, zmid


def kinetic_energy_specific(ds, template):
    ke = ds.ke
    ke.name = "specific_kinetic_energy"
    ke.attrs.update(units="m2 s-2", source="ke")
    return ke


def atmospheric_column_storage(ds):
    temperature = air_temperature(ds)
    total_moisture = total_moisture_mixing_ratio(ds)
    layer_mass, zmid = layer_mass_per_area(
        ds, temperature, total_moisture
    )
    vapour_mass_fraction = water_vapour_mass_fraction(ds, total_moisture)
    z_dim = vertical_dim(temperature)

    dry_static_column = ((CP * temperature + G * zmid) * layer_mass).sum(
        dim=z_dim, skipna=True
    )
    dry_static_column.name = "dry_static_energy"
    dry_static_column.attrs["units"] = "J m-2"

    latent_column = (LV * vapour_mass_fraction * layer_mass).sum(
        dim=z_dim, skipna=True
    )
    latent_column.name = "latent_energy"
    latent_column.attrs.update(units="J m-2", note="water-vapour latent storage")

    ke = kinetic_energy_specific(ds, temperature)
    kinetic_column = (ke * layer_mass).sum(dim=z_dim, skipna=True)
    kinetic_column.name = "kinetic_energy"
    kinetic_column.attrs.update(units="J m-2", note="ke")

    storage_ds = xr.Dataset(
        {
            "dry_static_energy": dry_static_column,
            "latent_energy": latent_column,
            "kinetic_energy": kinetic_column,
        }
    )
    return global_area_mean(storage_ds).load()


def global_energy_budget_fields(ds):
    """
    Return 2-D flux terms in W m-2.

    Positive TOA and surface radiative terms are downward. Surface turbulent
    fluxes are positive upward into the atmosphere, matching the existing
    mean-state convention. Atmospheric forcing is external atmospheric energy
    input: atmospheric radiation plus surface sensible and latent heat.
    """
    net_toa_sw = ds.swdnt - ds.swupt
    net_toa_lw = ds.lwdnt - ds.lwupt
    net_sfc_sw = ds.swdnb - ds.swupb
    net_sfc_lw = ds.lwdnb - ds.lwupb

    net_toa = net_toa_sw + net_toa_lw
    surface_radiation = net_sfc_sw + net_sfc_lw
    atmospheric_radiation = net_toa - surface_radiation

    surface_latent = ds.lh
    surface_sensible = ds.hfx
    surface_turbulent = surface_latent + surface_sensible

    fields = {
        "net_toa": net_toa,
        "surface_radiation": surface_radiation,
        "fixed_sst_surface_uptake": surface_radiation - surface_turbulent,
        "atmospheric_radiation": atmospheric_radiation,
        "surface_sensible": surface_sensible,
        "surface_latent": surface_latent,
        "atmospheric_forcing": atmospheric_radiation + surface_turbulent,
    }

    precip = precip_rate_from_accum(total_precip_accum(ds))
    fields["precip_latent_release"] = precip * LV / 86400.0

    for name, field in fields.items():
        field.name = name
        field.attrs["units"] = "W m-2"

    return fields


def derive_budget_series(ds):
    flux_fields = global_energy_budget_fields(ds)
    flux_series = {
        name: global_area_mean(field).load()
        for name, field in flux_fields.items()
    }

    storage_energy = atmospheric_column_storage(ds)
    dry = storage_tendency(storage_energy.dry_static_energy, "dry_static_storage_tendency")
    latent = storage_tendency(storage_energy.latent_energy, "latent_storage_tendency")
    kinetic = storage_tendency(storage_energy.kinetic_energy, "kinetic_storage_tendency")
    storage = dry + latent + kinetic
    storage.name = "atmospheric_storage_tendency"
    storage.attrs["units"] = "W m-2"

    forcing, storage, dry, latent, kinetic = xr.align(
        flux_series["atmospheric_forcing"], storage, dry, latent, kinetic, join="inner"
    )
    residual = storage - forcing
    residual.name = "residual_full"
    residual.attrs["units"] = "W m-2"

    residual_dry_latent = dry + latent - forcing
    residual_dry_latent.name = "residual_dry_latent_only"
    residual_dry_latent.attrs["units"] = "W m-2"

    residual_no_storage = -forcing
    residual_no_storage.name = "residual_no_storage"
    residual_no_storage.attrs["units"] = "W m-2"

    series = dict(flux_series)
    series.update(
        {
            "dry_static_storage_tendency": dry,
            "latent_storage_tendency": latent,
            "kinetic_storage_tendency": kinetic,
            "atmospheric_storage_tendency": storage,
            "residual_full": residual,
            "residual_dry_latent_only": residual_dry_latent,
            "residual_no_storage": residual_no_storage,
        }
    )
    return series, storage_energy


def term_summary(sim_name, term, ts, origin):
    values = np.asarray(ts.values, dtype=float)
    t_days, cumulative = cumulative_energy_mj_m2(ts, origin=origin)
    span_seconds = (t_days[-1] - t_days[0]) * 86400.0 if t_days.size > 1 else np.nan
    cumulative_mean = (
        cumulative[-1] * 1.0e6 / span_seconds
        if np.isfinite(span_seconds) and span_seconds > 0.0
        else np.nan
    )
    return {
        "simulation": sim_name,
        "term": term,
        "label": TERM_LABELS.get(term, term),
        "mean_w_m2": float(np.nanmean(values)),
        "std_w_m2": float(np.nanstd(values)),
        "final_cumulative_mj_m2": float(cumulative[-1]) if cumulative.size else np.nan,
        "cumulative_equivalent_w_m2": float(cumulative_mean),
    }


def timeseries_rows(sim_name, term, ts, origin):
    t_days, cumulative = cumulative_energy_mj_m2(ts, origin=origin)
    for time_index, (time_day, flux, cumulative_energy) in enumerate(
        zip(t_days, np.asarray(ts.values, dtype=float), cumulative)
    ):
        yield {
            "simulation": sim_name,
            "term": term,
            "label": TERM_LABELS.get(term, term),
            "time_index": time_index,
            "time_days": float(time_day),
            "flux_w_m2": float(flux),
            "cumulative_mj_m2": float(cumulative_energy),
        }


def energy_budget_uncertainty(entries):
    """Return independent-climate intervals from cached energy time series."""

    entry_by_name = {entry["simulation"]: entry for entry in entries}
    comparisons = (
        (sim_names[1], sim_names[0]),
        (sim_names[2], sim_names[0]),
        (sim_names[1], sim_names[2]),
    )
    rows = []
    block_days = BLOCK_LENGTH_DAYS["energy_budget"]
    for term_index, term in enumerate(SERIES_NAMES):
        blocks = {}
        for simulation, entry in entry_by_name.items():
            if term not in entry["series"]:
                continue
            series = entry["series"][term]
            blocks[simulation] = np.asarray(
                [
                    float(
                        series.isel(Time=time_slice)
                        .mean("Time", skipna=True)
                        .compute()
                        .values
                    )
                    for time_slice in time_block_slices(
                        series, block_days=block_days, complete=True
                    )
                ],
                dtype=float,
            )
        for comparison_index, (simulation, reference) in enumerate(comparisons):
            if simulation not in blocks or reference not in blocks:
                continue
            result = independent_bootstrap_difference(
                blocks[simulation],
                blocks[reference],
                rng=np.random.default_rng(
                    BOOTSTRAP_SEED + 4000 + 10 * term_index + comparison_index
                ),
                n_resamples=BOOTSTRAP_REPLICATES,
            )
            rows.append(
                {
                    "term": term,
                    "label": TERM_LABELS.get(term, term),
                    "simulation": simulation,
                    "reference": reference,
                    "mean_response_w_m2": result["estimate"],
                    "ci_low_w_m2": result["interval_low"],
                    "ci_high_w_m2": result["interval_high"],
                    "sign_probability": result["sign_probability"],
                    "n_simulation_blocks": result["n_simulation"],
                    "n_reference_blocks": result["n_control"],
                    "block_days": block_days,
                    "resampling": "independent complete time blocks by simulation",
                }
            )
    return pd.DataFrame(rows)


def print_budget_summary(summary_df):
    print("\nGlobal atmospheric energy-budget closure")
    print("Sign convention: positive forcing/storage means atmospheric energy gain.")
    print("Residual = atmospheric storage tendency - atmospheric forcing.")
    print("Precip latent release is internal bookkeeping and is not added to total forcing.\n")

    key_terms = [
        "residual_full",
        "atmospheric_storage_tendency",
        "dry_static_storage_tendency",
        "latent_storage_tendency",
        "kinetic_storage_tendency",
        "atmospheric_forcing",
        "atmospheric_radiation",
        "surface_sensible",
        "surface_latent",
        "fixed_sst_surface_uptake",
        "net_toa",
        "residual_dry_latent_only",
        "precip_latent_release",
    ]
    table = summary_df[summary_df["term"].isin(key_terms)].copy()
    table["term"] = pd.Categorical(table["term"], categories=key_terms, ordered=True)
    table = table.sort_values(["simulation", "term"])
    print(table.pivot(index="simulation", columns="term", values="mean_w_m2"))

    for sim_name, sub in table.groupby("simulation", sort=False):
        values = sub.set_index("term")
        def get(term, column="mean_w_m2"):
            return float(values.loc[term, column]) if term in values.index else np.nan

        print(f"\n{sim_name}:")
        print(f"  mean atmospheric residual: {get('residual_full'):.3f} W m-2")
        print(f"  mean storage tendency: {get('atmospheric_storage_tendency'):.3f} W m-2")
        print(f"    dry/static: {get('dry_static_storage_tendency'):.3f} W m-2")
        print(f"    latent:     {get('latent_storage_tendency'):.3f} W m-2")
        print(f"    KE:         {get('kinetic_storage_tendency'):.3f} W m-2")
        print(f"  mean atmospheric forcing: {get('atmospheric_forcing'):.3f} W m-2")
        print(f"    radiation: {get('atmospheric_radiation'):.3f} W m-2")
        print(f"    sensible:  {get('surface_sensible'):.3f} W m-2")
        print(f"    latent:    {get('surface_latent'):.3f} W m-2")
        print(f"  fixed-SST/surface uptake: {get('fixed_sst_surface_uptake'):.3f} W m-2")
        print(f"  residual if KE omitted: {get('residual_dry_latent_only'):.3f} W m-2")
        print(
            "  cumulative residual: "
            f"{get('residual_full', 'final_cumulative_mj_m2'):.3f} MJ m-2 "
            f"({get('residual_full', 'cumulative_equivalent_w_m2'):.3f} W m-2 equivalent)"
        )


def plot_timeseries(entries, figsave):
    if not should_run_figure(1):
        return

    fig, ax = plt.subplots(
        len(entries), 3, figsize=(13, 3.2 * len(entries)),
        constrained_layout=True, sharex="col"
    )
    if len(entries) == 1:
        ax = np.array([ax])

    system_terms = ["net_toa", "fixed_sst_surface_uptake", "atmospheric_forcing"]
    closure_terms = ["atmospheric_forcing", "atmospheric_storage_tendency", "residual_full"]
    storage_terms = [
        "dry_static_storage_tendency",
        "latent_storage_tendency",
        "kinetic_storage_tendency",
    ]
    styles = {
        "net_toa": dict(color="0.15", lw=1.4, ls="-"),
        "fixed_sst_surface_uptake": dict(color="tab:orange", lw=1.3, ls="-"),
        "atmospheric_forcing": dict(color="tab:blue", lw=1.4, ls="-"),
        "atmospheric_storage_tendency": dict(color="tab:green", lw=1.4, ls="-"),
        "residual_full": dict(color="tab:red", lw=1.4, ls="-"),
        "dry_static_storage_tendency": dict(color="tab:purple", lw=1.2, ls="-"),
        "latent_storage_tendency": dict(color="tab:cyan", lw=1.2, ls="-"),
        "kinetic_storage_tendency": dict(color="tab:brown", lw=1.2, ls="-"),
    }

    for row, entry in enumerate(entries):
        sim_name = entry["simulation"]
        series = entry["series"]
        origin = entry["origin"]
        annotate_panel_label(ax[row, 0], sim_name)

        for col, terms in enumerate((system_terms, closure_terms, storage_terms)):
            for term in terms:
                if term not in series:
                    continue
                ts = series[term]
                t_days = time_days_since_start(ts, origin=origin)
                ax[row, col].plot(
                    t_days, ts.values,
                    label=TERM_LABELS.get(term, term),
                    **styles.get(term, {}),
                )
            ax[row, col].axhline(0.0, color="0.35", lw=0.7)
            if col == 0:
                ax[row, col].set_ylabel(r"W m$^{-2}$")

    ax[0, 0].set_title("System fluxes, surface separate")
    ax[0, 1].set_title("Atmospheric closure")
    ax[0, 2].set_title("Storage tendency components")
    for col in range(3):
        ax[-1, col].set_xlabel("Time since analysis start (days)")
        ax[0, col].legend(loc="best", fontsize=7)

    figsave.save(fig, "atmospheric_closure_timeseries", number=1)
    print("Done with figure: global_atmospheric_closure_timeseries")


def plot_cumulative(entries, figsave):
    if not should_run_figure(2):
        return

    fig, ax = plt.subplots(
        len(entries), 3, figsize=(13, 3.2 * len(entries)),
        constrained_layout=True, sharex="col"
    )
    if len(entries) == 1:
        ax = np.array([ax])

    closure_terms = ["atmospheric_forcing", "atmospheric_storage_tendency", "residual_full"]
    residual_terms = ["residual_no_storage", "residual_dry_latent_only", "residual_full"]
    surface_terms = ["net_toa", "fixed_sst_surface_uptake", "atmospheric_forcing"]
    styles = {
        "atmospheric_forcing": dict(color="tab:blue", lw=1.4, ls="-"),
        "atmospheric_storage_tendency": dict(color="tab:green", lw=1.4, ls="-"),
        "residual_full": dict(color="tab:red", lw=1.5, ls="-"),
        "residual_dry_latent_only": dict(color="tab:brown", lw=1.2, ls="--"),
        "residual_no_storage": dict(color="0.45", lw=1.1, ls=":"),
        "net_toa": dict(color="0.15", lw=1.3, ls="-"),
        "fixed_sst_surface_uptake": dict(color="tab:orange", lw=1.3, ls="-"),
    }

    for row, entry in enumerate(entries):
        sim_name = entry["simulation"]
        series = entry["series"]
        origin = entry["origin"]
        annotate_panel_label(ax[row, 0], sim_name)

        for col, terms in enumerate((closure_terms, residual_terms, surface_terms)):
            for term in terms:
                if term not in series:
                    continue
                ts = series[term]
                t_days, cumulative = cumulative_energy_mj_m2(ts, origin=origin)
                ax[row, col].plot(
                    t_days, cumulative,
                    label=TERM_LABELS.get(term, term),
                    **styles.get(term, {}),
                )
            ax[row, col].axhline(0.0, color="0.35", lw=0.7)
            if col == 0:
                ax[row, col].set_ylabel(r"MJ m$^{-2}$")

    ax[0, 0].set_title("Cumulative atmospheric closure")
    ax[0, 1].set_title("Residual after adding storage terms")
    ax[0, 2].set_title("Cumulative system and fixed-SST uptake")
    for col in range(3):
        ax[-1, col].set_xlabel("Time since analysis start (days)")
        ax[0, col].legend(loc="best", fontsize=7)

    figsave.save(fig, "atmospheric_closure_cumulative", number=2)
    print("Done with figure: global_atmospheric_closure_cumulative")


def plot_decomposition(entries, figsave):
    if not should_run_figure(3):
        return

    summary_rows = []
    for entry in entries:
        sim_name = entry["simulation"]
        origin = entry["origin"]
        for term in (
            "atmospheric_radiation",
            "surface_sensible",
            "surface_latent",
            "dry_static_storage_tendency",
            "latent_storage_tendency",
            "kinetic_storage_tendency",
            "residual_full",
            "fixed_sst_surface_uptake",
            "precip_latent_release",
        ):
            if term in entry["series"]:
                summary_rows.append(term_summary(sim_name, term, entry["series"][term], origin))

    df = pd.DataFrame(summary_rows)
    term_order = [
        "atmospheric_radiation",
        "surface_sensible",
        "surface_latent",
        "dry_static_storage_tendency",
        "latent_storage_tendency",
        "kinetic_storage_tendency",
        "residual_full",
        "fixed_sst_surface_uptake",
        "precip_latent_release",
    ]
    colors = {
        "atmospheric_radiation": "tab:purple",
        "surface_sensible": "tab:red",
        "surface_latent": "tab:blue",
        "dry_static_storage_tendency": "tab:green",
        "latent_storage_tendency": "tab:cyan",
        "kinetic_storage_tendency": "tab:brown",
        "residual_full": "0.2",
        "fixed_sst_surface_uptake": "tab:orange",
        "precip_latent_release": "0.55",
    }

    fig, ax = plt.subplots(1, 1, figsize=(11, 4.8), constrained_layout=True)
    x = np.arange(len(entries))
    width = 0.08
    offsets = (np.arange(len(term_order)) - 0.5 * (len(term_order) - 1)) * width

    for j, term in enumerate(term_order):
        sub = df[df["term"] == term].set_index("simulation")
        vals = [sub.loc[entry["simulation"], "mean_w_m2"] if entry["simulation"] in sub.index else np.nan for entry in entries]
        ax.bar(
            x + offsets[j], vals, width=width,
            color=colors.get(term, "0.5"),
            label=TERM_LABELS.get(term, term),
        )

    ax.axhline(0.0, color="0.25", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([entry["simulation"] for entry in entries])
    ax.set_ylabel(r"Mean contribution (W m$^{-2}$)")
    ax.set_title("Atmospheric closure decomposition; precip latent release is internal")
    ax.legend(loc="best", fontsize=7, ncol=3)

    figsave.save(fig, "atmospheric_budget_decomposition", number=3)
    print("Done with figure: global_atmospheric_budget_decomposition")


def energy_cache_path(simulation):
    safe = simulation.replace(" ", "_")
    return os.path.join(ENERGY_CACHE_ROOT, f"{safe}_energy_budget.nc")


def energy_cache_manifest(
    data_loc, simulation, *, include_sources=True, test_times=None
):
    source_paths = (
        glob.glob(os.path.join(root_dir, data_loc, LATLON_PATTERN))
        if include_sources else None
    )
    return cache_manifest(
        "mean_state.energy_budget",
        schema=ENERGY_CACHE_SCHEMA,
        configuration={
            "simulation": simulation,
            "analysis_start": ANALYSIS_START,
            "analysis_end": ANALYSIS_END,
            "test_times": test_times,
            "variables": needed_vars,
            "constants": {
                "rd": RD,
                "cp": CP,
                "gravity": G,
                "latent_heat": LV,
            },
            "density_formulation": (
                "rho_m = rho_d * (1 + qv + qc + qr + qi + qs + qg)"
            ),
            "storage_mass_basis": "total moist air",
        },
        source_paths=source_paths,
    )


def energy_cache_is_current(path, expected_manifest, *, validate_sources=True):
    if not os.path.isfile(path):
        return False
    try:
        with xr.open_dataset(path) as cached:
            if cached.attrs.get("cache_schema") != ENERGY_CACHE_SCHEMA:
                return False
            if validate_sources:
                return (
                    cached.attrs.get("manifest_sha256")
                    == expected_manifest["manifest_sha256"]
                )
            saved = json.loads(cached.attrs.get("cache_manifest", "null"))
            return cache_manifest_matches(
                saved, expected_manifest, compare_sources=False
            )
    except (OSError, ValueError):
        return False


def write_energy_cache(path, series, storage_energy, manifest):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    dataset = xr.Dataset({**series, **dict(storage_energy.data_vars)})
    dataset.attrs.update(
        cache_schema=ENERGY_CACHE_SCHEMA,
        manifest_sha256=manifest["manifest_sha256"],
        cache_manifest=json.dumps(manifest, sort_keys=True),
    )
    temporary = path + ".tmp"
    netcdf_safe_dataset(dataset).to_netcdf(temporary)
    os.replace(temporary, path)


def read_energy_cache(path):
    with xr.open_dataset(path) as cached:
        dataset = cached.load()
    series = {name: dataset[name] for name in SERIES_NAMES if name in dataset}
    storage_energy = dataset[list(STORAGE_NAMES)]
    return series, storage_energy


def prepare_energy_entry(
    data_loc,
    simulation,
    *,
    figures_only=False,
    rebuild_cache=False,
    test_times=None,
):
    path = energy_cache_path(simulation)
    manifest = energy_cache_manifest(
        data_loc,
        simulation,
        include_sources=not figures_only,
        test_times=test_times,
    )
    current = energy_cache_is_current(
        path, manifest, validate_sources=not figures_only
    )
    if current and not rebuild_cache:
        print(f"{simulation}: using energy-budget cache {path}")
        series, storage_energy = read_energy_cache(path)
    elif figures_only:
        raise FileNotFoundError(
            "Figure-only mode requires a current energy-budget cache: "
            f"{path}"
        )
    else:
        ds = load_latlon_dataset(
            os.path.join(root_dir, data_loc),
            latlon_pattern=LATLON_PATTERN,
            chunks=latlon_chunks,
        )
        try:
            ds = standardize_latlon_dataset(ds, needed_vars)
            if test_times is not None:
                ds = ds.isel(Time=slice(0, int(test_times)))
            series, storage_energy = derive_budget_series(ds)
            write_energy_cache(path, series, storage_energy, manifest)
            series, storage_energy = read_energy_cache(path)
        finally:
            ds.close()
    origin = storage_energy.Time.values[0]
    return {
        "simulation": simulation,
        "series": series,
        "storage_energy": storage_energy,
        "origin": origin,
    }


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use bounded actual-data subsets and isolated test products.",
    )
    parser.add_argument("--test-times", type=int, default=8)
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=(1, 2, 3),
        default=None,
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help="Build figures only from current energy-budget caches.",
    )
    cache_mode.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild all energy-budget caches before plotting.",
    )
    args = parser.parse_args(argv)
    if args.test_times < 3:
        parser.error("--test-times must be at least 3")
    return args


def main(argv=None):
    global ENERGY_CACHE_ROOT, figures_to_run, save_loc, should_run_figure
    args = parse_args(argv)
    if args.test:
        test_label = f"all_{args.test_times:03d}times"
        ENERGY_CACHE_ROOT = diagnostic_cache_dir(
            "energy_budget", "testing", test_label
        )
        save_loc = os.path.join(save_loc, "testing", test_label)
    figures_to_run = args.figures or []
    should_run_figure = partial(figure_should_run, figures_to_run)
    figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)
    entries = [
        prepare_energy_entry(
            data_loc,
            simulation,
            figures_only=args.figures_only,
            rebuild_cache=args.rebuild_cache,
            test_times=args.test_times if args.test else None,
        )
        for data_loc, simulation in zip(data_locs, sim_names)
    ]
    summary_rows = []
    ts_rows = []

    for entry in entries:
        simulation = entry["simulation"]
        origin = entry["origin"]
        series = entry["series"]
        for term, ts in series.items():
            summary_rows.append(term_summary(simulation, term, ts, origin))
            ts_rows.extend(timeseries_rows(simulation, term, ts, origin))

    if not entries:
        print("Skipping global atmospheric energy budget: no simulations could be processed")
        return

    summary_df = pd.DataFrame(summary_rows)
    timeseries_df = pd.DataFrame(ts_rows)
    uncertainty_df = (
        pd.DataFrame()
        if args.test
        else energy_budget_uncertainty(entries)
    )
    if args.test:
        print("Test mode omits formal energy-budget uncertainty intervals")
    if save:
        summary_df.to_csv(
            os.path.join(save_loc, "global_atmospheric_energy_budget_summary.csv"),
            index=False,
        )
        timeseries_df.to_csv(
            os.path.join(save_loc, "global_atmospheric_energy_budget_timeseries.csv"),
            index=False,
        )
        uncertainty_df.to_csv(
            os.path.join(
                save_loc,
                "global_atmospheric_energy_budget_uncertainty.csv",
            ),
            index=False,
        )

    print_budget_summary(summary_df)
    plot_timeseries(entries, figsave)
    plot_cumulative(entries, figsave)
    plot_decomposition(entries, figsave)


if __name__ == "__main__":
    main()
