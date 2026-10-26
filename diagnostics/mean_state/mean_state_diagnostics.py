#!/usr/bin/env python
# coding: utf-8



import argparse
import os, sys
from functools import partial

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from matplotlib import colors
from tropopause import tropopause_profile, tropopause_profile_single
from wavenumber_frequency_functions import (
    genDispersionCurves,
)
from diagnostics.diagnostic_helpers import (
    FigureSaver, N2_latheight, add_period_reference_lines_frequency,
    add_split_difference_colorbars, add_symmetric_colorbar,
    annotate_panel_label, contourf_split_difference,
    control_diff_label, diagnostic_output_dir, difference_cmap,
    figure_should_run,
    figure_should_run_any, get_time_values,
    hemispheric_fold, isobaric_lat_mean,
    lat_height_mesh, lat_mean,
    make_spectrum_cmap, midlevel_height_from_latlon,
    model_lat_field, nice_levels, plot_Tu_latpres, plot_rhw_latpres,
    positive_levels, region_symmetric_levels,
    progress, relative_frequency_from_histogram, safe_nanpercentile,
    set_latheight_labels, set_latpres_axis,
    simulation_colour, split_scales_warranted,
    symmetric_levels,
    tropopause_on_latitudes, tropopause_region_masks,
    vertical_velocity_from_zm,
)
from diagnostics.mean_state.mean_state_cache import (
    configure_test_mode,
    prepare_products,
    split_simulations,
)
from diagnostics.mean_state.publication_profiles import (
    PROFILE_VARIANTS, environment_profile_figure, prepare_profile_inputs,
)
from diagnostics.analysis_config import BOOTSTRAP_SEED, FIELD_BOOTSTRAP_REPLICATES
from diagnostics.uncertainty import independent_bootstrap_field_difference
from diagnostics.analysis_config import (
    SIMULATION_ARCHIVES,
    SIMULATION_LABELS as PROJECT_SIMULATION_LABELS,
    SIMULATION_ROOT,
    ACTIVE_SIMULATION_NAMES,
)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Build mean-state diagnostics.")
    parser.add_argument(
        "--test",
        action="store_true",
        help="Use bounded actual-data subsets and isolated test products.",
    )
    parser.add_argument(
        "--test-times",
        type=int,
        default=100,
        help="Maximum source times used by test mode. The default supports test spectra.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=[*range(1, 15), 16, 17, 18, 19, 20, 21],
        default=None,
        help="Build only these numbered figures.",
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help="Build figures only from current compact mean-state caches.",
    )
    cache_mode.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild selected compact mean-state caches before plotting.",
    )
    args = parser.parse_args(argv)
    if args.test_times < 2:
        parser.error("--test-times must be at least 2")
    return args


runtime_args = parse_args()
progress("Starting mean-state diagnostics")
save = True
save_loc = diagnostic_output_dir('/home/cwp29/tmp/mpas/crf_on/final/mean_state/')
if runtime_args.test:
    configure_test_mode(runtime_args.test_times)
    save_loc = os.path.join(
        save_loc,
        "testing",
        f"all_{runtime_args.test_times:03d}times",
    )

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1:  rain_stats
#   2:  profiles
#   3:  temp_relhum
#   4:  eddy_fluxes
#   5:  pr_dist
#   6:  duration_dist
#   7:  spectra
#   8:  hovmoller
#   9:  zgrid_horizontal_variation
#   10: cldfrac_latheight
#   11: vorticity_latheight
#   12: olrtoa_maps
#   13: ke_latheight
#   14: eke_latpres
#   15: moved to energy_budget.py
#   16: temperature_wind_latheight, w_rh_latheight, tem_latheight
#   17: mslp_maps
#   18: scalar_constituents_latheight
#   19: precipitable-water and CAPE distributions and responses
#   20: static-stability latitude-height fields and responses
#   21: cold-point tropopause height and pressure profiles
#
# WHAT TO LOOK FOR:
#   * Use mean-state figures to establish the baseline structural response:
#     coherent latitude-height changes in T, RH, w, cloud fraction, and
#     hydrometeors are more meaningful than isolated map features.
#   * Compare precipitation PDFs, spectra, and Hovmoller plots before treating
#     small mean precipitation shifts as dynamically important.
#   * Treat cloud and OLR changes together with temperature/static-stability
#     and vertical-motion changes; high cloud alone is mechanistically
#     ambiguous.
figures_to_run = runtime_args.figures or []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)


def save_figure(fig, stem, number=None, dpi=None):
    """Save and release a figure so long runs do not retain its artists."""
    try:
        return figsave.save(fig, stem, number=number, dpi=dpi)
    finally:
        plt.close(fig)


should_run_figure = partial(figure_should_run, figures_to_run)
should_run_any = partial(figure_should_run_any, figures_to_run)


spectrum_period_reference_days = [3.0, 6.0, 30.0]


add_period_reference_lines_frequency = partial(
    add_period_reference_lines_frequency,
    periods=spectrum_period_reference_days,
)


RD = 287.0
CP = 1004.0
P0_PA = 100000.0
G = 9.80665
def isobaric_eddy_kinetic_energy(ds):
    u = ds.uzonal_isobaric
    v = ds.umeridional_isobaric
    eddy_dims = [dim for dim in ("Time", "longitude") if dim in u.dims]
    ua = u - u.mean(dim=eddy_dims, skipna=True)
    va = v - v.mean(dim=eddy_dims, skipna=True)
    eke = 0.5 * (ua * ua + va * va).mean(dim=eddy_dims, skipna=True)
    eke.name = "eke_isobaric"
    eke.attrs.update(units="m2 s-2", long_name="isobaric eddy kinetic energy")
    return hemispheric_fold(eke.transpose("pres", "latitude"), parity="even")


def positive_contour_levels(fields, n=21, fallback=1.0):
    values = np.concatenate(
        [np.asarray(field.values, dtype=float).ravel() for field in fields]
    )
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("Cannot plot non-finite positive fields")
    vmax = np.max(values)
    if vmax < 0.0:
        raise ValueError("Positive-field plot contains only negative values")
    if vmax == 0.0:
        vmax = fallback
    return positive_levels(vmax, n=n)


def symmetric_contour_levels(fields, ndiv=11, fallback=1.0):
    values = np.concatenate(
        [np.asarray(field.values, dtype=float).ravel() for field in fields]
    )
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("Cannot plot non-finite difference fields")
    vmax = np.max(np.abs(values))
    if vmax == 0.0:
        vmax = fallback
    return symmetric_levels(vmax, ndiv=ndiv)


def latlon_time_mean_field(ds, variable, scale=1.0, name=None, hemispheric_parity=None):
    da = ds[variable].mean(dim="Time", skipna=True)
    da = (da * scale).transpose("latitude", "longitude")
    da = hemispheric_fold(da, parity=hemispheric_parity).load()
    da.name = name or variable
    return da


def latheight_time_mean_field(zds, spec):
    if spec.get("derived") == "temperature":
        da = zds.theta * (zds.pressure / P0_PA) ** (RD / CP)
    elif spec.get("derived") == "vertical_velocity":
        da = vertical_velocity_from_zm(zds)
    else:
        da = zds[spec["var"]]

    if "Time" in da.dims:
        da = da.mean(dim="Time", skipna=True)

    da = model_lat_field(da) * spec.get("scale", 1.0)
    da = hemispheric_fold(da, parity=spec.get("hemispheric_parity", "even")).load()
    da.name = spec["label"]
    return da


def ensure_z_index_coord(da):
    if "z" in da.dims and "z" not in da.coords:
        da = da.assign_coords(z=np.arange(da.sizes["z"]))
    return da


def scalar_constituent_latheight(ds, spec, zmid=None):
    varnames = spec["vars"] if "vars" in spec else (spec["var"],)
    missing = [varname for varname in varnames if varname not in ds.data_vars]
    if missing:
        raise ValueError(f"{missing} not found")

    da = None
    for varname in varnames:
        term = ds[varname]
        da = term if da is None else da + term
    mean_dims = [dim for dim in ("Time", "longitude") if dim in da.dims]
    if mean_dims:
        da = da.mean(dim=mean_dims, skipna=True)

    da = model_lat_field(da) * spec.get("scale", 1.0)
    da = hemispheric_fold(da, parity=spec.get("hemispheric_parity", "even"))
    da = ensure_z_index_coord(da).load()
    da.name = spec["label"]
    if zmid is None:
        zmid = midlevel_height_from_latlon(ds)
    return lat_height_mesh(da, zmid, ds.latitude.values)


def positive_percentile_levels(fields, percentile=99.5, n=21, fallback=1.0):
    vmax = np.nanmax([
        safe_nanpercentile(
            field.values, percentile, zero_value=fallback
        )
        for field in fields
    ])
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback
    return positive_levels(vmax, n=n)


def positive_log_levels(fields, percentile=99.5, n=21, vmin=1.0e-4, fallback=1.0):
    finite = np.concatenate([
        np.asarray(field.values, dtype=float).ravel()
        for field in fields
    ])
    finite = finite[np.isfinite(finite)]
    if not finite.size:
        raise ValueError("Cannot plot a non-finite logarithmic field")
    vals = finite[finite > 0.0]
    if not vals.size:
        if np.any(finite < 0.0):
            raise ValueError(
                "Logarithmic positive-field plot contains only non-positive "
                "values"
            )
        vmax = fallback
    else:
        vmax = np.nanpercentile(vals, percentile)
    if not np.isfinite(vmax):
        raise ValueError("Calculated logarithmic contour scale is non-finite")
    if vmax <= vmin:
        vmax = vmin * 10.0
    return np.geomspace(vmin, vmax, n)


def scalar_raw_levels_and_norm(fields, spec):
    raw_scale = spec.get("raw_scale", "linear")
    if raw_scale == "log":
        levels = positive_log_levels(
            fields,
            percentile=99.5,
            vmin=spec.get("raw_log_min", 1.0e-4),
            fallback=spec.get("raw_fallback", 1.0),
        )
        return levels, colors.LogNorm(vmin=levels[0], vmax=levels[-1]), "both"
    if raw_scale != "linear":
        raise ValueError(f"Unknown raw scale {raw_scale!r} for {spec['label']}")

    levels = positive_percentile_levels(
        fields,
        percentile=99.5,
        fallback=spec.get("raw_fallback", 1.0),
    )
    return levels, None, "max"


def scalar_raw_values(field, spec):
    values = np.asarray(field.values, dtype=float)
    if spec.get("raw_scale") == "log":
        return np.ma.masked_less_equal(np.ma.masked_invalid(values), 0.0)
    return values


def symmetric_percentile_levels(fields, percentile=99.0, ndiv=10, fallback=1.0):
    vmax = np.nanmax([
        safe_nanpercentile(
            np.abs(field.values), percentile, zero_value=fallback
        )
        for field in fields
    ])
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = fallback
    return symmetric_levels(vmax, ndiv=ndiv)


def field_for_level_limits(field, spec):
    min_height = spec.get("level_min_height_m")
    if min_height is None or "z" not in field.coords:
        return field

    limited = field.where(field.z >= min_height, drop=True)
    if limited.sizes.get("z", 0) == 0:
        raise ValueError(
            f"{spec['label']}: no vertical levels satisfy minimum height "
            f"{min_height} m"
        )
    return limited


def fields_for_level_limits(fields, spec):
    return [field_for_level_limits(field, spec) for field in fields]


def tropopause_split_diff_context(diff_entries, spec, percentile=None, ndiv=10):
    """Masks and per-region levels for tropopause-split difference panels.

    ``diff_entries`` maps sim index -> (values2d, z2d_m, latitudes), with the
    vertical mesh in metres. Returns (masks, tropo_levels, strat_levels,
    split) where ``masks`` maps sim index -> (tropo_mask, strat_mask) using
    that simulation's cold-point tropopause, and ``split`` says whether the
    two regions differ enough in scale to warrant separate colorbars.
    """
    masks = {}
    for i, entry in diff_entries.items():
        values, z2d, lats = entry[0], entry[1], entry[2]
        cpt = tropopause_on_latitudes(CPTs[i][0], dss_zm[i].latitude.values, lats)
        masks[i] = tropopause_region_masks(z2d, cpt)

    min_height = spec.get("level_min_height_m")

    def level_mask(mask, z2d):
        return mask & (z2d >= min_height) if min_height is not None else mask

    fallback = spec.get("diff_fallback", 1.0)
    tropo_levels = region_symmetric_levels(
        [(entry[0], level_mask(masks[i][0], entry[1])) for i, entry in diff_entries.items()],
        ndiv=ndiv, zero_field_scale=fallback, percentile=percentile,
    )
    strat_levels = region_symmetric_levels(
        [(entry[0], level_mask(masks[i][1], entry[1])) for i, entry in diff_entries.items()],
        ndiv=ndiv, zero_field_scale=fallback, percentile=percentile,
    )
    split = (
        spec.get("diff_split_tropopause", False)
        and bool(diff_entries)
        and split_scales_warranted(tropo_levels, strat_levels)
    )
    return masks, tropo_levels, strat_levels, split


def latlon_contour_levels(fields, symmetric=False, n=21, fallback=1.0):
    if symmetric:
        return symmetric_contour_levels(fields, ndiv=n-1, fallback=fallback)

    values = np.concatenate(
        [np.asarray(field.values, dtype=float).ravel() for field in fields]
    )
    values = values[np.isfinite(values)]
    if not values.size:
        raise ValueError("Cannot plot a non-finite latitude-longitude field")
    vmin = np.min(values)
    vmax = np.max(values)
    if np.isclose(vmin, vmax):
        half_width = abs(vmin) * 0.01 if vmin else fallback
        vmin -= half_width
        vmax += half_width

    return nice_levels(vmin, vmax, n=n)


LATHEIGHT_MEAN_STATE_SPECS = [
    {
        "label": "Temperature",
        "derived": "temperature",
        "units": "K",
        "cmap": "coolwarm",
        "raw_symmetric": False,
        "raw_fallback": 300.0,
        "diff_fallback": 2.0,
        "diff_split_tropopause": True,
        "hemispheric_parity": "even",
    },
    {
        "label": "Zonal wind",
        "var": "uReconstructZonal",
        "units": r"m s$^{-1}$",
        "cmap": "seismic",
        "raw_symmetric": True,
        "raw_fallback": 10.0,
        "diff_fallback": 2.0,
        "diff_split_tropopause": True,
        "hemispheric_parity": "even",
    },
    {
        "label": "Meridional wind",
        "var": "uReconstructMeridional",
        "units": r"m s$^{-1}$",
        "cmap": "seismic",
        "raw_symmetric": True,
        "raw_fallback": 10.0,
        "diff_fallback": 2.0,
        "diff_split_tropopause": True,
        "hemispheric_parity": "odd",
    },
]

LATHEIGHT_W_RH_SPECS = [
    {
        "label": "Vertical velocity",
        "derived": "vertical_velocity",
        "units": r"m s$^{-1}$",
        "cmap": "seismic",
        "raw_symmetric": True,
        "raw_fallback": 0.05,
        "diff_fallback": 0.01,
        "diff_split_tropopause": True,
        "hemispheric_parity": "even",
    },
    {
        "label": "Relative humidity",
        "var": "relhum",
        "units": "%",
        "cmap": "BrBG",
        "raw_symmetric": False,
        "raw_fallback": 100.0,
        "diff_fallback": 10.0,
        "diff_split_tropopause": True,
        "hemispheric_parity": "even",
    },
]

MSLP_LATLON_SPEC = {
    "label": "MSLP",
    "var": "mslp",
    "units": "hPa",
    "scale": 0.01,
    "cmap": "viridis",
    "raw_symmetric": False,
    "raw_fallback": 1010.0,
    "diff_fallback": 2.0,
    "hemispheric_parity": "even",
}

SCALAR_CONSTITUENT_GROUPS = [
    {
        "name": "vapour",
        "title": "Water vapour",
        "fields": [
            {
                "var": "qv",
                "label": "Water vapour",
                "symbol": r"$q_v$",
                "units": r"g kg$^{-1}$",
                "scale": 1000.0,
                "cmap": "YlGnBu",
                "raw_scale": "log",
                "raw_log_min": 1.0e-4,
                "raw_fallback": 20.0,
                "diff_fallback": 1.0,
                "diff_split_tropopause": True,
                "hemispheric_parity": "even",
            },
        ],
    },
    {
        "name": "cloud_hydrometeors",
        "title": "Cloud hydrometeors",
        "fields": [
            {
                "vars": ("qi", "qc"),
                "label": "Cloud condensate",
                "symbol": r"$q_i + q_c$",
                "units": r"g kg$^{-1}$",
                "scale": 1000.0,
                "cmap": "YlGnBu",
                "raw_fallback": 0.1,
                "diff_fallback": 0.02,
                "hemispheric_parity": "even",
            },
        ],
    },
    {
        "name": "precipitating_hydrometeors",
        "title": "Precipitating hydrometeors",
        "fields": [
            {
                "vars": ("qg", "qs", "qr"),
                "label": "Precipitating condensate",
                "symbol": r"$q_g + q_s + q_r$",
                "units": r"g kg$^{-1}$",
                "scale": 1000.0,
                "cmap": "PuBu",
                "raw_fallback": 0.05,
                "diff_fallback": 0.01,
                "hemispheric_parity": "even",
            },
        ],
    },
]

root_dir = str(SIMULATION_ROOT)
data_locs = list(SIMULATION_ARCHIVES.values())
sim_names = list(PROJECT_SIMULATION_LABELS.values())

cols = [simulation_colour(name) for name in sim_names]



latlon_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw",
    "qv", "qc", "qr", "qi", "qs", "qg", "relhum", "w",
    "mslp",
    "uzonal_isobaric", "umeridional_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric", "theta",
    "uReconstructZonal", "uReconstructMeridional",
    "cldfrac", "olrtoa", "zgrid", "pressure", "vorticity", "ke", "cape",
]
zonal_mean_vars = [
    "qv_isobaric", "relhum", "rh_isobaric", "t_isobaric", "theta",
    "uReconstructZonal", "uReconstructMeridional", "uzonal_isobaric",
    "umeridional_isobaric", "omg_isobaric", "cldfrac", "pressure",
    "vorticity", "ke", "w",
]

# Match the native 4-D isobaric chunks spatially. The previous 60x120 chunks
# split every 180x360 HDF5 chunk into nine Dask tasks and made figure 4 spend
# hours repeatedly traversing an unnecessarily large graph.
latlon_chunks = {"Time": 4, "latitude": 180, "longitude": 360}
zm_chunks = {"Time": -1}

progress("Preparing compact mean-state calculation products")
cached_products = prepare_products(
    root_dir,
    data_locs,
    sim_names,
    ([number for number in figures_to_run if number != 2] or [0]) if figures_to_run else [],
    latlon_vars,
    zonal_mean_vars,
    latlon_chunks,
    zm_chunks,
    figures_only=runtime_args.figures_only,
    rebuild=runtime_args.rebuild_cache,
)
progress("Mean-state calculation products are ready")

if "means" in cached_products:
    means_latlon, means_zonal = cached_products["means"]
    dss = split_simulations(means_latlon, sim_names)
    dss_zm = split_simulations(means_zonal, sim_names)
else:
    dss = [xr.Dataset() for _ in sim_names]
    dss_zm = [xr.Dataset() for _ in sim_names]

rain_cache = cached_products.get("rain")
distribution_cache = cached_products.get("distributions")
spectrum_cache = cached_products.get("spectra")
eddy_cache = cached_products.get("eddy")

if should_run_figure(1):
    # rainc, rainnc contain accumulated rainfall from simulation start to given timestep
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.5), constrained_layout=True)

    for i, simulation in enumerate(sim_names):
        product = rain_cache.sel(simulation=simulation)
        precip_water = product.precip_water

        # Use this dataset's own time coordinate
        t_pw = get_time_values(precip_water)

        precip_rate = product.precip_rate_global

        t_rate = get_time_values(precip_rate)

        ax[0].plot(t_pw, precip_water.values, color=cols[i], label=sim_names[i])
        ax[1].plot(t_rate, precip_rate.values, color=cols[i])

    ax[0].set_ylim(18, 23)
    ax[0].set_ylabel(r"Precipitable water (mm)")

    ax[1].set_ylim(2, 4)
    ax[1].set_ylabel(r"Rainfall rate (mm day$^{-1}$)")

    ax[1].set_xlabel('Time (days)')

    fig.legend(loc='outside lower center', ncol=4)

    save_figure(fig, "rain_stats", number=1)
    print("Done with figure: comp_rain_stats")


if should_run_figure(2):
    profile_products, _profile_sources = prepare_profile_inputs(
        ACTIVE_SIMULATION_NAMES, figures_only=runtime_args.figures_only,
        rebuild=runtime_args.rebuild_cache,
        time_count=min(runtime_args.test_times, 40) if runtime_args.test else None,
    )
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8.5, "axes.labelsize": 8,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "pdf.fonttype": 42}):
        for suffix, include_lpo, lower, caption in PROFILE_VARIANTS:
            if runtime_args.test and include_lpo:
                continue
            fig = environment_profile_figure(profile_products, include_lpo=include_lpo, lower=lower)
            if runtime_args.test:
                fig.suptitle("TEST: bounded profile inputs; intervals are not publication estimates")
                caption = "Bounded test rendering only; short blocks do not establish production uncertainty.\n"
            path = figsave.save(fig, "profiles" + suffix, number=2)
            fig.savefig(os.path.splitext(path)[0] + ".pdf")
            with open(os.path.splitext(path)[0] + "_caption.txt", "w", encoding="utf-8") as stream:
                stream.write(caption)
            plt.close(fig)
    print("Done with figure: comp_profiles")


# pre-compute CPTs for plotting

CPTs = None
if should_run_any([3, 10, 11, 13, 14, 16, 18, 20, 21]):
    CPTs = []

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        theta = hemispheric_fold(zds.theta.mean(dim="Time"), parity="even")
        pressure = hemispheric_fold(zds.pressure.mean(dim="Time"), parity="even")

        T = theta * (pressure / P0_PA) ** (RD / CP)
        T = model_lat_field(T)
        pressure = model_lat_field(pressure)
        z = zds.z.values

        ztps = []
        ptps = []
        for j in range(len(zds.latitude.values)):
            ttp, ptp, ztp, _ = tropopause_profile(T.values[:, j], pressure.values[:, j], z)
            ztps.append(ztp)
            ptps.append(ptp)

        CPTs.append([ztps, ptps])
        progress(
            f"Calculated cold-point tropopause profiles for {sim_names[i]} "
            f"at {len(ztps)} latitudes"
        )

    CPTs = np.array(CPTs)



if should_run_figure(3):
    fig, ax = plt.subplots(len(dss), 2, figsize=(8, 3.0*len(dss)), sharey=True, constrained_layout=True)

    if len(dss) == 1:
        ax = np.array([ax])

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        ubar_da = isobaric_lat_mean(zds.uzonal_isobaric, hemispheric_parity="even")
        vbar_da = isobaric_lat_mean(zds.umeridional_isobaric, hemispheric_parity="odd")
        omega_bar_da = isobaric_lat_mean(
            zds.omg_isobaric, hemispheric_parity="even"
        ) / 100.0
        omega_bar_da.attrs.update(
            units="hPa s-1", source_variable="omg_isobaric"
        )
        Tbar_da = isobaric_lat_mean(zds.t_isobaric, hemispheric_parity="even")
        rhbar_da = isobaric_lat_mean(zds.rh_isobaric, hemispheric_parity="even")

        ubar = ubar_da.values
        vbar = vbar_da.values
        omega_bar = omega_bar_da.values
        Tbar = Tbar_da.values
        rhbar = rhbar_da.values

        T_im, _, _, _ = plot_Tu_latpres(ax[i, 0], zds.latitude.values, zds.pres.values, Tbar, ubar)
        rh_im, _ = plot_rhw_latpres(
            ax[i, 1], zds.latitude.values, zds.pres.values,
            rhbar, vbar, omega_bar,
        )

        annotate_panel_label(ax[i, 0], sim_names[i])
        annotate_panel_label(ax[i, 1], sim_names[i])

        ax[i, 0].plot(zds.latitude.values, CPTs[i][1]/100., color='k', lw=1, ls=':')
        ax[i, 1].plot(zds.latitude.values, CPTs[i][1]/100., color='k', lw=1, ls=':')

    for j in range(ax.shape[0]-1):
        ax[j, 0].tick_params(labelbottom=False)
        ax[j, 1].tick_params(labelbottom=False)

    ax[-1, 0].set_xlabel("Latitude (degrees)")
    ax[-1, 1].set_xlabel("Latitude (degrees)")

    for j in range(ax.shape[0]):
        ax[j, 0].set_ylabel("Pressure (hPa)")

    for a in ax.ravel():
        set_latpres_axis(a, xlabel=False, ylabel=False, xlim=(-90, 90))

    fig.colorbar(T_im, ax=ax[-1, 0], location='bottom', label="temperature (K)")
    fig.colorbar(rh_im, ax=ax[-1, 1], location='bottom', label="relative humidity (%)")

    save_figure(fig, "temp_relhum", number=3)
    print("Done with figure: comp_temp_relhum")


if should_run_figure(4):
    # Meridional eddy fluxes

    fig, ax = plt.subplots(4, 4, figsize=(8, 8), constrained_layout='True')

    for i, simulation in enumerate(sim_names):
        product = eddy_cache.sel(simulation=simulation)
        eke = product.eke
        memf = product.momentum_flux
        mehf = product.heat_flux
        mewvf = product.vapour_flux
        eke = hemispheric_fold(eke, parity="even")
        memf = hemispheric_fold(memf, parity="odd")
        mehf = hemispheric_fold(mehf, parity="odd")
        mewvf = hemispheric_fold(mewvf, parity="odd")

        for j, var in enumerate([eke, memf, mehf, mewvf]):
            if j == 0:
                ax[j, 0].plot(var.latitude.values, var[0], color=cols[i], label=sim_names[i])
            else:
                ax[j, 0].plot(var.latitude.values, var[0], color=cols[i])
            ax[j, 1].plot(var.latitude.values, var[1], color=cols[i])
            ax[j, 2].plot(var.latitude.values, var[2], color=cols[i])
            ax[j, 3].plot(var.latitude.values, var[3], color=cols[i])

    for a in ax.ravel():
        a.set_xlim(-90, 90)

    ax[0, 0].set_title("100 hPa")
    ax[0, 1].set_title("250 hPa")
    ax[0, 2].set_title("700 hPa")
    ax[0, 3].set_title("850 hPa")

    ax[0, 0].set_ylabel("Eddy kinetic \n energy (m$^2$ s$^{-2}$)")
    ax[1, 0].set_ylabel("Meridional eddy \n momentum flux (m$^2$ s$^{-2}$)")
    ax[2, 0].set_ylabel("Meridional heat \n flux (K m s$^{-1}$)")
    ax[3, 0].set_ylabel("Meridional wv \n flux (g kg$^{-1}$ m s$^{-1}$)")

    fig.legend(loc='outside lower center', ncol=4)

    save_figure(fig, "eddy_fluxes", number=4)
    print("Done with figure: comp_eddy_fluxes")


DISTRIBUTION_DEEP_TROPICAL_LAT_BOUNDS = (-10.0, 10.0)
DISTRIBUTION_DOMAIN_STYLES = (("Global", "-"), ("10S-10N", "--"))


if should_run_figure(5):
    # Normalised rainfall intensity distribution
    fig, ax = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True, sharey=True)

    rain_edges = distribution_cache.rain_bin_edges.isel(simulation=0).values
    imerg_histograms = distribution_cache.imerg_rain_histogram
    for domain_index, (domain, linestyle) in enumerate(DISTRIBUTION_DOMAIN_STYLES):
        counts = imerg_histograms.isel(domain=domain_index).values
        edges = rain_edges
        x, h = relative_frequency_from_histogram(counts, edges, smooth=10)
        ax[0].plot(x, h, color='k', ls=linestyle, label=f'IMERG (Obs), {domain}')
    ax[0].set_yscale('log')

    for s, simulation in enumerate(sim_names):
        product = distribution_cache.sel(simulation=simulation)
        for i, component in enumerate(["total", "convective", "resolved"]):
            histograms = product.rain_histogram.sel(component=component)
            for domain_index, (domain, linestyle) in enumerate(DISTRIBUTION_DOMAIN_STYLES):
                counts = histograms.isel(domain=domain_index).values
                edges = rain_edges
                x, h = relative_frequency_from_histogram(counts, edges, smooth=10)
                label = f"{sim_names[s]}, {domain}" if i == 0 else None
                ax[i].plot(x, h, color=cols[s], ls=linestyle, label=label)
            ax[i].set_yscale('log')

    fig.legend(loc='outside lower center', ncol=4)

    for a in ax:
        a.set_xlim(0, 300)
        a.set_ylim(1e-7, 1)
        a.set_xlabel("Rainfall rate (mm/day)")

    ax[0].set_title("Total")
    ax[0].set_ylabel("Normalised frequency")
    ax[1].set_title("Parameterised")
    ax[2].set_title("Resolved")

    save_figure(fig, "pr_dist", number=5)
    print("Done with figure: comp_pr_dist")


if should_run_figure(6):
    fig, ax = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True, sharey=True)

    duration_edges = distribution_cache.duration_bin_edges.isel(simulation=0).values
    imerg_histograms = distribution_cache.imerg_duration_histogram
    for domain_index, (domain, linestyle) in enumerate(DISTRIBUTION_DOMAIN_STYLES):
        counts = imerg_histograms.isel(domain=domain_index).values
        edges = duration_edges
        x, h = relative_frequency_from_histogram(counts, edges, smooth=5)
        ax[0].plot(x, h, color='k', ls=linestyle, label=f'IMERG (Obs), {domain}')
    ax[0].set_yscale('log')

    for s, simulation in enumerate(sim_names):
        product = distribution_cache.sel(simulation=simulation)
        for i, component in enumerate(["total", "convective", "resolved"]):
            histograms = product.duration_histogram.sel(component=component)
            for domain_index, (domain, linestyle) in enumerate(DISTRIBUTION_DOMAIN_STYLES):
                counts = histograms.isel(domain=domain_index).values
                edges = duration_edges
                x, h = relative_frequency_from_histogram(counts, edges, smooth=5)
                label = f"{sim_names[s]}, {domain}" if i == 0 else None
                ax[i].plot(x, h, color=cols[s], ls=linestyle, label=label)
            ax[i].set_yscale('log')

    for a in ax:
        a.set_xlim(1, 60)
        a.set_xlabel("Duration (days)")

    fig.legend(loc='outside lower center', ncol=4)

    ax[0].set_title("Total")
    ax[0].set_ylabel("Normalised frequency")
    ax[1].set_title("Parameterised")
    ax[2].set_title("Resolved")

    save_figure(fig, "duration_dist", number=6)
    print("Done with figure: comp_duration_dist")


if should_run_figure(19):
    # Plot the absolute distributions as well as their experiment responses.
    # Simulations represent independent statistical climates, so each
    # response bootstrap uses separate block indices.
    fig, axes = plt.subplots(
        2, 2,
        figsize=(9, 6.2),
        constrained_layout=True,
        squeeze=False,
    )
    comparison_specs = [
        (
            index,
            0,
            f"{sim_names[index]} - {sim_names[0]}",
            cols[index],
        )
        for index in range(1, len(sim_names))
    ]
    comparison_specs.extend(
        (
            1,
            index,
            f"{sim_names[1]} - {sim_names[index]}",
            "0.25",
        )
        for index in range(2, len(sim_names))
    )
    field_specs = (
        ("pw", "Precipitable water", "kg m$^{-2}$", (0.0, 80.0)),
        ("cape", "CAPE", "J kg$^{-1}$", (0.0, 5000.0)),
    )
    for field_index, (name, label, units, x_limits) in enumerate(field_specs):
        absolute_axis = axes[0, field_index]
        response_axis = axes[1, field_index]
        centres = distribution_cache[f"{name}_bin"].values
        for simulation_index, simulation_name in enumerate(sim_names):
            block_pdfs = distribution_cache.sel(simulation=simulation_name)[
                f"{name}_pdf_blocks"
            ].values
            absolute_axis.plot(
                centres,
                np.nanmean(block_pdfs, axis=0),
                color=cols[simulation_index],
                linewidth=1.2,
                label=simulation_name,
            )
        for comparison_index, (run_index, reference_index, comparison_label, colour) in enumerate(
            comparison_specs
        ):
            simulation = distribution_cache.sel(simulation=sim_names[run_index])[
                f"{name}_pdf_blocks"
            ].values
            control = distribution_cache.sel(simulation=sim_names[reference_index])[
                f"{name}_pdf_blocks"
            ].values
            if runtime_args.test:
                estimate = np.nanmean(simulation, axis=0) - np.nanmean(
                    control, axis=0
                )
                low = high = np.full_like(estimate, np.nan)
            else:
                estimate, low, high = independent_bootstrap_field_difference(
                    simulation,
                    control,
                    rng=np.random.default_rng(
                        BOOTSTRAP_SEED
                        + 1900
                        + 10 * field_index
                        + comparison_index
                    ),
                    n_resamples=FIELD_BOOTSTRAP_REPLICATES,
                )
            response_axis.plot(
                centres,
                estimate,
                color=colour,
                linewidth=1.2,
                label=comparison_label,
            )
            response_axis.fill_between(
                centres, low, high, color=colour, alpha=0.12
            )
        absolute_axis.set_xlim(*x_limits)
        absolute_axis.set_title(label)
        absolute_axis.set_ylabel("Probability density")
        absolute_axis.grid(alpha=0.18)
        response_axis.axhline(0.0, color="0.6", linewidth=0.8)
        response_axis.set_yscale("symlog", linthresh=1.0e-4)
        response_axis.set_xlim(*x_limits)
        response_axis.set_xlabel(f"{label} ({units})")
        response_axis.set_ylabel("PDF response")
        response_axis.grid(alpha=0.18)
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].legend(frameon=False, fontsize=8)
    save_figure(fig, "pw_cape_distributions_and_responses", number=19)
    print("Done with figure: pw_cape_distributions_and_responses")


if should_run_figure(20):
    n2_fields = [N2_latheight(zds) * 1.0e4 for zds in dss_zm]
    raw_levels = np.linspace(0.0, 8.0, 25)
    response_levels = symmetric_levels(1.0, ndiv=12)
    fig, raw_axes = plt.subplots(
        1,
        len(dss_zm),
        figsize=(3.7 * len(dss_zm), 3.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    fig2, response_axes = plt.subplots(
        1,
        len(dss_zm),
        figsize=(3.7 * len(dss_zm), 3.6),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    raw_axes = raw_axes[0]
    response_axes = response_axes[0]
    raw_image = control_image = response_image = None
    control_n2 = n2_fields[0]
    for index, (zds, n2) in enumerate(zip(dss_zm, n2_fields)):
        field, latitude, height = lat_height_mesh(
            n2, zds.z, zds.latitude.values
        )
        raw_image = raw_axes[index].contourf(
            latitude,
            height / 1000.0,
            field.values,
            levels=raw_levels,
            cmap="YlGnBu",
            extend="max",
        )
        if index == 0:
            control_image = response_axes[index].contourf(
                latitude,
                height / 1000.0,
                field.values,
                levels=raw_levels,
                cmap="YlGnBu",
                extend="max",
            )
        else:
            aligned, reference = xr.align(n2, control_n2, join="exact")
            response = aligned - reference
            response_image = response_axes[index].contourf(
                latitude,
                height / 1000.0,
                response.values,
                levels=response_levels,
                cmap=difference_cmap(response_levels),
                extend="both",
            )
        for axis in (raw_axes[index], response_axes[index]):
            axis.plot(
                zds.latitude.values,
                CPTs[index][0] / 1000.0,
                color="k",
                linewidth=1.0,
                linestyle=":",
            )
            set_latheight_labels(axis, ylabel=index == 0)
        annotate_panel_label(raw_axes[index], sim_names[index])
        annotate_panel_label(
            response_axes[index], control_diff_label(sim_names, index)
        )
    fig.colorbar(
        raw_image,
        ax=raw_axes,
        location="right",
        label=r"$N^2$ ($10^{-4}$ s$^{-2}$)",
    )
    fig2.colorbar(
        control_image,
        ax=response_axes[0],
        location="right",
        label=r"$N^2$ ($10^{-4}$ s$^{-2}$)",
    )
    if response_image is not None:
        add_symmetric_colorbar(
            fig2,
            response_image,
            response_axes[1:],
            response_levels,
            label=r"$\Delta N^2$ ($10^{-4}$ s$^{-2}$)",
        )
    save_figure(fig, "N2_latheight", number=20)
    save_figure(fig2, "N2_latheight_diff", number=20)
    print("Done with figure: N2_latheight")


if should_run_figure(21):
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(9, 6),
        constrained_layout=True,
        sharex=True,
    )
    control_height = CPTs[0][0] / 1000.0
    control_pressure = CPTs[0][1] / 100.0
    for index, zds in enumerate(dss_zm):
        latitude = zds.latitude.values
        height = CPTs[index][0] / 1000.0
        pressure = CPTs[index][1] / 100.0
        axes[0, 0].plot(
            latitude, height, color=cols[index], label=sim_names[index]
        )
        axes[0, 1].plot(latitude, pressure, color=cols[index])
        if index > 0:
            axes[1, 0].plot(
                latitude,
                height - control_height,
                color=cols[index],
                label=f"{sim_names[index]} - {sim_names[0]}",
            )
            axes[1, 1].plot(
                latitude,
                pressure - control_pressure,
                color=cols[index],
            )
    axes[0, 0].set_ylabel("Cold-point height (km)")
    axes[0, 1].set_ylabel("Cold-point pressure (hPa)")
    axes[1, 0].set_ylabel(r"$\Delta$ height (km)")
    axes[1, 1].set_ylabel(r"$\Delta$ pressure (hPa)")
    for axis in axes.ravel():
        axis.set_xlim(-90.0, 90.0)
        axis.axvline(0.0, color="0.7", linewidth=0.5)
    axes[1, 0].axhline(0.0, color="0.7", linewidth=0.5)
    axes[1, 1].axhline(0.0, color="0.7", linewidth=0.5)
    axes[1, 0].set_xlabel("Latitude (degrees)")
    axes[1, 1].set_xlabel("Latitude (degrees)")
    axes[0, 0].legend(frameon=False)
    axes[1, 0].legend(frameon=False)
    axes[0, 1].invert_yaxis()
    save_figure(fig, "cold_point_tropopause_profiles", number=21)
    print("Done with figure: cold_point_tropopause_profiles")


if should_run_figure(7):
    spec_levels = np.linspace(0, 5, 26)
    spec_cmap, spec_norm = make_spectrum_cmap(spec_levels)
    spec_diff_levels = symmetric_levels(2.0, ndiv=11)
    spec_diff_cmap = difference_cmap(spec_diff_levels)
    spec_diff_norm = colors.BoundaryNorm(spec_diff_levels, spec_diff_cmap.N)

    fig, ax = plt.subplots(len(data_locs), 2, figsize=(8, 8), constrained_layout=True)
    fig2, ax2 = plt.subplots(len(data_locs), 2, figsize=(8, 8), constrained_layout=True)

    if len(data_locs) == 1:
        ax = np.array([ax])
        ax2 = np.array([ax2])

    control_sym_spec = None
    control_asym_spec = None
    sym_spec_im = asym_spec_im = None
    sym_diff_im = asym_diff_im = None

    for d, simulation in enumerate(sim_names):
        product = spectrum_cache.sel(simulation=simulation)
        sym_sig_spectrum = product.symmetric_normalised_power
        asym_sig_spectrum = product.antisymmetric_normalised_power

        wn = sym_sig_spectrum.wavenumber
        f = sym_sig_spectrum.frequency

        # Use boxes rather than contour shading.  Values <= 1 are white/light, so
        # the plot emphasizes enhanced spectral power above the background.
        sym_spec_im = ax[d, 0].pcolormesh(wn, f, sym_sig_spectrum.T, cmap=spec_cmap, norm=spec_norm, shading='auto')
        asym_spec_im = ax[d, 1].pcolormesh(wn, f, asym_sig_spectrum.T, cmap=spec_cmap, norm=spec_norm, shading='auto')

        # Compare only significant excess power above background.
        sym_excess = xr.where(sym_sig_spectrum > 1.0, sym_sig_spectrum - 1.0, 0.0)
        asym_excess = xr.where(asym_sig_spectrum > 1.0, asym_sig_spectrum - 1.0, 0.0)

        if d == 0:
            control_sym_excess = sym_excess
            control_asym_excess = asym_excess

            # Top row: show control excess power, not raw normalised power.
            sym_diff_im = ax2[d, 0].pcolormesh( wn, f, sym_sig_spectrum.T, cmap=spec_cmap, norm=spec_norm, shading='auto' )
            asym_diff_im = ax2[d, 1].pcolormesh( wn, f, asym_sig_spectrum.T, cmap=spec_cmap, norm=spec_norm, shading='auto'
            )

        else:
            c_sym_excess = control_sym_excess.interp( wavenumber=sym_excess.wavenumber, frequency=sym_excess.frequency )
            c_asym_excess = control_asym_excess.interp( wavenumber=asym_excess.wavenumber, frequency=asym_excess.frequency )

            sym_diff = sym_excess - c_sym_excess
            asym_diff = asym_excess - c_asym_excess

            sym_diff_im = ax2[d, 0].pcolormesh( wn, f, sym_diff.T, cmap=spec_diff_cmap, norm=spec_diff_norm, shading='auto' )
            asym_diff_im = ax2[d, 1].pcolormesh( wn, f, asym_diff.T, cmap=spec_diff_cmap, norm=spec_diff_norm, shading='auto' )

        freq, wavenum = genDispersionCurves()

        labels = ['MRG', 'IG', 'EIG', 'Kelvin', 'ER', 'IG']

        for i in range(3):
            for j in range(3):
                for axes in (ax, ax2):
                    axes[d, 1].plot(wavenum[i, j], freq[i, j], color='r', alpha=0.5)
                    axes[d, 0].plot(wavenum[i+3, j], freq[i+3, j], color='r', alpha=0.5)

        # Decorations
        for axes in (ax, ax2):
            for a in axes[d]:
                a.set_xlim(-15, 15)
                a.set_ylim(0, 0.5)
                add_period_reference_lines_frequency(a)
                if d == len(data_locs)-1:
                    a.set_xlabel('Zonal wavenumber')
                else:
                    a.tick_params(labelbottom=False)

            axes[d, 0].set_ylabel('Frequency (day$^{-1}$)')
            axes[d, 1].tick_params(labelleft=False)
            axes[d, 0].axvline(0, color='k', ls='--')
            axes[d, 1].axvline(0, color='k', ls='--')
            annotate_panel_label(axes[d, 0], sim_names[d])
            annotate_panel_label(axes[d, 1], sim_names[d])

        if d == 0:
            for axes in (ax, ax2):
                axes[d, 0].annotate('ER', (-11, 0.12), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
                axes[d, 0].annotate('Kelvin', (10, 0.4), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
                axes[d, 0].annotate('IG', (-5, 0.4), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
                axes[d, 1].annotate('MRG', (-12, 0.07), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
                axes[d, 1].annotate('IG', (-5, 0.43), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
                axes[d, 1].annotate('EIG', (11, 0.32), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))

    ax[0, 0].set_title('Symmetric')
    ax[0, 1].set_title('Anti-symmetric')
    ax2[0, 0].set_title('Symmetric')
    ax2[0, 1].set_title('Anti-symmetric')

    fig.colorbar(sym_spec_im, ax=ax[-1], location='bottom', shrink=0.5, label="Normalised power")
    add_symmetric_colorbar(fig2, sym_diff_im if sym_diff_im is not None else sym_spec_im, ax2[-1], spec_diff_levels, location='bottom', shrink=0.5, label=r"Difference in excess normalised power above 1" )

    save_figure(fig, "spectra", number=7)
    save_figure(fig2, "spectra_diff", number=7)
    print("Done with figure: comp_spectra")

# Precip colourmap definition
precip_cols = [
    [255/255., 255/255., 255/255.], # white
    [205/255., 236/255., 251/255.], # light blue
    [157/255., 218/255., 247/255.], # blue
    [72/255., 142/255., 202/255.], # dark blue
    [72/255., 158/255., 148/255.], # light green
    [73/255., 181/255., 70/255.], # green
    [150/255., 203/255., 80/255.], # green-yellow
    [250/255., 232/255., 92/255.], # yellow
    [248/255., 184/255., 73/255.], # dark yellow
    [245/255., 106/255., 41/255.], # orange
    [230/255., 74/255., 41/255.], # dark orange
    [211/255., 31/255., 40/255.], # red
    [183/255., 27/255., 34/255.], # darker red
    [146/255., 21/255., 25/255.], #dark red
    [110/255., 15/255., 15/255.], #dark red
]

levels = np.linspace(0.01, 25, len(precip_cols))

precip_cmap = colors.LinearSegmentedColormap.from_list(name = "precip_rate", colors=precip_cols, N=len(precip_cols))
precip_norm = colors.BoundaryNorm(boundaries=levels, ncolors=len(precip_cols)-1)



if should_run_figure(8):
    # Deep-tropical Hovmoller diagrams.  PW and OLR show zonally asymmetric
    # anomalies after removing each longitude's time mean and each time's
    # zonal mean.
    fig, ax = plt.subplots(
        3,
        len(dss),
        figsize=(3.4 * len(dss), 7.5),
        constrained_layout=True,
        sharex=True,
        sharey="row",
        squeeze=False,
    )

    precip_image = None
    for column, simulation in enumerate(sim_names):
        pr = rain_cache.precip_rate_hovmoller.sel(simulation=simulation)
        precip_image = ax[0, column].contourf(
            pr.longitude,
            pr.Time,
            pr.values,
            levels=levels,
            norm=precip_norm,
            cmap=precip_cmap,
            extend="max",
        )
        annotate_panel_label(ax[0, column], sim_names[column])

    anomaly_specs = (
        (
            "precip_water_hovmoller",
            "Precipitable-water anomaly",
            r"kg m$^{-2}$",
        ),
        ("olr_hovmoller", "OLR anomaly", r"W m$^{-2}$"),
    )
    anomaly_images = []
    for row, (variable, label, units) in enumerate(anomaly_specs, start=1):
        anomalies = []
        for simulation in sim_names:
            field = rain_cache[variable].sel(simulation=simulation)
            anomaly = field - field.mean("Time", skipna=True)
            anomaly = anomaly - anomaly.mean("longitude", skipna=True)
            anomalies.append(anomaly)
        limit = max(
            safe_nanpercentile(
                np.abs(anomaly.values), 98.0, zero_value=1.0
            )
            for anomaly in anomalies
        )
        anomaly_levels = symmetric_levels(limit, ndiv=19)
        image = None
        for column, anomaly in enumerate(anomalies):
            image = ax[row, column].contourf(
                anomaly.longitude,
                anomaly.Time,
                anomaly.values,
                levels=anomaly_levels,
                cmap=difference_cmap(anomaly_levels),
                extend="both",
            )
        anomaly_images.append((image, anomaly_levels, label, units, row))

    ax[0, 0].set_ylabel("Precipitation\ntime")
    ax[1, 0].set_ylabel("PW anomaly\ntime")
    ax[2, 0].set_ylabel("OLR anomaly\ntime")
    for row in range(3):
        for column in range(1, len(dss)):
            ax[row, column].tick_params(labelleft=False)
    for axis in ax[-1, :]:
        axis.set_xlabel("Longitude")

    fig.colorbar(
        precip_image,
        ax=ax[0, :],
        location="right",
        label=r"Precipitation rate (mm day$^{-1}$)",
    )
    for image, anomaly_levels, label, units, row in anomaly_images:
        add_symmetric_colorbar(
            fig,
            image,
            ax[row, :],
            anomaly_levels,
            label=f"{label} ({units})",
        )

    save_figure(fig, "hovmoller", number=8)
    print("Done with figure: comp_hovmoller")

if should_run_figure(9):
    height_anomaly_levels = symmetric_levels(1.0, ndiv=19)
    # Diagnostic: how much does zgrid vary horizontally?

    fig, ax = plt.subplots(len(dss), 2, figsize=(8, 3.2*len(dss)), constrained_layout=True, sharex='col', sharey='row')

    if len(dss) == 1:
        ax = np.array([ax])

    for i, ds in enumerate(dss):
        z_zonal = ds.zgrid_zonal_mean.values
        zmean = ds.zgrid_global_mean.values
        zmin = ds.zgrid_global_min.values
        zmax = ds.zgrid_global_max.values
        zstd = ds.zgrid_global_std.values

        # Zonal mean and anomaly from global mean
        z_anom = z_zonal - zmean[:, None]

        # ---- left: profile with spread ----
        ax[i, 0].plot(zmean / 1000., zmean / 1000., color=cols[i], lw=1.5)
        ax[i, 0].fill_betweenx(zmean / 1000., zmin / 1000., zmax / 1000.,
                               color=cols[i], alpha=0.15, label='min/max')
        ax[i, 0].fill_betweenx(zmean / 1000., (zmean - zstd) / 1000., (zmean + zstd) / 1000.,
                               color=cols[i], alpha=0.3, label=r'$\pm 1\sigma$')

        ax[i, 0].plot(zmean / 1000., zmean / 1000., color=cols[i], lw=1.5, label=sim_names[i])
        ax[i, 0].set_ylabel("Height (km)")
        annotate_panel_label(ax[i, 0], sim_names[i])
        if i == len(dss) - 1:
            ax[i, 0].set_xlabel("Global-mean height (km)")

        # ---- right: latitude variation ----
        im = ax[i, 1].contourf(
            ds.latitude.values,
            zmean / 1000.,
            z_anom / 1000.,
            levels=height_anomaly_levels,
            cmap=difference_cmap(height_anomaly_levels),
            extend='both'
        )
        annotate_panel_label(ax[i, 1], sim_names[i])
        if i == len(dss) - 1:
            ax[i, 1].set_xlabel("Latitude (degrees)")
        ax[i, 1].tick_params(labelleft=False)

        # optional: print a simple summary
        print(sim_names[i])
        print("max |zonal-mean height anomaly| (km):", np.nanmax(np.abs(z_anom)) / 1000.)
        print("max horizontal range at a level (km):", np.nanmax(zmax - zmin) / 1000.)

    # one colourbar for the anomaly panels
    add_symmetric_colorbar(fig, im, ax[:, 1], height_anomaly_levels, label='Height anomaly from global mean (km)')

    save_figure(fig, "zgrid_horizontal_variation", number=9)

    print("Done with figure: check_zgrid_horizontal_variation")

if should_run_figure(10):
    # Latitude-height section of zonal-mean cloud fraction

    fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)
    fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)

    if len(dss) == 1:
        ax = [ax]
        ax2 = [ax2]

    im = im2 = im3 = None
    control_cldfrac = None

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        cldfrac = zds.cldfrac.mean(dim="Time")
        cldfrac = hemispheric_fold(cldfrac, parity="even")
        zmid = zds.z
        cldfrac, lat2d, z2d = lat_height_mesh(cldfrac, zmid, zds.latitude.values)
        lat_for_cpt = zds.latitude.values

        im = ax[i].contourf(lat2d, z2d / 1000.0, cldfrac.values, levels=np.linspace(0, 1, 21), cmap='binary', extend='max')
        ax[i].plot(lat_for_cpt, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        annotate_panel_label(ax[i], sim_names[i])
        set_latheight_labels(ax[i], xlabel=(i == len(dss)-1))
        if i < len(dss)-1:
            ax[i].tick_params(labelbottom=False)

        if i == 0:
            control_cldfrac = cldfrac.copy()

            im2 = ax2[i].contourf(
                lat2d, z2d / 1000.0, cldfrac.values,
                levels=np.linspace(0, 1, 21),
                cmap='binary',
                extend='max'
            )
        else:
            cldfrac, c_cldfrac = xr.align(cldfrac, control_cldfrac, join="exact")
            cldfrac_diff = cldfrac.values - c_cldfrac.values

            im3 = ax2[i].contourf(
                lat2d, z2d / 1000.0, cldfrac_diff,
                levels=symmetric_levels(0.1, ndiv=11),
                cmap=difference_cmap(symmetric_levels(0.1, ndiv=11)),
                extend='both'
            )

        ax2[i].plot(lat_for_cpt, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
        set_latheight_labels(ax2[i], xlabel=(i == len(dss)-1))
        if i < len(dss)-1:
            ax2[i].tick_params(labelbottom=False)

    fig.colorbar(im, ax=ax, location='right', label='Cloud fraction')
    fig2.colorbar(im2, ax=ax2[0], location='right', label='Cloud fraction')
    if im3 is not None:
        add_symmetric_colorbar(fig2, im3, ax2[1:], symmetric_levels(0.1, ndiv=11), label='Cloud-fraction difference')

    save_figure(fig, "cldfrac_latheight", number=10)
    save_figure(fig2, "cldfrac_latheight_diff", number=10)
    print("Done with figure: comp_cldfrac_latheight")


if should_run_figure(11):
    # Latitude-height section of zonal-mean vorticity

    fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)
    fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)

    if len(dss) == 1:
        ax = [ax]
        ax2 = [ax2]

    vort_plots = {}
    for i, zds in enumerate(dss_zm):
        vort = zds.vorticity.mean(dim="Time")
        vort = hemispheric_fold(vort, parity="odd")
        zmid = zds.z
        vort, lat2d, z2d = lat_height_mesh(vort, zmid, zds.latitude.values)
        vort_plots[i] = (vort, lat2d, z2d)

    control_vort = vort_plots[0][0] if 0 in vort_plots else None
    vort_diffs = {}
    if control_vort is not None:
        for i, (vort, lat2d, z2d) in vort_plots.items():
            if i == 0:
                continue
            vort, c_vort = xr.align(vort, control_vort, join="exact")
            vort_diffs[i] = vort - c_vort

    dlevels = (
        symmetric_contour_levels(list(vort_diffs.values()), ndiv=11)
        if vort_diffs
        else symmetric_levels(1.0, ndiv=11)
    )
    split_masks = split_tropo_levels = split_strat_levels = None
    split_diff = False
    if vort_diffs:
        split_masks, split_tropo_levels, split_strat_levels, split_diff = (
            tropopause_split_diff_context(
                {i: (vort_diffs[i].values, vort_plots[i][2], vort_diffs[i].latitude.values) for i in vort_diffs},
                {"diff_split_tropopause": True},
                ndiv=11,
            )
        )

    im = im2 = im3 = tropo_im = strat_im = None

    for i, zds in enumerate(dss_zm):
        if i not in vort_plots:
            continue
        vort, lat2d, z2d = vort_plots[i]

        vmax = np.nanmax(np.abs(vort.values))
        levels = symmetric_levels(vmax, ndiv=19)

        im = ax[i].contourf(
            lat2d,
            z2d / 1000.0,
            vort.values,
            levels=levels,
            cmap=difference_cmap(levels),
            extend='both',
        )
        ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        annotate_panel_label(ax[i], sim_names[i])
        set_latheight_labels(ax[i], xlabel=(i == len(dss)-1))
        if i < len(dss)-1:
            ax[i].tick_params(labelbottom=False)

        if i == 0:
            im2 = ax2[i].contourf(
                lat2d,
                z2d / 1000.0,
                vort.values,
                levels=levels,
                cmap=difference_cmap(levels),
                extend='both',
            )
        elif i in vort_diffs:
            if split_diff:
                tropo_im, strat_im = contourf_split_difference(
                    ax2[i], lat2d, z2d/1000., vort_diffs[i].values,
                    split_masks[i][0], split_tropo_levels, split_strat_levels)
            else:
                im3 = ax2[i].contourf(lat2d, z2d/1000., vort_diffs[i].values, levels=dlevels, cmap=difference_cmap(dlevels), extend='both')

        ax2[i].plot(zds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
        set_latheight_labels(ax2[i], xlabel=(i == len(dss)-1))
        if i < len(dss)-1:
            ax2[i].tick_params(labelbottom=False)

    add_symmetric_colorbar(fig, im, ax, levels, label='Vorticity (s$^{-1}$)')
    fig2.colorbar(im2, ax=ax2[0], location='right', label='Vorticity (s$^{-1}$)')
    if split_diff and tropo_im is not None:
        add_split_difference_colorbars(fig2, tropo_im, strat_im, ax2[1:], split_tropo_levels, split_strat_levels, label='Vorticity diff. (s$^{-1}$)')
    elif im3 is not None:
        add_symmetric_colorbar(fig2, im3, ax2[1:], dlevels, label='Vorticity difference (s$^{-1}$)')

    save_figure(fig, "vorticity_latheight", number=11)
    save_figure(fig2, "vorticity_latheight_diff", number=11)
    print("Done with figure: comp_vorticity_latheight")


if should_run_figure(12):
    # Histogram of OLR and time-mean lat/lon maps

    map_profile_widths = [1.0] * len(dss) + [0.72]
    fig, axes = plt.subplots(
        1, len(dss) + 1,
        figsize=(12, 3.5),
        constrained_layout=True,
        sharey=True,
        squeeze=False,
        gridspec_kw={"width_ratios": map_profile_widths},
    )
    fig2, axes2 = plt.subplots(
        1, len(dss) + 1,
        figsize=(12, 3.5),
        constrained_layout=True,
        sharey=True,
        squeeze=False,
        gridspec_kw={"width_ratios": map_profile_widths},
    )
    fig3, ax3 = plt.subplots(figsize=(8, 3), constrained_layout=True)
    ax = axes[0, :-1]
    ax_profile = axes[0, -1]
    ax2 = axes2[0, :-1]
    ax2_profile = axes2[0, -1]

    olr_cols = cols
    bins = np.linspace(100, 350, 80)
    im = im2 = im3 = None
    control_olr = None

    for i, ds in enumerate(dss):
        olr = ds.olrtoa
        olr_mean = olr.mean(dim='Time')
        olr_mean = hemispheric_fold(olr_mean, parity="even")
        olr_zonal_mean = olr_mean.mean(dim="longitude", skipna=True).load()
        ax_profile.plot(olr_zonal_mean.values, olr_zonal_mean.latitude.values,
                        color=olr_cols[i], label=sim_names[i])

        histograms = ds.olr_histogram
        olr_edges = ds.olr_bin_edges.values
        for domain_index, (domain, linestyle) in enumerate(DISTRIBUTION_DOMAIN_STYLES):
            h = histograms.isel(domain=domain_index).values
            b = olr_edges
            ax3.plot(
                0.5 * (b[1:] + b[:-1]),
                h / h.sum(),
                color=olr_cols[i],
                ls=linestyle,
                label=f"{sim_names[i]}, {domain}",
            )

        im = ax[i].contourf(
            ds.longitude.values,
            ds.latitude.values,
            olr_mean.values,
            levels=np.linspace(200, 300, 151),
            norm=colors.PowerNorm(gamma=0.5), cmap='Blues',
            extend='both'
        )
        annotate_panel_label(ax[i], sim_names[i])
        ax[i].set_xlabel('Longitude')
        if i == 0:
            ax[i].set_ylabel('Latitude')
            control_olr = olr_mean
            im2 = ax2[i].contourf(
                ds.longitude.values,
                ds.latitude.values,
                olr_mean.values,
                levels=np.linspace(200, 300, 151),
                norm=colors.PowerNorm(gamma=0.5), cmap='Blues',
                extend='both'
            )
        else:
            ax[i].tick_params(labelleft=False)
            olr_mean, control_interp = xr.align(olr_mean, control_olr, join="exact")
            olr_diff = (olr_mean - control_interp).load()
            im3 = ax2[i].contourf(
                ds.longitude.values,
                ds.latitude.values,
                olr_diff.values,
                levels=symmetric_levels(20.0, ndiv=11),
                cmap=difference_cmap(symmetric_levels(20.0, ndiv=11)),
                extend='both'
            )
            olr_diff_zonal_mean = olr_diff.mean(dim="longitude", skipna=True)
            ax2_profile.plot(olr_diff_zonal_mean.values, olr_diff_zonal_mean.latitude.values,
                             color=olr_cols[i], label=sim_names[i])

        annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
        ax2[i].set_xlabel('Longitude')
        if i == 0:
            ax2[i].set_ylabel('Latitude')
            ax2_profile.plot(np.zeros_like(olr_zonal_mean.values), olr_zonal_mean.latitude.values,
                             color=olr_cols[i], label=sim_names[i])
        else:
            ax2[i].tick_params(labelleft=False)

    ax_profile.axhline(0, color="0.3", lw=0.6)
    ax_profile.set_title("Longitude mean")
    ax_profile.set_xlabel(r'OLR (W m$^{-2}$)')
    ax_profile.tick_params(labelleft=False)
    ax_profile.grid(True, axis="x", lw=0.4, alpha=0.5)
    ax_profile.legend(loc="best", fontsize=8, frameon=False)

    ax2_profile.axvline(0, color="0.3", lw=0.7)
    ax2_profile.axhline(0, color="0.3", lw=0.6)
    ax2_profile.set_title("Longitude mean")
    ax2_profile.set_xlabel(r'$\Delta$ OLR (W m$^{-2}$)')
    ax2_profile.tick_params(labelleft=False)
    ax2_profile.grid(True, axis="x", lw=0.4, alpha=0.5)
    ax2_profile.legend(loc="best", fontsize=8, frameon=False)

    ax3.set_xlabel(r'OLR (W m$^{-2}$)')
    ax3.set_ylabel('Normalised frequency')
    ax3.legend(loc='best')

    fig.colorbar(im, ax=list(ax) + [ax_profile], location='right', label=r'OLR (W m$^{-2}$)')
    fig2.colorbar(im2, ax=ax2[0], location='right', label=r'OLR (W m$^{-2}$)')
    if im3 is not None:
        add_symmetric_colorbar(
            fig2,
            im3,
            list(ax2[1:]) + [ax2_profile],
            symmetric_levels(20.0, ndiv=11),
            label=r'OLR difference (W m$^{-2}$)',
        )

    save_figure(fig, "olrtoa_maps", number=12)
    save_figure(fig2, "olrtoa_maps_diff", number=12)
    save_figure(fig3, "olrtoa_hist", number=12)
    print("Done with figure: comp_olrtoa_hist_maps")

    

if should_run_figure(16):
    # Time-mean latitude-height sections using a shared raw/control-difference layout.
    latheight_groups = [
        {
            "title": "temperature/wind",
            "specs": LATHEIGHT_MEAN_STATE_SPECS,
            "raw_name": "temperature_wind_latheight",
            "diff_name": "temperature_wind_latheight_diff",
        },
        {
            "title": "w/RH",
            "specs": LATHEIGHT_W_RH_SPECS,
            "raw_name": "w_rh_latheight",
            "diff_name": "w_rh_latheight_diff",
        },
    ]

    for group in latheight_groups:
        latheight_entries = []
        for spec in group["specs"]:
            fields_by_sim = {}
            for i, zds in enumerate(dss_zm):
                fields_by_sim[i] = latheight_time_mean_field(zds, spec)

            latheight_entries.append((spec, fields_by_sim))

        if not latheight_entries:
            print(f"Skipping {group['title']} latitude-height plots: no requested fields were available")
            continue

        nrows = len(latheight_entries)
        ncols = len(dss_zm)
        fig, ax = plt.subplots(
            nrows, ncols,
            figsize=(3.7*ncols, 2.8*nrows),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        fig2, ax2 = plt.subplots(
            nrows, ncols,
            figsize=(3.7*ncols, 2.8*nrows),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        for row, (spec, fields_by_sim) in enumerate(latheight_entries):
            raw_fields = list(fields_by_sim.values())
            raw_level_fields = fields_for_level_limits(raw_fields, spec)
            if spec.get("raw_percentile") is not None and spec.get("raw_symmetric", False):
                raw_levels = symmetric_percentile_levels(
                    raw_level_fields,
                    percentile=spec.get("raw_percentile", 99.0),
                    ndiv=10,
                    fallback=spec.get("raw_fallback", 1.0),
                )
            else:
                raw_levels = latlon_contour_levels(
                    raw_level_fields,
                    symmetric=spec.get("raw_symmetric", False),
                    fallback=spec.get("raw_fallback", 1.0),
                )
            control_field = fields_by_sim[0]
            diff_fields = {}
            for i, field in fields_by_sim.items():
                if i == 0:
                    continue
                field, control_interp = xr.align(field, control_field, join="exact")
                diff_fields[i] = (field - control_interp).load()
            diff_level_fields = fields_for_level_limits(list(diff_fields.values()), spec)

            if diff_fields and spec.get("diff_percentile") is not None:
                diff_levels = symmetric_percentile_levels(
                    diff_level_fields,
                    percentile=spec.get("diff_percentile", 99.0),
                    ndiv=10,
                    fallback=spec.get("diff_fallback", 1.0),
                )
            else:
                diff_levels = (
                    symmetric_contour_levels(
                        diff_level_fields,
                        ndiv=10,
                        fallback=spec.get("diff_fallback", 1.0),
                    )
                    if diff_fields
                    else symmetric_levels(spec.get("diff_fallback", 1.0), ndiv=10)
                )

            diff_plots = {}
            for i, diff in diff_fields.items():
                diff_plot, diff_lat2d, diff_z2d = lat_height_mesh(
                    diff, dss_zm[i].z, diff.latitude.values
                )
                diff_plots[i] = (diff_plot.values, diff_z2d, diff.latitude.values, diff_lat2d)

            split_masks = split_tropo_levels = split_strat_levels = None
            split_diff = False
            if diff_plots and spec.get("diff_split_tropopause", False):
                split_masks, split_tropo_levels, split_strat_levels, split_diff = (
                    tropopause_split_diff_context(
                        diff_plots, spec, percentile=spec.get("diff_percentile")
                    )
                )

            raw_im = control_im = diff_im = tropo_im = strat_im = None
            for i in range(ncols):
                field = fields_by_sim.get(i)
                for axes in (ax[row, i], ax2[row, i]):
                    set_latheight_labels(axes, xlabel=(row == nrows-1), ylabel=(i == 0))
                    if i == 0:
                        axes.set_ylabel(f"{spec['label']}\nHeight (km)")
                    else:
                        axes.tick_params(labelleft=False)
                    if row < nrows-1:
                        axes.tick_params(labelbottom=False)

                if field is None:
                    ax[row, i].set_visible(False)
                    ax2[row, i].set_visible(False)
                    continue

                zmid = dss_zm[i].z
                field_plot, lat2d, z2d = lat_height_mesh(field, zmid, field.latitude.values)
                raw_im = ax[row, i].contourf(
                    lat2d,
                    z2d / 1000.0,
                    field_plot.values,
                    levels=raw_levels,
                    cmap=spec["cmap"],
                    extend="both",
                )
                ax[row, i].plot(dss_zm[i].latitude.values, CPTs[i][0]/1000.0, color="k", lw=1, ls=":")
                annotate_panel_label(ax[row, i], sim_names[i])

                if i == 0:
                    control_im = ax2[row, i].contourf(
                        lat2d,
                        z2d / 1000.0,
                        field_plot.values,
                        levels=raw_levels,
                        cmap=spec["cmap"],
                        extend="both",
                    )
                else:
                    diff_values, diff_z2d, _, diff_lat2d = diff_plots[i]
                    if split_diff:
                        tropo_im, strat_im = contourf_split_difference(
                            ax2[row, i],
                            diff_lat2d,
                            diff_z2d / 1000.0,
                            diff_values,
                            split_masks[i][0],
                            split_tropo_levels,
                            split_strat_levels,
                        )
                    else:
                        diff_im = ax2[row, i].contourf(
                            diff_lat2d,
                            diff_z2d / 1000.0,
                            diff_values,
                            levels=diff_levels,
                            cmap=difference_cmap(diff_levels),
                            extend="both",
                        )
                ax2[row, i].plot(dss_zm[i].latitude.values, CPTs[i][0]/1000.0, color="k", lw=1, ls=":")
                annotate_panel_label(ax2[row, i], control_diff_label(sim_names, i))

            label = f"{spec['label']} ({spec['units']})"
            if raw_im is not None:
                fig.colorbar(raw_im, ax=ax[row, :], location="right", label=label)
            if control_im is not None:
                fig2.colorbar(control_im, ax=ax2[row, 0], location="right", label=label)
            if split_diff and tropo_im is not None:
                add_split_difference_colorbars(
                    fig2,
                    tropo_im,
                    strat_im,
                    ax2[row, 1:],
                    split_tropo_levels,
                    split_strat_levels,
                    label=f"diff. ({spec['units']})",
                )
            elif diff_im is not None:
                add_symmetric_colorbar(
                    fig2,
                    diff_im,
                    ax2[row, 1:],
                    diff_levels,
                    label=f"{spec['label']} difference ({spec['units']})",
                )

        save_figure(fig, group["raw_name"], number=16)
        save_figure(fig2, group["diff_name"], number=16)
        print(f"Done with figure: comp_{group['raw_name']}")

    

if should_run_figure(17):
    # Time-mean MSLP lat/lon maps, using the same raw/control-difference layout
    # as the OLR maps.

    spec = MSLP_LATLON_SPEC
    fields_by_sim = {}
    for i, ds in enumerate(dss):
        fields_by_sim[i] = latlon_time_mean_field(
            ds,
            spec["var"],
            scale=spec.get("scale", 1.0),
            name=spec["label"],
            hemispheric_parity=spec.get("hemispheric_parity", "even"),
        )

    if 0 in fields_by_sim:
        map_profile_widths = [1.0] * len(dss) + [0.72]
        fig, axes = plt.subplots(
            1, len(dss) + 1,
            figsize=(12, 3.5),
            constrained_layout=True,
            sharey=True,
            squeeze=False,
            gridspec_kw={"width_ratios": map_profile_widths},
        )
        fig2, axes2 = plt.subplots(
            1, len(dss) + 1,
            figsize=(12, 3.5),
            constrained_layout=True,
            sharey=True,
            squeeze=False,
            gridspec_kw={"width_ratios": map_profile_widths},
        )
        ax = axes[0, :-1]
        ax_profile = axes[0, -1]
        ax2 = axes2[0, :-1]
        ax2_profile = axes2[0, -1]

        raw_levels = latlon_contour_levels(
            list(fields_by_sim.values()),
            symmetric=spec.get("raw_symmetric", False),
            fallback=spec.get("raw_fallback", 1.0),
        )
        control_field = fields_by_sim[0]
        diff_fields = {}
        for i, field in fields_by_sim.items():
            if i == 0:
                continue
            field, control_interp = xr.align(field, control_field, join="exact")
            diff_fields[i] = (field - control_interp).load()

        diff_levels = (
            symmetric_contour_levels(
                list(diff_fields.values()),
                ndiv=10,
                fallback=spec.get("diff_fallback", 1.0),
            )
            if diff_fields
            else symmetric_levels(spec.get("diff_fallback", 1.0), ndiv=10)
        )

        raw_im = control_im = diff_im = None
        for i in range(len(dss)):
            field = fields_by_sim.get(i)
            if field is None:
                ax[i].set_visible(False)
                ax2[i].set_visible(False)
                continue

            raw_im = ax[i].contourf(
                field.longitude.values,
                field.latitude.values,
                field.values,
                levels=raw_levels,
                cmap=spec["cmap"],
                extend="both",
            )
            annotate_panel_label(ax[i], sim_names[i])
            ax[i].set_xlabel("Longitude")
            zonal_mean = field.mean(dim="longitude", skipna=True).load()
            ax_profile.plot(zonal_mean.values, zonal_mean.latitude.values, color=cols[i], label=sim_names[i])

            if i == 0:
                ax[i].set_ylabel("Latitude")
                control_im = ax2[i].contourf(
                    field.longitude.values,
                    field.latitude.values,
                    field.values,
                    levels=raw_levels,
                    cmap=spec["cmap"],
                    extend="both",
                )
                ax2_profile.plot(np.zeros_like(zonal_mean.values), zonal_mean.latitude.values,
                                 color=cols[i], label=sim_names[i])
            else:
                ax[i].tick_params(labelleft=False)
                diff = diff_fields[i]
                diff_im = ax2[i].contourf(
                    diff.longitude.values,
                    diff.latitude.values,
                    diff.values,
                    levels=diff_levels,
                    cmap=difference_cmap(diff_levels),
                    extend="both",
                )
                diff_zonal_mean = diff.mean(dim="longitude", skipna=True).load()
                ax2_profile.plot(diff_zonal_mean.values, diff_zonal_mean.latitude.values,
                                 color=cols[i], label=sim_names[i])

            annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
            ax2[i].set_xlabel("Longitude")
            if i == 0:
                ax2[i].set_ylabel("Latitude")
            else:
                ax2[i].tick_params(labelleft=False)

        label = f"{spec['label']} ({spec['units']})"
        ax_profile.axhline(0, color="0.3", lw=0.6)
        ax_profile.set_title("Longitude mean")
        ax_profile.set_xlabel(label)
        ax_profile.tick_params(labelleft=False)
        ax_profile.grid(True, axis="x", lw=0.4, alpha=0.5)
        ax_profile.legend(loc="best", fontsize=8, frameon=False)

        ax2_profile.axvline(0, color="0.3", lw=0.7)
        ax2_profile.axhline(0, color="0.3", lw=0.6)
        ax2_profile.set_title("Longitude mean")
        ax2_profile.set_xlabel(f"{spec['label']} difference ({spec['units']})")
        ax2_profile.tick_params(labelleft=False)
        ax2_profile.grid(True, axis="x", lw=0.4, alpha=0.5)
        ax2_profile.legend(loc="best", fontsize=8, frameon=False)

        if raw_im is not None:
            fig.colorbar(raw_im, ax=list(ax) + [ax_profile], location="right", label=label)
        if control_im is not None:
            fig2.colorbar(control_im, ax=ax2[0], location="right", label=label)
        if diff_im is not None:
            add_symmetric_colorbar(
                fig2,
                diff_im,
                list(ax2[1:]) + [ax2_profile],
                diff_levels,
                label=f"{spec['label']} difference ({spec['units']})",
            )

        save_figure(fig, "mslp_maps", number=17)
        save_figure(fig2, "mslp_maps_diff", number=17)
        print("Done with figure: comp_mslp_maps")
    else:
        print("Skipping MSLP lat/lon maps: control field is missing")

    

if should_run_figure(18):
    # Time-mean latitude-height sections of MPAS moist scalar constituents.
    # Grouping separates vapour, suspended cloud condensate, and precipitating
    # condensate.

    zmid_cache = {}
    for i, ds in enumerate(dss):
        try:
            zmid_cache[i] = ds.zmid_latitude
        except Exception as e:
            raise RuntimeError(
                f"Scalar constituent heights failed for {sim_names[i]}"
            ) from e

    for group in SCALAR_CONSTITUENT_GROUPS:
        scalar_entries = []
        for spec in group["fields"]:
            fields_by_sim = {}
            meshes_by_sim = {}
            for i, ds in enumerate(dss):
                if i not in zmid_cache:
                    continue
                try:
                    field, lat2d, z2d = scalar_constituent_latheight(ds, spec, zmid=zmid_cache[i])
                except ValueError as e:
                    print(f"Skipping {spec['label']} latitude-height plot for {sim_names[i]}: {e}")
                    continue
                fields_by_sim[i] = field
                meshes_by_sim[i] = (lat2d, z2d)

            if 0 not in fields_by_sim:
                print(f"Skipping {spec['label']} latitude-height plots: control field is missing")
                continue

            scalar_entries.append((spec, fields_by_sim, meshes_by_sim))

        if not scalar_entries:
            print(f"Skipping {group['title']} scalar-constituent plots: no requested fields were available")
            continue

        nrows = len(scalar_entries)
        ncols = len(dss)
        fig, ax = plt.subplots(
            nrows, ncols,
            figsize=(3.7*ncols, 2.8*nrows),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )
        fig2, ax2 = plt.subplots(
            nrows, ncols,
            figsize=(3.7*ncols, 2.8*nrows),
            constrained_layout=True,
            squeeze=False,
            sharex=True,
            sharey=True,
        )

        for row, (spec, fields_by_sim, meshes_by_sim) in enumerate(scalar_entries):
            raw_levels, raw_norm, raw_extend = scalar_raw_levels_and_norm(list(fields_by_sim.values()), spec)
            control_field = fields_by_sim[0]
            diff_fields = {}
            for i, field in fields_by_sim.items():
                if i == 0:
                    continue
                field, control_interp = xr.align(field, control_field, join="exact")
                diff_fields[i] = (field - control_interp).load()

            diff_levels = (
                symmetric_percentile_levels(
                    list(diff_fields.values()),
                    percentile=99.0,
                    fallback=spec.get("diff_fallback", 1.0),
                )
                if diff_fields
                else symmetric_levels(spec.get("diff_fallback", 1.0), ndiv=10)
            )

            diff_plots = {}
            for i, diff in diff_fields.items():
                diff_lat2d, diff_z2d = meshes_by_sim[i]
                diff_plots[i] = (diff.values, diff_z2d, diff.latitude.values, diff_lat2d)

            split_masks = split_tropo_levels = split_strat_levels = None
            split_diff = False
            if diff_plots and spec.get("diff_split_tropopause", False):
                split_masks, split_tropo_levels, split_strat_levels, split_diff = (
                    tropopause_split_diff_context(diff_plots, spec, percentile=99.0)
                )

            raw_im = control_im = diff_im = tropo_im = strat_im = None
            for i in range(ncols):
                field = fields_by_sim.get(i)
                for axes in (ax[row, i], ax2[row, i]):
                    set_latheight_labels(axes, xlabel=(row == nrows-1), ylabel=(i == 0))
                    if i == 0:
                        axes.set_ylabel(f"{spec['symbol']}\nHeight (km)")
                    else:
                        axes.tick_params(labelleft=False)
                    if row < nrows-1:
                        axes.tick_params(labelbottom=False)

                if field is None:
                    ax[row, i].set_visible(False)
                    ax2[row, i].set_visible(False)
                    continue

                lat2d, z2d = meshes_by_sim[i]
                raw_values = scalar_raw_values(field, spec)
                raw_im = ax[row, i].contourf(
                    lat2d,
                    z2d / 1000.0,
                    raw_values,
                    levels=raw_levels,
                    cmap=spec["cmap"],
                    norm=raw_norm,
                    extend=raw_extend,
                )
                ax[row, i].plot(dss_zm[i].latitude.values, CPTs[i][0]/1000.0, color="k", lw=1, ls=":")
                annotate_panel_label(ax[row, i], sim_names[i])

                if i == 0:
                    control_im = ax2[row, i].contourf(
                        lat2d,
                        z2d / 1000.0,
                        raw_values,
                        levels=raw_levels,
                        cmap=spec["cmap"],
                        norm=raw_norm,
                        extend=raw_extend,
                    )
                else:
                    diff = diff_fields[i]
                    if split_diff:
                        tropo_im, strat_im = contourf_split_difference(
                            ax2[row, i],
                            lat2d,
                            z2d / 1000.0,
                            diff.values,
                            split_masks[i][0],
                            split_tropo_levels,
                            split_strat_levels,
                        )
                    else:
                        diff_im = ax2[row, i].contourf(
                            lat2d,
                            z2d / 1000.0,
                            diff.values,
                            levels=diff_levels,
                            cmap=difference_cmap(diff_levels),
                            extend="both",
                        )

                ax2[row, i].plot(dss_zm[i].latitude.values, CPTs[i][0]/1000.0, color="k", lw=1, ls=":")
                annotate_panel_label(ax2[row, i], control_diff_label(sim_names, i))

            label = f"{spec['label']} ({spec['units']})"
            if raw_im is not None:
                fig.colorbar(raw_im, ax=ax[row, :], location="right", label=label)
            if control_im is not None:
                fig2.colorbar(control_im, ax=ax2[row, 0], location="right", label=label)
            if split_diff and tropo_im is not None:
                add_split_difference_colorbars(
                    fig2,
                    tropo_im,
                    strat_im,
                    ax2[row, 1:],
                    split_tropo_levels,
                    split_strat_levels,
                    label=f"diff. ({spec['units']})",
                )
            elif diff_im is not None:
                add_symmetric_colorbar(
                    fig2,
                    diff_im,
                    ax2[row, 1:],
                    diff_levels,
                    label=f"{spec['label']} difference ({spec['units']})",
                )

        fig.suptitle(group["title"])
        fig2.suptitle(f"{group['title']} differences")
        save_figure(fig, f"scalar_constituents_{group['name']}_latheight", number=18)
        save_figure(fig2, f"scalar_constituents_{group['name']}_latheight_diff", number=18)
        print(f"Done with figure: comp_scalar_constituents_{group['name']}_latheight")

    

if should_run_figure(13):
    # Latitude-height section of zonal-mean kinetic energy

    fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)
    fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)

    if len(dss) == 1:
        ax = [ax]
        ax2 = [ax2]

    ke_plots = {}
    for i, zds in enumerate(dss_zm):
        ke = zds.ke.mean(dim="Time")
        ke = hemispheric_fold(ke, parity="even")
        zmid = zds.z
        ke, lat2d, z2d = lat_height_mesh(ke, zmid, zds.latitude.values)
        ke_plots[i] = (ke, lat2d, z2d)

    control_ke = ke_plots[0][0] if 0 in ke_plots else None
    ke_diffs = {}
    if control_ke is not None:
        for i, (ke, lat2d, z2d) in ke_plots.items():
            if i == 0:
                continue
            ke, c_ke = xr.align(ke, control_ke, join="exact")
            ke_diffs[i] = ke - c_ke

    ke_diff_levels = symmetric_levels(100.0, ndiv=11)
    split_masks = split_tropo_levels = split_strat_levels = None
    split_diff = False
    if ke_diffs:
        split_masks, split_tropo_levels, split_strat_levels, split_diff = (
            tropopause_split_diff_context(
                {i: (ke_diffs[i].values, ke_plots[i][2], ke_diffs[i].latitude.values) for i in ke_diffs},
                {"diff_split_tropopause": True, "diff_fallback": 100.0},
                ndiv=11,
            )
        )

    im = im2 = im3 = tropo_im = strat_im = None

    for i, zds in enumerate(dss_zm):
        if i not in ke_plots:
            continue
        ke, lat2d, z2d = ke_plots[i]

        ke_levels = positive_levels(np.nanmax(ke.values), n=21)
        im = ax[i].contourf(lat2d, z2d/1000., ke.values, levels=ke_levels, cmap='YlGnBu', extend='max')
        ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        annotate_panel_label(ax[i], sim_names[i])
        set_latheight_labels(ax[i], xlabel=(i == len(dss)-1))
        if i < len(dss)-1:
            ax[i].tick_params(labelbottom=False)

        if i == 0:
            im2 = ax2[i].contourf(lat2d, z2d/1000., ke.values, levels=ke_levels, cmap='YlGnBu', extend='max')
        elif i in ke_diffs:
            if split_diff:
                tropo_im, strat_im = contourf_split_difference(
                    ax2[i], lat2d, z2d/1000., ke_diffs[i].values,
                    split_masks[i][0], split_tropo_levels, split_strat_levels)
            else:
                im3 = ax2[i].contourf(lat2d, z2d/1000., ke_diffs[i].values, levels=ke_diff_levels, cmap=difference_cmap(ke_diff_levels), extend='both')

        ax2[i].plot(zds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
        set_latheight_labels(ax2[i], xlabel=(i == len(dss)-1))
        if i < len(dss)-1:
            ax2[i].tick_params(labelbottom=False)

    fig.colorbar(im, ax=ax, location='right', label=r'Kinetic energy (m$^2$ s$^{-2}$)')
    fig2.colorbar(im2, ax=ax2[0], location='right', label=r'Kinetic energy (m$^2$ s$^{-2}$)')
    if split_diff and tropo_im is not None:
        add_split_difference_colorbars(fig2, tropo_im, strat_im, ax2[1:], split_tropo_levels, split_strat_levels, label=r'KE difference (m$^2$ s$^{-2}$)')
    elif im3 is not None:
        add_symmetric_colorbar(fig2, im3, ax2[1:], ke_diff_levels, label=r'KE difference (m$^2$ s$^{-2}$)')

    save_figure(fig, "ke_latheight", number=13)
    save_figure(fig2, "ke_latheight_diff", number=13)
    print("Done with figure: comp_ke_latheight")

    

# Latitude-pressure section of isobaric eddy kinetic energy

if should_run_figure(14):
    eke_entries = []

    for i, ds in enumerate(dss):
        eke = hemispheric_fold(
            ds.eke_isobaric.transpose("pres", "latitude"), parity="even"
        ).load()
        eke_entries.append((i, eke))

    if eke_entries:
        nrows = len(eke_entries)
        fig, ax = plt.subplots(nrows, 1, figsize=(8, 3.5*nrows), constrained_layout=True, sharex=True)
        fig2, ax2 = plt.subplots(nrows, 1, figsize=(8, 3.5*nrows), constrained_layout=True, sharex=True)

        if nrows == 1:
            ax = [ax]
            ax2 = [ax2]

        eke_fields = [eke for _, eke in eke_entries]
        eke_levels = positive_contour_levels(eke_fields)
        diff_by_sim = {}
        control_index, control_eke = eke_entries[0]

        for i, eke in eke_entries[1:]:
            eke, c_eke = xr.align(eke, control_eke, join="exact")
            diff_by_sim[i] = eke - c_eke

        diff_levels = (
            symmetric_contour_levels(list(diff_by_sim.values()), ndiv=11)
            if diff_by_sim
            else symmetric_levels(1.0, ndiv=11)
        )

        diff_masks = {}
        for i, diff in diff_by_sim.items():
            pres2d = np.broadcast_to(diff.pres.values[:, None], diff.shape)
            cpt_hpa = tropopause_on_latitudes(
                CPTs[i][1] / 100.0, dss_zm[i].latitude.values, diff.latitude.values
            )
            diff_masks[i] = tropopause_region_masks(pres2d, cpt_hpa, coord="pressure")

        split_tropo_levels = split_strat_levels = None
        split_diff = False
        if diff_by_sim:
            split_tropo_levels = region_symmetric_levels(
                [(diff_by_sim[i].values, diff_masks[i][0]) for i in diff_by_sim], ndiv=11
            )
            split_strat_levels = region_symmetric_levels(
                [(diff_by_sim[i].values, diff_masks[i][1]) for i in diff_by_sim], ndiv=11
            )
            split_diff = split_scales_warranted(split_tropo_levels, split_strat_levels)

        im = im2 = im3 = tropo_im = strat_im = None

        for row, (i, eke) in enumerate(eke_entries):
            im = ax[row].contourf(
                eke.latitude.values,
                eke.pres.values,
                eke.values,
                levels=eke_levels,
                cmap="YlGnBu",
                extend="max"
            )
            ax[row].plot(dss[i].latitude.values, CPTs[i][1]/100., color="k", lw=1, ls=":")
            annotate_panel_label(ax[row], sim_names[i])
            set_latpres_axis(ax[row], xlabel=(row == nrows-1), ylabel=True, xlim=(-90, 90), ylim=(1000, 50))
            if row < nrows-1:
                ax[row].tick_params(labelbottom=False)

            if row == 0:
                im2 = ax2[row].contourf(
                    eke.latitude.values,
                    eke.pres.values,
                    eke.values,
                    levels=eke_levels,
                    cmap="YlGnBu",
                    extend="max"
                )
            else:
                diff = diff_by_sim[i]
                if split_diff:
                    tropo_im, strat_im = contourf_split_difference(
                        ax2[row], eke.latitude.values, eke.pres.values, diff.values,
                        diff_masks[i][0], split_tropo_levels, split_strat_levels)
                else:
                    im3 = ax2[row].contourf(
                        eke.latitude.values,
                        eke.pres.values,
                        diff.values,
                        levels=diff_levels,
                        cmap=difference_cmap(diff_levels),
                        extend="both"
                    )

            ax2[row].plot(dss[i].latitude.values, CPTs[i][1]/100., color="k", lw=1, ls=":")
            annotate_panel_label(ax2[row], control_diff_label(sim_names, i, control=control_index))
            set_latpres_axis(ax2[row], xlabel=(row == nrows-1), ylabel=True, xlim=(-90, 90), ylim=(1000, 50))
            if row < nrows-1:
                ax2[row].tick_params(labelbottom=False)

        fig.colorbar(im, ax=ax, location="right", label=r"Eddy kinetic energy (m$^2$ s$^{-2}$)")
        fig2.colorbar(im2, ax=ax2[0], location="right", label=r"Eddy kinetic energy (m$^2$ s$^{-2}$)")
        if split_diff and tropo_im is not None:
            add_split_difference_colorbars(fig2, tropo_im, strat_im, ax2[1:], split_tropo_levels, split_strat_levels, label=r"EKE difference (m$^2$ s$^{-2}$)")
        elif im3 is not None:
            add_symmetric_colorbar(fig2, im3, ax2[1:], diff_levels, label=r"EKE difference (m$^2$ s$^{-2}$)")

        save_figure(fig, "eke_latpres", number=14)
        save_figure(fig2, "eke_latpres_diff", number=14)
        print("Done with figure: comp_eke_latpres")
    else:
        print("Skipping EKE latitude-pressure plot: no simulations had the required winds")

progress("Completed mean-state diagnostics")
