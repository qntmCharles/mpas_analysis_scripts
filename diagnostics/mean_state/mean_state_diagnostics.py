#!/usr/bin/env python
# coding: utf-8



import os, glob, sys
from functools import partial

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
from matplotlib import pyplot as plt
import xarray as xr
from dask import compute as dask_compute
from matplotlib import colors
from tropopause import tropopause_profile, tropopause_profile_single
from wavenumber_frequency_functions import (
    genDispersionCurves,
    smooth_wavefreq,
    spacetime_power,
)
from diagnostics.diagnostic_helpers import (
    FigureSaver, LATLON_PATTERN, ZONAL_MEAN_PATTERN,
    accumulated_precipitation_components, add_period_reference_lines_frequency,
    add_split_difference_colorbars, add_symmetric_colorbar,
    annotate_panel_label, contourf_split_difference,
    control_diff_label, diagnostic_output_dir, difference_cmap,
    cosine_weighted_latitude_coarsen, cosine_weighted_power_input,
    domain_histograms, duration_domain_histograms, figure_should_run,
    figure_should_run_any, get_time_values,
    hemispheric_fold, isobaric_lat_mean,
    isobaric_vertical_velocity_from_zm, lat_height_mesh, lat_mean,
    load_latlon_dataset, make_spectrum_cmap, midlevel_height_from_latlon,
    model_lat_field, nice_levels, plot_Tu_latpres, plot_rhw_latpres,
    positive_levels, precip_rate_from_accum, region_symmetric_levels,
    relative_frequency_from_histogram, safe_nanpercentile,
    set_latheight_labels, set_latpres_axis,
    simulation_colour, split_scales_warranted, standardize_latlon_dataset,
    symmetric_levels,
    tropopause_on_latitudes, tropopause_region_masks,
    vertical_velocity_from_zm,
)


save = True
save_loc = diagnostic_output_dir('/home/cwp29/tmp/mpas/crf_on/final/mean_state/')

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
figures_to_run = []
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

root_dir = '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/'
data_locs = ['control_new', 'u_nudge', 'T_nudge']
sim_names = ['control', 'nudged u', 'nudged T']

cols = [simulation_colour(name) for name in sim_names]



dss = []
dss_zm = []

latlon_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw",
    "qv", "qc", "qr", "qi", "qs", "qg", "relhum", "w",
    "mslp",
    "uzonal_isobaric", "umeridional_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric", "theta",
    "uReconstructZonal", "uReconstructMeridional",
    "cldfrac", "olrtoa", "zgrid", "pressure", "vorticity", "ke",
]
zonal_mean_vars = [
    "qv_isobaric", "relhum", "rh_isobaric", "t_isobaric", "theta",
    "uReconstructZonal", "uReconstructMeridional", "uzonal_isobaric",
    "umeridional_isobaric", "cldfrac", "pressure", "vorticity", "ke", "w",
]

# Match the native 4-D isobaric chunks spatially. The previous 60x120 chunks
# split every 180x360 HDF5 chunk into nine Dask tasks and made figure 4 spend
# hours repeatedly traversing an unnecessarily large graph.
latlon_chunks = {"Time": 4, "latitude": 180, "longitude": 360}
zm_chunks = {"Time": -1}


for i, d in enumerate(data_locs):
    ds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=LATLON_PATTERN,
        chunks=latlon_chunks
    )
    ds = standardize_latlon_dataset(ds, latlon_vars)

    zds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=ZONAL_MEAN_PATTERN,
        chunks=zm_chunks,
    )
    zds = standardize_latlon_dataset(zds, zonal_mean_vars)

    # cheaper if data are exactly 6-hourly and aligned
    #ds = ds.coarsen(Time=4, boundary="trim").mean()
    #zds = zds.coarsen(Time=4, boundary="trim").mean()

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)

if should_run_figure(1):
    # rainc, rainnc contain accumulated rainfall from simulation start to given timestep
    fig, ax = plt.subplots(2, 1, figsize=(8, 4.5), constrained_layout=True)

    for i, ds in enumerate(dss):
        precip, _, _ = accumulated_precipitation_components(ds)

        precip_mean = lat_mean(precip.mean(dim="longitude"))
        precip_water = lat_mean(ds.precipw.mean(dim="longitude"))

        # Use this dataset's own time coordinate
        t_pw = get_time_values(precip_water)

        # Rainfall rate has one fewer Time point, so use diff() and its own Time coord
        precip_rate = precip_rate_from_accum(precip_mean)

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
    # Deep-tropical-mean potential-temperature and zonal-wind profiles.

    fig, ax = plt.subplots(1, 4, figsize=(8, 4), sharey=True, constrained_layout=True)

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        theta = lat_mean(
            zds.theta.sel(latitude=slice(-10, 10)).mean(dim="Time")
        )
        pressure = lat_mean(
            zds.pressure.sel(latitude=slice(-10, 10)).mean(dim="Time")
        )
        u = lat_mean(
            zds.uReconstructZonal.sel(latitude=slice(-10, 10)).mean(dim="Time")
        )

        T = theta * (pressure / P0_PA) ** (RD / CP)

        z = zds.z.values

        # Brunt-Väisälä frequency squared: N^2 = (g/theta) dtheta/dz
        g = 9.80665
        dtheta_dz = np.gradient(theta.values, z)
        N2 = g / theta.values * dtheta_dz

        print(z, pressure.values)
        ttp, ptp, ztp, _ = tropopause_profile_single(T.values, pressure.values, z)
        print(ttp, ptp, ztp)

        for a in ax:
            a.axhline(ztp/1000., color=cols[i], ls=':')

        ax[0].plot(theta.values, z/1000., color=cols[i], label=sim_names[i])
        ax[1].plot(T.values, z/1000., color=cols[i])
        ax[2].plot(u.values, z/1000., color=cols[i])
        ax[3].plot(N2 * 1e4, z/1000., color=cols[i])

    ax[0].set_xlabel(r'Potential temperature $\theta$ (K)')
    ax[1].set_xlabel(r'Temperature $T$ (K)')
    ax[1].set_xlim(150, 300)
    ax[2].set_xlabel(r'Zonal mean wind $u$ (m s$^{-1}$)')
    ax[3].set_xlabel(r'$N^2$ ($10^{-4}$ s$^{-2}$)')
    ax[3].axvline(0, color='grey', lw=0.5, ls='--')
    ax[0].set_ylabel('Height (km)')

    fig.legend(loc='outside lower center', ncol=len(dss))

    save_figure(fig, "profiles", number=2)
    print("Done with figure: comp_profiles")


# pre-compute CPTs for plotting

CPTs = None
if should_run_any([3, 10, 11, 13, 14, 16, 18]):
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
            print(ztp)

        CPTs.append([ztps, ptps])

    CPTs = np.array(CPTs)



if should_run_figure(3):
    fig, ax = plt.subplots(len(dss), 2, figsize=(8, 3.0*len(dss)), sharey=True, constrained_layout=True)

    if len(dss) == 1:
        ax = np.array([ax])

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        ubar_da = isobaric_lat_mean(zds.uzonal_isobaric, hemispheric_parity="even")
        vbar_da = isobaric_lat_mean(zds.umeridional_isobaric, hemispheric_parity="odd")
        wbar_da = isobaric_vertical_velocity_from_zm(zds)
        Tbar_da = isobaric_lat_mean(zds.t_isobaric, hemispheric_parity="even")
        rhbar_da = isobaric_lat_mean(zds.rh_isobaric, hemispheric_parity="even")

        ubar = ubar_da.values
        vbar = vbar_da.values
        wbar = wbar_da.values
        Tbar = Tbar_da.values
        rhbar = rhbar_da.values

        T_im, _, _, _ = plot_Tu_latpres(ax[i, 0], zds.latitude.values, zds.pres.values, Tbar, ubar)
        rh_im, _ = plot_rhw_latpres(ax[i, 1], zds.latitude.values, zds.pres.values, rhbar, vbar, wbar, Tbar)

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

    for i, ds in enumerate(dss):
        pressure_levels = [100, 250, 700, 850]
        u = ds.uzonal_isobaric.sel(pres=pressure_levels).astype(np.float64)
        v = ds.umeridional_isobaric.sel(pres=pressure_levels).astype(np.float64)
        temperature = ds.t_isobaric.sel(pres=pressure_levels).astype(np.float64)
        water_vapour = ds.qv_isobaric.sel(pres=pressure_levels).astype(np.float64)
        mean_dims = ('longitude', 'Time')
        u_mean = u.mean(dim=mean_dims)
        v_mean = v.mean(dim=mean_dims)
        temperature_mean = temperature.mean(dim=mean_dims)
        water_vapour_mean = water_vapour.mean(dim=mean_dims)

        # Covariance identities let Dask stream all required moments in one
        # pass rather than first forming four full anomaly fields. Float64
        # reductions avoid cancellation in mean(x*y) - mean(x)*mean(y).
        eke = 0.5 * (
            (u*u).mean(dim=mean_dims) - u_mean*u_mean
            + (v*v).mean(dim=mean_dims) - v_mean*v_mean
        )
        memf = (u*v).mean(dim=mean_dims) - u_mean*v_mean
        mehf = (v*temperature).mean(dim=mean_dims) - v_mean*temperature_mean
        mewvf = (v*water_vapour).mean(dim=mean_dims) - v_mean*water_vapour_mean
        eke = hemispheric_fold(eke, parity="even")
        memf = hemispheric_fold(memf, parity="odd")
        mehf = hemispheric_fold(mehf, parity="odd")
        mewvf = hemispheric_fold(mewvf, parity="odd")

        # Compute all four reductions in one shared Dask graph so source chunks
        # fan out to every required moment before being released.
        eke, memf, mehf, mewvf = dask_compute(eke, memf, mehf, mewvf)

        for j, var in enumerate([eke, memf, mehf, mewvf]):
            if j == 0:
                ax[j, 0].plot(ds.latitude.values, var[0], color=cols[i], label=sim_names[i])
            else:
                ax[j, 0].plot(ds.latitude.values, var[0], color=cols[i])
            ax[j, 1].plot(ds.latitude.values, var[1], color=cols[i])
            ax[j, 2].plot(ds.latitude.values, var[2], color=cols[i])
            ax[j, 3].plot(ds.latitude.values, var[3], color=cols[i])

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


imerg_ds = None
imerg_pr = None
if should_run_any([5, 6]):
    imerg_ds = xr.open_mfdataset(list(sorted(glob.glob('/data/phh1-hosted/cwp29/imerg/*3IMERG.*.nc4')))[:300], concat_dim='time', combine='nested', engine="netcdf4")

    imerg_pr = cosine_weighted_latitude_coarsen(
        imerg_ds.precipitation,
        5,
        lat_dim="lat",
    ).coarsen(lon=5, boundary="trim").mean()



DISTRIBUTION_DEEP_TROPICAL_LAT_BOUNDS = (-10.0, 10.0)
DISTRIBUTION_DOMAIN_STYLES = (("Global", "-"), ("10S-10N", "--"))


if should_run_figure(5):
    # Normalised rainfall intensity distribution
    fig, ax = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True, sharey=True)

    imerg_histograms = domain_histograms(
        imerg_pr, bins=300, value_range=(0, 300),
        deep_tropical_lat_bounds=DISTRIBUTION_DEEP_TROPICAL_LAT_BOUNDS,
    )
    for (domain, linestyle), (counts, edges) in zip(DISTRIBUTION_DOMAIN_STYLES, imerg_histograms):
        x, h = relative_frequency_from_histogram(counts, edges, smooth=10)
        ax[0].plot(x, h, color='k', ls=linestyle, label=f'IMERG (Obs), {domain}')
    ax[0].set_yscale('log')

    for s, ds in enumerate(dss):
        precip, precip_con, precip_res = accumulated_precipitation_components(ds)

        for i, precip_var in enumerate([precip, precip_con, precip_res]):
            rate = precip_rate_from_accum(precip_var)
            histograms = domain_histograms(
                rate, bins=300, value_range=(0, 300),
                deep_tropical_lat_bounds=DISTRIBUTION_DEEP_TROPICAL_LAT_BOUNDS,
            )
            for (domain, linestyle), (counts, edges) in zip(DISTRIBUTION_DOMAIN_STYLES, histograms):
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

    imerg_histograms = duration_domain_histograms(
        imerg_pr, threshold=5, bins=59, value_range=(1, 60),
        deep_tropical_lat_bounds=DISTRIBUTION_DEEP_TROPICAL_LAT_BOUNDS,
    )
    for (domain, linestyle), (counts, edges) in zip(DISTRIBUTION_DOMAIN_STYLES, imerg_histograms):
        x, h = relative_frequency_from_histogram(counts, edges, smooth=5)
        ax[0].plot(x, h, color='k', ls=linestyle, label=f'IMERG (Obs), {domain}')
    ax[0].set_yscale('log')

    for s, ds in enumerate(dss):
        precip, precip_con, precip_res = accumulated_precipitation_components(ds)

        for i, precip_var in enumerate([precip, precip_con, precip_res]):
            rate = precip_rate_from_accum(precip_var)
            histograms = duration_domain_histograms(
                rate, threshold=5, bins=59, value_range=(1, 60),
                deep_tropical_lat_bounds=DISTRIBUTION_DEEP_TROPICAL_LAT_BOUNDS,
            )
            for (domain, linestyle), (counts, edges) in zip(DISTRIBUTION_DOMAIN_STYLES, histograms):
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


if imerg_ds is not None:
    imerg_ds.close()
    imerg_ds = None
    imerg_pr = None

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

    for d, ds in enumerate(dss):
        precip, _, _ = accumulated_precipitation_components(ds)
        rate = precip_rate_from_accum(precip)

        # The spectrum only uses 10S-10N. Subset while the field is lazy so
        # load() materialises about one ninth of the former global array.
        pr = rate.sel(
            latitude=slice(*DISTRIBUTION_DEEP_TROPICAL_LAT_BOUNDS)
        ).rename({'latitude':'lat', 'longitude':'lon'})
        pr = cosine_weighted_power_input(pr, lat_dim="lat")
        pr = pr.load()
        print(
            f"Loaded deep-tropical precipitation rate for {sim_names[d]}: "
            f"shape={pr.shape}, dtype={pr.dtype}, size={pr.nbytes / 2**20:.1f} MiB"
        )

        spd = 4

        raw_spectrum = spacetime_power(pr, lataggreg='sum', latitude_bounds=(-10, 10), dosymmetries=True,
                                       segsize=96, noverlap=60, spd=spd)

        sym_smooth_spectrum = smooth_wavefreq(raw_spectrum.sel(component='symmetric'), freq_ax = 1)
        asym_smooth_spectrum = smooth_wavefreq(raw_spectrum.sel(component='antisymmetric'), freq_ax = 1)

        sym_sig_spectrum = raw_spectrum.sel(component='symmetric')/sym_smooth_spectrum
        asym_sig_spectrum = raw_spectrum.sel(component='antisymmetric')/asym_smooth_spectrum

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

    del pr, rate, precip

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
    # Hovmoller diagrams of daily precipitation rate

    fig, ax = plt.subplots(1, len(dss), figsize=(8, 3), constrained_layout=True)

    for d, ds in enumerate(dss):
        precip, _, _ = accumulated_precipitation_components(ds)
        precip = lat_mean(precip.sel(latitude=slice(-10, 10)))

        pr = precip_rate_from_accum(precip)

        hovmoller_im = ax[d].contourf(precip.longitude, pr.Time, pr.values, levels=levels, norm=precip_norm, cmap=precip_cmap, extend='max')

        annotate_panel_label(ax[d], sim_names[d])
        if d > 0:
            ax[d].tick_params(labelleft=False)

    ax[0].set_ylabel("time (days)")

    fig.colorbar(hovmoller_im, ax=ax[-1], location='right', label="Precipitation rate (mm/h)")

    for a in ax:
        a.set_xlabel("Longitude")

    save_figure(fig, "hovmoller", number=8)
    print("Done with figure: comp_hovmoller")

if should_run_figure(9):
    height_anomaly_levels = symmetric_levels(1.0, ndiv=19)
    # Diagnostic: how much does zgrid vary horizontally?

    fig, ax = plt.subplots(len(dss), 2, figsize=(8, 3.2*len(dss)), constrained_layout=True, sharex='col', sharey='row')

    if len(dss) == 1:
        ax = np.array([ax])

    for i, ds in enumerate(dss):
        # zgrid: (nVertLevelsP1, latitude, longitude)
        zgrid = ds.zgrid

        # Midpoint heights from interface heights
        z_lower = zgrid.isel(nVertLevelsP1=slice(0, -1)).values
        z_upper = zgrid.isel(nVertLevelsP1=slice(1, None)).values
        zmid = 0.5 * (z_lower + z_upper)   # (z, latitude, longitude)

        # Horizontal diagnostics
        z_zonal = np.nanmean(zmid, axis=2)            # (z, latitude)
        zmean = lat_mean(
            xr.DataArray(
                z_zonal,
                dims=("z", "latitude"),
                coords={"latitude": ds.latitude},
            )
        ).values                                      # global mean profile
        zmin = np.nanmin(zmid, axis=(1, 2))           # global min at each level
        zmax = np.nanmax(zmid, axis=(1, 2))           # global max at each level
        zstd = np.nanstd(zmid, axis=(1, 2))           # global std at each level

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

        histograms = domain_histograms(
            olr, bins=bins,
            deep_tropical_lat_bounds=DISTRIBUTION_DEEP_TROPICAL_LAT_BOUNDS,
        )
        for (domain, linestyle), (h, b) in zip(DISTRIBUTION_DOMAIN_STYLES, histograms):
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
            zmid_cache[i] = midlevel_height_from_latlon(ds)
        except Exception as e:
            print(f"Skipping scalar constituent height cache for {sim_names[i]}: {e}")

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
        eke_entries.append((i, isobaric_eddy_kinetic_energy(ds).load()))

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
