#!/usr/bin/env python
# coding: utf-8

import os, sys
from functools import partial

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.ndimage import uniform_filter1d
import xarray as xr
import dask
from tropopause import tropopause_profile
from diagnostics.analysis_config import ALL_SIMULATION_LABELS, simulation_cache_name
from diagnostics.diagnostic_helpers import (
    FigureSaver, LATLON_PATTERN, N2_latheight, ZONAL_MEAN_PATTERN,
    add_symmetric_colorbar, annotate_panel_label, control_diff_label,
    diagnostic_cache_dir, diagnostic_output_dir, difference_cmap,
    figure_should_run, figure_should_run_any, finite_or_nan, get_time_values,
    hemispheric_fold, lat_height_mesh,
    load_latlon_dataset, midlevel_height_from_latlon, model_lat_field,
    positive_levels, precip_rate_from_accum, precipitation_rate_from_dataset,
    safe_nanpercentile, set_latheight_labels, simulation_colour,
    standardize_latlon_dataset,
    symmetric_levels, time_step_days, total_precip_accum,
)
from diagnostics.convection.high_cloud_classification import (
    HighCloudThresholds,
    cached_classification_is_current,
)

save = True
save_loc = diagnostic_output_dir('/home/cwp29/tmp/mpas/crf_on/final/convection/')

# Set to [] or None to run all figures.  Otherwise list the numbered figures
# to regenerate, e.g. [18] for the conditional cloud diagnostics.
#
# Figure numbers:
#   1:  diag_symmetry_breaking
#   2:  diag_zonal_wavenumber_variance
#   3:  diag_hovmoller_anomalies
#   4:  diag_N2_latheight
#   5:  diag_zonal_wind_shear_latheight
#   6:  diag_tropopause_profiles
#   7:  moved to refined_high_cloud_classification_diagnostics.py Figure 5
#   8:  moved to refined_high_cloud_classification_diagnostics.py Figure 2
#   9:  diag_convective_distributions
#   10: diag_olr_precip_coupling
#   11: diag_radiative_heating_precip
#   12: diag_deep_tropical_lag_correlations
#   13: comp_zonalmean_convective_state
#   14: comp_hydrometeor_latheight
#   15: superseded by refined high-cloud Figures 4-5
#   16: comp_precip_conditioned_on_pw_cape_cin
#   17: moved to refined high-cloud Figure 6
#   18: rain-matched part moved to refined Figure 6; environment curves omitted
#   19: comp_atmos_radiative_heating_proxy
#   20: comp_latitude_band_timeseries_extended
#   21: diag_meridional_wavenumber_variance
figures_to_run = []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)


def save_figure(number, fig, stem, dpi=300):
    try:
        return figsave.save(fig, stem, number=number, dpi=dpi)
    finally:
        plt.close(fig)


should_run_figure = partial(figure_should_run, figures_to_run)
should_run_any = partial(figure_should_run_any, figures_to_run)


# Avoid giant Dask chunks when selecting latitude/longitude subsets from full lat-lon fields.
dask.config.set({'array.slicing.split_large_chunks': True})

root_dir = '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/'
data_locs = ['control_new', 'u_nudge', 'T_nudge']
simulation_ids = ('control', 'nudged-u', 'nudged-T')
sim_names = [ALL_SIMULATION_LABELS[name] for name in simulation_ids]

CF_THRESH = 0.05
P_HIGH_HPA = 440.0
P_PRECIP_MIN = 12.0  # 0.5 mm h-1 PF threshold (Feng et al. 2021)
classification_cache_dir = diagnostic_cache_dir("classification")

cols = [simulation_colour(name) for name in simulation_ids]



dss = []
dss_zm = []

latlon_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw",
    "uzonal_isobaric", "umeridional_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric", "theta",
    "uReconstructZonal", "cldfrac", "olrtoa", "zgrid", "pressure", "vorticity", "ke",
    # Extra 2-D variables for convective/radiative mechanism diagnostics.
    # These are much cheaper than adding full 4-D qv/rho, so load them by default.
    "cape", "cin", "lcl", "lfc", "lh", "hfx",
    "lwdnt", "lwdnb", "lwupb", "lwupt", "swdnt", "swdnb", "swupb", "swupt",
]

# Figures 14 and 20 need the full model-level hydrometeor fields.  They are not
# present in the zonal-mean files, so retain them from the lat-lon datasets and
# let condensate_latheight fall back to a lazy zonal mean of those fields.
if should_run_any([14, 20]):
    latlon_vars.extend(["qc", "qr", "qi", "qs", "qg"])

zonal_mean_vars = [
    "uzonal_isobaric", "umeridional_isobaric", "rh_isobaric", "t_isobaric",
    "qv_isobaric", "theta", "uReconstructZonal", "cldfrac", "pressure",
    "vorticity", "ke", "w",
]

# Chunk full latitude-longitude fields spatially as well as in time.
# This avoids loading an entire global field when a diagnostic only needs the tropics.
latlon_chunks = {"Time": 8, "latitude": 60, "longitude": 120}
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

# Pre-compute CPTs only for figures that draw them.
CPTs = None
if should_run_any([4, 5, 6, 14]):
    CPTs = []

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        theta = hemispheric_fold(zds.theta.mean(dim="Time"), parity="even")
        pressure = hemispheric_fold(zds.pressure.mean(dim="Time"), parity="even")

        T = theta * (pressure / 100000.)**(287.0 / 1004.0)
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

# -----------------------------------------------------------------------------
# Additional mechanism diagnostics for QBO-like nudging experiments
# -----------------------------------------------------------------------------
# These diagnostics deliberately do *not* assume that the simulations contain an
# propagating intraseasonal mode. They first test whether symmetry is broken,
# then compare UTLS stability, cloud depth, convective thresholds, radiative
# coupling, and simple moisture-convection lag relationships.
#
# Suggested reading of the figures:
#   * Low zonal-asymmetric variance in precip/OLR means there is probably no
#     useful coherent cloud/rain pattern to composite.
#   * A strong T-nudge response in N2, cloud-top height, OLR-per-precipitation,
#     or high-cloud fraction points to the thermal/static-stability pathway.
#   * A strong U-nudge response in zonal asymmetry, k=1-3 power, eddy fluxes, or
#     lag structure points more towards a dynamical/wave-mediated pathway.

def sel_time_like(da, ref):
    """Select exact times from da matching ref.Time; used after precip differencing."""
    return da.sel(Time=ref.Time)

def symmetric_latitude_band(da, latmax=10):
    return da.sel(latitude=slice(-latmax, latmax))

def lat_weights(da):
    return np.cos(np.deg2rad(da.latitude))

def area_mean(da, latmax=None):
    x = symmetric_latitude_band(da, latmax) if latmax is not None else da
    return x.weighted(lat_weights(x)).mean(dim=[d for d in ("latitude", "longitude") if d in x.dims])

def weighted_var(da, dims, weights=None):
    """Weighted variance over dims, keeping NaNs out."""
    if weights is None:
        m = da.mean(dim=dims, skipna=True)
        return ((da - m) ** 2).mean(dim=dims, skipna=True)
    m = da.weighted(weights).mean(dim=dims, skipna=True)
    return ((da - m) ** 2).weighted(weights).mean(dim=dims, skipna=True)

def hemispheric_components(da, lat_dim="latitude"):
    """Return hemispherically symmetric and antisymmetric parts on the original latitude grid."""
    flipped = da.isel({lat_dim: slice(None, None, -1)})
    flipped = flipped.assign_coords({lat_dim: da[lat_dim].values})
    sym = 0.5 * (da + flipped)
    asym = 0.5 * (da - flipped)
    return sym, asym

def fraction_zonal_asymmetry(da, latmax=10):
    """
    Fraction of deep-tropical anomaly variance in k>0 components.

    The anomaly is formed by removing the local time mean.  The k=0 part is the
    longitude mean of this time-varying anomaly; k>0 is the residual.
    """
    x = symmetric_latitude_band(da, latmax=latmax)
    x = x - x.mean(dim="Time", skipna=True)
    xstar = x - x.mean(dim="longitude", skipna=True)
    dims = [d for d in ("Time", "latitude", "longitude") if d in x.dims]
    w = lat_weights(x)
    total = weighted_var(x, dims=dims, weights=w)
    asym = weighted_var(xstar, dims=dims, weights=w)
    return float((asym / total).values)

def fraction_hemispheric_antisymmetry(da):
    """Fraction of anomaly variance in the hemispherically antisymmetric component."""
    x = da - da.mean(dim="Time", skipna=True)
    xs, xa = hemispheric_components(x)
    dims = [d for d in ("Time", "latitude", "longitude") if d in x.dims]
    w = lat_weights(x)
    vs = weighted_var(xs, dims=dims, weights=w)
    va = weighted_var(xa, dims=dims, weights=w)
    return float((va / (vs + va)).values)

_classification_cache = {}


classification_thresholds = partial(
    HighCloudThresholds,
    cf_thresh=CF_THRESH,
    high_cloud_top_pressure_hpa=P_HIGH_HPA,
    surface_precip_min_mm_day=P_PRECIP_MIN,
)


def shared_high_cloud_classification(ds):
    """Load the mandatory shared classification; never rebuild or fall back."""
    try:
        index = next(i for i, candidate in enumerate(dss) if candidate is ds)
    except StopIteration as exc:
        raise ValueError("Dataset is not one of the configured simulations") from exc
    if index in _classification_cache:
        return _classification_cache[index]

    path = os.path.join(
        classification_cache_dir,
        f"{simulation_cache_name(simulation_ids[index])}_classified_high_cloud.nc",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{sim_names[index]}: required shared high-cloud classification is missing: {path}. "
            "Run refined_high_cloud_classification_diagnostics.py first."
        )
    cached = xr.open_dataset(path, chunks={})
    if not cached_classification_is_current(cached, classification_thresholds()):
        cached.close()
        raise RuntimeError(f"{sim_names[index]}: stale high-cloud classification cache: {path}")
    required = (
        "cloud_top_pressure",
        "cloud_top_height",
        "high_cloud_fraction",
        "high_cloud_mask",
    )
    missing = [name for name in required if name not in cached.data_vars]
    if missing:
        cached.close()
        raise RuntimeError(f"{sim_names[index]}: shared classification lacks {missing}: {path}")
    _classification_cache[index] = cached
    print(f"Using shared high-cloud classification for {sim_names[index]}: {path}")
    return cached


def high_cloud_fraction(ds):
    return finite_or_nan(shared_high_cloud_classification(ds).high_cloud_fraction)


def cloud_top_height(ds):
    """Cloud-top height in km from the mandatory shared classification."""
    return finite_or_nan(shared_high_cloud_classification(ds).cloud_top_height)

def sample_flatten(*arrays, max_points=300000):
    """Flatten matching DataArrays/arrays and take a deterministic stride sample."""
    vals = [np.asarray(a).ravel() for a in arrays]
    mask = np.ones(vals[0].shape, dtype=bool)
    for v in vals:
        mask &= np.isfinite(v)
    vals = [v[mask] for v in vals]
    if vals[0].size > max_points:
        step = int(np.ceil(vals[0].size / max_points))
        vals = [v[::step] for v in vals]
    return vals

def binned_mean(x, y, bins, min_count=20):
    """Mean y in bins of x."""
    x, y = sample_flatten(x, y, max_points=1000000)
    ind = np.digitize(x, bins) - 1
    centres = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(len(centres), np.nan)
    count = np.zeros(len(centres), dtype=int)
    for i in range(len(centres)):
        m = ind == i
        count[i] = np.count_nonzero(m)
        if count[i] >= min_count:
            mean[i] = np.nanmean(y[m])
    return centres, mean, count

def lag_corr(x, y, max_lag_steps):
    """
    Correlation between x(t) and y(t+lag).

    Positive lag therefore means x leads y.  This is useful for asking whether
    column moisture leads rainfall/OLR changes.
    """
    x = np.asarray(x)
    y = np.asarray(y)
    out = []
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    for lag in lags:
        if lag < 0:
            xs, ys = x[-lag:], y[:lag]
        elif lag > 0:
            xs, ys = x[:-lag], y[lag:]
        else:
            xs, ys = x, y
        mask = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(mask) < 3:
            out.append(np.nan)
        else:
            out.append(np.corrcoef(xs[mask], ys[mask])[0, 1])
    return lags, np.asarray(out)

def net_atmospheric_radiation(ds):
    """
    Net atmospheric radiative heating proxy in W m-2.

    Positive means radiative energy convergence into the atmospheric column:
        net_down_TOA - net_down_surface
    using all-sky SW and LW fluxes.
    """
    net_toa = (ds.swdnt - ds.swupt) + (ds.lwdnt - ds.lwupt)
    net_sfc = (ds.swdnb - ds.swupb) + (ds.lwdnb - ds.lwupb)
    ratm = net_toa - net_sfc
    ratm.name = "net_atmospheric_radiation"
    ratm.attrs["units"] = "W m-2"
    return ratm

def time_aligned_fields(ds):
    """Common 2-D fields aligned to precipitation-rate time."""
    pr = precipitation_rate_from_dataset(ds)
    fields = {
        "Precip": pr,
        "OLR": sel_time_like(ds.olrtoa, pr),
        "PW": sel_time_like(ds.precipw, pr),
        "CAPE": sel_time_like(ds.cape, pr),
        "CIN": sel_time_like(ds.cin, pr),
        "High cloud": sel_time_like(high_cloud_fraction(ds), pr),
    }
    # Do not add R_atm here: it is a compound field made from eight fluxes,
    # so constructing it repeatedly inside generic diagnostics can create large
    # Dask graphs.  The dedicated radiative diagnostic below computes it once
    # and samples it conservatively.
    return fields

def strided_matching_samples(*arrays, latmax=30, max_time=24, max_lat=80, max_lon=160):
    """
    Take the same coarse, deterministic stride sample from matching lat-lon-time fields.

    This is deliberately applied before calling .values, so histograms and binned
    relationships do not materialise a full compound Dask array.
    """
    xs = [symmetric_latitude_band(a, latmax=latmax) for a in arrays]
    indexers = {}
    ref = xs[0]
    limits = {"Time": max_time, "latitude": max_lat, "longitude": max_lon}
    for dim, max_n in limits.items():
        if dim in ref.dims and ref.sizes[dim] > max_n:
            step = int(np.ceil(ref.sizes[dim] / max_n))
            indexers[dim] = slice(0, None, step)
    vals = [np.asarray(x.isel(indexers).values).ravel() for x in xs]
    mask = np.ones(vals[0].shape, dtype=bool)
    for v in vals:
        mask &= np.isfinite(v)
    return [v[mask] for v in vals]

# -----------------------------------------------------------------------------
# Additional helpers preserved from zm_latlon_misc_augmented.py
# -----------------------------------------------------------------------------
# These support the non-duplicated mechanism diagnostics appended below.

g = 9.80665

def deep_tropical_mean_latlon(da, lat_bounds=(-10, 10)):
    """Area-weighted latitude mean and arithmetic longitude mean."""
    da = finite_or_nan(da).sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
    w = np.cos(np.deg2rad(da.latitude))
    return da.weighted(w).mean(dim="latitude").mean(dim="longitude")

def binned_statistic_1d(x, y, bins, min_count=20):
    """Return bin centres, mean, interquartile range and counts."""
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x = x[ok]
    y = y[ok]

    inds = np.digitize(x, bins) - 1
    xc = 0.5 * (bins[1:] + bins[:-1])
    mean = np.full(len(xc), np.nan)
    p25 = np.full(len(xc), np.nan)
    p75 = np.full(len(xc), np.nan)
    count = np.zeros(len(xc), dtype=int)

    for k in range(len(xc)):
        vals = y[inds == k]
        count[k] = vals.size
        if vals.size >= min_count:
            mean[k] = np.nanmean(vals)
            p25[k] = np.nanpercentile(vals, 25)
            p75[k] = np.nanpercentile(vals, 75)

    return xc, mean, p25, p75, count

def add_cpt_height(ax, i):
    ax.plot(dss_zm[i].latitude.values, CPTs[i][0] / 1000.0, color="k", lw=1, ls=":")

def condensate_latheight(ds, zds, varlist):
    """Return time-mean zonal-mean condensate in g/kg as (z, latitude)."""
    available_zm = [v for v in varlist if v in zds.data_vars]
    available_ll = [v for v in varlist if v in ds.data_vars]

    if len(available_zm) > 0:
        q = sum(finite_or_nan(zds[v]) for v in available_zm).mean(dim="Time") * 1000.0
        q = hemispheric_fold(q, parity="even")
        zmid = zds.z
        return lat_height_mesh(q, zmid, zds.latitude.values)

    if len(available_ll) > 0:
        q = sum(finite_or_nan(ds[v]) for v in available_ll).mean(dim=("Time", "longitude")) * 1000.0
        q = model_lat_field(q)
        q = hemispheric_fold(q, parity="even")
        zmid = midlevel_height_from_latlon(ds)
        return lat_height_mesh(q, zmid, ds.latitude.values)

    raise ValueError(f"None of {varlist} found")

def atmospheric_radiative_heating_proxy(ds):
    """
    Approximate all-sky atmospheric radiative heating proxy:
      net TOA minus net surface radiation.

    Check signs carefully before publication; this is intended as a relative
    diagnostic across the simulations.
    """
    required = ["swdnt", "swupt", "swdnb", "swupb", "lwdnt", "lwupt", "lwdnb", "lwupb"]
    missing = [v for v in required if v not in ds.data_vars]
    if missing:
        raise ValueError(f"Missing radiation variables: {missing}")

    net_toa_lw = ds.lwdnt - ds.lwupt
    net_sfc_lw = ds.lwdnb - ds.lwupb
    atm_lw = net_toa_lw - net_sfc_lw

    net_toa_sw = ds.swdnt - ds.swupt
    net_sfc_sw = ds.swdnb - ds.swupb
    atm_sw = net_toa_sw - net_sfc_sw

    return atm_lw, atm_sw, atm_lw + atm_sw

# In[ ]:

# 1. Symmetry-breaking diagnostics.
# What to look for:
#   * If deep-tropical k>0 variance is very small, do not use
#     propagation-phase composites.
#   * If nudging increases k>0 or antisymmetric variance, it is changing wave or
#     instability behaviour rather than just the zonal-mean convective state.


if should_run_figure(1):
    symmetry_rows = []
    for i, ds in enumerate(dss):
        fields = time_aligned_fields(ds)
        test_fields = {}
        for key in ("Precip", "OLR", "PW", "High cloud"):
            if key in fields:
                test_fields[key] = fields[key]
        if "uzonal_isobaric" in ds:
            try:
                u850 = ds.uzonal_isobaric.sel(pres=850)
                test_fields["u850"] = sel_time_like(u850, fields["Precip"])
            except Exception:
                pass
        for name, da in test_fields.items():
            try:
                symmetry_rows.append({
                    "simulation": sim_names[i],
                    "variable": name,
                    "zonal_asym_fraction_10S10N": fraction_zonal_asymmetry(da, latmax=10),
                    "hemispheric_antisym_fraction_global": fraction_hemispheric_antisymmetry(da),
                })
            except Exception as e:
                print(f"Skipping symmetry metric for {sim_names[i]} {name}: {e}")

    symmetry_df = pd.DataFrame(symmetry_rows)
    print(symmetry_df)

    if len(symmetry_df) > 0:
        variables = list(symmetry_df.variable.unique())
        x = np.arange(len(variables))
        width = 0.8 / len(dss)

        fig, ax = plt.subplots(2, 1, figsize=(9, 5.5), constrained_layout=True, sharex=True)
        for i, name in enumerate(sim_names):
            sub = symmetry_df[symmetry_df.simulation == name].set_index("variable").reindex(variables)
            ax[0].bar(x + (i - (len(dss)-1)/2)*width, sub.zonal_asym_fraction_10S10N, width, color=cols[i], label=name)
            ax[1].bar(x + (i - (len(dss)-1)/2)*width, sub.hemispheric_antisym_fraction_global, width, color=cols[i])

        ax[0].set_ylabel(r"Deep-tropical $k>0$ variance fraction")
        ax[1].set_ylabel("Global antisymmetric variance fraction")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(variables)
        ax[0].legend(loc="best")
        ax[0].set_title("Symmetry-breaking diagnostics")

        save_figure(1, fig, "symmetry_breaking")
        print("Done with figure: diag_symmetry_breaking")
else:
    symmetry_df = pd.DataFrame()

# In[ ]:

# 2. Zonal wavenumber variance fractions of deep-tropical-mean anomalies.
# What to look for:
#   * k=0 dominance: mostly zonally symmetric variability.
#   * enhanced k=1-3 precip/OLR/PW variance: a necessary, but not sufficient,
#     condition for planetary-scale organisation.

def deep_tropical_zonal_series(da, latmax=10):
    x = symmetric_latitude_band(da, latmax=latmax)
    return x.weighted(lat_weights(x)).mean(dim="latitude")

def deep_tropical_meridional_series(da, latmax=10):
    x = symmetric_latitude_band(da, latmax=latmax)
    return x.mean(dim="longitude", skipna=True)

def spatial_wavenumber_variance_fraction(x, spatial_dim, max_k=15):
    x = x - x.mean(dim="Time", skipna=True)
    x = x.transpose("Time", spatial_dim)
    arr = np.asarray(x.values)
    if arr.ndim != 2:
        raise ValueError(f"Expected (Time, {spatial_dim}), got shape {arr.shape}")
    # Fill occasional NaNs with the spatial mean at that time before FFT.
    if np.any(~np.isfinite(arr)):
        row_mean = np.nanmean(arr, axis=1, keepdims=True)
        arr = np.where(np.isfinite(arr), arr, row_mean)
    fft = np.fft.rfft(arr, axis=1)
    power = np.abs(fft) ** 2
    if power.shape[1] > 2:
        power[:, 1:-1] *= 2.0
    pk = np.nanmean(power, axis=0)
    pk = pk / np.nansum(pk)
    kk = np.arange(pk.size)
    keep = kk <= max_k
    return kk[keep], pk[keep]

def zonal_wavenumber_variance_fraction(da, latmax=10, max_k=15):
    x = deep_tropical_zonal_series(da, latmax=latmax)
    return spatial_wavenumber_variance_fraction(x, "longitude", max_k=max_k)

def meridional_wavenumber_variance_fraction(da, latmax=10, max_k=15):
    x = deep_tropical_meridional_series(da, latmax=latmax)
    return spatial_wavenumber_variance_fraction(x, "latitude", max_k=max_k)

if should_run_figure(2):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True, sharey=True)
    plot_vars = [("Precip", None), ("OLR", "olrtoa"), ("PW", "precipw")]
    for j, (label, vname) in enumerate(plot_vars):
        for i, ds in enumerate(dss):
            try:
                if label == "Precip":
                    da = precipitation_rate_from_dataset(ds)
                else:
                    da = ds[vname]
                k, frac = zonal_wavenumber_variance_fraction(da, latmax=10, max_k=15)
                ax[j].plot(k, frac, marker="o", color=cols[i], label=sim_names[i] if j == 0 else None)
            except Exception as e:
                print(f"Skipping zonal wavenumber plot for {sim_names[i]} {label}: {e}")
        ax[j].set_title(label)
        ax[j].set_xlabel("Zonal wavenumber")
        ax[j].set_xlim(0, 15)
        ax[j].set_yscale("log")
    ax[0].set_ylabel("Fraction of deep-tropical anomaly variance")
    ax[0].legend(loc="best")

    save_figure(2, fig, "zonal_wavenumber_variance")
    print("Done with figure: diag_zonal_wavenumber_variance")

# In[ ]:

# 21. Meridional wavenumber fractions of deep-tropical zonal-mean anomalies.
# This repeats the figure-2 variance spectrum along latitude rather than longitude.
# What to look for:
#   * Low meridional-wavenumber dominance indicates broad meridional structure.
#   * enhanced higher meridional wavenumbers: narrower or more latitudinally
#     structured precip/OLR/PW anomalies.

if should_run_figure(21):
    fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True, sharey=True)
    plot_vars = [("Precip", None), ("OLR", "olrtoa"), ("PW", "precipw")]
    for j, (label, vname) in enumerate(plot_vars):
        for i, ds in enumerate(dss):
            try:
                if label == "Precip":
                    da = precipitation_rate_from_dataset(ds)
                else:
                    da = ds[vname]
                k, frac = meridional_wavenumber_variance_fraction(da, latmax=10, max_k=15)
                ax[j].plot(k, frac, marker="o", color=cols[i], label=sim_names[i] if j == 0 else None)
            except Exception as e:
                print(f"Skipping meridional wavenumber plot for {sim_names[i]} {label}: {e}")
        ax[j].set_title(label)
        ax[j].set_xlabel("Meridional wavenumber")
        ax[j].set_xlim(0, 15)
        ax[j].set_yscale("log")
    ax[0].set_ylabel("Fraction of deep-tropical anomaly variance")
    ax[0].legend(loc="best")

    save_figure(21, fig, "meridional_wavenumber_variance")
    print("Done with figure: diag_meridional_wavenumber_variance")

# In[ ]:

# 3. Equatorial Hovmoller anomalies for precipitation, OLR and precipitable water.
# These are better than raw fields for checking whether there is any coherent
# propagating cloud/rain envelope.

if should_run_figure(3):
    hov_vars = [("Precip", None, "mm day$^{-1}$"), ("OLR", "olrtoa", "W m$^{-2}$"), ("PW", "precipw", "kg m$^{-2}$")]
    fig, ax = plt.subplots(len(hov_vars), len(dss), figsize=(3.3*len(dss), 7.5), constrained_layout=True, sharex=True)
    if len(dss) == 1:
        ax = ax[:, None]

    for j, (label, vname, units) in enumerate(hov_vars):
        vmax = None
        fields = []
        for ds in dss:
            if label == "Precip":
                da = precipitation_rate_from_dataset(ds)
            else:
                da = ds[vname]
            x = deep_tropical_zonal_series(da, latmax=10)
            # Remove time mean at each longitude and instantaneous zonal mean.
            x = x - x.mean(dim="Time", skipna=True)
            x = x - x.mean(dim="longitude", skipna=True)
            fields.append(x)
            vv = np.nanpercentile(np.abs(x.values), 98)
            vmax = vv if vmax is None else max(vmax, vv)
        if vmax == 0 or not np.isfinite(vmax):
            vmax = 1.0
        levels_hov = symmetric_levels(vmax, ndiv=19)
        for i, x in enumerate(fields):
            im = ax[j, i].contourf(x.longitude, x.Time, x.values, levels=levels_hov, cmap=difference_cmap(levels_hov), extend="both")
            annotate_panel_label(ax[j, i], sim_names[i])
            if i == 0:
                ax[j, i].set_ylabel(f"{label}\ntime")
            else:
                ax[j, i].tick_params(labelleft=False)
            if j == len(hov_vars)-1:
                ax[j, i].set_xlabel("Longitude")
        add_symmetric_colorbar(fig, im, ax[j, :], levels_hov, label=f"{label}' ({units})")

    save_figure(3, fig, "hovmoller_anomalies")
    print("Done with figure: diag_hovmoller_anomalies")

# In[ ]:

# 4. Latitude-height static stability N^2.
# What to look for:
#   * Does T nudging reduce/increase N2 near the tropical tropopause?
#   * Does U nudging induce a secondary N2 response despite only nudging wind?

if should_run_figure(4):
    fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.3*len(dss)), constrained_layout=True, sharex=True)
    fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.3*len(dss)), constrained_layout=True, sharex=True)
    if len(dss) == 1:
        ax = [ax]
        ax2 = [ax2]

    control_N2 = None
    im = im2 = im3 = None
    for i, zds in enumerate(dss_zm):
        N2 = N2_latheight(zds)
        zmid = zds.z
        field, lat2d, z2d = lat_height_mesh(N2 * 1e4, zmid, zds.latitude.values)
        im = ax[i].contourf(lat2d, z2d/1000., field.values, levels=np.linspace(0, 8, 25), cmap="YlGnBu", extend="max")
        ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color="k", lw=1, ls=":")
        annotate_panel_label(ax[i], sim_names[i])
        set_latheight_labels(ax[i], xlabel=(i == len(dss)-1))
        if i < len(dss)-1:
            ax[i].tick_params(labelbottom=False)

        if i == 0:
            control_N2 = N2
            im2 = ax2[i].contourf(lat2d, z2d/1000., field.values, levels=np.linspace(0, 8, 25), cmap="YlGnBu", extend="max")
        else:
            cN2 = control_N2.interp(z=N2.z, latitude=N2.latitude)
            diff = (N2 - cN2) * 1e4
            n2_diff_levels = symmetric_levels(1.0, ndiv=12)
            im3 = ax2[i].contourf(lat2d, z2d/1000., diff.values, levels=n2_diff_levels, cmap=difference_cmap(n2_diff_levels), extend="both")
        ax2[i].plot(zds.latitude.values, CPTs[i][0]/1000., color="k", lw=1, ls=":")
        annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
        set_latheight_labels(ax2[i], xlabel=(i == len(dss)-1))
        if i < len(dss)-1:
            ax2[i].tick_params(labelbottom=False)

    fig.colorbar(im, ax=ax, location="right", label=r"$N^2$ ($10^{-4}$ s$^{-2}$)")
    fig2.colorbar(im2, ax=ax2[0], location="right", label=r"$N^2$ ($10^{-4}$ s$^{-2}$)")
    if im3 is not None:
        add_symmetric_colorbar(fig2, im3, ax2[1:], symmetric_levels(1.0, ndiv=12), label=r"$\Delta N^2$ ($10^{-4}$ s$^{-2}$)")

    save_figure(4, fig, "N2_latheight")
    save_figure(4, fig2, "N2_latheight_diff")
    print("Done with figure: diag_N2_latheight")

# In[ ]:

# 4b. Latitude-height zonal wind shear.
# What to look for:
#   * Does U nudging directly sharpen or weaken tropical shear layers?
#   * Does T nudging induce a secondary shear response consistent with
#     thermal-wind adjustment?

def zonal_wind_shear_latheight(zds):
    u = model_lat_field(zds.uReconstructZonal.mean(dim="Time"))
    u = hemispheric_fold(u, parity="even")
    z = zds.z.values
    du_dz = np.gradient(u.values, z, axis=0)
    shear = xr.DataArray(
        du_dz,
        dims=u.dims,
        coords=u.coords,
        name="du_dz",
        attrs={"units": "s-1"},
    )
    return hemispheric_fold(shear, parity="even")

if should_run_figure(5):
    if all("uReconstructZonal" in zds.data_vars for zds in dss_zm):
        fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.3*len(dss)), constrained_layout=True, sharex=True)
        fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.3*len(dss)), constrained_layout=True, sharex=True)
        if len(dss) == 1:
            ax = [ax]
            ax2 = [ax2]

        control_shear = None
        im = im2 = im3 = None
        shear_levels = symmetric_levels(8.0, ndiv=16)
        shear_diff_levels = symmetric_levels(4.0, ndiv=16)

        for i, zds in enumerate(dss_zm):
            shear = zonal_wind_shear_latheight(zds)
            zmid = zds.z
            field, lat2d, z2d = lat_height_mesh(shear * 1000.0, zmid, zds.latitude.values)

            im = ax[i].contourf(
                lat2d,
                z2d / 1000.0,
                field.values,
                levels=shear_levels,
                cmap=difference_cmap(shear_levels),
                extend="both",
            )
            ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color="k", lw=1, ls=":")
            annotate_panel_label(ax[i], sim_names[i])
            set_latheight_labels(ax[i], xlabel=(i == len(dss)-1))
            if i < len(dss)-1:
                ax[i].tick_params(labelbottom=False)

            if i == 0:
                control_shear = shear
                im2 = ax2[i].contourf(
                    lat2d,
                    z2d / 1000.0,
                    field.values,
                    levels=shear_levels,
                    cmap=difference_cmap(shear_levels),
                    extend="both",
                )
            else:
                cshear = control_shear.interp(z=shear.z, latitude=shear.latitude)
                diff = (shear - cshear) * 1000.0
                im3 = ax2[i].contourf(lat2d, z2d/1000., diff.values, levels=shear_diff_levels, cmap=difference_cmap(shear_diff_levels), extend="both")
            ax2[i].plot(zds.latitude.values, CPTs[i][0]/1000., color="k", lw=1, ls=":")
            annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
            set_latheight_labels(ax2[i], xlabel=(i == len(dss)-1))
            if i < len(dss)-1:
                ax2[i].tick_params(labelbottom=False)

        add_symmetric_colorbar(fig, im, ax, shear_levels, label=r"$\partial u / \partial z$ (m s$^{-1}$ km$^{-1}$)")
        add_symmetric_colorbar(fig2, im2, ax2[0], shear_levels, label=r"$\partial u / \partial z$ (m s$^{-1}$ km$^{-1}$)")
        if im3 is not None:
            add_symmetric_colorbar(fig2, im3, ax2[1:], shear_diff_levels, label=r"$\Delta \partial u / \partial z$ (m s$^{-1}$ km$^{-1}$)")

        save_figure(5, fig, "zonal_wind_shear_latheight")
        save_figure(5, fig2, "zonal_wind_shear_latheight_diff")
        print("Done with figure: diag_zonal_wind_shear_latheight")
    else:
        print("Skipping zonal wind shear latitude-height plots: uReconstructZonal not present in all zm datasets")

# In[ ]:

# 5. Tropopause height and pressure profiles, plus nudged-control differences.
# What to look for:
#   * Tropical tropopause height/pressure shifts are a compact measure of the
#     UTLS thermal response.
#   * A T-nudged shift without a similar U-nudged shift points to a direct
#     thermal pathway.

if should_run_figure(6):
    fig, ax = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True, sharex=True)
    for i, zds in enumerate(dss_zm):
        lat = zds.latitude.values
        ztp_km = CPTs[i][0] / 1000.0
        ptp_hpa = CPTs[i][1] / 100.0
        ax[0, 0].plot(lat, ztp_km, color=cols[i], label=sim_names[i])
        ax[0, 1].plot(lat, ptp_hpa, color=cols[i])
        if i > 0:
            cztp = np.interp(lat, dss_zm[0].latitude.values, CPTs[0][0] / 1000.0)
            cptp = np.interp(lat, dss_zm[0].latitude.values, CPTs[0][1] / 100.0)
            ax[1, 0].plot(lat, ztp_km - cztp, color=cols[i], label=sim_names[i] + " - control")
            ax[1, 1].plot(lat, ptp_hpa - cptp, color=cols[i])

    ax[0, 0].set_ylabel("Tropopause height (km)")
    ax[0, 1].set_ylabel("Tropopause pressure (hPa)")
    ax[1, 0].set_ylabel(r"$\Delta$ height (km)")
    ax[1, 1].set_ylabel(r"$\Delta$ pressure (hPa)")
    for a in ax.ravel():
        a.set_xlim(-90, 90)
        a.axvline(0, color="0.7", lw=0.5)
    ax[1, 0].axhline(0, color="0.7", lw=0.5)
    ax[1, 1].axhline(0, color="0.7", lw=0.5)
    ax[1, 0].set_xlabel("Latitude")
    ax[1, 1].set_xlabel("Latitude")
    ax[0, 0].legend(loc="best")
    ax[1, 0].legend(loc="best")
    ax[0, 1].invert_yaxis()

    save_figure(6, fig, "tropopause_profiles")
    print("Done with figure: diag_tropopause_profiles")

# High-cloud spatial responses and cloud-top distributions now use the shared
# class decomposition in refined_high_cloud_classification_diagnostics.py
# Figures 5 and 2, respectively.

# In[ ]:

# 7. Convective distributions: rainfall, OLR, PW, CAPE, CIN.
# What to look for:
#   * A rainfall tail change without a PW/CAPE shift suggests cloud-depth or
#     stability changes rather than just moisture availability.
#   * T nudging that changes OLR/PW/high-cloud distributions more than rainfall
#     is especially suggestive of a radiative/cloud pathway.

if should_run_figure(9):
    hist_specs = [
        ("Precip", precipitation_rate_from_dataset, np.linspace(0, 80, 121), "Rainfall rate (mm day$^{-1}$)", True),
        ("OLR", lambda ds: ds.olrtoa, np.linspace(100, 350, 121), r"OLR (W m$^{-2}$)", False),
        ("PW", lambda ds: ds.precipw, np.linspace(0, 80, 121), r"Precipitable water (kg m$^{-2}$)", False),
        ("CAPE", lambda ds: ds.cape if "cape" in ds else None, np.linspace(0, 5000, 121), r"CAPE (J kg$^{-1}$)", False),
        ("CIN", lambda ds: ds.cin if "cin" in ds else None, np.linspace(-500, 50, 121), r"CIN (J kg$^{-1}$)", False),
    ]

    fig, ax = plt.subplots(1, len(hist_specs), figsize=(3.1*len(hist_specs), 3.4), constrained_layout=True)
    for j, (name, func, bins, xlabel, logy) in enumerate(hist_specs):
        any_plotted = False
        for i, ds in enumerate(dss):
            da = func(ds)
            if da is None:
                continue
            vals = symmetric_latitude_band(da, latmax=30).values.ravel()
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                continue
            h, b = np.histogram(vals, bins=bins, density=True)
            h = uniform_filter1d(h, size=3)
            ax[j].plot(0.5*(b[1:]+b[:-1]), h, color=cols[i], label=sim_names[i] if j == 0 else None)
            any_plotted = True
        ax[j].set_title(name)
        ax[j].set_xlabel(xlabel)
        if logy:
            ax[j].set_yscale("log")
        if not any_plotted:
            ax[j].text(0.5, 0.5, "not loaded", transform=ax[j].transAxes, ha="center", va="center")
    ax[0].set_ylabel("Probability density")
    ax[0].legend(loc="best")

    save_figure(9, fig, "convective_distributions")
    print("Done with figure: diag_convective_distributions")

# In[ ]:

# 9. OLR-precipitation coupling / greenhouse-enhancement-like proxy.
# This is not a formal clear-sky CRF decomposition, but the slope tells you how
# much OLR decreases per unit rainfall anomaly.
# What to look for:
#   * More negative slope in T-nudged run: deeper/more radiatively active clouds
#     per unit precipitation.
#   * Similar slope but shifted precipitation distribution: mostly amplitude/state change.

if should_run_figure(10):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.8), constrained_layout=True)
    slope_rows = []
    for i, ds in enumerate(dss):
        fields = time_aligned_fields(ds)
        if "OLR" not in fields:
            continue
        pr = symmetric_latitude_band(fields["Precip"], latmax=30)
        olr = symmetric_latitude_band(fields["OLR"], latmax=30)

        pr_anom = pr - pr.mean(dim="Time", skipna=True)
        olr_anom = olr - olr.mean(dim="Time", skipna=True)
        x, y = sample_flatten(pr_anom.values, olr_anom.values, max_points=150000)
        if x.size < 10:
            continue
        slope, intercept = np.polyfit(x, y, 1)
        slope_rows.append({"simulation": sim_names[i], "dOLR_dP": slope})

        hb = ax[0].hexbin(x, y, gridsize=60, mincnt=1, linewidths=0.0)
        xs = np.linspace(np.nanpercentile(x, 1), np.nanpercentile(x, 99), 100)
        ax[0].plot(xs, slope*xs + intercept, color=cols[i], lw=2, label=f"{sim_names[i]}: {slope:.2f}")

        centres, mean, count = binned_mean(pr_anom.values, olr_anom.values, bins=np.linspace(-30, 60, 46), min_count=50)
        ax[1].plot(centres, mean, color=cols[i], marker="o", ms=3, label=sim_names[i])

    ax[0].axhline(0, color="0.7", lw=0.5)
    ax[0].axvline(0, color="0.7", lw=0.5)
    ax[0].set_xlabel(r"Precipitation anomaly (mm day$^{-1}$)")
    ax[0].set_ylabel(r"OLR anomaly (W m$^{-2}$)")
    ax[0].legend(loc="best", title=r"slope $dOLR'/dP'$")
    ax[1].axhline(0, color="0.7", lw=0.5)
    ax[1].axvline(0, color="0.7", lw=0.5)
    ax[1].set_xlabel(r"Precipitation anomaly (mm day$^{-1}$)")
    ax[1].set_ylabel(r"Binned mean OLR anomaly (W m$^{-2}$)")
    ax[1].legend(loc="best")

    slope_df = pd.DataFrame(slope_rows)
    print(slope_df)
    save_figure(10, fig, "olr_precip_coupling")
    print("Done with figure: diag_olr_precip_coupling")

# In[ ]:

# 10. Atmospheric-column radiative heating proxy and coupling to precipitation.
# What to look for:
#   * A systematic R_atm shift suggests the nudging changes column radiative
#     energy input.
#   * A changed R_atm(precip) curve suggests cloud-radiative feedback changes.
#
# Memory note:
#   R_atm is built from eight full latitude-longitude flux fields.  Do not call
#   .values on the full 30S-30N field.  The histogram and curve below use
#   a deterministic stride sample before loading data into memory.

if should_run_figure(11):
    rad_needed = ["swdnt", "swupt", "lwdnt", "lwupt", "swdnb", "swupb", "lwdnb", "lwupb"]
    have_radiation = any(all(v in ds.data_vars for v in rad_needed) for ds in dss)

    if have_radiation:
        fig, ax = plt.subplots(1, 3, figsize=(11, 3.5), constrained_layout=True)
        for i, ds in enumerate(dss):
            if not all(v in ds.data_vars for v in rad_needed):
                continue

            ratm = net_atmospheric_radiation(ds)
            pr = precipitation_rate_from_dataset(ds)
            ratm_pr = sel_time_like(ratm, pr)

            # This is small after area_mean, so loading it is safe.
            tmean = area_mean(ratm, latmax=30).load()
            ax[0].plot(get_time_values(tmean), tmean.values, color=cols[i], label=sim_names[i])

            # Histogram: stride before .values.
            vals, = strided_matching_samples(ratm, latmax=30, max_time=32, max_lat=80, max_lon=160)
            if vals.size > 0:
                h, b = np.histogram(vals, bins=np.linspace(-300, 300, 121), density=True)
                h = uniform_filter1d(h, size=3)
                ax[1].plot(0.5*(b[1:]+b[:-1]), h, color=cols[i])

            # Conditional curve: sample precipitation and R_atm on the same sparse grid.
            pr_vals, ratm_vals = strided_matching_samples(
                pr, ratm_pr, latmax=30, max_time=32, max_lat=80, max_lon=160
            )
            if pr_vals.size > 0:
                centres, mean, count = binned_mean(
                    pr_vals, ratm_vals, bins=np.linspace(0, 80, 41), min_count=20
                )
                ax[2].plot(centres, mean, color=cols[i])

            # Encourage Dask/matplotlib to release intermediate references between simulations.
            del ratm, pr, ratm_pr

        ax[0].set_xlabel("Time")
        ax[0].set_ylabel(r"30S-30N mean $R_{atm}$ (W m$^{-2}$)")
        ax[0].legend(loc="best")
        ax[1].set_xlabel(r"$R_{atm}$ (W m$^{-2}$)")
        ax[1].set_ylabel("Probability density")
        ax[2].set_xlabel(r"Precipitation (mm day$^{-1}$)")
        ax[2].set_ylabel(r"Mean $R_{atm}$ (W m$^{-2}$)")
        save_figure(11, fig, "radiative_heating_precip")
        print("Done with figure: diag_radiative_heating_precip")
    else:
        print("Skipping radiative-heating diagnostics: radiation flux variables not loaded")

# In[ ]:

# 11. Moisture-convection lag correlations using deep-tropical means.
# What to look for:
#   * PW leading precipitation is a recharge-discharge signature.
#   * If nudging changes the lag of maximum correlation, it changes the timing of
#     moistening relative to convection even without a coherent envelope.

if should_run_figure(12):
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5), constrained_layout=True, sharey=True)
    for i, ds in enumerate(dss):
        fields = time_aligned_fields(ds)
        if "PW" not in fields or "OLR" not in fields:
            continue
        pr_ts = area_mean(fields["Precip"], latmax=10)
        pw_ts = area_mean(fields["PW"], latmax=10)
        olr_ts = area_mean(fields["OLR"], latmax=10)

        dt_days_here = time_step_days(fields["Precip"])
        max_steps = max(1, int(round(10.0 / dt_days_here)))
        lags, c_pr = lag_corr(pw_ts.values, pr_ts.values, max_steps)
        _, c_olr = lag_corr(pw_ts.values, olr_ts.values, max_steps)
        lag_days = lags * dt_days_here

        ax[0].plot(lag_days, c_pr, color=cols[i], label=sim_names[i])
        ax[1].plot(lag_days, c_olr, color=cols[i])

    for a in ax:
        a.axvline(0, color="0.7", lw=0.5)
        a.axhline(0, color="0.7", lw=0.5)
        a.set_xlabel("Lag (days; positive means PW leads)")
    ax[0].set_ylabel("Correlation")
    ax[0].set_title("PW vs precipitation")
    ax[1].set_title("PW vs OLR")
    ax[0].legend(loc="best")

    save_figure(12, fig, "deep_tropical_lag_correlations")
    print("Done with figure: diag_deep_tropical_lag_correlations")


# ============================================================
# Additional diagnostics preserved from zm_latlon_misc_augmented.py
# ============================================================
# The older misc script overlapped with the diagnostics above in its symmetry,
# N2, Hovmoller, and OLR-precipitation analyses.  Those duplicated analyses are
# intentionally not repeated here; the blocks below preserve the distinct
# convective-state, hydrometeor, cloud-top, radiation-profile, and time-series
# diagnostics.

# ============================================================
# 3. Zonal-mean convective state and differences from control
# ============================================================
# What to look for:
#   - Broad OLR/high-cloud changes without a comparable precipitation change
#     suggest altered cloud depth/radiative effects rather than simply stronger rain.
#   - PW/CAPE/CIN shifts indicate that the lower-tropospheric convective regime
#     has changed, not just the UTLS.

if should_run_figure(13):
    surface_vars = [
        ("Precip", None, r"Precipitation (mm day$^{-1}$)"),
        ("olrtoa", "olrtoa", r"OLR (W m$^{-2}$)"),
        ("precipw", "precipw", r"Precipitable water (kg m$^{-2}$)"),
        ("cape", "cape", r"CAPE (J kg$^{-1}$)"),
        ("cin", "cin", r"CIN (J kg$^{-1}$)"),
    ]

    fig, ax = plt.subplots(len(surface_vars), 1, figsize=(8, 2.1 * len(surface_vars)), constrained_layout=True, sharex=True)
    fig2, ax2 = plt.subplots(len(surface_vars), 1, figsize=(8, 2.1 * len(surface_vars)), constrained_layout=True, sharex=True)

    for j, (label, var, ylabel) in enumerate(surface_vars):
        control = None
        for i, ds in enumerate(dss):
            if label == "Precip":
                x = precip_rate_from_accum(total_precip_accum(ds))
            else:
                if var not in ds.data_vars:
                    continue
                x = finite_or_nan(ds[var])

            xm = x.mean(dim=("Time", "longitude"))
            xm = hemispheric_fold(xm, parity="even")
            ax[j].plot(ds.latitude.values, xm.values, color=cols[i], label=sim_names[i])

            if i == 0:
                control = xm
                ax2[j].axhline(0, color="k", lw=0.7)
            else:
                c = control.interp(latitude=xm.latitude)
                ax2[j].plot(ds.latitude.values, (xm - c).values, color=cols[i], label=sim_names[i])

        ax[j].set_ylabel(ylabel)
        ax2[j].set_ylabel(r"$\Delta$ " + ylabel)
        for a in (ax[j], ax2[j]):
            a.axvline(0, color="grey", lw=0.5)
            a.set_xlim(-40, 40)

    ax[-1].set_xlabel("Latitude")
    ax2[-1].set_xlabel("Latitude")
    fig.legend(loc="outside lower center", ncol=len(dss))
    fig2.legend(loc="outside lower center", ncol=max(1, len(dss) - 1))

    if save:
        save_figure(13, fig, "zonalmean_convective_state")
        save_figure(13, fig2, "zonalmean_convective_state_diff")
    print("Done with figure: comp_zonalmean_convective_state")

# ============================================================
# 5. Hydrometeor latitude-height sections
# ============================================================
# What to look for:
#   - More upper-tropospheric ice/snow/graupel in T_nudge without much more rain
#     points to deeper/anvil cloud changes rather than just greater rain amount.

if should_run_figure(14):
    hydro_groups = [
        ("liquid_plus_rain", ["qc", "qr"], r"$q_c+q_r$ (g kg$^{-1}$)"),
        ("ice_plus_snow_graupel", ["qi", "qs", "qg"], r"$q_i+q_s+q_g$ (g kg$^{-1}$)"),
        ("total_condensate", ["qc", "qr", "qi", "qs", "qg"], r"Total condensate (g kg$^{-1}$)"),
    ]

    for outname, varlist, cbar_label in hydro_groups:
        fig = fig2 = None
        try:
            fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.4 * len(dss)), constrained_layout=True, sharex=True)
            fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.4 * len(dss)), constrained_layout=True, sharex=True)
            if len(dss) == 1:
                ax = [ax]
                ax2 = [ax2]

            control_q = None
            im = im2 = im3 = None

            # Use a common absolute colour scale from the control where possible.
            q0, _, _ = condensate_latheight(dss[0], dss_zm[0], varlist)
            vmax = safe_nanpercentile(
                q0.values, 99.5, zero_value=1e-4
            )
            levs = positive_levels(vmax, n=21)

            for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
                q, lat2d, z2d = condensate_latheight(ds, zds, varlist)

                im = ax[i].contourf(lat2d, z2d / 1000.0, q.values, levels=levs, cmap="YlGnBu", extend="max")
                add_cpt_height(ax[i], i)
                annotate_panel_label(ax[i], sim_names[i])
                set_latheight_labels(ax[i], xlabel=(i == len(dss) - 1))
                if i < len(dss) - 1:
                    ax[i].tick_params(labelbottom=False)

                if i == 0:
                    control_q = q
                    im2 = ax2[i].contourf(lat2d, z2d / 1000.0, q.values, levels=levs, cmap="YlGnBu", extend="max")
                else:
                    c = control_q.interp(z=q.z, latitude=q.latitude, kwargs={"fill_value": np.nan})
                    diff = q - c
                    dmax = safe_nanpercentile(
                        np.abs(diff.values),
                        99.0,
                        zero_value=max(vmax * 0.25, 1e-6),
                    )
                    q_diff_levels = symmetric_levels(dmax, ndiv=11)
                    im3 = ax2[i].contourf(lat2d, z2d / 1000.0, diff.values, levels=q_diff_levels, cmap=difference_cmap(q_diff_levels), extend="both")

                add_cpt_height(ax2[i], i)
                annotate_panel_label(ax2[i], control_diff_label(sim_names, i))
                set_latheight_labels(ax2[i], xlabel=(i == len(dss) - 1))
                if i < len(dss) - 1:
                    ax2[i].tick_params(labelbottom=False)

            fig.colorbar(im, ax=ax, location="right", label=cbar_label)
            fig2.colorbar(im2, ax=ax2[0], location="right", label=cbar_label)
            if im3 is not None:
                add_symmetric_colorbar(fig2, im3, ax2[1:], im3.levels, label=r"$\Delta$ " + cbar_label)

            if save:
                save_figure(14, fig, f"{outname}_latheight")
                save_figure(14, fig2, f"{outname}_latheight_diff")
            print(f"Done with figure: comp_{outname}_latheight")
        except ValueError as e:
            print(f"Skipping {outname}: {e}")
        finally:
            if fig is not None:
                plt.close(fig)
            if fig2 is not None:
                plt.close(fig2)

# Class-separated spatial/latitude responses now appear in refined high-cloud
# Figure 5; vertical class responses appear in Figure 4.

# ============================================================
# 7. Conditional convective relationships
# ============================================================
# What to look for:
#   - For the same precipitation, lower OLR or higher cloud top in T_nudge implies
#     deeper/more radiatively active cloud, not just more rain.
#   - Different precip(PW/CAPE/CIN) curves imply changed convective threshold.

if should_run_figure(16):
    conditional_specs = [
        ("precipw", r"Precipitable water (kg m$^{-2}$)", np.linspace(0, 80, 41)),
        ("cape", r"CAPE (J kg$^{-1}$)", np.linspace(0, 6000, 41)),
        ("cin", r"CIN (J kg$^{-1}$)", np.linspace(-500, 100, 41)),
    ]

    fig, ax = plt.subplots(1, len(conditional_specs), figsize=(12, 3.4), constrained_layout=True)
    if len(conditional_specs) == 1:
        ax = [ax]

    for j, (var, xlabel, bins) in enumerate(conditional_specs):
        for i, ds in enumerate(dss):
            if var not in ds.data_vars:
                continue
            pr = precip_rate_from_accum(total_precip_accum(ds))
            x = finite_or_nan(ds[var]).sel(Time=pr.Time)
            x = x.sel(latitude=slice(-20, 20))
            pr_sub = pr.sel(latitude=slice(-20, 20))
            x, pr_sub = xr.align(x, pr_sub, join="inner")
            xc, ym, y25, y75, count = binned_statistic_1d(x.values, pr_sub.values, bins)
            ax[j].plot(xc, ym, color=cols[i], label=sim_names[i])
            ax[j].fill_between(xc, y25, y75, color=cols[i], alpha=0.15)
        ax[j].set_xlabel(xlabel)
        ax[j].set_ylabel(r"Precipitation (mm day$^{-1}$)")
        ax[j].grid(alpha=0.2)
    ax[0].legend(loc="best")

    if save:
        save_figure(16, fig, "precip_conditioned_on_pw_cape_cin")
    print("Done with figure: comp_precip_conditioned_on_pw_cape_cin")

if should_run_any([17, 18]):
    print(
        "Figures 17-18 moved to refined_high_cloud_classification_diagnostics.py "
        "Figure 6. PW/CAPE/CIN-conditioned high-cloud curves were not retained; "
        "the broader precipitation-environment diagnostic remains in Figure 16."
    )

# ============================================================
# 8. Radiation budget proxy
# ============================================================
# What to look for:
#   - Tropical LW-atmospheric heating changes consistent with high-cloud changes
#     support a cloud-radiative pathway.

if should_run_figure(19):
    try:
        fig, ax = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True, sharex=True)
        fig2, ax2 = plt.subplots(3, 1, figsize=(8, 7), constrained_layout=True, sharex=True)
        control_profiles = None

        for i, ds in enumerate(dss):
            atm_lw, atm_sw, atm_rad = atmospheric_radiative_heating_proxy(ds)
            profiles = [
                hemispheric_fold(atm_lw.mean(dim=("Time", "longitude")), parity="even"),
                hemispheric_fold(atm_sw.mean(dim=("Time", "longitude")), parity="even"),
                hemispheric_fold(atm_rad.mean(dim=("Time", "longitude")), parity="even"),
            ]

            for j, prof in enumerate(profiles):
                ax[j].plot(ds.latitude, prof, color=cols[i], label=sim_names[i])

            if i == 0:
                control_profiles = profiles
                for a in ax2:
                    a.axhline(0, color="k", lw=0.7)
            else:
                for j, prof in enumerate(profiles):
                    c = control_profiles[j].interp(latitude=prof.latitude)
                    ax2[j].plot(ds.latitude, prof - c, color=cols[i], label=sim_names[i])

        labels = ["Atmos LW proxy", "Atmos SW proxy", "Atmos net rad proxy"]
        for j in range(3):
            ax[j].set_ylabel(labels[j] + "\n(W m$^{-2}$)")
            ax2[j].set_ylabel(r"$\Delta$ " + labels[j] + "\n(W m$^{-2}$)")
            for a in (ax[j], ax2[j]):
                a.axvline(0, color="grey", lw=0.5)
                a.set_xlim(-90, 90)
                a.grid(alpha=0.2)
        ax[-1].set_xlabel("Latitude")
        ax2[-1].set_xlabel("Latitude")
        fig.legend(loc="outside lower center", ncol=len(dss))
        fig2.legend(loc="outside lower center", ncol=max(1, len(dss) - 1))

        if save:
            save_figure(19, fig, "atmos_radiative_heating_proxy")
            save_figure(19, fig2, "atmos_radiative_heating_proxy_diff")
        print("Done with figure: comp_atmos_radiative_heating_proxy")
    except ValueError as e:
        print(f"Skipping radiation budget proxy: {e}")

# ============================================================
# 10. Extended latitude-band time series
# ============================================================
# What to look for:
#   - Coherent changes in PW -> rain -> OLR/high-cloud timing indicate changed
#     convection-radiation coupling even without coherent propagation.

if should_run_figure(20):
    fig, ax = plt.subplots(5, 1, figsize=(8, 8), constrained_layout=True, sharex=True)

    for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
        pr = precip_rate_from_accum(total_precip_accum(ds))
        pr_deep_trop = deep_tropical_mean_latlon(pr)
        pw_deep_trop = deep_tropical_mean_latlon(ds.precipw)
        olr_deep_trop = deep_tropical_mean_latlon(ds.olrtoa)

        ax[0].plot(
            get_time_values(pr_deep_trop), pr_deep_trop,
            color=cols[i], label=sim_names[i],
        )
        ax[1].plot(
            get_time_values(pw_deep_trop), pw_deep_trop, color=cols[i]
        )
        ax[2].plot(
            get_time_values(olr_deep_trop), olr_deep_trop, color=cols[i]
        )

        high_cld_ts = deep_tropical_mean_latlon(high_cloud_fraction(ds))
        ax[3].plot(get_time_values(high_cld_ts), high_cld_ts, color=cols[i])

        hydro = [v for v in ["qc", "qr", "qi", "qs", "qg"] if v in ds.data_vars]
        if len(hydro) > 0:
            qtot = sum(finite_or_nan(ds[v]) for v in hydro)
            qtot = qtot.sel(latitude=slice(-20, 20))
            qtot_trop = qtot.weighted(lat_weights(qtot)).mean(
                dim=[d for d in qtot.dims if d != "Time"],
                skipna=True,
            ) * 1000.0
            ax[4].plot(get_time_values(qtot_trop), qtot_trop, color=cols[i])

    ax[0].set_ylabel(r"Rain\n(mm day$^{-1}$)")
    ax[1].set_ylabel("PW")
    ax[2].set_ylabel("OLR")
    ax[3].set_ylabel("High cloud\nCTP <= 440 hPa")
    ax[4].set_ylabel(r"Mean condensate\n(g kg$^{-1}$)")
    ax[4].set_xlabel("Time")
    fig.legend(loc="outside lower center", ncol=len(dss))

    if save:
        save_figure(20, fig, "latitude_band_timeseries_extended")
    print("Done with figure: comp_latitude_band_timeseries_extended")

print("All combined symmetry/convection diagnostics complete.")
