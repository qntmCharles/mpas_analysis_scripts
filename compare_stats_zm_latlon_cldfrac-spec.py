#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os, glob, sys, numbers
sys.path.append("../")

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy.interpolate import griddata, RegularGridInterpolator
import pygeode as pyg
from jfunctions import *
import scipy.interpolate as intp
from scipy.ndimage import uniform_filter1d
import xarray as xr
from wavenumber_frequency_functions import *
from matplotlib import colors
from tropopause import *
from diagnostics.diagnostic_helpers import (
    FigureSaver,
    all_figures_requested,
    figure_should_run,
    figure_should_run_any,
    standardize_latlon_dataset as standardize_in_file_pressure_axis,
)

save = True
save_loc = '/home/cwp29/tmp/mpas/crf_on/'
figures_to_run = []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def run_all_figures():
    return all_figures_requested(figures_to_run)


# In[2]:


##### config_microp_re on/off test #####
#root_dir = '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/'
#data_locs = ['spinup/latlon_05deg.nc', 'microp_re_test/latlon_regridded_05deg.nc']
#sim_names = ['default', 'microp_re']

##### CRF on/off test #####
#root_dir = '/store/atmos-adk33/cwp29/mpas-a/data/qu_120km/'
#data_locs = ['control', 'crf_test']
#latlon_file = ['', 'latlon_spinup.nc']
#sim_names = ['crf on', 'crf off']

root_dir = '/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/'
data_locs = ['control', 'u_nudge', 'T_nudge']
latlon_file = ['latlon.2000-*.nc', 'latlon.2000-*.nc', 'latlon.2000-*.nc']
zm_latlon_file = ['zm_latlon.2000-*.nc', 'zm_latlon.2000-*.nc', 'zm_latlon.2000-*.nc']
sim_names = ['control', 'nudged u', 'nudged T']

cols = ['k', 'b', 'r']

# Use ndays of data ending at a fixed analysis date
ndays = 200
dt = 6
analysis_end_date = '2000-11-21'

time_format = 'time' #or 'int'


# In[3]:


def load_latlon_dataset(
    data_dir,
    latlon_pattern="latlon.2000-*.nc",
    diag_file="latlon_diag.nc",
    hist_file="latlon_hist.nc",
    cldfrac_file="cldfrac.nc",
    time_format="int",
    dt=6,
    ndays=None,
    end_date=None,
    chunks=None,
    prefer_pattern=False,
    add_cldfrac_file=True,
):
    diag_path = os.path.join(data_dir, diag_file)
    hist_path = os.path.join(data_dir, hist_file)
    cldfrac_path = os.path.join(data_dir, cldfrac_file)
    latlon_files = sorted(glob.glob(os.path.join(data_dir, latlon_pattern)))

    # Open lazily; rechunk later if needed.  For pre-computed zonal means,
    # prefer the explicit zm_latlon pattern over any latlon_diag/latlon_hist files
    # that may also exist in the same directory.
    if prefer_pattern and latlon_files:
        print(data_dir, latlon_pattern)
        ds = xr.open_mfdataset(latlon_files, combine="by_coords", chunks={}, data_vars="minimal", coords="minimal", compat="override")
        print(ds)

    elif os.path.exists(diag_path) and os.path.exists(hist_path):
        dsd = xr.open_dataset(diag_path, chunks={})
        dsh = xr.open_dataset(hist_path, chunks={})

        if time_format == "int":
            dsd = dsd.isel(Time=slice(0, None, 2))
        else:
            dsd = dsd.sel(Time=dsh.Time)

        ds = xr.merge([dsh, dsd], join="exact", compat="override")

    elif latlon_files:
        print(data_dir, latlon_pattern)
        ds = xr.open_mfdataset(latlon_files, combine="by_coords", chunks={}, data_vars="minimal", coords="minimal", compat="override")
        print(ds)

    else:
        raise FileNotFoundError(
            f"Could not find either {latlon_pattern} or both {diag_file} and {hist_file} in {data_dir}"
        )

    # Trim early, before adding extra variables
    if ndays is not None:
        if time_format == "int":
            ntime = ndays * int(24 / dt)
            ds = ds.isel(Time=slice(-ntime, None))
        else:
            t_end = pd.to_datetime(end_date) if end_date is not None else pd.to_datetime(ds.Time.max().item())
            t_start = t_end - pd.Timedelta(days=ndays)
            ds = ds.sel(Time=slice(t_start, t_end))

    if add_cldfrac_file and "cldfrac" not in ds.data_vars and os.path.exists(cldfrac_path):
        ds_cld = xr.open_dataset(cldfrac_path, chunks={})
    
        if "cldfrac" in ds_cld:
            # Align cldfrac to ds.Time; missing times become NaN.
            cldfrac = ds_cld["cldfrac"].reindex(Time=ds.Time)
    
            # Standardise the model-level dimension name to "z".
            if "nVertLevels" in cldfrac.dims:
                cldfrac = cldfrac.rename({"nVertLevels": "z"})
            elif "nVertLevelsP1" in cldfrac.dims:
                raise ValueError("cldfrac appears to be on interface levels, not mid-levels")
            elif "z" not in cldfrac.dims:
                raise ValueError(f"Could not find vertical dimension in cldfrac dims {cldfrac.dims}")
    
            # Replace cldfrac.z with a geometric-height coordinate.
            #
            # zgrid in the full latlon files is interface height:
            #     zgrid(nVertLevelsP1, latitude, longitude)
            #
            # A dimension coordinate must be 1D, so use the horizontal-mean
            # mid-level height as the z coordinate.
            if "zgrid" not in ds.data_vars:
                raise ValueError("Cannot assign geometric z coordinate to cldfrac: zgrid is missing")
    
            zgrid_mean = ds.zgrid.mean(dim=("latitude", "longitude"))
    
            z_lower = zgrid_mean.isel(nVertLevelsP1=slice(0, -1)).values
            z_upper = zgrid_mean.isel(nVertLevelsP1=slice(1, None)).values
            zmid = 0.5 * (z_lower + z_upper)
    
            if cldfrac.sizes["z"] != zmid.size:
                raise ValueError(
                    f"cldfrac has {cldfrac.sizes['z']} vertical levels, "
                    f"but zgrid implies {zmid.size} mid-levels"
                )
    
            cldfrac = cldfrac.assign_coords(z=("z", zmid))
            cldfrac["z"].attrs["units"] = "m"
            cldfrac["z"].attrs["long_name"] = "geometric height of layer midpoints"
    
            ds = ds.assign(cldfrac=cldfrac)

    # Rechunk only at the end
    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds


# In[4]:


dss = []
dss_zm = []

needed_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw",
    "uzonal_isobaric", "umeridional_isobaric", "omg_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric", "theta",
    "uReconstructZonal", "cldfrac", "olrtoa", "zgrid", "pressure", "vorticity", "ke"
]

check_vars = ['t_isobaric']



def get_time_values(ds):
    """Return a plotting-friendly time coordinate for this dataset only."""
    if issubclass(ds.Time.values.dtype.type, numbers.Integral):
        return ds.Time.values * dt / 24.0
    return ds.Time.values

def vertical_dim(da):
    """Find the model-level vertical dimension in either full lat/lon or zm_latlon data."""
    for dim in ("z", "nVertLevels", "nVertLevelsP1"):
        if dim in da.dims:
            return dim
    raise ValueError(f"No known model-level vertical dimension found in {da.dims!r}")

def midlevel_height_from_latlon(ds, latitude_sel=None, mean_latitude=False):
    """
    Geometric midlevel height from full latlon zgrid.

    From the latlon ncdump, zgrid has dimensions
    (nVertLevelsP1, latitude, longitude), with no Time dimension.
    This is used only for variables that are unavailable in zm_latlon
    and therefore still need an on-the-fly zonal mean from the full latlon files.
    """
    zgrid = ds.zgrid
    if latitude_sel is not None:
        zgrid = zgrid.sel(latitude=latitude_sel)

    zgrid = zgrid.mean(dim="longitude")
    if mean_latitude:
        zgrid = zgrid.mean(dim="latitude")

    # Use raw arrays for the midpoint so xarray does not align face-level
    # coordinates (0..N-1 with 1..N), which would silently drop the first
    # and last interfaces.
    lower = zgrid.isel(nVertLevelsP1=slice(0, -1)).values
    upper = zgrid.isel(nVertLevelsP1=slice(1, None)).values
    zmid_values = 0.5 * (lower + upper)

    if mean_latitude:
        return xr.DataArray(
            zmid_values,
            dims=("z",),
            coords={"z": np.arange(zmid_values.shape[0])},
            name="zmid",
        )

    return xr.DataArray(
        zmid_values,
        dims=("z", "latitude"),
        coords={
            "z": np.arange(zmid_values.shape[0]),
            "latitude": zgrid.latitude,
        },
        name="zmid",
    )


def height_from_zm_latlon(zds):
    """
    Geometric model-level height from precomputed zm_latlon files.

    From the zm_latlon ncdump, model-level zonal-mean variables use
    dimensions (Time, z, latitude), and z is the matching 1D geometric
    height coordinate.
    """
    return zds["z"]


def model_lat_field(da):
    """Return a DataArray ordered as (z, latitude) for latitude-height plots."""
    lev_dim = vertical_dim(da)
    if lev_dim != "z":
        da = da.rename({lev_dim: "z"})
    return da.transpose("z", "latitude")

def lat_height_mesh(field, zmid, lat):
    """Build 2D latitude and geometric-height arrays matching a (z, latitude) field."""
    field = model_lat_field(field)
    zmid = model_lat_field(zmid) if "latitude" in zmid.dims else zmid.rename({vertical_dim(zmid): "z"})
    lat2d = np.broadcast_to(lat[None, :], field.shape)
    zvals = zmid.values
    if zvals.ndim == 1:
        zvals = np.broadcast_to(zvals[:, None], field.shape)
    return field, lat2d, zvals

def standardize_latlon_dataset(ds, needed_vars=None):
    return standardize_in_file_pressure_axis(ds, needed_vars)
    
for i, d in enumerate(data_locs):
    """
    if i == 0:
        dsd = xr.open_dataset(os.path.join(root_dir, 'control/latlon_diag.nc'), chunks={'Time': 4})
        dsh = xr.open_dataset(os.path.join(root_dir, 'control/latlon_hist.nc'), chunks={'Time': 4})
        #print(dsd)
        #print(dsh)
        
        # keep only times present in the 6h dataset
        if time_format == 'int':
            dsd_6h = dsd.isel(Time=slice(0, None, 2))
        else:
            dsd_6h = dsd.sel(Time=dsh.Time)
        
        # merge variables into one dataset
        ds = xr.merge([dsh, dsd_6h], join="exact") 
        
    else:
        ds = xr.open_dataset(os.path.join(root_dir, d), chunks={'Time': 4})

    # Dataset before time restriction
    print(ds.data_vars)

    if time_format == 'int':
        ds = ds.sel(Time=slice(-1-ndays*int(24/dt), -1))
    else:
        # Restrict time
        t_end = pd.to_datetime(ds.Time.max().item())
        t_start = t_end - pd.Timedelta(days=ndays)
        print(t_start, t_end)
        ds = ds.sel(Time=slice(t_start, t_end))
    """
    ds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks={"Time": -1}
    )
    ds = standardize_latlon_dataset(ds, needed_vars)

    zds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=zm_latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks={"Time": -1},
        prefer_pattern=True,
        add_cldfrac_file=False
    )
    zds = standardize_latlon_dataset(zds, needed_vars)

    # cheaper if data are exactly 6-hourly and aligned
    #ds = ds.coarsen(Time=4, boundary="trim").mean()
    #zds = zds.coarsen(Time=4, boundary="trim").mean()

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)

def lat_mean(da, lat_dim='latitude'):
    w = np.cos(np.deg2rad(da[lat_dim]))
    return da.weighted(w).mean(dim=lat_dim)

# Plotting helper functions

def plot_Tu_latpres(a, lats, pres, Tbar, ubar):
    T_im = a.contourf(lats, pres, Tbar, levels=np.arange(190, 305, 5), extend='both', cmap='coolwarm')
    u_im_neg = a.contour(lats, pres, ubar, levels=np.arange(-20, 0, 5), colors='k', linestyles=':', linewidths=0.5)
    u_im_zero = a.contour(lats, pres, ubar, levels=[0], colors='k', linewidths=1)
    u_im_pos = a.contour(lats, pres, ubar, levels=np.arange(5, 55, 5), colors='k', linewidths=0.5)
    a.clabel(u_im_pos, u_im_pos.levels, fontsize=6)
    a.clabel(u_im_neg, u_im_neg.levels, fontsize=6)

    return T_im, u_im_neg, u_im_zero, u_im_pos

def plot_rhv_latpres(a, lats, pres, rhbar, vbar, omgbar):
    xi = np.linspace(lats.min(), lats.max(), len(lats))
    yi = np.linspace(pres.min(), pres.max(), len(pres))

    X, Y = np.meshgrid(xi, yi)

    U = intp.interpn((lats, pres), vbar.T, (X, Y))
    V = intp.interpn((lats, pres), omgbar.T, (X, Y))

    rh_im = a.contourf(lats, pres, rhbar, levels=np.arange(0, 105, 5), cmap='BrBG')
    medcirc_im = a.streamplot(X, Y, U/111111., V/100., color='k', linewidth=1, density=1)

    return rh_im, medcirc_im

def isobaric_lat_mean(da):
    """Return an isobaric time-mean field ordered as (pres, latitude)."""
    return da.mean(dim="Time").transpose("pres", "latitude")

def interp_like_control(field, control_field):
    """Interpolate the control field onto another field's pressure/latitude grid before differencing."""
    return control_field.interp(pres=field.pres, latitude=field.latitude)

def set_latheight_labels(ax, xlabel=True):
    ax.set_xlim(-90, 90)
    if xlabel:
        ax.set_xlabel("Latitude (degrees)")
    ax.set_ylabel("Height (km)")

def symmetric_levels(max_abs, ndiv=11):
    """Symmetric colour boundaries with an odd number of colour divisions."""
    return np.linspace(-max_abs, max_abs, ndiv + 1)

def make_diff_norm(levels):
    return colors.BoundaryNorm(levels, plt.get_cmap("bwr").N)

def make_spectrum_cmap(levels):
    """Colormap that leaves normalised power <= 1 nearly white and emphasizes > 1."""
    base = plt.get_cmap("YlGnBu")
    centres = 0.5 * (levels[:-1] + levels[1:])
    cols_spec = []
    for c in centres:
        if c <= 1.0:
            cols_spec.append((1.0, 1.0, 1.0, 1.0))
        else:
            x = (c - 1.0) / (levels[-1] - 1.0)
            cols_spec.append(base(0.15 + 0.85 * x))
    cmap = colors.ListedColormap(cols_spec, name="spec_gt1")
    cmap.set_over(base(1.0))
    cmap.set_under((1.0, 1.0, 1.0, 1.0))
    norm = colors.BoundaryNorm(levels, cmap.N)
    return cmap, norm

# pre-compute CPTs for plotting

CPTs = []

for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
    theta = zds.theta.mean(dim="Time")
    pressure = zds.pressure.mean(dim="Time")

    T = theta * (pressure / 100000.)**(287.0 / 1004.0)
    T = model_lat_field(T)
    pressure = model_lat_field(pressure)
    z = height_from_zm_latlon(zds).values

    ztps = []
    ptps = []
    for j in range(len(zds.latitude.values)):
        ttp, ptp, ztp, _ = tropopause_profile(T.values[:, j], pressure.values[:, j], z)
        ztps.append(ztp)
        ptps.append(ptp)
        print(ztp)

    CPTs.append([ztps, ptps])

CPTs = np.array(CPTs)

spec_levels = np.linspace(0, 2.5, 26)
spec_cmap, spec_norm = make_spectrum_cmap(spec_levels)
spec_diff_levels = symmetric_levels(1.5, ndiv=7)
spec_diff_norm = make_diff_norm(spec_diff_levels)

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
    precip = ds.rainc + ds.i_rainc*100. + ds.rainnc + ds.i_rainnc*100.

    if time_format == 'int':
        precip_rate = precip.diff("Time") / (dt / 24.0)
    else:
        dt_days = (
            precip.Time.diff("Time") / np.timedelta64(1, "D")
        )
        precip_rate = precip.diff("Time") / dt_days
        
    pr = precip_rate.rename({'latitude':'lat', 'longitude':'lon'})
    pr = pr.load()
    print(pr)

    spd = 4
        
    raw_spectrum = spacetime_power(pr, lataggreg='mean', latitude_bounds=(-10, 10), dosymmetries=True,
                                   segsize=48, noverlap=30, spd=spd)

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
    
        sym_diff_im = ax2[d, 0].pcolormesh( wn, f, sym_diff.T, cmap='bwr', norm=spec_diff_norm, shading='auto' )
        asym_diff_im = ax2[d, 1].pcolormesh( wn, f, asym_diff.T, cmap='bwr', norm=spec_diff_norm, shading='auto' )
    
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
            if d == len(data_locs)-1:
                a.set_xlabel('Zonal wavenumber')
            else:
                a.set_xticklabels([])

        axes[d, 0].set_ylabel('Frequency (day$^{-1}$)')
        axes[d, 1].set_yticklabels([])
        axes[d, 0].axvline(0, color='k', ls='--')
        axes[d, 1].axvline(0, color='k', ls='--')
        axes[d, 0].annotate(sim_names[d], (-14, 0.45), fontweight='bold', bbox=dict(facecolor='w', alpha=0.7))

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
fig2.colorbar( sym_diff_im if sym_diff_im is not None else sym_spec_im, ax=ax2[-1], location='bottom', shrink=0.5, label=r"Difference in excess normalised power above 1" )

figsave.save(fig, "spectra", number=1)
if save: fig2.savefig(save_loc+'comp_spectra_diff.png', dpi=300)
print("Done with figure: comp_spectra")

# Latitude-height section of zonal-mean cloud fraction

fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)
fig2, ax2 = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)

if len(dss) == 1:
    ax = [ax]
    ax2 = [ax2]

im = im2 = im3 = None
control_cldfrac = None

def hemispheric_symmetrise_lat(da, lat_dim="latitude"):
    """
    Average a latitude field with its hemispherically flipped version.

    Assumes the latitude grid is symmetric and ordered monotonically, e.g.
    -89.75 ... 89.75. The returned DataArray keeps the original latitude
    coordinate.
    """
    flipped = da.isel({lat_dim: slice(None, None, -1)})
    flipped = flipped.assign_coords({lat_dim: da[lat_dim].values})
    return 0.5 * (da + flipped)

for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
    if "cldfrac" in zds.data_vars:
        # Newer zm_latlon files may contain precomputed zonal-mean cloud fraction
        # with dimensions (Time, z, latitude).
        cldfrac = zds.cldfrac.mean(dim="Time")
        zmid = height_from_zm_latlon(zds)
        cldfrac, lat2d, z2d = lat_height_mesh(cldfrac, zmid, zds.latitude.values)
        lat_for_cpt = zds.latitude.values

    elif "cldfrac" in ds.data_vars:
        # Older runs have no cldfrac in latlon/zm_latlon, but do have a separate
        # cldfrac.nc.  That file is attached only to the full latlon dataset, so
        # compute the zonal mean here.
        cldfrac = ds.cldfrac.mean(dim=("Time", "longitude"))
        cldfrac = model_lat_field(cldfrac)
        
        if i == 0:
            cldfrac = hemispheric_symmetrise_lat(cldfrac)
        
        zmid = midlevel_height_from_latlon(ds)
        cldfrac, lat2d, z2d = lat_height_mesh(cldfrac, zmid, ds.latitude.values)
        lat_for_cpt = ds.latitude.values
    else:
        print(f"Skipping cloud-fraction plot for {sim_names[i]}: no cldfrac in zm_latlon, latlon, or cldfrac.nc")
        continue

    im = ax[i].contourf(lat2d, z2d / 1000.0, cldfrac.values, levels=np.linspace(0, 1, 21), cmap='binary', extend='max')
    ax[i].plot(lat_for_cpt, CPTs[i][0]/1000., color='k', lw=1, ls=':')
    ax[i].set_title(sim_names[i])
    set_latheight_labels(ax[i], xlabel=(i == len(dss)-1))
    if i < len(dss)-1:
        ax[i].set_xticklabels([])

    if i == 0:
        control_cldfrac = cldfrac.copy()
    
        im2 = ax2[i].contourf(
            lat2d, z2d / 1000.0, cldfrac.values,
            levels=np.linspace(0, 1, 21),
            cmap='binary',
            extend='max'
        )
    else:
        # Put control cloud fraction onto this simulation's z/latitude grid
        c_cldfrac = control_cldfrac.interp(
            z=cldfrac.z,
            kwargs={"fill_value": np.nan}
        )
    
        # Then subtract arrays
        cldfrac_diff = cldfrac.values - c_cldfrac.values
    
        im3 = ax2[i].contourf(
            lat2d, z2d / 1000.0, cldfrac_diff,
            levels=symmetric_levels(0.1, ndiv=11),
            cmap='bwr',
            extend='both'
        )
    

    ax2[i].plot(lat_for_cpt, CPTs[i][0]/1000., color='k', lw=1, ls=':')
    ax2[i].set_title(sim_names[i])
    set_latheight_labels(ax2[i], xlabel=(i == len(dss)-1))
    if i < len(dss)-1:
        ax2[i].set_xticklabels([])

fig.colorbar(im, ax=ax, location='right', label='Cloud fraction')
fig2.colorbar(im2, ax=ax2[0], location='right', label='Cloud fraction')
if im3 is not None:
    fig2.colorbar(im3, ax=ax2[1:], location='right', label='Cloud-fraction difference')

figsave.save(fig, "cldfrac_latheight", number=2)
if save: fig2.savefig(save_loc+'comp_cldfrac_latheight_diff.png', dpi=300)
print("Done with figure: comp_cldfrac_latheight")

# Diagnostic: how much does zgrid vary horizontally?

fig, ax = plt.subplots(len(dss), 2, figsize=(8, 3.2*len(dss)), constrained_layout=True, sharex='col')

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
    zmean = np.nanmean(zmid, axis=(1, 2))         # global mean profile
    zmin = np.nanmin(zmid, axis=(1, 2))           # global min at each level
    zmax = np.nanmax(zmid, axis=(1, 2))           # global max at each level
    zstd = np.nanstd(zmid, axis=(1, 2))           # global std at each level

    # Zonal mean and anomaly from global mean
    z_zonal = np.nanmean(zmid, axis=2)            # (z, latitude)
    z_anom = z_zonal - zmean[:, None]

    # ---- left: profile with spread ----
    ax[i, 0].plot(zmean / 1000., zmean / 1000., color=cols[i], lw=1.5)
    ax[i, 0].fill_betweenx(zmean / 1000., zmin / 1000., zmax / 1000.,
                           color=cols[i], alpha=0.15, label='min/max')
    ax[i, 0].fill_betweenx(zmean / 1000., (zmean - zstd) / 1000., (zmean + zstd) / 1000.,
                           color=cols[i], alpha=0.3, label=r'$\pm 1\sigma$')

    ax[i, 0].plot(zmean / 1000., zmean / 1000., color=cols[i], lw=1.5, label=sim_names[i])
    ax[i, 0].set_ylabel("Height (km)")
    ax[i, 0].set_title(sim_names[i])
    if i == len(dss) - 1:
        ax[i, 0].set_xlabel("Global-mean height (km)")

    # ---- right: latitude variation ----
    im = ax[i, 1].contourf(
        ds.latitude.values,
        zmean / 1000.,
        z_anom / 1000.,
        levels=np.linspace(-1.0, 1.0, 21),
        cmap='bwr',
        extend='both'
    )
    ax[i, 1].set_title(sim_names[i])
    if i == len(dss) - 1:
        ax[i, 1].set_xlabel("Latitude (degrees)")
    ax[i, 1].set_ylabel("Height (km)")

    # optional: print a simple summary
    print(sim_names[i])
    print("max |zonal-mean height anomaly| (km):", np.nanmax(np.abs(z_anom)) / 1000.)
    print("max horizontal range at a level (km):", np.nanmax(zmax - zmin) / 1000.)

# one colourbar for the anomaly panels
fig.colorbar(im, ax=ax[:, 1], location='right', label='Height anomaly from global mean (km)')

figsave.save(fig, "zgrid_horizontal_variation", number=3)

print("Done with figure: check_zgrid_horizontal_variation")
