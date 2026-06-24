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


#### config_microp_re on/off test #####
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

# Use final 'ndays' of data
ndays = 200
dt = 6

time_format = 'time' #or 'int'

def load_latlon_dataset(
    data_dir,
    latlon_pattern="latlon.2000-*.nc",
    diag_file="latlon_diag.nc",
    hist_file="latlon_hist.nc",
    cldfrac_file="cldfrac.nc",
    time_format="int",
    dt=6,
    ndays=None,
    chunks=None,
    prefer_pattern=False,
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
            t_end = pd.to_datetime(ds.Time.max().item())
            t_start = t_end - pd.Timedelta(days=ndays)
            ds = ds.sel(Time=slice(t_start, t_end))

    # Add cldfrac only for matching times, without a full merge
    if os.path.exists(cldfrac_path):
        ds_cld = xr.open_dataset(cldfrac_path, chunks={})

        if "cldfrac" in ds_cld:
            # align cldfrac to ds.Time; missing times become NaN
            cldfrac = ds_cld["cldfrac"].reindex(Time=ds.Time)
            ds = ds.assign(cldfrac=cldfrac)

    # Rechunk only at the end
    if chunks is not None:
        ds = ds.chunk(chunks)

    return ds

dss = []
dss_zm = []

needed_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw",
    "uzonal_isobaric", "umeridional_isobaric", "omg_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric", "theta", "relhum",
    "uReconstructZonal", "uReconstructMeridional", "cldfrac", "olrtoa", "zgrid", "pressure", "vorticity", "ke"
]

check_vars = ['t_isobaric']


def mean_if_present(da, dims):
    """Mean over dims that exist; useful for full lat/lon and pre-zonal-mean files."""
    dims = [d for d in dims if d in da.dims]
    return da.mean(dim=dims) if dims else da

def vertical_dim(da):
    """Return the model-level vertical dimension used by either latlon or zm_latlon files."""
    for dim in ("z", "nVertLevels"):
        if dim in da.dims:
            return dim
    raise ValueError(f"Could not identify vertical dimension in {da.dims}")

def vertical_coord(ds, da=None):
    """Return 1D height/level coordinate for model-level zm_latlon fields.

    zm_latlon files use a 1D `z` coordinate directly. Full latlon files use `zgrid`
    on interfaces, so convert to cell-center heights after zonal/time averaging.
    """
    if da is not None and "z" in da.dims and "z" in ds.coords:
        return ds["z"].values
    if "z" in ds.coords:
        return ds["z"].values
    if "zgrid" in ds:
        zgrid = mean_if_present(ds.zgrid, ["Time", "longitude", "latitude"] if da is None else ["Time", "longitude"])
        return 0.5 * (zgrid.isel(nVertLevelsP1=slice(0, -1)).values +
                      zgrid.isel(nVertLevelsP1=slice(1, None)).values)
    raise KeyError("No `z` coordinate or `zgrid` variable available for vertical coordinate")

def zonal_time_mean(ds, varname):
    """Use precomputed zonal means when present, otherwise calculate from full latlon."""
    return mean_if_present(ds[varname], ["Time", "longitude"])

def get_zm_or_full(ds, zds, varname):
    """Prefer zm_latlon variable; fall back to full latlon variable when zm lacks it."""
    return zds if varname in zds.data_vars else ds

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
        dt=dt,
        ndays=ndays,
        chunks={"Time": -1}
    )
    ds = standardize_latlon_dataset(ds, needed_vars)

    zds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=zm_latlon_file[i],
        time_format=time_format,
        dt=dt,
        ndays=ndays,
        chunks={"Time": -1},
        prefer_pattern=True
    )
    zds = standardize_latlon_dataset(zds, needed_vars)

    # cheaper if data are exactly 6-hourly and aligned
    #ds = ds.coarsen(Time=4, boundary="trim").mean()
    #zds = zds.coarsen(Time=4, boundary="trim").mean()

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)

if issubclass(ds.Time.values.dtype.type, numbers.Integral):
    time = ds.Time.values * dt/24.
else:
    time = ds.Time.values
print(time)


# In[5]:


def lat_mean(da, lat_dim='latitude'):
    w = np.cos(np.deg2rad(da[lat_dim]))
    return da.weighted(w).mean(dim=lat_dim)


def plot_time_coord(ds):
    """Return a plotting-friendly time coordinate for this dataset only.

    Avoid using a single global time array because different input files can
    have slightly different endpoint inclusion after the ndays trim.
    """
    if issubclass(ds.Time.values.dtype.type, numbers.Integral):
        return ds.Time.values * dt / 24.0
    return ds.Time.values

def z_at_lat(z, j):
    """Return vertical coordinate for one latitude from either 1D zm z or 2D z(lat)."""
    z = np.asarray(z)
    if z.ndim == 1:
        return z
    return z[:, j]


# In[6]:


# rainc, rainnc contain accumulated rainfall from simulation start to given timestep
fig, ax = plt.subplots(2, 1, figsize=(8, 4.5), constrained_layout=True)

for i, ds in enumerate(dss):
    # Precipitation and precipitable water are not in zm_latlon, so keep using full latlon here.
    # Use each dataset's own time coordinate: after time trimming, files can differ by one
    # endpoint depending on whether the 200-day range is inclusive.
    ds_time = plot_time_coord(ds)
    precip = ds['rainnc'] + ds['i_rainnc']*100. + ds['rainc'] + ds['i_rainc']*100.
    precip_global = lat_mean(precip.mean(dim='longitude'))
    precip_rate = np.gradient(precip_global.values)
    
    precip_water = lat_mean(ds.precipw.mean(dim='longitude')).values

    ax[0].plot(ds_time, precip_water, color=cols[i], label=sim_names[i])
    ax[1].plot(ds_time, precip_rate, color=cols[i])

ax[0].set_ylim(18, 23)
ax[0].set_ylabel(r"Precipitable water (mm)")

ax[1].set_ylim(2, 4)
ax[1].set_ylabel(r"Rainfall rate (mm day$^{-1}$)")

ax[1].set_xlabel('Time (days)')

fig.legend(loc='outside lower center', ncol=4)

figsave.save(fig, "rain_stats", number=1)


# In[7]:


# Vertical profiles of tropical-mean potential temperature (theta) and zonal-mean wind (uReconstructZonal)

fig, ax = plt.subplots(1, 3, figsize=(8, 4), sharey=True, constrained_layout=True)

for i, zds in enumerate(dss_zm):
    theta = mean_if_present(zds.theta.sel(latitude=slice(-10, 10)), ['Time', 'latitude', 'longitude'])
    pressure = mean_if_present(zds.pressure.sel(latitude=slice(-10, 10)), ['Time', 'latitude', 'longitude'])
    u = mean_if_present(zds.uReconstructZonal.sel(latitude=slice(-10, 10)), ['Time', 'latitude', 'longitude'])

    T = theta * (pressure / 100000.)**(287.0 / 1004.0)
    T_iso = mean_if_present(zds.t_isobaric.sel(latitude=slice(-10, 10)), ['Time', 'latitude', 'longitude'])
    
    z = vertical_coord(zds, theta)

    print(z, pressure.values)
    ttp, ptp, ztp, _ = tropopause_profile(T.values, pressure.values, z)
    print(ttp, ptp, ztp)

    for a in ax:
        a.axhline(ztp/1000., color=cols[i], ls=':')
        
    ax[0].plot(theta.values, z/1000., color=cols[i], label=sim_names[i])
    ax[1].plot(T.values, z/1000., color=cols[i])
    ax[2].plot(u.values, z/1000., color=cols[i])
    
ax[0].set_xlabel(r'Potential temperature $\theta$ (K)')
ax[1].set_xlabel(r'Temperature $T$ (K)')
ax[1].set_xlim(150, 300)
ax[2].set_xlabel(r'Zonal mean wind $u$ (m s$^{-1}$)')
ax[0].set_ylabel('Height (km)')

fig.legend(loc='outside lower center', ncol=len(dss))

figsave.save(fig, "profiles", number=2)


# In[8]:


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


# In[ ]:


# pre-compute CPTs for plotting

CPTs = []

for i, zds in enumerate(dss_zm):
    theta = mean_if_present(zds.theta, ['Time', 'longitude'])
    pressure = mean_if_present(zds.pressure, ['Time', 'longitude'])
    u = mean_if_present(zds.uReconstructZonal, ['Time', 'longitude'])

    T = theta * (pressure / 100000.)**(287.0 / 1004.0)
    
    z = vertical_coord(zds, theta)
    ztps = []
    ptps = []
    for j in range(len(zds.latitude.values)):
        zj = z_at_lat(z, j)
        ttp, ptp, ztp, _ = tropopause_profile(T.values[:, j], pressure.values[:, j], zj)
        ztps.append(ztp)
        ptps.append(ptp)
        print(ztp)

    CPTs.append([ztps, ptps])

CPTs = np.array(CPTs)


# In[ ]:


fig, ax = plt.subplots(len(dss_zm), 2, figsize=(8, 3.5*len(dss_zm)), sharey=True, constrained_layout=True)

for i, (ds, zds) in enumerate(zip(dss, dss_zm)):
    ubar = mean_if_present(zds.uzonal_isobaric, ['longitude', 'Time']).values
    vbar = mean_if_present(zds.umeridional_isobaric, ['longitude', 'Time']).values
    omgbar = mean_if_present(zds.omg_isobaric, ['longitude', 'Time']).values
    # zm_latlon has model-level relhum but no rh_isobaric; fall back to full latlon for pressure-level RH.
    rh_src = get_zm_or_full(ds, zds, 'rh_isobaric')
    rhbar = mean_if_present(rh_src.rh_isobaric, ['longitude', 'Time']).values
    Tbar = mean_if_present(zds.t_isobaric, ['longitude', 'Time']).values
    
    T_im, _, _, _ = plot_Tu_latpres(ax[i, 0], zds.latitude.values, zds.pres.values, Tbar, ubar)
    rh_im, _ = plot_rhv_latpres(ax[i, 1], zds.latitude.values, zds.pres.values, rhbar, vbar, omgbar)
    
    ax[i, 0].annotate(sim_names[i], (-75, 15), fontweight='bold', bbox=dict(facecolor='w', alpha=0.7))
    ax[i, 1].annotate(sim_names[i], (-75, 15), fontweight='bold', bbox=dict(facecolor='w', alpha=0.7))

    ax[i, 0].plot(zds.latitude.values, CPTs[i][1]/100., color='k', lw=1, ls=':')
    ax[i, 1].plot(zds.latitude.values, CPTs[i][1]/100., color='k', lw=1, ls=':')

for a in ax[:-1, :].ravel():
    a.set_xticklabels([])

ax[-1, 0].set_xlabel("Latitude (degrees)")
ax[-1, 1].set_xlabel("Latitude (degrees)")
    
for i in range(ax.shape[0]):
    ax[i, 0].set_ylabel("Pressure (hPa)")
    
ax[0, 0].invert_yaxis()
ax[0, 0].set_yscale('log')
    
for a in ax.ravel():
    a.set_xlim(-90, 90)
    a.set_ylim(1000, 10)
    
fig.colorbar(T_im, ax=ax[-1, 0], location='bottom', label="temperature (K)")
fig.colorbar(rh_im, ax=ax[-1, 1], location='bottom', label="relative humidity (%)")

figsave.save(fig, "temp_relhum", number=3)


# In[ ]:


# Meridional eddy fluxes

fig, ax = plt.subplots(4, 2, figsize=(8, 8), constrained_layout='True')

for i, ds in enumerate(dss):
    ua = ds.uzonal_isobaric.sel(pres=[250, 700]) - ds.uzonal_isobaric.sel(pres=[250, 700]).mean(dim=('longitude', 'Time'))
    va = ds.umeridional_isobaric.sel(pres=[250, 700]) - ds.umeridional_isobaric.sel(pres=[250, 700]).mean(dim=('longitude', 'Time')) 
    Ta = ds.t_isobaric.sel(pres=[250, 700]) - ds.umeridional_isobaric.sel(pres=[250, 700]).mean(dim=('longitude', 'Time'))
    qva = ds.qv_isobaric.sel(pres=[250, 700]) - ds.qv_isobaric.sel(pres=[250, 700]).mean(dim=('longitude', 'Time'))
    
    eke = (ua*ua + va*va).mean(dim=('longitude', 'Time'))
    memf = (ua*va).mean(dim=('longitude', 'Time'))
    mehf = (va*Ta).mean(dim=('longitude', 'Time'))
    mewvf = (va*qva).mean(dim=('longitude', 'Time'))

    for j, var in enumerate([eke, memf, mehf, mewvf]):
        if j == 0:
            ax[j, 0].plot(ds.latitude.values, var[0], color=cols[i], label=sim_names[i])
        else:
            ax[j, 0].plot(ds.latitude.values, var[0], color=cols[i])
        ax[j, 1].plot(ds.latitude.values, var[1], color=cols[i])

for a in ax.ravel():
    a.set_xlim(-90, 90)

ax[0, 0].set_ylabel("Eddy kinetic \n energy (m$^2$ s$^{-2}$)")
ax[1, 0].set_ylabel("Meridional eddy \n momentum flux (m$^2$ s$^{-2}$)")
ax[2, 0].set_ylabel("Meridional heat \n flux (K m s$^{-1}$)")
ax[3, 0].set_ylabel("Meridional wv \n flux (g kg$^{-1}$ m s$^{-1}$)")

fig.legend(loc='outside lower center', ncol=4)
        
figsave.save(fig, "eddy_fluxes", number=4)


# In[ ]:


imerg_ds = xr.open_mfdataset(list(sorted(glob.glob('/data/phh1-hosted/cwp29/imerg/*3IMERG.*.nc4')))[:100], concat_dim='time', combine='nested')

imerg_pr = imerg_ds.precipitation.coarsen(lat=10).mean().coarsen(lon=10).mean()


# In[ ]:


# Normalised rainfall intensity distribution
from scipy.ndimage import uniform_filter1d

fig, ax = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True, sharey=True)

h, bins = np.histogram(imerg_pr, bins=300, range=(0, 300))
h = uniform_filter1d(h, size=10)
ax[0].plot(0.5*(bins[1:] + bins[:-1]), h/h[0], color='k', ls=':', label='IMERG (Obs)')
ax[0].set_yscale('log')

for s, ds in enumerate(dss):
    precip_con = ds.rainc + ds.i_rainc*100.
    precip_res = ds.rainnc + ds.i_rainnc*100.
    precip = precip_con + precip_res
    
    for i, precip_var in enumerate([precip, precip_con, precip_res]):
        if time_format == 'int':
            precip_rate = precip_var.diff("Time") / (dt / 24.0)
        else:
            dt_days = (
                precip_var.Time.diff("Time") / np.timedelta64(1, "D")
            )
            precip_rate = precip_var.diff("Time") / dt_days
        h, bins = np.histogram(precip_rate.values, bins=300, range=(0, 300))
        h = uniform_filter1d(h, size=10)
        if i == 0:
            ax[i].plot(0.5*(bins[1:] + bins[:-1]), h/h[0], color=cols[s], label=sim_names[s])
        else:
            ax[i].plot(0.5*(bins[1:] + bins[:-1]), h/h[0], color=cols[s])
        ax[i].set_yscale('log')

fig.legend(loc='outside lower center', ncol=5)

for a in ax:
    a.set_xlim(0, 300)
    a.set_ylim(1e-7, 1)
    a.set_xlabel("Rainfall rate (mm/day)")

ax[0].set_title("Total")
ax[0].set_ylabel("Normalised frequency")
ax[1].set_title("Parameterised")
ax[2].set_title("Resolved")
    
figsave.save(fig, "pr_dist", number=5)


# In[ ]:


def calculate_rain_durations(pr_var, thresh):
    pr_thresh = np.where(pr_var > thresh, 1, 0)
    
    durations = []

    for i in range(pr_thresh.shape[1]):
        for j in range(pr_thresh.shape[2]):
            timeseries = pr_thresh[:, i, j]
    
            count = 0
            for k in range(pr_thresh.shape[0]):
                if timeseries[k] > 0:
                    count += 1
                else:
                    if count > 0:
                        durations.append(count)
                        count = 0
                        
    return durations
                    
fig, ax = plt.subplots(1, 3, figsize=(8, 3), constrained_layout=True, sharey=True)

h, bins = np.histogram(calculate_rain_durations(imerg_pr, 5), bins=39, range=(1, 40))
h = uniform_filter1d(h, size=5)
ax[0].plot(0.5*(bins[1:] + bins[:-1]), h/h[0], color='k', ls=':', label='IMERG (Obs)')
ax[0].set_yscale('log')

for s, ds in enumerate(dss):
    precip_con = ds.rainc + ds.i_rainc*100.
    precip_res = ds.rainnc + ds.i_rainnc*100.
    precip = precip_con + precip_res

    for i, precip_var in enumerate([precip, precip_con, precip_res]):
        if time_format == 'int':
            precip_rate = precip_var.diff("Time") / (dt / 24.0)
        else:
            dt_days = (
                precip_var.Time.diff("Time") / np.timedelta64(1, "D")
            )
            precip_rate = precip_var.diff("Time") / dt_days
        h, bins = np.histogram(calculate_rain_durations(precip_rate.values, 5), bins=39, range=(1, 40))
        h = uniform_filter1d(h, size=5)
        if i == 0:
            ax[i].plot(0.5*(bins[1:] + bins[:-1]), h/h[0], color=cols[s], label=sim_names[s])
        else:
            ax[i].plot(0.5*(bins[1:] + bins[:-1]), h/h[0], color=cols[s])
        ax[i].set_yscale('log')

for a in ax:
    a.set_xlim(1, 60)
    a.set_xlabel("Duration (days)")
    
fig.legend(loc='outside lower center', ncol=5)

ax[0].set_title("Total")
ax[0].set_ylabel("Normalised frequency")
ax[1].set_title("Parameterised")
ax[2].set_title("Resolved")

figsave.save(fig, "duration_dist", number=6)


# In[ ]:


spec_levels = np.linspace(0, 3.0, 26)

fig, ax = plt.subplots(len(data_locs), 2, figsize=(8, 8), constrained_layout=True)

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
                                   segsize=32, noverlap=16, spd=spd)

    sym_smooth_spectrum = smooth_wavefreq(raw_spectrum.sel(component='symmetric'), freq_ax = 1)
    asym_smooth_spectrum = smooth_wavefreq(raw_spectrum.sel(component='antisymmetric'), freq_ax = 1)
    
    sym_sig_spectrum = raw_spectrum.sel(component='symmetric')/sym_smooth_spectrum
    asym_sig_spectrum = raw_spectrum.sel(component='antisymmetric')/asym_smooth_spectrum

    wn = sym_sig_spectrum.wavenumber
    f = sym_sig_spectrum.frequency

    sym_spec_im = ax[d, 0].contourf(wn, f, sym_sig_spectrum.T, levels=spec_levels, cmap='YlGnBu', extend='max')
    
    asym_spec_im = ax[d, 1].contourf(wn, f, asym_sig_spectrum.T, levels=spec_levels, cmap='YlGnBu', extend='max')
    
    freq, wavenum = genDispersionCurves()
    
    labels = ['MRG', 'IG', 'EIG', 'Kelvin', 'ER', 'IG']

    for i in range(3):
        for j in range(3):
            ax[d, 1].plot(wavenum[i, j], freq[i, j], color='r', alpha=0.5)
            ax[d, 0].plot(wavenum[i+3, j], freq[i+3, j], color='r', alpha=0.5)
    
    # Decorations
    for a in ax[d]:
        a.set_xlim(-15, 15)
        a.set_ylim(0, 0.5)
        
        if d == len(data_locs)-1:
            a.set_xlabel('Zonal wavenumber')
        else:
            a.set_xticklabels([])

    ax[d, 0].set_ylabel('Frequency (day$^{-1}$)')
    ax[d, 1].set_yticklabels([])
    
    ax[d, 0].axvline(0, color='k', ls='--')
    ax[d, 1].axvline(0, color='k', ls='--')
    
    ax[d, 0].annotate(sim_names[d], (-14, 0.45), fontweight='bold', bbox=dict(facecolor='w', alpha=0.7))

    if d == 0:
        ax[d, 0].annotate('ER', (-11, 0.12), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
        ax[d, 0].annotate('Kelvin', (10, 0.4), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
        ax[d, 0].annotate('IG', (-5, 0.4), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
        
        ax[d, 1].annotate('MRG', (-12, 0.07), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
        ax[d, 1].annotate('IG', (-5, 0.43), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))
        ax[d, 1].annotate('EIG', (11, 0.32), bbox=dict(facecolor='w', edgecolor='k', alpha=0.7))

ax[0, 0].set_title('Symmetric')
ax[0, 1].set_title('Anti-symmetric')

fig.colorbar(sym_spec_im, ax=ax[-1] ,location='bottom', shrink=0.5, label="Normalised power")

figsave.save(fig, "spectra", number=7)


# In[ ]:


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


# In[ ]:


# Hovmoller diagrams of daily precipitation rate
    
fig, ax = plt.subplots(1, len(dss), figsize=(8, 3), constrained_layout=True)

for d, ds in enumerate(dss):
    precip = ds.rainc + ds.i_rainc*100. + ds.rainnc + ds.i_rainnc*100.
    precip = precip.sel(latitude=slice(-10, 10)).mean(dim='latitude')
    
    if time_format == 'int':
        pr = precip.diff("Time") / (dt / 24.0)
    else:
        dt_days = (
            precip.Time.diff("Time") / np.timedelta64(1, "D")
        )
        pr = precip.diff("Time") / dt_days
    
    hovmoller_im = ax[d].contourf(precip.longitude, pr.Time, pr.values, levels=levels, norm=precip_norm, cmap=precip_cmap, extend='max')

    ax[d].set_title(sim_names[d])
    
ax[0].set_ylabel("time (days)")
    
fig.colorbar(hovmoller_im, ax=ax[-1], location='right', label="Precipitation rate (mm/h)")
    
for a in ax:
    a.set_xlabel("Longitude")

figsave.save(fig, "hovmoller", number=8)


# In[ ]:


# Latitude-height section of zonal-mean cloud fraction

fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)

if len(dss) == 1:
    ax = [ax]

for i, ds in enumerate(dss):
    # cldfrac is not present in zm_latlon, so compute this zonal mean from full latlon.
    cldfrac = ds.cldfrac.mean(dim=('Time', 'longitude'))

    zgrid = ds.zgrid.mean(dim=('Time', 'longitude'))
    z = 0.5 * (zgrid.isel(nVertLevelsP1=slice(0, -1)).values +
               zgrid.isel(nVertLevelsP1=slice(1, None)).values)

    lat = ds.latitude.values
    lat2d = np.broadcast_to(lat[None, :], cldfrac.shape)

    im = ax[i].contourf(lat2d, z / 1000.0, cldfrac.values, levels=np.linspace(0, 1, 21), cmap='binary', extend='max')
    ax[i].plot(ds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
    ax[i].plot(ds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
    
    ax[i].set_title(sim_names[i])
    ax[i].set_ylabel('Height (km)')
    ax[i].set_xlim(-90, 90)

ax[-1].set_xlabel('Latitude (degrees)')
fig.colorbar(im, ax=ax, location='right', label='Cloud fraction')

figsave.save(fig, "cldfrac_latheight", number=9)


# In[ ]:


# Latitude-height section of zonal-mean vorticity

fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)

if len(dss) == 1:
    ax = [ax]

for i, zds in enumerate(dss_zm):
    if 'vorticity' in zds.data_vars:
        vort = mean_if_present(zds.vorticity, ['Time', 'longitude']).transpose('z', 'latitude')
    
        z = vertical_coord(zds, vort)
    
        lat = zds.latitude.values
        lat2d = np.broadcast_to(lat[None, :], vort.shape)
    
        vmax = np.nanmax(np.abs(vort.values))
        levels = np.linspace(-vmax, vmax, 21)
    
        im = ax[i].contourf(lat2d, z/1000., vort.values, levels=levels, cmap='coolwarm', extend='both')
        ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        
        ax[i].set_title(sim_names[i])
        ax[i].set_ylabel('Height (km)')
        ax[i].set_xlim(-90, 90)

ax[-1].set_xlabel('Latitude (degrees)')
fig.colorbar(im, ax=ax, location='right', label='Vorticity')

figsave.save(fig, "vorticity_latheight", number=10)


# In[ ]:


# Histogram of OLR and time-mean lat/lon maps

fig, ax = plt.subplots(len(dss), 2, figsize=(10, 3.5*len(dss)), constrained_layout=True)

if len(dss) == 1:
    ax = [ax]

olr_cols = cols
bins = np.linspace(100, 350, 80)

for i, ds in enumerate(dss):
    olr = ds.olrtoa
    olr_mean = olr.mean(dim='Time')

    h, b = np.histogram(olr.values.ravel(), bins=bins)
    ax[i, 0].plot(0.5 * (b[1:] + b[:-1]), h / h.sum(), color=olr_cols[i])
    ax[i, 0].set_title(sim_names[i])
    ax[i, 0].set_xlabel(r'OLR (W m$^{-2}$)')
    ax[i, 0].set_ylabel('Normalised frequency')

    im = ax[i, 1].contourf(
        ds.longitude.values,
        ds.latitude.values,
        olr_mean.values,
        levels = np.linspace(200, 300, 151),
        norm=colors.PowerNorm(gamma=0.5), cmap='Blues',
        extend='both'
    )
    ax[i, 1].set_title(sim_names[i])
    ax[i, 1].set_xlabel('Longitude')
    ax[i, 1].set_ylabel('Latitude')

fig.colorbar(im, ax=ax[:, 1], location='right', label=r'OLR (W m$^{-2}$)')

figsave.save(fig, "olrtoa_hist_maps", number=11)


# In[ ]:


# Latitude-height section of zonal-mean kinetic energy

fig, ax = plt.subplots(len(dss), 1, figsize=(8, 3.5*len(dss)), constrained_layout=True, sharex=True)

if len(dss) == 1:
    ax = [ax]

for i, zds in enumerate(dss_zm):
    if 'ke' in zds.data_vars:
        ke = mean_if_present(zds.ke, ['Time', 'longitude']).transpose('z', 'latitude')
    
        z = vertical_coord(zds, ke)
    
        lat = zds.latitude.values
        lat2d = np.broadcast_to(lat[None, :], ke.shape)
    
        im = ax[i].contourf(lat2d, z/1000., ke.values, levels=np.linspace(0, np.nanmax(ke.values), 21), cmap='YlGnBu', extend='max')
        ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        ax[i].plot(zds.latitude.values, CPTs[i][0]/1000., color='k', lw=1, ls=':')
        ax[i].set_title(sim_names[i])
        ax[i].set_ylabel('Height (km)')
        ax[i].set_xlim(-90, 90)

ax[-1].set_xlabel('Latitude (degrees)')
fig.colorbar(im, ax=ax, location='right', label=r'Kinetic energy (m$^2$ s$^{-2}$)')

figsave.save(fig, "ke_latheight", number=12)


# In[ ]:
