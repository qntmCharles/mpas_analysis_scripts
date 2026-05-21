#!/usr/bin/env python
# coding: utf-8

"""
Equatorial wave diagnostics for QBO-like UTLS nudging experiments.

This script is intended to sit alongside:
  * mean_state_diagnostics.py
  * symmetry_convection_diagnostics.py
  * thermal_wind_scaled_response.py
  * matched_pattern_response.py

It deliberately does not assume that the aquaplanet contains an MJO analogue.
Instead it asks whether the precipitation/cloud response is associated with
particular convectively coupled equatorial wave (CCEW) bands, and whether the
u-nudged/T-nudged UTLS changes alter the precipitation response to those waves.

The filtering uses Straub/Kiladis-style k-f boxes via the low-level kf_filter.
For the current wave list this means:
The current analysis wave list is chosen for this simulation set:
  Kelvin:       k=1..14,    period=2.5..20 d,  equivalent depth=8..90 m
  EIG:          k=1..14,    period=1.8..4.5 d, equivalent depth=8..90 m
  IG symmetric: k=-15..-1,  period=1.4..2.6 d, equivalent depth=8..90 m
  IG antisym.:  k=-15..-1,  period=1.25..2 d,  equivalent depth=8..90 m
  AD sym/asym:  k=-28..-4,  period=5..20 d,    no depth constraint
  ER:           k=-10..-1,  period=10..40 d,   equivalent depth=8..90 m

AD is a broad westward disturbance filter rather than a single Matsuno normal
mode, so this script analyzes its symmetric and antisymmetric projections
separately.

The script caches filtered fields as NetCDF files in save_loc/wave_cache so that
figures can be regenerated without re-running the k-f filters.
"""

import os, sys, numbers, glob

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import colors
from matplotlib.lines import Line2D
from scipy.ndimage import uniform_filter1d
import xarray as xr
import dask

from diagnostic_helpers import *
from wavenumber_frequency_functions import spacetime_power, smooth_wavefreq

try:
    from waveFilter import kf_filter
except ModuleNotFoundError:
    # If this script is moved into a subdirectory, keep the failure message clear.
    raise ModuleNotFoundError(
        "Could not import kf_filter from waveFilter. "
        "Put waveFilter.py on PYTHONPATH or in the same scripts tree."
    )

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

save = True
save_loc = "/home/cwp29/tmp/mpas/crf_on/waves/"

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1:  wave_filtered_hovmollers_precip
#   2:  wave_power_fractions_and_ratios
#   3:  wave_amplitude_lag_correlations
#   4:  wave_conditioned_precip_olr_pw
#   5:  intense_precip_wave_attribution
#   6:  vertical_wave_power_profiles
#   7:  critical_level_phase_speed_diagnostics
#   8:  wave_filtered_eddy_fluxes_HEAVY
#   9:  symmetric_antisymmetric_wave_spectra
#       plus period-axis and wavenumber-phase-speed variants
figures_to_run = [1, 2, 3, 4, 5, 6, 7, 8, 9]
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

# Figure 8 is deliberately excluded from the default list because it filters
# latitude-resolved u and v. Add 8 to figures_to_run after the cheaper figures
# have run and cached their filtered fields.

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control", "u_nudge", "T_nudge"]
latlon_file = ["latlon.2000-*.nc", "latlon.2000-*.nc", "latlon.2000-*.nc"]
zm_latlon_file = ["zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc"]
sim_names = ["control", "nudged u", "nudged T"]
cols = ["k", "b", "r"]

# Use ndays of data ending at a fixed analysis date.
ndays = 200
dt = 6
analysis_end_date = "2000-11-21"
time_format = "time"  # or "int"
obs_per_day = int(round(24 / dt))

# User-defined filtering latitude band.  The cache filename includes this band,
# so changing it creates a separate set of filtered-wave files.
#
# This is the full band for symmetric/even signals.  Antisymmetric/odd signals
# are projected from the paired hemispheres (0..L and -L..0), where L is the
# larger absolute latitude bound below.
filter_lat_bounds = (-10, 10)
filter_avg_lat = True

# Heavier latitude-resolved filtering in figure 8 can use its own wider band.
eddy_flux_lat_bounds = (-20, 20)

# Analysis wave bands.  Names with suffixes are unique analysis IDs for parity
# projections of the same underlying k-f filter.
wave_names = ["Kelvin", "EIG", "IG_sym", "IG_antisym", "AD_sym", "AD_antisym", "ER"]

# Latitude projection by wave.  Choices are:
#   symmetric:       average over filter_lat_bounds, suitable for even modes
#   antisymmetric:   0.5 * (0..L - -L..0), suitable for odd modes
#   hemispheric_sum: 0.5 * (0..L + -L..0), paired hemispheres without assigning
#                    a Matsuno parity
#   north/south:     one hemisphere only
#
# Identifications used here:
#   Kelvin and ER are symmetric/even.
#   EIG is the eastward n=0 inertio-gravity branch, antisymmetric/odd.
#   IG_sym and IG_antisym separate the westward n=1 and n=2 IG branches.
#   AD is a broad westward disturbance band, not a single Matsuno normal mode,
#   so both paired same-sign and odd projections are analyzed.
wave_latitude_projection = {
    "Kelvin": "symmetric",
    "EIG": "antisymmetric",
    "IG_sym": "symmetric",
    "IG_antisym": "antisymmetric",
    "AD_sym": "hemispheric_sum",
    "AD_antisym": "antisymmetric",
    "AD": "hemispheric_sum",
    "ER": "symmetric",
}

# Variables used in the wave diagnostics.  Names are interpreted by
# get_filter_field(); cldfrac without a height is the vertical mean, cldfrac5km
# means the nearest cloud-fraction level to 5 km, and pressure-level fields are
# parsed from names like u850, omega250, T100, v850, qv700, RH500.
power_fields = [
    "Precip", "OLR", "PW", "cldfrac",
    "u850", "u250", "u100",
    "v850", "v250",
    "omega500", "omega250",
    "T250", "T100",
]

lag_target_fields = ["Precip", "OLR", "PW", "cldfrac"]
conditioned_fields = ["Precip", "OLR", "PW", "cldfrac"]
vertical_profile_waves = wave_names
vertical_profile_fields = ["u", "omega", "T", "cldfrac"]
vertical_profile_levels = [100, 150, 200, 250, 300, 500, 700, 850]
cldfrac_profile_heights_km = [2, 5, 8, 11, 14, 17]

# Figure 8 keeps latitude resolved, so hemispheric projections such as AD_sym
# and AD_antisym are not applied there. Use the underlying AD filter once.
eddy_flux_waves = ["Kelvin", "EIG", "IG_sym", "IG_antisym", "AD", "ER"]
eddy_flux_levels = [100, 250, 850]

spectrum_field = "Precip"
spectrum_lat_bounds = filter_lat_bounds
spectrum_segment_days = 96
spectrum_overlap_days = 60
spectrum_wavenumber_limits = (-30, 16)
spectrum_frequency_limits = (0, 0.6)
spectrum_period_reference_days = [3.0, 6.0, 30.0]
spectrum_phase_speed_lines_ms = [5.0, 10.0, 20.0, 40.0]
spectrum_phase_speed_limits = (-160.0, 160.0)
spectrum_phase_speed_step = 2.0
spectrum_outline_colors = {
    "Kelvin": "tab:orange",
    "EIG": "tab:red",
    "IG_sym": "tab:pink",
    "IG_antisym": "tab:purple",
    "AD_sym": "tab:green",
    "AD_antisym": "tab:olive",
    "ER": "tab:blue",
}
earth_radius_m = 6.371e6

# Input/output cache for k-f filtered fields.
cache_dir = '/store/atmos-adk33/cwp29/cache/'
input_cache_dir = os.path.join(cache_dir, "filter_inputs")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(input_cache_dir, exist_ok=True)

# Avoid giant chunks when selecting latitude/longitude subsets.
dask.config.set({"array.slicing.split_large_chunks": True})

needed_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "olrtoa",
    "cldfrac",
    "uzonal_isobaric", "umeridional_isobaric", "omg_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric",
    "theta", "pressure", "zgrid",
]

# Keep full lat-lon fields chunked in both time and space.  The k-f filters
# force loading of tropical lat/lon arrays, but spatial chunks still help for
# preliminary selection and any non-filtered diagnostics.
latlon_chunks = {"Time": 8, "latitude": 60, "longitude": 120}
zm_chunks = {"Time": -1}


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def run_all_figures():
    return all_figures_requested(figures_to_run)


# -----------------------------------------------------------------------------
# Load data
# -----------------------------------------------------------------------------

dss = []
dss_zm = []
for i, d in enumerate(data_locs):
    ds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks=latlon_chunks,
        add_cldfrac_file=True,
    )
    ds = standardize_latlon_dataset(ds, needed_vars)

    zds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=zm_latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks=zm_chunks,
        prefer_pattern=True,
        add_cldfrac_file=False,
    )
    zds = standardize_latlon_dataset(zds, needed_vars)

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)


# -----------------------------------------------------------------------------
# Wave-box metadata and helper functions
# -----------------------------------------------------------------------------

# This dictionary drives filtering, labels, phase-speed ranges, and spectrum
# outlines. AD is a westward disturbance filter rather than a single equatorial
# normal mode.
mis = -999.0
WAVE_BOXES = {
    "Kelvin": dict(
        k=(1, 14), period=(2.5, 20.0), depth=(8, 90),
        label="Kelvin", filter_name="Kelvin", components=("symmetric",),
    ),
    "EIG": dict(
        k=(1, 14), period=(1.8, 4.5), depth=(8, 90),
        label="EIG", filter_name="IG0", components=("antisymmetric",),
    ),
    "IG_sym": dict(
        k=(-15, -1), period=(1.4, 2.6), depth=(8, 90),
        label="IG sym", filter_name="IG1", components=("symmetric",),
    ),
    "IG_antisym": dict(
        k=(-15, -1), period=(1.25, 2.0), depth=(8, 90),
        label="IG antisym", filter_name="IG2", components=("antisymmetric",),
    ),
    "AD_sym": dict(
        k=(-28, -4), period=(5.0, 20.0), depth=(mis, mis),
        label="AD sym", filter_name="AD", components=("symmetric",),
    ),
    "AD_antisym": dict(
        k=(-28, -4), period=(5.0, 20.0), depth=(mis, mis),
        label="AD antisym", filter_name="AD", components=("antisymmetric",),
    ),
    "AD": dict(
        k=(-28, -4), period=(5.0, 20.0), depth=(mis, mis),
        label="AD", filter_name="AD", components=("symmetric", "antisymmetric"),
    ),
    "ER": dict(
        k=(-10, -1), period=(10.0, 40.0), depth=(8, 90),
        label="ER", filter_name="ER", components=("symmetric",),
    ),
}


def wave_label(wave_name):
    return WAVE_BOXES.get(wave_name, {}).get("label", wave_name)


def safe_name(x):
    return str(x).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


def precip_rate(ds):
    pr = precip_rate_from_accum(total_precip_accum(ds), dt=dt)
    pr.name = "Precip"
    pr.attrs["units"] = "mm day-1"
    return finite_or_nan(pr)


def sel_time_like(da, ref):
    return da.sel(Time=ref.Time)


def cldfrac_level_field_name(height_km):
    label = f"{float(height_km):g}".replace(".", "p")
    return f"cldfrac{label}km"


def parse_cldfrac_height_km(field_name):
    if field_name == "cldfrac":
        return None
    prefix = "cldfrac"
    suffix = "km"
    if not (field_name.startswith(prefix) and field_name.endswith(suffix)):
        raise ValueError(f"Unrecognised cloud-fraction field name: {field_name}")
    text = field_name[len(prefix):-len(suffix)].replace("p", ".")
    if not text:
        raise ValueError(f"No height supplied in {field_name}")
    return float(text)


def available_cldfrac_times(cf):
    """Return times present in the reindexed cloud-fraction file."""
    indexers = {dim: 0 for dim in cf.dims if dim != "Time"}
    valid = cf.isel(indexers).notnull().load()
    return valid.Time.values[np.asarray(valid.values, dtype=bool)]


def sel_common_times(da, ref):
    common = da.Time.to_index().intersection(ref.Time.to_index())
    return da.sel(Time=common)


def get_cldfrac_field(ds, field_name):
    if "cldfrac" not in ds.data_vars:
        raise ValueError("cldfrac is not available in this dataset")

    cf = finite_or_nan(ds.cldfrac)
    zdim = vertical_dim(cf)
    if zdim != "z":
        cf = cf.rename({zdim: "z"})

    height_km = parse_cldfrac_height_km(field_name)
    if height_km is None:
        out = cf.mean(dim="z", skipna=True)
        out.name = "cldfrac"
        return out.sel(Time=available_cldfrac_times(cf))

    if "z" not in cf.coords:
        raise ValueError(f"Cannot select {field_name}: cldfrac has no height coordinate")
    out = cf.sel(z=height_km * 1000.0, method="nearest")
    out.name = field_name
    out.attrs["selected_height_km"] = height_km
    return out.sel(Time=available_cldfrac_times(cf))


def get_filter_field(ds, field_name):
    """Return a lat-lon-time DataArray for a named field used by the wave filters."""
    pr = precip_rate(ds)

    if field_name == "Precip":
        return pr
    if field_name == "OLR":
        return finite_or_nan(ds.olrtoa).sel(Time=pr.Time)
    if field_name == "PW":
        return finite_or_nan(ds.precipw).sel(Time=pr.Time)
    if field_name == "cldfrac" or field_name.startswith("cldfrac"):
        return sel_common_times(get_cldfrac_field(ds, field_name), pr)

    # Pressure-level fields encoded as u850, v250, omega500, T100, qv700, RH500.
    prefixes = [
        ("omega", "omg_isobaric"),
        ("u", "uzonal_isobaric"),
        ("v", "umeridional_isobaric"),
        ("T", "t_isobaric"),
        ("qv", "qv_isobaric"),
        ("RH", "rh_isobaric"),
    ]
    for prefix, var in prefixes:
        if field_name.startswith(prefix):
            level_text = field_name[len(prefix):]
            if not level_text:
                raise ValueError(f"No pressure level supplied in {field_name}")
            plev = float(level_text)
            if var not in ds.data_vars:
                raise ValueError(f"{var} is not available in this dataset")
            out = finite_or_nan(ds[var]).sel(pres=plev, method="nearest").sel(Time=pr.Time)
            out.name = field_name
            return out

    raise ValueError(f"Unrecognised field name: {field_name}")


def field_units(field_name):
    if field_name == "Precip":
        return "mm day$^{-1}$"
    if field_name == "OLR":
        return "W m$^{-2}$"
    if field_name == "PW":
        return "kg m$^{-2}$"
    if field_name == "cldfrac" or field_name.startswith("cldfrac"):
        return "fraction"
    if field_name.startswith("omega"):
        return "Pa s$^{-1}$"
    if field_name.startswith("T"):
        return "K"
    if field_name.startswith("qv"):
        return "kg kg$^{-1}$"
    if field_name.startswith("RH"):
        return "%"
    return "m s$^{-1}$"


def netcdf_safe_attrs(attrs):
    """Convert Python/numpy bool attributes to NetCDF4-compatible metadata."""
    safe = {}
    for key, value in attrs.items():
        if isinstance(value, (bool, np.bool_)):
            safe[key] = int(value)
        else:
            safe[key] = value
    return safe


def netcdf_safe_dataset(ds):
    """Return a shallow copy with bool attrs converted before NetCDF writing."""
    out = ds.copy(deep=False)
    out.attrs = netcdf_safe_attrs(out.attrs)
    for name in out.variables:
        out[name].attrs = netcdf_safe_attrs(out[name].attrs)
    return out


def to_filter_input_dataset(da, obs_var="x"):
    """Standardise MPAS lat/lon DataArray to the file expected by kf_filter."""
    x = finite_or_nan(da)
    rename = {}
    if "latitude" in x.dims:
        rename["latitude"] = "lat"
    if "longitude" in x.dims:
        rename["longitude"] = "lon"
    x = x.rename(rename)
    x = x.transpose("Time", "lat", "lon")
    x.name = obs_var
    return x.to_dataset()


def write_filter_input_if_needed(ds, sim_index, field_name):
    """Cache a standard NetCDF input for kf_filter."""
    fname = os.path.join(input_cache_dir, f"{safe_name(sim_names[sim_index])}_{safe_name(field_name)}.nc")
    if os.path.exists(fname):
        return fname

    print(f"Writing filter input: {fname}")
    da = get_filter_field(ds, field_name)
    fds = to_filter_input_dataset(da, obs_var="x")
    encoding = {"x": {"dtype": "float32"}}
    fds.to_netcdf(fname, encoding=encoding)
    return fname


def normalise_filtered_dims(wave_da):
    """Rename filter output dimensions back to MPAS-like names."""
    out = wave_da
    rename = {}
    if "time" in out.dims:
        rename["time"] = "Time"
    if "lon" in out.dims:
        rename["lon"] = "longitude"
    if "lat" in out.dims:
        rename["lat"] = "latitude"
    if rename:
        out = out.rename(rename)
    return out


def lat_bound_label(value):
    """Stable, compact latitude labels for cache filenames."""
    value = float(value)
    if value.is_integer():
        return str(int(value))
    return f"{value:g}"


def normalise_lat_bounds(lat_bounds):
    if lat_bounds is None:
        lat_bounds = filter_lat_bounds
    if len(lat_bounds) != 2:
        raise ValueError(f"lat_bounds must contain two values, got {lat_bounds}")
    lo, hi = float(lat_bounds[0]), float(lat_bounds[1])
    if lo == hi:
        raise ValueError(f"lat_bounds must span a non-zero latitude range, got {lat_bounds}")
    if lo > hi:
        lo, hi = hi, lo
    return lo, hi


def lat_bounds_attr(lat_bounds):
    lo, hi = normalise_lat_bounds(lat_bounds)
    return f"{lat_bound_label(lo)},{lat_bound_label(hi)}"


def lat_bounds_cache_tag(lat_bounds):
    lo, hi = normalise_lat_bounds(lat_bounds)
    return f"{lat_bound_label(lo)}_{lat_bound_label(hi)}".replace("-", "m").replace(".", "p")


def hemisphere_bounds_from_lat_bounds(lat_bounds):
    """Return paired south/north bounds using the larger absolute bound."""
    lo, hi = normalise_lat_bounds(lat_bounds)
    half_width = max(abs(lo), abs(hi))
    return (-half_width, 0.0), (0.0, half_width)


def lat_projection_for_wave(wave_name, projection=None):
    """Return the configured latitude projection for a wave/disturbance band."""
    valid = {"symmetric", "antisymmetric", "hemispheric_sum", "north", "south", "direct", "latitude_resolved"}
    if projection is None:
        projection = wave_latitude_projection.get(wave_name, "symmetric")
    if projection not in valid:
        raise ValueError(f"Unknown latitude projection {projection!r}; choose from {sorted(valid)}")
    return projection


def filtered_wave_cache_path(sim_index, field_name, wave_name, lat_bounds, avgLat, projection):
    lat_tag = lat_bounds_cache_tag(lat_bounds)
    avg_tag = "avgLat" if avgLat else "allLat"
    cache_name = (
        f"{safe_name(sim_names[sim_index])}_{safe_name(field_name)}_{safe_name(wave_name)}_"
        f"{lat_tag}_{avg_tag}_{safe_name(projection)}.nc"
    )
    return os.path.join(cache_dir, cache_name)


def wave_cache_matches(wave_da, field_name, wave_name, avgLat, lat_bounds, projection):
    avg_attr = wave_da.attrs.get("avgLat", -1)
    filter_name = WAVE_BOXES[wave_name].get("filter_name", wave_name)
    return (
        wave_da.attrs.get("field") == field_name
        and wave_da.attrs.get("wave_name") == wave_name
        and wave_da.attrs.get("filter_name") == filter_name
        and int(avg_attr) == int(avgLat)
        and wave_da.attrs.get("filter_lat_bounds") == lat_bounds_attr(lat_bounds)
        and wave_da.attrs.get("latitude_projection") == projection
    )


def read_cached_wave(cache_path, field_name, wave_name, avgLat, lat_bounds, projection, force=False):
    if force or not os.path.exists(cache_path):
        return None
    try:
        cached = xr.open_dataset(cache_path)
        wave_cached = cached["wave"]
        if wave_cache_matches(wave_cached, field_name, wave_name, avgLat, lat_bounds, projection):
            return wave_cached
        print(f"Regenerating stale cached wave file {cache_path}")
        cached.close()
        os.remove(cache_path)
    except Exception as exc:
        print(f"Removing unreadable cached wave file {cache_path}: {exc}")
        try:
            os.remove(cache_path)
        except OSError:
            pass
    return None


def write_wave_cache(wave, cache_path):
    tmp_cache_path = cache_path + ".tmp"
    netcdf_safe_dataset(wave.to_dataset()).to_netcdf(tmp_cache_path)
    os.replace(tmp_cache_path, cache_path)


def filter_input_with_wave_box(input_path, obs_var, wave_name, lat_bounds, avgLat):
    """Run the low-level Wheeler-Kiladis k-f filter for a configured wave box."""
    box = WAVE_BOXES[wave_name]
    filter_name = box.get("filter_name", wave_name)
    k0, k1 = box["k"]
    p0, p1 = box["period"]
    h0, h1 = box["depth"]

    with xr.open_dataset(input_path) as infile:
        obs_time = infile.Time.load()
        lon = infile.lon.load()
        obs_data = infile[obs_var].sel(lat=slice(*lat_bounds))

        # Match the original MPAS aqua helper padding, which reduces
        # wrap-around artifacts before the filtered field is trimmed back.
        filt_time = np.arange(0, len(obs_time) + 702)

        if avgLat:
            obs_data = obs_data.mean("lat")
            filt_data = np.zeros((len(filt_time), 1, len(lon)), dtype="f")
            filt_data[0:obs_data.shape[0], 0, :] = obs_data.values
            coords = [obs_time, lon]
            dims = ["time", "lon"]
        else:
            lat = obs_data.lat.load()
            filt_data = np.zeros((len(filt_time), len(lat), len(lon)), dtype="f")
            filt_data[0:obs_data.shape[0], :, :] = obs_data.values
            coords = [obs_time, lat, lon]
            dims = ["time", "lat", "lon"]

    wave = np.zeros_like(filt_data)
    for lat_counter in range(filt_data.shape[1]):
        wave[:, lat_counter, :] = kf_filter(
            filt_data[:, lat_counter, :],
            obs_per_day,
            p0, p1,
            k0, k1,
            h0, h1,
            filter_name,
        )

    if avgLat:
        out_values = wave[0:len(coords[0]), 0, :]
    else:
        out_values = wave[0:len(coords[0]), :, :]
    return xr.DataArray(out_values, coords=coords, dims=dims)


def direct_filtered_wave(ds, sim_index, field_name, wave_name, avgLat, lat_bounds, projection, force=False):
    """
    Return a directly k-f filtered field for one latitude band.

    Higher-level latitude projection logic composes these direct filtered fields
    where needed.
    """
    if wave_name not in WAVE_BOXES:
        raise ValueError(f"{wave_name} is not in the supported wave list: {list(WAVE_BOXES)}")
    lat_bounds = normalise_lat_bounds(lat_bounds)

    cache_path = filtered_wave_cache_path(sim_index, field_name, wave_name, lat_bounds, avgLat, projection)
    cached_wave = read_cached_wave(cache_path, field_name, wave_name, avgLat, lat_bounds, projection, force=force)
    if cached_wave is not None:
        return cached_wave

    input_path = write_filter_input_if_needed(ds, sim_index, field_name)
    print(
        f"Filtering {sim_names[sim_index]} {field_name} {wave_name} "
        f"lat={lat_bounds} avgLat={avgLat} projection={projection}"
    )
    wave = filter_input_with_wave_box(input_path, "x", wave_name, lat_bounds, avgLat)
    wave = normalise_filtered_dims(wave)
    wave.name = "wave"
    wave.attrs.update(
        field=field_name,
        wave_name=wave_name,
        filter_name=WAVE_BOXES[wave_name].get("filter_name", wave_name),
        source_simulation=sim_names[sim_index],
        filter_lat_min=lat_bounds[0],
        filter_lat_max=lat_bounds[1],
        filter_lat_bounds=lat_bounds_attr(lat_bounds),
        latitude_projection=projection,
        avgLat=int(avgLat),
        units=field_units(field_name),
    )
    write_wave_cache(wave, cache_path)
    return xr.open_dataset(cache_path)["wave"]


def projected_filtered_wave(ds, sim_index, field_name, wave_name, lat_bounds, projection, force=False):
    """Compose paired-hemisphere filtered fields for an avgLat=True projection."""
    lat_bounds = normalise_lat_bounds(lat_bounds)
    cache_path = filtered_wave_cache_path(sim_index, field_name, wave_name, lat_bounds, True, projection)
    cached_wave = read_cached_wave(cache_path, field_name, wave_name, True, lat_bounds, projection, force=force)
    if cached_wave is not None:
        return cached_wave

    south_bounds, north_bounds = hemisphere_bounds_from_lat_bounds(lat_bounds)
    north = direct_filtered_wave(
        ds, sim_index, field_name, wave_name, True, north_bounds,
        f"{projection}_north_component", force=force
    )
    south = direct_filtered_wave(
        ds, sim_index, field_name, wave_name, True, south_bounds,
        f"{projection}_south_component", force=force
    )
    north, south = xr.align(north, south, join="inner")

    if projection == "antisymmetric":
        # Equivalent to adding the southern signal after flipping its odd sign.
        wave = 0.5 * (north - south)
        combination = "0.5*(north-south)"
    elif projection == "hemispheric_sum":
        wave = 0.5 * (north + south)
        combination = "0.5*(north+south)"
    else:
        raise ValueError(f"Projection {projection!r} does not use paired hemispheres")

    wave = wave.load()
    wave.name = "wave"
    wave.attrs.update(
        field=field_name,
        wave_name=wave_name,
        filter_name=WAVE_BOXES[wave_name].get("filter_name", wave_name),
        source_simulation=sim_names[sim_index],
        filter_lat_min=lat_bounds[0],
        filter_lat_max=lat_bounds[1],
        filter_lat_bounds=lat_bounds_attr(lat_bounds),
        latitude_projection=projection,
        hemisphere_combination=combination,
        south_component_bounds=lat_bounds_attr(south_bounds),
        north_component_bounds=lat_bounds_attr(north_bounds),
        avgLat=1,
        units=field_units(field_name),
    )
    write_wave_cache(wave, cache_path)
    return xr.open_dataset(cache_path)["wave"]


def get_filtered_wave(ds, sim_index, field_name, wave_name, avgLat=True, lat_bounds=None, projection=None, force=False):
    """
    Return a k-f filtered field using the configured Wheeler-Kiladis wave box.

    The filtered output is cached.  With avgLat=True, the latitude band is first
    projected according to wave_latitude_projection unless projection is
    supplied explicitly.  With avgLat=False, the latitude-resolved field is
    filtered directly and no parity projection is applied.
    """
    lat_bounds = normalise_lat_bounds(lat_bounds)
    if not avgLat:
        return direct_filtered_wave(
            ds, sim_index, field_name, wave_name, False, lat_bounds,
            "latitude_resolved", force=force
        )

    projection = lat_projection_for_wave(wave_name, projection)
    if projection == "symmetric":
        return direct_filtered_wave(ds, sim_index, field_name, wave_name, True, lat_bounds, projection, force=force)
    if projection == "north":
        _, north_bounds = hemisphere_bounds_from_lat_bounds(lat_bounds)
        return direct_filtered_wave(ds, sim_index, field_name, wave_name, True, north_bounds, projection, force=force)
    if projection == "south":
        south_bounds, _ = hemisphere_bounds_from_lat_bounds(lat_bounds)
        return direct_filtered_wave(ds, sim_index, field_name, wave_name, True, south_bounds, projection, force=force)
    if projection in ("antisymmetric", "hemispheric_sum"):
        return projected_filtered_wave(ds, sim_index, field_name, wave_name, lat_bounds, projection, force=force)
    if projection == "direct":
        return direct_filtered_wave(ds, sim_index, field_name, wave_name, True, lat_bounds, projection, force=force)
    raise ValueError(f"Unhandled latitude projection {projection!r}")


def project_latitude_field(da, projection, lat_bounds=None):
    """Project an unfiltered latitude field to match avgLat=True wave output."""
    if lat_bounds is None:
        lat_bounds = filter_lat_bounds
    lat_bounds = normalise_lat_bounds(lat_bounds)
    projection = lat_projection_for_wave("", projection)

    if projection in ("symmetric", "direct"):
        x = da.sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
        return x.mean(dim="latitude", skipna=True)
    if projection == "north":
        _, north_bounds = hemisphere_bounds_from_lat_bounds(lat_bounds)
        x = da.sel(latitude=slice(north_bounds[0], north_bounds[1]))
        return x.mean(dim="latitude", skipna=True)
    if projection == "south":
        south_bounds, _ = hemisphere_bounds_from_lat_bounds(lat_bounds)
        x = da.sel(latitude=slice(south_bounds[0], south_bounds[1]))
        return x.mean(dim="latitude", skipna=True)

    south_bounds, north_bounds = hemisphere_bounds_from_lat_bounds(lat_bounds)
    north = da.sel(latitude=slice(north_bounds[0], north_bounds[1])).mean(dim="latitude", skipna=True)
    south = da.sel(latitude=slice(south_bounds[0], south_bounds[1])).mean(dim="latitude", skipna=True)
    north, south = xr.align(north, south, join="inner")
    if projection == "antisymmetric":
        return 0.5 * (north - south)
    if projection == "hemispheric_sum":
        return 0.5 * (north + south)
    raise ValueError(f"Unhandled latitude projection {projection!r}")


def latmean_for_filter_field(ds, field_name, lat_bounds=None, projection="symmetric", wave_name=None):
    """Unfiltered counterpart to avgLat=True wave output: dimensions (Time, longitude)."""
    if wave_name is not None:
        projection = lat_projection_for_wave(wave_name)
    da = get_filter_field(ds, field_name)
    return project_latitude_field(da, projection, lat_bounds=lat_bounds)


def anomaly_variance_latmean(ds, field_name, lat_bounds=None, projection="symmetric", wave_name=None):
    x = latmean_for_filter_field(ds, field_name, lat_bounds=lat_bounds, projection=projection, wave_name=wave_name)
    x = x - x.mean(dim="Time", skipna=True)
    return float((x**2).mean(dim=("Time", "longitude"), skipna=True).values)


def wave_variance(wave_da):
    dims = [d for d in ("Time", "latitude", "longitude") if d in wave_da.dims]
    return float((wave_da**2).mean(dim=dims, skipna=True).values)


def wave_amplitude_index(wave_da):
    """RMS wave amplitude as a function of time."""
    dims = [d for d in ("latitude", "longitude") if d in wave_da.dims]
    amp = np.sqrt((wave_da**2).mean(dim=dims, skipna=True))
    amp.name = "wave_amplitude"
    return amp


def area_mean_filter_band(ds, field_name, lat_bounds=None, projection="symmetric", wave_name=None):
    """Time series of an unfiltered field over the filtering latitude band."""
    x = latmean_for_filter_field(ds, field_name, lat_bounds=lat_bounds, projection=projection, wave_name=wave_name)
    return x.mean(dim="longitude", skipna=True)


def lag_corr(x, y, max_lag_steps):
    """Correlation of x(t) with y(t+lag); positive lag means x leads y."""
    xv = np.asarray(x)
    yv = np.asarray(y)
    lags = np.arange(-max_lag_steps, max_lag_steps + 1)
    out = np.full(lags.size, np.nan)
    for j, lag in enumerate(lags):
        if lag < 0:
            xs, ys = xv[-lag:], yv[:lag]
        elif lag > 0:
            xs, ys = xv[:-lag], yv[lag:]
        else:
            xs, ys = xv, yv
        ok = np.isfinite(xs) & np.isfinite(ys)
        if np.count_nonzero(ok) >= 4:
            out[j] = np.corrcoef(xs[ok], ys[ok])[0, 1]
    return lags, out


def get_dt_days_from_time(da):
    if issubclass(da.Time.values.dtype.type, numbers.Integral):
        return dt / 24.0
    return float(np.median(da.Time.diff("Time") / np.timedelta64(1, "D")))


def binned_mean_1d(x, y, bins, min_count=10):
    xv = np.asarray(x).ravel()
    yv = np.asarray(y).ravel()
    ok = np.isfinite(xv) & np.isfinite(yv)
    xv = xv[ok]
    yv = yv[ok]
    idx = np.digitize(xv, bins) - 1
    xc = 0.5 * (bins[:-1] + bins[1:])
    mean = np.full(xc.size, np.nan)
    p25 = np.full(xc.size, np.nan)
    p75 = np.full(xc.size, np.nan)
    count = np.zeros(xc.size, dtype=int)
    for k in range(xc.size):
        vals = yv[idx == k]
        count[k] = vals.size
        if vals.size >= min_count:
            mean[k] = np.nanmean(vals)
            p25[k] = np.nanpercentile(vals, 25)
            p75[k] = np.nanpercentile(vals, 75)
    return xc, mean, p25, p75, count


def wave_phase_speed_range(wave_name):
    """Return phase-speed min/central/max in m s-1 from k-period box."""
    box = WAVE_BOXES[wave_name]
    k0, k1 = box["k"]
    p0, p1 = box["period"]
    kvals = np.arange(int(k0), int(k1) + 1) if k0 <= k1 else np.arange(int(k0), int(k1) - 1, -1)
    kvals = kvals[kvals != 0]
    periods = np.array([p0, p1], dtype=float)
    a = 6.371e6
    speeds = []
    for k in kvals:
        for P in periods:
            speeds.append(np.sign(k) * 2.0 * np.pi * a / (abs(k) * P * 86400.0))
    speeds = np.asarray(speeds)
    # Central speed: use central period and central absolute wavenumber.
    k_c = np.sign(np.mean(kvals)) * np.mean(np.abs(kvals))
    P_c = 0.5 * (p0 + p1)
    c_c = np.sign(k_c) * 2.0 * np.pi * a / (abs(k_c) * P_c * 86400.0)
    return np.nanmin(speeds), c_c, np.nanmax(speeds)


def shallow_water_frequency_cpd(wave_name, zonal_wavenumber, equivalent_depth):
    """Approximate shallow-water modal frequency in cycles per day."""
    k = np.asarray(zonal_wavenumber, dtype=float) / 6.37e6
    c = np.sqrt(9.8 * float(equivalent_depth))
    beta = 2.28e-11
    filter_name = WAVE_BOXES.get(wave_name, {}).get("filter_name", wave_name)

    if filter_name == "Kelvin":
        omega = k * c
    elif filter_name == "ER":
        omega = -beta * k / (k**2 + 3.0 * beta / c)
    elif filter_name == "IG1":
        omega = np.sqrt(3.0 * beta * c + k**2 * c**2)
    elif filter_name == "IG2":
        omega = np.sqrt(5.0 * beta * c + k**2 * c**2)
    elif filter_name in ("MRG", "IG0"):
        with np.errstate(divide="ignore", invalid="ignore"):
            omega = np.where(
                k > 0,
                k * c * (0.5 + 0.5 * np.sqrt(1.0 + 4.0 * beta / (k**2 * c))),
                k * c * (0.5 - 0.5 * np.sqrt(1.0 + 4.0 * beta / (k**2 * c))),
            )
        omega = np.where(np.isclose(k, 0.0), np.sqrt(beta * c), omega)
    else:
        raise ValueError(f"No dispersion-outline helper for {wave_name}")

    return np.abs(omega) * 86400.0 / (2.0 * np.pi)


def wave_filter_mask_on_spectrum(wavenumber, frequency, wave_name):
    """Boolean mask for a configured wave-filter region on a spectrum grid."""
    box = WAVE_BOXES[wave_name]
    k0, k1 = box["k"]
    p0, p1 = box["period"]
    k_min, k_max = min(k0, k1), max(k0, k1)
    f_min = 1.0 / max(p0, p1)
    f_max = 1.0 / min(p0, p1)

    kgrid, fgrid = np.meshgrid(np.asarray(wavenumber), np.asarray(frequency), indexing="ij")
    mask = (
        (kgrid >= k_min)
        & (kgrid <= k_max)
        & (np.abs(fgrid) >= f_min)
        & (np.abs(fgrid) <= f_max)
    )

    h0, h1 = box["depth"]
    if h0 != mis and h1 != mis:
        f0 = shallow_water_frequency_cpd(wave_name, kgrid, h0)
        f1 = shallow_water_frequency_cpd(wave_name, kgrid, h1)
        f_low = np.minimum(f0, f1)
        f_high = np.maximum(f0, f1)
        mask &= (np.abs(fgrid) >= f_low) & (np.abs(fgrid) <= f_high)

    return mask


def period_days_from_frequency(frequency):
    freq = np.asarray(frequency, dtype=float)
    out = np.full_like(freq, np.nan, dtype=float)
    ok = freq > 0
    out[ok] = 1.0 / freq[ok]
    return out


def frequency_from_zonal_phase_speed(wavenumber, phase_speed_ms):
    return (
        np.asarray(phase_speed_ms, dtype=float)
        * np.asarray(wavenumber, dtype=float)
        * 86400.0
        / (2.0 * np.pi * earth_radius_m)
    )


def zonal_phase_speed_from_frequency(wavenumber, frequency):
    with np.errstate(divide="ignore", invalid="ignore"):
        return (
            2.0 * np.pi * earth_radius_m
            * np.asarray(frequency, dtype=float)
            / (np.asarray(wavenumber, dtype=float) * 86400.0)
        )


def transform_wavenumber_frequency_to_phase_speed(wavenumber, frequency, values, phase_speed_axis):
    """
    Interpolate values from (zonal wavenumber, frequency) to
    (zonal wavenumber, zonal phase speed).

    The plotted quantity is already a normalised power ratio, so this is a
    coordinate transform for visual diagnosis rather than a variance-preserving
    spectral-density transform.
    """
    wn = np.asarray(wavenumber, dtype=float)
    freq = np.asarray(frequency, dtype=float)
    vals = np.asarray(values, dtype=float)
    phase_speed_axis = np.asarray(phase_speed_axis, dtype=float)

    if vals.shape != (wn.size, freq.size):
        raise ValueError(
            f"Expected values with shape {(wn.size, freq.size)}, got {vals.shape}"
        )

    fmask = freq > 0
    freq_pos = freq[fmask]
    order = np.argsort(freq_pos)
    freq_pos = freq_pos[order]
    vals_pos = vals[:, fmask][:, order]

    kmask = wn != 0
    wn_out = wn[kmask]
    out = np.full((phase_speed_axis.size, wn_out.size), np.nan)
    source_indices = np.where(kmask)[0]
    for out_j, src_j in enumerate(source_indices):
        target_freq = frequency_from_zonal_phase_speed(wn[src_j], phase_speed_axis)
        valid = target_freq > 0
        out[valid, out_j] = np.interp(
            target_freq[valid],
            freq_pos,
            vals_pos[src_j, :],
            left=np.nan,
            right=np.nan,
        )
    return wn_out, phase_speed_axis, out


def phase_speed_axis_values():
    lo, hi = spectrum_phase_speed_limits
    step = spectrum_phase_speed_step
    return np.arange(lo, hi + 0.5 * step, step)


def add_period_reference_lines_frequency(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ymin, ymax = sorted((y0, y1))
    for period in spectrum_period_reference_days:
        freq = 1.0 / period
        if ymin <= freq <= ymax:
            ax.axhline(freq, color="0.25", lw=0.8, ls=":", alpha=0.8)
            ax.text(x1, freq, f"{period:g} d", ha="right", va="bottom",
                    fontsize=7, color="0.25")


def add_period_reference_lines_period(ax):
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    ymin, ymax = sorted((y0, y1))
    for period in spectrum_period_reference_days:
        if ymin <= period <= ymax:
            ax.axhline(period, color="0.25", lw=0.8, ls=":", alpha=0.8)
            ax.text(x1, period, f"{period:g} d", ha="right", va="bottom",
                    fontsize=7, color="0.25")


def add_phase_speed_lines_period_axis(ax, wavenumber):
    k = np.asarray(wavenumber, dtype=float)
    k = k[k != 0]
    y0, y1 = ax.get_ylim()
    ymin, ymax = sorted((y0, y1))
    for speed in spectrum_phase_speed_lines_ms:
        period = 2.0 * np.pi * earth_radius_m / (np.abs(k) * speed * 86400.0)
        ok = (period >= ymin) & (period <= ymax)
        if not np.any(ok):
            continue
        for branch in (k < 0, k > 0):
            branch_ok = ok & branch
            if np.any(branch_ok):
                ax.plot(k[branch_ok], period[branch_ok], color="0.35", lw=0.7,
                        ls="--", alpha=0.65)
        pos = ok & (k > 0)
        if np.any(pos):
            idx = np.where(pos)[0][-1]
            ax.text(k[idx], period[idx], f"{speed:g} m s$^{{-1}}$",
                    ha="right", va="bottom", fontsize=6, color="0.35")


def add_period_reference_lines_phase_speed(ax, wavenumber):
    k = np.asarray(wavenumber, dtype=float)
    k = k[k != 0]
    y0, y1 = ax.get_ylim()
    ymin, ymax = sorted((y0, y1))
    for period in spectrum_period_reference_days:
        c = 2.0 * np.pi * earth_radius_m / (k * period * 86400.0)
        ok = (c >= ymin) & (c <= ymax)
        if not np.any(ok):
            continue
        for branch in (k < 0, k > 0):
            branch_ok = ok & branch
            if np.any(branch_ok):
                ax.plot(k[branch_ok], c[branch_ok], color="0.25", lw=0.8,
                        ls=":", alpha=0.8)
        pos = ok & (k > 0)
        if np.any(pos):
            idx = np.where(pos)[0][-1]
            ax.text(k[idx], c[idx], f"{period:g} d",
                    ha="right", va="bottom", fontsize=7, color="0.25")


def add_wave_filter_region_outlines(ax, wavenumber, frequency, component):
    """Outline configured wave-filter regions that belong on one spectrum component."""
    for wname in wave_names:
        if component not in WAVE_BOXES[wname].get("components", ("symmetric", "antisymmetric")):
            continue
        mask = wave_filter_mask_on_spectrum(wavenumber, frequency, wname)
        if not np.any(mask):
            continue
        ax.contour(
            wavenumber,
            frequency,
            mask.T.astype(float),
            levels=[0.5],
            colors=[spectrum_outline_colors.get(wname, "0.1")],
            linewidths=1.4,
        )


def add_wave_filter_region_outlines_period(ax, wavenumber, frequency, component):
    """Outline configured filters on a period-vs-wavenumber spectrum."""
    freq = np.asarray(frequency, dtype=float)
    fmask = freq > 0
    freq_pos = freq[fmask]
    period = period_days_from_frequency(freq_pos)
    order = np.argsort(period)
    freq_plot = freq_pos[order]
    period_plot = period[order]
    for wname in wave_names:
        if component not in WAVE_BOXES[wname].get("components", ("symmetric", "antisymmetric")):
            continue
        mask = wave_filter_mask_on_spectrum(wavenumber, freq_plot, wname)
        if not np.any(mask):
            continue
        ax.contour(
            wavenumber,
            period_plot,
            mask.T.astype(float),
            levels=[0.5],
            colors=[spectrum_outline_colors.get(wname, "0.1")],
            linewidths=1.4,
        )


def add_wave_filter_region_outlines_phase_speed(ax, wavenumber, frequency, phase_speed_axis, component):
    """Outline configured filters on a phase-speed-vs-wavenumber spectrum."""
    for wname in wave_names:
        if component not in WAVE_BOXES[wname].get("components", ("symmetric", "antisymmetric")):
            continue
        mask = wave_filter_mask_on_spectrum(wavenumber, frequency, wname).astype(float)
        if not np.any(mask):
            continue
        wn_phase, c_phase, mask_phase = transform_wavenumber_frequency_to_phase_speed(
            wavenumber, frequency, mask, phase_speed_axis
        )
        if not np.any(np.isfinite(mask_phase)) or np.nanmax(mask_phase) < 0.5:
            continue
        ax.contour(
            wn_phase,
            c_phase,
            mask_phase,
            levels=[0.5],
            colors=[spectrum_outline_colors.get(wname, "0.1")],
            linewidths=1.4,
        )


def wave_filter_region_legend_handles():
    return [
        Line2D(
            [0], [0],
            color=spectrum_outline_colors.get(wname, "0.1"),
            lw=1.8,
            label=wave_label(wname),
        )
        for wname in wave_names
    ]


def spectrum_input_field(ds, field_name, lat_bounds):
    """Return a loaded lat-lon-time field with names expected by spacetime_power."""
    da = get_filter_field(ds, field_name)
    da = da.sel(latitude=slice(*normalise_lat_bounds(lat_bounds)))
    return da.rename({"latitude": "lat", "longitude": "lon"}).load()


# -----------------------------------------------------------------------------
# 1. Wave-filtered Hovmoller diagrams
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Which filter bands actually reconstruct coherent propagating precipitation?
#   * Does u_nudge increase a particular band, or simply broaden all wave-like
#     variability?
#   * Kelvin/EIG: eastward streaks. IG/AD/ER: mostly westward or mixed structure.

if should_run_figure(1):
    hov_field = "Precip"
    fields = {}
    vmax = 0.0
    for i, ds in enumerate(dss):
        for wname in wave_names:
            wav = get_filtered_wave(ds, i, hov_field, wname, avgLat=True)
            fields[(i, wname)] = wav
            try:
                vmax = max(vmax, float(np.nanpercentile(np.abs(wav.values), 98)))
            except Exception:
                pass
    if not np.isfinite(vmax) or vmax == 0.0:
        vmax = 1.0
    levels = symmetric_levels(vmax, ndiv=20)

    fig, ax = plt.subplots(len(dss), len(wave_names), figsize=(3.2 * len(wave_names), 2.3 * len(dss)),
                           constrained_layout=True, sharex=True, sharey=True)
    if len(dss) == 1:
        ax = np.array([ax])
    for i in range(len(dss)):
        for j, wname in enumerate(wave_names):
            wav = fields[(i, wname)]
            im = ax[i, j].contourf(wav.longitude, wav.Time, wav.values, levels=levels, cmap="bwr", extend="both")
            if i == 0:
                ax[i, j].set_title(wave_label(wname))
            if j == 0:
                ax[i, j].set_ylabel(sim_names[i])
            else:
                ax[i, j].set_yticklabels([])
            if i == len(dss) - 1:
                ax[i, j].set_xlabel("Longitude")
            annotate_panel_label(ax[i, j], sim_names[i] if j == 0 else wave_label(wname))
    add_symmetric_colorbar(fig, im, ax.ravel(), levels, label=f"{hov_field}' ({field_units(hov_field)})")
    figsave.save(fig, "wave_filtered_hovmollers_precip", number=1)
    print("Done with figure: wave_filtered_hovmollers_precip")


# -----------------------------------------------------------------------------
# 2. Wave-band variance fractions and nudged/control power ratios
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * A wave-band precipitation fraction tells you how much of the tropical
#     precip variance is attributable to each filtered band.
#   * Compare precipitation with dynamical fields. If u/omega/T wave power
#     increases but precipitation power does not, waves are modified aloft but
#     weakly coupled to rainfall. If precipitation power increases without a
#     dynamical increase, convective sensitivity has changed.

wave_metrics = None
if should_run_any([2, 3, 4, 5]):
    rows = []
    for i, ds in enumerate(dss):
        for field in power_fields:
            for wname in wave_names:
                projection = lat_projection_for_wave(wname)
                try:
                    total_var = anomaly_variance_latmean(
                        ds, field, lat_bounds=filter_lat_bounds, projection=projection
                    )
                    wav = get_filtered_wave(ds, i, field, wname, avgLat=True)
                    wvar = wave_variance(wav)
                    rows.append(dict(
                        simulation=sim_names[i],
                        simulation_index=i,
                        field=field,
                        wave=wname,
                        wave_label=wave_label(wname),
                        latitude_projection=projection,
                        total_variance=total_var,
                        wave_variance=wvar,
                        wave_fraction=wvar / total_var if total_var > 0 else np.nan,
                    ))
                except Exception as exc:
                    print(f"Skipping wave variance for {sim_names[i]} {field} {wname}: {exc}")
    wave_metrics = pd.DataFrame(rows)
    if len(wave_metrics) > 0:
        # Add nudged/control ratios field-by-field and wave-by-wave.
        control = wave_metrics[wave_metrics.simulation_index == 0].set_index(["field", "wave"])
        ratios = []
        for _, row in wave_metrics.iterrows():
            key = (row.field, row.wave)
            if key in control.index:
                cvar = control.loc[key, "wave_variance"]
                cfrac = control.loc[key, "wave_fraction"]
                ratios.append(dict(power_ratio_to_control=row.wave_variance / cvar if cvar > 0 else np.nan,
                                   fraction_ratio_to_control=row.wave_fraction / cfrac if cfrac > 0 else np.nan))
            else:
                ratios.append(dict(power_ratio_to_control=np.nan, fraction_ratio_to_control=np.nan))
        wave_metrics = pd.concat([wave_metrics.reset_index(drop=True), pd.DataFrame(ratios)], axis=1)
        print("\nWave-band variance metrics")
        print(wave_metrics.to_string(index=False))
        if save:
            wave_metrics.to_csv(os.path.join(save_loc, "wave_band_variance_metrics.csv"), index=False)

if should_run_figure(2) and wave_metrics is not None and len(wave_metrics) > 0:
    plot_fields = ["Precip", "OLR", "PW", "cldfrac", "u850", "omega500", "T100"]
    plot_fields = [f for f in plot_fields if f in set(wave_metrics.field)]

    fig, ax = plt.subplots(2, len(plot_fields), figsize=(3.2 * len(plot_fields), 6.0), constrained_layout=True)
    if len(plot_fields) == 1:
        ax = ax[:, None]

    for j, field in enumerate(plot_fields):
        sub = wave_metrics[wave_metrics.field == field]
        x = np.arange(len(wave_names))
        width = 0.8 / len(dss)
        for i, sim in enumerate(sim_names):
            s = sub[sub.simulation == sim].set_index("wave").reindex(wave_names)
            ax[0, j].bar(x + (i - (len(dss)-1)/2) * width, s.wave_fraction.values, width,
                         color=cols[i], label=sim if j == 0 else None)
            ax[1, j].bar(x + (i - (len(dss)-1)/2) * width, s.power_ratio_to_control.values, width,
                         color=cols[i])
        ax[0, j].set_title(field)
        ax[0, j].set_xticks(x)
        ax[0, j].set_xticklabels([])
        ax[1, j].set_xticks(x)
        ax[1, j].set_xticklabels([wave_label(w) for w in wave_names], rotation=35, ha="right")
        ax[1, j].axhline(1, color="0.6", lw=0.7)
        ax[0, j].set_yscale("log")
        #ax[1, j].set_yscale("log")
        ax[0, j].grid(alpha=0.2)
        ax[1, j].grid(alpha=0.2)
    ax[0, 0].set_ylabel("Wave-band variance / total variance")
    ax[1, 0].set_ylabel("Wave power ratio to control")
    ax[0, 0].legend(loc="best")
    figsave.save(fig, "wave_power_fractions_and_ratios", number=2)
    print("Done with figure: wave_power_fractions_and_ratios")


# -----------------------------------------------------------------------------
# 3. Lagged coupling between wave amplitude and domain-mean fields
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Positive lag means wave amplitude leads the target field.
#   * If u_nudge increases correlation at positive lags, waves may be triggering
#     or organising convection more effectively.
#   * If correlations are similar but wave power differs, occurrence/intensity
#     rather than coupling is the dominant change.

if should_run_figure(3):
    lag_days_max = 10.0
    lag_rows = []
    fig, ax = plt.subplots(len(wave_names), len(lag_target_fields),
                           figsize=(3.6 * len(lag_target_fields), 2.3 * len(wave_names)),
                           constrained_layout=True, sharex=True, sharey=True)
    if len(wave_names) == 1:
        ax = ax[None, :]
    if len(lag_target_fields) == 1:
        ax = ax[:, None]

    for wj, wname in enumerate(wave_names):
        projection = lat_projection_for_wave(wname)
        for tj, target in enumerate(lag_target_fields):
            for i, ds in enumerate(dss):
                try:
                    wav = get_filtered_wave(ds, i, "Precip", wname, avgLat=True)
                    amp = wave_amplitude_index(wav)
                    ts = area_mean_filter_band(
                        ds, target, lat_bounds=filter_lat_bounds, projection=projection
                    )
                    amp, ts = xr.align(amp, ts, join="inner")
                    if amp.sizes.get("Time", 0) < 3:
                        continue
                    dt_days_here = get_dt_days_from_time(amp)
                    max_steps = max(1, int(round(lag_days_max / dt_days_here)))
                    lags, corr = lag_corr(amp.values, ts.values, max_lag_steps=max_steps)
                    lag_days = lags * dt_days_here
                    ax[wj, tj].plot(lag_days, corr, color=cols[i], label=sim_names[i] if (wj == 0 and tj == 0) else None)
                    for lg, cc in zip(lag_days, corr):
                        lag_rows.append(dict(simulation=sim_names[i], wave=wname,
                                             wave_label=wave_label(wname),
                                             latitude_projection=projection, target=target,
                                             lag_days=lg, correlation=cc))
                except Exception as exc:
                    print(f"Skipping lag correlation {sim_names[i]} {wname} {target}: {exc}")
            ax[wj, tj].axhline(0, color="0.6", lw=0.6)
            ax[wj, tj].axvline(0, color="0.6", lw=0.6)
            ax[wj, tj].grid(alpha=0.2)
            if wj == 0:
                ax[wj, tj].set_title(target)
            if tj == 0:
                ax[wj, tj].set_ylabel(wave_label(wname))
            if wj == len(wave_names) - 1:
                ax[wj, tj].set_xlabel("Lag (days; + means wave leads)")
    ax[0, 0].legend(loc="best")
    figsave.save(fig, "wave_amplitude_lag_correlations", number=3)
    lag_df = pd.DataFrame(lag_rows)
    if save and len(lag_df) > 0:
        lag_df.to_csv(os.path.join(save_loc, "wave_amplitude_lag_correlations.csv"), index=False)
    print("Done with figure: wave_amplitude_lag_correlations")


# -----------------------------------------------------------------------------
# 4. Conditional mean fields as a function of wave amplitude
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * This separates 'more waves' from 'stronger convective response to the same
#     wave amplitude'.
#   * Curves shifted upward in precipitation at fixed normalised wave amplitude
#     imply changed convective sensitivity.

if should_run_figure(4):
    cond_rows = []
    amp_bins = np.linspace(0, 3.5, 22)
    fig, ax = plt.subplots(len(wave_names), len(conditioned_fields),
                           figsize=(3.6 * len(conditioned_fields), 2.3 * len(wave_names)),
                           constrained_layout=True, sharex=True)
    if len(wave_names) == 1:
        ax = ax[None, :]
    if len(conditioned_fields) == 1:
        ax = ax[:, None]

    for wj, wname in enumerate(wave_names):
        projection = lat_projection_for_wave(wname)
        for tj, target in enumerate(conditioned_fields):
            for i, ds in enumerate(dss):
                try:
                    wav = get_filtered_wave(ds, i, "Precip", wname, avgLat=True)
                    # Local wave amplitude in longitude-time, normalised by its std.
                    a = np.abs(wav)
                    a_std = float(a.std(dim=("Time", "longitude"), skipna=True).values)
                    if not np.isfinite(a_std) or a_std == 0:
                        continue
                    a_norm = a / a_std
                    y = latmean_for_filter_field(
                        ds, target, lat_bounds=filter_lat_bounds, projection=projection
                    )
                    a_norm, y = xr.align(a_norm, y, join="inner")
                    if a_norm.sizes.get("Time", 0) == 0:
                        continue
                    xc, ym, y25, y75, count = binned_mean_1d(a_norm.values, y.values, amp_bins, min_count=20)
                    ax[wj, tj].plot(xc, ym, color=cols[i], label=sim_names[i] if (wj == 0 and tj == 0) else None)
                    ax[wj, tj].fill_between(xc, y25, y75, color=cols[i], alpha=0.12, linewidth=0)
                    for xx, yy, cnum in zip(xc, ym, count):
                        cond_rows.append(dict(simulation=sim_names[i], wave=wname,
                                              wave_label=wave_label(wname),
                                              latitude_projection=projection, target=target,
                                              amp_norm_bin=xx, mean=yy, count=cnum))
                except Exception as exc:
                    print(f"Skipping conditional curve {sim_names[i]} {wname} {target}: {exc}")
            if wj == 0:
                ax[wj, tj].set_title(target)
            if tj == 0:
                ax[wj, tj].set_ylabel(wave_label(wname))
            if wj == len(wave_names) - 1:
                ax[wj, tj].set_xlabel(r"$|P'_W| / \sigma(|P'_W|)$")
            ax[wj, tj].grid(alpha=0.2)
    ax[0, 0].legend(loc="best")
    figsave.save(fig, "wave_conditioned_precip_olr_pw", number=4)
    cond_df = pd.DataFrame(cond_rows)
    if save and len(cond_df) > 0:
        cond_df.to_csv(os.path.join(save_loc, "wave_conditioned_precip_olr_pw.csv"), index=False)
    print("Done with figure: wave_conditioned_precip_olr_pw")


# -----------------------------------------------------------------------------
# 5. Intense-precipitation attribution to wave bands
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Which wave band is largest at high-precipitation longitudes/times?
#   * If u_nudge increases intense precipitation only when Kelvin/EIG/IG/AD/ER
#     amplitude is large, that wave band is a candidate mediator.

if should_run_figure(5):
    attribution_rows = []
    percentile = 95.0
    for i, ds in enumerate(dss):
        try:
            pr = latmean_for_filter_field(ds, "Precip", lat_bounds=filter_lat_bounds, projection="direct")
            pr_vals = np.asarray(pr.values)
            threshold = np.nanpercentile(pr_vals, percentile)
            events = pr >= threshold
            norm_waves = {}
            for wname in wave_names:
                wav = get_filtered_wave(ds, i, "Precip", wname, avgLat=True).sel(Time=pr.Time)
                std = float(wav.std(dim=("Time", "longitude"), skipna=True).values)
                norm_waves[wname] = np.abs(wav / std) if std > 0 else xr.zeros_like(wav)
            # Stack wave amplitudes into shape (wave, Time, longitude).
            stack = xr.concat([norm_waves[w] for w in wave_names], dim=xr.DataArray(wave_names, dims="wave", name="wave"))
            winner_index = stack.argmax(dim="wave")
            winner = xr.DataArray(np.array(wave_names, dtype=object)[winner_index.values], dims=winner_index.dims, coords=winner_index.coords)
            for wname in wave_names:
                projection = lat_projection_for_wave(wname)
                frac = float(((winner == wname).where(events)).sum().values / events.sum().values)
                mean_amp = float(norm_waves[wname].where(events).mean().values)
                attribution_rows.append(dict(simulation=sim_names[i], wave=wname,
                                             wave_label=wave_label(wname),
                                             latitude_projection=projection, intense_event_fraction=frac,
                                             mean_normalised_amplitude_at_events=mean_amp,
                                             precip_percentile=percentile, precip_threshold=threshold))
        except Exception as exc:
            print(f"Skipping event attribution for {sim_names[i]}: {exc}")

    attrib = pd.DataFrame(attribution_rows)
    if len(attrib) > 0:
        print("\nIntense-precipitation wave attribution")
        print(attrib.to_string(index=False))
        if save:
            attrib.to_csv(os.path.join(save_loc, "intense_precip_wave_attribution.csv"), index=False)

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.6), constrained_layout=True)
        x = np.arange(len(wave_names))
        width = 0.8 / len(dss)
        for i, sim in enumerate(sim_names):
            sub = attrib[attrib.simulation == sim].set_index("wave").reindex(wave_names)
            ax[0].bar(x + (i - (len(dss)-1)/2) * width, sub.intense_event_fraction, width, color=cols[i], label=sim)
            ax[1].bar(x + (i - (len(dss)-1)/2) * width, sub.mean_normalised_amplitude_at_events, width, color=cols[i])
        ax[0].set_ylabel(f"Fraction of P > p{percentile:g} events\nwhere wave is largest")
        ax[1].set_ylabel("Mean normalised wave amplitude\nat intense events")
        for a in ax:
            a.set_xticks(x)
            a.set_xticklabels([wave_label(w) for w in wave_names], rotation=30, ha="right")
            a.grid(alpha=0.2)
        ax[0].legend(loc="best")
        figsave.save(fig, "intense_precip_wave_attribution", number=5)
        print("Done with figure: intense_precip_wave_attribution")


# -----------------------------------------------------------------------------
# 6. Vertical profiles of wave-filtered power
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Does u_nudge amplify UTLS wave variance in u/T/omega more than T_nudge?
#   * Does the precipitation response line up with increased lower-tropospheric
#     ascent, upper-tropospheric temperature variance, or near-tropopause wave
#     activity?

if should_run_figure(6):
    profile_rows = []
    for i, ds in enumerate(dss):
        for wname in vertical_profile_waves:
            projection = lat_projection_for_wave(wname)
            for var in vertical_profile_fields:
                if var == "cldfrac":
                    levels = cldfrac_profile_heights_km
                    field_names = [cldfrac_level_field_name(zkm) for zkm in levels]
                    vertical_axis = "height_km"
                    vertical_units = "km"
                else:
                    levels = vertical_profile_levels
                    field_names = [f"{var}{plev}" for plev in levels]
                    vertical_axis = "pressure_hpa"
                    vertical_units = "hPa"
                for level, field in zip(levels, field_names):
                    try:
                        wav = get_filtered_wave(ds, i, field, wname, avgLat=True)
                        profile_rows.append(dict(simulation=sim_names[i], simulation_index=i,
                                                 wave=wname, wave_label=wave_label(wname),
                                                 latitude_projection=projection,
                                                 field=var,
                                                 vertical_coord=level,
                                                 vertical_axis=vertical_axis,
                                                 vertical_units=vertical_units,
                                                 wave_variance=wave_variance(wav)))
                    except Exception as exc:
                        print(f"Skipping vertical wave power {sim_names[i]} {wname} {field}: {exc}")
    prof = pd.DataFrame(profile_rows)
    if len(prof) > 0:
        if save:
            prof.to_csv(os.path.join(save_loc, "vertical_wave_power_profiles.csv"), index=False)

        fig, ax = plt.subplots(len(vertical_profile_fields), len(vertical_profile_waves),
                               figsize=(3.2 * len(vertical_profile_waves), 3.0 * len(vertical_profile_fields)),
                               constrained_layout=True, sharey="row")
        if len(vertical_profile_fields) == 1:
            ax = ax[None, :]
        if len(vertical_profile_waves) == 1:
            ax = ax[:, None]
        for fj, var in enumerate(vertical_profile_fields):
            for wj, wname in enumerate(vertical_profile_waves):
                for i, sim in enumerate(sim_names):
                    sub = prof[(prof.simulation == sim) & (prof.wave == wname) & (prof.field == var)].sort_values("vertical_coord")
                    if len(sub) == 0:
                        continue
                    vals = sub.wave_variance.values
                    y = sub.vertical_coord.values
                    ax[fj, wj].plot(vals, y, color=cols[i], label=sim)
                    ax[fj, wj].set_xscale("log")
                if var == "cldfrac":
                    ax[fj, wj].set_ylim(
                        min(cldfrac_profile_heights_km),
                        max(cldfrac_profile_heights_km),
                    )
                else:
                    ax[fj, wj].invert_yaxis()
                    ax[fj, wj].set_yscale("log")
                    ax[fj, wj].set_ylim(1000, 70)
                ax[fj, wj].grid(alpha=0.2)
                if fj == 0:
                    ax[fj, wj].set_title(wave_label(wname))
                if wj == 0:
                    ylabel = f"{var}\nHeight (km)" if var == "cldfrac" else f"{var}\nPressure (hPa)"
                    ax[fj, wj].set_ylabel(ylabel)
                if fj == len(vertical_profile_fields) - 1:
                    ax[fj, wj].set_xlabel("wave variance")
        ax[0, 0].legend(loc="best")
        fig.suptitle("Vertical wave-power profiles")
        figsave.save(fig, "vertical_wave_power_profiles", number=6)
        print("Done with figure: vertical_wave_power_profiles")


# -----------------------------------------------------------------------------
# 7. Intrinsic phase-speed / critical-level diagnostics
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * A line where Ubar approaches the wave phase speed is a potential critical
#     or absorption layer.  If u_nudge moves such a layer into/out of the UTLS,
#     the wave--convection coupling can change without a large mean precip shift.

if should_run_figure(7):
    speed_rows = []
    for wname in wave_names:
        cmin, ccen, cmax = wave_phase_speed_range(wname)
        speed_rows.append(dict(wave=wname, wave_label=wave_label(wname),
                               latitude_projection=lat_projection_for_wave(wname),
                               c_min_ms=cmin, c_central_ms=ccen, c_max_ms=cmax,
                               k_range=str(WAVE_BOXES[wname]["k"]), period_range_days=str(WAVE_BOXES[wname]["period"])))
    speed_df = pd.DataFrame(speed_rows)
    print("\nWave-box phase-speed estimates")
    print(speed_df.to_string(index=False))
    if save:
        speed_df.to_csv(os.path.join(save_loc, "wave_box_phase_speed_estimates.csv"), index=False)

    fig, ax = plt.subplots(1, len(dss_zm), figsize=(3.6 * len(dss_zm), 4.2), constrained_layout=True, sharey=True)
    if len(dss_zm) == 1:
        ax = [ax]
    for i, zds in enumerate(dss_zm):
        u = zds.uzonal_isobaric.sel(latitude=slice(filter_lat_bounds[0], filter_lat_bounds[1])).mean(dim=("Time", "latitude"))
        ax[i].plot(u.values, u.pres.values, color=cols[i], lw=1.8, label=sim_names[i])
        for _, row in speed_df.iterrows():
            # Only draw the central phase speed as a labelled line, with the full range faintly shaded.
            ax[i].axvline(row.c_central_ms, color="0.5", lw=0.8, ls="--")
            ax[i].text(row.c_central_ms, 80, row.wave_label, rotation=90, ha="right", va="top", fontsize=7)
            #ax[i].axvspan(min(row.c_min_ms, row.c_max_ms), max(row.c_min_ms, row.c_max_ms), color="0.8", alpha=0.08)
        ax[i].axvline(0, color="0.7", lw=0.6)
        ax[i].set_title(sim_names[i])
        ax[i].set_xlabel(r"$\overline{u}$ and $c$ (m s$^{-1}$)")
        ax[i].set_yscale("log")
        ax[i].invert_yaxis()
        ax[i].set_ylim(1000, 50)
        ax[i].grid(alpha=0.2)
        annotate_panel_label(ax[i], sim_names[i])
    ax[0].set_ylabel("Pressure (hPa)")
    fig.suptitle(f"Tropical mean wind profile over {filter_lat_bounds[0]}--{filter_lat_bounds[1]}° with wave phase-speed ranges")
    figsave.save(fig, "critical_level_phase_speed_profiles", number=7)
    print("Done with figure: critical_level_phase_speed_profiles")


# -----------------------------------------------------------------------------
# 8. Wave-filtered eddy fluxes (heavier, latitude-resolved filtering)
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Kelvin-like waves should be dominated by u/omega/T structure rather than
#     meridional momentum flux.  AD/ER/IG changes in u'v', EKE or vorticity-like
#     flow would point to rotational-wave mediation.
#   * Run this after the cheaper figures, because avgLat=False applies the k-f
#     filter separately to every retained latitude.

if should_run_figure(8):
    flux_rows = []
    for i, ds in enumerate(dss):
        for wname in eddy_flux_waves:
            for plev in eddy_flux_levels:
                try:
                    uw = get_filtered_wave(ds, i, f"u{plev}", wname, avgLat=False, lat_bounds=eddy_flux_lat_bounds)
                    vw = get_filtered_wave(ds, i, f"v{plev}", wname, avgLat=False, lat_bounds=eddy_flux_lat_bounds)
                    Tw = get_filtered_wave(ds, i, f"T{plev}", wname, avgLat=False, lat_bounds=eddy_flux_lat_bounds)
                    uw, vw, Tw = xr.align(uw, vw, Tw, join="inner")
                    uv = (uw * vw).mean(dim=("Time", "longitude"), skipna=True)
                    vT = (vw * Tw).mean(dim=("Time", "longitude"), skipna=True)
                    eke = 0.5 * (uw**2 + vw**2).mean(dim=("Time", "longitude"), skipna=True)
                    for lat, uvv, vTv, ekev in zip(uv.latitude.values, uv.values, vT.values, eke.values):
                        flux_rows.append(dict(simulation=sim_names[i], wave=wname,
                                              wave_label=wave_label(wname), pres=plev,
                                              latitude_projection="latitude_resolved",
                                              latitude=lat, uv=uvv, vT=vTv, eke=ekev))
                except Exception as exc:
                    print(f"Skipping wave-filtered eddy flux {sim_names[i]} {wname} {plev}: {exc}")
    flux = pd.DataFrame(flux_rows)
    if len(flux) > 0:
        if save:
            flux.to_csv(os.path.join(save_loc, "wave_filtered_eddy_fluxes.csv"), index=False)
        fig, ax = plt.subplots(len(eddy_flux_waves), len(eddy_flux_levels),
                               figsize=(3.2 * len(eddy_flux_levels), 2.5 * len(eddy_flux_waves)),
                               constrained_layout=True, sharex=True)
        if len(eddy_flux_waves) == 1:
            ax = ax[None, :]
        if len(eddy_flux_levels) == 1:
            ax = ax[:, None]
        for wj, wname in enumerate(eddy_flux_waves):
            for pj, plev in enumerate(eddy_flux_levels):
                for i, sim in enumerate(sim_names):
                    sub = flux[(flux.simulation == sim) & (flux.wave == wname) & (flux.pres == plev)]
                    ax[wj, pj].plot(sub.latitude, sub.uv, color=cols[i], label=sim if (wj == 0 and pj == 0) else None)
                ax[wj, pj].axhline(0, color="0.7", lw=0.6)
                ax[wj, pj].axvline(0, color="0.7", lw=0.6)
                ax[wj, pj].grid(alpha=0.2)
                if wj == 0:
                    ax[wj, pj].set_title(f"{plev} hPa")
                if pj == 0:
                    ax[wj, pj].set_ylabel(f"{wave_label(wname)}\n$u'_Wv'_W$")
                if wj == len(eddy_flux_waves) - 1:
                    ax[wj, pj].set_xlabel("Latitude")
        ax[0, 0].legend(loc="best")
        figsave.save(fig, "wave_filtered_eddy_momentum_fluxes", number=8)
        print("Done with figure: wave_filtered_eddy_momentum_fluxes")


# -----------------------------------------------------------------------------
# 9. Symmetric and antisymmetric wavenumber-frequency spectra
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Do the configured filter regions sit over coherent spectral peaks?
#   * Are nudged-run changes concentrated in symmetric/even or antisymmetric/odd
#     power, or are they broad changes to the convective background?

if should_run_figure(9):
    spec_levels = np.linspace(0, 2.5, 26)
    spec_cmap, spec_norm = make_spectrum_cmap(spec_levels)

    components = ("symmetric", "antisymmetric")
    spectra_by_sim = []
    for i, ds in enumerate(dss):
        try:
            sp_input = spectrum_input_field(ds, spectrum_field, spectrum_lat_bounds)
            raw_spectrum = spacetime_power(
                sp_input,
                lataggreg="mean",
                latitude_bounds=normalise_lat_bounds(spectrum_lat_bounds),
                dosymmetries=True,
                segsize=spectrum_segment_days,
                noverlap=spectrum_overlap_days,
                spd=obs_per_day,
            )

            spectra = {}
            for component in components:
                raw_component = raw_spectrum.sel(component=component)
                background = smooth_wavefreq(raw_component, freq_name="frequency")
                spectra[component] = xr.where(background > 0, raw_component / background, np.nan)
            spectra_by_sim.append(spectra)
        except Exception as exc:
            spectra_by_sim.append(None)
            print(f"Skipping symmetric/antisymmetric spectrum for {sim_names[i]}: {exc}")

    fig, ax = plt.subplots(len(dss), 2, figsize=(9.0, 2.8 * len(dss)),
                           constrained_layout=True, sharex=True, sharey=True)
    if len(dss) == 1:
        ax = np.array([ax])

    spec_im = None
    for i, spectra in enumerate(spectra_by_sim):
        if spectra is None:
            continue
        for cj, component in enumerate(components):
            spectrum = spectra[component].transpose("wavenumber", "frequency")
            wn = spectrum.wavenumber.values
            freq = spectrum.frequency.values
            spec_im = ax[i, cj].pcolormesh(
                wn, freq, spectrum.values.T,
                cmap=spec_cmap, norm=spec_norm, shading="auto",
            )
            add_wave_filter_region_outlines(ax[i, cj], wn, freq, component)
            ax[i, cj].axvline(0, color="0.4", lw=0.7, ls="--")
            ax[i, cj].set_xlim(*spectrum_wavenumber_limits)
            ax[i, cj].set_ylim(*spectrum_frequency_limits)
            add_period_reference_lines_frequency(ax[i, cj])
            ax[i, cj].grid(alpha=0.15)
            if i == 0:
                ax[i, cj].set_title(component.capitalize())
            if cj == 0:
                ax[i, cj].set_ylabel(f"{sim_names[i]}\nFrequency (day$^{{-1}}$)")
            else:
                ax[i, cj].tick_params(labelleft=False)
            if i == len(dss) - 1:
                ax[i, cj].set_xlabel("Zonal wavenumber")
            annotate_panel_label(ax[i, cj], sim_names[i])

    if spec_im is not None:
        fig.colorbar(spec_im, ax=ax.ravel(), location="right", shrink=0.85, label="Normalised power")
        fig.legend(
            handles=wave_filter_region_legend_handles(),
            loc="outside lower center",
            ncol=min(4, len(wave_names)),
            title="Filter regions",
        )
        fig.suptitle(
            f"{spectrum_field} symmetric/antisymmetric spectra over "
            f"{spectrum_lat_bounds[0]}--{spectrum_lat_bounds[1]}°"
        )
        figsave.save(fig, "symmetric_antisymmetric_wave_spectra", number=9)
        print("Done with figure: symmetric_antisymmetric_wave_spectra")

    period_fig, period_ax = plt.subplots(len(dss), 2, figsize=(9.0, 2.8 * len(dss)),
                                         constrained_layout=True, sharex=True, sharey=True)
    if len(dss) == 1:
        period_ax = np.array([period_ax])

    period_im = None
    period_min = 1.0 / spectrum_frequency_limits[1]
    for i, spectra in enumerate(spectra_by_sim):
        if spectra is None:
            continue
        for cj, component in enumerate(components):
            spectrum = spectra[component].transpose("wavenumber", "frequency")
            wn = spectrum.wavenumber.values
            freq = spectrum.frequency.values
            fmask = freq > 0
            freq_pos = freq[fmask]
            period = period_days_from_frequency(freq_pos)
            order = np.argsort(period)
            period_plot = period[order]
            values = spectrum.values[:, fmask][:, order]

            period_im = period_ax[i, cj].pcolormesh(
                wn, period_plot, values.T,
                cmap=spec_cmap, norm=spec_norm, shading="auto",
            )
            add_wave_filter_region_outlines_period(period_ax[i, cj], wn, freq, component)
            add_phase_speed_lines_period_axis(period_ax[i, cj], wn)
            period_ax[i, cj].axvline(0, color="0.4", lw=0.7, ls="--")
            period_ax[i, cj].set_xlim(*spectrum_wavenumber_limits)
            period_ax[i, cj].set_yscale("log")
            period_ax[i, cj].set_ylim(np.nanmax(period_plot), period_min)
            add_period_reference_lines_period(period_ax[i, cj])
            yticks = [p for p in [2, 3, 6, 10, 20, 30, 60, 90]
                      if period_min <= p <= np.nanmax(period_plot)]
            period_ax[i, cj].set_yticks(yticks)
            period_ax[i, cj].set_yticklabels([f"{p:g}" for p in yticks])
            period_ax[i, cj].grid(alpha=0.15)
            if i == 0:
                period_ax[i, cj].set_title(component.capitalize())
            if cj == 0:
                period_ax[i, cj].set_ylabel(f"{sim_names[i]}\nPeriod (days)")
            else:
                period_ax[i, cj].tick_params(labelleft=False)
            if i == len(dss) - 1:
                period_ax[i, cj].set_xlabel("Zonal wavenumber")
            annotate_panel_label(period_ax[i, cj], sim_names[i])

    if period_im is not None:
        period_fig.colorbar(period_im, ax=period_ax.ravel(), location="right",
                            shrink=0.85, label="Normalised power")
        period_fig.legend(
            handles=wave_filter_region_legend_handles(),
            loc="outside lower center",
            ncol=min(4, len(wave_names)),
            title="Filter regions",
        )
        period_fig.suptitle(
            f"{spectrum_field} symmetric/antisymmetric period spectra over "
            f"{spectrum_lat_bounds[0]}--{spectrum_lat_bounds[1]}°"
        )
        figsave.save(period_fig, "symmetric_antisymmetric_wave_spectra_period", number=9)
        print("Done with figure: symmetric_antisymmetric_wave_spectra_period")

    phase_speed_axis = phase_speed_axis_values()
    phase_fig, phase_ax = plt.subplots(len(dss), 2, figsize=(9.0, 2.8 * len(dss)),
                                       constrained_layout=True, sharex=True, sharey=True)
    if len(dss) == 1:
        phase_ax = np.array([phase_ax])

    phase_im = None
    for i, spectra in enumerate(spectra_by_sim):
        if spectra is None:
            continue
        for cj, component in enumerate(components):
            spectrum = spectra[component].transpose("wavenumber", "frequency")
            wn = spectrum.wavenumber.values
            freq = spectrum.frequency.values
            wn_phase, c_phase, phase_values = transform_wavenumber_frequency_to_phase_speed(
                wn, freq, spectrum.values, phase_speed_axis
            )
            phase_im = phase_ax[i, cj].pcolormesh(
                wn_phase, c_phase, phase_values,
                cmap=spec_cmap, norm=spec_norm, shading="auto",
            )
            add_wave_filter_region_outlines_phase_speed(phase_ax[i, cj], wn, freq, c_phase, component)
            phase_ax[i, cj].axvline(0, color="0.4", lw=0.7, ls="--")
            phase_ax[i, cj].axhline(0, color="0.4", lw=0.7, ls="--")
            phase_ax[i, cj].set_xlim(*spectrum_wavenumber_limits)
            phase_ax[i, cj].set_ylim(*spectrum_phase_speed_limits)
            add_period_reference_lines_phase_speed(phase_ax[i, cj], wn_phase)
            phase_ax[i, cj].grid(alpha=0.15)
            if i == 0:
                phase_ax[i, cj].set_title(component.capitalize())
            if cj == 0:
                phase_ax[i, cj].set_ylabel(f"{sim_names[i]}\nZonal phase speed (m s$^{{-1}}$)")
            else:
                phase_ax[i, cj].tick_params(labelleft=False)
            if i == len(dss) - 1:
                phase_ax[i, cj].set_xlabel("Zonal wavenumber")
            annotate_panel_label(phase_ax[i, cj], sim_names[i])

    if phase_im is not None:
        phase_fig.colorbar(phase_im, ax=phase_ax.ravel(), location="right",
                           shrink=0.85, label="Normalised power")
        phase_fig.legend(
            handles=wave_filter_region_legend_handles(),
            loc="outside lower center",
            ncol=min(4, len(wave_names)),
            title="Filter regions",
        )
        phase_fig.suptitle(
            f"{spectrum_field} symmetric/antisymmetric wavenumber-phase-speed spectra over "
            f"{spectrum_lat_bounds[0]}--{spectrum_lat_bounds[1]}°"
        )
        figsave.save(phase_fig, "symmetric_antisymmetric_wave_spectra_phase_speed", number=9)
        print("Done with figure: symmetric_antisymmetric_wave_spectra_phase_speed")

print("\nFinished equatorial_wave_diagnostics.py")
