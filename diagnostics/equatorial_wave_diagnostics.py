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
The current analysis wave list is chosen for this simulation set:
  Kelvin:       k=1..14,    period=2.5..20 d,  equivalent depth=8..90 m
  EIG:          k=1..14,    period=1.8..4.5 d, equivalent depth=8..90 m
  IG antisym.:  k=-15..-1,  period=1.25..2 d,  equivalent depth=8..90 m
  Easterly:     k=-28..-6,  period=2.5..7 d,   no depth constraint
  ER:           k=-10..-1,  period=10..40 d,   equivalent depth=8..90 m

Wave-phase-conditioned cloud-classification analysis (previously figures 10
and 12 here) lives in convective_pathway_diagnostics.py, which conditions the
shared high-cloud classes on Kelvin, Easterly, and ER phase.

The easterly-wave definition follows Rios-Berrios et al. (2023). It is a
rectangular TD/easterly-wave band rather than a shallow-water normal mode.

The script caches filtered fields as NetCDF files beside the source simulations
so that figures can be regenerated without re-running the k-f filters.
"""

import os, sys, numbers, json

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
import xarray as xr
import dask

from diagnostic_helpers import *
from equatorial_wave_config import (
    MISSING_EQUIVALENT_DEPTH,
    WAVE_BOXES,
    WAVE_LATITUDE_PROJECTION,
    WAVE_VARIABILITY_FIELDS,
)
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
save_loc = diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/waves/")

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1:  wave_filtered_hovmollers_precip
#   2:  wave_power_absolute_and_change (+ wave_power_fraction_of_total)
#   3:  wave_amplitude_lag_correlations
#   4:  wave_conditioned_precip_olr_pw
#   5:  intense_precip_wave_attribution
#   6:  vertical_wave_power_profiles
#   7:  critical_level_phase_speed_diagnostics
#   8:  wave_filtered_eddy_fluxes_HEAVY
#   9:  symmetric_antisymmetric_wave_spectra
#       plus period-axis and wavenumber-phase-speed variants
#  10:  wave_phase_speed_power_distributions
#   11: compact_wave_attribution_metrics
#
# Earlier figures 10 and 12 (Kelvin-conditioned cloud decomposition / cloud-top
# pressure) moved to convective_pathway_diagnostics.py figure 10, which covers
# Kelvin, Easterly, and ER phase conditioning.
figures_to_run = [2, 7]
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

# Restart/resume controls.  With the default full run, existing figure files are
# treated as completed products; set this to False if you explicitly want to
# redraw figures from cached/intermediate data.
skip_existing_figures = False
force_recompute_caches = False

# Figure 8 is deliberately excluded from the default list because it filters
# latitude-resolved u and v. Add 8 to figures_to_run after the cheaper figures
# have run and cached their filtered fields.

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
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

WAVE_ATTRIBUTION_FIGURE_NUMBER = 11
ATTRIBUTION_EPS = 1.0e-12

# Heavier latitude-resolved filtering in figure 8 can use its own wider band.
eddy_flux_lat_bounds = (-20, 20)

# Analysis wave bands.  Names with suffixes are unique analysis IDs for parity
# projections of the same underlying k-f filter.
wave_names = ["Kelvin", "EIG", "IG_antisym", "Easterly", "ER"]

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
#   Easterly is the direct 10 S--10 N mean used by Rios-Berrios et al. (2023),
#   without assigning the rectangular TD/easterly-wave box a Matsuno parity.
wave_latitude_projection = dict(WAVE_LATITUDE_PROJECTION)

# Variables used in the wave diagnostics.  Names are interpreted by
# get_filter_field(); cldfrac without a height is the vertical mean, cldfrac5km
# means the nearest cloud-fraction level to 5 km, and pressure-level fields are
# parsed from names like u850, w500, T100, v850, qv700, RH500.
power_fields = list(WAVE_VARIABILITY_FIELDS)

lag_target_fields = ["Precip", "OLR", "PW", "cldfrac"]
conditioned_fields = ["Precip", "OLR", "PW", "cldfrac"]
vertical_profile_waves = wave_names
vertical_profile_fields = ["u", "w", "T", "cldfrac"]
vertical_profile_levels = [50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 950, 1000]
cldfrac_profile_heights_km = [2, 5, 8, 11, 14, 17, 20]

# Figure 8 keeps latitude resolved and applies each wave box directly.
eddy_flux_waves = list(wave_names)
eddy_flux_levels = [70, 100, 250, 850]

spectrum_field = "Precip"
spectrum_lat_bounds = filter_lat_bounds
spectrum_segment_days = 48
spectrum_overlap_days = 30
spectrum_wavenumber_limits = (-30, 16)
spectrum_frequency_limits = (0, 0.6)
spectrum_period_reference_days = [3.0, 6.0, 30.0]
spectrum_phase_speed_lines_ms = [5.0, 10.0, 20.0, 40.0]
spectrum_phase_speed_limits = (-160.0, 160.0)
spectrum_phase_speed_step = 2.0
spectrum_components = ("symmetric", "antisymmetric")
phase_speed_distribution_limits = (-400.0, 300.0)
phase_speed_distribution_bin_width = 5.0
phase_speed_distribution_power_mode = "excess_normalised_power"
# Shared x-axis for figure 7's wind profiles and phase-speed distributions.
critical_level_phase_speed_xlim = (-40.0, 25.0)
spectrum_outline_colors = {
    "Kelvin": "tab:orange",
    "EIG": "tab:red",
    "IG_sym": "tab:pink",
    "IG_antisym": "tab:purple",
    "Easterly": "tab:green",
    "ER": "tab:blue",
}
earth_radius_m = 6.371e6

# Input/output cache for k-f filtered fields.
cache_dir = diagnostic_cache_dir("wave_cache", root_dir=root_dir)
input_cache_dir = os.path.join(cache_dir, "filter_inputs")
derived_cache_dir = os.path.join(cache_dir, "derived")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(input_cache_dir, exist_ok=True)
os.makedirs(derived_cache_dir, exist_ok=True)

# Bump this when the pressure-level source fields or their coordinate handling
# changes. Existing pressure-level filter inputs/wave caches without this marker
# are regenerated automatically.
PRESSURE_FIELD_CACHE_VERSION = "pressure-axis-fixed"
PRESSURE_FIELD_PREFIXES = ("w", "u", "v", "T", "qv", "RH")
DERIVED_CACHE_VERSION = "equatorial-wave-derived-v2"
SPECTRUM_CACHE_VERSION = "equatorial-wave-spectrum-v1"

# Avoid giant chunks when selecting latitude/longitude subsets.
dask.config.set({"array.slicing.split_large_chunks": True})

needed_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "olrtoa",
    "cldfrac",
    "uzonal_isobaric", "umeridional_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric",
    "theta", "pressure", "zgrid", "w",
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


def numbered_figure_path(number, stem):
    return os.path.join(save_loc, f"{number:02d}_{stem}.png")


def figure_outputs_exist(number, stems):
    if isinstance(stems, str):
        stems = [stems]
    return all(os.path.exists(numbered_figure_path(number, stem)) for stem in stems)


def should_build_figure(number, stems):
    if not should_run_figure(number):
        return False
    if save and skip_existing_figures and not force_recompute_caches and figure_outputs_exist(number, stems):
        if isinstance(stems, str):
            stems = [stems]
        print(
            f"Skipping figure {number}: existing output file(s) found "
            f"({', '.join(f'{number:02d}_{stem}.png' for stem in stems)})"
        )
        return False
    return True


def save_figure(number, fig, stem):
    return figsave.save(fig, stem, number=number)


def jsonable_value(value):
    if isinstance(value, dict):
        return {str(k): jsonable_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable_value(v) for v in value]
    if isinstance(value, np.ndarray):
        return jsonable_value(value.tolist())
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def derived_cache_path(cache_name, suffix):
    return os.path.join(derived_cache_dir, f"{safe_name(cache_name)}.{suffix}")


def read_cache_metadata(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_cache_metadata(path, signature):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as handle:
        json.dump(jsonable_value(signature), handle, indent=2, sort_keys=True)
    os.replace(tmp, path)


def cache_signature(kind, extra=None):
    signature = dict(
        cache_version=DERIVED_CACHE_VERSION,
        kind=kind,
        root_dir=os.path.abspath(root_dir),
        data_locs=list(data_locs),
        sim_names=list(sim_names),
        latlon_file=list(latlon_file),
        zm_latlon_file=list(zm_latlon_file),
        ndays=ndays,
        dt_hours=dt,
        obs_per_day=obs_per_day,
        analysis_end_date=analysis_end_date,
        time_format=time_format,
        filter_lat_bounds=jsonable_value(filter_lat_bounds),
        filter_avg_lat=filter_avg_lat,
        wave_names=list(wave_names),
        wave_boxes=jsonable_value(WAVE_BOXES) if "WAVE_BOXES" in globals() else None,
        wave_latitude_projection=jsonable_value(wave_latitude_projection),
        pressure_field_cache_version=PRESSURE_FIELD_CACHE_VERSION,
    )
    if extra:
        signature.update(jsonable_value(extra))
    return signature


def cache_metadata_matches(metadata_path, signature, label):
    if force_recompute_caches:
        print(f"Ignoring cached {label}: force_recompute_caches is enabled.")
        return False
    try:
        cached = read_cache_metadata(metadata_path)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"Ignoring cached {label}: could not read metadata ({exc}).")
        return False
    if cached is None:
        return False
    signature = jsonable_value(signature)
    changed = sorted(
        key for key in set(cached) | set(signature)
        if cached.get(key) != signature.get(key)
    )
    if changed:
        shown = ", ".join(changed[:6])
        suffix = "" if len(changed) <= 6 else f", +{len(changed) - 6} more"
        print(f"Ignoring cached {label}: metadata changed ({shown}{suffix}).")
        return False
    return True


def figure_metadata_path(number, stem):
    return derived_cache_path(f"figure_{number:02d}_{stem}", "metadata.json")


def figure_outputs_current(number, stems, signature):
    if isinstance(stems, str):
        stems = [stems]
    if not figure_outputs_exist(number, stems):
        return False
    return all(
        cache_metadata_matches(
            figure_metadata_path(number, stem),
            signature,
            f"figure {number:02d}_{stem}",
        )
        for stem in stems
    )


def should_build_versioned_figure(number, stems, signature):
    if not should_run_figure(number):
        return False
    if (
        save
        and skip_existing_figures
        and not force_recompute_caches
        and figure_outputs_current(number, stems, signature)
    ):
        if isinstance(stems, str):
            stems = [stems]
        print(
            f"Skipping figure {number}: existing output file(s) and metadata match "
            f"({', '.join(f'{number:02d}_{stem}.png' for stem in stems)})"
        )
        return False
    return True


def write_figure_output_metadata(number, stems, signature):
    if not save:
        return
    if isinstance(stems, str):
        stems = [stems]
    for stem in stems:
        write_cache_metadata(figure_metadata_path(number, stem), signature)


def load_cached_table(cache_name, signature, output_filename=None, required_columns=None):
    """Load a derived CSV cache, seeding it from save_loc output if available."""
    csv_path = derived_cache_path(cache_name, "csv")
    metadata_path = derived_cache_path(cache_name, "metadata.json")
    required_columns = set(required_columns or ())

    if cache_metadata_matches(metadata_path, signature, cache_name) and os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            if required_columns.issubset(df.columns):
                print(f"Using cached table {csv_path}")
                return df
        except Exception as exc:
            print(f"Ignoring cached table {csv_path}: {exc}")

    if output_filename is not None and not force_recompute_caches:
        output_path = os.path.join(save_loc, output_filename)
        if os.path.exists(output_path):
            try:
                df = pd.read_csv(output_path)
                if required_columns.issubset(df.columns):
                    print(f"Using existing output table as cache: {output_path}")
                    try:
                        write_cached_table(cache_name, df, signature)
                    except Exception as exc:
                        print(f"Could not seed derived cache for {cache_name}: {exc}")
                    return df
            except Exception as exc:
                print(f"Ignoring existing output table {output_path}: {exc}")
    return None


def write_cached_table(cache_name, df, signature):
    csv_path = derived_cache_path(cache_name, "csv")
    metadata_path = derived_cache_path(cache_name, "metadata.json")
    tmp = csv_path + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, csv_path)
    write_cache_metadata(metadata_path, signature)


def table_key_set(df, key_columns):
    if df is None or len(df) == 0 or any(col not in df.columns for col in key_columns):
        return set()
    return set(map(tuple, df[list(key_columns)].itertuples(index=False, name=None)))


def append_cached_rows(cache_name, existing, rows, signature):
    if not rows:
        return existing if existing is not None else pd.DataFrame()
    new_df = pd.DataFrame(rows)
    if existing is not None and len(existing) > 0:
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df
    write_cached_table(cache_name, combined, signature)
    return combined


def load_cached_dataset(cache_name, signature):
    nc_path = derived_cache_path(cache_name, "nc")
    metadata_path = derived_cache_path(cache_name, "metadata.json")
    if not (os.path.exists(nc_path) and cache_metadata_matches(metadata_path, signature, cache_name)):
        return None
    try:
        print(f"Using cached dataset {nc_path}")
        return xr.open_dataset(nc_path, chunks={})
    except Exception as exc:
        print(f"Ignoring cached dataset {nc_path}: {exc}")
        return None


def write_cached_dataset(cache_name, ds, signature):
    nc_path = derived_cache_path(cache_name, "nc")
    metadata_path = derived_cache_path(cache_name, "metadata.json")
    tmp = nc_path + ".tmp"
    netcdf_safe_dataset(ds).to_netcdf(tmp)
    os.replace(tmp, nc_path)
    write_cache_metadata(metadata_path, signature)


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
    )
    zds = standardize_latlon_dataset(zds, needed_vars)

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)


# -----------------------------------------------------------------------------
# Wave-box metadata and helper functions
# -----------------------------------------------------------------------------

mis = MISSING_EQUIVALENT_DEPTH


def wave_label(wave_name):
    return WAVE_BOXES.get(wave_name, {}).get("label", wave_name)


def safe_name(x):
    return str(x).replace(" ", "_").replace("/", "-").replace(".", "p").replace("-", "m")


def precip_rate(ds):
    pr = precip_rate_from_accum(total_precip_accum(ds), dt=dt)
    pr.name = "Precip"
    pr.attrs["units"] = "mm day-1"
    return finite_or_nan(pr)


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
        return out

    if "z" not in cf.coords:
        raise ValueError(f"Cannot select {field_name}: cldfrac has no height coordinate")
    out = cf.sel(z=height_km * 1000.0, method="nearest")
    out.name = field_name
    out.attrs["selected_height_km"] = height_km
    return out


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

    # Pressure-level fields encoded as u850, v250, w500, T100, qv700, RH500.
    prefixes = [
        ("w", None),
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
            if var is None:
                out = vertical_velocity_at_pressure(ds, plev).sel(Time=pr.Time)
                out.name = field_name
                return out
            if var not in ds.data_vars:
                raise ValueError(f"{var} is not available in this dataset")
            out = finite_or_nan(ds[var]).sel(pres=plev, method="nearest").sel(Time=pr.Time)
            out.name = field_name
            return out

    raise ValueError(f"Unrecognised field name: {field_name}")


def is_pressure_level_field(field_name):
    for prefix in PRESSURE_FIELD_PREFIXES:
        if field_name.startswith(prefix):
            level_text = field_name[len(prefix):]
            if not level_text:
                return False
            try:
                float(level_text)
            except ValueError:
                return False
            return True
    return False


def field_units(field_name):
    if field_name == "Precip":
        return "mm day$^{-1}$"
    if field_name == "OLR":
        return "W m$^{-2}$"
    if field_name == "PW":
        return "kg m$^{-2}$"
    if field_name == "cldfrac" or field_name.startswith("cldfrac"):
        return "fraction"
    if is_pressure_level_field(field_name) and field_name.startswith("w"):
        return "m s$^{-1}$"
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


def filter_input_cache_matches(fname, field_name):
    if not os.path.exists(fname):
        return False
    if not is_pressure_level_field(field_name):
        return True
    try:
        with xr.open_dataset(fname) as cached:
            return cached.attrs.get("cache_version") == PRESSURE_FIELD_CACHE_VERSION
    except Exception:
        return False


def write_filter_input_if_needed(ds, sim_index, field_name):
    """Cache a standard NetCDF input for kf_filter."""
    fname = os.path.join(input_cache_dir, f"{safe_name(sim_names[sim_index])}_{safe_name(field_name)}.nc")
    if filter_input_cache_matches(fname, field_name):
        return fname
    if os.path.exists(fname):
        print(f"Regenerating stale filter input: {fname}")

    print(f"Writing filter input: {fname}")
    da = get_filter_field(ds, field_name)
    fds = to_filter_input_dataset(da, obs_var="x")
    fds.attrs.update(source_field=field_name, source_simulation=sim_names[sim_index])
    if is_pressure_level_field(field_name):
        fds.attrs["cache_version"] = PRESSURE_FIELD_CACHE_VERSION
    encoding = {"x": {"dtype": "float32"}}
    tmp_fname = fname + ".tmp"
    fds.to_netcdf(tmp_fname, encoding=encoding)
    os.replace(tmp_fname, fname)
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
    if (
        is_pressure_level_field(field_name)
        and wave_da.attrs.get("cache_version") != PRESSURE_FIELD_CACHE_VERSION
    ):
        return False
    return (
        wave_da.attrs.get("field") == field_name
        and wave_da.attrs.get("wave_name") == wave_name
        and wave_da.attrs.get("filter_name") == filter_name
        and int(avg_attr) == int(avgLat)
        and wave_da.attrs.get("filter_lat_bounds") == lat_bounds_attr(lat_bounds)
        and wave_da.attrs.get("latitude_projection") == projection
        and wave_da.attrs.get("wave_box")
        == json.dumps(jsonable_value(WAVE_BOXES[wave_name]), sort_keys=True)
    )


def read_cached_wave(cache_path, field_name, wave_name, avgLat, lat_bounds, projection, force=False):
    if force or not os.path.exists(cache_path):
        return None
    try:
        cached = xr.open_dataset(cache_path)
        wave_cached = cached["wave"]
        if wave_cache_matches(wave_cached, field_name, wave_name, avgLat, lat_bounds, projection):
            print(f"Using cached filtered wave {cache_path}")
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
        wave_box=json.dumps(jsonable_value(WAVE_BOXES[wave_name]), sort_keys=True),
    )
    if is_pressure_level_field(field_name):
        wave.attrs["cache_version"] = PRESSURE_FIELD_CACHE_VERSION
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
        wave_box=json.dumps(jsonable_value(WAVE_BOXES[wave_name]), sort_keys=True),
    )
    if is_pressure_level_field(field_name):
        wave.attrs["cache_version"] = PRESSURE_FIELD_CACHE_VERSION
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


def compute_intense_precip_wave_attribution(percentile=95.0):
    signature = cache_signature(
        "intense_precip_wave_attribution",
        extra=dict(
            percentile=percentile,
            waves=list(wave_names),
            filter_lat_bounds=jsonable_value(filter_lat_bounds),
        ),
    )
    cached = load_cached_table(
        "intense_precip_wave_attribution",
        signature,
        output_filename="intense_precip_wave_attribution.csv",
        required_columns=("simulation", "wave", "intense_event_fraction", "mean_normalised_amplitude_at_events"),
    )
    if cached is not None:
        return cached

    attribution_rows = []
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
            stack = xr.concat([norm_waves[w] for w in wave_names], dim=xr.DataArray(wave_names, dims="wave", name="wave"))
            winner_index = stack.argmax(dim="wave")
            winner = xr.DataArray(
                np.array(wave_names, dtype=object)[winner_index.values],
                dims=winner_index.dims,
                coords=winner_index.coords,
            )
            event_count = float(events.sum().values)
            for wname in wave_names:
                projection = lat_projection_for_wave(wname)
                frac = float(((winner == wname).where(events)).sum().values / event_count) if event_count > 0 else np.nan
                mean_amp = float(norm_waves[wname].where(events).mean().values)
                attribution_rows.append(
                    dict(
                        simulation=sim_names[i],
                        simulation_index=i,
                        wave=wname,
                        wave_label=wave_label(wname),
                        latitude_projection=projection,
                        intense_event_fraction=frac,
                        mean_normalised_amplitude_at_events=mean_amp,
                        precip_percentile=percentile,
                        precip_threshold=threshold,
                    )
                )
        except Exception as exc:
            print(f"Skipping event attribution for {sim_names[i]}: {exc}")
    out = pd.DataFrame(attribution_rows)
    if len(out) > 0:
        write_cached_table("intense_precip_wave_attribution", out, signature)
    return out


def attribution_row(metric_name, variable, wave_band, condition, values_by_sim, units, notes):
    control = float(values_by_sim.get(sim_names[0], np.nan))
    nudged_u = float(values_by_sim.get(sim_names[1], np.nan))
    nudged_T = float(values_by_sim.get(sim_names[2], np.nan))
    delta_u = nudged_u - control
    delta_T = nudged_T - control
    wind_unique = delta_u - delta_T
    thermal_like = delta_T
    denom = np.abs(wind_unique) + np.abs(thermal_like) + ATTRIBUTION_EPS
    return dict(
        metric_name=metric_name,
        variable=variable,
        wave_band=wave_band,
        condition=condition,
        control=control,
        nudged_u=nudged_u,
        nudged_T=nudged_T,
        delta_u=delta_u,
        delta_T=delta_T,
        wind_unique=wind_unique,
        thermal_like=thermal_like,
        wind_fraction=wind_unique / denom,
        thermal_fraction=thermal_like / denom,
        units=units,
        notes=notes,
    )


def build_compact_wave_attribution_table(wave_metrics_df, intense_attribution_df=None):
    rows = []

    if wave_metrics_df is not None and len(wave_metrics_df) > 0:
        wave_metrics_df = add_common_power_fractions(wave_metrics_df)
        for (field, wave), sub in wave_metrics_df.groupby(["field", "wave"]):
            values = sub.set_index("simulation")["wave_fraction_of_total"].to_dict()
            rows.append(
                attribution_row(
                    "wave_variance_fraction",
                    field,
                    wave_label(wave),
                    "all",
                    values,
                    "fraction",
                    "Filtered wave power (variance of the filtered anomaly) divided by the field's "
                    "total latitude-mean power (symmetric + antisymmetric)",
                )
            )

    if intense_attribution_df is not None and len(intense_attribution_df) > 0:
        for wave, sub in intense_attribution_df.groupby("wave"):
            frac_values = sub.set_index("simulation")["intense_event_fraction"].to_dict()
            rows.append(
                attribution_row(
                    "p95_event_largest_wave_fraction",
                    "Precip",
                    wave_label(wave),
                    "p95_precip_events",
                    frac_values,
                    "fraction",
                    "Fraction of p95 precipitation events where this normalised wave amplitude is largest",
                )
            )
            amp_values = sub.set_index("simulation")["mean_normalised_amplitude_at_events"].to_dict()
            rows.append(
                attribution_row(
                    "p95_event_mean_normalised_wave_amplitude",
                    "Precip",
                    wave_label(wave),
                    "p95_precip_events",
                    amp_values,
                    "sigma",
                    "Mean absolute wave amplitude normalised by each run's wave standard deviation at p95 precipitation events",
                )
            )

    table = pd.DataFrame(rows)
    if len(table) > 0:
        table = table[
            [
                "metric_name",
                "variable",
                "wave_band",
                "condition",
                "control",
                "nudged_u",
                "nudged_T",
                "delta_u",
                "delta_T",
                "wind_unique",
                "thermal_like",
                "wind_fraction",
                "thermal_fraction",
                "units",
                "notes",
            ]
        ]
    return table


def plot_attribution_change_heatmaps(table):
    """Two annotated panels of the wave-power-fraction change from control.

    Panel 1 is the u-nudged change, panel 2 the T-nudged change, on a shared
    diverging scale. The part of panel 1 missing from panel 2 is the
    wind/shear-mediated response; no derived quantity is needed to read it.
    """
    sub = table[(table.metric_name == "wave_variance_fraction") & (table.condition == "all")]
    if len(sub) == 0:
        return
    panels = [
        ("delta_u", f"{sim_names[1]} $-$ {sim_names[0]}"),
        ("delta_T", f"{sim_names[2]} $-$ {sim_names[0]}"),
    ]
    pivots = {}
    wave_order = None
    field_order = None
    for value_col, _ in panels:
        pivot = sub.pivot_table(index="wave_band", columns="variable", values=value_col, aggfunc="mean")
        if wave_order is None:
            wave_order = [wave_label(w) for w in wave_names if wave_label(w) in pivot.index]
            field_order = [f for f in power_fields if f in pivot.columns]
        # Percentage points of the total anomaly variance.
        pivots[value_col] = 100.0 * pivot.reindex(index=wave_order, columns=field_order).values.astype(float)
    if len(wave_order) == 0 or len(field_order) == 0:
        return
    max_abs = np.nanmax([np.nanmax(np.abs(v)) for v in pivots.values()])
    if not np.isfinite(max_abs) or max_abs == 0:
        max_abs = 1.0
    levels = symmetric_levels(max_abs, ndiv=11)
    cmap = difference_cmap(levels)
    norm = colors.BoundaryNorm(levels, cmap.N)
    fig, axes = plt.subplots(
        1, len(panels),
        figsize=(1.1 * len(field_order) * len(panels) + 2.5, 0.55 * len(wave_order) + 2.5),
        constrained_layout=True, sharey=True,
    )
    for ax, (value_col, title) in zip(np.atleast_1d(axes), panels):
        values = pivots[value_col]
        im = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")
        for r in range(values.shape[0]):
            for c in range(values.shape[1]):
                v = values[r, c]
                if np.isfinite(v):
                    ax.text(c, r, f"{v:+.2g}", ha="center", va="center", fontsize=7,
                            color="k" if abs(v) < 0.7 * max_abs else "w")
        ax.set_xticks(np.arange(len(field_order)))
        ax.set_xticklabels(field_order, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(wave_order)))
        ax.set_yticklabels(wave_order)
        ax.set_title(title)
        ax.grid(False)
    add_symmetric_colorbar(
        fig, im, np.atleast_1d(axes).ravel(), levels,
        label="Wave-band power fraction change (percentage points)",
    )
    fig.suptitle("Wave power change from control (u-nudged response absent from the T-nudged panel is wind-mediated)")
    save_figure(WAVE_ATTRIBUTION_FIGURE_NUMBER, fig, "wave_attribution_change_heatmaps")


def find_attribution_selection(table, metric_name, variable, wave_band, condition):
    sub = table[
        (table.metric_name == metric_name)
        & (table.variable == variable)
        & (table.wave_band == wave_band)
        & (table.condition == condition)
    ]
    if len(sub) == 0:
        return None
    return sub.iloc[0]


def plot_attribution_summary_bars(table):
    selections = [
        ("Kelvin precip power", "wave_variance_fraction", "Precip", "Kelvin", "all"),
        ("Kelvin OLR power", "wave_variance_fraction", "OLR", "Kelvin", "all"),
        ("Kelvin PW power", "wave_variance_fraction", "PW", "Kelvin", "all"),
        ("Kelvin cldfrac power", "wave_variance_fraction", "cldfrac", "Kelvin", "all"),
        ("Easterly cldfrac power", "wave_variance_fraction", "cldfrac", "Easterly", "all"),
        ("ER cldfrac power", "wave_variance_fraction", "cldfrac", "ER", "all"),
        ("Kelvin T100 power", "wave_variance_fraction", "T100", "Kelvin", "all"),
        ("Kelvin w500 power", "wave_variance_fraction", "w500", "Kelvin", "all"),
        ("Kelvin u850 power", "wave_variance_fraction", "u850", "Kelvin", "all"),
    ]
    rows = []
    labels = []
    for label, metric_name, variable, wave_band, condition in selections:
        row = find_attribution_selection(table, metric_name, variable, wave_band, condition)
        if row is not None:
            rows.append(row)
            labels.append(label)
    if len(rows) == 0:
        return
    x = np.arange(len(rows))
    width = 0.25
    delta_T = np.asarray([row.delta_T for row in rows], dtype=float)
    wind_unique = np.asarray([row.wind_unique for row in rows], dtype=float)
    delta_u = np.asarray([row.delta_u for row in rows], dtype=float)
    fig, ax = plt.subplots(figsize=(max(10.0, 0.8 * len(rows)), 4.5), constrained_layout=True)
    ax.bar(x - width, delta_T, width, color=cols[2],
           label=f"{sim_names[2]} $-$ control (reproduced by T-nudging)")
    ax.bar(x, wind_unique, width, color="0.25",
           label="$\\Delta$u $-$ $\\Delta$T (extra u-nudged response)")
    ax.bar(x + width, delta_u, width, color=cols[1],
           label=f"{sim_names[1]} $-$ control (total u-nudged)")
    ax.axhline(0, color="0.65", lw=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Metric anomaly")
    ax.grid(axis="y", alpha=0.2)
    ax.legend(loc="best")
    ax.set_title("Wave-power attribution: T-reproduced vs extra u-nudged response")
    save_figure(WAVE_ATTRIBUTION_FIGURE_NUMBER, fig, "wave_attribution_summary_bars")


def plot_compact_wave_attribution(table):
    if len(table) == 0:
        return
    plot_attribution_change_heatmaps(table)
    plot_attribution_summary_bars(table)


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
        angular_frequency = k * c
    elif filter_name == "ER":
        angular_frequency = -beta * k / (k**2 + 3.0 * beta / c)
    elif filter_name == "IG1":
        angular_frequency = np.sqrt(3.0 * beta * c + k**2 * c**2)
    elif filter_name == "IG2":
        angular_frequency = np.sqrt(5.0 * beta * c + k**2 * c**2)
    elif filter_name in ("MRG", "IG0"):
        with np.errstate(divide="ignore", invalid="ignore"):
            angular_frequency = np.where(
                k > 0,
                k * c * (0.5 + 0.5 * np.sqrt(1.0 + 4.0 * beta / (k**2 * c))),
                k * c * (0.5 - 0.5 * np.sqrt(1.0 + 4.0 * beta / (k**2 * c))),
            )
        angular_frequency = np.where(np.isclose(k, 0.0), np.sqrt(beta * c), angular_frequency)
    else:
        raise ValueError(f"No dispersion-outline helper for {wave_name}")

    return np.abs(angular_frequency) * 86400.0 / (2.0 * np.pi)


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


def spectrum_cache_signature(sim_index):
    return cache_signature(
        "normalised_wave_spectrum",
        extra=dict(
            cache_version=SPECTRUM_CACHE_VERSION,
            simulation=sim_names[sim_index],
            spectrum_field=spectrum_field,
            spectrum_lat_bounds=jsonable_value(spectrum_lat_bounds),
            spectrum_segment_days=spectrum_segment_days,
            spectrum_overlap_days=spectrum_overlap_days,
            obs_per_day=obs_per_day,
            components=list(spectrum_components),
        ),
    )


def load_cached_spectra(sim_index):
    cache_name = f"normalised_wave_spectrum_{safe_name(sim_names[sim_index])}"
    cached = load_cached_dataset(cache_name, spectrum_cache_signature(sim_index))
    if cached is None or "normalised_power" not in cached:
        return None
    return {
        str(component): cached["normalised_power"].sel(component=component)
        for component in cached.component.values
    }


def write_cached_spectra(sim_index, spectra):
    components = list(spectra)
    spectrum_stack = xr.concat(
        [spectra[component] for component in components],
        dim=xr.DataArray(components, dims="component", name="component"),
    )
    ds_out = spectrum_stack.to_dataset(name="normalised_power")
    ds_out.attrs.update(
        description="Cached normalised symmetric/antisymmetric wave spectrum",
        simulation=sim_names[sim_index],
        field=spectrum_field,
    )
    cache_name = f"normalised_wave_spectrum_{safe_name(sim_names[sim_index])}"
    write_cached_dataset(cache_name, ds_out, spectrum_cache_signature(sim_index))


def compute_normalised_spectra_for_sim(ds, sim_index):
    spectra = load_cached_spectra(sim_index)
    if spectra is not None:
        return spectra

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
    for component in spectrum_components:
        raw_component = raw_spectrum.sel(component=component)
        background = smooth_wavefreq(raw_component, freq_name="frequency")
        spectra[component] = xr.where(background > 0, raw_component / background, np.nan).load()
    write_cached_spectra(sim_index, spectra)
    return spectra


def load_or_compute_normalised_spectra_by_sim():
    spectra_by_sim = []
    for i, ds in enumerate(dss):
        try:
            spectra_by_sim.append(compute_normalised_spectra_for_sim(ds, i))
        except Exception as exc:
            spectra_by_sim.append(None)
            print(f"Skipping symmetric/antisymmetric spectrum for {sim_names[i]}: {exc}")
    return spectra_by_sim


def phase_speed_distribution_centres():
    lo, hi = phase_speed_distribution_limits
    step = phase_speed_distribution_bin_width
    return np.arange(lo, hi + 0.5 * step, step)


def phase_speed_distribution_edges():
    centres = phase_speed_distribution_centres()
    step = phase_speed_distribution_bin_width
    return np.concatenate(([centres[0] - 0.5 * step], centres + 0.5 * step))


def phase_speed_distribution_weights(values):
    values = np.asarray(values, dtype=float)
    if phase_speed_distribution_power_mode == "excess_normalised_power":
        return np.where(np.isfinite(values) & (values > 1.0), values - 1.0, 0.0)
    if phase_speed_distribution_power_mode == "normalised_power":
        return np.where(np.isfinite(values) & (values > 0.0), values, 0.0)
    raise ValueError(f"Unknown phase-speed distribution mode: {phase_speed_distribution_power_mode}")


def phase_speed_distribution_signature():
    return cache_signature(
        "wave_phase_speed_power_distributions",
        extra=dict(
            spectrum_field=spectrum_field,
            spectrum_lat_bounds=jsonable_value(spectrum_lat_bounds),
            spectrum_segment_days=spectrum_segment_days,
            spectrum_overlap_days=spectrum_overlap_days,
            spectrum_components=list(spectrum_components),
            phase_speed_distribution_limits=jsonable_value(phase_speed_distribution_limits),
            phase_speed_distribution_bin_width=phase_speed_distribution_bin_width,
            phase_speed_distribution_power_mode=phase_speed_distribution_power_mode,
        ),
    )


def compute_wave_phase_speed_distributions():
    """
    Return phase-speed distributions made by binning the plotted spectrum.

    This is not a sampled phase-speed PDF. It is a phase-speed projection of the
    normalised wavenumber-frequency spectrum: spectral power inside each wave
    filter is binned by c = omega/k and normalised to unit area for each
    simulation-wave pair.
    """
    signature = phase_speed_distribution_signature()
    dist = load_cached_table(
        "wave_phase_speed_power_distributions",
        signature,
        output_filename="wave_phase_speed_power_distributions.csv",
        required_columns=(
            "simulation", "simulation_index", "wave", "phase_speed_ms",
            "density_per_ms", "fractional_power", "weighted_power",
        ),
    )
    if dist is not None:
        return dist

    spectra_by_sim = load_or_compute_normalised_spectra_by_sim()
    centres = phase_speed_distribution_centres()
    edges = phase_speed_distribution_edges()
    bin_width = phase_speed_distribution_bin_width
    rows = []

    for i, spectra in enumerate(spectra_by_sim):
        if spectra is None:
            continue
        for wname in wave_names:
            hist = np.zeros(centres.size, dtype=float)
            component_names = [
                component for component in WAVE_BOXES[wname].get("components", spectrum_components)
                if component in spectra
            ]
            for component in component_names:
                spectrum = spectra[component].transpose("wavenumber", "frequency")
                wn = np.asarray(spectrum.wavenumber.values, dtype=float)
                freq = np.asarray(spectrum.frequency.values, dtype=float)
                values = np.asarray(spectrum.values, dtype=float)
                kgrid, fgrid = np.meshgrid(wn, freq, indexing="ij")
                mask = wave_filter_mask_on_spectrum(wn, freq, wname)
                mask &= (kgrid != 0.0) & (fgrid > 0.0)
                weights = phase_speed_distribution_weights(values)
                phase_speed = zonal_phase_speed_from_frequency(kgrid, fgrid)
                valid = mask & np.isfinite(phase_speed) & np.isfinite(weights) & (weights > 0.0)
                if not np.any(valid):
                    continue
                hist += np.histogram(
                    phase_speed[valid],
                    bins=edges,
                    weights=weights[valid],
                )[0]

            total_weight = float(np.nansum(hist))
            if total_weight > 0.0:
                fractional_power = hist / total_weight
                density = fractional_power / bin_width
            else:
                fractional_power = np.full_like(hist, np.nan, dtype=float)
                density = np.full_like(hist, np.nan, dtype=float)

            cmin, ccen, cmax = wave_phase_speed_range(wname)
            for c, power, frac, dens in zip(centres, hist, fractional_power, density):
                rows.append(
                    dict(
                        simulation=sim_names[i],
                        simulation_index=i,
                        wave=wname,
                        wave_label=wave_label(wname),
                        phase_speed_ms=c,
                        weighted_power=power,
                        fractional_power=frac,
                        density_per_ms=dens,
                        total_weighted_power=total_weight,
                        power_mode=phase_speed_distribution_power_mode,
                        c_min_box_ms=cmin,
                        c_central_box_ms=ccen,
                        c_max_box_ms=cmax,
                    )
                )

    dist = pd.DataFrame(rows)
    write_cached_table("wave_phase_speed_power_distributions", dist, signature)
    if save:
        dist.to_csv(os.path.join(save_loc, "wave_phase_speed_power_distributions.csv"), index=False)
    return dist


def add_common_power_fractions(df):
    """Add each band's share of a single per-field total power.

    The full-band latitude mean and the antisymmetric hemisphere difference
    are the even/odd decomposition of the latitude-mean signal, so their
    variances sum to one projection-independent total per field and
    simulation (the Wheeler-Kiladis symmetric + antisymmetric convention).
    Unlike wave_fraction, these shares are comparable across wave bands and
    approximately summable where the k-f boxes do not overlap.
    """
    out = df.copy()
    out["wave_fraction_of_total"] = np.nan
    if "total_variance" not in out.columns or "latitude_projection" not in out.columns:
        return out
    for (sim, field), sub in out.groupby(["simulation", "field"]):
        t_sym = sub.loc[
            sub.latitude_projection.isin(("symmetric", "direct")), "total_variance"
        ].median()
        t_anti = sub.loc[
            sub.latitude_projection == "antisymmetric", "total_variance"
        ].median()
        if np.isfinite(t_sym) and np.isfinite(t_anti) and (t_sym + t_anti) > 0:
            out.loc[sub.index, "wave_fraction_of_total"] = (
                sub.wave_variance / (t_sym + t_anti)
            )
    return out


def add_wave_metric_ratios(wave_metrics):
    if wave_metrics is None or len(wave_metrics) == 0:
        return wave_metrics
    base = wave_metrics.drop(
        columns=["power_ratio_to_control", "fraction_ratio_to_control"],
        errors="ignore",
    ).copy()
    control = base[base.simulation_index == 0].set_index(["field", "wave"])
    ratios = []
    for _, row in base.iterrows():
        key = (row.field, row.wave)
        if key in control.index:
            cvar = control.loc[key, "wave_variance"]
            cfrac = control.loc[key, "wave_fraction"]
            ratios.append(
                dict(
                    power_ratio_to_control=row.wave_variance / cvar if cvar > 0 else np.nan,
                    fraction_ratio_to_control=row.wave_fraction / cfrac if cfrac > 0 else np.nan,
                )
            )
        else:
            ratios.append(dict(power_ratio_to_control=np.nan, fraction_ratio_to_control=np.nan))
    return pd.concat([base.reset_index(drop=True), pd.DataFrame(ratios)], axis=1)


# -----------------------------------------------------------------------------
# 1. Wave-filtered Hovmoller diagrams
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Which filter bands actually reconstruct coherent propagating precipitation?
#   * Does u_nudge increase a particular band, or simply broaden all wave-like
#     variability?
#   * Kelvin/EIG: eastward streaks. IG/Easterly/ER: mostly westward structure.

if should_build_figure(1, "wave_filtered_hovmollers_precip"):
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
            im = ax[i, j].contourf(wav.longitude, wav.Time, wav.values, levels=levels, cmap=difference_cmap(levels), extend="both")
            if i == 0:
                ax[i, j].set_title(wave_label(wname))
            if j == 0:
                ax[i, j].set_ylabel(sim_names[i])
            else:
                ax[i, j].tick_params(labelleft=False)
            if i == len(dss) - 1:
                ax[i, j].set_xlabel("Longitude")
            annotate_panel_label(ax[i, j], sim_names[i] if j == 0 else wave_label(wname))
    add_symmetric_colorbar(fig, im, ax.ravel(), levels, label=f"{hov_field}' ({field_units(hov_field)})")
    figsave.save(fig, "wave_filtered_hovmollers_precip", number=1)
    print("Done with figure: wave_filtered_hovmollers_precip")


# -----------------------------------------------------------------------------
# 2. Absolute wave-band power and change relative to control
# -----------------------------------------------------------------------------
# Here "power" is the variance of the k-f filtered anomaly, which by Parseval
# equals the spectral power integrated over the wave's filter box.
# WHAT TO LOOK FOR:
#   * The top row of the main figure shows which wave bands actually carry
#     strong absolute power in each field; the bottom row shows the nudged-run
#     change in percent. Read the rows together: a large percentage change
#     matters most in a band whose absolute power is non-negligible.
#   * The companion fraction-of-total figure gives context on how much of each
#     field's variability the band explains at all.
#   * Compare precipitation with dynamical fields. If u/w/T wave power
#     increases but precipitation power does not, waves are modified aloft but
#     weakly coupled to rainfall. If precipitation power increases without a
#     dynamical increase, convective sensitivity has changed.

wave_metrics = None
if should_run_any([2, WAVE_ATTRIBUTION_FIGURE_NUMBER]):
    wave_metrics_signature = cache_signature(
        "wave_band_variance_metrics",
        extra=dict(
            fields=list(power_fields),
            waves=list(wave_names),
            filter_lat_bounds=jsonable_value(filter_lat_bounds),
        ),
    )
    wave_metrics = load_cached_table(
        "wave_band_variance_metrics",
        wave_metrics_signature,
        output_filename="wave_band_variance_metrics.csv",
        required_columns=("simulation", "simulation_index", "field", "wave", "wave_variance", "wave_fraction"),
    )
    if wave_metrics is None:
        wave_metrics = pd.DataFrame()
    else:
        wave_metrics = wave_metrics.drop(
            columns=["power_ratio_to_control", "fraction_ratio_to_control"],
            errors="ignore",
        )

    completed_metric_keys = table_key_set(wave_metrics, ("simulation_index", "field", "wave"))
    for i, ds in enumerate(dss):
        for field in power_fields:
            for wname in wave_names:
                key = (i, field, wname)
                if key in completed_metric_keys:
                    continue
                projection = lat_projection_for_wave(wname)
                try:
                    total_var = anomaly_variance_latmean(
                        ds, field, lat_bounds=filter_lat_bounds, projection=projection
                    )
                    wav = get_filtered_wave(ds, i, field, wname, avgLat=True)
                    wvar = wave_variance(wav)
                    row = dict(
                        simulation=sim_names[i],
                        simulation_index=i,
                        field=field,
                        wave=wname,
                        wave_label=wave_label(wname),
                        latitude_projection=projection,
                        total_variance=total_var,
                        wave_variance=wvar,
                        wave_fraction=wvar / total_var if total_var > 0 else np.nan,
                    )
                    wave_metrics = append_cached_rows(
                        "wave_band_variance_metrics",
                        wave_metrics,
                        [row],
                        wave_metrics_signature,
                    )
                    completed_metric_keys.add(key)
                except Exception as exc:
                    print(f"Skipping wave variance for {sim_names[i]} {field} {wname}: {exc}")
    if len(wave_metrics) > 0:
        # Add nudged/control ratios field-by-field and wave-by-wave.
        wave_metrics = add_wave_metric_ratios(wave_metrics)
        write_cached_table("wave_band_variance_metrics", wave_metrics, wave_metrics_signature)
        print("\nWave-band variance metrics")
        print(wave_metrics.to_string(index=False))
        if save:
            wave_metrics.to_csv(os.path.join(save_loc, "wave_band_variance_metrics.csv"), index=False)

if should_build_figure(2, ["wave_power_absolute_and_change", "wave_power_fraction_of_total"]) \
        and wave_metrics is not None and len(wave_metrics) > 0:
    plot_fields = ["Precip", "OLR", "PW", "cldfrac", "u850", "w500", "T100"]
    plot_fields = [f for f in plot_fields if f in set(wave_metrics.field)]

    # Main figure: absolute wave-band power (log) and percentage change from
    # control. The change row makes quantitative what the log bars cannot show.
    fig, ax = plt.subplots(2, len(plot_fields), figsize=(3.2 * len(plot_fields), 6.0), constrained_layout=True)
    if len(plot_fields) == 1:
        ax = ax[:, None]

    for j, field in enumerate(plot_fields):
        sub = wave_metrics[wave_metrics.field == field]
        x = np.arange(len(wave_names))
        width = 0.8 / len(dss)
        for i, sim in enumerate(sim_names):
            s = sub[sub.simulation == sim].set_index("wave").reindex(wave_names)
            offset = (i - (len(dss) - 1) / 2) * width
            ax[0, j].bar(x + offset, s.wave_variance.values, width,
                         color=cols[i], label=sim if j == 0 else None)
            if i > 0:
                pct = 100.0 * (s.power_ratio_to_control.values - 1.0)
                ax[1, j].bar(x + offset, pct, width, color=cols[i])
        ax[0, j].set_title(field)
        ax[0, j].set_xticks(x)
        ax[0, j].tick_params(labelbottom=False)
        ax[1, j].set_xticks(x)
        ax[1, j].set_xticklabels([wave_label(w) for w in wave_names], rotation=35, ha="right")
        ax[1, j].axhline(0, color="0.6", lw=0.7)
        ax[0, j].set_yscale("log")
        ax[0, j].grid(alpha=0.2)
        ax[1, j].grid(alpha=0.2)
    ax[0, 0].set_ylabel("Wave-band power (field units$^2$)")
    ax[1, 0].set_ylabel("Wave-band power change vs control (%)")
    ax[0, 0].legend(loc="best", fontsize=8)
    figsave.save(fig, "wave_power_absolute_and_change", number=2)
    print("Done with figure: wave_power_absolute_and_change")

    # Context figure: how much of each field's total anomaly power each wave
    # band explains, as one annotated percentage heatmap per simulation.
    # One denominator per field and simulation (symmetric + antisymmetric
    # latitude-mean power), so cells are comparable across wave bands and
    # roughly summable; the remainder is unfiltered background variability.
    metrics_common = add_common_power_fractions(wave_metrics)
    frac_pivots = []
    for sim in sim_names:
        sub = metrics_common[metrics_common.simulation == sim]
        pivot = sub.pivot_table(index="wave", columns="field", values="wave_fraction_of_total", aggfunc="mean")
        frac_pivots.append(
            100.0 * pivot.reindex(index=wave_names, columns=plot_fields).values.astype(float)
        )
    positive = np.concatenate(
        [v[np.isfinite(v) & (v > 0)].ravel() for v in frac_pivots]
    )
    if positive.size > 0:
        norm = colors.LogNorm(vmin=positive.min(), vmax=positive.max())
        fig, axes = plt.subplots(
            1, len(sim_names),
            figsize=(0.62 * len(plot_fields) * len(sim_names) + 3.0, 0.5 * len(wave_names) + 2.2),
            constrained_layout=True, sharey=True,
        )
        for k, (axk, sim) in enumerate(zip(np.atleast_1d(axes), sim_names)):
            values = frac_pivots[k]
            im = axk.imshow(values, cmap="viridis", norm=norm, aspect="auto")
            for r in range(values.shape[0]):
                for c in range(values.shape[1]):
                    v = values[r, c]
                    if np.isfinite(v) and v > 0:
                        axk.text(c, r, f"{v:.2g}", ha="center", va="center", fontsize=7,
                                 color="k" if norm(v) > 0.6 else "w")
            axk.set_title(sim)
            axk.set_xticks(np.arange(len(plot_fields)))
            axk.set_xticklabels(plot_fields, rotation=35, ha="right")
            axk.set_yticks(np.arange(len(wave_names)))
            axk.set_yticklabels([wave_label(w) for w in wave_names])
            axk.grid(False)
        fig.colorbar(im, ax=np.atleast_1d(axes).ravel(), shrink=0.9,
                     label="Share of the field's total power (%)")
        figsave.save(fig, "wave_power_fraction_of_total", number=2)
        print("Done with figure: wave_power_fraction_of_total")


# -----------------------------------------------------------------------------
# 3. Lagged coupling between wave amplitude and domain-mean fields
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Positive lag means wave amplitude leads the target field.
#   * If u_nudge increases correlation at positive lags, waves may be triggering
#     or organising convection more effectively.
#   * If correlations are similar but wave power differs, occurrence/intensity
#     rather than coupling is the dominant change.

if should_build_figure(3, "wave_amplitude_lag_correlations"):
    lag_days_max = 10.0
    lag_signature = cache_signature(
        "wave_amplitude_lag_correlations",
        extra=dict(
            waves=list(wave_names),
            lag_target_fields=list(lag_target_fields),
            lag_days_max=lag_days_max,
            filter_lat_bounds=jsonable_value(filter_lat_bounds),
        ),
    )
    lag_df = load_cached_table(
        "wave_amplitude_lag_correlations",
        lag_signature,
        output_filename="wave_amplitude_lag_correlations.csv",
        required_columns=("simulation", "wave", "target", "lag_days", "correlation"),
    )
    if lag_df is None:
        lag_df = pd.DataFrame()
    completed_lag_keys = table_key_set(lag_df, ("simulation", "wave", "target"))

    for wname in wave_names:
        projection = lat_projection_for_wave(wname)
        for target in lag_target_fields:
            for i, ds in enumerate(dss):
                key = (sim_names[i], wname, target)
                if key in completed_lag_keys:
                    continue
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
                    rows = [
                        dict(
                            simulation=sim_names[i],
                            wave=wname,
                            wave_label=wave_label(wname),
                            latitude_projection=projection,
                            target=target,
                            lag_days=lg,
                            correlation=cc,
                        )
                        for lg, cc in zip(lag_days, corr)
                    ]
                    lag_df = append_cached_rows(
                        "wave_amplitude_lag_correlations",
                        lag_df,
                        rows,
                        lag_signature,
                    )
                    completed_lag_keys.add(key)
                except Exception as exc:
                    print(f"Skipping lag correlation {sim_names[i]} {wname} {target}: {exc}")

    fig, ax = plt.subplots(len(wave_names), len(lag_target_fields),
                           figsize=(3.6 * len(lag_target_fields), 2.3 * len(wave_names)),
                           constrained_layout=True, sharex=True, sharey=True)
    if len(wave_names) == 1:
        ax = ax[None, :]
    if len(lag_target_fields) == 1:
        ax = ax[:, None]

    for wj, wname in enumerate(wave_names):
        for tj, target in enumerate(lag_target_fields):
            for i, ds in enumerate(dss):
                s = lag_df[
                    (lag_df.simulation == sim_names[i])
                    & (lag_df.wave == wname)
                    & (lag_df.target == target)
                ].sort_values("lag_days")
                if len(s) == 0:
                    continue
                ax[wj, tj].plot(
                    s.lag_days,
                    s.correlation,
                    color=cols[i],
                    label=sim_names[i] if (wj == 0 and tj == 0) else None,
                )
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

if should_build_figure(4, "wave_conditioned_precip_olr_pw"):
    amp_bins = np.linspace(0, 3.5, 22)
    cond_signature = cache_signature(
        "wave_conditioned_precip_olr_pw",
        extra=dict(
            waves=list(wave_names),
            conditioned_fields=list(conditioned_fields),
            amp_bins=jsonable_value(amp_bins),
            filter_lat_bounds=jsonable_value(filter_lat_bounds),
        ),
    )
    cond_df = load_cached_table(
        "wave_conditioned_precip_olr_pw",
        cond_signature,
        output_filename="wave_conditioned_precip_olr_pw.csv",
        required_columns=("simulation", "wave", "target", "amp_norm_bin", "mean", "count"),
    )
    if cond_df is None:
        cond_df = pd.DataFrame()
    completed_cond_keys = table_key_set(cond_df, ("simulation", "wave", "target"))

    for wname in wave_names:
        projection = lat_projection_for_wave(wname)
        for target in conditioned_fields:
            for i, ds in enumerate(dss):
                key = (sim_names[i], wname, target)
                if key in completed_cond_keys:
                    continue
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
                    rows = [
                        dict(
                            simulation=sim_names[i],
                            wave=wname,
                            wave_label=wave_label(wname),
                            latitude_projection=projection,
                            target=target,
                            amp_norm_bin=xx,
                            mean=yy,
                            p25=lo,
                            p75=hi,
                            count=cnum,
                        )
                        for xx, yy, lo, hi, cnum in zip(xc, ym, y25, y75, count)
                    ]
                    cond_df = append_cached_rows(
                        "wave_conditioned_precip_olr_pw",
                        cond_df,
                        rows,
                        cond_signature,
                    )
                    completed_cond_keys.add(key)
                except Exception as exc:
                    print(f"Skipping conditional curve {sim_names[i]} {wname} {target}: {exc}")

    fig, ax = plt.subplots(len(wave_names), len(conditioned_fields),
                           figsize=(3.6 * len(conditioned_fields), 2.3 * len(wave_names)),
                           constrained_layout=True, sharex=True)
    if len(wave_names) == 1:
        ax = ax[None, :]
    if len(conditioned_fields) == 1:
        ax = ax[:, None]

    for wj, wname in enumerate(wave_names):
        for tj, target in enumerate(conditioned_fields):
            for i, ds in enumerate(dss):
                s = cond_df[
                    (cond_df.simulation == sim_names[i])
                    & (cond_df.wave == wname)
                    & (cond_df.target == target)
                ].sort_values("amp_norm_bin")
                if len(s) == 0:
                    continue
                ax[wj, tj].plot(
                    s.amp_norm_bin,
                    s["mean"],
                    color=cols[i],
                    label=sim_names[i] if (wj == 0 and tj == 0) else None,
                )
                if {"p25", "p75"}.issubset(s.columns):
                    ax[wj, tj].fill_between(s.amp_norm_bin, s.p25, s.p75, color=cols[i], alpha=0.12, linewidth=0)
            if wj == 0:
                ax[wj, tj].set_title(target)
            if tj == 0:
                ax[wj, tj].set_ylabel(wave_label(wname))
            if wj == len(wave_names) - 1:
                ax[wj, tj].set_xlabel(r"$|P'_W| / \sigma(|P'_W|)$")
            ax[wj, tj].grid(alpha=0.2)
    ax[0, 0].legend(loc="best")
    figsave.save(fig, "wave_conditioned_precip_olr_pw", number=4)
    if save and len(cond_df) > 0:
        cond_df.to_csv(os.path.join(save_loc, "wave_conditioned_precip_olr_pw.csv"), index=False)
    print("Done with figure: wave_conditioned_precip_olr_pw")


# -----------------------------------------------------------------------------
# 5. Intense-precipitation attribution to wave bands
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Which wave band is largest at high-precipitation longitudes/times?
#   * If u_nudge increases intense precipitation only when Kelvin/EIG/IG/Easterly/ER
#     amplitude is large, that wave band is a candidate mediator.

intense_precip_attribution = None
if should_build_figure(5, "intense_precip_wave_attribution"):
    percentile = 95.0
    intense_precip_attribution = compute_intense_precip_wave_attribution(percentile=percentile)
    attrib = intense_precip_attribution
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
#   * Does u_nudge amplify UTLS wave variance in u/T/w more than T_nudge?
#   * Does the precipitation response line up with increased lower-tropospheric
#     ascent, upper-tropospheric temperature variance, or near-tropopause wave
#     activity?

if should_build_figure(6, "vertical_wave_power_profiles"):
    profile_signature = cache_signature(
        "vertical_wave_power_profiles",
        extra=dict(
            vertical_profile_waves=list(vertical_profile_waves),
            vertical_profile_fields=list(vertical_profile_fields),
            vertical_profile_levels=jsonable_value(vertical_profile_levels),
            cldfrac_profile_heights_km=jsonable_value(cldfrac_profile_heights_km),
            filter_lat_bounds=jsonable_value(filter_lat_bounds),
        ),
    )
    prof = load_cached_table(
        "vertical_wave_power_profiles",
        profile_signature,
        output_filename="vertical_wave_power_profiles.csv",
        required_columns=("simulation", "simulation_index", "wave", "field", "vertical_coord", "wave_variance"),
    )
    if prof is None:
        prof = pd.DataFrame()
    completed_profile_keys = table_key_set(prof, ("simulation_index", "wave", "field", "vertical_coord"))

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
                    key = (i, wname, var, level)
                    if key in completed_profile_keys:
                        continue
                    try:
                        wav = get_filtered_wave(ds, i, field, wname, avgLat=True)
                        row = dict(simulation=sim_names[i], simulation_index=i,
                                   wave=wname, wave_label=wave_label(wname),
                                   latitude_projection=projection,
                                   field=var,
                                   vertical_coord=level,
                                   vertical_axis=vertical_axis,
                                   vertical_units=vertical_units,
                                   wave_variance=wave_variance(wav))
                        prof = append_cached_rows(
                            "vertical_wave_power_profiles",
                            prof,
                            [row],
                            profile_signature,
                        )
                        completed_profile_keys.add(key)
                    except Exception as exc:
                        print(f"Skipping vertical wave power {sim_names[i]} {wname} {field}: {exc}")
    if len(prof) > 0:
        write_cached_table("vertical_wave_power_profiles", prof, profile_signature)
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
                    ax[fj, wj].set_yscale("log")
                    ax[fj, wj].set_ylim(1000, 50)
                ax[fj, wj].grid(alpha=0.2)
                if fj == 0:
                    ax[fj, wj].set_title(wave_label(wname))
                if wj == 0:
                    ylabel = f"{var}\nHeight (km)" if var == "cldfrac" else f"{var}\nPressure (hPa)"
                    ax[fj, wj].set_ylabel(ylabel)
                if fj == len(vertical_profile_fields) - 1:
                    ax[fj, wj].set_xlabel("wave power")
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
#   * The vertical lines are each band's power-weighted mean phase speed from
#     that simulation's own spectrum (matching the distributions below), not a
#     fixed property of the filter box.

if should_build_figure(7, "critical_level_phase_speed_profiles"):
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

    phase_speed_dist = compute_wave_phase_speed_distributions()

    def power_weighted_mean_phase_speed(sim_index, wname):
        if len(phase_speed_dist) == 0:
            return np.nan
        sub = phase_speed_dist[
            (phase_speed_dist.simulation_index == sim_index) & (phase_speed_dist.wave == wname)
        ]
        w = np.clip(np.nan_to_num(sub.density_per_ms.values, nan=0.0), 0.0, None)
        if w.sum() <= 0:
            return np.nan
        return float(np.sum(sub.phase_speed_ms.values * w) / w.sum())

    fig, ax = plt.subplots(
        2, len(dss_zm),
        figsize=(3.8 * len(dss_zm), 6.2),
        constrained_layout=True,
        sharey="row",
        sharex="col",
        gridspec_kw={"height_ratios": [2.4, 1.2]},
    )
    if len(dss_zm) == 1:
        ax = np.asarray(ax).reshape(2, 1)
    profile_ax = ax[0, :]
    dist_ax = ax[1, :]

    for i, zds in enumerate(dss_zm):
        a = profile_ax[i]
        u = zds.uzonal_isobaric.sel(latitude=slice(filter_lat_bounds[0], filter_lat_bounds[1])).mean(dim=("Time", "latitude"))
        a.plot(u.values, u.pres.values, color=cols[i], lw=1.8, label=sim_names[i])
        for wname in wave_names:
            # This simulation's power-weighted mean phase speed for the band,
            # colour-matched to the distribution below. Fall back to the
            # filter-box centre estimate if no distribution is available.
            c_mean = power_weighted_mean_phase_speed(i, wname)
            if not np.isfinite(c_mean):
                c_mean = float(speed_df.loc[speed_df.wave == wname, "c_central_ms"].iloc[0])
            a.axvline(c_mean, color=spectrum_outline_colors.get(wname, "0.5"), lw=1.0, ls="--", alpha=0.8)
        a.axvline(0, color="0.7", lw=0.6)
        a.set_title(sim_names[i])
        a.set_yscale("log")
        a.set_ylim(1000, 50)
        a.grid(alpha=0.2)
        annotate_panel_label(a, sim_names[i])

        b = dist_ax[i]
        sim_dist = phase_speed_dist[phase_speed_dist.simulation_index == i] if len(phase_speed_dist) else pd.DataFrame()
        for wname in wave_names:
            sub = sim_dist[sim_dist.wave == wname].sort_values("phase_speed_ms") if len(sim_dist) else pd.DataFrame()
            if len(sub) == 0 or not np.any(np.isfinite(sub.density_per_ms.values)):
                continue
            b.plot(
                sub.phase_speed_ms,
                sub.density_per_ms,
                color=spectrum_outline_colors.get(wname, "0.1"),
                lw=1.5,
                label=wave_label(wname) if i == 0 else None,
            )
        b.axvline(0, color="0.7", lw=0.6)
        b.set_xlim(*critical_level_phase_speed_xlim)
        b.set_xlabel(r"$\overline{u}$ and zonal phase speed (m s$^{-1}$)")
        b.grid(alpha=0.2)
        if i == 0:
            b.set_ylabel("Power-weighted\ndensity")
        else:
            b.tick_params(labelleft=False)
    profile_ax[0].set_ylabel("Pressure (hPa)")
    if len(phase_speed_dist) > 0:
        dist_ax[0].legend(loc="best", fontsize=7, ncol=2)
    fig.suptitle(
        f"Tropical mean wind profile with power-weighted mean phase speeds (dashed) and "
        f"phase-speed distributions over {filter_lat_bounds[0]}--{filter_lat_bounds[1]}°"
    )
    figsave.save(fig, "critical_level_phase_speed_profiles", number=7)
    print("Done with figure: critical_level_phase_speed_profiles")


# -----------------------------------------------------------------------------
# 8. Wave-filtered eddy fluxes (heavier, latitude-resolved filtering)
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Kelvin-like waves should be dominated by u/w/T structure rather than
#     meridional momentum flux. Easterly/ER/IG changes in u'v', EKE or vorticity-like
#     flow would point to rotational-wave mediation.
#   * Run this after the cheaper figures, because avgLat=False applies the k-f
#     filter separately to every retained latitude.

if should_build_figure(8, "wave_filtered_eddy_momentum_fluxes"):
    flux_signature = cache_signature(
        "wave_filtered_eddy_fluxes",
        extra=dict(
            eddy_flux_waves=list(eddy_flux_waves),
            eddy_flux_levels=jsonable_value(eddy_flux_levels),
            eddy_flux_lat_bounds=jsonable_value(eddy_flux_lat_bounds),
        ),
    )
    flux = load_cached_table(
        "wave_filtered_eddy_fluxes",
        flux_signature,
        output_filename="wave_filtered_eddy_fluxes.csv",
        required_columns=("simulation", "wave", "pres", "latitude", "uv", "vT", "eke"),
    )
    if flux is None:
        flux = pd.DataFrame()
    completed_flux_keys = table_key_set(flux, ("simulation", "wave", "pres"))

    for i, ds in enumerate(dss):
        for wname in eddy_flux_waves:
            for plev in eddy_flux_levels:
                key = (sim_names[i], wname, plev)
                if key in completed_flux_keys:
                    continue
                try:
                    uw = get_filtered_wave(ds, i, f"u{plev}", wname, avgLat=False, lat_bounds=eddy_flux_lat_bounds)
                    vw = get_filtered_wave(ds, i, f"v{plev}", wname, avgLat=False, lat_bounds=eddy_flux_lat_bounds)
                    Tw = get_filtered_wave(ds, i, f"T{plev}", wname, avgLat=False, lat_bounds=eddy_flux_lat_bounds)
                    uw, vw, Tw = xr.align(uw, vw, Tw, join="inner")
                    uv = (uw * vw).mean(dim=("Time", "longitude"), skipna=True)
                    vT = (vw * Tw).mean(dim=("Time", "longitude"), skipna=True)
                    eke = 0.5 * (uw**2 + vw**2).mean(dim=("Time", "longitude"), skipna=True)
                    rows = []
                    for lat, uvv, vTv, ekev in zip(uv.latitude.values, uv.values, vT.values, eke.values):
                        rows.append(dict(simulation=sim_names[i], wave=wname,
                                         wave_label=wave_label(wname), pres=plev,
                                         latitude_projection="latitude_resolved",
                                         latitude=lat, uv=uvv, vT=vTv, eke=ekev))
                    flux = append_cached_rows(
                        "wave_filtered_eddy_fluxes",
                        flux,
                        rows,
                        flux_signature,
                    )
                    completed_flux_keys.add(key)
                except Exception as exc:
                    print(f"Skipping wave-filtered eddy flux {sim_names[i]} {wname} {plev}: {exc}")
    if len(flux) > 0:
        write_cached_table("wave_filtered_eddy_fluxes", flux, flux_signature)
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

if should_build_figure(
    9,
    [
        "symmetric_antisymmetric_wave_spectra",
        "symmetric_antisymmetric_wave_spectra_period",
        "symmetric_antisymmetric_wave_spectra_phase_speed",
    ],
):
    spec_levels = np.linspace(0, 3.0, 26)
    spec_cmap, spec_norm = make_spectrum_cmap(spec_levels)

    components = spectrum_components
    spectra_by_sim = load_or_compute_normalised_spectra_by_sim()

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


# -----------------------------------------------------------------------------
# 10. Phase-speed distributions by simulation
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Whether a simulation changes the preferred phase speed of a wave band, not
#     only its integrated wave-box power.
#   * Broad or multi-peaked distributions mean the single phase-speed line in
#     the critical-level diagnostic should be interpreted cautiously.

if should_build_figure(10, "wave_phase_speed_power_distributions"):
    dist = compute_wave_phase_speed_distributions()
    if len(dist) > 0:
        if save:
            dist.to_csv(os.path.join(save_loc, "wave_phase_speed_power_distributions.csv"), index=False)

        ncols = min(3, len(wave_names))
        nrows = int(np.ceil(len(wave_names) / ncols))
        fig, ax = plt.subplots(
            nrows, ncols,
            figsize=(4.0 * ncols, 2.8 * nrows),
            constrained_layout=True,
            sharey=True,
        )
        axes = np.asarray(ax).reshape(-1)
        for wj, wname in enumerate(wave_names):
            a = axes[wj]
            wave_dist = dist[dist.wave == wname]
            for i, sim in enumerate(sim_names):
                sub = wave_dist[wave_dist.simulation == sim].sort_values("phase_speed_ms")
                if len(sub) == 0 or not np.any(np.isfinite(sub.density_per_ms.values)):
                    continue
                a.plot(sub.phase_speed_ms, sub.density_per_ms, color=cols[i], lw=1.7, label=sim)
            cmin, ccen, cmax = wave_phase_speed_range(wname)
            a.axvline(0, color="0.75", lw=0.6)
            a.axvline(ccen, color="0.45", lw=0.8, ls="--")
            occupied = wave_dist[np.isfinite(wave_dist.weighted_power) & (wave_dist.weighted_power > 0.0)]
            if len(occupied) > 0:
                x0 = float(occupied.phase_speed_ms.min())
                x1 = float(occupied.phase_speed_ms.max())
            else:
                x0, x1 = sorted((cmin, cmax))
            margin = max(10.0, 0.08 * max(1.0, x1 - x0))
            x0 = max(phase_speed_distribution_limits[0], x0 - margin)
            x1 = min(phase_speed_distribution_limits[1], x1 + margin)
            if x0 >= x1:
                x0, x1 = phase_speed_distribution_limits
            a.set_xlim(x0, x1)
            a.set_title(wave_label(wname))
            a.grid(alpha=0.2)
            if wj % ncols == 0:
                a.set_ylabel("Power-weighted\ndensity")
            if wj >= (nrows - 1) * ncols:
                a.set_xlabel(r"Zonal phase speed (m s$^{-1}$)")
        for a in axes[len(wave_names):]:
            a.set_visible(False)
        axes[0].legend(loc="best", fontsize=8)
        fig.suptitle(
            f"Spectrum-derived phase-speed distributions "
            f"({phase_speed_distribution_power_mode.replace('_', ' ')})"
        )
        figsave.save(fig, "wave_phase_speed_power_distributions", number=10)
        print("Done with figure: wave_phase_speed_power_distributions")
    else:
        print("Skipping figure: wave_phase_speed_power_distributions (no distribution data)")


# -----------------------------------------------------------------------------
# 11. Compact attribution metrics
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * The paired heatmaps show each run's wave-power change from control; the
#     part of the u-nudged change missing from the T-nudged panel is the
#     wind/shear-mediated response.
#   * Large cloud/PW/T100/momentum changes without matching precipitation
#     changes suggest altered coupling and organisation rather than just
#     stronger bulk rainfall.

compact_attribution_figure_stems = [
    "wave_attribution_change_heatmaps",
    "wave_attribution_summary_bars",
]
compact_signature = cache_signature(
    "compact_wave_attribution_metrics",
    extra=dict(
        wave_metric_fields=list(power_fields),
        waves=list(wave_names),
        intense_precip_percentile=95.0,
        attribution_eps=ATTRIBUTION_EPS,
        fraction_denominator="symmetric_plus_antisymmetric",
    ),
)
if should_build_versioned_figure(WAVE_ATTRIBUTION_FIGURE_NUMBER, compact_attribution_figure_stems, compact_signature):
    compact_attribution = load_cached_table(
        "compact_wave_attribution_metrics",
        compact_signature,
        output_filename="wave_attribution_metrics.csv",
        required_columns=("metric_name", "variable", "wave_band", "condition", "wind_unique"),
    )
    if compact_attribution is None:
        if intense_precip_attribution is None:
            intense_precip_attribution = compute_intense_precip_wave_attribution(percentile=95.0)
            if save and len(intense_precip_attribution) > 0:
                intense_precip_attribution.to_csv(os.path.join(save_loc, "intense_precip_wave_attribution.csv"), index=False)

        compact_attribution = build_compact_wave_attribution_table(
            wave_metrics,
            intense_attribution_df=intense_precip_attribution,
        )
        if len(compact_attribution) > 0:
            write_cached_table("compact_wave_attribution_metrics", compact_attribution, compact_signature)
    if len(compact_attribution) > 0:
        print("\nCompact wave attribution metrics")
        print(compact_attribution.to_string(index=False))
        if save:
            compact_attribution.to_csv(os.path.join(save_loc, "wave_attribution_metrics.csv"), index=False)
        plot_compact_wave_attribution(compact_attribution)
        write_figure_output_metadata(
            WAVE_ATTRIBUTION_FIGURE_NUMBER,
            compact_attribution_figure_stems,
            compact_signature,
        )
        print("Done with figure: compact_wave_attribution_metrics")
    else:
        print("Skipping compact wave attribution: no source metrics were available")

print("\nFinished equatorial_wave_diagnostics.py")
