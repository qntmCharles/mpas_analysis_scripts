#!/usr/bin/env python
# coding: utf-8

"""
Equatorial wave diagnostics for QBO-like UTLS nudging experiments.

This script is intended to sit alongside:
  * mean_state_diagnostics.py
  * symmetry_convection_diagnostics.py

It does not assign a planetary intraseasonal mode to this aquaplanet.
Instead it asks whether the precipitation/cloud response is associated with
particular convectively coupled equatorial wave (CCEW) bands, and whether the
u-nudged/T-nudged UTLS changes alter the precipitation response to those waves.

The filtering uses Straub/Kiladis-style k-f boxes via the low-level kf_filter.
The diagnostic-specific wave collections are chosen for this simulation set:
  Eastward intraseasonal: k=1..5, period=20..100 d, symmetric
  Kelvin:       k=1..14,    period=2.5..20 d,  equivalent depth=8..90 m
  EIG:          k=1..14,    period=1.8..4.5 d, equivalent depth=8..90 m
  IG antisym.:  k=-15..-1,  period=1.25..2 d,  equivalent depth=8..90 m
  Easterly:     k=-28..-6,  period=2.5..7 d,   no depth constraint
  ER:           k=-10..-1,  period=10..40 d,   equivalent depth=8..90 m

Wave-filtered total high cloud and attribution classes are regressed on the
same-band precipitation predictors here. Track-following composites reuse this
script's caches in ``lpo_lifecycle_analysis.py``.

The easterly-wave definition follows Rios-Berrios et al. (2023). It is a
rectangular TD/easterly-wave band rather than a shallow-water normal mode.

The script caches filtered fields beside the source simulations so that figures
can be regenerated without re-running the k-f filters. All machine-readable
CSV and NetCDF products remain in that fixed data-cache tree; only rendered
figures are written below ``~/tmp``.
"""

import argparse
import os, sys, numbers, json
from functools import partial

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
from scipy import fft, signal

from diagnostics.diagnostic_helpers import (
    ANALYSIS_END, ANALYSIS_START, FigureSaver, LATLON_PATTERN,
    ZONAL_MEAN_PATTERN, add_period_reference_lines_frequency,
    align_vertical_velocity_to_reference,
    add_symmetric_colorbar,
    annotate_panel_label,
    binned_mean_1d, configured_label, cosine_weighted_power_input,
    diagnostic_cache_dir,
    diagnostic_output_dir, difference_cmap, figure_should_run,
    figure_should_run_any, jsonable_value,
    load_latlon_dataset, make_spectrum_cmap, netcdf_safe_dataset,
    mpas_moist_air_density, native_midlevel_height,
    numbered_figure_saver, precip_rate_from_accum, read_json_file,
    pressure_to_pa, progress, safe_name, simulation_colour,
    standardize_latlon_dataset, symmetric_levels,
    total_precip_accum, vertical_dim, write_json_file,
)
from diagnostics.waves.equatorial_wave_config import (
    ALL_CACHED_FILTERS,
    EASTWARD_INTRASEASONAL_FILTER,
    EDDY_FLUX_FILTERS,
    FILTERED_WAVE_PRODUCT_VERSION,
    LPO_LIFECYCLE_FILTERS,
    MISSING_EQUIVALENT_DEPTH,
    SHORT_LAG_FILTERS,
    SHORT_PERIOD_CRITICAL_LEVEL_FILTERS,
    TOP_HEAVINESS_FILTERS,
    VERTICAL_STRUCTURE_FILTERS,
    WAVE_RESPONSE_FILTERS,
    WAVE_BOXES,
    WAVE_CONFIG_VERSION,
    WAVE_LATITUDE_PROJECTION,
    WAVE_VARIABILITY_FIELDS,
)
from diagnostics.waves.cached_wave_products import (
    LPO_WAVE_SECTION_CACHE_VERSION,
    lpo_wave_section_cache_path,
    wave_response_contrasts_product_path,
    wave_response_metrics_product_path,
)
from diagnostics.convection.lpo_tracking import (
    DEFAULT_TEST_DAYS as DEFAULT_LPO_TEST_DAYS,
    DEFAULT_TEST_MINIMUM_DURATION_DAYS,
    PRODUCT_VERSION as LPO_TRACKING_PRODUCT_VERSION,
    SIMULATION_DIRS as LPO_SIMULATION_DIRS,
    TROPICAL_CENTROID_LIMIT_DEG,
    require_actual_test_products,
    require_publication_products,
)
from diagnostics.convection.high_cloud_classification import (
    CLASSIFICATION_CACHE_VERSION,
    HighCloudThresholds,
    cached_classification_is_current,
)
from diagnostics.convection.lpo_lifecycle_sections import (
    RAY_HEIGHT_M,
    RAY_RELATIVE_LATITUDE_DEGREES,
    RAY_RELATIVE_LONGITUDE_DEGREES,
)
from diagnostics.waves.wave_metrics import (
    METRICS_VERSION as WAVE_METRICS_VERSION,
    block_bootstrap_filtered_wave_metrics,
    buoyancy_frequency_squared,
    circular_time_block_indices,
    common_vertical_motion_eofs,
    complex_top_heaviness,
    filtered_wave_metrics,
    filtered_wave_phase_diagnostics,
    kelvin_group_velocities,
    pressure_layer_weights,
    project_onto_eofs,
    record_subsets,
    resolved_frequency_count,
    spectral_coefficients_for_box,
    upward_pressure_velocity,
)
from diagnostics.ttl import TTL_BASE_PRESSURE_HPA
from wavenumber_frequency_functions import spacetime_power, smooth_wavefreq
from waveFilter import kf_filter

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

DEFAULT_WAVE_TEST_DAYS = 40
MINIMUM_WAVE_TEST_DAYS = 21


def parse_command_line(argv=None):
    """Parse the executable wave-diagnostic interface."""

    parser = argparse.ArgumentParser(
        description=(
            "Build equatorial-wave caches, metrics and figures from the three "
            "configured MPAS-A simulations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Use a bounded subset of the actual simulations and isolated test "
            "caches and figures."
        ),
    )
    parser.add_argument(
        "--test-start",
        default=None,
        metavar="TIME",
        help="First six-hourly time in the actual-data test subset.",
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_WAVE_TEST_DAYS,
        metavar="N",
        help=(
            "Length of the wave test record. At least 21 days is required so "
            "the differenced precipitation record resolves the shortest "
            "eastward-intraseasonal period."
        ),
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=range(1, 15),
        default=None,
        help="Build only these numbered figures.",
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--rebuild-cache",
        "--force-recompute-caches",
        dest="force_recompute_caches",
        action="store_true",
        help=(
            "Ignore reusable wave, intermediate and product caches once per "
            "cache path during this run, then replace them atomically."
        ),
    )
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Build selected figures without opening raw model histories. "
            "Fail when a required reusable product is absent or stale."
        ),
    )
    parser.add_argument(
        "--allow-stale-lpo-products",
        action="store_true",
        help=(
            "Use existing publication LPO tables with an older product "
            "version when rebuilding Figure 14 wave sections."
        ),
    )
    args = parser.parse_args(argv)
    if args.test and args.test_days < MINIMUM_WAVE_TEST_DAYS:
        parser.error(
            f"--test-days must be at least {MINIMUM_WAVE_TEST_DAYS} to "
            "resolve the eastward-intraseasonal filter"
        )
    if not args.test and args.test_start is not None:
        parser.error("--test-start requires --test")
    if args.test and args.allow_stale_lpo_products:
        parser.error("--allow-stale-lpo-products supports publication mode only")
    if args.figures:
        args.figures = list(dict.fromkeys(args.figures))
    return args


runtime_args = parse_command_line()
test_mode = bool(runtime_args.test)
analysis_start = pd.Timestamp(runtime_args.test_start or ANALYSIS_START)
analysis_end = (
    analysis_start
    + pd.Timedelta(days=runtime_args.test_days)
    - pd.Timedelta(hours=6)
    if test_mode
    else pd.Timestamp(ANALYSIS_END)
)
if analysis_start.minute or analysis_start.second or analysis_start.hour % 6:
    raise SystemExit(
        "--test-start must lie on the simulation's six-hourly time grid"
    )
ANALYSIS_START = analysis_start.strftime("%Y-%m-%d %H:%M:%S")
ANALYSIS_END = analysis_end.strftime("%Y-%m-%d %H:%M:%S")

save = True
save_loc = diagnostic_output_dir(
    "/home/cwp29/tmp/mpas/crf_on/final/waves/testing/"
    if test_mode
    else "/home/cwp29/tmp/mpas/crf_on/final/waves/"
)

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
#       plus standard T100 and u100 frequency-wavenumber spectra
#  10:  wave_phase_speed_power_distributions
#   11: wave precipitation amplitude and cloud/OLR response efficiency
#   12: Sakaeda/Inoue two-EOF top-heaviness
#   13: Lim-Son Kelvin reference energy dispersion
#   14: compact LPO-following wave sections (explicit prerequisite)
#
# LPO lifecycle composites live in ``lpo_lifecycle_analysis.py`` and reuse
# these wave caches. Wave-response regressions are owned by this script.
figures_to_run = runtime_args.figures or [1, 2, 6, 7, 11, 12, 13, 14]
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)

# Restart/resume controls.  With the default full run, existing figure files are
# treated as completed products; set this to False if you explicitly want to
# redraw figures from cached/intermediate data.
skip_existing_figures = False
force_recompute_caches = bool(runtime_args.force_recompute_caches)
_rebuilt_filter_input_paths = set()
_rebuilt_wave_cache_paths = set()

# Figure 8 is deliberately excluded from the default list because it filters
# latitude-resolved u and v. Add 8 to figures_to_run after the cheaper figures
# have run and cached their filtered fields.

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control_new", "u_nudge", "T_nudge"]
sim_names = ["control", "nudged u", "nudged T"]
cols = [simulation_colour(name) for name in sim_names]

dt = 6
obs_per_day = int(round(24 / dt))

# User-defined filtering latitude band.  The cache filename includes this band,
# so changing it creates a separate set of filtered-wave files.
#
# This is the full band for symmetric/even signals.  Antisymmetric/odd signals
# are projected from the paired hemispheres (0..L and -L..0), where L is the
# larger absolute latitude bound below.
filter_lat_bounds = (-10, 10)
filter_avg_lat = True

WAVE_RESPONSE_FIGURE_NUMBER = 11
WAVE_RESPONSE_PRODUCT_VERSION = (
    "wave-response-efficiency-v3-tropics-20deg"
)
WAVE_RESPONSE_BOOTSTRAP_RESAMPLES = 2000
WAVE_RESPONSE_BOOTSTRAP_SEED = 20240602
WAVE_RESPONSE_CI_PERCENTILES = (2.5, 97.5)
WAVE_RESPONSE_ROBUST_SIGN_THRESHOLD = 0.95
WAVE_RESPONSE_SCATTER_SAMPLE = 20000
WAVE_RESPONSE_BOOTSTRAP_REFERENCE = (
    "Kunsch (1989), Annals of Statistics, doi:10.1214/aos/1176347265"
)
WAVE_RESPONSE_CLOUD_FIELDS = (
    ("total_high_cloud", "Total high cloud", "high_cloud"),
    ("active_high_cloud", "Active deep convection", "active_high_cloud"),
    (
        "stratiform_high_cloud",
        "Stratiform precipitating high cloud",
        "stratiform_high_cloud",
    ),
    (
        "nonprecipitating_high_cloud",
        "Non-precipitating high cloud",
        "nonprecipitating_high_cloud",
    ),
    (
        "ttl_reaching_high_cloud",
        "TTL-reaching high cloud",
        "ttl_reaching_high_cloud",
    ),
)

# Heavier latitude-resolved filtering in figure 8 can use its own wider band.
eddy_flux_lat_bounds = (-20, 20)

# Analysis collections are defined centrally so slow intraseasonal variability
# is not forced into short-lag or shallow-water diagnostics.
wave_names = list(ALL_CACHED_FILTERS)
lag_wave_names = list(SHORT_LAG_FILTERS)
critical_level_wave_names = list(SHORT_PERIOD_CRITICAL_LEVEL_FILTERS)

# Latitude projection by wave.  Choices are:
#   symmetric:       average over filter_lat_bounds, suitable for even modes
#   antisymmetric:   0.5 * (0..L - -L..0), suitable for odd modes
#   direct:          average without imposing parity
#
# Identifications used here:
#   Kelvin and ER are symmetric/even.
#   EIG is the eastward n=0 inertio-gravity branch, antisymmetric/odd.
#   IG_antisym isolates the westward n=2 IG branch.
#   Easterly is the direct 10 S--10 N mean used by Rios-Berrios et al. (2023),
#   without assigning the rectangular TD/easterly-wave box a Matsuno parity.
wave_latitude_projection = dict(WAVE_LATITUDE_PROJECTION)

# Variables used in the wave diagnostics.  Names are interpreted by
# get_filter_field(); cldfrac without a height is the vertical mean, cldfrac5km
# means the nearest cloud-fraction level to 5 km, and pressure-level fields are
# parsed from names like u850, w500, T100, v850, qv700, RH500. Here ``w``
# denotes upward-positive pressure velocity (-omega), not geometric velocity.
power_fields = list(WAVE_VARIABILITY_FIELDS)

lag_target_fields = ["Precip", "OLR", "PW", "cldfrac"]
conditioned_fields = ["Precip", "OLR", "PW", "cldfrac"]
vertical_profile_waves = list(VERTICAL_STRUCTURE_FILTERS)
vertical_profile_fields = ["u", "w", "T", "cldfrac"]
vertical_profile_levels = [50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 950, 1000]
cldfrac_profile_heights_km = [2, 5, 8, 11, 14, 17, 20]
top_heaviness_pressure_hpa = np.asarray(
    [100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 950, 1000],
    dtype=float,
)
kelvin_reference_k_m1 = 1.57e-7
kelvin_reference_m_m1 = -6.28e-4
kelvin_sensitivity_wavenumbers = (1, 2, 3)
kelvin_sensitivity_vertical_wavelengths_km = (5.0, 10.0, 15.0)
kelvin_profile_height_m = np.arange(0.0, 25_000.1, 250.0)

# Compact Figure-14 caches retain only the LPO-following samples needed by the
# lifecycle diagnostic. Filtering is still performed here over the complete
# configured time/longitude record. The relative-longitude stencil, rather
# than a pre-averaged box, lets the lifecycle renderer compare point, 2.5,
# 3.75 and 5 degree half-widths without filtering or reopening model data.
LPO_SECTION_DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES = 5.0
LPO_SECTION_LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES = (0.0, 2.5, 3.75, 5.0)
LPO_SECTION_RELATIVE_LONGITUDE_DEGREES = np.arange(-7.5, 7.51, 0.5)
LPO_SECTION_RELATIVE_LATITUDE_DEGREES = np.arange(-10.0, 10.01, 0.5)
LPO_SECTION_PRESSURE_HPA = np.asarray(
    [50, 70, 100, 150, 200, 250, 300, 400, 500, 600, 700, 800, 850, 900, 950, 1000],
    dtype=float,
)
LPO_SECTION_VERTICAL_FIELDS = ("T", "w", "qv", "RH", "u", "N2")
LPO_SECTION_SURFACE_FIELDS = ("Precip", "PW", "OLR")

# Figure 8 keeps latitude resolved and applies each wave box directly.
eddy_flux_waves = list(EDDY_FLUX_FILTERS)
eddy_flux_levels = [70, 100, 250, 850]

spectrum_field = "Precip"
spectrum_additional_fields = (
    (
        "T100",
        "Temperature at 100 hPa",
        "temperature_100hpa_symmetric_antisymmetric_wave_spectra",
    ),
    (
        "u100",
        "Zonal wind at 100 hPa",
        "zonal_wind_100hpa_symmetric_antisymmetric_wave_spectra",
    ),
)
spectrum_lat_bounds = filter_lat_bounds
if test_mode:
    # The default test record is 40 days, while the publication spectrum uses
    # 48-day segments. Keep a useful multiweek spectrum in test mode without
    # asking the rolling-window code to construct a segment longer than the
    # actual-data subset.
    spectrum_segment_days = min(32, runtime_args.test_days - 1)
    spectrum_overlap_days = min(
        20, int(round(0.625 * spectrum_segment_days))
    )
else:
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
phase_speed_distribution_bin_width = 2.0
phase_speed_distribution_power_mode = "excess_normalised_power"
# Shared x-axis for figure 7's wind profiles and phase-speed distributions.
critical_level_phase_speed_xlim = (-40.0, 25.0)
spectrum_outline_colors = {
    "Kelvin": "tab:orange",
    "EIG": "tab:red",
    "IG_antisym": "tab:purple",
    "Easterly": "tab:green",
    "ER": "tab:blue",
}
earth_radius_m = 6.371e6

# Input/output cache for k-f filtered fields.
cache_dir = diagnostic_cache_dir(
    "wave_cache",
    *(("testing",) if test_mode else ()),
)
input_cache_dir = os.path.join(cache_dir, "filter_inputs")
derived_cache_dir = os.path.join(cache_dir, "derived")
product_output_dir = os.path.join(cache_dir, "products")
os.makedirs(cache_dir, exist_ok=True)
os.makedirs(input_cache_dir, exist_ok=True)
os.makedirs(derived_cache_dir, exist_ok=True)
os.makedirs(product_output_dir, exist_ok=True)
if test_mode:
    progress(f"Wave test cache root: {cache_dir}")
    progress(f"Wave test data products: {product_output_dir}")
    progress(f"Wave test figures: {save_loc}")

# Bump this when the pressure-level source fields or their coordinate handling
# changes. Existing pressure-level filter inputs/wave caches without this marker
# are regenerated automatically.
PRESSURE_FIELD_CACHE_VERSION = (
    "pressure-axis-v3-corrected-t-nudge-omega"
)
FILTER_INPUT_CACHE_VERSION = (
    "equatorial-wave-filter-input-v4-corrected-t-nudge-omega"
)
FILTERED_WAVE_CACHE_VERSION = FILTERED_WAVE_PRODUCT_VERSION
PRESSURE_FIELD_PREFIXES = ("w", "u", "v", "T", "qv", "RH")
DERIVED_CACHE_VERSION = "equatorial-wave-derived-v10-cosine-latitude"
SPECTRUM_CACHE_VERSION = "equatorial-wave-spectrum-v4-cosine-latitude"

# Avoid giant chunks when selecting latitude/longitude subsets.
dask.config.set({"array.slicing.split_large_chunks": True})

latlon_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "olrtoa",
    "cldfrac",
    "uzonal_isobaric", "umeridional_isobaric",
    "rh_isobaric", "t_isobaric", "qv_isobaric", "omg_isobaric",
    "theta", "pressure", "zgrid", "w", "rho", "qv", "qc", "qr",
    "qi", "qs", "qg", "relhum",
    "uReconstructZonal",
]
zonal_mean_vars = [
    "cldfrac", "uzonal_isobaric", "umeridional_isobaric", "rh_isobaric",
    "t_isobaric", "qv_isobaric", "theta", "pressure", "w",
    "uReconstructZonal",
]

# Keep full lat-lon fields chunked in both time and space.  The k-f filters
# force loading of tropical lat/lon arrays, but spatial chunks still help for
# preliminary selection and any non-filtered diagnostics.
latlon_chunks = {"Time": 8, "latitude": 60, "longitude": 120}
zm_chunks = {"Time": -1}


should_run_figure = partial(figure_should_run, figures_to_run)
should_run_any = partial(figure_should_run_any, figures_to_run)
def numbered_figure_path(number, stem):
    return os.path.join(figsave.save_loc, f"{number:02d}_{stem}.png")


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


save_figure = numbered_figure_saver(figsave)


def derived_cache_path(cache_name, suffix):
    return os.path.join(derived_cache_dir, f"{safe_name(cache_name)}.{suffix}")


read_cache_metadata = read_json_file
write_cache_metadata = write_json_file


def cache_signature(kind, extra=None):
    signature = dict(
        cache_version=DERIVED_CACHE_VERSION,
        kind=kind,
        root_dir=os.path.abspath(root_dir),
        data_locs=list(data_locs),
        sim_names=list(sim_names),
        latlon_pattern=LATLON_PATTERN,
        zonal_mean_pattern=ZONAL_MEAN_PATTERN,
        analysis_start_date=ANALYSIS_START,
        dt_hours=dt,
        obs_per_day=obs_per_day,
        analysis_end_date=ANALYSIS_END,
        filter_lat_bounds=jsonable_value(filter_lat_bounds),
        filter_avg_lat=filter_avg_lat,
        wave_names=list(wave_names),
        wave_boxes=jsonable_value(WAVE_BOXES) if "WAVE_BOXES" in globals() else None,
        wave_config_version=WAVE_CONFIG_VERSION,
        wave_latitude_projection=jsonable_value(wave_latitude_projection),
        pressure_field_cache_version=PRESSURE_FIELD_CACHE_VERSION,
        filter_input_cache_version=FILTER_INPUT_CACHE_VERSION,
        filtered_wave_cache_version=FILTERED_WAVE_CACHE_VERSION,
        classification_cache_version=CLASSIFICATION_CACHE_VERSION,
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
    """Load a current derived CSV from the single configured cache."""
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
        # Derived products are compact plotting/summary datasets. Load them
        # eagerly so cached and freshly computed products have identical array
        # semantics, then close the backing file immediately. Large filtered
        # wave caches are handled separately and remain lazy.
        with xr.open_dataset(nc_path) as cached:
            return cached.load()
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

classification_cache_dir = diagnostic_cache_dir(
    "classification_cache"
)


def load_wave_cloud_responses(simulation):
    """Load the current shared cloud product as float occurrence fields."""
    path = os.path.join(
        classification_cache_dir,
        f"{simulation.replace(' ', '_')}_classified_high_cloud.nc",
    )
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"{simulation}: missing shared high-cloud product {path}. Run "
            "refined_high_cloud_classification_diagnostics.py first."
        )
    classified = xr.open_dataset(path, chunks={})
    if not cached_classification_is_current(classified, HighCloudThresholds()):
        classified.close()
        raise RuntimeError(
            f"{simulation}: stale high-cloud product {path}; regenerate it before "
            "wave filtering."
        )
    required = (
        "high_cloud_mask",
        "convective_precipitating_high_cloud_mask",
        "stratiform_precipitating_high_cloud_mask",
        "nonprecipitating_high_cloud_mask",
        "cloud_top_pressure",
    )
    missing = [name for name in required if name not in classified]
    if missing:
        classified.close()
        raise RuntimeError(f"{simulation}: shared high-cloud product lacks {missing}")
    high = classified.high_cloud_mask.fillna(False).astype(bool)
    responses = xr.Dataset(
        {
            "high_cloud": high.astype("float32"),
            "active_high_cloud": classified[
                "convective_precipitating_high_cloud_mask"
            ].fillna(False).astype("float32"),
            "stratiform_high_cloud": classified[
                "stratiform_precipitating_high_cloud_mask"
            ].fillna(False).astype("float32"),
            "nonprecipitating_high_cloud": classified[
                "nonprecipitating_high_cloud_mask"
            ].fillna(False).astype("float32"),
            "ttl_reaching_high_cloud": (
                high
                & (
                    classified.cloud_top_pressure
                    <= TTL_BASE_PRESSURE_HPA
                )
            ).fillna(False).astype("float32"),
        }
    )
    responses.attrs.update(
        source_classification_path=os.path.abspath(path),
        classification_cache_version=CLASSIFICATION_CACHE_VERSION,
        ttl_cloud_top_pressure_threshold_hpa=TTL_BASE_PRESSURE_HPA,
    )
    return responses


dss = []
dss_zm = []
if runtime_args.figures_only:
    dss = [None] * len(data_locs)
    dss_zm = [None] * len(data_locs)
    progress("Figure-only mode: raw model histories remain closed.")
else:
    for i, d in enumerate(data_locs):
        ds = load_latlon_dataset(
            os.path.join(root_dir, d),
            latlon_pattern=LATLON_PATTERN,
            chunks=latlon_chunks,
            start=ANALYSIS_START,
            end=ANALYSIS_END,
        )
        ds = standardize_latlon_dataset(ds, latlon_vars)
        expected_time = precip_rate_from_accum(total_precip_accum(ds)).Time
        cloud_responses = load_wave_cloud_responses(sim_names[i]).sel(
            Time=expected_time
        )
        if not np.array_equal(
            cloud_responses.Time.values, expected_time.values
        ):
            raise RuntimeError(
                f"{sim_names[i]}: shared high-cloud classification Time "
                "coordinate does not exactly match the precipitation-rate record"
            )
        ds = ds.sel(Time=expected_time)
        ds = xr.merge([ds, cloud_responses], compat="override", join="exact")

        zds = load_latlon_dataset(
            os.path.join(root_dir, d),
            latlon_pattern=ZONAL_MEAN_PATTERN,
            chunks=zm_chunks,
            start=ANALYSIS_START,
            end=ANALYSIS_END,
        )
        zds = standardize_latlon_dataset(zds, zonal_mean_vars)

        progress(
            f"Loaded {sim_names[i]} test={test_mode}: "
            f"{ds.sizes['Time']} lat-lon times, "
            f"{zds.sizes['Time']} zonal-mean times"
        )
        dss.append(ds)
        dss_zm.append(zds)


# -----------------------------------------------------------------------------
# Wave-box metadata and helper functions
# -----------------------------------------------------------------------------

mis = MISSING_EQUIVALENT_DEPTH


wave_label = partial(configured_label, WAVE_BOXES)


def precip_rate(ds):
    pr = precip_rate_from_accum(total_precip_accum(ds))
    pr.name = "Precip"
    pr.attrs["units"] = "mm day-1"
    return pr


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

    cf = ds.cldfrac
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
        return ds.olrtoa.sel(Time=pr.Time)
    if field_name == "PW":
        return ds.precipw.sel(Time=pr.Time)
    if field_name == "cldfrac" or field_name.startswith("cldfrac"):
        return sel_common_times(get_cldfrac_field(ds, field_name), pr)
    if field_name in {
        "high_cloud",
        "active_high_cloud",
        "stratiform_high_cloud",
        "nonprecipitating_high_cloud",
        "ttl_reaching_high_cloud",
    }:
        if field_name not in ds:
            raise RuntimeError(
                f"{field_name} is unavailable; the current shared high-cloud "
                "classification product is required."
            )
        return sel_common_times(ds[field_name].astype(float), pr)

    # Pressure-level fields encoded as u850, v250, w500, or T100. The ``w``
    # cache family is retained as a stable diagnostic name, but its defined
    # quantity is upward-positive pressure velocity (-omega) in Pa s-1.
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
                out = upward_pressure_velocity(
                    ds,
                    plev,
                    times=pr.Time,
                )
                out.name = field_name
                return out
            out = ds[var].sel(pres=plev).sel(Time=pr.Time)
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
    if (
        field_name == "cldfrac"
        or field_name.startswith("cldfrac")
        or field_name.endswith("_high_cloud")
        or field_name == "high_cloud"
    ):
        return "fraction"
    if is_pressure_level_field(field_name) and field_name.startswith("w"):
        return "Pa s$^{-1}$"
    if field_name.startswith("T"):
        return "K"
    if field_name.startswith("qv"):
        return "kg kg$^{-1}$"
    if field_name.startswith("RH"):
        return "%"
    return "m s$^{-1}$"


def field_display_label(field_name):
    """Return compact labels that preserve the defined physical quantity."""

    if field_name.startswith("w") and is_pressure_level_field(field_name):
        return rf"$-\omega_{{{field_name[1:]}}}$"
    if field_name == "w":
        return r"$-\omega$"
    return field_name


def to_filter_input_dataset(da, obs_var="x"):
    """Standardise MPAS lat/lon DataArray to the file expected by kf_filter."""
    x = da
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
    try:
        with xr.open_dataset(fname) as cached:
            return (
                cached.attrs.get("cache_version") == FILTER_INPUT_CACHE_VERSION
                and cached.attrs.get("source_field") == field_name
                and cached.attrs.get("analysis_start") == ANALYSIS_START
                and cached.attrs.get("analysis_end") == ANALYSIS_END
                and int(cached.attrs.get("test_subset", -1))
                == int(test_mode)
                and (
                    not is_pressure_level_field(field_name)
                    or cached.attrs.get("pressure_field_cache_version")
                    == PRESSURE_FIELD_CACHE_VERSION
                )
                and (
                    field_name
                    not in {
                        "high_cloud",
                        "active_high_cloud",
                        "stratiform_high_cloud",
                        "nonprecipitating_high_cloud",
                        "ttl_reaching_high_cloud",
                    }
                    or cached.attrs.get("classification_cache_version")
                    == CLASSIFICATION_CACHE_VERSION
                )
                and (
                    field_name != "ttl_reaching_high_cloud"
                    or float(
                        cached.attrs.get(
                            "ttl_cloud_top_pressure_threshold_hpa",
                            np.nan,
                        )
                    )
                    == TTL_BASE_PRESSURE_HPA
                )
            )
    except Exception:
        return False


def write_filter_input_if_needed(ds, sim_index, field_name):
    """Cache a standard NetCDF input for kf_filter."""
    fname = os.path.join(input_cache_dir, f"{safe_name(sim_names[sim_index])}_{safe_name(field_name)}.nc")
    force_this_path = (
        force_recompute_caches
        and fname not in _rebuilt_filter_input_paths
    )
    if not force_this_path and filter_input_cache_matches(fname, field_name):
        return fname
    if os.path.exists(fname):
        print(f"Regenerating stale filter input: {fname}")

    print(f"Writing filter input: {fname}")
    da = get_filter_field(ds, field_name)
    fds = to_filter_input_dataset(da, obs_var="x")
    fds.attrs.update(
        source_field=field_name,
        source_simulation=sim_names[sim_index],
        cache_version=FILTER_INPUT_CACHE_VERSION,
        wave_config_version=WAVE_CONFIG_VERSION,
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        test_subset=int(test_mode),
    )
    if is_pressure_level_field(field_name):
        fds.attrs["pressure_field_cache_version"] = PRESSURE_FIELD_CACHE_VERSION
    if field_name in {
        "high_cloud",
        "active_high_cloud",
        "stratiform_high_cloud",
        "nonprecipitating_high_cloud",
        "ttl_reaching_high_cloud",
    }:
        fds.attrs["classification_cache_version"] = CLASSIFICATION_CACHE_VERSION
    if field_name == "ttl_reaching_high_cloud":
        fds.attrs["ttl_cloud_top_pressure_threshold_hpa"] = (
            TTL_BASE_PRESSURE_HPA
        )
    encoding = {"x": {"dtype": "float32"}}
    tmp_fname = fname + ".tmp"
    fds.to_netcdf(tmp_fname, encoding=encoding)
    os.replace(tmp_fname, fname)
    _rebuilt_filter_input_paths.add(fname)
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


def cosine_latitude_mean(
    field,
    dimensions="latitude",
    *,
    lat_dim="latitude",
):
    """Return a cosine-weighted mean over one latitude-bearing reduction."""
    dimensions = (
        [dimensions] if isinstance(dimensions, str) else list(dimensions)
    )
    if lat_dim not in dimensions:
        raise ValueError(f"{lat_dim!r} must be one of {dimensions!r}")
    weights = np.cos(np.deg2rad(field[lat_dim])).clip(min=0.0)
    return field.weighted(weights).mean(dim=dimensions, skipna=True)


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
    valid = {"symmetric", "antisymmetric", "direct"}
    if projection is None:
        projection = wave_latitude_projection[wave_name]
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
    filter_name = WAVE_BOXES[wave_name]["filter_name"]
    if wave_da.attrs.get("cache_version") != FILTERED_WAVE_CACHE_VERSION:
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
        and wave_da.attrs.get("wave_config_version") == WAVE_CONFIG_VERSION
        and wave_da.attrs.get("analysis_start") == ANALYSIS_START
        and wave_da.attrs.get("analysis_end") == ANALYSIS_END
        and int(wave_da.attrs.get("test_subset", -1)) == int(test_mode)
        and (
            not is_pressure_level_field(field_name)
            or wave_da.attrs.get("pressure_field_cache_version")
            == PRESSURE_FIELD_CACHE_VERSION
        )
        and (
            field_name
            not in {
                "high_cloud",
                "active_high_cloud",
                "stratiform_high_cloud",
                "nonprecipitating_high_cloud",
                "ttl_reaching_high_cloud",
            }
            or wave_da.attrs.get("classification_cache_version")
            == CLASSIFICATION_CACHE_VERSION
        )
        and (
            field_name != "ttl_reaching_high_cloud"
            or float(
                wave_da.attrs.get(
                    "ttl_cloud_top_pressure_threshold_hpa",
                    np.nan,
                )
            )
            == TTL_BASE_PRESSURE_HPA
        )
    )


def read_cached_wave(cache_path, field_name, wave_name, avgLat, lat_bounds, projection, force=False):
    force_this_path = (
        force
        or (
            force_recompute_caches
            and cache_path not in _rebuilt_wave_cache_paths
        )
    )
    if force_this_path or not os.path.exists(cache_path):
        return None
    cached = xr.open_dataset(cache_path)
    wave_cached = cached["wave"]
    if wave_cache_matches(wave_cached, field_name, wave_name, avgLat, lat_bounds, projection):
        print(f"Using cached filtered wave {cache_path}")
        return wave_cached
    print(f"Regenerating stale cached wave file {cache_path}")
    cached.close()
    return None


def write_wave_cache(wave, cache_path):
    tmp_cache_path = cache_path + ".tmp"
    netcdf_safe_dataset(wave.to_dataset()).to_netcdf(tmp_cache_path)
    os.replace(tmp_cache_path, cache_path)
    _rebuilt_wave_cache_paths.add(cache_path)


def filter_input_with_wave_box(input_path, obs_var, wave_name, lat_bounds, avgLat):
    """Run the low-level Wheeler-Kiladis k-f filter for a configured wave box."""
    box = WAVE_BOXES[wave_name]
    filter_name = box["filter_name"]
    k0, k1 = box["k"]
    p0, p1 = box["period"]
    h0, h1 = box["depth"]

    with xr.open_dataset(input_path) as infile:
        obs_time = infile.Time.load()
        if (
            wave_name == EASTWARD_INTRASEASONAL_FILTER
            and resolved_frequency_count(
                len(obs_time), dt, WAVE_BOXES[wave_name]["period"]
            )
            < 1
        ):
            raise ValueError(
                "Record is too short to resolve any Fourier frequency in the "
                "20-100-day eastward-intraseasonal band"
            )
        lon = infile.lon.load()
        obs_data = infile[obs_var].sel(lat=slice(*lat_bounds))

        if avgLat:
            obs_data = cosine_latitude_mean(
                obs_data, "lat", lat_dim="lat"
            )
            filt_data = np.asarray(obs_data.values[:, None, :], dtype="f")
            coords = [obs_time, lon]
            dims = ["time", "lon"]
        else:
            lat = obs_data.lat.load()
            filt_data = np.asarray(obs_data.values, dtype="f")
            coords = [obs_time, lat, lon]
            dims = ["time", "lat", "lon"]

    # kf_filter detrends and applies a Tukey taper to the complete record
    # before its FFT. Do not append zero-valued samples: that introduces a
    # large artificial step at the data boundary and contaminates low-
    # frequency bands, especially the 20--100-day intraseasonal filter.
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
        filter_name=WAVE_BOXES[wave_name]["filter_name"],
        source_simulation=sim_names[sim_index],
        filter_lat_min=lat_bounds[0],
        filter_lat_max=lat_bounds[1],
        filter_lat_bounds=lat_bounds_attr(lat_bounds),
        latitude_projection=projection,
        avgLat=int(avgLat),
        units=field_units(field_name),
        wave_box=json.dumps(jsonable_value(WAVE_BOXES[wave_name]), sort_keys=True),
        wave_config_version=WAVE_CONFIG_VERSION,
        cache_version=FILTERED_WAVE_CACHE_VERSION,
        resolved_fourier_frequency_count=resolved_frequency_count(
            wave.sizes["Time"], dt, WAVE_BOXES[wave_name]["period"]
        ),
        finite_record_caution=(
            "few independent 20-100-day periods; temporal edges require caution"
            if wave_name == EASTWARD_INTRASEASONAL_FILTER
            else ""
        ),
        temporal_padding="none; complete record detrended and Tukey tapered by kf_filter",
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        test_subset=int(test_mode),
    )
    if is_pressure_level_field(field_name):
        wave.attrs["pressure_field_cache_version"] = PRESSURE_FIELD_CACHE_VERSION
    if field_name in {
        "high_cloud",
        "active_high_cloud",
        "stratiform_high_cloud",
        "nonprecipitating_high_cloud",
        "ttl_reaching_high_cloud",
    }:
        wave.attrs["classification_cache_version"] = CLASSIFICATION_CACHE_VERSION
    if field_name == "ttl_reaching_high_cloud":
        wave.attrs["ttl_cloud_top_pressure_threshold_hpa"] = (
            TTL_BASE_PRESSURE_HPA
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
    north, south = xr.align(north, south, join="exact")
    if projection != "antisymmetric":
        raise ValueError(f"Projection {projection!r} does not use paired hemispheres")
    wave = 0.5 * (north - south)
    combination = "0.5*(north-south)"

    wave = wave.load()
    wave.name = "wave"
    wave.attrs.update(
        field=field_name,
        wave_name=wave_name,
        filter_name=WAVE_BOXES[wave_name]["filter_name"],
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
        wave_config_version=WAVE_CONFIG_VERSION,
        cache_version=FILTERED_WAVE_CACHE_VERSION,
        resolved_fourier_frequency_count=resolved_frequency_count(
            wave.sizes["Time"], dt, WAVE_BOXES[wave_name]["period"]
        ),
        finite_record_caution=(
            "few independent 20-100-day periods; temporal edges require caution"
            if wave_name == EASTWARD_INTRASEASONAL_FILTER
            else ""
        ),
        analysis_start=ANALYSIS_START,
        analysis_end=ANALYSIS_END,
        test_subset=int(test_mode),
    )
    if is_pressure_level_field(field_name):
        wave.attrs["pressure_field_cache_version"] = PRESSURE_FIELD_CACHE_VERSION
    if field_name in {
        "high_cloud",
        "active_high_cloud",
        "stratiform_high_cloud",
        "nonprecipitating_high_cloud",
        "ttl_reaching_high_cloud",
    }:
        wave.attrs["classification_cache_version"] = CLASSIFICATION_CACHE_VERSION
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
    if projection == "antisymmetric":
        return projected_filtered_wave(ds, sim_index, field_name, wave_name, lat_bounds, projection, force=force)
    if projection == "direct":
        return direct_filtered_wave(ds, sim_index, field_name, wave_name, True, lat_bounds, projection, force=force)
    raise ValueError(f"Unhandled latitude projection {projection!r}")


def _lpo_wave_tracking_steps(
    *, test=False, allow_stale_products=False
):
    """Read only the stable tracking tables needed for compact wave caches."""

    mode_directory = (
        require_actual_test_products(
            start=runtime_args.test_start,
            days=DEFAULT_LPO_TEST_DAYS,
            minimum_duration_days=DEFAULT_TEST_MINIMUM_DURATION_DAYS,
        )
        if test
        else require_publication_products(
            allow_stale_product_version=allow_stale_products
        )
    )
    tables = {}
    for simulation in LPO_SIMULATION_DIRS:
        standard = mode_directory / "primary" / simulation / "standard"
        timing = pd.read_csv(
            standard / "lpt_event_relative_diagnostics.csv"
        )
        timing["Time"] = pd.to_datetime(timing.Time)
        selected = timing[
            timing.primary_tropical_sample.astype(bool)
            & timing.reference_in_record_subset.astype(bool)
            & timing.in_primary_lifecycle_window.astype(bool)
        ].copy()
        selected = selected.sort_values(
            ["event_id", "lag_hours", "Time"]
        ).drop_duplicates(["event_id", "lag_hours"], keep="last")
        geometric = pd.read_csv(
            standard / "lpt_track_steps.csv",
            usecols=[
                "event_id", "Time", "centroid_lon", "centroid_lat",
            ],
        )
        geometric["Time"] = pd.to_datetime(geometric.Time)
        selected = selected.merge(
            geometric,
            on=["event_id", "Time"],
            how="left",
            validate="one_to_one",
        )
        required = {
            "event_id",
            "Time",
            "lag_hours",
            "precip_weighted_centroid_lon",
            "precip_weighted_centroid_lat",
            "centroid_lon",
            "centroid_lat",
        }
        missing = sorted(required - set(selected.columns))
        if missing:
            raise RuntimeError(
                f"{standard}: tracking timing table lacks {missing}"
            )
        geometric_columns = ["centroid_lon", "centroid_lat"]
        if selected[geometric_columns].isna().any().any():
            count = int(selected[geometric_columns].isna().any(axis=1).sum())
            raise RuntimeError(
                f"{standard}: {count} selected lifecycle rows lack geometric "
                "centroids"
            )
        tables[simulation] = selected.reset_index(drop=True)
    return tables


def _filter_2d_wave(values, wave_name):
    """Filter one complete time-longitude field with a configured wave box."""

    box = WAVE_BOXES[wave_name]
    filtered = kf_filter(
        np.asarray(values, dtype="f"),
        obs_per_day,
        box["period"][0],
        box["period"][1],
        box["k"][0],
        box["k"][1],
        box["depth"][0],
        box["depth"][1],
        box["filter_name"],
    )
    return np.asarray(filtered[: np.shape(values)[0]], dtype=float)


_BATCH_FILTER_MASKS = {}


def _filter_2d_wave_batch(values, wave_name):
    """Apply the configured filter to a batch of time-longitude fields."""

    values = np.asarray(values, dtype="f")
    if values.ndim != 3:
        raise ValueError("Batch wave filters require level, time, longitude")
    batch_count, time_count, longitude_count = values.shape
    key = (time_count, longitude_count, wave_name)
    mask = _BATCH_FILTER_MASKS.get(key)
    if mask is None:
        rng = np.random.default_rng(20260804)
        reference = rng.standard_normal(
            (time_count, longitude_count), dtype=np.float32
        )
        box = WAVE_BOXES[wave_name]
        _filtered, mask = kf_filter(
            reference,
            obs_per_day,
            box["period"][0],
            box["period"][1],
            box["k"][0],
            box["k"][1],
            box["depth"][0],
            box["depth"][1],
            box["filter_name"],
            debug=True,
        )
        mask = np.asarray(mask, dtype=bool)
        _BATCH_FILTER_MASKS[key] = mask
    original = values[:, :, ::-1].transpose(0, 2, 1)
    detrended = signal.detrend(original, axis=-1, type="linear")
    taper = signal.windows.tukey(time_count, 0.05, True)
    spectrum = fft.rfft2(
        detrended * taper[None, None, :],
        axes=(-2, -1),
        workers=4,
    )
    spectrum *= mask[None, :, :]
    reconstructed = fft.irfft2(
        spectrum,
        s=(longitude_count, time_count),
        axes=(-2, -1),
        workers=4,
    )
    output = reconstructed.transpose(0, 2, 1)[:, :, ::-1]
    return np.asarray(output[:batch_count], dtype=float)


def _tracking_time_indices(source_time, steps):
    source_index = pd.DatetimeIndex(pd.to_datetime(source_time))
    requested = pd.DatetimeIndex(pd.to_datetime(steps.Time))
    indices = source_index.get_indexer(requested)
    if np.any(indices < 0):
        missing = requested[indices < 0]
        raise RuntimeError(
            "Wave-filter input does not contain exact saved LPO times: "
            + ", ".join(str(value) for value in missing[:4])
        )
    return indices


def _sample_filtered_relative_longitudes(
    filtered,
    longitude,
    time_indices,
    centroid_longitude,
):
    """Sample a periodic field on the configured LPO-relative stencil."""

    longitude = np.asarray(longitude, dtype=float)
    centroids = np.asarray(centroid_longitude, dtype=float)
    relative = LPO_SECTION_RELATIVE_LONGITUDE_DEGREES
    output = np.full((len(time_indices), len(relative)), np.nan)
    for index, (time_index, centroid) in enumerate(
        zip(time_indices, centroids)
    ):
        targets = np.mod(centroid + relative, 360.0)
        longitude_indices = np.asarray(
            [
                np.argmin(
                    np.abs(((longitude - target + 180.0) % 360.0) - 180.0)
                )
                for target in targets
            ],
            dtype=int,
        )
        output[index] = np.asarray(
            filtered[int(time_index), longitude_indices], dtype=float
        )
    return output


def _wave_phase_diagnostics(filtered, time_indices, longitude, centroids):
    """Return Rios-Berrios-style eight-phase metadata at LPO centroids.

    The construction uses filtered precipitation and its time derivative.
    Phase 5 is the positive precipitation maximum, phases 4 and 6 are its
    growing and decaying flanks, and amplitude is the radius in the jointly
    standardised phase plane.
    """

    result = filtered_wave_phase_diagnostics(
        filtered,
        time_indices,
        longitude,
        centroids,
        observations_per_day=obs_per_day,
    )
    return (
        result["phase_angle_degrees"],
        result["phase_number"],
        result["normalized_amplitude"],
        result["strong_amplitude"],
    )


def _latitude_resolved_lpo_wave_values(
    field,
    steps,
    wave_names,
):
    """Filter one surface field by latitude and retain moving LPO sections."""

    field = field.transpose("Time", "latitude", "longitude")
    latitude = np.asarray(field.latitude.values, dtype=float)
    relative_latitude = LPO_SECTION_RELATIVE_LATITUDE_DEGREES
    targets = (
        np.asarray(steps.precip_weighted_centroid_lat, dtype=float)[:, None]
        + relative_latitude[None, :]
    )
    latitude_indices = np.abs(
        latitude[None, None, :] - targets[:, :, None]
    ).argmin(axis=2)
    nearest = latitude[latitude_indices]
    if np.nanmax(np.abs(nearest - targets)) > 0.251:
        raise RuntimeError(
            "LPO relative-latitude section extends beyond the fixed "
            "half-degree wave-filter grid"
        )
    time_indices = _tracking_time_indices(field.Time.values, steps)
    output = np.full(
        (
            len(steps),
            len(relative_latitude),
            len(LPO_SECTION_RELATIVE_LONGITUDE_DEGREES),
            len(wave_names),
        ),
        np.nan,
    )
    unique_latitude_indices = np.unique(latitude_indices)
    sources = np.asarray(
        field.isel(latitude=unique_latitude_indices).load(), dtype="f"
    ).transpose(1, 0, 2)
    for wave_index, wave_name in enumerate(wave_names):
        filtered_batch = _filter_2d_wave_batch(sources, wave_name)
        for batch_index, latitude_index in enumerate(
            unique_latitude_indices
        ):
            positions = np.argwhere(latitude_indices == latitude_index)
            filtered = filtered_batch[batch_index]
            step_indices = np.unique(positions[:, 0])
            sampled = _sample_filtered_relative_longitudes(
                filtered,
                field.longitude.values,
                time_indices[step_indices],
                np.asarray(
                    steps.precip_weighted_centroid_lon,
                    dtype=float,
                )[step_indices],
            )
            sample_lookup = dict(zip(step_indices, sampled))
            for step_index, relative_index in positions:
                output[step_index, relative_index, :, wave_index] = (
                    sample_lookup[step_index]
                )
    return output


def _latitude_mean_lpo_wave_values(field, steps, wave_names):
    """Filter means within 10 degrees of each saved LPO latitude."""

    field = field.transpose("Time", "latitude", "longitude")
    centroid_latitude = np.asarray(
        steps.precip_weighted_centroid_lat, dtype=float
    )
    latitude_lower = float(np.nanmin(centroid_latitude) - 10.5)
    latitude_upper = float(np.nanmax(centroid_latitude) + 10.5)
    field = field.sel(
        latitude=slice(latitude_lower, latitude_upper)
    ).load()
    latitude = np.asarray(field.latitude, dtype=float)
    centre_indices = np.abs(
        latitude[None, :]
        - centroid_latitude[:, None]
    ).argmin(axis=1)
    time_indices = _tracking_time_indices(field.Time.values, steps)
    output = np.full(
        (
            len(steps),
            len(LPO_SECTION_RELATIVE_LONGITUDE_DEGREES),
            len(wave_names),
        ),
        np.nan,
    )
    rms = np.full(len(wave_names), np.nan)
    rms_collect = [[] for _wave_name in wave_names]
    unique_centre_indices = np.unique(centre_indices)
    sources = np.stack(
        [
            np.asarray(
                cosine_latitude_mean(
                    field.where(
                        np.abs(field.latitude - latitude[centre_index])
                        <= 10.0,
                        drop=True,
                    )
                ),
                dtype="f",
            )
            for centre_index in unique_centre_indices
        ]
    )
    for wave_index, wave_name in enumerate(wave_names):
        filtered_batch = _filter_2d_wave_batch(sources, wave_name)
        for batch_index, centre_index in enumerate(unique_centre_indices):
            positions = np.flatnonzero(centre_indices == centre_index)
            filtered = filtered_batch[batch_index]
            output[positions, :, wave_index] = (
                _sample_filtered_relative_longitudes(
                    filtered,
                    field.longitude.values,
                    time_indices[positions],
                    np.asarray(
                        steps.precip_weighted_centroid_lon, dtype=float
                    )[positions],
                )
            )
            finite = np.isfinite(filtered)
            if finite.any():
                rms_collect[wave_index].append(
                    float(np.sqrt(np.mean(filtered[finite] ** 2)))
                )
    for wave_index, values in enumerate(rms_collect):
        if values:
            rms[wave_index] = float(np.mean(values))
    return output, rms


def _latitude_mean_lpo_values(field, steps):
    """Sample unfiltered means within 10 degrees of each LPO latitude."""

    field = field.transpose("Time", "latitude", "longitude")
    centroid_latitude = np.asarray(
        steps.precip_weighted_centroid_lat, dtype=float
    )
    latitude_lower = float(np.nanmin(centroid_latitude) - 10.5)
    latitude_upper = float(np.nanmax(centroid_latitude) + 10.5)
    field = field.sel(
        latitude=slice(latitude_lower, latitude_upper)
    ).load()
    latitude = np.asarray(field.latitude, dtype=float)
    centre_indices = np.abs(
        latitude[None, :]
        - centroid_latitude[:, None]
    ).argmin(axis=1)
    time_indices = _tracking_time_indices(field.Time.values, steps)
    output = np.full(
        (
            len(steps),
            len(LPO_SECTION_RELATIVE_LONGITUDE_DEGREES),
        ),
        np.nan,
    )
    for centre_index in np.unique(centre_indices):
        positions = np.flatnonzero(centre_indices == centre_index)
        centre = latitude[centre_index]
        source = np.asarray(
            cosine_latitude_mean(
                field.where(
                    np.abs(field.latitude - centre) <= 10.0, drop=True
                )
            ),
            dtype=float,
        )
        output[positions] = _sample_filtered_relative_longitudes(
            source,
            field.longitude.values,
            time_indices[positions],
            np.asarray(
                steps.precip_weighted_centroid_lon, dtype=float
            )[positions],
        )
    return output


def _native_sampling_geometry(
    field,
    steps,
    *,
    relative_longitude=LPO_SECTION_RELATIVE_LONGITUDE_DEGREES,
    latitude_column="precip_weighted_centroid_lat",
    longitude_column="precip_weighted_centroid_lon",
    additional_latitude_bounds=None,
    time_independent=False,
):
    """Return bounded coordinates and index arrays for native profiles."""

    centroid_latitude = np.asarray(
        steps[latitude_column], dtype=float
    )
    latitude_lower = float(np.nanmin(centroid_latitude) - 10.5)
    latitude_upper = float(np.nanmax(centroid_latitude) + 10.5)
    if additional_latitude_bounds is not None:
        extra_lower, extra_upper = normalise_lat_bounds(
            additional_latitude_bounds
        )
        latitude_lower = min(latitude_lower, extra_lower)
        latitude_upper = max(latitude_upper, extra_upper)
    bounded = field.sel(latitude=slice(latitude_lower, latitude_upper))
    latitude = np.asarray(bounded.latitude, dtype=float)
    centre_indices = np.abs(
        latitude[None, :] - centroid_latitude[:, None]
    ).argmin(axis=1)
    time_indices = (
        np.zeros(len(steps), dtype=int)
        if time_independent
        else _tracking_time_indices(bounded.Time.values, steps)
    )
    longitude = np.asarray(bounded.longitude, dtype=float)
    centroid_longitude = np.asarray(
        steps[longitude_column], dtype=float
    )
    relative_longitude = np.asarray(relative_longitude, dtype=float)
    targets = np.mod(
        centroid_longitude[:, None]
        + relative_longitude[None, :],
        360.0,
    )
    longitude_indices = np.abs(
        ((longitude[None, None, :] - targets[:, :, None] + 180.0) % 360.0)
        - 180.0
    ).argmin(axis=2)
    return (
        bounded,
        latitude,
        centre_indices,
        time_indices,
        longitude_indices,
    )


def _latitude_mean_native_lpo_wave_values(
    field,
    steps,
    wave_names,
    *,
    block_size=25,
    progress_label="native field",
    relative_longitude=LPO_SECTION_RELATIVE_LONGITUDE_DEGREES,
    latitude_column="precip_weighted_centroid_lat",
    longitude_column="precip_weighted_centroid_lon",
):
    """Filter native levels in vectorized blocks around LPO latitudes."""

    native_dimension = vertical_dim(field)
    field = field.transpose(
        "Time", "latitude", "longitude", native_dimension
    )
    (
        bounded,
        latitude,
        centre_indices,
        time_indices,
        longitude_indices,
    ) = _native_sampling_geometry(
        field,
        steps,
        relative_longitude=relative_longitude,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        additional_latitude_bounds=(
            filter_lat_bounds
            if any(
                WAVE_LATITUDE_PROJECTION[name] == "antisymmetric"
                for name in wave_names
            )
            else None
        ),
    )
    relative_longitude = np.asarray(relative_longitude, dtype=float)
    level_count = bounded.sizes[native_dimension]
    output = np.full(
        (
            len(steps),
            level_count,
            len(relative_longitude),
            len(wave_names),
        ),
        np.nan,
    )
    for start in range(0, level_count, block_size):
        stop = min(start + block_size, level_count)
        progress(
            f"{progress_label}: native levels {start + 1}-{stop}/"
            f"{level_count}"
        )
        slab = bounded.isel(
            {native_dimension: slice(start, stop)}
        ).load()
        for wave_index, wave_name in enumerate(wave_names):
            if WAVE_LATITUDE_PROJECTION[wave_name] == "antisymmetric":
                source = np.asarray(
                    project_latitude_field(
                        slab, "antisymmetric", filter_lat_bounds
                    ),
                    dtype="f",
                ).transpose(2, 0, 1)
                filtered = _filter_2d_wave_batch(source, wave_name)
                sampled = filtered[
                    :, time_indices[:, None], longitude_indices
                ].transpose(1, 0, 2)
                output[:, start:stop, :, wave_index] = sampled
                continue
            for centre_index in np.unique(centre_indices):
                positions = np.flatnonzero(centre_indices == centre_index)
                centre = latitude[centre_index]
                source = np.asarray(
                    cosine_latitude_mean(
                        slab.where(
                            np.abs(slab.latitude - centre) <= 10.0,
                            drop=True,
                        )
                    ),
                    dtype="f",
                ).transpose(2, 0, 1)
                filtered = _filter_2d_wave_batch(source, wave_name)
                sampled = filtered[
                    :, time_indices[positions, None],
                    longitude_indices[positions]
                ].transpose(1, 0, 2)
                output[positions, start:stop, :, wave_index] = sampled
    return output


def _latitude_mean_native_lpo_values(
    field,
    steps,
    *,
    block_size=25,
    relative_longitude=LPO_SECTION_RELATIVE_LONGITUDE_DEGREES,
    latitude_column="precip_weighted_centroid_lat",
    longitude_column="precip_weighted_centroid_lon",
):
    """Sample unfiltered native levels in bounded vectorized blocks."""

    native_dimension = vertical_dim(field)
    time_independent = "Time" not in field.dims
    if time_independent:
        field = field.expand_dims(Time=np.asarray([0]))
    field = field.transpose(
        "Time", "latitude", "longitude", native_dimension
    )
    (
        bounded,
        latitude,
        centre_indices,
        time_indices,
        longitude_indices,
    ) = _native_sampling_geometry(
        field,
        steps,
        relative_longitude=relative_longitude,
        latitude_column=latitude_column,
        longitude_column=longitude_column,
        time_independent=time_independent,
    )
    relative_longitude = np.asarray(relative_longitude, dtype=float)
    level_count = bounded.sizes[native_dimension]
    output = np.full(
        (
            len(steps),
            level_count,
            len(relative_longitude),
        ),
        np.nan,
    )
    for start in range(0, level_count, block_size):
        stop = min(start + block_size, level_count)
        slab = bounded.isel(
            {native_dimension: slice(start, stop)}
        ).load()
        for centre_index in np.unique(centre_indices):
            positions = np.flatnonzero(centre_indices == centre_index)
            centre = latitude[centre_index]
            source = np.asarray(
                cosine_latitude_mean(
                    slab.where(
                        np.abs(slab.latitude - centre) <= 10.0,
                        drop=True,
                    )
                ),
                dtype=float,
            ).transpose(2, 0, 1)
            sampled = source[
                :, time_indices[positions, None],
                longitude_indices[positions]
            ].transpose(1, 0, 2)
            output[positions, start:stop, :] = sampled
    return output


def _native_lpo_vertical_fields(ds):
    """Return native-level fields required by filtered LPO composites."""

    required = {
        "theta", "pressure", "rho", "w", "qv", "qc", "qr", "qi",
        "qs", "qg", "relhum",
        "uReconstructZonal",
    }
    missing = sorted(required - set(ds.data_vars))
    if missing:
        raise RuntimeError(
            f"Native filtered LPO profiles require fields {missing}"
        )
    time = precip_rate(ds).Time
    pressure_pa = pressure_to_pa(ds.pressure).sel(Time=time)
    theta = ds.theta.sel(Time=time)
    height = native_midlevel_height(ds, theta)
    native_dimension = vertical_dim(theta)
    static_stability = xr.apply_ufunc(
        buoyancy_frequency_squared,
        theta,
        height,
        kwargs={"mask_nonpositive": False},
        input_core_dims=[[native_dimension], [native_dimension]],
        output_core_dims=[[native_dimension]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )
    temperature = theta * (pressure_pa / 100_000.0) ** (287.0 / 1004.0)
    specific_humidity = ds.qv.sel(Time=time) / (
        1.0 + ds.qv.sel(Time=time)
    )
    return {
        "T": temperature,
        "w": (
            9.80665
            * mpas_moist_air_density(ds).sel(Time=time)
            * align_vertical_velocity_to_reference(
                ds.w.sel(Time=time), ds.rho.sel(Time=time)
            )
        ),
        "qv": 1000.0 * specific_humidity,
        "RH": ds.relhum.sel(Time=time),
        "u": ds.uReconstructZonal.sel(Time=time),
        "N2": static_stability,
        "pressure_hpa": pressure_pa / 100.0,
        "height_m": height,
    }


def _peak_filtered_temperature_ray_sections(native_fields, steps):
    """Return geometric lag-zero Kelvin and MRG temperature sections."""

    peak = steps[np.isclose(steps.lag_hours, 0.0)].copy()
    peak = peak.sort_values(["event_id", "Time"]).drop_duplicates(
        "event_id", keep="last"
    ).reset_index(drop=True)
    wave_names = ("Kelvin", "MRG")
    filtered_native = _latitude_mean_native_lpo_wave_values(
        native_fields["T"],
        peak,
        wave_names,
        progress_label="geometric peak filtered temperature",
        relative_longitude=RAY_RELATIVE_LONGITUDE_DEGREES,
        latitude_column="centroid_lat",
        longitude_column="centroid_lon",
    )
    native_height = _latitude_mean_native_lpo_values(
        native_fields["height_m"],
        peak,
        relative_longitude=RAY_RELATIVE_LONGITUDE_DEGREES,
        latitude_column="centroid_lat",
        longitude_column="centroid_lon",
    )
    output = np.full(
        (
            len(peak), len(RAY_HEIGHT_M),
            len(RAY_RELATIVE_LONGITUDE_DEGREES), len(wave_names),
        ),
        np.nan,
        dtype=np.float32,
    )
    for event_index in range(len(peak)):
        for longitude_index in range(len(RAY_RELATIVE_LONGITUDE_DEGREES)):
            height = np.asarray(
                native_height[event_index, :, longitude_index], dtype=float
            )
            for wave_index in range(len(wave_names)):
                temperature = np.asarray(
                    filtered_native[
                        event_index, :, longitude_index, wave_index
                    ],
                    dtype=float,
                )
                valid = np.isfinite(height) & np.isfinite(temperature)
                if valid.sum() < 2:
                    continue
                order = np.argsort(height[valid])
                z = height[valid][order]
                value = temperature[valid][order]
                unique = np.r_[True, np.diff(z) > 0.0]
                z, value = z[unique], value[unique]
                if len(z) < 2:
                    continue
                inside = (RAY_HEIGHT_M >= z[0]) & (RAY_HEIGHT_M <= z[-1])
                output[
                    event_index, inside, longitude_index, wave_index
                ] = np.interp(RAY_HEIGHT_M[inside], z, value)
    return xr.Dataset(
        {
            "ray_filtered_temperature": (
                (
                    "peak_event", "ray_height_m",
                    "ray_relative_longitude", "ray_wave_band",
                ),
                output,
            ),
            "peak_event_id": (
                ("peak_event",), peak.event_id.astype(str).values
            ),
        },
        coords={
            "peak_event": np.arange(len(peak)),
            "ray_height_m": RAY_HEIGHT_M,
            "ray_relative_longitude": RAY_RELATIVE_LONGITUDE_DEGREES,
            "ray_wave_band": np.asarray(wave_names, dtype=str),
        },
        attrs={
            "ray_centroid": "geometric LPT centroid",
            "MRG_latitude_projection": "antisymmetric 10S-10N",
            "Kelvin_vertical_wavelength_km": 10.0,
            "MRG_zonal_wavenumber": -2,
            "MRG_vertical_wavelength_km": 10.0,
        },
    )


def _precipitation_phase_at_lpo(field, steps, wave_names):
    """Diagnose phase from each LPO-centred 20-degree latitude band."""

    field = field.transpose("Time", "latitude", "longitude")
    latitude = np.asarray(field.latitude, dtype=float)
    centre_indices = np.abs(
        latitude[None, :]
        - np.asarray(steps.precip_weighted_centroid_lat, dtype=float)[:, None]
    ).argmin(axis=1)
    time_indices = _tracking_time_indices(field.Time.values, steps)
    shape = (len(steps), len(wave_names))
    angle = np.full(shape, np.nan)
    phase = np.full(shape, np.nan)
    amplitude = np.full(shape, np.nan)
    strong = np.zeros(shape, dtype=bool)
    rms = np.full(len(wave_names), np.nan)
    rms_collect = [[] for _wave_name in wave_names]
    for wave_index, wave_name in enumerate(wave_names):
        if WAVE_LATITUDE_PROJECTION[wave_name] != "antisymmetric":
            continue
        source = np.asarray(
            project_latitude_field(
                field, "antisymmetric", filter_lat_bounds
            ),
            dtype="f",
        )
        filtered = _filter_2d_wave(source, wave_name)
        result = _wave_phase_diagnostics(
            filtered,
            time_indices,
            field.longitude.values,
            np.asarray(
                steps.precip_weighted_centroid_lon, dtype=float
            ),
        )
        angle[:, wave_index] = result[0]
        phase[:, wave_index] = result[1]
        amplitude[:, wave_index] = result[2]
        strong[:, wave_index] = result[3]
        finite = np.isfinite(filtered)
        if finite.any():
            rms_collect[wave_index].append(
                float(np.sqrt(np.mean(filtered[finite] ** 2)))
            )
    for centre_index in np.unique(centre_indices):
        positions = np.flatnonzero(centre_indices == centre_index)
        centre = latitude[centre_index]
        source = np.asarray(
            cosine_latitude_mean(
                field.where(
                    np.abs(field.latitude - centre) <= 10.0, drop=True
                )
            ),
            dtype="f",
        )
        for wave_index, wave_name in enumerate(wave_names):
            if WAVE_LATITUDE_PROJECTION[wave_name] == "antisymmetric":
                continue
            filtered = _filter_2d_wave(source, wave_name)
            result = _wave_phase_diagnostics(
                filtered,
                time_indices[positions],
                field.longitude.values,
                np.asarray(
                    steps.precip_weighted_centroid_lon, dtype=float
                )[positions],
            )
            angle[positions, wave_index] = result[0]
            phase[positions, wave_index] = result[1]
            amplitude[positions, wave_index] = result[2]
            strong[positions, wave_index] = result[3]
            finite = np.isfinite(filtered)
            if finite.any():
                rms_collect[wave_index].append(
                    float(np.sqrt(np.mean(filtered[finite] ** 2)))
                )
    for wave_index, values in enumerate(rms_collect):
        if values:
            rms[wave_index] = float(np.mean(values))
    return angle, phase, amplitude, strong, rms


def _random_reference_steps(field, steps, simulation):
    """Create matched random tracks moving at the mean saved LPO speed."""

    zonal_speeds = []
    for _event_id, event in steps.groupby("event_id", sort=False):
        event = event.sort_values("Time")
        if len(event) < 2:
            continue
        longitude = np.rad2deg(
            np.unwrap(
                np.deg2rad(
                    np.asarray(
                        event.precip_weighted_centroid_lon, dtype=float
                    )
                )
            )
        )
        elapsed_hours = (
            pd.to_datetime(event.Time).diff()
            / pd.Timedelta(hours=1)
        ).to_numpy(dtype=float)[1:]
        valid = np.isfinite(elapsed_hours) & (elapsed_hours > 0.0)
        zonal_speeds.extend(
            (np.diff(longitude)[valid] / elapsed_hours[valid]).tolist()
        )
    mean_speed_degrees_per_hour = (
        float(np.nanmean(zonal_speeds)) if zonal_speeds else 0.0
    )
    source_time = pd.DatetimeIndex(pd.to_datetime(field.Time.values))
    seed = 20260804 + list(LPO_SIMULATION_DIRS).index(simulation)
    rng = np.random.default_rng(seed)
    records = []
    for event_id, event in steps.groupby("event_id", sort=False):
        event = event.sort_values("lag_hours")
        lag_hours = np.asarray(event.lag_hours, dtype=int)
        earliest = pd.Timedelta(hours=int(lag_hours.min()))
        latest = pd.Timedelta(hours=int(lag_hours.max()))
        valid_reference = source_time[
            (source_time + earliest >= source_time[0])
            & (source_time + latest <= source_time[-1])
        ]
        if not len(valid_reference):
            raise RuntimeError(
                "Wave record cannot contain the matched random lifecycle"
            )
        reference_time = valid_reference[
            int(rng.integers(0, len(valid_reference)))
        ]
        reference_longitude = float(rng.uniform(0.0, 360.0))
        reference_latitude = float(
            rng.uniform(
                -TROPICAL_CENTROID_LIMIT_DEG,
                TROPICAL_CENTROID_LIMIT_DEG,
            )
        )
        for lag in lag_hours:
            records.append(
                {
                    "event_id": str(event_id),
                    "lag_hours": int(lag),
                    "Time": reference_time + pd.Timedelta(hours=int(lag)),
                    "precip_weighted_centroid_lon": np.mod(
                        reference_longitude
                        + mean_speed_degrees_per_hour * lag,
                        360.0,
                    ),
                    "precip_weighted_centroid_lat": reference_latitude,
                }
            )
    output = pd.DataFrame(records)
    output.attrs["mean_lpo_speed_degrees_per_hour"] = (
        mean_speed_degrees_per_hour
    )
    return output


def build_compact_lpo_wave_sections(ds, simulation, steps):
    """Build the compact prefiltered interface consumed by LPO lifecycle."""

    wave_names = tuple(LPO_LIFECYCLE_FILTERS)
    step_count = len(steps)
    surface_shape = (
        step_count,
        len(LPO_SECTION_RELATIVE_LATITUDE_DEGREES),
        len(LPO_SECTION_RELATIVE_LONGITUDE_DEGREES),
        len(wave_names),
    )
    native_fields = _native_lpo_vertical_fields(ds)
    native_template = native_fields["T"]
    native_dimension = vertical_dim(native_template)
    native_level_count = native_template.sizes[native_dimension]
    vertical_shape = (
        step_count,
        native_level_count,
        len(LPO_SECTION_RELATIVE_LONGITUDE_DEGREES),
        len(wave_names),
    )
    surface = {
        name: np.full(surface_shape, np.nan)
        for name in LPO_SECTION_SURFACE_FIELDS
    }
    vertical = {
        name: np.full(vertical_shape, np.nan)
        for name in LPO_SECTION_VERTICAL_FIELDS
    }
    precipitation_rms = np.full(len(wave_names), np.nan)
    phase_angle = np.full((step_count, len(wave_names)), np.nan)
    phase_number = np.full((step_count, len(wave_names)), np.nan)
    normalized_amplitude = np.full((step_count, len(wave_names)), np.nan)
    strong_amplitude = np.zeros((step_count, len(wave_names)), dtype=bool)
    random_phase_angle = np.full((step_count, len(wave_names)), np.nan)
    random_phase_number = np.full((step_count, len(wave_names)), np.nan)
    random_normalized_amplitude = np.full(
        (step_count, len(wave_names)), np.nan
    )
    random_strong_amplitude = np.zeros(
        (step_count, len(wave_names)), dtype=bool
    )
    random_reference_speed = np.nan

    for field_name in LPO_SECTION_SURFACE_FIELDS:
        progress(
            f"{simulation}: compact LPO latitude sections for {field_name}"
        )
        field = get_filter_field(ds, field_name)
        surface[field_name] = _latitude_resolved_lpo_wave_values(
            field, steps, wave_names
        )
        if field_name == "Precip":
            (
                phase_angle,
                phase_number,
                normalized_amplitude,
                strong_amplitude,
                precipitation_rms,
            ) = _precipitation_phase_at_lpo(
                field,
                steps,
                wave_names,
            )
            random_steps = _random_reference_steps(
                field, steps, simulation
            )
            random_reference_speed = random_steps.attrs[
                "mean_lpo_speed_degrees_per_hour"
            ]
            (
                random_phase_angle,
                random_phase_number,
                random_normalized_amplitude,
                random_strong_amplitude,
                _random_precipitation_rms,
            ) = _precipitation_phase_at_lpo(
                field,
                random_steps,
                wave_names,
            )

    for field_name in LPO_SECTION_VERTICAL_FIELDS:
        vertical[field_name] = _latitude_mean_native_lpo_wave_values(
            native_fields[field_name],
            steps,
            wave_names,
            progress_label=f"{simulation}: compact LPO {field_name}",
        )
    native_pressure = _latitude_mean_native_lpo_values(
        native_fields["pressure_hpa"], steps
    )

    coords = {
        "event_step": np.arange(step_count),
        "relative_latitude": LPO_SECTION_RELATIVE_LATITUDE_DEGREES,
        "relative_longitude": LPO_SECTION_RELATIVE_LONGITUDE_DEGREES,
        "native_level": np.arange(native_level_count),
        "wave_band": np.asarray(wave_names, dtype=str),
    }
    dataset = xr.Dataset(
        {
            "event_id": (("event_step",), steps.event_id.astype(str)),
            "Time": (("event_step",), pd.to_datetime(steps.Time)),
            "lag_hours": (("event_step",), steps.lag_hours.astype(int)),
            "precip_weighted_centroid_lon": (
                ("event_step",),
                steps.precip_weighted_centroid_lon.astype(float),
            ),
            "precip_weighted_centroid_lat": (
                ("event_step",),
                steps.precip_weighted_centroid_lat.astype(float),
            ),
            "filtered_precipitation": (
                (
                    "event_step", "relative_latitude",
                    "relative_longitude", "wave_band",
                ),
                surface["Precip"],
            ),
            "filtered_precipitable_water": (
                (
                    "event_step", "relative_latitude",
                    "relative_longitude", "wave_band",
                ),
                surface["PW"],
            ),
            "filtered_olr": (
                (
                    "event_step", "relative_latitude",
                    "relative_longitude", "wave_band",
                ),
                surface["OLR"],
            ),
            "filtered_temperature": (
                (
                    "event_step", "native_level",
                    "relative_longitude", "wave_band",
                ),
                vertical["T"],
            ),
            "filtered_upward_pressure_velocity": (
                (
                    "event_step", "native_level",
                    "relative_longitude", "wave_band",
                ),
                vertical["w"],
            ),
            "filtered_specific_humidity": (
                (
                    "event_step", "native_level",
                    "relative_longitude", "wave_band",
                ),
                vertical["qv"],
            ),
            "filtered_relative_humidity": (
                (
                    "event_step", "native_level",
                    "relative_longitude", "wave_band",
                ),
                vertical["RH"],
            ),
            "filtered_zonal_wind": (
                (
                    "event_step", "native_level",
                    "relative_longitude", "wave_band",
                ),
                vertical["u"],
            ),
            "filtered_static_stability": (
                (
                    "event_step", "native_level",
                    "relative_longitude", "wave_band",
                ),
                vertical["N2"],
            ),
            "filtered_native_pressure_hpa": (
                (
                    "event_step", "native_level",
                    "relative_longitude",
                ),
                native_pressure,
            ),
            "wave_phase_angle_degrees": (
                ("event_step", "wave_band"), phase_angle
            ),
            "wave_phase_number": (
                ("event_step", "wave_band"), phase_number
            ),
            "wave_normalized_amplitude": (
                ("event_step", "wave_band"), normalized_amplitude
            ),
            "wave_strong_amplitude": (
                ("event_step", "wave_band"), strong_amplitude
            ),
            "random_wave_phase_angle_degrees": (
                ("event_step", "wave_band"), random_phase_angle
            ),
            "random_wave_phase_number": (
                ("event_step", "wave_band"), random_phase_number
            ),
            "random_wave_normalized_amplitude": (
                ("event_step", "wave_band"), random_normalized_amplitude
            ),
            "random_wave_strong_amplitude": (
                ("event_step", "wave_band"), random_strong_amplitude
            ),
            "precipitation_wave_rms": (
                ("wave_band",), precipitation_rms
            ),
        },
        coords=coords,
        attrs={
            "cache_version": LPO_WAVE_SECTION_CACHE_VERSION,
            "wave_config_version": WAVE_CONFIG_VERSION,
            "tracking_product_version": LPO_TRACKING_PRODUCT_VERSION,
            "source_simulation": simulation,
            "test_subset": int(test_mode),
            "event_selection": (
                "saved primary_tropical_sample and reference_in_record_subset; "
                "every available sample within -72 to +72 hours"
            ),
            "default_longitude_half_width_degrees": (
                LPO_SECTION_DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES
            ),
            "longitude_half_width_sensitivity_degrees": json.dumps(
                LPO_SECTION_LONGITUDE_HALF_WIDTH_SENSITIVITY_DEGREES
            ),
            "vertical_latitude_projection": (
                "symmetric/direct bands use +/-10 degrees around each saved "
                "LPO latitude; antisymmetric bands use 0.5 times the "
                "10S-10N north-minus-south projection"
            ),
            "vertical_coordinate": (
                "native MPAS levels with sampled pressure in hPa"
            ),
            "surface_latitude_geometry": (
                "latitude relative to saved precipitation-weighted centroid"
            ),
            "filtering_owner": "equatorial_wave_diagnostics.py",
            "phase_method": (
                "Rios-Berrios-style eight phases from standardized filtered "
                "precipitation and its time derivative at the LPO centroid; "
                "phase 5 is the positive precipitation maximum; MRG uses "
                "the equatorial antisymmetric precipitation projection"
            ),
            "phase_is_not_lpo_lag": True,
            "random_phase_reference": (
                "one matched random trajectory per LPO event within the "
                f"{TROPICAL_CENTROID_LIMIT_DEG:g}-degree tropical acceptance "
                "range; trajectories use the "
                "simulation mean LPO zonal speed"
            ),
            "random_reference_mean_lpo_speed_degrees_per_hour": (
                random_reference_speed
            ),
        },
    )
    for name, units in (
        ("filtered_precipitation", "mm day-1"),
        ("filtered_precipitable_water", "kg m-2"),
        ("filtered_olr", "W m-2"),
        ("filtered_temperature", "K"),
        ("filtered_upward_pressure_velocity", "Pa s-1"),
        ("filtered_specific_humidity", "g kg-1"),
        ("filtered_relative_humidity", "%"),
        ("filtered_zonal_wind", "m s-1"),
        ("filtered_static_stability", "s-2"),
        ("filtered_native_pressure_hpa", "hPa"),
        ("precipitation_wave_rms", "mm day-1"),
    ):
        dataset[name].attrs["units"] = units
    dataset.filtered_relative_humidity.attrs[
        "source_variable"
    ] = "relhum"
    dataset.filtered_static_stability.attrs[
        "input_nonpositive_values"
    ] = "retained before filtering"
    dataset.filtered_upward_pressure_velocity.attrs[
        "sign_convention"
    ] = "positive values represent ascent (-omega)"
    ray_sections = _peak_filtered_temperature_ray_sections(
        native_fields, steps
    )
    dataset = xr.merge([dataset, ray_sections], compat="no_conflicts")
    dataset.attrs.update(ray_sections.attrs)
    return dataset


def project_latitude_field(da, projection, lat_bounds=None):
    """Project an unfiltered latitude field to match avgLat=True wave output."""
    if lat_bounds is None:
        lat_bounds = filter_lat_bounds
    lat_bounds = normalise_lat_bounds(lat_bounds)
    projection = lat_projection_for_wave("", projection)

    if projection in ("symmetric", "direct"):
        x = da.sel(latitude=slice(lat_bounds[0], lat_bounds[1]))
        return cosine_latitude_mean(x)
    south_bounds, north_bounds = hemisphere_bounds_from_lat_bounds(lat_bounds)
    north = cosine_latitude_mean(
        da.sel(latitude=slice(north_bounds[0], north_bounds[1]))
    )
    south = cosine_latitude_mean(
        da.sel(latitude=slice(south_bounds[0], south_bounds[1]))
    )
    north, south = xr.align(north, south, join="exact")
    if projection == "antisymmetric":
        return 0.5 * (north - south)
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
    if "latitude" in dims:
        variance = cosine_latitude_mean(wave_da**2, dims)
    else:
        variance = (wave_da**2).mean(dim=dims, skipna=True)
    return float(variance.values)


def wave_amplitude_index(wave_da):
    """RMS wave amplitude as a function of time."""
    dims = [d for d in ("latitude", "longitude") if d in wave_da.dims]
    if "latitude" in dims:
        variance = cosine_latitude_mean(wave_da**2, dims)
    else:
        variance = (wave_da**2).mean(dim=dims, skipna=True)
    amp = np.sqrt(variance)
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
        pr = latmean_for_filter_field(
            ds,
            "Precip",
            lat_bounds=filter_lat_bounds,
            projection="direct",
        )
        pr_vals = np.asarray(pr.values)
        threshold = np.nanpercentile(pr_vals, percentile)
        events = pr >= threshold
        norm_waves = {}
        for wname in wave_names:
            wav = get_filtered_wave(
                ds, i, "Precip", wname, avgLat=True
            ).sel(Time=pr.Time)
            std = float(
                wav.std(dim=("Time", "longitude"), skipna=True).values
            )
            if not np.isfinite(std) or std <= 0:
                raise RuntimeError(
                    f"{sim_names[i]} {wname}: filtered precipitation has "
                    "zero or invalid variance"
                )
            norm_waves[wname] = np.abs(wav / std)
        stack = xr.concat(
            [norm_waves[w] for w in wave_names],
            dim=xr.DataArray(wave_names, dims="wave", name="wave"),
        )
        winner_index = stack.argmax(dim="wave")
        winner = xr.DataArray(
            np.array(wave_names, dtype=object)[winner_index.values],
            dims=winner_index.dims,
            coords=winner_index.coords,
        )
        event_count = float(events.sum().values)
        if event_count <= 0:
            raise RuntimeError(
                f"{sim_names[i]}: no finite intense-precipitation samples"
            )
        for wname in wave_names:
            projection = lat_projection_for_wave(wname)
            frac = float(
                ((winner == wname).where(events)).sum().values / event_count
            )
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
    out = pd.DataFrame(attribution_rows)
    if len(out) > 0:
        write_cached_table("intense_precip_wave_attribution", out, signature)
    return out


def _wave_response_effective_segment_count(sample_count, interval_hours):
    """Count the existing 48-day/30-day-overlap spectral segments."""

    segment = int(round(48.0 * 24.0 / interval_hours))
    step = int(round((48.0 - 30.0) * 24.0 / interval_hours))
    if sample_count < segment:
        return 0
    return 1 + (sample_count - segment) // max(step, 1)


def _wave_response_block_days(wave_name):
    """Use the geometric midpoint of each wave period range."""

    period = np.asarray(WAVE_BOXES[wave_name]["period"], dtype=float)
    return float(np.sqrt(np.prod(period)))


def _wave_response_bootstrap_indices(wave_name, subset_name, sample_count):
    """Create reproducible indices shared across simulations and responses."""

    subset_offset = {
        "full_record": 0,
        "first_half": 1,
        "second_half": 2,
    }[subset_name]
    wave_offset = WAVE_RESPONSE_FILTERS.index(wave_name)
    rng = np.random.default_rng(
        WAVE_RESPONSE_BOOTSTRAP_SEED + 10 * wave_offset + subset_offset
    )
    requested_days = _wave_response_block_days(wave_name)
    block_length = max(1, int(round(requested_days * obs_per_day)))
    block_length = min(block_length, int(sample_count))
    indices = circular_time_block_indices(
        sample_count,
        block_length,
        WAVE_RESPONSE_BOOTSTRAP_RESAMPLES,
        rng,
    )
    return indices, block_length / obs_per_day


def _wave_response_bootstrap_summary(estimate, draws):
    """Summarise one central percentile interval and sign consistency."""

    draws = np.asarray(draws, dtype=float)
    draws = draws[np.isfinite(draws)]
    if draws.size == 0 or not np.isfinite(estimate):
        return {
            "ci_low": np.nan,
            "ci_high": np.nan,
            "bootstrap_noise": np.nan,
            "sign_robustness": np.nan,
            "ci_excludes_zero": False,
            "robust": False,
        }
    ci_low, ci_high = np.percentile(
        draws, WAVE_RESPONSE_CI_PERCENTILES
    )
    sign = np.sign(estimate)
    sign_robustness = (
        np.nan
        if sign == 0.0
        else float(np.mean(np.sign(draws) == sign))
    )
    excludes_zero = bool(ci_low > 0.0 or ci_high < 0.0)
    return {
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "bootstrap_noise": float(np.std(draws, ddof=1)),
        "sign_robustness": sign_robustness,
        "ci_excludes_zero": excludes_zero,
        "robust": bool(
            excludes_zero
            and sign_robustness >= WAVE_RESPONSE_ROBUST_SIGN_THRESHOLD
        ),
    }


def _wave_response_metric_row(
    simulation,
    wave_name,
    subset_name,
    response_key,
    response_label,
    response_kind,
    predictor,
    response,
    bootstrap_indices,
    bootstrap_block_days,
):
    """One same-band precipitation/response regression estimate."""

    predictor, response = xr.align(predictor, response, join="exact")
    subset_masks = record_subsets(
        predictor.Time.values,
        include_halves=(wave_name == EASTWARD_INTRASEASONAL_FILTER),
    )
    indices = np.flatnonzero(subset_masks[subset_name])
    x = predictor.isel(Time=indices).load()
    y = response.isel(Time=indices).load()
    frequency_count = resolved_frequency_count(
        x.sizes["Time"], dt, WAVE_BOXES[wave_name]["period"]
    )
    if frequency_count < 1:
        if subset_name == "full_record":
            raise RuntimeError(
                f"{simulation}: record is too short for "
                f"{wave_label(wave_name)}"
            )
        metrics = {
            "precipitation_rms_mm_day": np.nan,
            "precipitation_variance_mm2_day2": np.nan,
            "covariance": np.nan,
            "regression_slope": np.nan,
        }
        bootstrap_draws = np.full(
            WAVE_RESPONSE_BOOTSTRAP_RESAMPLES, np.nan
        )
        status = "insufficient_record_for_filter"
    else:
        metrics = filtered_wave_metrics(
            x.values,
            y.values,
            response_kind=response_kind,
        )
        bootstrap_draws = block_bootstrap_filtered_wave_metrics(
            x.values,
            y.values,
            response_kind=response_kind,
            time_indices=bootstrap_indices,
        )["regression_slope"]
        status = "ok"
    bootstrap = _wave_response_bootstrap_summary(
        metrics["regression_slope"], bootstrap_draws
    )
    units = (
        "W m-2 per mm day-1"
        if response_kind == "toa_longwave"
        else "cloud fraction per mm day-1"
    )
    return {
        "product_version": WAVE_RESPONSE_PRODUCT_VERSION,
        "wave_metrics_version": WAVE_METRICS_VERSION,
        "wave_config_version": WAVE_CONFIG_VERSION,
        "simulation": simulation,
        "wave": wave_name,
        "wave_label": wave_label(wave_name),
        "sensitivity_subset": subset_name,
        "response_variable": response_key,
        "response_label": response_label,
        "response_kind": response_kind,
        **metrics,
        "units": units,
        "filter_definition": json.dumps(
            jsonable_value(WAVE_BOXES[wave_name]), sort_keys=True
        ),
        "latitude_projection": wave_latitude_projection[wave_name],
        "resolved_fourier_frequency_count": frequency_count,
        "effective_segment_count": (
            np.nan
            if wave_name == EASTWARD_INTRASEASONAL_FILTER
            else _wave_response_effective_segment_count(
                x.sizes["Time"], dt
            )
        ),
        "regression_slope_ci_low": bootstrap["ci_low"],
        "regression_slope_ci_high": bootstrap["ci_high"],
        "regression_slope_bootstrap_noise": bootstrap["bootstrap_noise"],
        "regression_slope_sign_robustness": bootstrap["sign_robustness"],
        "regression_slope_ci_excludes_zero": bootstrap[
            "ci_excludes_zero"
        ],
        "regression_slope_robust": bootstrap["robust"],
        "bootstrap_method": (
            "circular moving-block bootstrap of complete longitude fields"
        ),
        "bootstrap_reference": WAVE_RESPONSE_BOOTSTRAP_REFERENCE,
        "bootstrap_interval": "central 95% percentile interval",
        "bootstrap_resamples": WAVE_RESPONSE_BOOTSTRAP_RESAMPLES,
        "bootstrap_block_days": bootstrap_block_days,
        "bootstrap_blocks_per_resample": int(
            np.ceil(
                x.sizes["Time"]
                / max(int(round(bootstrap_block_days * obs_per_day)), 1)
            )
        ),
        "finite_record_caution": (
            "20-100-day band has few representative-period blocks; compare "
            "the full-record and half-record estimates"
            if wave_name == EASTWARD_INTRASEASONAL_FILTER
            else ""
        ),
        "sample_count": int(x.size),
        "time_sample_count": int(x.sizes["Time"]),
        "status": status,
    }, bootstrap_draws


def compute_wave_response_metric_table():
    """Separate rainfall amplitude, cloud efficiency and TOA-LW response."""

    rows = []
    bootstrap_draws = {}
    reference_times = {}
    for simulation_index, ds in enumerate(dss):
        simulation = sim_names[simulation_index]
        for wave_name in WAVE_RESPONSE_FILTERS:
            predictor = get_filtered_wave(
                ds, simulation_index, "Precip", wave_name, avgLat=True
            )
            subsets = tuple(
                record_subsets(
                    predictor.Time.values,
                    include_halves=(
                        wave_name == EASTWARD_INTRASEASONAL_FILTER
                    ),
                )
            )
            responses = {
                "toa_longwave_response": (
                    "TOA longwave response per unit precipitation",
                    "toa_longwave",
                    get_filtered_wave(
                        ds,
                        simulation_index,
                        "OLR",
                        wave_name,
                        avgLat=True,
                    ),
                )
            }
            for response_key, response_label, field_name in (
                WAVE_RESPONSE_CLOUD_FIELDS
            ):
                responses[response_key] = (
                    response_label,
                    "cloud",
                    get_filtered_wave(
                        ds,
                        simulation_index,
                        field_name,
                        wave_name,
                        avgLat=True,
                    ),
                )
            for subset_name in subsets:
                subset_mask = record_subsets(
                    predictor.Time.values,
                    include_halves=(
                        wave_name == EASTWARD_INTRASEASONAL_FILTER
                    ),
                )[subset_name]
                subset_times = predictor.Time.values[subset_mask]
                time_key = (wave_name, subset_name)
                if time_key in reference_times:
                    if not np.array_equal(reference_times[time_key], subset_times):
                        raise RuntimeError(
                            f"{wave_label(wave_name)} {subset_name}: simulation "
                            "times do not support paired block resampling"
                        )
                else:
                    reference_times[time_key] = subset_times.copy()
                bootstrap_indices, bootstrap_block_days = (
                    _wave_response_bootstrap_indices(
                        wave_name,
                        subset_name,
                        subset_times.size,
                    )
                )
                for response_key, (
                    response_label,
                    response_kind,
                    response,
                ) in responses.items():
                    row, draws = _wave_response_metric_row(
                        simulation,
                        wave_name,
                        subset_name,
                        response_key,
                        response_label,
                        response_kind,
                        predictor,
                        response,
                        bootstrap_indices,
                        bootstrap_block_days,
                    )
                    rows.append(row)
                    bootstrap_draws[
                        (
                            simulation,
                            wave_name,
                            subset_name,
                            response_key,
                            "regression_slope",
                        )
                    ] = draws
    table = pd.DataFrame(rows)
    table.attrs["wave_response_bootstrap_draws"] = bootstrap_draws
    return table


def wave_response_direct_contrasts(table):
    """Direct experiment differences without thermal scaling or attribution."""

    bootstrap_draws = table.attrs.get("wave_response_bootstrap_draws", {})
    index_columns = ["wave", "sensitivity_subset", "response_variable"]
    metadata = table.drop_duplicates(index_columns).set_index(index_columns)
    rows = []
    for metric in (
        "precipitation_rms_mm_day",
        "precipitation_variance_mm2_day2",
        "covariance",
        "regression_slope",
    ):
        pivot = table.pivot(
            index=index_columns,
            columns="simulation",
            values=metric,
        )
        for index, values in pivot.iterrows():
            common = dict(zip(index_columns, index))
            for key, label, lhs, rhs in (
                (
                    "U_minus_control",
                    "u-nudged - control",
                    sim_names[1],
                    sim_names[0],
                ),
                (
                    "T_minus_control",
                    "T-nudged - control",
                    sim_names[2],
                    sim_names[0],
                ),
                (
                    "U_minus_T",
                    "u-nudged - T-nudged",
                    sim_names[1],
                    sim_names[2],
                ),
            ):
                bootstrap = {
                    "ci_low": np.nan,
                    "ci_high": np.nan,
                    "bootstrap_noise": np.nan,
                    "sign_robustness": np.nan,
                    "ci_excludes_zero": np.nan,
                    "robust": np.nan,
                }
                lhs_key = (
                    lhs,
                    common["wave"],
                    common["sensitivity_subset"],
                    common["response_variable"],
                    metric,
                )
                rhs_key = (
                    rhs,
                    common["wave"],
                    common["sensitivity_subset"],
                    common["response_variable"],
                    metric,
                )
                if lhs_key in bootstrap_draws and rhs_key in bootstrap_draws:
                    bootstrap = _wave_response_bootstrap_summary(
                        values[lhs] - values[rhs],
                        bootstrap_draws[lhs_key] - bootstrap_draws[rhs_key],
                    )
                rows.append(
                    {
                        **common,
                        "contrast_key": key,
                        "contrast_label": label,
                        "metric": metric,
                        "response_change": values[lhs] - values[rhs],
                        "units": metadata.loc[index, "units"],
                        **bootstrap,
                        "bootstrap_method": (
                            "paired circular moving-block bootstrap of "
                            "complete longitude fields"
                            if metric == "regression_slope"
                            else ""
                        ),
                        "bootstrap_interval": (
                            "central 95% percentile interval"
                            if metric == "regression_slope"
                            else ""
                        ),
                        "bootstrap_resamples": (
                            WAVE_RESPONSE_BOOTSTRAP_RESAMPLES
                            if metric == "regression_slope"
                            else 0
                        ),
                    }
                )
    return pd.DataFrame(rows)


def _wave_response_panel_values(table, wave_name, metric):
    subset = table[
        (table.wave == wave_name)
        & (table.sensitivity_subset == "full_record")
    ]
    if metric == "precipitation_rms_mm_day":
        subset = subset[subset.response_variable == "total_high_cloud"]
    elif metric == "total_high_cloud_efficiency":
        subset = subset[subset.response_variable == "total_high_cloud"]
        metric = "regression_slope"
    elif metric == "toa_longwave_response":
        subset = subset[
            subset.response_variable == "toa_longwave_response"
        ]
        metric = "regression_slope"
    return subset.set_index("simulation")[metric].reindex(sim_names)


def _draw_wave_response_errorbars(axis, x, values, lower, upper):
    """Draw central percentile intervals using the project bar style."""

    values = np.asarray(values, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    valid = np.isfinite(values) & np.isfinite(lower) & np.isfinite(upper)
    if not valid.any():
        return
    axis.errorbar(
        np.asarray(x)[valid],
        values[valid],
        yerr=np.vstack(
            (
                np.maximum(values[valid] - lower[valid], 0.0),
                np.maximum(upper[valid] - values[valid], 0.0),
            )
        ),
        color="0.15",
        fmt="none",
        lw=1.1,
        capsize=2,
        zorder=4,
    )


def plot_wave_response_metrics(table, contrasts):
    """Figure 11: physical rainfall amplitude and same-band efficiencies."""

    metrics = (
        ("precipitation_rms_mm_day", "Precipitation RMS\n(mm day$^{-1}$)"),
        (
            "total_high_cloud_efficiency",
            "Total-high-cloud efficiency\n(fraction per mm day$^{-1}$)",
        ),
        (
            "toa_longwave_response",
            "TOA longwave response\n(W m$^{-2}$ per mm day$^{-1}$)",
        ),
    )
    fig, axes = plt.subplots(
        len(WAVE_RESPONSE_FILTERS),
        len(metrics),
        figsize=(12, 2.5 * len(WAVE_RESPONSE_FILTERS)),
        constrained_layout=True,
    )
    for row, wave_name in enumerate(WAVE_RESPONSE_FILTERS):
        for column, (metric, label) in enumerate(metrics):
            axis = axes[row, column]
            values = _wave_response_panel_values(table, wave_name, metric)
            axis.bar(
                np.arange(len(sim_names)),
                values.values,
                color=cols,
                width=0.72,
            )
            if metric != "precipitation_rms_mm_day":
                response_variable = (
                    "toa_longwave_response"
                    if metric == "toa_longwave_response"
                    else "total_high_cloud"
                )
                interval = table[
                    (table.wave == wave_name)
                    & (table.sensitivity_subset == "full_record")
                    & (table.response_variable == response_variable)
                ].set_index("simulation").reindex(sim_names)
                _draw_wave_response_errorbars(
                    axis,
                    np.arange(len(sim_names)),
                    values.values,
                    interval.regression_slope_ci_low.values,
                    interval.regression_slope_ci_high.values,
                )
            axis.axhline(0, color="0.6", lw=0.7)
            axis.grid(axis="y", alpha=0.2)
            if row == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(wave_label(wave_name))
            if row == len(WAVE_RESPONSE_FILTERS) - 1:
                axis.set_xticks(
                    range(len(sim_names)), sim_names, rotation=25, ha="right"
                )
            else:
                axis.set_xticks([])
    fig.suptitle(
        "Wave precipitation amplitude and response per unit precipitation\n"
        "Error bars show 95% block-bootstrap intervals for regressions"
    )
    save_figure(
        WAVE_RESPONSE_FIGURE_NUMBER,
        fig,
        "wave_precipitation_cloud_olr_response",
    )

    contrast_keys = ("U_minus_control", "T_minus_control", "U_minus_T")
    contrast_labels = (
        r"$u$-nudged − control",
        r"$T$-nudged − control",
        r"$u$-nudged − $T$-nudged",
    )
    fig, axes = plt.subplots(
        len(WAVE_RESPONSE_FILTERS),
        len(metrics),
        figsize=(12, 2.5 * len(WAVE_RESPONSE_FILTERS)),
        constrained_layout=True,
    )
    for row, wave_name in enumerate(WAVE_RESPONSE_FILTERS):
        for column, (metric, label) in enumerate(metrics):
            response_variable = (
                "toa_longwave_response"
                if metric == "toa_longwave_response"
                else "total_high_cloud"
            )
            contrast_metric = (
                "precipitation_rms_mm_day"
                if metric == "precipitation_rms_mm_day"
                else "regression_slope"
            )
            subset = contrasts[
                (contrasts.wave == wave_name)
                & (contrasts.sensitivity_subset == "full_record")
                & (contrasts.response_variable == response_variable)
                & (contrasts.metric == contrast_metric)
            ].set_index("contrast_key")
            axis = axes[row, column]
            values = subset.response_change.reindex(contrast_keys)
            axis.bar(
                np.arange(len(contrast_keys)),
                values.values,
                color=(cols[1], cols[2], "0.35"),
                width=0.72,
            )
            if contrast_metric == "regression_slope":
                interval = subset.reindex(contrast_keys)
                _draw_wave_response_errorbars(
                    axis,
                    np.arange(len(contrast_keys)),
                    values.values,
                    interval.ci_low.values,
                    interval.ci_high.values,
                )
            axis.axhline(0, color="0.6", lw=0.7)
            axis.grid(axis="y", alpha=0.2)
            if row == 0:
                axis.set_title(label)
            if column == 0:
                axis.set_ylabel(wave_label(wave_name))
            if row == len(WAVE_RESPONSE_FILTERS) - 1:
                axis.set_xticks(
                    range(len(contrast_keys)),
                    contrast_labels,
                    rotation=25,
                    ha="right",
                )
            else:
                axis.set_xticks([])
    fig.suptitle(
        "Direct wave-response differences; no thermal scaling\n"
        "Error bars show paired 95% block-bootstrap intervals"
    )
    save_figure(
        WAVE_RESPONSE_FIGURE_NUMBER,
        fig,
        "wave_precipitation_cloud_olr_response_differences",
    )


def plot_wave_cloud_class_efficiencies(table, contrasts):
    """Plot each filtered cloud response and its direct contrasts."""

    cloud_responses = tuple(
        (response_key, response_label)
        for response_key, response_label, _field_name
        in WAVE_RESPONSE_CLOUD_FIELDS
    )
    fig, axes = plt.subplots(
        len(WAVE_RESPONSE_FILTERS),
        len(cloud_responses),
        figsize=(3.5 * len(cloud_responses), 2.5 * len(WAVE_RESPONSE_FILTERS)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, wave_name in enumerate(WAVE_RESPONSE_FILTERS):
        for column, (response_key, response_label) in enumerate(
            cloud_responses
        ):
            axis = axes[row, column]
            subset = table[
                (table.wave == wave_name)
                & (table.sensitivity_subset == "full_record")
                & (table.response_variable == response_key)
            ].set_index("simulation").reindex(sim_names)
            values = subset.regression_slope.to_numpy(dtype=float)
            axis.bar(
                np.arange(len(sim_names)),
                values,
                color=cols,
                width=0.72,
            )
            _draw_wave_response_errorbars(
                axis,
                np.arange(len(sim_names)),
                values,
                subset.regression_slope_ci_low.values,
                subset.regression_slope_ci_high.values,
            )
            axis.axhline(0.0, color="0.6", lw=0.7)
            axis.grid(axis="y", alpha=0.2)
            if row == 0:
                axis.set_title(response_label)
            if column == 0:
                axis.set_ylabel(wave_label(wave_name))
            if row == len(WAVE_RESPONSE_FILTERS) - 1:
                axis.set_xticks(
                    range(len(sim_names)),
                    sim_names,
                    rotation=25,
                    ha="right",
                )
            else:
                axis.set_xticks([])
    fig.suptitle(
        "Wave-filtered cloud-class efficiency "
        "(fraction per mm day$^{-1}$)\n"
        "Error bars show 95% block-bootstrap intervals"
    )
    save_figure(
        WAVE_RESPONSE_FIGURE_NUMBER,
        fig,
        "wave_precipitation_cloud_class_efficiencies",
    )

    contrast_keys = ("U_minus_control", "T_minus_control", "U_minus_T")
    contrast_labels = (
        r"$u$-nudged − control",
        r"$T$-nudged − control",
        r"$u$-nudged − $T$-nudged",
    )
    fig, axes = plt.subplots(
        len(WAVE_RESPONSE_FILTERS),
        len(cloud_responses),
        figsize=(3.5 * len(cloud_responses), 2.5 * len(WAVE_RESPONSE_FILTERS)),
        constrained_layout=True,
        squeeze=False,
    )
    for row, wave_name in enumerate(WAVE_RESPONSE_FILTERS):
        for column, (response_key, response_label) in enumerate(
            cloud_responses
        ):
            axis = axes[row, column]
            subset = contrasts[
                (contrasts.wave == wave_name)
                & (contrasts.sensitivity_subset == "full_record")
                & (contrasts.response_variable == response_key)
                & (contrasts.metric == "regression_slope")
            ].set_index("contrast_key").reindex(contrast_keys)
            values = subset.response_change.to_numpy(dtype=float)
            axis.bar(
                np.arange(len(contrast_keys)),
                values,
                color=(cols[1], cols[2], "0.35"),
                width=0.72,
            )
            _draw_wave_response_errorbars(
                axis,
                np.arange(len(contrast_keys)),
                values,
                subset.ci_low.values,
                subset.ci_high.values,
            )
            axis.axhline(0.0, color="0.6", lw=0.7)
            axis.grid(axis="y", alpha=0.2)
            if row == 0:
                axis.set_title(response_label)
            if column == 0:
                axis.set_ylabel(wave_label(wave_name))
            if row == len(WAVE_RESPONSE_FILTERS) - 1:
                axis.set_xticks(
                    range(len(contrast_keys)),
                    contrast_labels,
                    rotation=25,
                    ha="right",
                )
            else:
                axis.set_xticks([])
    fig.suptitle(
        "Direct cloud-class efficiency differences; no thermal scaling\n"
        "Error bars show paired 95% block-bootstrap intervals"
    )
    save_figure(
        WAVE_RESPONSE_FIGURE_NUMBER,
        fig,
        "wave_precipitation_cloud_class_efficiency_differences",
    )


def _plot_wave_response_scatter(
    table,
    response_field,
    response_variable,
    response_kind,
    output_stem,
    response_label=None,
):
    """Plot the filtered values used by each Figure 11 regression."""

    is_longwave = response_kind == "toa_longwave"
    is_total_cloud = response_variable == "total_high_cloud"
    response_symbol = (
        r"$L_f$"
        if is_longwave
        else (r"$H_f$" if is_total_cloud else r"$H_{f,c}$")
    )
    response_units = r"W m$^{-2}$" if is_longwave else "cloud fraction"
    fig, axes = plt.subplots(
        len(WAVE_RESPONSE_FILTERS),
        len(sim_names),
        figsize=(11.5, 2.6 * len(WAVE_RESPONSE_FILTERS)),
        constrained_layout=True,
        squeeze=False,
    )
    for row_index, wave_name in enumerate(WAVE_RESPONSE_FILTERS):
        for simulation_index, ds in enumerate(dss):
            axis = axes[row_index, simulation_index]
            predictor = get_filtered_wave(
                ds,
                simulation_index,
                "Precip",
                wave_name,
                avgLat=True,
            )
            response = get_filtered_wave(
                ds,
                simulation_index,
                response_field,
                wave_name,
                avgLat=True,
            )
            predictor, response = xr.align(
                predictor, response, join="exact"
            )
            x = np.asarray(predictor.load().values, dtype=float).ravel()
            y = np.asarray(response.load().values, dtype=float).ravel()
            finite = np.isfinite(x) & np.isfinite(y)
            x = x[finite]
            y = y[finite]
            if x.size == 0:
                raise RuntimeError(
                    f"{sim_names[simulation_index]} {wave_label(wave_name)}: "
                    "scatter regression has no finite values"
                )
            sample_count = min(WAVE_RESPONSE_SCATTER_SAMPLE, x.size)
            rng = np.random.default_rng(
                WAVE_RESPONSE_BOOTSTRAP_SEED
                + 100
                + 10 * row_index
                + simulation_index
            )
            sample = rng.choice(x.size, size=sample_count, replace=False)
            axis.scatter(
                x[sample],
                y[sample],
                s=3,
                alpha=0.08,
                color=cols[simulation_index],
                edgecolors="none",
                rasterized=True,
            )
            metric_row = table[
                (table.simulation == sim_names[simulation_index])
                & (table.wave == wave_name)
                & (table.sensitivity_subset == "full_record")
                & (table.response_variable == response_variable)
            ].iloc[0]
            efficiency = float(metric_row.regression_slope)
            raw_slope = -efficiency if is_longwave else efficiency
            line_limit = float(np.nanpercentile(np.abs(x), 99.0))
            line_x = np.asarray([-line_limit, line_limit])
            axis.plot(line_x, raw_slope * line_x, color="0.1", lw=1.2)
            interval_label = (
                f"[{metric_row.regression_slope_ci_low:.3g}, "
                f"{metric_row.regression_slope_ci_high:.3g}]"
            )
            coefficient_label = (
                r"$-\beta_L$"
                if is_longwave
                else (r"$\beta_H$" if is_total_cloud else r"$\beta_{H,c}$")
            )
            axis.text(
                0.03,
                0.97,
                f"{coefficient_label} = {efficiency:.3g}\n"
                f"95% block interval {interval_label}\n"
                f"{sample_count:,} of {x.size:,} points shown",
                transform=axis.transAxes,
                va="top",
                ha="left",
                fontsize=7.5,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.75),
            )
            axis.axhline(0.0, color="0.6", lw=0.7)
            axis.axvline(0.0, color="0.6", lw=0.7)
            axis.grid(alpha=0.2)
            if row_index == 0:
                axis.set_title(sim_names[simulation_index])
            if simulation_index == 0:
                axis.set_ylabel(
                    f"{wave_label(wave_name)}\n{response_symbol} "
                    f"({response_units})"
                )
            if row_index == len(WAVE_RESPONSE_FILTERS) - 1:
                axis.set_xlabel(r"$P_f$ (mm day$^{-1}$)")
    if is_longwave:
        relation = r"$L_f$ versus $P_f$; Figure 11 reports $-\beta_L$"
    elif is_total_cloud:
        relation = r"$H_f$ versus $P_f$; Figure 11 reports $\beta_H$"
    else:
        relation = (
            f"{response_label}: "
            r"$H_{f,c}$ versus $P_f$; Figure 11 reports $\beta_{H,c}$"
        )
    fig.suptitle(f"Wave-filtered regression values: {relation}")
    save_figure(WAVE_RESPONSE_FIGURE_NUMBER, fig, output_stem)


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
    filter_name = WAVE_BOXES[wave_name]["filter_name"]

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


add_period_reference_lines_frequency = partial(
    add_period_reference_lines_frequency,
    periods=spectrum_period_reference_days,
)


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
        if component not in WAVE_BOXES[wname]["components"]:
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
        if component not in WAVE_BOXES[wname]["components"]:
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
        if component not in WAVE_BOXES[wname]["components"]:
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
    da = da.rename({"latitude": "lat", "longitude": "lon"}).load()
    required_samples = int(spectrum_segment_days * obs_per_day)
    if da.sizes["Time"] < required_samples:
        raise ValueError(
            f"{field_name} spectrum requires at least {required_samples} "
            f"samples ({spectrum_segment_days} days), but the selected record "
            f"contains {da.sizes['Time']}"
        )
    finite = np.isfinite(np.asarray(da.values))
    if not np.all(finite):
        raise ValueError(
            f"{field_name} spectrum input contains "
            f"{finite.size - np.count_nonzero(finite)} non-finite values"
        )
    return da


def spectrum_cache_signature(sim_index, field_name=spectrum_field):
    return cache_signature(
        "normalised_wave_spectrum",
        extra=dict(
            cache_version=SPECTRUM_CACHE_VERSION,
            simulation=sim_names[sim_index],
            spectrum_field=field_name,
            spectrum_lat_bounds=jsonable_value(spectrum_lat_bounds),
            spectrum_segment_days=spectrum_segment_days,
            spectrum_overlap_days=spectrum_overlap_days,
            obs_per_day=obs_per_day,
            components=list(spectrum_components),
        ),
    )


def spectrum_cache_name(sim_index, field_name):
    simulation_name = safe_name(sim_names[sim_index])
    if field_name == spectrum_field:
        return f"normalised_wave_spectrum_{simulation_name}"
    return f"normalised_wave_spectrum_{safe_name(field_name)}_{simulation_name}"


def load_cached_spectra(sim_index, field_name=spectrum_field):
    cache_name = spectrum_cache_name(sim_index, field_name)
    cached = load_cached_dataset(
        cache_name,
        spectrum_cache_signature(sim_index, field_name),
    )
    if cached is None or "normalised_power" not in cached:
        return None
    return {
        str(component): cached["normalised_power"].sel(component=component)
        for component in cached.component.values
    }


def write_cached_spectra(sim_index, spectra, field_name=spectrum_field):
    components = list(spectra)
    spectrum_stack = xr.concat(
        [spectra[component] for component in components],
        dim=xr.DataArray(components, dims="component", name="component"),
    )
    ds_out = spectrum_stack.to_dataset(name="normalised_power")
    ds_out.attrs.update(
        description="Cached normalised symmetric/antisymmetric wave spectrum",
        simulation=sim_names[sim_index],
        field=field_name,
    )
    cache_name = spectrum_cache_name(sim_index, field_name)
    write_cached_dataset(
        cache_name,
        ds_out,
        spectrum_cache_signature(sim_index, field_name),
    )


def compute_normalised_spectra_for_sim(ds, sim_index, field_name=spectrum_field):
    spectra = load_cached_spectra(sim_index, field_name)
    if spectra is not None:
        return spectra

    sp_input = spectrum_input_field(ds, field_name, spectrum_lat_bounds)
    sp_input = cosine_weighted_power_input(sp_input, lat_dim="lat")
    raw_spectrum = spacetime_power(
        sp_input,
        lataggreg="sum",
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
    write_cached_spectra(sim_index, spectra, field_name)
    return spectra


def load_or_compute_normalised_spectra_by_sim(field_name=spectrum_field):
    return [
        compute_normalised_spectra_for_sim(ds, i, field_name)
        for i, ds in enumerate(dss)
    ]


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
        for wname in critical_level_wave_names:
            hist = np.zeros(centres.size, dtype=float)
            component_names = [
                component for component in WAVE_BOXES[wname]["components"]
                if component in spectra
            ]
            if not component_names:
                raise RuntimeError(
                    f"{sim_names[i]} {wname}: none of the configured spectrum "
                    f"components {WAVE_BOXES[wname]['components']} are present"
                )
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
        dist.to_csv(
            os.path.join(product_output_dir, "wave_phase_speed_power_distributions.csv"),
            index=False,
        )
    return dist


def add_matching_projection_power_fractions(df):
    """Expose each band's share of its matching projected anomaly power.

    Symmetric/direct filters use the 10 S--10 N mean; antisymmetric filters
    use the paired-hemisphere difference. Fractions with different latitude
    projections therefore provide within-band context but are not intended
    to sum across every row.
    """
    out = df.copy()
    out["wave_fraction_of_projected_total"] = out["wave_fraction"]
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


def upward_pressure_velocity_profiles(ds):
    """Return upward-positive pressure velocity on one common pressure grid."""

    precipitation_times = precip_rate(ds).Time
    profiles = [
        project_latitude_field(
            upward_pressure_velocity(
                ds,
                pressure_hpa,
                times=precipitation_times,
            ),
            "symmetric",
            lat_bounds=filter_lat_bounds,
        ).astype("float32")
        for pressure_hpa in top_heaviness_pressure_hpa
    ]
    out = xr.concat(
        profiles,
        dim=xr.IndexVariable("pressure_hpa", top_heaviness_pressure_hpa),
    ).transpose("Time", "longitude", "pressure_hpa")
    out.name = "upward_pressure_velocity"
    out.attrs.update(
        units="Pa s-1",
        sign_convention="positive values represent ascent (-omega)",
        vertical_motion_source="native MPAS omg_isobaric",
        conversion=(
            "stored profile is -omg_isobaric so positive values represent "
            "ascent"
        ),
        latitude_projection="symmetric mean over 10S-10N",
    )
    return out


def _finite_profile_samples(profile):
    values = np.asarray(profile.values, dtype=float)
    samples = values.reshape(-1, values.shape[-1])
    valid = np.all(np.isfinite(samples), axis=1)
    if np.count_nonzero(valid) < 10:
        raise ValueError("Too few finite pressure-velocity profiles for EOFs")
    return samples[valid]


def compute_top_heaviness_products():
    profiles = [upward_pressure_velocity_profiles(ds).load() for ds in dss]
    weights = pressure_layer_weights(top_heaviness_pressure_hpa)
    sample_sets = [_finite_profile_samples(profile) for profile in profiles]
    minimum_count = min(samples.shape[0] for samples in sample_sets)
    bases = {
        "unfiltered_control": common_vertical_motion_eofs(
            sample_sets[0], top_heaviness_pressure_hpa, weights=weights
        ),
        "equally_weighted_pooled": common_vertical_motion_eofs(
            np.concatenate(
                [samples[:minimum_count] for samples in sample_sets], axis=0
            ),
            top_heaviness_pressure_hpa,
            weights=weights,
        ),
    }
    for source, basis in bases.items():
        coherence = abs(np.sum(basis["eofs"][0] * weights))
        if coherence < 0.15:
            raise RuntimeError(
                f"{source}: EOF1 is not a physically coherent deep vertical-motion "
                f"mode (weighted column coherence={coherence:.3f})"
            )

    rows = []
    for source, basis in bases.items():
        for simulation_index, (simulation, profile) in enumerate(zip(sim_names, profiles)):
            values = np.asarray(profile.values, dtype=float)
            if not np.all(np.isfinite(values)):
                pressure_mean = np.nanmean(
                    values.reshape(-1, values.shape[-1]), axis=0
                )
                bad = ~np.isfinite(values)
                values[bad] = np.broadcast_to(pressure_mean, values.shape)[bad]
            pcs = project_onto_eofs(
                values.reshape(-1, values.shape[-1]),
                basis["eofs"],
                basis["weights"],
            ).reshape(values.shape[0], values.shape[1], 2)
            for wave_name in TOP_HEAVINESS_FILTERS:
                subsets = record_subsets(
                    profile.Time.values,
                    include_halves=(wave_name == EASTWARD_INTRASEASONAL_FILTER),
                )
                for subset_name, subset_mask in subsets.items():
                    selected = pcs[subset_mask]
                    box = WAVE_BOXES[wave_name]
                    resolved = resolved_frequency_count(
                        selected.shape[0], dt, box["period"]
                    )
                    row = {
                        "wave_metrics_version": WAVE_METRICS_VERSION,
                        "wave_config_version": WAVE_CONFIG_VERSION,
                        "simulation": simulation,
                        "simulation_index": simulation_index,
                        "wave": wave_name,
                        "wave_label": wave_label(wave_name),
                        "sensitivity_subset": subset_name,
                        "pressure_min_hpa": float(top_heaviness_pressure_hpa.min()),
                        "pressure_max_hpa": float(top_heaviness_pressure_hpa.max()),
                        "eof_basis_source": source,
                        "eof1_explained_variance": float(
                            basis["explained_variance"][0]
                        ),
                        "eof2_explained_variance": float(
                            basis["explained_variance"][1]
                        ),
                        "resolved_fourier_frequency_count": resolved,
                        "vertical_motion_source": profile.attrs[
                            "vertical_motion_source"
                        ],
                        "eof_sign_convention": basis["convention"],
                        "status": "ok",
                    }
                    if resolved < 1:
                        if subset_name == "full_record":
                            raise RuntimeError(
                                f"{simulation}: full record is too short for "
                                f"{wave_label(wave_name)} top-heaviness"
                            )
                        row.update(
                            tau_r=np.nan, tau_i=np.nan, pc1_power=np.nan,
                            pc2_power=np.nan, spectral_coefficient_count=0,
                            status="insufficient_record_for_filter",
                        )
                    else:
                        o1, count = spectral_coefficients_for_box(
                            selected[:, :, 0], interval_hours=dt,
                            k_bounds=box["k"], period_bounds_days=box["period"],
                        )
                        o2, count2 = spectral_coefficients_for_box(
                            selected[:, :, 1], interval_hours=dt,
                            k_bounds=box["k"], period_bounds_days=box["period"],
                        )
                        result = complex_top_heaviness(o1, o2)
                        result["spectral_coefficient_count"] = min(count, count2)
                        row.update(result)
                    rows.append(row)

    basis_names = list(bases)
    basis_ds = xr.Dataset(
        {
            "eof": (
                ("basis_source", "eof_number", "pressure_hpa"),
                np.stack([bases[name]["eofs"] for name in basis_names]),
            ),
            "pressure_layer_weight": (
                ("basis_source", "pressure_hpa"),
                np.stack([bases[name]["weights"] for name in basis_names]),
            ),
            "explained_variance": (
                ("basis_source", "eof_number"),
                np.stack([bases[name]["explained_variance"] for name in basis_names]),
            ),
        },
        coords={
            "basis_source": basis_names,
            "eof_number": [1, 2],
            "pressure_hpa": top_heaviness_pressure_hpa,
        },
        attrs={
            "wave_metrics_version": WAVE_METRICS_VERSION,
            "wave_config_version": WAVE_CONFIG_VERSION,
            "vertical_motion_convention": profiles[0].attrs["sign_convention"],
            "conversion": profiles[0].attrs["conversion"],
            "basis_definition": (
                "primary: unfiltered control tropical -omega anomalies; "
                "validation: equal samples pooled across three simulations"
            ),
        },
    )
    basis_ds.eof.attrs.update(
        units="normalised Pa s-1 structure",
        normalization="sum(eof^2 * pressure_layer_weight) = 1",
    )
    basis_ds.pressure_layer_weight.attrs.update(
        units="1", definition="normalised pressure-layer thickness"
    )
    basis_ds.explained_variance.attrs["units"] = "1"
    return pd.DataFrame(rows), basis_ds


def native_kelvin_background_profile(zds):
    required = {"uReconstructZonal", "theta", "pressure"}
    missing = sorted(required - set(zds.data_vars))
    if missing:
        raise RuntimeError(
            "Kelvin dispersion zonal-mean input lacks required native MPAS "
            f"fields: {missing}"
    )
    theta = zds.theta
    wind = zds.uReconstructZonal
    zdim = vertical_dim(theta)
    if zdim != "z" or "z" not in theta.coords:
        raise RuntimeError(
            "Kelvin dispersion requires the zonal-mean product's physical "
            "'z' coordinate"
        )
    height = np.asarray(theta.z.values, dtype=float)
    if height.ndim != 1 or height.size != theta.sizes[zdim]:
        raise RuntimeError(
            "Kelvin dispersion requires a one-dimensional zonal-mean "
            "height coordinate"
        )
    if vertical_dim(wind) != zdim:
        wind = wind.rename({vertical_dim(wind): zdim})
    theta_subset = theta.sel(latitude=slice(*filter_lat_bounds))
    wind_subset = wind.sel(latitude=slice(*filter_lat_bounds))
    pressure_subset = pressure_to_pa(zds.pressure).sel(
        latitude=slice(*filter_lat_bounds)
    ) / 100.0
    n2 = xr.apply_ufunc(
        buoyancy_frequency_squared,
        theta_subset,
        xr.DataArray(height, dims=(zdim,), coords={zdim: theta_subset[zdim]}),
        input_core_dims=[[zdim], [zdim]],
        output_core_dims=[[zdim]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )
    def deep_tropical_mean(field):
        latitude_weights = np.cos(np.deg2rad(field.latitude))
        reduced = field.weighted(latitude_weights).mean(
            "latitude", skipna=True
        )
        remaining = [dim for dim in reduced.dims if dim != zdim]
        return reduced.mean(remaining, skipna=True) if remaining else reduced

    n2_mean = np.asarray(deep_tropical_mean(n2).values, dtype=float)
    n_mean = np.sqrt(n2_mean)
    wind_mean = np.asarray(
        deep_tropical_mean(wind_subset).values, dtype=float
    )
    pressure_mean = np.asarray(
        deep_tropical_mean(pressure_subset).values, dtype=float
    )
    order = np.argsort(height)
    height, n2_mean, n_mean, wind_mean, pressure_mean = (
        height[order], n2_mean[order], n_mean[order], wind_mean[order],
        pressure_mean[order],
    )
    valid = (
        np.isfinite(height)
        & np.isfinite(n2_mean)
        & np.isfinite(n_mean)
        & np.isfinite(wind_mean)
        & np.isfinite(pressure_mean)
        & (pressure_mean > 0.0)
    )
    height, n2_mean, n_mean, wind_mean, pressure_mean = (
        height[valid], n2_mean[valid], n_mean[valid], wind_mean[valid],
        pressure_mean[valid],
    )
    if height.size < 3:
        raise RuntimeError("Insufficient native-height levels for Kelvin dispersion")
    return {
        "height_m": kelvin_profile_height_m,
        "u": np.interp(
            kelvin_profile_height_m, height, wind_mean, left=np.nan, right=np.nan
        ),
        "n": np.interp(
            kelvin_profile_height_m, height, n_mean, left=np.nan, right=np.nan
        ),
        "n2": np.interp(
            kelvin_profile_height_m, height, n2_mean,
            left=np.nan, right=np.nan,
        ),
        "pressure_hpa": np.interp(
            kelvin_profile_height_m, height, pressure_mean,
            left=np.nan, right=np.nan,
        ),
    }


def compute_kelvin_dispersion_products():
    backgrounds = [native_kelvin_background_profile(zds) for zds in dss_zm]
    shape = (
        len(sim_names), len(kelvin_sensitivity_wavenumbers),
        len(kelvin_sensitivity_vertical_wavelengths_km),
        kelvin_profile_height_m.size,
    )
    variables = {
        name: np.full(shape, np.nan)
        for name in ("u", "n", "n2", "pressure", "cgx", "cgz", "slope")
    }
    for si, background in enumerate(backgrounds):
        for ki, zonal_wavenumber in enumerate(kelvin_sensitivity_wavenumbers):
            k_value = kelvin_reference_k_m1 * zonal_wavenumber
            for mi, wavelength_km in enumerate(
                kelvin_sensitivity_vertical_wavelengths_km
            ):
                m_value = kelvin_reference_m_m1 * (10.0 / wavelength_km)
                result = kelvin_group_velocities(
                    background["u"], background["n"],
                    zonal_wavenumber_m1=k_value,
                    vertical_wavenumber_m1=m_value,
                    minimum_abs_cgx_m_s=0.1,
                )
                variables["u"][si, ki, mi] = background["u"]
                variables["n"][si, ki, mi] = background["n"]
                variables["n2"][si, ki, mi] = background["n2"]
                variables["pressure"][si, ki, mi] = background[
                    "pressure_hpa"
                ]
                variables["cgx"][si, ki, mi] = result["cgx_m_s"]
                variables["cgz"][si, ki, mi] = result["cgz_m_s"]
                variables["slope"][si, ki, mi] = result["ray_slope"]
    dims = ("simulation", "zonal_wavenumber", "vertical_wavelength_km", "height_m")
    coords = {
        "simulation": np.asarray(sim_names, dtype=str),
        "zonal_wavenumber": np.asarray(
            kelvin_sensitivity_wavenumbers, dtype=int
        ),
        "vertical_wavelength_km": np.asarray(
            kelvin_sensitivity_vertical_wavelengths_km, dtype=float
        ),
        "height_m": kelvin_profile_height_m,
    }
    dataset = xr.Dataset(
        {
            "background_zonal_wind": (dims, variables["u"]),
            "buoyancy_frequency": (dims, variables["n"]),
            "static_stability": (dims, variables["n2"]),
            "background_pressure_hpa": (dims, variables["pressure"]),
            "horizontal_group_velocity": (dims, variables["cgx"]),
            "vertical_group_velocity": (dims, variables["cgz"]),
            "ray_slope": (dims, variables["slope"]),
        },
        coords=coords,
        attrs={
            "diagnostic": "Kelvin-wave reference energy dispersion",
            "primary_zonal_wavenumber": 1,
            "primary_k_m-1": kelvin_reference_k_m1,
            "primary_vertical_wavelength_km": 10.0,
            "primary_m_m-1": kelvin_reference_m_m1,
            "latitude_bounds": str(filter_lat_bounds),
            "vertical_wavenumber_convention": (
                "m<0 denotes upward energy propagation; Cgz=N*k/m^2 is positive"
            ),
            "near_zero_cgx_mask_m_s": 0.1,
            "wave_metrics_version": WAVE_METRICS_VERSION,
            "wave_config_version": WAVE_CONFIG_VERSION,
        },
    )
    for name, units, long_name in (
        ("background_zonal_wind", "m s-1", "deep-tropical-mean background zonal wind"),
        ("buoyancy_frequency", "s-1", "deep-tropical-mean buoyancy frequency"),
        ("static_stability", "s-2", "deep-tropical-mean static stability"),
        ("background_pressure_hpa", "hPa", "deep-tropical-mean pressure"),
        ("horizontal_group_velocity", "m s-1", "Kelvin reference horizontal group velocity"),
        ("vertical_group_velocity", "m s-1", "Kelvin reference vertical group velocity"),
        ("ray_slope", "1", "Kelvin reference vertical-to-horizontal ray slope"),
    ):
        dataset[name].attrs.update(units=units, long_name=long_name)
        difference_name = f"{name}_difference_from_control"
        dataset[difference_name] = (
            dataset[name] - dataset[name].sel(simulation="control")
        )
        dataset[difference_name].attrs.update(
            units=units,
            long_name=f"{long_name}; simulation minus control",
        )
    primary = dataset.sel(zonal_wavenumber=1, vertical_wavelength_km=10.0)
    layer = primary.sel(height_m=slice(14_000.0, 20_000.0))
    rows = []
    for simulation in sim_names:
        source = layer.sel(simulation=simulation)
        rows.append(
            {
                "simulation": simulation,
                "height_range_km": "14-20",
                "layer_mean_cgx_m_s": float(
                    source.horizontal_group_velocity.mean(skipna=True)
                ),
                "minimum_cgx_m_s": float(
                    source.horizontal_group_velocity.min(skipna=True)
                ),
                "layer_mean_cgz_m_s": float(
                    source.vertical_group_velocity.mean(skipna=True)
                ),
                "layer_mean_ray_slope": float(source.ray_slope.mean(skipna=True)),
                "calculation_label": (
                    "Kelvin-wave reference; not moist-mode group velocity"
                ),
            }
        )
    return dataset, pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# 1. Wave-filtered Hovmoller diagrams
# -----------------------------------------------------------------------------
# WHAT TO LOOK FOR:
#   * Which filter bands actually reconstruct coherent propagating precipitation?
#   * Does u_nudge increase a particular band, or simply broaden all wave-like
#     variability?
#   * Kelvin/EIG: eastward streaks. IG/Easterly/ER: mostly westward structure.

hovmoller_fields = ("Precip", "PW", "OLR")
hovmoller_stems = [f"wave_filtered_hovmollers_{name.lower()}" for name in hovmoller_fields]
if should_build_figure(1, hovmoller_stems):
    for hov_field, stem in zip(hovmoller_fields, hovmoller_stems):
        fields = {}
        vmax = 0.0
        for i, ds in enumerate(dss):
            for wname in wave_names:
                wav = get_filtered_wave(ds, i, hov_field, wname, avgLat=True)
                fields[(i, wname)] = wav
                finite = np.abs(np.asarray(wav.values, dtype=float))
                finite = finite[np.isfinite(finite)]
                if finite.size == 0:
                    raise RuntimeError(
                        f"{sim_names[i]} {wname} {hov_field}: filtered field "
                        "contains no finite values"
                    )
                vmax = max(vmax, float(np.percentile(finite, 98)))
        if not np.isfinite(vmax) or vmax == 0.0:
            raise RuntimeError(
                f"{hov_field}: all configured filtered fields have zero or "
                "invalid amplitude"
            )
        levels = symmetric_levels(vmax, ndiv=20)
        fig, ax = plt.subplots(
            len(dss), len(wave_names),
            figsize=(3.2 * len(wave_names), 2.3 * len(dss)),
            constrained_layout=True, sharex=True, sharey=True,
        )
        if len(dss) == 1:
            ax = np.array([ax])
        for i in range(len(dss)):
            for j, wname in enumerate(wave_names):
                wav = fields[(i, wname)]
                im = ax[i, j].contourf(
                    wav.longitude, wav.Time, wav.values, levels=levels,
                    cmap=difference_cmap(levels), extend="both",
                )
                if i == 0:
                    ax[i, j].set_title(wave_label(wname))
                if j == 0:
                    ax[i, j].set_ylabel(sim_names[i])
                else:
                    ax[i, j].tick_params(labelleft=False)
                if i == len(dss) - 1:
                    ax[i, j].set_xlabel("Longitude")
                annotate_panel_label(
                    ax[i, j], sim_names[i] if j == 0 else wave_label(wname)
                )
        add_symmetric_colorbar(
            fig, im, ax.ravel(), levels,
            label=f"{hov_field}' ({field_units(hov_field)})",
        )
        figsave.save(fig, stem, number=1)
        print(f"Done with figure: {stem}")


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
#   * The companion fraction figure gives context on how much of the matching
#     latitude-projected anomaly variability lies in each band. Rows using
#     different symmetry projections should not be summed.
#   * Compare precipitation with dynamical fields. If u/-omega/T wave power
#     increases but precipitation power does not, waves are modified aloft but
#     weakly coupled to rainfall. If precipitation power increases without a
#     dynamical increase, convective sensitivity has changed.

wave_metrics = None
if should_run_any([2, WAVE_RESPONSE_FIGURE_NUMBER, 13]):
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
                total_var = anomaly_variance_latmean(
                    ds,
                    field,
                    lat_bounds=filter_lat_bounds,
                    projection=projection,
                )
                if not np.isfinite(total_var) or total_var <= 0:
                    raise RuntimeError(
                        f"{sim_names[i]} {field}: total anomaly variance is "
                        "zero or invalid"
                    )
                wav = get_filtered_wave(ds, i, field, wname, avgLat=True)
                wvar = wave_variance(wav)
                wave_fraction = wvar / total_var
                if not np.isfinite(wave_fraction) or wave_fraction < 0:
                    raise RuntimeError(
                        f"{sim_names[i]} {field} {wname}: invalid filtered "
                        f"power fraction {wave_fraction}"
                    )
                if wave_fraction > 1.01:
                    raise RuntimeError(
                        f"{sim_names[i]} {field} {wname}: filtered power is "
                        f"{100.0 * wave_fraction:.1f}% of the matching "
                        "projected anomaly power. A band-pass reconstruction "
                        "cannot exceed its source variance by this amount; "
                        "regenerate the filtered-wave cache."
                    )
                row = dict(
                    simulation=sim_names[i],
                    simulation_index=i,
                    field=field,
                    wave=wname,
                    wave_label=wave_label(wname),
                    latitude_projection=projection,
                    units=field_units(field),
                    total_variance=total_var,
                    wave_variance=wvar,
                    wave_fraction=wave_fraction,
                )
                wave_metrics = append_cached_rows(
                    "wave_band_variance_metrics",
                    wave_metrics,
                    [row],
                    wave_metrics_signature,
                )
                completed_metric_keys.add(key)
    if len(wave_metrics) > 0:
        # Add nudged/control ratios field-by-field and wave-by-wave.
        wave_metrics = add_wave_metric_ratios(wave_metrics)
        write_cached_table("wave_band_variance_metrics", wave_metrics, wave_metrics_signature)
        print(
            f"Wave-band variance metrics ready: {len(wave_metrics)} rows "
            f"for {wave_metrics['field'].nunique()} fields and "
            f"{wave_metrics['wave'].nunique()} filters"
        )
        if save:
            metrics_output_path = os.path.join(
                product_output_dir, "wave_band_variance_metrics.csv"
            )
            wave_metrics.to_csv(metrics_output_path, index=False)
            print(f"Done with table file: {metrics_output_path}")

if should_build_figure(2, ["wave_power_absolute_and_change", "wave_power_fraction_of_total"]) \
        and wave_metrics is not None and len(wave_metrics) > 0:
    plot_fields = ["Precip", "OLR", "PW", "high_cloud", "u850", "w500", "T100"]
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
        ax[0, j].set_title(field_display_label(field))
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

    # Context figure: how much of the matching latitude-projected anomaly
    # power lies in each wave band. Symmetric/direct and antisymmetric rows use
    # their respective projected totals and should not be summed together.
    metrics_common = add_matching_projection_power_fractions(wave_metrics)
    frac_pivots = []
    for sim in sim_names:
        sub = metrics_common[metrics_common.simulation == sim]
        pivot = sub.pivot_table(
            index="wave",
            columns="field",
            values="wave_fraction_of_projected_total",
            aggfunc="mean",
        )
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
                     label="Share of projected anomaly power (%)")
        fig.suptitle(
            "Wave-band share of matching projected anomaly power"
        )
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
            waves=list(lag_wave_names),
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

    for wname in lag_wave_names:
        projection = lat_projection_for_wave(wname)
        for target in lag_target_fields:
            for i, ds in enumerate(dss):
                key = (sim_names[i], wname, target)
                if key in completed_lag_keys:
                    continue
                wav = get_filtered_wave(
                    ds, i, "Precip", wname, avgLat=True
                )
                amp = wave_amplitude_index(wav)
                ts = area_mean_filter_band(
                    ds,
                    target,
                    lat_bounds=filter_lat_bounds,
                    projection=projection,
                )
                amp, ts = xr.align(amp, ts, join="inner")
                if amp.sizes["Time"] < 3:
                    raise RuntimeError(
                        f"{sim_names[i]} {wname} {target}: fewer than three "
                        "common times for lag correlation"
                    )
                dt_days_here = get_dt_days_from_time(amp)
                max_steps = max(1, int(round(lag_days_max / dt_days_here)))
                lags, corr = lag_corr(
                    amp.values, ts.values, max_lag_steps=max_steps
                )
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

    fig, ax = plt.subplots(len(lag_wave_names), len(lag_target_fields),
                           figsize=(3.6 * len(lag_target_fields), 2.3 * len(lag_wave_names)),
                           constrained_layout=True, sharex=True, sharey=True)
    if len(lag_wave_names) == 1:
        ax = ax[None, :]
    if len(lag_target_fields) == 1:
        ax = ax[:, None]

    for wj, wname in enumerate(lag_wave_names):
        for tj, target in enumerate(lag_target_fields):
            for i, ds in enumerate(dss):
                s = lag_df[
                    (lag_df.simulation == sim_names[i])
                    & (lag_df.wave == wname)
                    & (lag_df.target == target)
                ].sort_values("lag_days")
                if len(s) == 0:
                    raise RuntimeError(
                        f"Missing lag-correlation output for {sim_names[i]} "
                        f"{wname} {target}"
                    )
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
            if wj == len(lag_wave_names) - 1:
                ax[wj, tj].set_xlabel("Lag (days; + means wave leads)")
    ax[0, 0].legend(loc="best")
    figsave.save(fig, "wave_amplitude_lag_correlations", number=3)
    if save and len(lag_df) > 0:
        lag_df.to_csv(
            os.path.join(product_output_dir, "wave_amplitude_lag_correlations.csv"),
            index=False,
        )
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
                wav = get_filtered_wave(
                    ds, i, "Precip", wname, avgLat=True
                )
                # Local wave amplitude in longitude-time, normalised by its std.
                a = np.abs(wav)
                a_std = float(
                    a.std(dim=("Time", "longitude"), skipna=True).values
                )
                if not np.isfinite(a_std) or a_std <= 0:
                    raise RuntimeError(
                        f"{sim_names[i]} {wname}: wave amplitude has zero or "
                        "invalid variance"
                    )
                a_norm = a / a_std
                y = latmean_for_filter_field(
                    ds,
                    target,
                    lat_bounds=filter_lat_bounds,
                    projection=projection,
                )
                a_norm, y = xr.align(a_norm, y, join="inner")
                if a_norm.sizes["Time"] == 0:
                    raise RuntimeError(
                        f"{sim_names[i]} {wname} {target}: no common times "
                        "for conditional composite"
                    )
                xc, ym, y25, y75, count = binned_mean_1d(
                    a_norm.values, y.values, amp_bins, min_count=20
                )
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
                    for xx, yy, lo, hi, cnum in zip(
                        xc, ym, y25, y75, count
                    )
                ]
                cond_df = append_cached_rows(
                    "wave_conditioned_precip_olr_pw",
                    cond_df,
                    rows,
                    cond_signature,
                )
                completed_cond_keys.add(key)

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
                    raise RuntimeError(
                        f"Missing conditional-response output for "
                        f"{sim_names[i]} {wname} {target}"
                    )
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
        cond_df.to_csv(
            os.path.join(product_output_dir, "wave_conditioned_precip_olr_pw.csv"),
            index=False,
        )
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
            attrib.to_csv(
                os.path.join(product_output_dir, "intense_precip_wave_attribution.csv"),
                index=False,
            )

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
        required_columns=("simulation", "simulation_index", "wave", "field", "vertical_coord", "wave_variance", "sensitivity_subset"),
    )
    if prof is None:
        prof = pd.DataFrame()
    completed_profile_keys = table_key_set(
        prof,
        ("simulation_index", "wave", "field", "vertical_coord", "sensitivity_subset"),
    )

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
                    subset_names = (
                        ("full_record", "first_half", "second_half")
                        if wname == EASTWARD_INTRASEASONAL_FILTER
                        else ("full_record",)
                    )
                    if all(
                        (i, wname, var, level, subset_name)
                        in completed_profile_keys
                        for subset_name in subset_names
                    ):
                        continue
                    wav = get_filtered_wave(
                        ds, i, field, wname, avgLat=True
                    )
                    masks = record_subsets(
                        wav.Time.values,
                        include_halves=(
                            wname == EASTWARD_INTRASEASONAL_FILTER
                        ),
                    )
                    rows = []
                    for subset_name, mask in masks.items():
                        key = (i, wname, var, level, subset_name)
                        if key in completed_profile_keys:
                            continue
                        selected = wav.isel(Time=np.flatnonzero(mask))
                        resolved = resolved_frequency_count(
                            selected.sizes["Time"],
                            dt,
                            WAVE_BOXES[wname]["period"],
                        )
                        rows.append(
                            dict(
                                simulation=sim_names[i],
                                simulation_index=i,
                                wave=wname,
                                wave_label=wave_label(wname),
                                latitude_projection=projection,
                                field=var,
                                vertical_coord=level,
                                vertical_axis=vertical_axis,
                                vertical_units=vertical_units,
                                sensitivity_subset=subset_name,
                                resolved_fourier_frequency_count=resolved,
                                status=(
                                    "ok"
                                    if resolved > 0
                                    else "insufficient_record_for_filter"
                                ),
                                source_field=(
                                    "omg_isobaric" if var == "w" else field
                                ),
                                sign_convention=(
                                    "positive ascent (-omega)"
                                    if var == "w"
                                    else ""
                                ),
                                wave_variance=(
                                    wave_variance(selected)
                                    if resolved > 0
                                    else np.nan
                                ),
                            )
                        )
                        completed_profile_keys.add(key)
                    prof = append_cached_rows(
                        "vertical_wave_power_profiles",
                        prof,
                        rows,
                        profile_signature,
                    )
    if len(prof) > 0:
        write_cached_table("vertical_wave_power_profiles", prof, profile_signature)
        if save:
            prof.to_csv(
                os.path.join(product_output_dir, "vertical_wave_power_profiles.csv"),
                index=False,
            )

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
                    sub = prof[
                        (prof.simulation == sim)
                        & (prof.wave == wname)
                        & (prof.field == var)
                        & (prof.sensitivity_subset == "full_record")
                    ].sort_values("vertical_coord")
                    if len(sub) == 0:
                        raise RuntimeError(
                            f"Missing vertical wave profile for {sim} "
                            f"{wname} {var}"
                        )
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
                    variable_label = field_display_label(var)
                    ylabel = (
                        f"{variable_label}\nHeight (km)"
                        if var == "cldfrac"
                        else f"{variable_label}\nPressure (hPa)"
                    )
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
    for wname in critical_level_wave_names:
        cmin, ccen, cmax = wave_phase_speed_range(wname)
        speed_rows.append(dict(wave=wname, wave_label=wave_label(wname),
                               latitude_projection=lat_projection_for_wave(wname),
                               c_min_ms=cmin, c_central_ms=ccen, c_max_ms=cmax,
                               k_range=str(WAVE_BOXES[wname]["k"]), period_range_days=str(WAVE_BOXES[wname]["period"])))
    speed_df = pd.DataFrame(speed_rows)
    print("\nWave-box phase-speed estimates")
    print(speed_df.to_string(index=False))
    if save:
        speed_df.to_csv(
            os.path.join(product_output_dir, "wave_box_phase_speed_estimates.csv"),
            index=False,
        )

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
        u = cosine_latitude_mean(
            zds.uzonal_isobaric.sel(
                latitude=slice(filter_lat_bounds[0], filter_lat_bounds[1])
            ),
            ("Time", "latitude"),
        )
        a.plot(u.values, u.pres.values, color=cols[i], lw=1.8, label=sim_names[i])
        for wname in critical_level_wave_names:
            # This simulation's power-weighted mean phase speed for the band,
            # colour-matched to the distribution below. Fall back to the
            c_mean = power_weighted_mean_phase_speed(i, wname)
            if not np.isfinite(c_mean):
                raise RuntimeError(
                    f"Missing finite power-weighted phase speed for "
                    f"{sim_names[i]} {wname}"
                )
            a.axvline(c_mean, color=spectrum_outline_colors.get(wname, "0.5"), lw=1.0, ls="--", alpha=0.8)
        a.axvline(0, color="0.7", lw=0.6)
        a.set_title(sim_names[i])
        a.set_yscale("log")
        a.set_ylim(1000, 50)
        a.grid(alpha=0.2)
        annotate_panel_label(a, sim_names[i])

        b = dist_ax[i]
        sim_dist = phase_speed_dist[phase_speed_dist.simulation_index == i] if len(phase_speed_dist) else pd.DataFrame()
        for wname in critical_level_wave_names:
            sub = sim_dist[sim_dist.wave == wname].sort_values("phase_speed_ms") if len(sim_dist) else pd.DataFrame()
            if len(sub) == 0 or not np.any(np.isfinite(sub.density_per_ms.values)):
                raise RuntimeError(
                    f"Missing finite phase-speed distribution for "
                    f"{sim_names[i]} {wname}"
                )
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
        f"Deep-tropical mean wind profile with power-weighted mean phase speeds (dashed) and "
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
                uw = get_filtered_wave(
                    ds,
                    i,
                    f"u{plev}",
                    wname,
                    avgLat=False,
                    lat_bounds=eddy_flux_lat_bounds,
                )
                vw = get_filtered_wave(
                    ds,
                    i,
                    f"v{plev}",
                    wname,
                    avgLat=False,
                    lat_bounds=eddy_flux_lat_bounds,
                )
                Tw = get_filtered_wave(
                    ds,
                    i,
                    f"T{plev}",
                    wname,
                    avgLat=False,
                    lat_bounds=eddy_flux_lat_bounds,
                )
                uw, vw, Tw = xr.align(uw, vw, Tw, join="exact")
                uv = (uw * vw).mean(
                    dim=("Time", "longitude"), skipna=True
                )
                vT = (vw * Tw).mean(
                    dim=("Time", "longitude"), skipna=True
                )
                eke = 0.5 * (uw**2 + vw**2).mean(
                    dim=("Time", "longitude"), skipna=True
                )
                rows = []
                for lat, uvv, vTv, ekev in zip(
                    uv.latitude.values, uv.values, vT.values, eke.values
                ):
                    rows.append(
                        dict(
                            simulation=sim_names[i],
                            wave=wname,
                            wave_label=wave_label(wname),
                            pres=plev,
                            latitude_projection="latitude_resolved",
                            latitude=lat,
                            uv=uvv,
                            vT=vTv,
                            eke=ekev,
                        )
                    )
                flux = append_cached_rows(
                    "wave_filtered_eddy_fluxes",
                    flux,
                    rows,
                    flux_signature,
                )
                completed_flux_keys.add(key)
    if len(flux) > 0:
        write_cached_table("wave_filtered_eddy_fluxes", flux, flux_signature)
        if save:
            flux.to_csv(
                os.path.join(product_output_dir, "wave_filtered_eddy_fluxes.csv"),
                index=False,
            )
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


def plot_frequency_wavenumber_spectra(
    spectra_by_sim,
    field_title,
    output_stem,
    spectrum_cmap,
    spectrum_norm,
):
    """Plot the standard Figure-9 frequency-versus-wavenumber presentation."""
    components = spectrum_components
    fig, ax = plt.subplots(
        len(dss),
        2,
        figsize=(9.0, 2.8 * len(dss)),
        constrained_layout=True,
        sharex=True,
        sharey=True,
    )
    if len(dss) == 1:
        ax = np.array([ax])

    spec_im = None
    for i, spectra in enumerate(spectra_by_sim):
        for cj, component in enumerate(components):
            if component not in spectra:
                raise RuntimeError(
                    f"{sim_names[i]} spectrum lacks configured component "
                    f"{component!r}"
                )
            spectrum = spectra[component].transpose("wavenumber", "frequency")
            wn = spectrum.wavenumber.values
            freq = spectrum.frequency.values
            spec_im = ax[i, cj].pcolormesh(
                wn,
                freq,
                spectrum.values.T,
                cmap=spectrum_cmap,
                norm=spectrum_norm,
                shading="auto",
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
                ax[i, cj].set_ylabel(
                    f"{sim_names[i]}\nFrequency (day$^{{-1}}$)"
                )
            else:
                ax[i, cj].tick_params(labelleft=False)
            if i == len(dss) - 1:
                ax[i, cj].set_xlabel("Zonal wavenumber")
            annotate_panel_label(ax[i, cj], sim_names[i])

    if spec_im is None:
        return
    fig.colorbar(
        spec_im,
        ax=ax.ravel(),
        location="right",
        shrink=0.85,
        label="Normalised power",
    )
    fig.legend(
        handles=wave_filter_region_legend_handles(),
        loc="outside lower center",
        ncol=min(4, len(wave_names)),
        title="Filter regions",
    )
    fig.suptitle(
        f"{field_title} symmetric/antisymmetric spectra over "
        f"{spectrum_lat_bounds[0]}--{spectrum_lat_bounds[1]}°"
    )
    figsave.save(fig, output_stem, number=9)
    print(f"Done with figure: {output_stem}")


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
        *[field[2] for field in spectrum_additional_fields],
    ],
):
    spec_levels = np.linspace(0, 3.0, 26)
    spec_cmap, spec_norm = make_spectrum_cmap(spec_levels)

    components = spectrum_components
    spectra_by_sim = load_or_compute_normalised_spectra_by_sim()
    plot_frequency_wavenumber_spectra(
        spectra_by_sim,
        spectrum_field,
        "symmetric_antisymmetric_wave_spectra",
        spec_cmap,
        spec_norm,
    )
    for field_name, field_title, output_stem in spectrum_additional_fields:
        field_spectra_by_sim = load_or_compute_normalised_spectra_by_sim(
            field_name
        )
        plot_frequency_wavenumber_spectra(
            field_spectra_by_sim,
            field_title,
            output_stem,
            spec_cmap,
            spec_norm,
        )

    period_fig, period_ax = plt.subplots(len(dss), 2, figsize=(9.0, 2.8 * len(dss)),
                                         constrained_layout=True, sharex=True, sharey=True)
    if len(dss) == 1:
        period_ax = np.array([period_ax])

    period_im = None
    period_min = 1.0 / spectrum_frequency_limits[1]
    for i, spectra in enumerate(spectra_by_sim):
        for cj, component in enumerate(components):
            if component not in spectra:
                raise RuntimeError(
                    f"{sim_names[i]} spectrum lacks configured component "
                    f"{component!r}"
                )
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
        for cj, component in enumerate(components):
            if component not in spectra:
                raise RuntimeError(
                    f"{sim_names[i]} spectrum lacks configured component "
                    f"{component!r}"
                )
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
            dist.to_csv(
                os.path.join(product_output_dir, "wave_phase_speed_power_distributions.csv"),
                index=False,
            )

        ncols = min(3, len(critical_level_wave_names))
        nrows = int(np.ceil(len(critical_level_wave_names) / ncols))
        fig, ax = plt.subplots(
            nrows, ncols,
            figsize=(4.0 * ncols, 2.8 * nrows),
            constrained_layout=True,
            sharey=True,
        )
        axes = np.asarray(ax).reshape(-1)
        for wj, wname in enumerate(critical_level_wave_names):
            a = axes[wj]
            wave_dist = dist[dist.wave == wname]
            for i, sim in enumerate(sim_names):
                sub = wave_dist[wave_dist.simulation == sim].sort_values("phase_speed_ms")
                if len(sub) == 0 or not np.any(np.isfinite(sub.density_per_ms.values)):
                    raise RuntimeError(
                        f"Missing finite phase-speed distribution for "
                        f"{sim} {wname}"
                    )
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
        for a in axes[len(critical_level_wave_names):]:
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
# 11. Wave precipitation amplitude and cloud/OLR response efficiency
# -----------------------------------------------------------------------------
# Rainfall amplitude, high-cloud production per unit rainfall, and TOA
# longwave response per unit rainfall are separate physical quantities. Cloud
# and OLR responses are filtered in exactly the same band as precipitation.
# P_f is filtered precipitation. H_f,c is filtered cloud-class occurrence.
# L_f is filtered outgoing longwave radiation.

wave_response_cloud_scatter_stems = {
    response_key: (
        "wave_precipitation_high_cloud_scatter"
        if response_key == "total_high_cloud"
        else f"wave_precipitation_{response_key}_scatter"
    )
    for response_key, _response_label, _field_name
    in WAVE_RESPONSE_CLOUD_FIELDS
}
wave_response_figure_stems = [
    "wave_precipitation_cloud_olr_response",
    "wave_precipitation_cloud_olr_response_differences",
    "wave_precipitation_cloud_class_efficiencies",
    "wave_precipitation_cloud_class_efficiency_differences",
    *wave_response_cloud_scatter_stems.values(),
    "wave_precipitation_olr_scatter",
]
wave_response_signature = cache_signature(
    "wave_response_efficiency_metrics",
    extra={
        "product_version": WAVE_RESPONSE_PRODUCT_VERSION,
        "wave_metrics_version": WAVE_METRICS_VERSION,
        "waves": list(WAVE_RESPONSE_FILTERS),
        "cloud_responses": [
            response[0] for response in WAVE_RESPONSE_CLOUD_FIELDS
        ],
        "predictor": "same-band filtered precipitation",
        "toa_longwave_sign": (
            "positive slope means OLR reduction per unit precipitation"
        ),
        "bootstrap_method": (
            "paired circular moving-block bootstrap of complete longitude "
            "fields"
        ),
        "bootstrap_resamples": WAVE_RESPONSE_BOOTSTRAP_RESAMPLES,
        "bootstrap_seed": WAVE_RESPONSE_BOOTSTRAP_SEED,
        "bootstrap_ci_percentiles": list(WAVE_RESPONSE_CI_PERCENTILES),
        "bootstrap_block_length": (
            "geometric midpoint of each wave period range"
        ),
        "bootstrap_reference": WAVE_RESPONSE_BOOTSTRAP_REFERENCE,
        "scatter_sample_count": WAVE_RESPONSE_SCATTER_SAMPLE,
        "eastward_intraseasonal_subsets": [
            "full_record",
            "first_half",
            "second_half",
        ],
    },
)
if should_build_versioned_figure(
    WAVE_RESPONSE_FIGURE_NUMBER,
    wave_response_figure_stems,
    wave_response_signature,
):
    wave_response = load_cached_table(
        "wave_response_efficiency_metrics",
        wave_response_signature,
        output_filename="wave_response_efficiency_metrics.csv",
        required_columns=(
            "simulation",
            "wave",
            "response_variable",
            "precipitation_rms_mm_day",
            "regression_slope",
            "sensitivity_subset",
            "regression_slope_ci_low",
            "regression_slope_ci_high",
        ),
    )
    wave_response_contrasts = load_cached_table(
        "wave_response_efficiency_contrasts",
        wave_response_signature,
        output_filename="wave_response_efficiency_contrasts.csv",
        required_columns=(
            "wave",
            "response_variable",
            "contrast_key",
            "metric",
            "response_change",
            "ci_low",
            "ci_high",
        ),
    )
    if wave_response is None or wave_response_contrasts is None:
        wave_response = compute_wave_response_metric_table()
        wave_response_contrasts = wave_response_direct_contrasts(
            wave_response
        )
        write_cached_table(
            "wave_response_efficiency_metrics",
            wave_response,
            wave_response_signature,
        )
        write_cached_table(
            "wave_response_efficiency_contrasts",
            wave_response_contrasts,
            wave_response_signature,
        )
    if save:
        wave_response.to_csv(
            wave_response_metrics_product_path(test=test_mode),
            index=False,
        )
        wave_response_contrasts.to_csv(
            wave_response_contrasts_product_path(test=test_mode),
            index=False,
        )
    plot_wave_response_metrics(wave_response, wave_response_contrasts)
    plot_wave_cloud_class_efficiencies(
        wave_response, wave_response_contrasts
    )
    for response_key, response_label, field_name in (
        WAVE_RESPONSE_CLOUD_FIELDS
    ):
        _plot_wave_response_scatter(
            wave_response,
            field_name,
            response_key,
            "cloud",
            wave_response_cloud_scatter_stems[response_key],
            response_label=response_label,
        )
    _plot_wave_response_scatter(
        wave_response,
        "OLR",
        "toa_longwave_response",
        "toa_longwave",
        "wave_precipitation_olr_scatter",
    )
    write_figure_output_metadata(
        WAVE_RESPONSE_FIGURE_NUMBER,
        wave_response_figure_stems,
        wave_response_signature,
    )
    print("Done with figure: wave precipitation/cloud/OLR response")


# -----------------------------------------------------------------------------
# 12. Sakaeda/Inoue two-EOF top-heaviness
# -----------------------------------------------------------------------------

if should_build_figure(12, "two_eof_top_heaviness"):
    top_signature = cache_signature(
        "two_eof_top_heaviness",
        extra={
            "wave_metrics_version": WAVE_METRICS_VERSION,
            "waves": list(TOP_HEAVINESS_FILTERS),
            "pressure_hpa": top_heaviness_pressure_hpa.tolist(),
            "basis_sources": ["unfiltered_control", "equally_weighted_pooled"],
            "vertical_motion": "native MPAS -omg_isobaric only",
        },
    )
    top_table = load_cached_table(
        "two_eof_top_heaviness",
        top_signature,
        output_filename="two_eof_top_heaviness_metrics.csv",
        required_columns=(
            "simulation", "wave", "tau_r", "tau_i", "pc1_power",
            "pc2_power", "eof_basis_source", "sensitivity_subset",
        ),
    )
    basis_dataset = load_cached_dataset(
        "vertical_motion_eof_basis", top_signature
    )
    if top_table is None or basis_dataset is None:
        top_table, basis_dataset = compute_top_heaviness_products()
        write_cached_table("two_eof_top_heaviness", top_table, top_signature)
        write_cached_dataset(
            "vertical_motion_eof_basis", basis_dataset, top_signature
        )
    if save:
        top_table.to_csv(
            os.path.join(product_output_dir, "two_eof_top_heaviness_metrics.csv"),
            index=False,
        )
        netcdf_safe_dataset(basis_dataset).to_netcdf(
            os.path.join(product_output_dir, "vertical_motion_eof_basis.nc")
        )

    primary = top_table[
        (top_table.eof_basis_source == "unfiltered_control")
        & (top_table.sensitivity_subset == "full_record")
    ]
    fig, axes = plt.subplots(1, 4, figsize=(16.0, 4.2), constrained_layout=True)
    x = np.arange(len(TOP_HEAVINESS_FILTERS))
    width = 0.8 / len(sim_names)
    for simulation_index, (simulation, color) in enumerate(zip(sim_names, cols)):
        selected = (
            primary[primary.simulation == simulation]
            .set_index("wave").reindex(TOP_HEAVINESS_FILTERS)
        )
        if selected[["tau_r", "tau_i"]].isna().any().any():
            raise RuntimeError(
                f"Top-heaviness output is incomplete for {simulation}; "
                f"required waves are {TOP_HEAVINESS_FILTERS}"
            )
        offset = (simulation_index - 1) * width
        axes[0].bar(x + offset, selected.tau_r, width, color=color, label=simulation)
        axes[1].bar(x + offset, selected.tau_i, width, color=color)
    for axis, title, ylabel in (
        (axes[0], r"Top-heaviness $\tau_r$", r"$\Re(\tau)$"),
        (axes[1], r"Vertical tilt $\tau_i$", r"$\Im(\tau)$"),
    ):
        axis.set_title(title)
        axis.set_xticks(x)
        axis.set_xticklabels(
            [wave_label(name) for name in TOP_HEAVINESS_FILTERS],
            rotation=28, ha="right",
        )
        axis.set_ylabel(ylabel)
        axis.axhline(0.0, color="0.55", lw=0.8)
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(frameon=False, fontsize=8)

    basis = basis_dataset.sel(basis_source="unfiltered_control")
    for eof_index, linestyle in ((1, "-"), (2, "--")):
        explained_variance = float(
            basis.explained_variance.sel(eof_number=eof_index).values
        )
        axes[2].plot(
            basis.eof.sel(eof_number=eof_index),
            basis.pressure_hpa,
            linestyle,
            label=(
                f"EOF{eof_index} "
                f"({100 * explained_variance:.1f}%)"
            ),
        )
    axes[2].set(
        title="Common control EOF basis",
        xlabel="Normalised structure",
        ylabel="Pressure (hPa)",
        yscale="log",
        ylim=(1000, 100),
    )
    axes[2].legend(frameon=False, fontsize=8)
    axes[2].grid(alpha=0.2)

    for simulation, color in zip(sim_names, cols):
        selected = primary[primary.simulation == simulation]
        for wave_name, linestyle in zip(
            TOP_HEAVINESS_FILTERS,
            ("-", "--", "-.", ":", (0, (3, 1, 1, 1))),
            strict=True,
        ):
            row = selected[selected.wave == wave_name]
            if len(row) != 1 or not np.isfinite(row.iloc[0].tau_r):
                raise RuntimeError(
                    f"Top-heaviness reconstruction requires one finite "
                    f"{simulation} {wave_name} row; found {len(row)}"
                )
            reconstruction = (
                basis.eof.sel(eof_number=1)
                + row.iloc[0].tau_r * basis.eof.sel(eof_number=2)
            )
            axes[3].plot(
                reconstruction, basis.pressure_hpa,
                color=color, linestyle=linestyle, alpha=0.8,
                label=f"{simulation}, {wave_label(wave_name)}",
            )
    axes[3].set(
        title="Reconstructed ascent structure",
        xlabel=r"EOF1 + $\tau_r$ EOF2",
        yscale="log",
        ylim=(1000, 100),
    )
    axes[3].grid(alpha=0.2)
    axes[3].legend(frameon=False, fontsize=6, ncol=2)
    fig.suptitle(
        "Two-EOF pressure-velocity top-heaviness; one physically oriented basis"
    )
    figsave.save(fig, "two_eof_top_heaviness", number=12)
    print("Done with figure: two_eof_top_heaviness")


# -----------------------------------------------------------------------------
# 13. Lim-Son Kelvin-wave reference energy dispersion
# -----------------------------------------------------------------------------

if should_build_figure(13, "kelvin_reference_energy_dispersion"):
    kelvin_signature = cache_signature(
        "kelvin_reference_energy_dispersion",
        extra={
            "wave_metrics_version": WAVE_METRICS_VERSION,
            "k_primary_m-1": kelvin_reference_k_m1,
            "m_primary_m-1": kelvin_reference_m_m1,
            "zonal_wavenumbers": list(kelvin_sensitivity_wavenumbers),
            "vertical_wavelengths_km": list(
                kelvin_sensitivity_vertical_wavelengths_km
            ),
            "latitude_bounds": list(filter_lat_bounds),
        },
    )
    kelvin_dataset = load_cached_dataset(
        "kelvin_reference_energy_dispersion", kelvin_signature
    )
    kelvin_summary = load_cached_table(
        "kelvin_reference_energy_dispersion_summary",
        kelvin_signature,
        output_filename="kelvin_reference_energy_dispersion_summary.csv",
        required_columns=(
            "simulation", "layer_mean_cgx_m_s", "minimum_cgx_m_s",
            "layer_mean_cgz_m_s", "layer_mean_ray_slope",
            "Kelvin_T100_rms", "Kelvin_high_cloud_rms",
            f"{EASTWARD_INTRASEASONAL_FILTER}_T100_rms",
            f"{EASTWARD_INTRASEASONAL_FILTER}_high_cloud_rms",
        ),
    )
    if kelvin_dataset is None or kelvin_summary is None:
        kelvin_dataset, kelvin_summary = compute_kelvin_dispersion_products()
        if wave_metrics is None or wave_metrics.empty:
            raise RuntimeError(
                "Kelvin dispersion context requires the current wave-band "
                "variance metrics"
            )
        for contextual_wave in ("Kelvin", EASTWARD_INTRASEASONAL_FILTER):
            for contextual_field in ("T100", "high_cloud"):
                values = {}
                for simulation in sim_names:
                    match = wave_metrics[
                        (wave_metrics.simulation == simulation)
                        & (wave_metrics.wave == contextual_wave)
                        & (wave_metrics.field == contextual_field)
                    ]
                    if len(match) != 1:
                        raise RuntimeError(
                            "Kelvin dispersion context requires exactly one "
                            f"{simulation} {contextual_wave} "
                            f"{contextual_field} metric; found {len(match)}"
                        )
                    variance = float(match.iloc[0].wave_variance)
                    if not np.isfinite(variance) or variance < 0.0:
                        raise RuntimeError(
                            f"Invalid wave variance for {simulation} "
                            f"{contextual_wave} {contextual_field}: {variance}"
                        )
                    values[simulation] = np.sqrt(variance)
                kelvin_summary[
                    f"{contextual_wave}_{contextual_field}_rms"
                ] = kelvin_summary.simulation.map(values)
        write_cached_dataset(
            "kelvin_reference_energy_dispersion",
            kelvin_dataset, kelvin_signature,
        )
        write_cached_table(
            "kelvin_reference_energy_dispersion_summary",
            kelvin_summary, kelvin_signature,
        )
    if save:
        netcdf_safe_dataset(kelvin_dataset).to_netcdf(
            os.path.join(product_output_dir, "kelvin_reference_energy_dispersion.nc")
        )
        kelvin_summary.to_csv(
            os.path.join(
                product_output_dir, "kelvin_reference_energy_dispersion_summary.csv"
            ),
            index=False,
        )

    variables = (
        ("background_zonal_wind", r"$U$ (m s$^{-1}$)"),
        ("static_stability", r"$N^2$ (s$^{-2}$)"),
        ("horizontal_group_velocity", r"$C_{gx}$ (m s$^{-1}$)"),
        ("vertical_group_velocity", r"$C_{gz}$ (m s$^{-1}$)"),
        ("ray_slope", r"$C_{gz}/C_{gx}$"),
    )
    fig, axes = plt.subplots(
        2, len(variables), figsize=(15.5, 7.2),
        constrained_layout=True, sharey=True,
    )
    for row, zonal_wavenumber in enumerate((1, 2)):
        selected = kelvin_dataset.sel(
            zonal_wavenumber=zonal_wavenumber,
            vertical_wavelength_km=10.0,
        )
        for column, (variable, label) in enumerate(variables):
            for simulation, color in zip(sim_names, cols):
                source = selected.sel(simulation=simulation)
                axes[row, column].plot(
                    source[variable], source.background_pressure_hpa,
                    color=color, label=simulation,
                )
            axes[row, column].axvline(0.0, color="0.6", lw=0.8)
            axes[row, column].grid(alpha=0.2)
            axes[row, column].set_xlabel(label)
            axes[row, column].set_yscale("log")
            axes[row, column].set_ylim(1000.0, 50.0)
            if row == 0:
                axes[row, column].set_title(label)
        axes[row, 0].set_ylabel(
            f"k={zonal_wavenumber}\nPressure (hPa)"
        )
    axes[0, 0].legend(frameon=False, fontsize=8)
    latitude_title = (
        f"{abs(filter_lat_bounds[0]):g}°S–{filter_lat_bounds[1]:g}°N"
    )
    fig.suptitle(
        "Kelvin-wave reference energy dispersion; k=1 and 2; "
        f"10-km vertical wavelength; {latitude_title}"
    )
    figsave.save(fig, "kelvin_reference_energy_dispersion", number=13)
    print("Done with figure: kelvin_reference_energy_dispersion")


# -----------------------------------------------------------------------------
# 14. Compact LPO-following sections for the lifecycle diagnostic
# -----------------------------------------------------------------------------
# This numbered action intentionally writes no figure. It is an explicit,
# restartable prerequisite: expensive filtering remains owned by this script,
# while lifecycle compositing and figure refinements read only these compact
# event-level stencils.

if 14 in figures_to_run:
    tracking_steps = _lpo_wave_tracking_steps(
        test=test_mode,
        allow_stale_products=runtime_args.allow_stale_lpo_products,
    )
    lifecycle_cache_fields = ("Precip", "PW", "OLR", "T100", "u100")
    for simulation_index, (simulation, dataset) in enumerate(
        zip(LPO_SIMULATION_DIRS, dss)
    ):
        progress(
            f"{simulation}: validating lifecycle filtered-wave caches"
        )
        for wave_name in LPO_LIFECYCLE_FILTERS:
            for field_name in lifecycle_cache_fields:
                get_filtered_wave(
                    dataset,
                    simulation_index,
                    field_name,
                    wave_name,
                    avgLat=True,
                )
        output_path = lpo_wave_section_cache_path(
            simulation, test=test_mode
        )
        reuse = False
        if os.path.exists(output_path) and not force_recompute_caches:
            try:
                with xr.open_dataset(output_path) as existing:
                    reuse = (
                        existing.attrs.get("cache_version")
                        == LPO_WAVE_SECTION_CACHE_VERSION
                        and existing.attrs.get("tracking_product_version")
                        == LPO_TRACKING_PRODUCT_VERSION
                        and existing.attrs.get("source_simulation")
                        == simulation
                        and int(existing.attrs.get("test_subset", -1))
                        == int(test_mode)
                        and "relative_longitude" in existing.sizes
                        and "wave_phase_number" in existing
                        and "MRG" in set(
                            existing.wave_band.values.astype(str)
                        )
                        and "filtered_static_stability" in existing
                        and "ray_filtered_temperature" in existing
                    )
            except (OSError, ValueError):
                reuse = False
        if reuse:
            progress(f"Reusing compact LPO wave sections: {output_path}")
            continue
        sections = build_compact_lpo_wave_sections(
            dataset,
            simulation,
            tracking_steps[simulation],
        )
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        temporary_path = output_path + ".tmp"
        netcdf_safe_dataset(sections).to_netcdf(temporary_path)
        os.replace(temporary_path, output_path)
        progress(f"Wrote compact LPO wave sections: {output_path}")

print("\nFinished equatorial_wave_diagnostics.py")
