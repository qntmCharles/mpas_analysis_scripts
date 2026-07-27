#!/usr/bin/env python3
"""Large-Scale Precipitation Tracking (LPT) for MPAS convective organisation.

This module is the stable project-side interface to the published LPT code.
Object identification, overlap linking, and merger/split handling are delegated
to the unmodified upstream driver. MPAS preprocessing, scientific sample
selection, lifecycle diagnostics, standardised outputs, matching, and figures
remain here.

The exact upstream revision is pinned in
``external/lpt-python-public.lock.json``. Install it as a separate checkout and
set ``MPAS_LPT_SOURCE`` to that directory::

    git clone https://github.com/uwincm/lpt-python-public.git /path/to/lpt
    git -C /path/to/lpt checkout be94a3e7afbd107b2cd852fcf166493e1c3fe472
    export MPAS_LPT_SOURCE=/path/to/lpt

The adapter rejects a different commit or modified tracked source. Original
upstream object, system, composite-mask, and individual-mask outputs are
preserved below each run's ``upstream_lpt`` directory.

Publication mode is the default. It applies one configuration to all three
simulations: six-hourly precipitation, a complete 72-hour running mean,
Gaussian sigma 5 degrees, 15 mm/day threshold, 300,000 km2 minimum area,
periodic longitude, overlap linking with merger/split families, and seven-day
minimum duration. Validation mode separately runs the controlled threshold,
sampling-phase, and record-half sensitivities.

The ``standard`` directories are the downstream interface. They contain event
and track-step tables, per-event masks, fixed-lag diagnostics, lifecycle
summaries, equal-event composites, and complete configuration metadata.
Figures are regenerated exclusively from these products without rerunning
tracking. ``convective_lifecycle_composite_diagnostics.py``,
``convective_pathway_diagnostics.py``, and the LPT section of
``significance_bootstrap_diagnostics.py`` consume this interface.

Method citation: Kerns and Chen (2020), JGR Atmospheres,
doi:10.1029/2020JD032373; https://github.com/uwincm/lpt-python-public.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import dataclasses
import hashlib
import importlib
import json
import os
import subprocess
import sys
import types
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
for _parent in (
    _THIS_DIR.parent,
    _THIS_DIR.parent.parent,
    _THIS_DIR.parent.parent.parent,
):
    if str(_parent) not in sys.path:
        sys.path.append(str(_parent))

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt
from scipy import ndimage

from diagnostics.diagnostic_helpers import (
    LATLON_PATTERN,
    diagnostic_cache_dir,
    diagnostic_output_dir,
    grid_cell_area_km2,
    load_latlon_dataset,
    precipitation_rate_from_dataset,
    standardize_latlon_dataset,
)


UPSTREAM_REPOSITORY = "https://github.com/uwincm/lpt-python-public.git"
UPSTREAM_COMMIT = "be94a3e7afbd107b2cd852fcf166493e1c3fe472"
UPSTREAM_CITATION = "Kerns and Chen (2020), doi:10.1029/2020JD032373"
PRODUCT_VERSION = "mpas-lpt-precipitation-envelope-v1"

SIMULATION_DIRS = {
    "control": "control_new",
    "nudged-u": "u_nudge",
    "nudged-T": "T_nudge",
}
SIMULATION_LABELS = {
    "control": "control",
    "nudged-u": r"$u$-nudged",
    "nudged-T": r"$T$-nudged",
}
SIMULATION_COLOURS = {
    "control": "black",
    "nudged-u": "tab:blue",
    "nudged-T": "tab:red",
}

TRACKING_FIGURES = (1, 2, 3)
LIFECYCLE_FIGURES = (4, 5, 6, 7)
PUBLICATION_FIGURES = TRACKING_FIGURES + LIFECYCLE_FIGURES
LIFECYCLE_LAGS_HOURS = tuple(range(-72, 73, 6))
TROPICAL_CENTROID_LIMIT_DEG = 15.0
TROPICAL_MIN_CONSECUTIVE_HOURS = 24.0
DEFAULT_ROOT = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on"
DEFAULT_OUTPUT = "/home/cwp29/tmp/mpas/crf_on/final/organisation_lpt"
DEFAULT_TRACKING_DOMAIN = (-50.0, 50.0)


@dataclasses.dataclass(frozen=True)
class LPTConfiguration:
    """Complete physical and linking configuration for one LPT run."""

    precipitation_threshold_mm_day: float = 15.0
    input_interval_hours: int = 6
    sampling_hours_utc: tuple[int, ...] = (0, 6, 12, 18)
    running_mean_hours: int = 72
    gaussian_sigma_degrees: float = 5.0
    gaussian_truncate_sigma: float = 3.0
    minimum_area_km2: float = 300_000.0
    periodic_longitude: bool = True
    minimum_overlap_fraction: float = 0.5
    bare_minimum_overlap_fraction: float = 0.1
    minimum_overlap_points: int = 0
    minimum_overlap_area_km2: float = 0.0
    allow_merge_split: bool = True
    break_up_merge_split: bool = False
    split_merger_min_hours: int = 0
    minimum_duration_hours: int = 7 * 24
    fall_below_threshold_max_hours: int = 0
    tracking_latitude_bounds: tuple[float, float] = DEFAULT_TRACKING_DOMAIN
    lifecycle_lags_hours: tuple[int, ...] = LIFECYCLE_LAGS_HOURS
    tropical_centroid_limit_degrees: float = TROPICAL_CENTROID_LIMIT_DEG
    tropical_min_consecutive_hours: float = TROPICAL_MIN_CONSECUTIVE_HOURS

    def __post_init__(self) -> None:
        if self.running_mean_hours % self.input_interval_hours:
            raise ValueError("72-hour mean must contain a whole number of input intervals")
        expected = 24 // self.input_interval_hours
        if len(self.sampling_hours_utc) != expected:
            raise ValueError(
                "sampling_hours_utc is inconsistent with input_interval_hours"
            )
        if sorted(set(self.sampling_hours_utc)) != list(self.sampling_hours_utc):
            raise ValueError("sampling_hours_utc must be unique and sorted")
        if self.minimum_duration_hours < 7 * 24:
            raise ValueError("publication LPTs must be retained for at least seven days")

    @property
    def identifier(self) -> str:
        hours = "".join(f"{hour:02d}" for hour in self.sampling_hours_utc)
        return (
            f"threshold{self.precipitation_threshold_mm_day:g}_"
            f"{self.input_interval_hours}h_{hours}utc"
        ).replace(".", "p")

    def as_dict(self) -> dict:
        return dataclasses.asdict(self)


PRIMARY_CONFIGURATION = LPTConfiguration()


def lpt_output_root(explicit: str | os.PathLike[str] | None = None) -> Path:
    """Return the root containing separate publication and validation trees."""

    return Path(explicit) if explicit is not None else Path(
        diagnostic_output_dir(DEFAULT_OUTPUT)
    )


def publication_output_directory(
    explicit: str | os.PathLike[str] | None = None,
) -> Path:
    """Return the stable root of the definitive LPT publication products."""

    return lpt_output_root(explicit) / "publication"


@dataclasses.dataclass(frozen=True)
class RunSpecification:
    """A publication or controlled validation tracking run."""

    name: str
    configuration: LPTConfiguration
    record_subset: str = "complete"
    reference_run: str | None = None

    @property
    def is_primary(self) -> bool:
        return (
            self.name == "primary"
            and self.configuration == PRIMARY_CONFIGURATION
            and self.record_subset == "complete"
        )


def publication_run_specifications() -> list[RunSpecification]:
    return [RunSpecification("primary", PRIMARY_CONFIGURATION)]


def validation_run_specifications() -> list[RunSpecification]:
    """Return the non-overwriting, controlled sensitivity suite."""

    specs: list[RunSpecification] = []
    for threshold in (12.0, 15.0, 18.0):
        name = f"threshold_{threshold:g}"
        specs.append(
            RunSpecification(
                name,
                dataclasses.replace(
                    PRIMARY_CONFIGURATION,
                    precipitation_threshold_mm_day=threshold,
                ),
                reference_run=None if threshold == 15.0 else "threshold_15",
            )
        )
    for name, hours in (("sampling_00_12", (0, 12)), ("sampling_06_18", (6, 18))):
        specs.append(
            RunSpecification(
                name,
                dataclasses.replace(
                    PRIMARY_CONFIGURATION,
                    input_interval_hours=12,
                    sampling_hours_utc=hours,
                ),
                reference_run="threshold_15",
            )
        )
    for subset in ("first_half", "second_half"):
        specs.append(
            RunSpecification(
                f"record_{subset}",
                PRIMARY_CONFIGURATION,
                record_subset=subset,
                reference_run="threshold_15",
            )
        )
    return specs


def configuration_metadata(
    specification: RunSpecification,
    *,
    simulation: str,
    upstream_source: str | os.PathLike[str] | None = None,
) -> dict:
    try:
        project_commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        project_commit = "unknown"
    return {
        "product_version": PRODUCT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "simulation": simulation,
        "run_name": specification.name,
        "record_subset": specification.record_subset,
        "reference_run": specification.reference_run,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source": None if upstream_source is None else str(upstream_source),
        "upstream_source_modified": False,
        "project_git_commit": project_commit,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "xarray_version": xr.__version__,
        "citation": UPSTREAM_CITATION,
        "configuration": specification.configuration.as_dict(),
        "selection": {
            "retain_all_valid_lpt_tracks": True,
            "primary_tropical_sample": (
                "precipitation-weighted envelope centroid within 15 degrees "
                "of the equator for at least 24 consecutive hours"
            ),
            "dynamical_selection_criteria": [],
            "lifecycle_reference": (
                "maximum area-integrated trailing-24-hour precipitation "
                "inside the tracked envelope"
            ),
            "primary_composite_requires_complete_lag_window": True,
            "event_weighting": "one equal weight per LPT event",
        },
    }


def _sha256_json(value: Mapping) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def upstream_source_path(explicit: str | None = None) -> Path:
    """Locate a separately installed checkout of the pinned upstream source."""

    candidates = [
        explicit,
        os.environ.get("MPAS_LPT_SOURCE"),
        Path(__file__).resolve().parents[2] / "external" / "lpt-python-public",
    ]
    for candidate in candidates:
        if candidate and (Path(candidate) / "lpt" / "helpers.py").is_file():
            return Path(candidate).resolve()
    raise RuntimeError(
        "Published LPT source was not found. Clone "
        f"{UPSTREAM_REPOSITORY} at commit {UPSTREAM_COMMIT} and set "
        "MPAS_LPT_SOURCE to the checkout described in this module's docstring."
    )


def verify_upstream_source(source: str | os.PathLike[str]) -> str:
    """Require the exact pinned Git revision and an unmodified work tree."""

    source = Path(source).resolve()
    try:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        status = subprocess.run(
            ["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(f"{source} is not a readable LPT Git checkout") from exc
    if commit != UPSTREAM_COMMIT:
        raise RuntimeError(
            f"LPT checkout is {commit}, but this analysis requires {UPSTREAM_COMMIT}"
        )
    if status:
        raise RuntimeError("The pinned upstream LPT checkout has local modifications")
    return commit


@contextlib.contextmanager
def imported_upstream(source: str | os.PathLike[str]):
    """Import the upstream namespace without altering its source tree."""

    source = Path(source).resolve()
    verify_upstream_source(source)
    source_string = str(source)
    sys.path.insert(0, source_string)
    old_context = sys.modules.get("context")
    try:
        lpt = importlib.import_module("lpt")
        context_module = types.ModuleType("context")
        context_module.lpt = lpt
        sys.modules["context"] = context_module
        for module_name in (
            "readdata",
            "helpers",
            "lptio",
            "plotting",
            "masks",
            "mjo_id",
        ):
            setattr(lpt, module_name, importlib.import_module(f"lpt.{module_name}"))
        driver = importlib.import_module("lpt.lpt_driver")
        lpt.lpt_driver = driver
        yield lpt
    finally:
        if old_context is None:
            sys.modules.pop("context", None)
        else:
            sys.modules["context"] = old_context
        with contextlib.suppress(ValueError):
            sys.path.remove(source_string)


def regular_grid_sigma_points(
    latitude: Sequence[float],
    longitude: Sequence[float],
    sigma_degrees: float = 5.0,
) -> float:
    """Translate the physical smoothing scale to the upstream grid-point unit."""

    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    if lat.size < 2 or lon.size < 2:
        raise ValueError("LPT needs at least two latitude and longitude points")
    dlat = np.abs(np.diff(lat))
    dlon = np.abs(np.diff(np.unwrap(np.deg2rad(lon)))) * 180.0 / np.pi
    if not np.allclose(dlat, np.median(dlat), rtol=1e-5, atol=1e-8):
        raise ValueError("LPT preprocessing requires regularly spaced latitude")
    if not np.allclose(dlon, np.median(dlon), rtol=1e-5, atol=1e-8):
        raise ValueError("LPT preprocessing requires regularly spaced longitude")
    if not np.isclose(np.median(dlat), np.median(dlon), rtol=1e-3):
        raise ValueError(
            "Published LPT accepts one Gaussian sigma; latitude and longitude "
            "grid spacing must therefore be equal"
        )
    if not np.isclose(
        np.ptp(np.sort(np.mod(lon, 360.0))) + np.median(dlon),
        360.0,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ValueError(
            "periodic-longitude LPT requires a complete, gap-free global "
            "longitude grid"
        )
    return float(sigma_degrees / np.median(dlat))


def validate_regular_time_axis(
    time: xr.DataArray | Sequence,
    interval_hours: int,
) -> pd.DatetimeIndex:
    values = pd.DatetimeIndex(pd.to_datetime(np.asarray(time)))
    if values.has_duplicates or not values.is_monotonic_increasing:
        raise ValueError("LPT input times must be unique and increasing")
    if len(values) > 1:
        expected = pd.Timedelta(hours=interval_hours)
        if not np.all(np.diff(values.values) == expected.to_timedelta64()):
            raise ValueError(f"LPT input must be complete and regular at {interval_hours} h")
    return values


def sample_precipitation(
    precipitation: xr.DataArray,
    configuration: LPTConfiguration,
) -> xr.DataArray:
    """Select a complete 6- or 12-hour phase without filling missing scenes."""

    times = pd.DatetimeIndex(pd.to_datetime(precipitation.Time.values))
    keep = np.isin(times.hour, configuration.sampling_hours_utc)
    sampled = precipitation.isel(Time=np.flatnonzero(keep))
    validate_regular_time_axis(sampled.Time.values, configuration.input_interval_hours)
    return sampled


def full_running_mean(
    precipitation: xr.DataArray,
    configuration: LPTConfiguration = PRIMARY_CONFIGURATION,
) -> xr.DataArray:
    """Trailing physical 72-hour mean with no partial cold-start values."""

    sampled = sample_precipitation(precipitation, configuration)
    window = configuration.running_mean_hours // configuration.input_interval_hours + 1
    return sampled.rolling(Time=window, min_periods=window).mean().dropna("Time", how="all")


def _upstream_gaussian_filter_2d(
    values: np.ndarray,
    *,
    sigma_points: float,
    truncate_sigma: float,
    periodic_longitude: bool,
) -> np.ndarray:
    """Apply the pinned upstream LPT spatial-filter boundary convention."""

    values = np.asarray(values, dtype=float)
    if periodic_longitude:
        # Kerns/Chen LPT pads longitude before applying scipy's reflecting
        # Gaussian filter, then removes the pad. This mirrors helpers.py in the
        # pinned upstream revision and keeps the project-side figure field
        # numerically consistent with the field used to identify objects.
        pad_width = int(np.ceil(sigma_points))
        padded = np.pad(
            values,
            ((0, 0), (pad_width, pad_width)),
            mode="wrap",
        )
        filtered = ndimage.gaussian_filter(
            padded,
            sigma_points,
            order=0,
            mode="reflect",
            cval=0.0,
            truncate=truncate_sigma,
        )
        return filtered[:, pad_width:-pad_width]
    return ndimage.gaussian_filter(
        values,
        sigma_points,
        order=0,
        mode="reflect",
        cval=0.0,
        truncate=truncate_sigma,
    )


def filtered_running_mean(
    precipitation: xr.DataArray,
    configuration: LPTConfiguration = PRIMARY_CONFIGURATION,
) -> xr.DataArray:
    """Return the complete 72-hour mean plus the published Gaussian filter."""

    running = full_running_mean(precipitation, configuration)
    sigma_points = regular_grid_sigma_points(
        running.latitude.values,
        running.longitude.values,
        configuration.gaussian_sigma_degrees,
    )
    filtered = xr.apply_ufunc(
        _upstream_gaussian_filter_2d,
        running,
        input_core_dims=[["latitude", "longitude"]],
        output_core_dims=[["latitude", "longitude"]],
        kwargs={
            "sigma_points": sigma_points,
            "truncate_sigma": configuration.gaussian_truncate_sigma,
            "periodic_longitude": configuration.periodic_longitude,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).transpose(*running.dims)
    filtered.name = "filtered_72h_precipitation_mm_day"
    filtered.attrs.update(
        {
            "long_name": (
                "complete 72-hour running-mean precipitation with Gaussian "
                "spatial smoothing"
            ),
            "units": "mm day-1",
            "gaussian_sigma_degrees": configuration.gaussian_sigma_degrees,
            "gaussian_sigma_grid_points": sigma_points,
            "periodic_longitude": configuration.periodic_longitude,
            "implementation": (
                "project-side reproduction of the pinned upstream LPT "
                "helpers.py filter for saved diagnostics only"
            ),
        }
    )
    return filtered


def _subset_reference_bounds(
    times: pd.DatetimeIndex,
    subset: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if subset == "complete":
        return times[0], times[-1]
    midpoint = times[0] + (times[-1] - times[0]) / 2
    if subset == "first_half":
        return times[0], midpoint.floor("h")
    if subset == "second_half":
        return midpoint.ceil("h"), times[-1]
    raise ValueError(f"Unknown record subset {subset!r}")


def tracking_and_input_bounds(
    times: Sequence,
    specification: RunSpecification,
) -> dict[str, pd.Timestamp]:
    """Return subset bounds with tracking and 72-hour preprocessing buffers."""

    index = pd.DatetimeIndex(pd.to_datetime(times))
    reference_start, reference_end = _subset_reference_bounds(
        index, specification.record_subset
    )
    if specification.record_subset == "complete":
        tracking_start, tracking_end = reference_start, reference_end
    else:
        # Seven days captures valid tracks whose reference maximum lies just
        # inside a record half; the extra 72 h below is preprocessing only.
        event_buffer = pd.Timedelta(
            hours=specification.configuration.minimum_duration_hours
        )
        tracking_start = max(index[0], reference_start - event_buffer)
        tracking_end = min(index[-1], reference_end + event_buffer)
    input_start = tracking_start - pd.Timedelta(
        hours=specification.configuration.running_mean_hours
    )
    if input_start < index[0]:
        tracking_start = index[0] + pd.Timedelta(
            hours=specification.configuration.running_mean_hours
        )
        input_start = index[0]
    return {
        "input_start": input_start,
        "tracking_start": tracking_start,
        "tracking_end": tracking_end,
        "reference_start": reference_start,
        "reference_end": reference_end,
    }


def write_upstream_precipitation_files(
    precipitation_mm_day: xr.DataArray,
    destination: str | os.PathLike[str],
    specification: RunSpecification,
) -> dict:
    """Write generic NetCDF inputs consumed unchanged by upstream LPT."""

    config = specification.configuration
    precip = sample_precipitation(precipitation_mm_day, config)
    times = pd.DatetimeIndex(pd.to_datetime(precip.Time.values))
    bounds = tracking_and_input_bounds(times, specification)
    precip = precip.sel(Time=slice(bounds["input_start"], bounds["tracking_end"]))
    precip = precip.sel(
        latitude=slice(*(
            config.tracking_latitude_bounds
            if precip.latitude[0] < precip.latitude[-1]
            else config.tracking_latitude_bounds[::-1]
        ))
    )
    if precip.sizes.get("latitude", 0) == 0:
        raise ValueError("Tracking latitude bounds do not intersect the MPAS grid")
    sigma_points = regular_grid_sigma_points(
        precip.latitude.values,
        precip.longitude.values,
        config.gaussian_sigma_degrees,
    )
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    for time_value in pd.DatetimeIndex(pd.to_datetime(precip.Time.values)):
        field = (
            precip.sel(Time=time_value)
            .rename({"latitude": "lat", "longitude": "lon"})
            .rename("rain")
            / 24.0
        )
        field.attrs.update(units="mm h-1", long_name="MPAS total precipitation rate")
        output = xr.Dataset({"rain": field.expand_dims(time=[time_value])})
        output.to_netcdf(
            destination / f"mpas_rain_{time_value:%Y%m%d%H}.nc",
            encoding={"rain": {"zlib": True, "complevel": 2, "dtype": "float32"}},
        )
    return {
        **{key: value.isoformat() for key, value in bounds.items()},
        "written_time_count": int(precip.sizes["Time"]),
        "gaussian_sigma_grid_points": sigma_points,
        "input_directory": str(destination),
    }


def upstream_option_dictionaries(
    input_directory: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    specification: RunSpecification,
    sigma_grid_points: float,
) -> tuple[dict, ...]:
    """Build complete upstream dictionaries without enabling MJO selection."""

    config = specification.configuration
    dataset = {
        "label": "mpas",
        "raw_data_parent_dir": str(input_directory),
        # The pinned upstream revision's single-time generic reader currently
        # calls its helper with a stale keyword.  Its published multi-time
        # generic reader is otherwise identical and works for one-time files,
        # so use that supported path without modifying upstream source.
        "raw_data_format": "generic_netcdf_with_multiple_times",
        "file_name_format": "mpas_rain_%Y%m%d%H.nc",
        "data_time_interval": config.input_interval_hours,
        "verbose": False,
        "longitude_variable_name": "lon",
        "latitude_variable_name": "lat",
        "time_variable_name": "time",
        "field_variable_name": "rain",
        "field_is_accumulated": False,
        "field_units": "mm h-1",
        "area": [
            0,
            360,
            config.tracking_latitude_bounds[0],
            config.tracking_latitude_bounds[1],
        ],
        "calendar": "standard",
    }
    output = {
        "img_dir": str(Path(output_directory) / "images"),
        "data_dir": str(Path(output_directory) / "data"),
        "sub_directory_format": "%Y/%m/%Y%m%d",
    }
    plotting = {
        "do_plotting": False,
        "plot_area": [0, 360, *config.tracking_latitude_bounds],
        "map_crs_name": "PlateCarree",
        "map_crs_central_longitude": 180.0,
        "map_vmin": 1.0,
        "map_vmax": 50.0,
        "map_cmap_base": "jet",
        "map_cmap_factor": 1.5,
        "map_cmap_under_color": "white",
        "map_cmap_over_color": None,
        "time_lon_range": [0, 360],
    }
    lpo = {
        "do_lpo_calc": True,
        "lpo_calc_n_cores": 1,
        "overwrite_existing_files": True,
        "multiply_factor": 24.0,
        "field_units": "mm d-1",
        "thresh": config.precipitation_threshold_mm_day,
        "object_is_gt_threshold": True,
        "thresh_or_equal": True,
        "accumulation_hours": config.running_mean_hours,
        "filter_stdev": sigma_grid_points,
        "filter_n_stdev_width": config.gaussian_truncate_sigma,
        "min_points": 1,
        "min_area": config.minimum_area_km2,
        "periodic_longitude": config.periodic_longitude,
        "cold_start_mode": False,
        "cold_start_const_period": 24.0,
        "do_lpo_mask": False,
        "mask_include_rain_rates": True,
        "mask_calc_volrain": True,
        "mask_detailed_output": True,
        "mask_calc_with_filter_radius": True,
        "mask_calc_with_accumulation_period": True,
        "mask_coarse_grid_factor": 0,
        "target_memory_for_writing_masks_MB": 2000,
        "mask_n_cores": 1,
    }
    lpt = {
        "do_lpt_calc": True,
        "min_overlap_points": config.minimum_overlap_points,
        "min_overlap_area": config.minimum_overlap_area_km2,
        "min_overlap_frac": config.minimum_overlap_fraction,
        "bare_min_overlap_frac": config.bare_minimum_overlap_fraction,
        "min_lp_objects_points": 1,
        "min_lp_objects_area": config.minimum_area_km2,
        "min_lpt_duration_hours": config.minimum_duration_hours,
        "fall_below_threshold_max_hours": config.fall_below_threshold_max_hours,
        "lpt_n_cores": 1,
        "do_lpt_individual_masks": True,
        "do_lpt_individual_group_masks": False,
        "individual_masks_begin_lptid": 0,
        "individual_masks_end_lptid": 100000,
        "individual_masks_mjo_only": False,
        "do_lpt_composite_mask": True,
        "do_mjo_lpt_composite_mask": False,
        "do_non_mjo_lpt_composite_mask": False,
        "mask_include_rain_rates": True,
        "mask_calc_volrain": True,
        "mask_detailed_output": True,
        "mask_calc_with_filter_radius": True,
        "mask_calc_with_accumulation_period": True,
        "mask_coarse_grid_factor": 0,
        "target_memory_for_writing_masks_MB": 2000,
        "mask_n_cores": 1,
    }
    merge_split = {
        "break_up_merge_split": config.break_up_merge_split,
        "allow_merge_split": config.allow_merge_split,
        "split_merger_min_hours": config.split_merger_min_hours,
    }
    mjo = {
        "do_mjo_id": False,
        "do_plotting": False,
        # Explicit sentinels document that no MJO/dynamical selection is run.
        "min_zonal_speed": -999.0,
        "min_lpt_duration": 0.0,
    }
    return dataset, plotting, output, lpo, lpt, merge_split, mjo


def run_upstream_lpt(
    source: str | os.PathLike[str],
    input_directory: str | os.PathLike[str],
    output_directory: str | os.PathLike[str],
    specification: RunSpecification,
    preprocessing_metadata: Mapping,
) -> Path:
    """Run the pinned upstream driver and return its original systems file."""

    sigma = float(preprocessing_metadata["gaussian_sigma_grid_points"])
    options = upstream_option_dictionaries(
        input_directory, output_directory, specification, sigma
    )
    begin = pd.Timestamp(preprocessing_metadata["tracking_start"])
    end = pd.Timestamp(preprocessing_metadata["tracking_end"])
    argv = ["mpas_lpt", begin.strftime("%Y%m%d%H"), end.strftime("%Y%m%d%H")]
    # Upstream atomically replaces systems NetCDF files through a relative
    # temporary name. Run on the output filesystem so POSIX rename remains
    # atomic even when the analysis checkout and data cache are different
    # mounts.
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    previous_directory = Path.cwd()
    try:
        os.chdir(output_directory)
        with imported_upstream(source) as lpt:
            lpt.lpt_driver.lpt_driver(*options, argv)
    finally:
        os.chdir(previous_directory)
    systems = sorted(Path(output_directory).glob("data/**/systems/lpt_systems_*.nc"))
    if len(systems) != 1:
        raise RuntimeError(
            f"Expected one upstream LPT systems file, found {len(systems)}"
        )
    return systems[0]


def consecutive_true_hours(values: Sequence[bool], step_hours: float) -> float:
    maximum = current = 0
    for value in np.asarray(values, dtype=bool):
        current = current + 1 if value else 0
        maximum = max(maximum, current)
    return float(maximum * step_hours)


def unwrap_longitude(longitude: Sequence[float]) -> np.ndarray:
    return np.rad2deg(np.unwrap(np.deg2rad(np.asarray(longitude, dtype=float))))


def propagation_metrics(
    time: Sequence,
    longitude: Sequence[float],
    latitude: Sequence[float],
) -> dict[str, float | str]:
    """Regression propagation speeds and net displacement after tracking."""

    stamps = pd.DatetimeIndex(pd.to_datetime(time))
    hours = np.asarray((stamps - stamps[0]) / pd.Timedelta(hours=1), dtype=float)
    lon = unwrap_longitude(longitude)
    lat = np.asarray(latitude, dtype=float)
    valid = np.isfinite(hours) & np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2 or np.ptp(hours[valid]) == 0:
        return {
            "zonal_speed_m_s": np.nan,
            "meridional_speed_m_s": np.nan,
            "total_zonal_displacement_km": np.nan,
            "propagation_class": "unknown",
        }
    lat_mean = np.nanmean(lat[valid])
    zonal_km_hour = np.polyfit(hours[valid], lon[valid], 1)[0] * 111.195 * np.cos(
        np.deg2rad(lat_mean)
    )
    meridional_km_hour = np.polyfit(hours[valid], lat[valid], 1)[0] * 111.195
    zonal_speed = zonal_km_hour / 3.6
    displacement = (
        (lon[valid][-1] - lon[valid][0])
        * 111.195
        * np.cos(np.deg2rad(lat_mean))
    )
    propagation_class = (
        "eastward"
        if zonal_speed > 1.0
        else "westward"
        if zonal_speed < -1.0
        else "weakly_propagating"
    )
    return {
        "zonal_speed_m_s": float(zonal_speed),
        "meridional_speed_m_s": float(meridional_km_hour / 3.6),
        "total_zonal_displacement_km": float(displacement),
        "propagation_class": propagation_class,
    }


def standardise_upstream_tracks(
    systems_file: str | os.PathLike[str],
    simulation: str,
    specification: RunSpecification,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert stitched upstream tracks to stable project tables."""

    with xr.open_dataset(systems_file) as ds:
        ds = ds.load()
    steps: list[pd.DataFrame] = []
    events: list[dict] = []
    reference_start, reference_end = _subset_reference_bounds(
        pd.DatetimeIndex(pd.to_datetime(ds.timestamp_stitched.values[np.isfinite(ds.lptid_stitched.values)])),
        specification.record_subset,
    )
    for index, upstream_id in enumerate(np.asarray(ds.lptid.values)):
        start = int(ds.lpt_begin_index.values[index])
        end = int(ds.lpt_end_index.values[index]) + 1
        event = pd.DataFrame(
            {
                "Time": pd.to_datetime(ds.timestamp_stitched.values[start:end]),
                "upstream_lpt_id": float(upstream_id),
                "centroid_lon": ds.centroid_lon_stitched.values[start:end],
                "centroid_lat": ds.centroid_lat_stitched.values[start:end],
                "area_km2": ds.area_stitched.values[start:end],
                "min_lon": ds.min_lon_stitched.values[start:end],
                "max_lon": ds.max_lon_stitched.values[start:end],
                "min_lat": ds.min_lat_stitched.values[start:end],
                "max_lat": ds.max_lat_stitched.values[start:end],
                "simultaneous_branches": ds.nobj_stitched.values[start:end],
                "mean_instantaneous_precip_mm_h": ds.amean_inst_field.values[start:end],
                "maximum_instantaneous_precip_mm_h": ds.max_inst_field.values[start:end],
                "mean_filtered_precip_mm_day": ds.amean_filtered_running_field.values[start:end],
            }
        )
        event = event[np.isfinite(event.area_km2) & event.Time.notna()].copy()
        if event.empty:
            continue
        event_id = f"{simulation}_{specification.name}_{index + 1:05d}"
        event.insert(0, "event_id", event_id)
        event.insert(0, "simulation", simulation)
        event["zonal_extent_degrees"] = circular_extent(event.min_lon, event.max_lon)
        event["meridional_extent_degrees"] = event.max_lat - event.min_lat
        event["tropical_centroid"] = (
            np.abs(event.centroid_lat)
            <= specification.configuration.tropical_centroid_limit_degrees
        )
        step_hours = (
            np.median(np.diff(event.Time) / np.timedelta64(1, "h"))
            if len(event) > 1
            else specification.configuration.input_interval_hours
        )
        tropical_hours = consecutive_true_hours(event.tropical_centroid, step_hours)
        propagation = propagation_metrics(
            event.Time, event.centroid_lon, event.centroid_lat
        )
        branch_count = np.asarray(event.simultaneous_branches, dtype=float)
        branch_change = np.diff(branch_count)
        events.append(
            {
                "simulation": simulation,
                "event_id": event_id,
                "upstream_lpt_id": float(upstream_id),
                "start_time": event.Time.min(),
                "end_time": event.Time.max(),
                "duration_hours": (
                    event.Time.max() - event.Time.min()
                ).total_seconds()
                / 3600.0,
                "maximum_area_km2": float(event.area_km2.max()),
                "mean_area_km2": float(event.area_km2.mean()),
                "tropical_consecutive_hours": tropical_hours,
                "primary_tropical_sample": (
                    tropical_hours
                    >= specification.configuration.tropical_min_consecutive_hours
                ),
                "merger_occurred": bool(np.any(branch_change < 0)),
                "split_occurred": bool(np.any(branch_change > 0)),
                "maximum_simultaneous_branches": int(np.nanmax(branch_count)),
                "record_subset": specification.record_subset,
                "reference_assignment_start": reference_start,
                "reference_assignment_end": reference_end,
                **propagation,
            }
        )
        steps.append(event)
    return (
        pd.concat(steps, ignore_index=True) if steps else pd.DataFrame(),
        pd.DataFrame(events),
    )


def circular_extent(minimum: Sequence[float], maximum: Sequence[float]) -> np.ndarray:
    extent = np.asarray(maximum, dtype=float) - np.asarray(minimum, dtype=float)
    return np.where(extent < 0, extent + 360.0, extent)


def precipitation_weighted_centroid(
    mask: xr.DataArray,
    precipitation: xr.DataArray,
    area_km2: xr.DataArray,
) -> tuple[float, float]:
    valid = mask.astype(bool) & np.isfinite(precipitation)
    weights = precipitation.where(valid, 0.0) * area_km2
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0:
        return np.nan, np.nan
    latitude = xr.broadcast(precipitation.latitude, precipitation)[0]
    longitude = xr.broadcast(precipitation.longitude, precipitation)[0]
    lat = float((latitude * weights).sum() / total)
    angle = np.deg2rad(longitude)
    sin_mean = float((np.sin(angle) * weights).sum() / total)
    cos_mean = float((np.cos(angle) * weights).sum() / total)
    lon = float(np.mod(np.rad2deg(np.arctan2(sin_mean, cos_mean)), 360.0))
    return lon, lat


def label_count_periodic(mask: np.ndarray) -> int:
    """Count 8-connected clusters while merging labels across longitude."""

    labels, count = ndimage.label(mask, structure=np.ones((3, 3), dtype=int))
    if count == 0:
        return 0
    parent = np.arange(count + 1)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        if left and right:
            parent[find(right)] = find(left)

    for row in range(mask.shape[0]):
        for offset in (-1, 0, 1):
            other = (row + offset) % mask.shape[0]
            if 0 <= row + offset < mask.shape[0]:
                union(labels[row, 0], labels[other, -1])
    return len({find(value) for value in range(1, count + 1)})


CLASS_MASKS = {
    "active_deep_convective": "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud": "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud": "nonprecipitating_high_cloud_mask",
    "total_high_cloud": "high_cloud_mask",
}


def _area_sum(mask: xr.DataArray, area: xr.DataArray) -> float:
    return float(area.where(mask.astype(bool), 0.0).sum())


def diagnose_event_steps(
    event_mask: xr.DataArray,
    precipitation_mm_day: xr.DataArray,
    classification: Mapping[str, xr.DataArray],
    model_fields: Mapping[str, xr.DataArray],
    *,
    event_id: str,
    simulation: str,
) -> pd.DataFrame:
    """Diagnose instantaneous cloud populations inside one tracked envelope."""

    precipitation, event_mask = xr.align(
        precipitation_mm_day, event_mask.astype(bool), join="inner"
    )
    if precipitation.sizes.get("Time", 0) == 0:
        return pd.DataFrame()
    area = grid_cell_area_km2(
        precipitation.latitude.values, precipitation.longitude.values
    )
    area = xr.DataArray(
        area,
        dims=("latitude", "longitude"),
        coords={
            "latitude": precipitation.latitude,
            "longitude": precipitation.longitude,
        },
    )
    # A full trailing 24 h mean. Missing early values remain missing and cannot
    # become a lifecycle reference.
    dt_hours = float(
        np.median(np.diff(pd.to_datetime(precipitation.Time.values)))
        / np.timedelta64(1, "h")
    )
    # MPAS rates at each six-hour output represent the preceding accumulation
    # interval, hence four six-hour rates form a physical 24-hour quantity.
    window_24h = int(round(24.0 / dt_hours))
    precip_24h = precipitation.rolling(
        Time=window_24h, min_periods=window_24h
    ).mean()
    rows: list[dict] = []
    for time_value in pd.to_datetime(event_mask.Time.values):
        mask = event_mask.sel(Time=time_value)
        precip = precipitation.sel(Time=time_value)
        precip24 = precip_24h.sel(Time=time_value)
        envelope_area = _area_sum(mask, area)
        precip_volume = float((precip.where(mask) * area).sum(skipna=True))
        precip24_volume = float((precip24.where(mask) * area).sum(skipna=True))
        lon, lat = precipitation_weighted_centroid(mask, precip, area)
        row = {
            "simulation": simulation,
            "event_id": event_id,
            "Time": time_value,
            "envelope_area_km2": envelope_area,
            "area_integrated_precip_km2_mm_day": precip_volume,
            "area_integrated_24h_precip_km2_mm_day": precip24_volume,
            "mean_precip_mm_day": float(precip.where(mask).mean(skipna=True)),
            "maximum_precip_mm_day": float(precip.where(mask).max(skipna=True)),
            "precip_weighted_centroid_lon": lon,
            "precip_weighted_centroid_lat": lat,
        }
        for short_name, variable_name in CLASS_MASKS.items():
            if variable_name not in classification:
                row[f"{short_name}_area_km2"] = np.nan
                continue
            class_mask = classification[variable_name].sel(Time=time_value) & mask
            row[f"{short_name}_area_km2"] = _area_sum(class_mask, area)
            if short_name == "active_deep_convective":
                row["active_cluster_count"] = label_count_periodic(
                    np.asarray(class_mask.values, dtype=bool)
                )
        convective = classification.get(
            "convective_precipitating_high_cloud_mask"
        )
        conv_mask = None
        if convective is not None:
            conv_mask = convective.sel(Time=time_value) & mask
            row["convective_precip_area_km2"] = _area_sum(
                conv_mask & (precip > 0), area
            )
            row["convective_precip_volume_km2_mm_day"] = float(
                (precip.where(conv_mask, 0.0) * area).sum()
            )
        stratiform = classification.get(
            "stratiform_precipitating_high_cloud_mask"
        )
        row["stratiform_precip_fraction"] = (
            float((precip.where(stratiform.sel(Time=time_value) & mask, 0.0) * area).sum())
            / precip_volume
            if stratiform is not None and precip_volume > 0
            else np.nan
        )
        if "cloud_top_height" in classification:
            high_mask = (
                classification["high_cloud_mask"].sel(Time=time_value) & mask
                if "high_cloud_mask" in classification
                else mask
            )
            cloud_top_height = classification["cloud_top_height"].sel(
                Time=time_value
            ).where(high_mask)
            row["mean_cloud_top_height"] = float(
                cloud_top_height.mean(skipna=True)
            )
            row["maximum_cloud_top_height"] = float(
                cloud_top_height.max(skipna=True)
            )
        for variable_name, statistic, output_name in (
            ("cloud_top_height", "mean", "mean_cloud_top_height"),
            ("cloud_top_height", "max", "maximum_cloud_top_height"),
            ("olr", "mean", "olr_w_m2"),
            ("olrtoa", "mean", "olr_w_m2"),
            ("w", "max", "maximum_vertical_velocity"),
            ("w", "quantile", "p95_vertical_velocity"),
            ("high_cloud_ice_mass", "sum", "high_cloud_ice_mass"),
            ("high_cloud_snow_mass", "sum", "high_cloud_snow_mass"),
            ("lw_cloud_radiative_effect", "mean", "lw_cloud_radiative_effect"),
            ("sw_cloud_radiative_effect", "mean", "sw_cloud_radiative_effect"),
            ("utls_temperature_anomaly", "mean", "utls_temperature_anomaly"),
            ("tropopause_static_stability", "mean", "tropopause_static_stability"),
            ("upper_tropospheric_zonal_wind", "mean", "upper_tropospheric_zonal_wind"),
            ("upper_tropospheric_vertical_shear", "mean", "upper_tropospheric_vertical_shear"),
        ):
            if output_name in row or variable_name not in model_fields:
                continue
            field = model_fields[variable_name]
            if "Time" in field.dims:
                field = field.sel(Time=time_value)
            selected = field.where(
                conv_mask
                if variable_name == "w" and conv_mask is not None
                else mask
            )
            if statistic == "mean":
                value = selected.mean(skipna=True)
            elif statistic == "max":
                value = selected.max(skipna=True)
            elif statistic == "quantile":
                value = selected.quantile(0.95, skipna=True)
            else:
                value = (selected * area).sum(skipna=True)
                if output_name in {
                    "high_cloud_ice_mass",
                    "high_cloud_snow_mass",
                }:
                    value = value * 1.0e6
            row[output_name] = float(value)
        rows.append(row)
    return pd.DataFrame(rows)


def read_upstream_event_masks(
    upstream_directory: str | os.PathLike[str],
    events: pd.DataFrame,
) -> dict[str, xr.DataArray]:
    """Read original individual LPT masks and expose project-standard geometry."""

    files = sorted(Path(upstream_directory).glob("data/**/systems/**/lpt_system_mask_*.nc"))
    by_upstream_id: dict[float, Path] = {}
    for path in files:
        with xr.open_dataset(path) as dataset:
            if "lptid" not in dataset:
                continue
            by_upstream_id[round(float(dataset.lptid.values[0]), 4)] = path
    masks: dict[str, xr.DataArray] = {}
    for event in events.itertuples():
        key = round(float(event.upstream_lpt_id), 4)
        if key not in by_upstream_id:
            raise RuntimeError(
                f"Missing original upstream individual mask for LPT {key}"
            )
        dataset = xr.open_dataset(by_upstream_id[key], chunks={"time": 1})
        variable = (
            "mask_at_end_time"
            if "mask_at_end_time" in dataset
            else "mask"
            if "mask" in dataset
            else None
        )
        if variable is None:
            dataset.close()
            raise RuntimeError(
                f"{by_upstream_id[key]} contains no LPT envelope mask"
            )
        mask = dataset[variable].rename(
            {"time": "Time", "lat": "latitude", "lon": "longitude"}
        )
        mask = mask.astype(bool).rename("lpt_envelope_mask")
        mask.attrs.update(
            {
                "long_name": "tracked LPT precipitation envelope",
                "geometry": "upstream LPT mask_at_end_time",
                "upstream_lpt_id": float(event.upstream_lpt_id),
            }
        )
        masks[event.event_id] = mask
    return masks


def project_model_fields(dataset: xr.Dataset, precipitation: xr.DataArray) -> dict:
    """Expose available MPAS diagnostics under the stable lifecycle names."""

    fields: dict[str, xr.DataArray] = {}
    aliases = {
        "olr": ("olrtoa", "olr"),
        "w": ("w", "vertical_velocity"),
        "lw_cloud_radiative_effect": ("lw_cloud_radiative_effect", "lw_cre"),
        "sw_cloud_radiative_effect": ("sw_cloud_radiative_effect", "sw_cre"),
        "utls_temperature_anomaly": ("utls_temperature_anomaly",),
        "tropopause_static_stability": ("tropopause_static_stability",),
        "upper_tropospheric_zonal_wind": ("upper_tropospheric_zonal_wind",),
        "upper_tropospheric_vertical_shear": ("upper_tropospheric_vertical_shear",),
        "high_cloud_ice_mass": ("high_cloud_ice_mass",),
        "high_cloud_snow_mass": ("high_cloud_snow_mass",),
    }
    for output_name, candidates in aliases.items():
        for candidate in candidates:
            if candidate in dataset:
                field = dataset[candidate]
                if "Time" in field.dims:
                    field = field.sel(Time=precipitation.Time)
                # Only 2-D column diagnostics can be masked directly here.
                if set(("latitude", "longitude")).issubset(field.dims) and not (
                    set(field.dims) - {"Time", "latitude", "longitude"}
                ):
                    fields[output_name] = field
                break
    if "w" in dataset:
        vertical = next(
            (dim for dim in ("z", "nVertLevelsP1", "nVertLevels") if dim in dataset.w.dims),
            None,
        )
        if vertical:
            fields["w"] = dataset.w.sel(Time=precipitation.Time).max(
                vertical, skipna=True
            )
    if "cldfrac" in dataset:
        fields["cloud_fraction_profile"] = dataset.cldfrac.sel(
            Time=precipitation.Time
        )
    if "t_isobaric" in dataset and "pres" in dataset.t_isobaric.dims:
        temperature = dataset.t_isobaric.sel(Time=precipitation.Time)
        utls = temperature.where(
            (temperature.pres >= 100.0) & (temperature.pres <= 300.0),
            drop=True,
        ).mean("pres", skipna=True)
        fields["utls_temperature_anomaly"] = utls - utls.mean("Time")
        t100 = temperature.sel(pres=100.0, method="nearest")
        t200 = temperature.sel(pres=200.0, method="nearest")
        theta100 = t100 * (1000.0 / float(t100.pres)) ** 0.286
        theta200 = t200 * (1000.0 / float(t200.pres)) ** 0.286
        stability = (theta100 - theta200).rename(
            "tropopause_static_stability"
        )
        stability.attrs.update(
            units="K per 100 hPa",
            definition="theta(near 100 hPa) minus theta(near 200 hPa)",
        )
        fields["tropopause_static_stability"] = stability
    if "uzonal_isobaric" in dataset and "pres" in dataset.uzonal_isobaric.dims:
        zonal_wind = dataset.uzonal_isobaric.sel(Time=precipitation.Time)
        fields["upper_tropospheric_zonal_wind"] = zonal_wind.where(
            (zonal_wind.pres >= 150.0) & (zonal_wind.pres <= 300.0),
            drop=True,
        ).mean("pres", skipna=True)
        fields["upper_tropospheric_vertical_shear"] = (
            zonal_wind.sel(pres=200.0, method="nearest")
            - zonal_wind.sel(pres=850.0, method="nearest")
        )
    if all(name in dataset for name in ("rho", "pressure", "zgrid")):
        dz = dataset.zgrid.diff("nVertLevelsP1").rename(
            {"nVertLevelsP1": "nVertLevels"}
        )
        density = dataset.rho.sel(Time=precipitation.Time)
        high = dataset.pressure.sel(Time=precipitation.Time) <= 44_000.0
        for source_name, output_name in (
            ("qi", "high_cloud_ice_mass"),
            ("qs", "high_cloud_snow_mass"),
        ):
            if source_name in dataset:
                fields[output_name] = (
                    dataset[source_name].sel(Time=precipitation.Time)
                    * density
                    * dz
                ).where(high).sum("nVertLevels", skipna=True)
    return fields


def postprocess_event_masks(
    *,
    upstream_directory: str | os.PathLike[str],
    standard_directory: str | os.PathLike[str],
    steps: pd.DataFrame,
    events: pd.DataFrame,
    precipitation: xr.DataArray,
    classification: Mapping[str, xr.DataArray],
    model_fields: Mapping[str, xr.DataArray],
    specification: RunSpecification,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize masks and produce all mask-dependent event diagnostics."""

    masks = read_upstream_event_masks(upstream_directory, events)
    standard_directory = Path(standard_directory)
    mask_directory = standard_directory / "event_masks"
    mask_directory.mkdir(parents=True, exist_ok=True)
    diagnostic_rows: list[pd.DataFrame] = []
    mask_index_rows = []
    filtered_precipitation = filtered_running_mean(
        precipitation, specification.configuration
    )
    for event in events.itertuples():
        mask = masks[event.event_id]
        event_times = pd.to_datetime(
            steps.loc[steps.event_id == event.event_id, "Time"]
        ).to_numpy()
        # Materialise only this sparse event mask. This avoids competing
        # NetCDF/Dask locks when the mask is used simultaneously to write the
        # self-contained product and sample model/classification files.
        mask = mask.sel(Time=event_times).load()
        raw_event_precipitation, mask = xr.align(
            precipitation.sel(Time=event_times), mask, join="inner"
        )
        filtered_event_precipitation = filtered_precipitation.sel(
            Time=mask.Time
        )
        regime = xr.zeros_like(mask, dtype=np.int8).rename(
            "embedded_cloud_regime"
        )
        regime.attrs.update(
            {
                "flag_values": [0, 1, 2, 3],
                "flag_meanings": (
                    "other active_deep_convection "
                    "stratiform_precipitating_high_cloud "
                    "nonprecipitating_high_cloud"
                ),
            }
        )
        for code, variable_name in (
            (3, "nonprecipitating_high_cloud_mask"),
            (2, "stratiform_precipitating_high_cloud_mask"),
            (1, "convective_precipitating_high_cloud_mask"),
        ):
            if variable_name in classification:
                regime = xr.where(
                    classification[variable_name].sel(Time=mask.Time) & mask,
                    code,
                    regime,
                )
        output_path = mask_directory / f"{event.event_id}_mask.nc"
        event_variables = {
            "lpt_envelope_mask": mask,
            "raw_precipitation_mm_day": raw_event_precipitation,
            "filtered_72h_precipitation_mm_day": filtered_event_precipitation,
            "embedded_cloud_regime": regime,
        }
        if "cloud_fraction_profile" in model_fields:
            area = xr.DataArray(
                grid_cell_area_km2(
                    mask.latitude.values, mask.longitude.values
                ),
                dims=("latitude", "longitude"),
                coords={
                    "latitude": mask.latitude,
                    "longitude": mask.longitude,
                },
            )
            weights = area.where(mask)
            profile = model_fields["cloud_fraction_profile"].sel(
                Time=mask.Time
            )
            event_variables["vertical_cloud_fraction"] = (
                (profile * weights).sum(
                    ("latitude", "longitude"), skipna=True
                )
                / weights.sum(("latitude", "longitude"), skipna=True)
            )
            event_variables["vertical_cloud_fraction"].attrs.update(
                {
                    "long_name": (
                        "area-weighted vertical cloud-fraction profile "
                        "inside the LPT envelope"
                    ),
                    "units": "1",
                }
            )
        xr.Dataset(
            event_variables,
            attrs={
                "event_id": event.event_id,
                "upstream_lpt_id": float(event.upstream_lpt_id),
                "purpose": (
                    "self-contained event geometry and fields sufficient for "
                    "tracking overview and event-centred figures"
                ),
            },
        ).to_netcdf(
            output_path,
            encoding={
                "lpt_envelope_mask": {
                    "dtype": "int8",
                    "zlib": True,
                    "complevel": 4,
                },
                "raw_precipitation_mm_day": {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 2,
                },
                "filtered_72h_precipitation_mm_day": {
                    "dtype": "float32",
                    "zlib": True,
                    "complevel": 2,
                },
                "embedded_cloud_regime": {
                    "dtype": "int8",
                    "zlib": True,
                    "complevel": 4,
                },
                **(
                    {
                        "vertical_cloud_fraction": {
                            "dtype": "float32",
                            "zlib": True,
                            "complevel": 2,
                        }
                    }
                    if "vertical_cloud_fraction" in event_variables
                    else {}
                ),
            },
        )
        mask_index_rows.append(
            {
                "simulation": event.simulation,
                "event_id": event.event_id,
                "upstream_lpt_id": event.upstream_lpt_id,
                "mask_file": str(output_path.relative_to(standard_directory)),
            }
        )
        diagnostic_rows.append(
            diagnose_event_steps(
                mask,
                precipitation,
                classification,
                model_fields,
                event_id=event.event_id,
                simulation=event.simulation,
            )
        )
    pd.DataFrame(mask_index_rows).to_csv(
        standard_directory / "lpt_event_mask_index.csv", index=False
    )
    tropical_filtered = filtered_precipitation.sel(
        latitude=slice(
            -15.0 if filtered_precipitation.latitude[0] < filtered_precipitation.latitude[-1] else 15.0,
            15.0 if filtered_precipitation.latitude[0] < filtered_precipitation.latitude[-1] else -15.0,
        )
    )
    latitude_weights = np.cos(np.deg2rad(tropical_filtered.latitude))
    xr.Dataset(
        {
            "equatorial_filtered_precipitation_mm_day": tropical_filtered.weighted(
                latitude_weights
            ).mean("latitude")
        },
        attrs={
            "latitude_band": "15S-15N cosine-latitude weighted mean",
            "filter": "complete trailing 72-hour mean, Gaussian sigma 5 degrees",
        },
    ).to_netcdf(
        standard_directory / "lpt_time_longitude_overview.nc",
        encoding={
            "equatorial_filtered_precipitation_mm_day": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 2,
            }
        },
    )
    diagnostics = (
        pd.concat(diagnostic_rows, ignore_index=True)
        if diagnostic_rows
        else pd.DataFrame()
    )
    lifecycle, timing_summary = assign_lifecycle_reference(
        diagnostics, specification.configuration
    )
    if lifecycle.empty:
        return lifecycle, events, timing_summary

    # Scientific tropical membership is based on the requested
    # precipitation-weighted centroid, not the upstream geometric centroid.
    event_updates = []
    reference_start, reference_end = _subset_reference_bounds(
        pd.DatetimeIndex(pd.to_datetime(precipitation.Time.values)),
        specification.record_subset,
    )
    for event_id, event_steps in lifecycle.groupby("event_id", sort=False):
        event_steps = event_steps.sort_values("Time")
        dt_hours = (
            float(np.median(np.diff(pd.to_datetime(event_steps.Time))) / np.timedelta64(1, "h"))
            if len(event_steps) > 1
            else specification.configuration.input_interval_hours
        )
        within = (
            np.abs(event_steps.precip_weighted_centroid_lat)
            <= specification.configuration.tropical_centroid_limit_degrees
        )
        tropical_hours = consecutive_true_hours(within, dt_hours)
        reference_time = pd.Timestamp(event_steps.reference_time.iloc[0])
        propagation = propagation_metrics(
            event_steps.Time,
            event_steps.precip_weighted_centroid_lon,
            event_steps.precip_weighted_centroid_lat,
        )
        event_updates.append(
            {
                "event_id": event_id,
                "reference_time": reference_time,
                "reference_in_record_subset": bool(
                    reference_start <= reference_time <= reference_end
                ),
                "tropical_consecutive_hours": tropical_hours,
                "primary_tropical_sample": bool(
                    tropical_hours
                    >= specification.configuration.tropical_min_consecutive_hours
                ),
                "complete_primary_lag_window": bool(
                    event_steps.complete_primary_lag_window.iloc[0]
                ),
                "event_integrated_precipitation": float(
                    np.trapz(
                        event_steps.area_integrated_precip_km2_mm_day,
                        x=(
                            pd.to_datetime(event_steps.Time)
                            - pd.Timestamp(event_steps.Time.iloc[0])
                        )
                        / pd.Timedelta(days=1),
                    )
                ),
                "envelope_area_time_km2_h": float(
                    np.trapz(
                        event_steps.envelope_area_km2,
                        x=(
                            pd.to_datetime(event_steps.Time)
                            - pd.Timestamp(event_steps.Time.iloc[0])
                        )
                        / pd.Timedelta(hours=1),
                    )
                ),
                "analysed_record_days": float(
                    (
                        pd.Timestamp(precipitation.Time.values[-1])
                        - pd.Timestamp(precipitation.Time.values[0])
                    )
                    / pd.Timedelta(days=1)
                ),
                "tracking_domain_area_km2": float(
                    np.sum(
                        grid_cell_area_km2(
                            precipitation.latitude.values,
                            precipitation.longitude.values,
                        )
                    )
                ),
                "maximum_area_integrated_24h_precipitation": float(
                    event_steps.area_integrated_24h_precip_km2_mm_day.max()
                ),
                "mean_precip_mm_day": float(event_steps.mean_precip_mm_day.mean()),
                "maximum_precip_mm_day": float(event_steps.maximum_precip_mm_day.max()),
                "maximum_zonal_extent_degrees": float(
                    steps.loc[
                        steps.event_id == event_id,
                        "zonal_extent_degrees",
                    ].max()
                ),
                "maximum_meridional_extent_degrees": float(
                    steps.loc[
                        steps.event_id == event_id,
                        "meridional_extent_degrees",
                    ].max()
                ),
                **propagation,
            }
        )
    updates = pd.DataFrame(event_updates).set_index("event_id")
    events = events.set_index("event_id")
    for column in updates:
        events[column] = updates[column]
    events = events.reset_index()
    for column in (
        "reference_in_record_subset",
        "primary_tropical_sample",
        "tropical_consecutive_hours",
    ):
        lifecycle[column] = lifecycle.event_id.map(
            updates[column].to_dict()
        )
    conversion_summary = event_lifecycle_summary(lifecycle)
    if not conversion_summary.empty:
        timing_summary = timing_summary.merge(
            conversion_summary,
            on=("simulation", "event_id"),
            how="left",
        )
    return lifecycle, events, timing_summary


def assign_lifecycle_reference(
    steps: pd.DataFrame,
    configuration: LPTConfiguration = PRIMARY_CONFIGURATION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign peak-precipitation t=0 and fixed physical lags to every event."""

    output: list[pd.DataFrame] = []
    summaries: list[dict] = []
    lag_set = set(configuration.lifecycle_lags_hours)
    for event_id, event in steps.groupby("event_id", sort=False):
        event = event.sort_values("Time").copy()
        valid = np.isfinite(event.area_integrated_24h_precip_km2_mm_day)
        if not valid.any():
            reference_time = pd.NaT
            event["lag_hours"] = np.nan
            complete = False
        else:
            peak_index = event.loc[
                valid, "area_integrated_24h_precip_km2_mm_day"
            ].idxmax()
            reference_time = pd.Timestamp(event.loc[peak_index, "Time"])
            event["lag_hours"] = (
                pd.to_datetime(event.Time) - reference_time
            ) / pd.Timedelta(hours=1)
            available = set(event.loc[np.isfinite(event.lag_hours), "lag_hours"].astype(int))
            complete = lag_set.issubset(available)
        event["reference_time"] = reference_time
        event["complete_primary_lag_window"] = complete
        event["in_primary_lifecycle_window"] = event.lag_hours.isin(lag_set)
        summary = {
            "simulation": event.simulation.iloc[0],
            "event_id": event_id,
            "reference_time": reference_time,
            "complete_primary_lag_window": complete,
        }
        for column in event.select_dtypes(include=[np.number]).columns:
            if column in {"lag_hours"} or not np.isfinite(event[column]).any():
                continue
            if column == "olr_w_m2":
                index = event[column].idxmin()
                prefix = "minimum"
            else:
                index = event[column].idxmax()
                prefix = "maximum"
            summary[f"{prefix}_{column}"] = float(event.loc[index, column])
            summary[f"{column}_peak_lag_hours"] = (
                float(event.loc[index, "lag_hours"])
                if pd.notna(reference_time)
                else np.nan
            )
        summaries.append(summary)
        output.append(event)
    return (
        pd.concat(output, ignore_index=True) if output else pd.DataFrame(),
        pd.DataFrame(summaries),
    )


def event_lifecycle_summary(
    lifecycle: pd.DataFrame,
    rainfall_decay_fraction: float = 0.25,
) -> pd.DataFrame:
    """Timing/conversion quantities used by publication Figure 5."""

    if "primary_tropical_sample" in lifecycle:
        lifecycle = lifecycle[lifecycle.primary_tropical_sample.astype(bool)]
    if "reference_in_record_subset" in lifecycle:
        lifecycle = lifecycle[
            lifecycle.reference_in_record_subset.astype(bool)
        ]
    rows: list[dict] = []
    columns = {
        "active_peak_lag_hours": "active_deep_convective_area_km2",
        "stratiform_peak_lag_hours": "stratiform_precipitating_high_cloud_area_km2",
        "nonprecip_high_cloud_peak_lag_hours": "nonprecipitating_high_cloud_area_km2",
        "olr_minimum_lag_hours": "olr_w_m2",
    }
    for event_id, event in lifecycle.groupby("event_id", sort=False):
        event = event.sort_values("lag_hours")
        row = {
            "simulation": event.simulation.iloc[0],
            "event_id": event_id,
            "rainfall_decay_fraction": rainfall_decay_fraction,
        }
        for output_name, column in columns.items():
            if column not in event or not np.isfinite(event[column]).any():
                row[output_name] = np.nan
                continue
            index = event[column].idxmin() if column == "olr_w_m2" else event[column].idxmax()
            row[output_name] = float(event.loc[index, "lag_hours"])
        rain = event.area_integrated_24h_precip_km2_mm_day
        peak = float(np.nanmax(rain)) if np.isfinite(rain).any() else np.nan
        after = event[event.lag_hours >= 0]
        below = after[after.area_integrated_24h_precip_km2_mm_day < rainfall_decay_fraction * peak]
        if (
            len(below)
            and "nonprecipitating_high_cloud_area_km2" in event
            and np.isfinite(event.nonprecipitating_high_cloud_area_km2).any()
        ):
            decay_time = float(below.lag_hours.iloc[0])
            cloudy_after = after[
                (after.lag_hours >= decay_time)
                & (
                    after.nonprecipitating_high_cloud_area_km2
                    >= rainfall_decay_fraction
                    * np.nanmax(event.nonprecipitating_high_cloud_area_km2)
                )
            ]
            row["high_cloud_persistence_after_rainfall_hours"] = (
                float(cloudy_after.lag_hours.max() - decay_time)
                if len(cloudy_after)
                else 0.0
            )
        else:
            row["high_cloud_persistence_after_rainfall_hours"] = np.nan
        active_peak = (
            np.nanmax(event.active_deep_convective_area_km2)
            if "active_deep_convective_area_km2" in event
            else np.nan
        )
        post_high = (
            np.nanmean(
                after.loc[
                    after.lag_hours > 0,
                    "nonprecipitating_high_cloud_area_km2",
                ]
            )
            if "nonprecipitating_high_cloud_area_km2" in after
            else np.nan
        )
        row["post_peak_high_cloud_to_peak_active_area_ratio"] = (
            post_high / active_peak if active_peak > 0 else np.nan
        )
        rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_event_composite(
    lifecycle: pd.DataFrame,
    variables: Sequence[str],
    *,
    n_bootstrap: int = 1000,
    seed: int = 20260727,
) -> xr.Dataset:
    """Equal-event composite; complete event sequences are resampled intact."""

    selected = lifecycle[
        lifecycle.complete_primary_lag_window.astype(bool)
        & lifecycle.in_primary_lifecycle_window.astype(bool)
    ].copy()
    if "primary_tropical_sample" in selected:
        selected = selected[selected.primary_tropical_sample.astype(bool)]
    if "reference_in_record_subset" in selected:
        selected = selected[selected.reference_in_record_subset.astype(bool)]
    lags = np.asarray(LIFECYCLE_LAGS_HOURS, dtype=int)
    simulations = [name for name in SIMULATION_DIRS if name in set(selected.simulation)]
    shape = (len(simulations), len(lags), len(variables))
    mean = np.full(shape, np.nan)
    area_weighted = np.full(shape, np.nan)
    precipitation_weighted = np.full(shape, np.nan)
    low = np.full(shape, np.nan)
    high = np.full(shape, np.nan)
    count = np.zeros(len(simulations), dtype=int)
    rng = np.random.default_rng(seed)
    bootstrap_by_simulation: dict[str, np.ndarray] = {}
    for isim, simulation in enumerate(simulations):
        sim = selected[selected.simulation == simulation]
        event_ids = sim.event_id.unique()
        count[isim] = len(event_ids)
        if not len(event_ids):
            continue
        cube = np.full((len(event_ids), len(lags), len(variables)), np.nan)
        area_weights = np.full(len(event_ids), np.nan)
        precipitation_weights = np.full(len(event_ids), np.nan)
        for ievent, event_id in enumerate(event_ids):
            event = sim[sim.event_id == event_id].set_index("lag_hours")
            if "envelope_area_km2" in event:
                area_weights[ievent] = float(np.nanmax(event.envelope_area_km2))
            if "area_integrated_precip_km2_mm_day" in event:
                precipitation_weights[ievent] = float(
                    np.nansum(event.area_integrated_precip_km2_mm_day)
                )
            for ilag, lag in enumerate(lags):
                if lag not in event.index:
                    continue
                for ivar, variable in enumerate(variables):
                    if variable in event:
                        cube[ievent, ilag, ivar] = event.loc[lag, variable]
        mean[isim] = np.nanmean(cube, axis=0)
        for weights, destination in (
            (area_weights, area_weighted),
            (precipitation_weights, precipitation_weighted),
        ):
            for ilag in range(len(lags)):
                for ivar in range(len(variables)):
                    values = cube[:, ilag, ivar]
                    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
                    if valid.any():
                        destination[isim, ilag, ivar] = np.average(
                            values[valid], weights=weights[valid]
                        )
        boot = np.empty((n_bootstrap, len(lags), len(variables)))
        for index in range(n_bootstrap):
            draw = rng.integers(0, len(event_ids), len(event_ids))
            boot[index] = np.nanmean(cube[draw], axis=0)
        bootstrap_by_simulation[simulation] = boot
        low[isim], high[isim] = np.nanpercentile(boot, [2.5, 97.5], axis=0)
    data_vars = {
            "event_equal_mean": (("simulation", "lag_hours", "variable"), mean),
            "area_weighted_mean": (
                ("simulation", "lag_hours", "variable"),
                area_weighted,
            ),
            "precipitation_weighted_mean": (
                ("simulation", "lag_hours", "variable"),
                precipitation_weighted,
            ),
            "ci_low": (("simulation", "lag_hours", "variable"), low),
            "ci_high": (("simulation", "lag_hours", "variable"), high),
            "event_count": (("simulation",), count),
    }
    contrasts = [
        simulation
        for simulation in ("nudged-u", "nudged-T")
        if simulation in bootstrap_by_simulation
        and "control" in bootstrap_by_simulation
    ]
    if contrasts:
        contrast_shape = (len(contrasts), len(lags), len(variables))
        difference_mean = np.full(contrast_shape, np.nan)
        difference_low = np.full(contrast_shape, np.nan)
        difference_high = np.full(contrast_shape, np.nan)
        contrast_event_count = np.zeros((len(contrasts), 2), dtype=int)
        simulation_index = {name: index for index, name in enumerate(simulations)}
        for icontrast, simulation in enumerate(contrasts):
            difference_mean[icontrast] = (
                mean[simulation_index[simulation]]
                - mean[simulation_index["control"]]
            )
            bootstrap_difference = (
                bootstrap_by_simulation[simulation]
                - bootstrap_by_simulation["control"]
            )
            difference_low[icontrast], difference_high[icontrast] = (
                np.nanpercentile(
                    bootstrap_difference,
                    [2.5, 97.5],
                    axis=0,
                )
            )
            contrast_event_count[icontrast] = (
                count[simulation_index[simulation]],
                count[simulation_index["control"]],
            )
        data_vars.update(
            {
                "difference_mean": (
                    ("contrast", "lag_hours", "variable"),
                    difference_mean,
                ),
                "difference_ci_low": (
                    ("contrast", "lag_hours", "variable"),
                    difference_low,
                ),
                "difference_ci_high": (
                    ("contrast", "lag_hours", "variable"),
                    difference_high,
                ),
                "contrast_event_count": (
                    ("contrast", "contrast_member"),
                    contrast_event_count,
                ),
            }
        )
    return xr.Dataset(
        data_vars,
        coords={
            "simulation": simulations,
            "lag_hours": lags,
            "variable": list(variables),
            **({"contrast": contrasts} if contrasts else {}),
            **(
                {"contrast_member": ["experiment", "control"]}
                if contrasts
                else {}
            ),
        },
        attrs={
            "sampling_unit": "complete LPT event",
            "bootstrap": f"{n_bootstrap} complete-event resamples",
            "event_weighting": "equal",
            "contrast_bootstrap": (
                "independent complete-event resampling within each simulation"
            ),
        },
    )


def spacetime_iou(default_mask: xr.DataArray, sensitivity_mask: xr.DataArray) -> float:
    left, right = xr.align(
        default_mask.astype(bool), sensitivity_mask.astype(bool), join="outer", fill_value=False
    )
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    return intersection / union if union else 0.0


def match_sensitivity_events(
    default_events: pd.DataFrame,
    sensitivity_events: pd.DataFrame,
    masks: Mapping[str, xr.DataArray],
) -> pd.DataFrame:
    """One-to-one greedy matching by decreasing space-time mask IoU."""

    candidates = []
    for default_id in default_events.event_id:
        for sensitivity_id in sensitivity_events.event_id:
            iou = spacetime_iou(masks[default_id], masks[sensitivity_id])
            if iou > 0:
                candidates.append((iou, default_id, sensitivity_id))
    candidates.sort(reverse=True)
    used_default: set[str] = set()
    used_sensitivity: set[str] = set()
    rows: list[dict] = []
    default_index = default_events.set_index("event_id")
    sensitivity_index = sensitivity_events.set_index("event_id")
    for iou, default_id, sensitivity_id in candidates:
        if default_id in used_default or sensitivity_id in used_sensitivity:
            continue
        used_default.add(default_id)
        used_sensitivity.add(sensitivity_id)
        left = default_index.loc[default_id]
        right = sensitivity_index.loc[sensitivity_id]
        left_mask, right_mask = xr.align(
            masks[default_id].astype(bool),
            masks[sensitivity_id].astype(bool),
            join="inner",
        )
        common_times = left_mask.Time.values[
            np.asarray((left_mask.any(("latitude", "longitude")) & right_mask.any(("latitude", "longitude"))).values)
        ]
        separations = []
        for time_value in common_times:
            # Mask centroid separation is a stable geometric validation metric;
            # scientific centroids remain precipitation weighted.
            lm = left_mask.sel(Time=time_value)
            rm = right_mask.sel(Time=time_value)
            ly, lx = np.where(lm.values)
            ry, rx = np.where(rm.values)
            if len(lx) and len(rx):
                lat1 = float(np.mean(lm.latitude.values[ly]))
                lat2 = float(np.mean(rm.latitude.values[ry]))
                lon1 = float(
                    np.mod(
                        np.rad2deg(
                            np.angle(
                                np.mean(
                                    np.exp(
                                        1j
                                        * np.deg2rad(
                                            lm.longitude.values[lx]
                                        )
                                    )
                                )
                            )
                        ),
                        360.0,
                    )
                )
                lon2 = float(
                    np.mod(
                        np.rad2deg(
                            np.angle(
                                np.mean(
                                    np.exp(
                                        1j
                                        * np.deg2rad(
                                            rm.longitude.values[rx]
                                        )
                                    )
                                )
                            )
                        ),
                        360.0,
                    )
                )
                dlat = np.deg2rad(lat2 - lat1)
                dlon = np.deg2rad((lon2 - lon1 + 180.0) % 360.0 - 180.0)
                hav = (
                    np.sin(dlat / 2.0) ** 2
                    + np.cos(np.deg2rad(lat1))
                    * np.cos(np.deg2rad(lat2))
                    * np.sin(dlon / 2.0) ** 2
                )
                separations.append(
                    2.0 * 6371.0 * np.arcsin(np.sqrt(hav))
                )
        rows.append(
            {
                "default_event_id": default_id,
                "sensitivity_event_id": sensitivity_id,
                "match_status": "matched",
                "spacetime_iou": iou,
                "peak_time_difference_hours": (
                    pd.Timestamp(right.reference_time)
                    - pd.Timestamp(left.reference_time)
                ).total_seconds()
                / 3600.0,
                "duration_ratio": right.duration_hours / left.duration_hours,
                "integrated_precipitation_ratio": (
                    right.event_integrated_precipitation
                    / left.event_integrated_precipitation
                ),
                "mean_centroid_separation_km": (
                    float(np.mean(separations)) if separations else np.nan
                ),
                "propagation_speed_difference_m_s": (
                    right.zonal_speed_m_s - left.zonal_speed_m_s
                ),
            }
        )
    for event_id in default_events.event_id:
        if event_id not in used_default:
            rows.append(
                {
                    "default_event_id": event_id,
                    "sensitivity_event_id": None,
                    "match_status": "unmatched_default",
                }
            )
    for event_id in sensitivity_events.event_id:
        if event_id not in used_sensitivity:
            rows.append(
                {
                    "default_event_id": None,
                    "sensitivity_event_id": event_id,
                    "match_status": "unmatched_sensitivity",
                }
            )
    return pd.DataFrame(rows)


def _read_standard_masks(standard_directory: Path) -> dict[str, xr.DataArray]:
    index_path = standard_directory / "lpt_event_mask_index.csv"
    if not index_path.exists():
        return {}
    index = pd.read_csv(index_path)
    return {
        row.event_id: xr.open_dataset(
            standard_directory / row.mask_file, chunks={"Time": 1}
        ).lpt_envelope_mask
        for row in index.itertuples()
    }


def build_validation_matches(validation_directory: str | os.PathLike[str]) -> None:
    """Match every sensitivity to the full 15 mm/day validation reference."""

    validation_directory = Path(validation_directory)
    reference = validation_directory / "threshold_15"
    if not reference.exists():
        raise RuntimeError("Validation matching requires the threshold_15 run")
    combined_rows = []
    for sensitivity in sorted(validation_directory.iterdir()):
        if (
            not sensitivity.is_dir()
            or sensitivity.name in {"threshold_15", "figures"}
        ):
            continue
        for simulation in SIMULATION_DIRS:
            default_standard = reference / simulation / "standard"
            sensitivity_standard = sensitivity / simulation / "standard"
            default_path = default_standard / "lpt_events.csv"
            sensitivity_path = sensitivity_standard / "lpt_events.csv"
            if not default_path.exists() or not sensitivity_path.exists():
                continue
            default = pd.read_csv(default_path)
            compared = pd.read_csv(sensitivity_path)
            default = default[
                default.primary_tropical_sample.astype(bool)
                & default.reference_in_record_subset.astype(bool)
            ]
            compared = compared[
                compared.primary_tropical_sample.astype(bool)
                & compared.reference_in_record_subset.astype(bool)
            ]
            masks = {
                **_read_standard_masks(default_standard),
                **_read_standard_masks(sensitivity_standard),
            }
            matching = match_sensitivity_events(default, compared, masks)
            matching.insert(0, "simulation", simulation)
            matching.insert(0, "sensitivity_run", sensitivity.name)
            combined_rows.append(matching)
            matching.to_csv(
                sensitivity_standard / "validation_event_matching.csv",
                index=False,
            )
    if combined_rows:
        pd.concat(combined_rows, ignore_index=True).to_csv(
            validation_directory / "validation_event_matching.csv",
            index=False,
        )


def write_standard_products(
    directory: str | os.PathLike[str],
    *,
    metadata: Mapping,
    track_steps: pd.DataFrame,
    events: pd.DataFrame,
    lifecycle: pd.DataFrame | None = None,
    lifecycle_summary: pd.DataFrame | None = None,
    composites: xr.Dataset | None = None,
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(dict(metadata))
    payload["configuration_sha256"] = _sha256_json(payload)
    (directory / "lpt_configuration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    track_steps.to_csv(directory / "lpt_track_steps.csv", index=False)
    events.to_csv(directory / "lpt_events.csv", index=False)
    if lifecycle is not None:
        lifecycle.to_csv(directory / "lpt_event_relative_diagnostics.csv", index=False)
    if lifecycle_summary is not None:
        lifecycle_summary.to_csv(directory / "lpt_event_lifecycle_summary.csv", index=False)
    if composites is not None:
        composites.attrs["configuration_sha256"] = payload["configuration_sha256"]
        composites.to_netcdf(directory / "lpt_event_equal_composites.nc")


def _simulation_paths(mode_directory: Path, simulation: str) -> dict[str, Path]:
    root = mode_directory / simulation
    return {
        "root": root,
        "preprocessed": root / "preprocessed_precipitation",
        "upstream": root / "upstream_lpt",
        "standard": root / "standard",
    }


def _load_classification(simulation: str, expected_time: xr.DataArray) -> dict:
    cache_dir = Path(diagnostic_cache_dir("classification_cache", root_dir=DEFAULT_ROOT))
    candidates = [
        cache_dir / f"{simulation.replace('-', '_')}_classified_high_cloud.nc",
        cache_dir
        / f"{SIMULATION_LABELS[simulation].replace('$', '').replace('-', '').replace(' ', '_')}_classified_high_cloud.nc",
    ]
    for candidate in candidates:
        if candidate.exists():
            dataset = xr.open_dataset(candidate, chunks={}).sel(
                Time=expected_time.values
            )
            return {name: dataset[name] for name in dataset.data_vars}
    raise RuntimeError(
        f"Missing shared high-cloud classification for {simulation}; run "
        "refined_high_cloud_classification_diagnostics.py first"
    )


def _load_simulation_dataset(root: str, simulation: str) -> xr.Dataset:
    variables = [
        "rainc",
        "rainnc",
        "i_rainc",
        "i_rainnc",
        "olrtoa",
        "cldfrac",
        "qv",
        "qc",
        "qi",
        "qr",
        "qs",
        "rho",
        "theta",
        "u",
        "w",
        "pressure",
        "zgrid",
        "t_isobaric",
        "uzonal_isobaric",
    ]
    dataset = load_latlon_dataset(
        os.path.join(root, SIMULATION_DIRS[simulation]),
        latlon_pattern=LATLON_PATTERN,
        chunks={"Time": 8, "latitude": 90, "longitude": 180},
    )
    return standardize_latlon_dataset(dataset, variables)


def run_specification(
    specification: RunSpecification,
    *,
    simulations: Sequence[str],
    root: str,
    output: Path,
    source: Path,
    stage: str,
) -> None:
    mode_directory = output / specification.name
    for simulation in simulations:
        paths = _simulation_paths(mode_directory, simulation)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        dataset = _load_simulation_dataset(root, simulation)
        precipitation = precipitation_rate_from_dataset(dataset)
        preprocessing_path = paths["root"] / "preprocessing_metadata.json"
        if stage in {"all", "preprocess"}:
            preprocessing = write_upstream_precipitation_files(
                precipitation, paths["preprocessed"], specification
            )
            preprocessing_path.write_text(
                json.dumps(preprocessing, indent=2, sort_keys=True) + "\n"
            )
        else:
            preprocessing = json.loads(preprocessing_path.read_text())
        if stage == "preprocess":
            dataset.close()
            continue
        if stage in {"all", "track"}:
            systems_file = run_upstream_lpt(
                source,
                paths["preprocessed"],
                paths["upstream"],
                specification,
                preprocessing,
            )
            # Preserve original upstream files in place and record their path.
            (paths["root"] / "upstream_systems_file.txt").write_text(
                str(systems_file) + "\n"
            )
        else:
            systems_file = Path(
                (paths["root"] / "upstream_systems_file.txt").read_text().strip()
            )
        if stage == "track":
            dataset.close()
            continue
        if stage in {"all", "postprocess"}:
            steps, events = standardise_upstream_tracks(
                systems_file, simulation, specification
            )
            classification = _load_classification(
                simulation, precipitation.Time
            )
            model_fields = project_model_fields(dataset, precipitation)
            lifecycle, events, lifecycle_summary = postprocess_event_masks(
                upstream_directory=paths["upstream"],
                standard_directory=paths["standard"],
                steps=steps,
                events=events,
                precipitation=precipitation,
                classification=classification,
                model_fields=model_fields,
                specification=specification,
            )
            metadata = configuration_metadata(
                specification, simulation=simulation, upstream_source=source
            )
            write_standard_products(
                paths["standard"],
                metadata=metadata,
                track_steps=steps,
                events=events,
                lifecycle=lifecycle,
                lifecycle_summary=lifecycle_summary,
            )
        dataset.close()
    if stage in {"all", "postprocess"}:
        lifecycle_files = sorted(
            mode_directory.glob(
                "*/standard/lpt_event_relative_diagnostics.csv"
            )
        )
        if lifecycle_files:
            combined = pd.concat(
                [pd.read_csv(path) for path in lifecycle_files],
                ignore_index=True,
            )
            variables = [
                name
                for name in (
                    "area_integrated_24h_precip_km2_mm_day",
                    "active_deep_convective_area_km2",
                    "stratiform_precipitating_high_cloud_area_km2",
                    "nonprecipitating_high_cloud_area_km2",
                    "olr_w_m2",
                )
                if name in combined
            ]
            composite = bootstrap_event_composite(combined, variables)
            composite.attrs.update(
                {
                    "product_version": PRODUCT_VERSION,
                    "run_name": specification.name,
                    "configuration": json.dumps(
                        specification.configuration.as_dict()
                    ),
                }
            )
            composite.to_netcdf(
                mode_directory / "lpt_event_equal_composites.nc"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track large-scale MPAS precipitation envelopes with the published "
            "LPT implementation and build event-relative cloud diagnostics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=("publication", "validation"), default="publication"
    )
    parser.add_argument(
        "--qc",
        action="store_true",
        help="Compatibility alias for --mode validation.",
    )
    parser.add_argument(
        "--stage",
        choices=("all", "preprocess", "track", "postprocess", "figures"),
        default="all",
    )
    parser.add_argument(
        "--simulation",
        choices=(*SIMULATION_DIRS, "all"),
        default="all",
    )
    parser.add_argument("--root", default=os.environ.get("MPAS_SIMULATION_ROOT", DEFAULT_ROOT))
    parser.add_argument(
        "--output",
        default=str(lpt_output_root()),
    )
    parser.add_argument("--upstream-source", default=None)
    parser.add_argument("--figures", nargs="+", type=int, choices=PUBLICATION_FIGURES)
    args = parser.parse_args(argv)
    if args.qc:
        args.mode = "validation"
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output = Path(args.output) / args.mode
    output.mkdir(parents=True, exist_ok=True)
    specs = (
        publication_run_specifications()
        if args.mode == "publication"
        else validation_run_specifications()
    )
    simulations = (
        tuple(SIMULATION_DIRS)
        if args.simulation == "all"
        else (args.simulation,)
    )
    if args.stage == "figures":
        build_figures_from_saved_outputs(
            output,
            mode=args.mode,
            figures=args.figures,
        )
        return 0
    source = upstream_source_path(args.upstream_source)
    verify_upstream_source(source)
    for specification in specs:
        run_specification(
            specification,
            simulations=simulations,
            root=args.root,
            output=output,
            source=source,
            stage=args.stage,
        )
    if args.mode == "validation" and args.stage in {"all", "postprocess"}:
        build_validation_matches(output)
    if args.stage == "all":
        build_figures_from_saved_outputs(
            output,
            mode=args.mode,
            figures=args.figures,
        )
    return 0


def read_standard_event_tables(
    mode_directory: str | os.PathLike[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the project-standard event and lifecycle tables for one run root."""

    mode_directory = Path(mode_directory)
    event_files = sorted(mode_directory.glob("*/standard/lpt_events.csv"))
    lifecycle_files = sorted(
        mode_directory.glob("*/standard/lpt_event_relative_diagnostics.csv")
    )
    events = (
        pd.concat([pd.read_csv(path) for path in event_files], ignore_index=True)
        if event_files
        else pd.DataFrame()
    )
    lifecycle = (
        pd.concat([pd.read_csv(path) for path in lifecycle_files], ignore_index=True)
        if lifecycle_files
        else pd.DataFrame()
    )
    return events, lifecycle


def require_publication_products(
    output: str | os.PathLike[str] | None = None,
) -> Path:
    """Validate the stable publication interface used by downstream scripts."""

    mode_directory = publication_output_directory(output)
    missing: list[str] = []
    for simulation in SIMULATION_DIRS:
        standard = mode_directory / "primary" / simulation / "standard"
        for filename in (
            "lpt_configuration.json",
            "lpt_events.csv",
            "lpt_track_steps.csv",
            "lpt_event_relative_diagnostics.csv",
            "lpt_event_lifecycle_summary.csv",
        ):
            path = standard / filename
            if not path.is_file():
                missing.append(str(path))
                continue
            if filename == "lpt_configuration.json":
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if metadata.get("product_version") != PRODUCT_VERSION:
                    raise RuntimeError(
                        f"{path} has product_version "
                        f"{metadata.get('product_version')!r}; expected "
                        f"{PRODUCT_VERSION!r}. Rebuild the LPT publication run."
                    )
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:8])
        raise FileNotFoundError(
            "Missing standard LPT publication products. Run "
            "lpt_precipitation_envelope_diagnostics.py first:\n"
            f"{preview}"
        )
    return mode_directory


def build_figures_from_saved_outputs(
    output: str | os.PathLike[str],
    *,
    mode: str,
    figures: Sequence[int] | None = None,
) -> None:
    """Regenerate figures exclusively from saved event-level products."""

    output = Path(output)
    figures = set(figures or (PUBLICATION_FIGURES if mode == "publication" else (8,)))
    if mode == "validation":
        _plot_validation_figure(output)
        return
    primary = output / "primary"
    events, lifecycle = read_standard_event_tables(primary)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    if 1 in figures and not events.empty:
        _plot_method_example(primary, events, figure_dir / "figure01_lpt_method.png")
    if 2 in figures and not events.empty:
        _plot_tracking_overview(primary, events, figure_dir / "figure02_lpt_tracking_overview.png")
    if 3 in figures and not events.empty:
        _plot_event_population(events, figure_dir / "figure03_lpt_population.png")
    if 4 in figures and not lifecycle.empty:
        _plot_lifecycle(lifecycle, figure_dir / "figure04_lpt_lifecycle.png")
    if 5 in figures and not lifecycle.empty:
        summary = event_lifecycle_summary(lifecycle)
        _plot_lifecycle_timing(summary, figure_dir / "figure05_lifecycle_timing.png")
    if 6 in figures and not lifecycle.empty:
        _plot_event_centred_evolution(
            primary, lifecycle, figure_dir / "figure06_event_centred_evolution.png"
        )
    if 7 in figures and not lifecycle.empty:
        _plot_dynamical_context(
            lifecycle, figure_dir / "figure07_utls_dynamical_context.png"
        )


def _event_product_path(primary: Path, simulation: str, event_id: str) -> Path:
    index = pd.read_csv(
        primary / simulation / "standard" / "lpt_event_mask_index.csv"
    ).set_index("event_id")
    return primary / simulation / "standard" / index.loc[event_id, "mask_file"]


def _plot_method_example(primary: Path, events: pd.DataFrame, output: Path) -> None:
    candidates = events[
        events.primary_tropical_sample.astype(bool)
        & events.complete_primary_lag_window.astype(bool)
    ].copy()
    if candidates.empty:
        candidates = events[events.primary_tropical_sample.astype(bool)].copy()
    if candidates.empty:
        return
    # A median-area event with branches/restructuring is representative without
    # selecting the cleanest-looking case.
    candidates["representative_score"] = np.abs(
        candidates.maximum_area_km2 - candidates.maximum_area_km2.median()
    )
    branched = candidates[candidates.maximum_simultaneous_branches > 1]
    chosen = (
        branched.sort_values("representative_score").iloc[0]
        if len(branched)
        else candidates.sort_values("representative_score").iloc[0]
    )
    dataset = xr.open_dataset(
        _event_product_path(primary, chosen.simulation, chosen.event_id)
    )
    reference = pd.Timestamp(chosen.reference_time)
    target_times = [reference + pd.Timedelta(hours=lag) for lag in (-48, 0, 48)]
    available = pd.DatetimeIndex(pd.to_datetime(dataset.Time.values))
    target_times = [available[np.argmin(np.abs(available - target))] for target in target_times]
    track_steps = pd.read_csv(
        primary / chosen.simulation / "standard" / "lpt_track_steps.csv",
        parse_dates=["Time"],
    )
    track_steps = track_steps[track_steps.event_id == chosen.event_id]
    fig, axes = plt.subplots(4, 3, figsize=(13, 10), constrained_layout=True, sharex=True, sharey=True)
    for column, time_value in enumerate(target_times):
        raw = dataset.raw_precipitation_mm_day.sel(Time=time_value)
        filtered = dataset.filtered_72h_precipitation_mm_day.sel(Time=time_value)
        mask = dataset.lpt_envelope_mask.sel(Time=time_value)
        regime = dataset.embedded_cloud_regime.sel(Time=time_value)
        axes[0, column].pcolormesh(raw.longitude, raw.latitude, raw, vmin=0, vmax=60, cmap="Blues")
        axes[1, column].pcolormesh(filtered.longitude, filtered.latitude, filtered, vmin=0, vmax=30, cmap="Blues")
        axes[2, column].pcolormesh(filtered.longitude, filtered.latitude, filtered, vmin=0, vmax=30, cmap="Blues")
        axes[2, column].contour(
            filtered.longitude,
            filtered.latitude,
            filtered,
            levels=[PRIMARY_CONFIGURATION.precipitation_threshold_mm_day],
            colors="black",
            linestyles="--",
            linewidths=0.8,
        )
        axes[2, column].contour(
            mask.longitude,
            mask.latitude,
            mask,
            levels=[0.5],
            colors="magenta",
            linewidths=1.5,
        )
        axes[3, column].pcolormesh(regime.longitude, regime.latitude, regime.where(mask), vmin=0.5, vmax=3.5, cmap="viridis")
        for row in range(4):
            recent = track_steps[
                (track_steps.Time <= time_value)
                & (track_steps.Time >= time_value - pd.Timedelta(days=2))
            ]
            axes[row, column].plot(
                recent.centroid_lon,
                recent.centroid_lat,
                color="magenta",
                lw=1.2,
            )
            axes[row, column].set_title(f"{(time_value-reference)/pd.Timedelta(days=1):+.1f} d" if row == 0 else "")
    for axis, label in zip(axes[:, 0], ("Raw six-hour rain", "72-hour + 5° filtered rain", "15 mm/day LPT envelope", "Embedded cloud regimes")):
        axis.set_ylabel(label + "\nlatitude")
    for axis in axes[-1]:
        axis.set_xlabel("longitude")
    fig.suptitle(f"Representative evolving LPT envelope: {chosen.event_id}")
    fig.savefig(output, dpi=300)
    plt.close(fig)
    dataset.close()


def _plot_tracking_overview(primary: Path, events: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True, sharex=True)
    for axis, simulation in zip(axes, SIMULATION_DIRS):
        standard = primary / simulation / "standard"
        overview = xr.open_dataset(standard / "lpt_time_longitude_overview.nc")
        field = overview.equatorial_filtered_precipitation_mm_day
        axis.pcolormesh(field.longitude, pd.to_datetime(field.Time.values), field, vmin=0, vmax=30, cmap="Blues", shading="auto")
        # Show every retained LPT in the tracking overview. The tropical
        # lifecycle rule is a downstream sample definition, not a tracker
        # acceptance criterion.
        sim_events = events[events.simulation == simulation]
        steps = pd.read_csv(standard / "lpt_track_steps.csv", parse_dates=["Time"])
        for event in sim_events.itertuples():
            track = steps[steps.event_id == event.event_id].reset_index(drop=True)
            product = xr.open_dataset(
                _event_product_path(primary, simulation, event.event_id)
            )
            envelope = product.lpt_envelope_mask.sel(
                latitude=slice(
                    -15 if product.latitude[0] < product.latitude[-1] else 15,
                    15 if product.latitude[0] < product.latitude[-1] else -15,
                )
            ).any("latitude")
            axis.contour(
                envelope.longitude,
                pd.to_datetime(envelope.Time.values),
                envelope,
                levels=[0.5],
                colors="white",
                linewidths=0.35,
                alpha=0.8,
            )
            axis.plot(track.centroid_lon, track.Time, color="magenta", lw=1.0)
            axis.scatter(
                [
                    track.iloc[
                        np.argmin(
                            np.abs(
                                track.Time
                                - pd.Timestamp(event.reference_time)
                            )
                        )
                    ].centroid_lon
                ],
                [pd.Timestamp(event.reference_time)],
                color="gold",
                edgecolor="black",
                s=24,
                zorder=4,
            )
            axis.text(track.centroid_lon.iloc[0], track.Time.iloc[0], event.event_id.rsplit("_", 1)[-1], fontsize=5)
            product.close()
        axis.set(ylabel=SIMULATION_LABELS[simulation], xlim=(0, 360))
        overview.close()
    axes[-1].set_xlabel("Longitude")
    fig.suptitle("Full-record 15°S–15°N filtered precipitation and LPT centroid tracks")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _centred_event_array(
    field: xr.DataArray,
    centroid_lon: float,
    centroid_lat: float,
) -> np.ndarray:
    values = np.asarray(field.values, dtype=float)
    ix = int(np.argmin(np.abs(((field.longitude.values - centroid_lon + 180) % 360) - 180)))
    iy = int(np.argmin(np.abs(field.latitude.values - centroid_lat)))
    target_x = values.shape[1] // 2
    target_y = values.shape[0] // 2
    values = np.roll(values, target_x - ix, axis=1)
    shift_y = target_y - iy
    shifted = np.full_like(values, np.nan)
    if shift_y >= 0:
        shifted[shift_y:] = values[: values.shape[0] - shift_y]
    else:
        shifted[:shift_y] = values[-shift_y:]
    return shifted


def _plot_event_centred_evolution(
    primary: Path,
    lifecycle: pd.DataFrame,
    output: Path,
) -> None:
    stages = (-48, 0, 48)
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True, sharex=True, sharey=True)
    for row, simulation in enumerate(SIMULATION_DIRS):
        sim = lifecycle[
            (lifecycle.simulation == simulation)
            & lifecycle.complete_primary_lag_window.astype(bool)
            & lifecycle.primary_tropical_sample.astype(bool)
            & lifecycle.reference_in_record_subset.astype(bool)
        ]
        for column, lag in enumerate(stages):
            precipitation_collect = []
            regimes_collect = {1: [], 2: [], 3: []}
            for event_id in sim.event_id.unique():
                event = sim[sim.event_id == event_id]
                sample = event[event.lag_hours == lag]
                if sample.empty:
                    continue
                dataset = xr.open_dataset(
                    _event_product_path(primary, simulation, event_id)
                )
                time_value = pd.Timestamp(sample.Time.iloc[0])
                precip = dataset.raw_precipitation_mm_day.sel(Time=time_value)
                regime = dataset.embedded_cloud_regime.sel(Time=time_value)
                lon = float(sample.precip_weighted_centroid_lon.iloc[0])
                lat = float(sample.precip_weighted_centroid_lat.iloc[0])
                precipitation_collect.append(
                    _centred_event_array(precip, lon, lat)
                )
                for code in regimes_collect:
                    regimes_collect[code].append(
                        _centred_event_array(regime == code, lon, lat)
                    )
                latitude = precip.latitude.values
                longitude = precip.longitude.values
                dataset.close()
            axis = axes[row, column]
            if precipitation_collect:
                mean_precip = np.nanmean(precipitation_collect, axis=0)
                dx = float(np.median(np.diff(longitude)))
                dy = float(np.median(np.diff(latitude)))
                x = (np.arange(mean_precip.shape[1]) - mean_precip.shape[1] // 2) * dx
                y = (np.arange(mean_precip.shape[0]) - mean_precip.shape[0] // 2) * dy
                axis.pcolormesh(x, y, mean_precip, vmin=0, vmax=30, cmap="Blues", shading="auto")
                for code, colour in ((1, "red"), (2, "gold"), (3, "cyan")):
                    frequency = np.nanmean(regimes_collect[code], axis=0)
                    if np.nanmax(frequency) > 0.1:
                        axis.contour(x, y, frequency, levels=[0.1], colors=colour, linewidths=1.0)
            axis.axvline(0, color="0.5", lw=0.5)
            axis.axhline(0, color="0.5", lw=0.5)
            if row == 0:
                axis.set_title(f"{lag / 24:+.0f} days")
            if column == 0:
                axis.set_ylabel(f"{SIMULATION_LABELS[simulation]}\nrelative latitude (°)")
    for axis in axes[-1]:
        axis.set_xlabel("Longitude relative to precipitation centroid (°)")
    fig.suptitle(
        "Event-centred precipitation; contours: active (red), stratiform (gold), non-precipitating high cloud (cyan)"
    )
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_dynamical_context(lifecycle: pd.DataFrame, output: Path) -> None:
    hypotheses = [
        ("utls_temperature_anomaly", "UTLS temperature anomaly"),
        ("tropopause_static_stability", "Tropopause static stability"),
        ("upper_tropospheric_zonal_wind", "Upper-tropospheric zonal wind"),
        ("upper_tropospheric_vertical_shear", "Vertical shear"),
    ]
    available = [(name, label) for name, label in hypotheses if name in lifecycle and np.isfinite(lifecycle[name]).any()]
    if not available:
        print("Figure 7 not written: no hypothesis-led UTLS/dynamical fields are available")
        return
    fig, axes = plt.subplots(len(available), 2, figsize=(11, 2.5 * len(available)), squeeze=False, constrained_layout=True)
    for row, (variable, label) in enumerate(available):
        composite = bootstrap_event_composite(lifecycle, [variable])
        control = composite.event_equal_mean.sel(simulation="control", variable=variable)
        for simulation in SIMULATION_DIRS:
            if simulation not in composite.simulation:
                continue
            mean = composite.event_equal_mean.sel(simulation=simulation, variable=variable)
            axes[row, 0].plot(composite.lag_hours / 24, mean, color=SIMULATION_COLOURS[simulation], label=SIMULATION_LABELS[simulation])
            if simulation != "control":
                axes[row, 1].plot(composite.lag_hours / 24, mean - control, color=SIMULATION_COLOURS[simulation], label=f"{SIMULATION_LABELS[simulation]} − control")
        axes[row, 0].set_ylabel(label)
        axes[row, 1].axhline(0, color="0.5", lw=0.7)
        for axis in axes[row]:
            axis.axvline(0, color="0.5", ls="--", lw=0.7)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False)
    axes[0, 1].legend(frameon=False)
    axes[-1, 0].set_xlabel("Lag from peak precipitation (days)")
    axes[-1, 1].set_xlabel("Lag from peak precipitation (days)")
    axes[0, 0].set_title("Event-centred environment")
    axes[0, 1].set_title("Experiment minus control")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_event_population(events: pd.DataFrame, output: Path) -> None:
    selected = events[events.primary_tropical_sample.astype(bool)]
    metrics = [
        ("duration_hours", "Duration (h)"),
        ("maximum_area_km2", r"Maximum area (km$^2$)"),
        (
            "event_integrated_precipitation",
            r"Event-integrated precipitation (km$^2$ mm)",
        ),
        ("zonal_speed_m_s", r"Zonal speed (m s$^{-1}$)"),
        ("total_zonal_displacement_km", "Net zonal displacement (km)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10), constrained_layout=True)
    axes[0, 0].set_title("Tropical events per 100 days")
    axes[0, 1].set_title("Envelope area-time coverage (%)")
    for index, simulation in enumerate(SIMULATION_DIRS):
        sim = selected[selected.simulation == simulation]
        if len(sim):
            days = float(sim.analysed_record_days.iloc[0])
            domain_area = float(sim.tracking_domain_area_km2.iloc[0])
            rate = 100.0 * len(sim) / days
            coverage = (
                100.0
                * sim.envelope_area_time_km2_h.sum()
                / (domain_area * days * 24.0)
            )
        else:
            rate = coverage = 0.0
        axes[0, 0].bar(index, rate, color=SIMULATION_COLOURS[simulation])
        axes[0, 1].bar(index, coverage, color=SIMULATION_COLOURS[simulation])
    for axis in axes[0, :2]:
        axis.set(
            xticks=range(3),
            xticklabels=[SIMULATION_LABELS[x] for x in SIMULATION_DIRS],
        )
    distribution_axes = [axes[0, 2], *axes[1, :], axes[2, 0]]
    for axis, (column, label) in zip(distribution_axes, metrics):
        for index, simulation in enumerate(SIMULATION_DIRS):
            values = selected.loc[selected.simulation == simulation, column]
            axis.scatter(
                np.full(len(values), index) + np.linspace(-0.08, 0.08, max(len(values), 1)),
                values,
                s=16,
                alpha=0.65,
                color=SIMULATION_COLOURS[simulation],
            )
            if len(values):
                axis.plot(index, np.nanmedian(values), marker="_", ms=18, color="gold")
        axis.set(xticks=range(3), xticklabels=[SIMULATION_LABELS[x] for x in SIMULATION_DIRS], ylabel=label)
        axis.grid(axis="y", alpha=0.2)
    axes[2, 1].axis("off")
    axes[2, 2].axis("off")
    fig.suptitle("Tropical LPT envelopes; each point is one event")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_lifecycle(lifecycle: pd.DataFrame, output: Path) -> None:
    variables = [
        ("area_integrated_24h_precip_km2_mm_day", "Integrated 24-h precipitation"),
        ("active_deep_convective_area_km2", "Active deep-convective area"),
        ("stratiform_precipitating_high_cloud_area_km2", "Stratiform high-cloud area"),
        ("nonprecipitating_high_cloud_area_km2", "Non-precipitating high-cloud area"),
        ("olr_w_m2", r"OLR (W m$^{-2}$)"),
    ]
    available = [(name, label) for name, label in variables if name in lifecycle]
    composite = bootstrap_event_composite(lifecycle, [name for name, _ in available])
    fig, axes = plt.subplots(
        len(available), 2, figsize=(11, 2.3 * len(available)), squeeze=False, constrained_layout=True
    )
    lags = composite.lag_hours / 24.0
    for row, (variable, label) in enumerate(available):
        for simulation in SIMULATION_DIRS:
            if simulation not in composite.simulation:
                continue
            mean = composite.event_equal_mean.sel(simulation=simulation, variable=variable)
            low = composite.ci_low.sel(simulation=simulation, variable=variable)
            high = composite.ci_high.sel(simulation=simulation, variable=variable)
            event_count = int(
                composite.event_count.sel(simulation=simulation)
            )
            axes[row, 0].plot(
                lags,
                mean,
                color=SIMULATION_COLOURS[simulation],
                label=f"{SIMULATION_LABELS[simulation]} (n={event_count})",
            )
            axes[row, 0].fill_between(lags, low, high, color=SIMULATION_COLOURS[simulation], alpha=0.12)
            if simulation != "control":
                difference = composite.difference_mean.sel(
                    contrast=simulation,
                    variable=variable,
                )
                difference_low = composite.difference_ci_low.sel(
                    contrast=simulation,
                    variable=variable,
                )
                difference_high = composite.difference_ci_high.sel(
                    contrast=simulation,
                    variable=variable,
                )
                axes[row, 1].plot(
                    lags,
                    difference,
                    color=SIMULATION_COLOURS[simulation],
                    label=f"{SIMULATION_LABELS[simulation]} − control",
                )
                axes[row, 1].fill_between(
                    lags,
                    difference_low,
                    difference_high,
                    color=SIMULATION_COLOURS[simulation],
                    alpha=0.12,
                )
        axes[row, 0].set_ylabel(label)
        axes[row, 1].axhline(0, color="0.5", lw=0.7)
        for axis in axes[row]:
            axis.axvline(0, color="0.5", ls="--", lw=0.7)
            axis.grid(alpha=0.2)
    axes[0, 0].legend(frameon=False, ncol=3)
    axes[0, 1].legend(frameon=False)
    axes[-1, 0].set_xlabel("Lag from peak envelope precipitation (days)")
    axes[-1, 1].set_xlabel("Lag from peak envelope precipitation (days)")
    axes[0, 0].set_title("Absolute event-equal composites")
    axes[0, 1].set_title("Experiment minus control")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_lifecycle_timing(summary: pd.DataFrame, output: Path) -> None:
    metrics = [
        ("active_peak_lag_hours", "Active-core peak"),
        ("stratiform_peak_lag_hours", "Stratiform peak"),
        ("nonprecip_high_cloud_peak_lag_hours", "Non-precipitating high-cloud peak"),
        ("olr_minimum_lag_hours", "OLR minimum"),
        ("high_cloud_persistence_after_rainfall_hours", "Post-rain high-cloud persistence"),
        ("post_peak_high_cloud_to_peak_active_area_ratio", "Post-peak high cloud / peak active area"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (column, title) in zip(axes.flat, metrics):
        for index, simulation in enumerate(SIMULATION_DIRS):
            values = summary.loc[summary.simulation == simulation, column].dropna()
            axis.scatter(
                np.full(len(values), index),
                values,
                s=14,
                alpha=0.55,
                color=SIMULATION_COLOURS[simulation],
            )
        axis.axhline(0, color="0.5", lw=0.7)
        axis.set(xticks=range(3), xticklabels=[SIMULATION_LABELS[x] for x in SIMULATION_DIRS], title=title)
    fig.suptitle("Lifecycle timing and conversion; one point per LPT event")
    fig.savefig(output, dpi=300)
    plt.close(fig)


def _plot_validation_figure(output: Path) -> None:
    rows = []
    for run_directory in sorted(output.iterdir()):
        if not run_directory.is_dir() or run_directory.name == "figures":
            continue
        events, _ = read_standard_event_tables(run_directory)
        if events.empty:
            continue
        events["run_name"] = run_directory.name
        rows.append(events)
    if not rows:
        print("Validation figure not written: no saved validation events")
        return
    events = pd.concat(rows, ignore_index=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    threshold_events = events[events.run_name.str.startswith("threshold_")].copy()
    threshold_events["threshold"] = threshold_events.run_name.str.split("_").str[-1].astype(float)
    counts = threshold_events.groupby(["threshold", "simulation"]).size().unstack(fill_value=0)
    counts.plot(ax=axes[0, 0], marker="o", title="Event count versus threshold")
    for simulation in SIMULATION_DIRS:
        subset = threshold_events[threshold_events.simulation == simulation]
        coverage = subset.groupby("threshold").apply(
            lambda group: 100
            * group.envelope_area_time_km2_h.sum()
            / (
                group.tracking_domain_area_km2.iloc[0]
                * group.analysed_record_days.iloc[0]
                * 24
            ),
        )
        axes[0, 0].plot(
            coverage.index,
            coverage,
            ls="--",
            color=SIMULATION_COLOURS[simulation],
            alpha=0.6,
        )
    for simulation in SIMULATION_DIRS:
        subset = threshold_events[threshold_events.simulation == simulation]
        axes[0, 1].scatter(subset.threshold, subset.duration_hours, s=12, alpha=0.5, label=SIMULATION_LABELS[simulation])
        axes[0, 2].scatter(subset.threshold, subset.zonal_speed_m_s, s=12, alpha=0.5)
    axes[0, 1].set_title("Duration versus threshold")
    axes[0, 2].set_title("Propagation speed versus threshold")
    matching_path = output / "validation_event_matching.csv"
    if matching_path.exists():
        matching = pd.read_csv(matching_path)
        sampling = matching[
            matching.sensitivity_run.isin(("sampling_00_12", "sampling_06_18"))
        ]
        recovered = (
            sampling.assign(matched=sampling.match_status == "matched")
            .groupby(["sensitivity_run", "simulation"])
            .matched.mean()
            .unstack(fill_value=0)
        )
        recovered.plot.bar(
            ax=axes[1, 0],
            title="Fraction of default events recovered",
        )
    composites = []
    for threshold in ("threshold_12", "threshold_15", "threshold_18"):
        path = output / threshold / "lpt_event_equal_composites.nc"
        if path.exists():
            composite = xr.open_dataset(path)
            if "area_integrated_24h_precip_km2_mm_day" in composite.variable:
                mean = composite.event_equal_mean.sel(
                    simulation="control",
                    variable="area_integrated_24h_precip_km2_mm_day",
                )
                axes[1, 1].plot(
                    composite.lag_hours / 24,
                    mean,
                    label=threshold.replace("threshold_", "") + " mm/day",
                )
            composites.append(composite)
    axes[1, 1].set_title("Primary lifecycle versus threshold")
    axes[1, 1].legend(frameon=False)
    halves = events[
        events.run_name.isin(("record_first_half", "record_second_half"))
    ]
    if len(halves):
        halves.groupby(["run_name", "simulation"]).size().unstack(fill_value=0).plot.bar(
            ax=axes[1, 2],
            title="First- and second-half event counts",
        )
    for composite in composites:
        composite.close()
    for axis in axes.flat:
        axis.tick_params(axis="x", rotation=35)
        axis.grid(alpha=0.2)
    axes[0, 1].legend(frameon=False)
    fig.savefig(figure_dir / "validation_lpt_sensitivity.png", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
