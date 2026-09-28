#!/usr/bin/env python3
"""Large precipitation object tracking and validation for MPAS-A.

This module is the stable project-side interface to the published LPT code.
Object identification, overlap linking, and merger/split handling are delegated
to the unmodified upstream LPT driver. This project entry point owns MPAS
preprocessing, LPO identification, precipitation event diagnostics, population
summaries, standard event masks and tables, and tracking-validation figures and
movies. Cloud/environment sampling and lifecycle composites belong to
``lpo_lifecycle_analysis.py``.

The exact upstream revision is pinned in
``external/lpt-python-public.lock.json`` and installed once at
``/store/atmos-adk33/cwp29/mpas-a/external/lpt-python-public``. Runs use that
shared checkout automatically; no source-path command-line option or
environment variable is required.

The adapter rejects a different commit or modified tracked source. Original
upstream object, system, composite-mask, and individual-mask outputs are
preserved below each run's ``upstream_lpt`` directory.

Large and regenerable products are kept below the central ``cache/lpt`` location.
Only rendered figures are written below the
usual ``~/tmp/mpas/crf_on/final/lpo_tracking`` figure tree. The tracking
script does not expose an alternative data-location option.

Publication Figure 1 writes one envelope-evolution file for every retained LPO.
Each file also shows concurrent LPO boundaries. Figure 2 writes one overview
per simulation with side-by-side 100-day panels. It shows every threshold and
size-qualified candidate envelope, but only retained LPOs receive centroid
tracks. A 15-day sliding-window MP4 uses the same presentation. Test mode uses
one panel for its short subset. Figure 3 separates property CDFs from compact
population summaries. Figure 4 maps every retained LPO track within 10°S–10°N.
All figures use saved standard products.

Publication mode is the default. It applies one configuration to all three
simulations: six-hourly precipitation, a complete 72-hour running mean,
Gaussian sigma 5 degrees, 12 mm/day threshold, 100 contiguous cells on the
0.5-degree MPAS grid, periodic longitude, and seven-day minimum duration. It
uses the Kerns--Chen object definition with two dataset-motivated linking
adaptations. Below-threshold reconnection is limited to 24 hours. Ambiguous
merger and split paths retain a largest-area lineage. Validation uses the first
100 tracking days from control only. It checks 11, 12, and 13 mm/day thresholds,
0-, 24-, and 72-hour gaps, 0.0, 0.1, and 0.2 mutual-overlap floors, and both
merger/split settings. The primary run supplies every shared reference value.

``--test`` is a quick visual check on a short, contiguous subset of the actual
MPAS records. It retains the publication threshold, filters, grid-cell
criterion, periodic geometry, and linking method. Tracking Figures 1--4 use
the complete saved test population; the envelope gallery and tracking overview
show every retained test track. Test products are isolated below
``testing/actual_subset_*`` and cannot be consumed as definitive publication
products. The test-only minimum retained duration is one day: the five-point
minimum required by the unmodified upstream spline smoother. Publication and
validation remain fixed at seven days.

The publication ``standard`` directories are the downstream interface. They
contain event and track-step tables, per-event masks, precipitation-relative
timing, population summaries, and complete configuration metadata.
``lpo_lifecycle_analysis.py`` consumes this interface. This script also owns
the LPO population uncertainty summaries.

Method citation: Kerns and Chen (2020), JGR Atmospheres,
doi:10.1029/2019JD032142; https://github.com/uwincm/lpt-python-public.
Track-validation figures follow the time-longitude presentation in Kerns and
Chen (2016), their Figure 4a, and the upstream
``lpt.plotting.plot_timelon_with_lpt`` implementation. They deliberately omit
the paper's MJO-index panel because propagation and wave behaviour are
diagnosed after tracking in this workflow.
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
import time
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
import cartopy.crs as ccrs
from matplotlib import animation, pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FuncFormatter, MaxNLocator
from cartopy.mpl.ticker import LatitudeFormatter, LongitudeFormatter
from scipy import ndimage

from diagnostics.diagnostic_helpers import (
    FigureSaver,
    SIMULATION_COLOURS,
    LATLON_PATTERN,
    cache_manifest,
    cache_manifest_matches,
    diagnostic_output_dir,
    grid_cell_area_km2,
    high_cloud_class_cmap_norm,
    high_cloud_class_legend_handles,
    load_latlon_dataset,
    netcdf_safe_dataset,
    precip_rate_cmap_norm,
    precipitation_rate_from_dataset,
    progress,
    read_json_file,
    standardize_latlon_dataset,
    write_csv_atomic,
    write_json_file,
    write_netcdf_atomic,
)
from diagnostics.analysis_config import (
    ACTIVE_SIMULATION_PROFILE,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SIMULATION_ARCHIVES,
    SIMULATION_PUBLICATION_LABELS,
    SIMULATION_ROOT,
    cache_directory,
)
from diagnostics.uncertainty import independent_bootstrap_difference
from diagnostics.waves.wave_metrics import sample_masked_field
from diagnostics.convection.high_cloud_classification import (
    HighCloudThresholds,
    cached_classification_is_current,
)


UPSTREAM_REPOSITORY = "https://github.com/uwincm/lpt-python-public.git"
UPSTREAM_COMMIT = "be94a3e7afbd107b2cd852fcf166493e1c3fe472"
UPSTREAM_SOURCE = Path(
    "/store/atmos-adk33/cwp29/mpas-a/external/lpt-python-public"
)
UPSTREAM_CITATION = "Kerns and Chen (2020), doi:10.1029/2019JD032142"
TRACK_FIGURE_CITATION = "Kerns and Chen (2016), doi:10.1002/2015JD024661, Figure 4"
PRODUCT_VERSION = "mpas-lpt-precipitation-envelope-v18-rain-volume-units"
TRACKING_CACHE_VERSION = "mpas-upstream-lpt-tracking-v3-strict-linking"
OWNER_CACHE_SCHEMA = "lpo-owner-v1-source-signed-standard-products"
TRACKING_OVERVIEW_LATITUDE_BAND = (
    "20S-20N cosine-latitude weighted mean"
)
UPSTREAM_TRACKING_METADATA_FILENAME = "upstream_tracking_metadata.json"
DISABLED_ABSOLUTE_OVERLAP_POINT_THRESHOLD = 2**63 - 1
UPSTREAM_STRICT_HALF_OVERLAP_THRESHOLD = float(np.nextafter(0.5, 0.0))
KERNS_CHEN_GRID_SPACING_DEGREES = 0.25
KERNS_CHEN_MINIMUM_OBJECT_GRID_CELLS = 400
MPAS_LPT_GRID_SPACING_DEGREES = 0.5
MPAS_MINIMUM_OBJECT_GRID_CELLS = int(
    round(
        KERNS_CHEN_MINIMUM_OBJECT_GRID_CELLS
        * (
            KERNS_CHEN_GRID_SPACING_DEGREES
            / MPAS_LPT_GRID_SPACING_DEGREES
        )
        ** 2
    )
)

SIMULATION_DIRS = dict(SIMULATION_ARCHIVES)
SIMULATION_LABELS = dict(SIMULATION_PUBLICATION_LABELS)
CENTROID_TRACK_COLOUR = "black"
LPO_ENVELOPE_COLOUR = "purple"
CENTROID_TRACK_LINEWIDTH = 2.4

TRACKING_FIGURES = (1, 2, 3, 4)
PUBLICATION_FIGURES = TRACKING_FIGURES
VALIDATION_FIGURES = (8,)
VALIDATION_RUN_NAMES = (
    "threshold_11",
    "threshold_12",
    "threshold_13",
    "gap_0",
    "gap_72",
    "overlap_0",
    "overlap_0p2",
    "merge_split_false",
)
VALIDATION_SIMULATIONS = ("control",)
VALIDATION_RECORD_DAYS = 100
VALIDATION_PARAMETER_FAMILIES = (
    (
        "Precipitation threshold",
        "mm day$^{-1}$",
        (("threshold_11", "11"), ("threshold_12", "12"), ("threshold_13", "13")),
    ),
    (
        "Gap allowance",
        "hours",
        (("gap_0", "0"), ("threshold_12", "24"), ("gap_72", "72")),
    ),
    (
        "Mutual-overlap floor",
        "fraction",
        (("overlap_0", "0.0"), ("threshold_12", "0.1"), ("overlap_0p2", "0.2")),
    ),
    (
        "Largest-area lineage",
        "setting",
        (("merge_split_false", "off"), ("threshold_12", "on")),
    ),
)
VALIDATION_MATCH_COLUMNS = (
    "default_event_id",
    "sensitivity_event_id",
    "match_status",
    "spacetime_iou",
    "peak_time_difference_hours",
    "duration_ratio",
    "integrated_precipitation_ratio",
    "mean_centroid_separation_km",
    "propagation_speed_difference_m_s",
)
DEFAULT_TEST_DAYS = 10
MINIMUM_TEST_DAYS = 4
DEFAULT_TEST_MINIMUM_DURATION_DAYS = 1.0
MINIMUM_UPSTREAM_SMOOTHING_DURATION_DAYS = 1.0
TEST_LIFECYCLE_LAGS_HOURS = LIFECYCLE_LAGS_HOURS = tuple(
    range(-72, 73, 6)
)
METHOD_EXAMPLE_LAGS_HOURS = (-72, -48, -24, 0, 24, 48, 72)
TRACKING_MOVIE_WINDOW_DAYS = 15
TRACKING_MOVIE_FRAME_STEP_HOURS = 24
TRACKING_MOVIE_FRAMES_PER_SECOND = 6
TROPICAL_CENTROID_LIMIT_DEG = 20.0
TRACK_MAP_LATITUDE_LIMIT_DEG = 10.0
TROPICAL_MIN_CONSECUTIVE_HOURS = 24.0
DEFAULT_ROOT = str(SIMULATION_ROOT)
DEFAULT_LPT_CACHE_ROOT = os.fspath(cache_directory("lpt"))
DEFAULT_LPT_FIGURE_ROOT = "/home/cwp29/tmp/mpas/crf_on/final"
DEFAULT_TRACKING_DOMAIN = (-50.0, 50.0)


@dataclasses.dataclass(frozen=True)
class LPTConfiguration:
    """Complete physical and linking configuration for one LPT run."""

    precipitation_threshold_mm_day: float = 12.0
    input_interval_hours: int = 6
    sampling_hours_utc: tuple[int, ...] = (0, 6, 12, 18)
    running_mean_hours: int = 72
    gaussian_sigma_degrees: float = 5.0
    gaussian_truncate_sigma: float = 3.0
    minimum_object_grid_cells: int = MPAS_MINIMUM_OBJECT_GRID_CELLS
    periodic_longitude: bool = True
    minimum_overlap_fraction: float = 0.5
    # Require the configured mutual-overlap floor. Disable the upstream
    # resolution-dependent grid-cell-count shortcut separately.
    bare_minimum_overlap_fraction: float = 0.10
    minimum_overlap_points: int = DISABLED_ABSOLUTE_OVERLAP_POINT_THRESHOLD
    minimum_overlap_area_km2: float = 0.0
    allow_merge_split: bool = True
    break_up_merge_split: bool = True
    split_merger_min_hours: int = 3 * 24
    minimum_duration_hours: int = 7 * 24
    allow_test_duration_below_seven_days: bool = False
    fall_below_threshold_max_hours: int = 24
    tracking_latitude_bounds: tuple[float, float] = DEFAULT_TRACKING_DOMAIN
    lifecycle_lags_hours: tuple[int, ...] = LIFECYCLE_LAGS_HOURS
    tropical_centroid_limit_degrees: float = TROPICAL_CENTROID_LIMIT_DEG
    tropical_min_consecutive_hours: float = TROPICAL_MIN_CONSECUTIVE_HOURS

    def __post_init__(self) -> None:
        if self.running_mean_hours % self.input_interval_hours:
            raise ValueError(
                "running mean must contain a whole number of input intervals"
            )
        expected = 24 // self.input_interval_hours
        if len(self.sampling_hours_utc) != expected:
            raise ValueError(
                "sampling_hours_utc is inconsistent with input_interval_hours"
            )
        if sorted(set(self.sampling_hours_utc)) != list(self.sampling_hours_utc):
            raise ValueError("sampling_hours_utc must be unique and sorted")
        if self.minimum_object_grid_cells < 1:
            raise ValueError("minimum_object_grid_cells must be positive")
        if (
            self.minimum_duration_hours < 7 * 24
            and not self.allow_test_duration_below_seven_days
        ):
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


def lpo_tracking_output_root() -> Path:
    """Return the data-repository root for LPT caches and saved products."""

    return Path(DEFAULT_LPT_CACHE_ROOT)


def lpo_tracking_figure_root() -> Path:
    """Return the home output root used only for rendered LPT figures."""

    return Path(diagnostic_output_dir(DEFAULT_LPT_FIGURE_ROOT))


def publication_output_directory() -> Path:
    """Return the stable root of the definitive LPT publication products."""

    return lpo_tracking_output_root() / "publication"


def actual_test_output_directory(
    output: str | os.PathLike[str],
    *,
    start: str | None,
    days: int,
    minimum_duration_hours: int,
) -> Path:
    """Return an isolated publication-shaped tree for an actual-data subset."""

    return (
        Path(output)
        / "testing"
        / actual_test_label(
            start=start,
            days=days,
            minimum_duration_hours=minimum_duration_hours,
        )
        / "publication"
    )


def actual_test_label(
    *,
    start: str | None,
    days: int,
    minimum_duration_hours: int,
) -> str:
    """Return the shared directory label for actual-data diagnostic tests."""

    start_label = "auto" if start is None else str(start)
    start_label = (
        start_label.replace(":", "")
        .replace(" ", "_")
        .replace("/", "-")
    )
    return (
        f"actual_subset_{start_label}_{days:03d}days_"
        f"min{minimum_duration_hours:03d}h"
    )


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


def actual_data_test_run_specifications(
    minimum_duration_days: float,
) -> list[RunSpecification]:
    """Publication method with only a clearly marked test retention cutoff."""

    minimum_duration_hours = int(round(24.0 * minimum_duration_days))
    configuration = dataclasses.replace(
        PRIMARY_CONFIGURATION,
        minimum_duration_hours=minimum_duration_hours,
        allow_test_duration_below_seven_days=True,
        lifecycle_lags_hours=TEST_LIFECYCLE_LAGS_HOURS,
    )
    return [RunSpecification("primary", configuration)]


def validation_run_specifications() -> list[RunSpecification]:
    """Return the control-only 100-day sensitivity suite."""

    specs: list[RunSpecification] = []
    for threshold in (11.0, 12.0, 13.0):
        name = f"threshold_{threshold:g}"
        specs.append(
            RunSpecification(
                name,
                dataclasses.replace(
                    PRIMARY_CONFIGURATION,
                    precipitation_threshold_mm_day=threshold,
                ),
                record_subset="first_100_days",
                reference_run=(
                    None if threshold == 12.0 else "threshold_12"
                ),
            )
        )
    for hours in (0, 72):
        specs.append(
            RunSpecification(
                f"gap_{hours}",
                dataclasses.replace(
                    PRIMARY_CONFIGURATION,
                    fall_below_threshold_max_hours=hours,
                ),
                record_subset="first_100_days",
                reference_run="threshold_12",
            )
        )
    for overlap, suffix in ((0.0, "0"), (0.2, "0p2")):
        specs.append(
            RunSpecification(
                f"overlap_{suffix}",
                dataclasses.replace(
                    PRIMARY_CONFIGURATION,
                    bare_minimum_overlap_fraction=overlap,
                ),
                record_subset="first_100_days",
                reference_run="threshold_12",
            )
        )
    specs.append(
        RunSpecification(
            "merge_split_false",
            dataclasses.replace(
                PRIMARY_CONFIGURATION,
                break_up_merge_split=False,
            ),
            record_subset="first_100_days",
            reference_run="threshold_12",
        )
    )
    return specs


def configuration_metadata(
    specification: RunSpecification,
    *,
    simulation: str,
    upstream_source: str | os.PathLike[str],
    test_subset: Mapping | None = None,
) -> dict:
    configuration = specification.configuration
    validation_departures = []
    if (
        configuration.precipitation_threshold_mm_day
        != PRIMARY_CONFIGURATION.precipitation_threshold_mm_day
    ):
        validation_departures.append(
            "precipitation threshold changed from "
            f"{PRIMARY_CONFIGURATION.precipitation_threshold_mm_day:g} to "
            f"{configuration.precipitation_threshold_mm_day:g} mm day-1"
        )
    if (
        configuration.running_mean_hours
        != PRIMARY_CONFIGURATION.running_mean_hours
    ):
        validation_departures.append(
            "temporal running mean changed from "
            f"{PRIMARY_CONFIGURATION.running_mean_hours} to "
            f"{configuration.running_mean_hours} hours"
        )
    if (
        configuration.gaussian_sigma_degrees
        != PRIMARY_CONFIGURATION.gaussian_sigma_degrees
    ):
        validation_departures.append(
            "Gaussian spatial-filter sigma changed from "
            f"{PRIMARY_CONFIGURATION.gaussian_sigma_degrees:g} to "
            f"{configuration.gaussian_sigma_degrees:g} degrees"
        )
    if configuration.sampling_hours_utc != PRIMARY_CONFIGURATION.sampling_hours_utc:
        validation_departures.append(
            "temporal sampling changed from all six-hourly times to "
            + ", ".join(f"{hour:02d} UTC" for hour in configuration.sampling_hours_utc)
        )
    if (
        configuration.fall_below_threshold_max_hours
        != PRIMARY_CONFIGURATION.fall_below_threshold_max_hours
    ):
        validation_departures.append(
            "below-threshold gap allowance changed from "
            f"{PRIMARY_CONFIGURATION.fall_below_threshold_max_hours} to "
            f"{configuration.fall_below_threshold_max_hours} hours"
        )
    if (
        configuration.bare_minimum_overlap_fraction
        != PRIMARY_CONFIGURATION.bare_minimum_overlap_fraction
    ):
        validation_departures.append(
            "mutual-overlap floor changed from "
            f"{PRIMARY_CONFIGURATION.bare_minimum_overlap_fraction:g} to "
            f"{configuration.bare_minimum_overlap_fraction:g}"
        )
    if (
        configuration.break_up_merge_split
        != PRIMARY_CONFIGURATION.break_up_merge_split
    ):
        validation_departures.append(
            "largest-area merger/split lineage handling changed from "
            f"{PRIMARY_CONFIGURATION.break_up_merge_split} to "
            f"{configuration.break_up_merge_split}"
        )
    try:
        project_commit = subprocess.run(
            ["git", "-C", str(Path(__file__).resolve().parents[2]), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            "Cannot record the required analysis Git revision"
        ) from exc
    metadata = {
        "product_version": PRODUCT_VERSION,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "simulation": simulation,
        "run_name": specification.name,
        "record_subset": specification.record_subset,
        "reference_run": specification.reference_run,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "upstream_source": str(upstream_source),
        "upstream_source_modified": False,
        "project_git_commit": project_commit,
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "xarray_version": xr.__version__,
        "citation": UPSTREAM_CITATION,
        "track_figure_presentation_citation": TRACK_FIGURE_CITATION,
        "track_figure_centroid_source": (
            "upstream centroid_lon, centroid_lat, and area_km2 from "
            "lpt_track_steps.csv"
        ),
        "lifecycle_centroid_source": (
            "precipitation-weighted envelope centroids remain saved in "
            "lpt_event_relative_diagnostics.csv"
        ),
        "configuration": configuration.as_dict(),
        "validation_departures_from_primary": validation_departures,
        "selection": {
            "retain_all_valid_lpt_tracks": True,
            "primary_tropical_sample": (
                "precipitation-weighted envelope centroid within "
                f"{configuration.tropical_centroid_limit_degrees:g} degrees "
                "of the equator for at least "
                f"{configuration.tropical_min_consecutive_hours:g} "
                "consecutive hours"
            ),
            "dynamical_selection_criteria": [],
            "lifecycle_reference": (
                "maximum trailing-24-hour integrated rain volume inside "
                "the tracked envelope"
            ),
            "primary_composite_requires_complete_lag_window": True,
            "event_weighting": "one equal weight per LPT event",
        },
        "kerns_chen_2020_methodology": {
            "method_summary": (
                "Kerns--Chen object definition with dataset-motivated "
                "linking adaptations"
            ),
            "matching_elements": [
                f"complete trailing {configuration.running_mean_hours}-hour "
                "precipitation mean",
                "Gaussian spatial smoothing with "
                f"{configuration.gaussian_sigma_degrees:g}-degree standard "
                "deviation and three-standard-deviation support",
                "strictly greater than "
                f"{configuration.precipitation_threshold_mm_day:g} mm day-1 "
                "LPO threshold",
                "minimum 100 contiguous cells on the 0.5-degree MPAS grid, "
                "scaled by cell count from 400 cells on the paper's "
                "0.25-degree grid; both upstream object-size area filters "
                "are disabled",
                "four-neighbour object connectivity",
                "periodic longitude",
                "half-of-either-object fractional overlap linking; the "
                "upstream absolute-grid-cell alternative is disabled",
                f"{configuration.bare_minimum_overlap_fraction:g} "
                "mutual-overlap floor",
                "minimum retained duration of seven days after reconnection",
                "tracking over 50S-50N",
            ],
            "dataset_motivated_linking_adaptations": [
                (
                    "below-threshold reconnection is limited to "
                    f"{configuration.fall_below_threshold_max_hours} hours "
                    "because the model record is complete and rainfall is "
                    "already averaged over 72 hours"
                ),
                (
                    "ambiguous merger/split paths are reduced to a "
                    "largest-area lineage so retained events do not duplicate "
                    "shared branches"
                    if configuration.break_up_merge_split
                    else "ambiguous merger/split paths remain joined for this "
                    "controlled sensitivity run"
                ),
            ],
            "deliberate_project_differences": [
                "six-hourly MPAS input rather than three-hourly TMPA input",
                (
                    "the first 100 complete tracking days are used for each "
                    "controlled sensitivity"
                    if specification.record_subset == "first_100_days"
                    else "the complete approximately 200-day simulation is "
                    "tracked in one run rather than overlapping annual "
                    "tracking windows"
                ),
                "no eastward-propagation or other MJO classification is used "
                "to select tracks",
                "the primary tropical sample requires a precipitation-weighted "
                "centroid within "
                f"{configuration.tropical_centroid_limit_degrees:g} degrees "
                "latitude for "
                f"{configuration.tropical_min_consecutive_hours:g} "
                "consecutive hours",
                "project diagnostics use the instantaneous threshold-envelope "
                "mask; paper rainfall attribution also used a one-sigma "
                "peripheral zone and the preceding three-day accumulation",
                "event lifecycle timing and embedded cloud/environment "
                "diagnostics are project additions",
            ],
        },
        "downstream_lifecycle_analysis": {
            "script": "lpo_lifecycle_analysis.py",
            "tracking_products_are_not_modified": True,
            "event_identification_and_reference_times_remain_authoritative": True,
        },
    }
    if test_subset is not None:
        if (
            specification.configuration.minimum_duration_hours
            != PRIMARY_CONFIGURATION.minimum_duration_hours
        ):
            methodology = metadata["kerns_chen_2020_methodology"]
            methodology["matching_elements"].remove(
                "minimum retained duration of seven days after reconnection"
            )
            methodology["deliberate_project_differences"].append(
                "the isolated actual-data figure test retains tracks for "
                f"{specification.configuration.minimum_duration_hours} hours "
                "instead of seven days; publication and validation retain "
                "the paper's seven-day minimum"
            )
        metadata["kerns_chen_2020_methodology"][
            "deliberate_project_differences"
        ].append(
            "the isolated actual-data tracking test records a -48 to +48-hour "
            "precipitation-relative timing window for downstream lifecycle "
            "testing"
        )
        metadata["test_subset"] = {
            **dict(test_subset),
            "uses_actual_simulation_data": True,
            "publication_minimum_duration_hours": (
                PRIMARY_CONFIGURATION.minimum_duration_hours
            ),
            "test_minimum_duration_hours": (
                specification.configuration.minimum_duration_hours
            ),
            "tracking_configuration_unchanged_apart_from_duration": True,
            "diagnostic_scope": (
                "tracking-validation check; the Figure 1 envelope gallery "
                "and Figure 2 overview use every retained test event"
            ),
            "figures_produced": list(PUBLICATION_FIGURES),
            "method_figure": (
                "one file per retained test event; unfiltered and "
                "Gaussian-filtered 72-hour precipitation following Kerns and "
                "Chen (2020) Figure 1"
            ),
            "displayed_lpo_threshold_mm_day": (
                specification.configuration.precipitation_threshold_mm_day
            ),
            "threshold_comparison": (
                "The 12 mm day-1 threshold follows Kerns and Chen (2020) "
                "and is applied consistently to MPAS tracking and figures"
            ),
            "high_cloud_classification_sampling": (
                "two-dimensional high-cloud context for tracking examples"
            ),
            "native_level_lifecycle_sampling_performed": False,
            "lifecycle_lag_window_hours": list(
                specification.configuration.lifecycle_lags_hours
            ),
            "event_equal_lifecycle_composite_produced": False,
            "definitive_publication_product": False,
        }
    return metadata


def _sha256_json(value: Mapping) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _tracking_signature_payload(
    specification: RunSpecification,
    preprocessing_metadata: Mapping,
) -> dict:
    """Return the complete project-side signature for upstream LPT tracking."""

    return {
        "tracking_cache_version": TRACKING_CACHE_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "run_name": specification.name,
        "record_subset": specification.record_subset,
        "configuration": specification.configuration.as_dict(),
        "preprocessing": dict(preprocessing_metadata),
    }


def write_upstream_tracking_metadata(
    path: str | os.PathLike[str],
    *,
    specification: RunSpecification,
    preprocessing_metadata: Mapping,
    systems_file: str | os.PathLike[str],
) -> None:
    """Record the exact configuration that produced one upstream track file."""

    payload = _tracking_signature_payload(
        specification,
        preprocessing_metadata,
    )
    metadata = {
        **payload,
        "tracking_sha256": _sha256_json(payload),
        "systems_file": str(Path(systems_file).resolve()),
    }
    Path(path).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def require_current_upstream_tracking(
    path: str | os.PathLike[str],
    *,
    specification: RunSpecification,
    preprocessing_metadata: Mapping,
    systems_file: str | os.PathLike[str],
) -> None:
    """Reject upstream tracks made with a stale project-side configuration."""

    path = Path(path)
    if not path.is_file():
        raise RuntimeError(
            f"Missing {path}; rerun LPT with --stage track before "
            "--stage postprocess"
        )
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected_payload = _tracking_signature_payload(
        specification,
        preprocessing_metadata,
    )
    saved_payload = {
        key: metadata.get(key)
        for key in expected_payload
    }
    saved_signature = metadata.get("tracking_sha256")
    if (
        saved_signature != _sha256_json(saved_payload)
        or saved_signature != _sha256_json(expected_payload)
    ):
        raise RuntimeError(
            f"{path} predates the current 100-cell LPO criterion or otherwise "
            "has a stale tracking configuration; rerun --stage track"
        )
    recorded_systems_file = Path(
        metadata.get("systems_file", "")
    ).resolve()
    expected_systems_file = Path(systems_file).resolve()
    if recorded_systems_file != expected_systems_file:
        raise RuntimeError(
            f"{path} records {recorded_systems_file}, not the selected "
            f"upstream file {expected_systems_file}; rerun --stage track"
        )
    if not expected_systems_file.is_file():
        raise FileNotFoundError(
            f"Recorded upstream systems file does not exist: "
            f"{expected_systems_file}"
        )


def upstream_source_path() -> Path:
    """Return the shared checkout of the pinned upstream LPT source."""

    if (UPSTREAM_SOURCE / "lpt" / "helpers.py").is_file():
        return UPSTREAM_SOURCE.resolve()
    raise RuntimeError(
        "Published LPT source was not found at the configured shared path "
        f"{UPSTREAM_SOURCE}. Expected {UPSTREAM_REPOSITORY} at commit "
        f"{UPSTREAM_COMMIT}."
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
    # MPAS lat-lon coordinates are commonly stored as float32. A nominal
    # 0.5-degree grid then has spacing jitter of order 1e-5 degrees, which is
    # representational rather than a physically irregular grid.
    coordinate_rtol = 1.0e-4
    coordinate_atol = 1.0e-6
    if not np.allclose(
        dlat,
        np.median(dlat),
        rtol=coordinate_rtol,
        atol=coordinate_atol,
    ):
        raise ValueError("LPT preprocessing requires regularly spaced latitude")
    if not np.allclose(
        dlon,
        np.median(dlon),
        rtol=coordinate_rtol,
        atol=coordinate_atol,
    ):
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


def periodic_longitude_field(field: xr.DataArray) -> xr.DataArray:
    """Return a complete periodic field on a sorted 0--360 degree axis."""

    longitude = np.mod(np.asarray(field.longitude.values, dtype=float), 360.0)
    if len(np.unique(longitude)) != len(longitude):
        raise ValueError(
            "Periodic longitude conversion produced duplicate coordinates"
        )
    return field.assign_coords(longitude=longitude).sortby("longitude")


def equatorial_envelope_for_plot(
    envelope: xr.DataArray,
    latitude_limit_degrees: float = TROPICAL_CENTROID_LIMIT_DEG,
) -> xr.DataArray:
    """Project an LPO mask into a 0--360 longitude--time boundary field."""

    latitude = envelope.latitude
    projected = envelope.sel(
        latitude=slice(
            -latitude_limit_degrees
            if latitude[0] < latitude[-1]
            else latitude_limit_degrees,
            latitude_limit_degrees
            if latitude[0] < latitude[-1]
            else -latitude_limit_degrees,
        )
    ).any("latitude")
    return periodic_longitude_field(projected)


def _periodic_candidate_objects_2d(
    above_threshold: np.ndarray,
    *,
    minimum_grid_cells: int,
) -> np.ndarray:
    """Retain size-qualified four-neighbour objects across the longitude seam."""

    mask = np.asarray(above_threshold, dtype=bool)
    structure = np.asarray(
        ((0, 1, 0), (1, 1, 1), (0, 1, 0)),
        dtype=np.int8,
    )
    labels, label_count = ndimage.label(mask, structure=structure)
    if label_count == 0:
        return np.zeros_like(mask, dtype=bool)
    parents = np.arange(label_count + 1)

    def root(label: int) -> int:
        while parents[label] != label:
            parents[label] = parents[parents[label]]
            label = parents[label]
        return int(label)

    for first, last in zip(labels[:, 0], labels[:, -1]):
        if first == 0 or last == 0:
            continue
        first_root = root(int(first))
        last_root = root(int(last))
        if first_root != last_root:
            parents[last_root] = first_root
    roots = np.asarray([root(label) for label in range(label_count + 1)])
    merged_labels = roots[labels]
    cell_counts = np.bincount(merged_labels.ravel())
    retained = cell_counts >= minimum_grid_cells
    retained[0] = False
    return retained[merged_labels]


def candidate_envelope_mask(
    filtered_precipitation: xr.DataArray,
    configuration: LPTConfiguration = PRIMARY_CONFIGURATION,
) -> xr.DataArray:
    """Return every threshold and size-qualified instantaneous LPO candidate."""

    required_dims = {"Time", "latitude", "longitude"}
    if not required_dims.issubset(filtered_precipitation.dims):
        raise ValueError(
            "Candidate envelopes require Time, latitude, and longitude dimensions"
        )
    candidates = xr.apply_ufunc(
        _periodic_candidate_objects_2d,
        filtered_precipitation
        > configuration.precipitation_threshold_mm_day,
        input_core_dims=[["latitude", "longitude"]],
        output_core_dims=[["latitude", "longitude"]],
        kwargs={
            "minimum_grid_cells": configuration.minimum_object_grid_cells,
        },
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool],
        dask_gufunc_kwargs={"allow_rechunk": True},
    ).transpose(*filtered_precipitation.dims)
    candidates = candidates.rename("candidate_lpo_envelope_mask")
    candidates.attrs.update(
        {
            "long_name": "instantaneous threshold and size-qualified LPO candidates",
            "precipitation_threshold_mm_day": (
                configuration.precipitation_threshold_mm_day
            ),
            "minimum_object_grid_cells": configuration.minimum_object_grid_cells,
            "connectivity": "four-neighbour with periodic longitude",
            "retention": "independent of later track duration and linking",
        }
    )
    return candidates


def full_running_mean(
    precipitation: xr.DataArray,
    configuration: LPTConfiguration = PRIMARY_CONFIGURATION,
) -> xr.DataArray:
    """Trailing physical running mean with no partial cold-start values."""

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
    """Return the complete running mean plus the configured Gaussian filter."""

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
            # NetCDF attributes do not have a portable boolean type.
            "periodic_longitude": int(configuration.periodic_longitude),
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
                f"complete {configuration.running_mean_hours}-hour "
                "running-mean precipitation with Gaussian spatial smoothing"
            ),
            "units": "mm day-1",
            "gaussian_sigma_degrees": configuration.gaussian_sigma_degrees,
            "gaussian_sigma_grid_points": sigma_points,
            "periodic_longitude": int(configuration.periodic_longitude),
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
    *,
    running_mean_hours: int = 0,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    if subset == "complete":
        return times[0], times[-1]
    if subset == "first_100_days":
        start = times[0] + pd.Timedelta(hours=running_mean_hours)
        end = start + pd.Timedelta(days=VALIDATION_RECORD_DAYS)
        if end > times[-1]:
            raise ValueError(
                "Validation requires 100 complete tracking days after the "
                f"{running_mean_hours}-hour preprocessing window"
            )
        return start, end
    midpoint_index = len(times) // 2
    if midpoint_index == 0:
        raise ValueError("Record is too short to divide into two subsets")
    if subset == "first_half":
        return times[0], times[midpoint_index - 1]
    if subset == "second_half":
        return times[midpoint_index], times[-1]
    raise ValueError(f"Unknown record subset {subset!r}")


def tracking_and_input_bounds(
    times: Sequence,
    specification: RunSpecification,
) -> dict[str, pd.Timestamp]:
    """Return subset bounds with the configured preprocessing buffer."""

    index = pd.DatetimeIndex(pd.to_datetime(times))
    reference_start, reference_end = _subset_reference_bounds(
        index,
        specification.record_subset,
        running_mean_hours=specification.configuration.running_mean_hours,
    )
    if specification.record_subset == "first_100_days":
        return {
            "input_start": index[0],
            "tracking_start": reference_start,
            "tracking_end": reference_end,
            "reference_start": reference_start,
            "reference_end": reference_end,
        }
    # Record-half validation assigns full-record tracks by lifecycle-reference
    # time. Tracking the complete record is the only finite buffer that cannot
    # truncate an event merely because it lasts longer than the seven-day
    # minimum retention threshold.
    tracking_start, tracking_end = index[0], index[-1]
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
    # The pinned generic reader applies dataset['area']=[0, 360, ...]
    # literally. MPAS files use -180..180, so write a sorted 0..360 coordinate
    # to expose the complete periodic domain rather than only its eastern half.
    precip = precip.assign_coords(
        longitude=np.mod(precip.longitude, 360.0)
    ).sortby("longitude")
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
        write_netcdf_atomic(
            output,
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
    inferred_grid_spacing = (
        config.gaussian_sigma_degrees / float(sigma_grid_points)
    )
    if not np.isclose(
        inferred_grid_spacing,
        MPAS_LPT_GRID_SPACING_DEGREES,
        rtol=1.0e-3,
        atol=1.0e-5,
    ):
        raise ValueError(
            "The 100-cell LPO minimum is defined for the fixed 0.5-degree "
            f"MPAS grid, but the supplied grid spacing is "
            f"{inferred_grid_spacing:g} degrees"
        )
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
        "thresh_or_equal": False,
        "accumulation_hours": config.running_mean_hours,
        "filter_stdev": sigma_grid_points,
        "filter_n_stdev_width": config.gaussian_truncate_sigma,
        "min_points": config.minimum_object_grid_cells,
        "min_area": 0.0,
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
        # Upstream compares with ">" whereas the paper specifies "at least
        # half". The immediately preceding representable float therefore
        # implements >= 0.5 without modifying the published linker.
        "min_overlap_frac": (
            UPSTREAM_STRICT_HALF_OVERLAP_THRESHOLD
            if config.minimum_overlap_fraction == 0.5
            else config.minimum_overlap_fraction
        ),
        "bare_min_overlap_frac": config.bare_minimum_overlap_fraction,
        "min_lp_objects_points": config.minimum_object_grid_cells,
        "min_lp_objects_area": 0.0,
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
    systems_filename = (
        f"lpt_systems_{options[0]['label']}_{argv[1]}_{argv[2]}.nc"
    )
    systems_text_filename = Path(systems_filename).with_suffix(".txt").name
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
            try:
                lpt.lpt_driver.lpt_driver(*options, argv)
            except IndexError as exc:
                # The pinned writer assumes TIMECLUSTERS[0] exists. Convert its
                # empty-track IndexError into an actionable project-side error
                # without modifying upstream source.
                systems_directory = Path(options[4]["outdir"])
                if (systems_directory / systems_text_filename).is_file():
                    raise RuntimeError(
                        "No LPT envelope satisfied the unchanged seven-day "
                        "duration criterion in this record window. Choose a "
                        "different --test-start or increase --test-days."
                    ) from exc
                raise
    finally:
        os.chdir(previous_directory)
    systems_directory = Path(options[4]["outdir"])
    systems_file = systems_directory / systems_filename
    if not systems_file.is_file():
        raise RuntimeError(
            "Upstream LPT did not create the systems file for the requested "
            f"tracking interval: {systems_file}"
        )
    return systems_file


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
    lon = np.asarray(longitude, dtype=float)
    lat = np.asarray(latitude, dtype=float)
    valid = np.isfinite(hours) & np.isfinite(lon) & np.isfinite(lat)
    if valid.sum() < 2 or np.ptp(hours[valid]) == 0:
        raise RuntimeError(
            "Propagation requires at least two finite, distinct track times "
            "with precipitation-weighted centroids"
        )
    # LPT permits short below-threshold gaps within one tracked system.  These
    # steps have empty masks and undefined centroids.  Remove them before
    # unwrapping because one NaN otherwise contaminates every later longitude.
    lon = unwrap_longitude(lon[valid])
    hours = hours[valid]
    lat = lat[valid]
    lat_mean = np.nanmean(lat)
    zonal_km_hour = np.polyfit(hours, lon, 1)[0] * 111.195 * np.cos(
        np.deg2rad(lat_mean)
    )
    meridional_km_hour = np.polyfit(hours, lat, 1)[0] * 111.195
    zonal_speed = zonal_km_hour / 3.6
    displacement = (
        (lon[-1] - lon[0])
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


TRACKING_FIGURE_CLASSIFICATION_FIELDS = (
    "convective_precipitating_high_cloud_mask",
    "stratiform_precipitating_high_cloud_mask",
    "nonprecipitating_high_cloud_mask",
)


def _require_mapping_fields(
    source: Mapping[str, xr.DataArray],
    required: Sequence[str],
    *,
    product: str,
) -> None:
    """Validate one fixed producer/consumer field contract."""

    missing = sorted(set(required) - set(source))
    if missing:
        raise RuntimeError(f"{product} lacks required fields: {missing}")


def _area_sum(mask: xr.DataArray, area: xr.DataArray) -> float:
    return float(area.where(mask.astype(bool), 0.0).sum())


def _select_horizontal_grid_like(
    source: xr.DataArray,
    target: xr.DataArray,
    *,
    context: str,
) -> xr.DataArray:
    """Select source cells matching a possibly cropped target grid."""

    selected = source
    for coordinate in ("latitude", "longitude"):
        target_values = np.asarray(target[coordinate].values, dtype=float)
        source_values = np.asarray(selected[coordinate].values, dtype=float)
        nearest_indices = np.abs(
            source_values[:, None] - target_values[None, :]
        ).argmin(axis=0)
        selected = selected.isel({coordinate: nearest_indices})
        selected_values = np.asarray(
            selected[coordinate].values,
            dtype=float,
        )
        if (
            selected_values.size != target_values.size
            or not np.allclose(
                selected_values,
                target_values,
                rtol=0.0,
                atol=1.0e-5,
            )
        ):
            raise RuntimeError(
                f"{context}: {coordinate} grid does not match the LPO mask"
            )
    return selected


def diagnose_envelope_steps(
    event_mask: xr.DataArray,
    precipitation_mm_day: xr.DataArray,
    *,
    event_id: str,
    simulation: str,
) -> pd.DataFrame:
    """Calculate the two-dimensional diagnostics needed by Figures 1--3."""

    event_mask = event_mask.astype(bool)
    source_area = grid_cell_area_km2(
        precipitation_mm_day.latitude.values,
        precipitation_mm_day.longitude.values,
    )
    precipitation_mm_day = _select_horizontal_grid_like(
        precipitation_mm_day,
        event_mask,
        context=f"{simulation}/{event_id} precipitation",
    )
    for coordinate in ("latitude", "longitude"):
        event_mask = event_mask.assign_coords(
            {coordinate: precipitation_mm_day[coordinate].values}
        )
    precipitation_times = pd.DatetimeIndex(
        pd.to_datetime(precipitation_mm_day.Time.values)
    )
    event_times = pd.DatetimeIndex(pd.to_datetime(event_mask.Time.values))
    event_positions = precipitation_times.get_indexer(event_times)
    if np.any(event_positions < 0):
        raise RuntimeError(
            f"{simulation}/{event_id}: LPT mask and precipitation have no "
            "complete common time axis"
        )
    if len(precipitation_times) < 2:
        raise RuntimeError(
            f"{simulation}/{event_id}: precipitation needs at least two times"
        )
    dt_hours = float(
        np.median(np.diff(precipitation_times))
        / np.timedelta64(1, "h")
    )
    window_24h = int(round(24.0 / dt_hours))
    first_position = max(0, int(event_positions.min()) - window_24h + 1)
    last_position = int(event_positions.max()) + 1

    spatial_support = np.asarray(event_mask.any("Time").values, dtype=bool)
    latitude_indices = np.flatnonzero(spatial_support.any(axis=1))
    longitude_indices = np.flatnonzero(spatial_support.any(axis=0))
    if not latitude_indices.size or not longitude_indices.size:
        raise RuntimeError(
            f"{simulation}/{event_id}: LPT mask contains no retained cells"
        )
    full_area = _select_horizontal_grid_like(
        source_area,
        event_mask,
        context=f"{simulation}/{event_id} cell area",
    )
    spatial_indexers = {
        "latitude": latitude_indices,
        "longitude": longitude_indices,
    }
    event_mask = event_mask.isel(spatial_indexers)
    area = full_area.isel(spatial_indexers)
    buffered_precipitation = precipitation_mm_day.isel(
        Time=slice(first_position, last_position),
        **spatial_indexers,
    ).load()
    precipitation_depth_24h = buffered_precipitation.rolling(
        Time=window_24h, min_periods=window_24h
    ).sum() * (dt_hours / 24.0)
    precipitation = buffered_precipitation.sel(Time=event_times.values)
    precipitation, event_mask, area = xr.align(
        precipitation,
        event_mask,
        area,
        join="exact",
    )
    rows: list[dict] = []
    for time_value in pd.to_datetime(event_mask.Time.values):
        mask = event_mask.sel(Time=time_value)
        precip = precipitation.sel(Time=time_value)
        precipitation_depth = precipitation_depth_24h.sel(Time=time_value)
        envelope_area = _area_sum(mask, area)
        precipitation_statistics = sample_masked_field(
            precip, mask, area
        )
        rain_volume_24h = sample_masked_field(
            precipitation_depth, mask, area
        )["area_integral"]
        lon, lat = precipitation_weighted_centroid(mask, precip, area)
        valid_precipitation = mask & np.isfinite(precip)
        maximum_precipitation = (
            float(precip.where(valid_precipitation).max(skipna=True))
            if bool(valid_precipitation.any())
            else np.nan
        )
        rows.append(
            {
                "simulation": simulation,
                "event_id": event_id,
                "Time": time_value,
                "envelope_area_km2": envelope_area,
                "area_integrated_precip_km2_mm_day": (
                    precipitation_statistics["area_integral"]
                ),
                "area_integrated_24h_precip_km2_mm": rain_volume_24h,
                "mean_precip_mm_day": precipitation_statistics["mean"],
                "maximum_precip_mm_day": maximum_precipitation,
                "precip_weighted_centroid_lon": lon,
                "precip_weighted_centroid_lat": lat,
            }
        )
    return pd.DataFrame(rows)


def tracking_domain_area_km2(
    latitude: Sequence[float],
    longitude: Sequence[float],
    latitude_bounds: Sequence[float],
) -> float:
    """Area of the latitude-limited periodic domain supplied to upstream LPT."""

    latitude = np.asarray(latitude, dtype=float)
    lower, upper = sorted(np.asarray(latitude_bounds, dtype=float))
    selected = latitude[(latitude >= lower) & (latitude <= upper)]
    if selected.size == 0:
        raise ValueError("Tracking latitude bounds do not intersect the grid")
    return float(np.sum(grid_cell_area_km2(selected, longitude)))


def read_upstream_event_masks(
    upstream_systems_file: str | os.PathLike[str],
    events: pd.DataFrame,
) -> dict[str, xr.DataArray]:
    """Read original individual LPT masks and expose project-standard geometry."""

    systems_file = Path(upstream_systems_file)
    filename_parts = systems_file.stem.rsplit("_", 2)
    if len(filename_parts) != 3 or not systems_file.name.startswith(
        "lpt_systems_"
    ):
        raise RuntimeError(
            f"Cannot identify the tracking interval from {systems_file}"
        )
    systems_directory = systems_file.parent / (
        f"{filename_parts[-2]}_{filename_parts[-1]}"
    )
    if not systems_directory.is_dir():
        raise FileNotFoundError(
            "Missing upstream individual-mask directory for the selected "
            f"systems file: {systems_directory}"
        )
    files = sorted(
        systems_directory.glob("lpt_system_mask_*.nc")
    )
    by_upstream_id: dict[float, Path] = {}
    for path in files:
        with xr.open_dataset(path) as dataset:
            if "lptid" not in dataset:
                raise RuntimeError(
                    f"Pinned upstream LPT mask has no 'lptid' variable: {path}"
                )
            by_upstream_id[round(float(dataset.lptid.values[0]), 4)] = path
    masks: dict[str, xr.DataArray] = {}
    for event in events.itertuples():
        key = round(float(event.upstream_lpt_id), 4)
        if key not in by_upstream_id:
            raise RuntimeError(
                f"Missing original upstream individual mask for LPT {key}"
            )
        # Honour the NetCDF chunk layout.  The mask variables are already
        # stored one time step per chunk, while the unused upstream track
        # variables use whole-event time chunks.  Overriding the entire
        # dataset to time=1 therefore splits those stored chunks and produces
        # an xarray performance warning for every event.
        dataset = xr.open_dataset(by_upstream_id[key], chunks={})
        if "mask_at_end_time" not in dataset:
            dataset.close()
            raise RuntimeError(
                f"{by_upstream_id[key]} lacks the configured detailed-output "
                "'mask_at_end_time' variable"
            )
        mask = dataset["mask_at_end_time"].rename(
            {"time": "Time", "lat": "latitude", "lon": "longitude"}
        )
        # Return to the project-wide signed convention so masks align exactly
        # with the original MPAS and classification coordinates.
        signed_longitude = (
            (mask.longitude.astype(float) + 180.0) % 360.0
        ) - 180.0
        mask = mask.assign_coords(longitude=signed_longitude).sortby(
            "longitude"
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


def postprocess_event_masks(
    *,
    upstream_systems_file: str | os.PathLike[str],
    standard_directory: str | os.PathLike[str],
    steps: pd.DataFrame,
    events: pd.DataFrame,
    precipitation: xr.DataArray,
    classification: Mapping[str, xr.DataArray],
    specification: RunSpecification,
    tracking_figure_test: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Standardize masks and produce precipitation-based LPO diagnostics."""

    _require_mapping_fields(
        classification,
        TRACKING_FIGURE_CLASSIFICATION_FIELDS,
        product="shared high-cloud classification",
    )
    masks = read_upstream_event_masks(
        upstream_systems_file,
        events,
    )
    method_event_ids: set[str] = set()
    if tracking_figure_test or specification.name == "primary":
        envelope_figure_event_ids = set(events.event_id.astype(str))
    else:
        envelope_figure_event_ids = set()
    if tracking_figure_test:
        method_event_ids = set(events.event_id.astype(str))
        example_times = np.sort(
            pd.to_datetime(
                steps.loc[
                    steps.event_id.isin(method_event_ids),
                    "Time",
                ]
            ).unique()
        )
        # The short test subset must be able to render the method example for
        # whichever retained event has a complete display window. Materialise
        # its shared classification fields before any NetCDF writer is open:
        # otherwise Dask can stall while concurrently reading the shared
        # classification cache and writing an event product on the same
        # filesystem.
        classification = {
            **classification,
            **{
                name: classification[name].sel(Time=example_times).load()
                for name in TRACKING_FIGURE_CLASSIFICATION_FIELDS
            },
        }
        events = events.copy()
        events["test_method_example"] = events.event_id.isin(method_event_ids)
    events["figure_envelope_example"] = events.event_id.astype(str).isin(
        envelope_figure_event_ids
    )
    standard_directory = Path(standard_directory)
    mask_directory = standard_directory / "event_masks"
    mask_directory.mkdir(parents=True, exist_ok=True)
    diagnostic_rows: list[pd.DataFrame] = []
    mask_index_rows = []
    lower_latitude, upper_latitude = sorted(
        specification.configuration.tracking_latitude_bounds
    )
    tracking_precipitation = precipitation.sel(
        latitude=slice(
            lower_latitude
            if precipitation.latitude[0] < precipitation.latitude[-1]
            else upper_latitude,
            upper_latitude
            if precipitation.latitude[0] < precipitation.latitude[-1]
            else lower_latitude,
        )
    )
    if tracking_figure_test:
        progress(
            f"{specification.name}/{events.simulation.iloc[0]}: loading the "
            "shared two-dimensional test fields once"
        )
        tracking_precipitation = tracking_precipitation.load()
    unfiltered_running_precipitation = full_running_mean(
        tracking_precipitation,
        specification.configuration,
    )
    if tracking_figure_test:
        unfiltered_running_precipitation = (
            unfiltered_running_precipitation.load()
        )
    filtered_precipitation = filtered_running_mean(
        tracking_precipitation, specification.configuration
    )
    if tracking_figure_test:
        filtered_precipitation = filtered_precipitation.load()
    candidate_envelopes = candidate_envelope_mask(
        filtered_precipitation,
        specification.configuration,
    )
    event_count = len(events)
    for event_number, event in enumerate(events.itertuples(), start=1):
        event_start = time.perf_counter()
        progress(
            f"{specification.name}/{event.simulation}: event "
            f"{event_number}/{event_count} {event.event_id} "
            "(tracking and precipitation event diagnostics)"
        )
        mask = masks[event.event_id]
        event_times = pd.to_datetime(
            steps.loc[steps.event_id == event.event_id, "Time"]
        ).to_numpy()
        # Materialise only this sparse event mask. This avoids competing
        # NetCDF/Dask locks when the mask is used simultaneously to write the
        # self-contained product and sample model/classification files.
        mask = mask.sel(Time=event_times).load()
        event_precipitation = _select_horizontal_grid_like(
            tracking_precipitation.sel(Time=event_times),
            mask,
            context=f"{event.simulation}/{event.event_id} precipitation",
        )
        # Keep numerically equivalent float32/float64 grids from becoming a
        # coordinate union when masks are combined with MPAS fields.
        for coordinate in ("latitude", "longitude"):
            mask = mask.assign_coords(
                {coordinate: event_precipitation[coordinate].values}
            )
        raw_event_precipitation, mask = xr.align(
            event_precipitation, mask, join="exact"
        )
        mask = mask.fillna(False).astype(bool)
        latitude_support = np.flatnonzero(
            np.asarray(
                mask.any(("Time", "longitude")).values,
                dtype=bool,
            )
        )
        if not latitude_support.size:
            raise RuntimeError(
                f"{event.simulation}/{event.event_id}: event mask contains "
                "no latitude support"
            )
        latitude_spacing = float(
            np.median(np.abs(np.diff(mask.latitude.values)))
        )
        context_cells = int(np.ceil(10.0 / latitude_spacing))
        latitude_slice = slice(
            max(0, int(latitude_support.min()) - context_cells),
            min(
                mask.sizes["latitude"],
                int(latitude_support.max()) + context_cells + 1,
            ),
        )
        mask = mask.isel(latitude=latitude_slice)
        raw_event_precipitation = raw_event_precipitation.isel(
            latitude=latitude_slice
        )
        filtered_event_precipitation = filtered_precipitation.sel(
            Time=mask.Time,
            latitude=mask.latitude,
            longitude=mask.longitude,
        )
        output_path = mask_directory / f"{event.event_id}_mask.nc"
        event_variables = {
            "lpt_envelope_mask": mask,
            "raw_precipitation_mm_day": raw_event_precipitation,
            "filtered_72h_precipitation_mm_day": filtered_event_precipitation,
        }
        if not tracking_figure_test or event.event_id in method_event_ids:
            regime_context = xr.zeros_like(mask, dtype=np.int8).rename(
                "cloud_regime_context"
            )
            for code, variable_name in (
                (3, "nonprecipitating_high_cloud_mask"),
                (2, "stratiform_precipitating_high_cloud_mask"),
                (1, "convective_precipitating_high_cloud_mask"),
            ):
                regime_context = xr.where(
                    classification[variable_name].sel(
                        Time=mask.Time,
                        latitude=mask.latitude,
                        longitude=mask.longitude,
                    ),
                    code,
                    regime_context,
                )
            regime_context = regime_context.rename("cloud_regime_context")
            regime_context.attrs.update(
                {
                    "flag_values": [0, 1, 2, 3],
                    "flag_meanings": (
                        "other active_deep_convection "
                        "stratiform_precipitating_high_cloud "
                        "nonprecipitating_high_cloud"
                    ),
                    "sampling_region": (
                        "complete saved event-context domain; use "
                        "lpt_envelope_mask to distinguish the envelope"
                    ),
                }
            )
            embedded_regime = regime_context.where(mask, 0).rename(
                "embedded_cloud_regime"
            )
            embedded_regime.attrs.update(regime_context.attrs)
            embedded_regime.attrs["sampling_region"] = (
                "inside lpt_envelope_mask only"
            )
            event_variables["cloud_regime_context"] = regime_context
            event_variables["embedded_cloud_regime"] = embedded_regime
        if event.event_id in envelope_figure_event_ids:
            event_variables["unfiltered_72h_precipitation_mm_day"] = (
                unfiltered_running_precipitation.sel(
                    Time=mask.Time,
                    latitude=mask.latitude,
                    longitude=mask.longitude,
                )
            )
        event_dataset = netcdf_safe_dataset(xr.Dataset(
            event_variables,
            attrs={
                "event_id": event.event_id,
                "upstream_lpt_id": float(event.upstream_lpt_id),
                "purpose": (
                    "self-contained event geometry and fields sufficient for "
                    "tracking overview and event-centred figures"
                ),
                "method_figure_design": (
                    "Kerns and Chen (2020) Figure 1-style comparison of "
                    "unfiltered and Gaussian-filtered 72-hour rain"
                ),
                "displayed_lpo_threshold_mm_day": (
                    specification.configuration.precipitation_threshold_mm_day
                ),
                "saved_context": (
                    "all longitudes and the union of event-mask latitudes "
                    "with a 10-degree latitude buffer"
                ),
            },
        ))
        progress(
            f"{specification.name}/{event.simulation}: materialising "
            f"{event.event_id} event-context fields"
        )
        write_netcdf_atomic(
            event_dataset,
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
                **(
                    {
                        "unfiltered_72h_precipitation_mm_day": {
                            "dtype": "float32",
                            "zlib": True,
                            "complevel": 2,
                        }
                    }
                    if "unfiltered_72h_precipitation_mm_day"
                    in event_variables
                    else {}
                ),
                **(
                    {
                        "embedded_cloud_regime": {
                            "dtype": "int8",
                            "zlib": True,
                            "complevel": 4,
                        },
                        "cloud_regime_context": {
                            "dtype": "int8",
                            "zlib": True,
                            "complevel": 4,
                        }
                    }
                    if "embedded_cloud_regime" in event_variables
                    else {}
                ),
            },
        )
        event_dataset.close()
        mask_index_rows.append(
            {
                "simulation": event.simulation,
                "event_id": event.event_id,
                "upstream_lpt_id": event.upstream_lpt_id,
                "mask_file": str(output_path.relative_to(standard_directory)),
            }
        )
        diagnostic_rows.append(
            diagnose_envelope_steps(
                mask,
                tracking_precipitation,
                event_id=event.event_id,
                simulation=event.simulation,
            )
        )
        progress(
            f"{specification.name}/{event.simulation}: completed "
            f"{event.event_id} in {time.perf_counter() - event_start:.1f} s"
        )
    write_csv_atomic(
        pd.DataFrame(mask_index_rows),
        standard_directory / "lpt_event_mask_index.csv", index=False
    )
    tropical_limit = (
        specification.configuration.tropical_centroid_limit_degrees
    )
    tropical_filtered = periodic_longitude_field(
        filtered_precipitation.sel(
            latitude=slice(
                -tropical_limit
                if filtered_precipitation.latitude[0]
                < filtered_precipitation.latitude[-1]
                else tropical_limit,
                tropical_limit
                if filtered_precipitation.latitude[0]
                < filtered_precipitation.latitude[-1]
                else -tropical_limit,
            )
        )
    )
    tropical_unfiltered = periodic_longitude_field(
        unfiltered_running_precipitation.sel(
            latitude=slice(
                -tropical_limit
                if unfiltered_running_precipitation.latitude[0]
                < unfiltered_running_precipitation.latitude[-1]
                else tropical_limit,
                tropical_limit
                if unfiltered_running_precipitation.latitude[0]
                < unfiltered_running_precipitation.latitude[-1]
                else -tropical_limit,
            )
        )
    )
    tropical_raw = periodic_longitude_field(
        tracking_precipitation.sel(
            Time=filtered_precipitation.Time,
            latitude=slice(
                -10.0
                if tracking_precipitation.latitude[0]
                < tracking_precipitation.latitude[-1]
                else 10.0,
                10.0
                if tracking_precipitation.latitude[0]
                < tracking_precipitation.latitude[-1]
                else -10.0,
            ),
        )
    )
    filtered_latitude_weights = np.cos(
        np.deg2rad(tropical_filtered.latitude)
    )
    unfiltered_latitude_weights = np.cos(
        np.deg2rad(tropical_unfiltered.latitude)
    )
    raw_latitude_weights = np.cos(np.deg2rad(tropical_raw.latitude))
    equatorial_candidates = equatorial_envelope_for_plot(
        candidate_envelopes
    )
    overview_dataset = xr.Dataset(
        {
            "equatorial_raw_precipitation_mm_day": tropical_raw.weighted(
                raw_latitude_weights
            ).mean("latitude"),
            "equatorial_unfiltered_72h_precipitation_mm_day": (
                tropical_unfiltered.weighted(
                    unfiltered_latitude_weights
                ).mean("latitude")
            ),
            "equatorial_filtered_precipitation_mm_day": tropical_filtered.weighted(
                filtered_latitude_weights
            ).mean("latitude"),
            "equatorial_candidate_lpo_envelope": equatorial_candidates,
        },
        attrs={
            "raw_precipitation_latitude_band": (
                "10S-10N cosine-latitude weighted mean"
            ),
            "filtered_precipitation_latitude_band": (
                TRACKING_OVERVIEW_LATITUDE_BAND
            ),
            "temporal_average": (
                "complete trailing "
                f"{specification.configuration.running_mean_hours}-hour mean"
            ),
            "spatial_filter": (
                "equatorial_raw_precipitation_mm_day is the instantaneous "
                "six-hourly precipitation rate with no temporal or spatial "
                "filter; "
                "equatorial_unfiltered_72h_precipitation_mm_day is the "
                "legacy-named temporally averaged field with no spatial "
                "smoothing; equatorial_filtered_precipitation_mm_day uses "
                "Gaussian sigma "
                f"{specification.configuration.gaussian_sigma_degrees:g} "
                "degrees"
            ),
            "legacy_unfiltered_variable_actual_running_mean_hours": (
                specification.configuration.running_mean_hours
            ),
            "gaussian_sigma_degrees": (
                specification.configuration.gaussian_sigma_degrees
            ),
            "candidate_envelope_definition": (
                "filtered precipitation exceeds the configured threshold, "
                "meets the instantaneous grid-cell minimum, and does not "
                "require later track retention"
            ),
        },
    )
    write_netcdf_atomic(
        overview_dataset,
        standard_directory / "lpt_time_longitude_overview.nc",
        encoding={
            "equatorial_raw_precipitation_mm_day": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 2,
            },
            "equatorial_unfiltered_72h_precipitation_mm_day": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 2,
            },
            "equatorial_filtered_precipitation_mm_day": {
                "dtype": "float32",
                "zlib": True,
                "complevel": 2,
            },
            "equatorial_candidate_lpo_envelope": {
                "dtype": "int8",
                "zlib": True,
                "complevel": 4,
            },
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
        running_mean_hours=specification.configuration.running_mean_hours,
    )
    events["reference_assignment_start"] = reference_start
    events["reference_assignment_end"] = reference_end
    reference_record_days = float(
        (reference_end - reference_start)
        / pd.Timedelta(days=1)
    )
    domain_area_km2 = tracking_domain_area_km2(
        precipitation.latitude.values,
        precipitation.longitude.values,
        specification.configuration.tracking_latitude_bounds,
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
                "analysed_record_days": reference_record_days,
                "tracking_domain_area_km2": domain_area_km2,
                "maximum_area_integrated_24h_precipitation": float(
                    event_steps.area_integrated_24h_precip_km2_mm.max()
                ),
                "mean_precip_mm_day": float(event_steps.mean_precip_mm_day.mean()),
                "maximum_precip_mm_day": float(event_steps.maximum_precip_mm_day.max()),
                "mean_filtered_precip_mm_day": float(
                    steps.loc[
                        steps.event_id == event_id,
                        "mean_filtered_precip_mm_day",
                    ].mean()
                ),
                "start_precip_weighted_centroid_lon": float(
                    np.mod(
                        event_steps.precip_weighted_centroid_lon.iloc[0],
                        360.0,
                    )
                ),
                "start_precip_weighted_centroid_lat": float(
                    event_steps.precip_weighted_centroid_lat.iloc[0]
                ),
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
    return lifecycle, events, timing_summary


def assign_lifecycle_reference(
    steps: pd.DataFrame,
    configuration: LPTConfiguration = PRIMARY_CONFIGURATION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Assign precipitation-peak t=0 and fixed physical lags to every event."""

    output: list[pd.DataFrame] = []
    summaries: list[dict] = []
    lag_set = set(configuration.lifecycle_lags_hours)
    for event_id, event in steps.groupby("event_id", sort=False):
        event = event.sort_values("Time").copy()
        valid = np.isfinite(event.area_integrated_24h_precip_km2_mm)
        if not valid.any():
            raise RuntimeError(
                f"{event_id}: no finite trailing-24-hour integrated rain "
                "volume is available for the lifecycle reference"
            )
        peak_index = event.loc[
            valid, "area_integrated_24h_precip_km2_mm"
        ].idxmax()
        reference_time = pd.Timestamp(event.loc[peak_index, "Time"])
        event["lag_hours"] = (
            pd.to_datetime(event.Time) - reference_time
        ) / pd.Timedelta(hours=1)
        available = set(
            event.loc[np.isfinite(event.lag_hours), "lag_hours"].astype(int)
        )
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
        if not separations:
            raise RuntimeError(
                f"Matched events {default_id} and {sensitivity_id} have "
                "positive space-time IoU but no common centroid samples"
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
                    float(np.mean(separations))
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
    return pd.DataFrame(rows, columns=VALIDATION_MATCH_COLUMNS)


def _read_standard_masks(standard_directory: Path) -> dict[str, xr.DataArray]:
    index_path = standard_directory / "lpt_event_mask_index.csv"
    if not index_path.exists():
        raise FileNotFoundError(
            f"Missing standard LPT event-mask index: {index_path}"
        )
    index = pd.read_csv(index_path)
    return {
        row.event_id: xr.open_dataset(
            standard_directory / row.mask_file, chunks={}
        ).lpt_envelope_mask
        for row in index.itertuples()
    }


def build_validation_matches(validation_directory: str | os.PathLike[str]) -> None:
    """Match every sensitivity to the full 12 mm/day validation reference."""

    validation_directory = Path(validation_directory)
    reference = validation_directory / "threshold_12"
    if not reference.exists():
        raise RuntimeError("Validation matching requires the threshold_12 run")
    combined_rows = []
    for sensitivity_name in VALIDATION_RUN_NAMES:
        if sensitivity_name == "threshold_12":
            continue
        sensitivity = validation_directory / sensitivity_name
        if not sensitivity.is_dir():
            raise FileNotFoundError(
                f"Missing configured validation run: {sensitivity}"
            )
        for simulation in VALIDATION_SIMULATIONS:
            default_standard = reference / simulation / "standard"
            sensitivity_standard = sensitivity / simulation / "standard"
            default_path = default_standard / "lpt_events.csv"
            sensitivity_path = sensitivity_standard / "lpt_events.csv"
            missing = [
                str(path)
                for path in (default_path, sensitivity_path)
                if not path.exists()
            ]
            if missing:
                raise FileNotFoundError(
                    "Validation matching lacks required event tables:\n"
                    + "\n".join(f"  {path}" for path in missing)
                )
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
) -> None:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = copy.deepcopy(dict(metadata))
    payload["standard_schema"] = {
        "track_step_columns": list(track_steps.columns),
        "event_columns": list(events.columns),
        "lifecycle_columns": (
            list(lifecycle.columns) if lifecycle is not None else []
        ),
        "lifecycle_summary_columns": (
            list(lifecycle_summary.columns)
            if lifecycle_summary is not None
            else []
        ),
        "event_mask_index": "lpt_event_mask_index.csv",
    }
    payload["configuration_sha256"] = _sha256_json(payload)
    (directory / "lpt_configuration.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    write_csv_atomic(track_steps, directory / "lpt_track_steps.csv", index=False)
    write_csv_atomic(events, directory / "lpt_events.csv", index=False)
    if lifecycle is not None:
        write_csv_atomic(
            lifecycle,
            directory / "lpt_event_relative_diagnostics.csv",
            index=False,
        )
    if lifecycle_summary is not None:
        write_csv_atomic(
            lifecycle_summary,
            directory / "lpt_event_lifecycle_summary.csv",
            index=False,
        )


def _simulation_paths(mode_directory: Path, simulation: str) -> dict[str, Path]:
    root = mode_directory / simulation
    return {
        "root": root,
        "preprocessed": root / "preprocessed_precipitation",
        "upstream": root / "upstream_lpt",
        "standard": root / "standard",
    }


def _classification_cache_path(simulation: str) -> Path:
    return cache_directory(
        "classification",
        f"{simulation.replace('-', '_')}_classified_high_cloud.nc",
    )


def lpo_owner_cache_manifest(
    specification: RunSpecification,
    simulation: str,
    *,
    test_start: str | None,
    test_days: int | None,
    tracking_figure_test: bool,
) -> dict:
    """Describe inputs that determine one standard LPO product family."""

    source_root = Path(DEFAULT_ROOT) / SIMULATION_DIRS[simulation]
    source_paths = sorted(source_root.glob(LATLON_PATTERN))
    source_paths.append(_classification_cache_path(simulation))
    return cache_manifest(
        "lpo_tracking.py",
        schema=OWNER_CACHE_SCHEMA,
        configuration={
            "product_version": PRODUCT_VERSION,
            "tracking_cache_version": TRACKING_CACHE_VERSION,
            "upstream_commit": UPSTREAM_COMMIT,
            "run_name": specification.name,
            "record_subset": specification.record_subset,
            "configuration": specification.configuration.as_dict(),
            "test_start": test_start,
            "test_days": test_days,
            "tracking_figure_test": bool(tracking_figure_test),
            "classification_fields": list(
                TRACKING_FIGURE_CLASSIFICATION_FIELDS
            ),
        },
        source_paths=source_paths,
    )


def standard_lpo_products_are_current(
    standard: Path,
    expected_manifest: Mapping,
) -> bool:
    """Return whether one complete standard family matches its inputs."""

    required = (
        "lpt_configuration.json",
        "lpt_events.csv",
        "lpt_track_steps.csv",
        "lpt_event_relative_diagnostics.csv",
        "lpt_event_lifecycle_summary.csv",
        "lpt_event_mask_index.csv",
    )
    if not all((standard / name).is_file() for name in required):
        return False
    if not cache_manifest_matches(
        read_json_file(standard / "lpt_cache_manifest.json"),
        expected_manifest,
    ):
        return False
    configuration = read_json_file(standard / "lpt_configuration.json")
    if not isinstance(configuration, dict):
        return False
    if configuration.get("product_version") != PRODUCT_VERSION:
        return False
    saved_hash = configuration.get("configuration_sha256")
    unhashed = dict(configuration)
    unhashed.pop("configuration_sha256", None)
    if saved_hash != _sha256_json(unhashed):
        return False
    try:
        mask_index = pd.read_csv(standard / "lpt_event_mask_index.csv")
    except (OSError, pd.errors.ParserError):
        return False
    if "mask_file" not in mask_index:
        return False
    return all((standard / value).is_file() for value in mask_index.mask_file)


def _load_classification(
    simulation: str,
    expected_time: xr.DataArray,
    required_fields: Sequence[str],
) -> dict:
    path = cache_directory(
        "classification",
        f"{simulation.replace('-', '_')}_classified_high_cloud.nc",
    )
    if not path.exists():
        raise FileNotFoundError(
            f"Missing shared high-cloud classification {path}; run "
            "refined_high_cloud_classification_diagnostics.py first"
        )
    dataset = xr.open_dataset(path, chunks={})
    if not cached_classification_is_current(
        dataset, HighCloudThresholds()
    ):
        dataset.close()
        raise RuntimeError(
            f"Stale shared high-cloud classification: {path}"
        )
    missing = sorted(set(required_fields) - set(dataset.data_vars))
    if missing:
        dataset.close()
        raise RuntimeError(
            f"Shared high-cloud classification {path} lacks required fields: "
            f"{missing}"
        )
    try:
        dataset = dataset.sel(Time=expected_time.values)
    except KeyError as exc:
        dataset.close()
        raise RuntimeError(
            f"Shared high-cloud classification {path} does not contain the "
            "exact LPT input times"
        ) from exc
    return {name: dataset[name] for name in required_fields}


def _load_simulation_dataset(
    simulation: str,
    *,
    tracking_figure_test: bool = False,
    include_lifecycle_fields: bool = False,
) -> xr.Dataset:
    precipitation_variables = [
        "rainc",
        "rainnc",
        "i_rainc",
        "i_rainnc",
    ]
    lifecycle_variables = [
        "olrtoa",
        "precipw",
        "cldfrac",
        "qc",
        "qr",
        "qi",
        "qs",
        "qg",
        "rho",
        "theta",
        "qv",
        "relhum",
        "w",
        "pressure",
        "zgrid",
        "t_isobaric",
        "uzonal_isobaric",
        "umeridional_isobaric",
        "qv_isobaric",
        "rh_isobaric",
        "omg_isobaric",
        "uReconstructZonal",
    ]
    variables = precipitation_variables + (
        lifecycle_variables
        if include_lifecycle_fields or not tracking_figure_test
        else []
    )
    dataset = load_latlon_dataset(
        os.path.join(DEFAULT_ROOT, SIMULATION_DIRS[simulation]),
        latlon_pattern=LATLON_PATTERN,
        chunks={"Time": 8, "latitude": 90, "longitude": 180},
    )
    return standardize_latlon_dataset(dataset, variables)


def select_actual_test_window(
    dataset: xr.Dataset,
    *,
    start: str | None,
    days: int,
) -> tuple[xr.Dataset, dict]:
    """Select a contiguous real-data window plus one precipitation-rate input."""

    if "Time" not in dataset.coords:
        raise RuntimeError("--test requires a datetime Time coordinate")
    index = pd.DatetimeIndex(pd.to_datetime(dataset.Time.values))
    if not isinstance(index, pd.DatetimeIndex) or index.hasnans:
        raise RuntimeError("--test requires a complete datetime Time coordinate")
    if len(index) < 2:
        raise RuntimeError("--test requires at least two source times")
    requested_start = (
        pd.Timestamp(start)
        if start is not None
        else pd.Timestamp(index[0]).normalize() + pd.Timedelta(days=1)
    )
    requested_end = requested_start + pd.Timedelta(days=days)
    start_candidates = np.flatnonzero(index >= requested_start)
    if not len(start_candidates):
        raise ValueError(
            f"--test-start {requested_start} is after the simulation record"
        )
    first_rate_index = int(start_candidates[0])
    if first_rate_index == 0:
        source_start_index = 0
        first_rate_index = 1
    else:
        source_start_index = first_rate_index - 1
    end_candidates = np.flatnonzero(index <= requested_end)
    if not len(end_candidates):
        raise ValueError(
            f"--test window ending {requested_end} precedes the simulation record"
        )
    source_end_index = int(end_candidates[-1])
    if source_end_index <= first_rate_index:
        raise ValueError("The requested --test window contains too few source times")
    actual_start = pd.Timestamp(index[first_rate_index])
    actual_end = pd.Timestamp(index[source_end_index])
    actual_days = (actual_end - actual_start) / pd.Timedelta(days=1)
    if actual_days < days - 1:
        raise ValueError(
            f"Only {actual_days:.2f} days are available from {actual_start}; "
            f"--test-days requested {days}"
        )
    selected = dataset.isel(
        Time=slice(source_start_index, source_end_index + 1)
    )
    metadata = {
        "requested_start": str(requested_start),
        "requested_days": int(days),
        "source_start": str(pd.Timestamp(index[source_start_index])),
        "diagnostic_start": str(actual_start),
        "diagnostic_end": str(actual_end),
        "source_time_count": int(selected.sizes["Time"]),
    }
    return selected, metadata


def run_specification(
    specification: RunSpecification,
    *,
    simulations: Sequence[str],
    output: Path,
    source: Path,
    stage: str,
    test_start: str | None = None,
    test_days: int | None = None,
    tracking_figure_test: bool = False,
    rebuild_cache: bool = False,
) -> None:
    mode_directory = output / specification.name
    for simulation in simulations:
        progress(
            f"{specification.name}/{simulation}: opening fixed MPAS inputs "
            f"below {DEFAULT_ROOT}"
        )
        paths = _simulation_paths(mode_directory, simulation)
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        expected_manifest = lpo_owner_cache_manifest(
            specification,
            simulation,
            test_start=test_start,
            test_days=test_days,
            tracking_figure_test=tracking_figure_test,
        )
        if (
            stage == "all"
            and not rebuild_cache
            and standard_lpo_products_are_current(
                paths["standard"], expected_manifest
            )
        ):
            progress(
                f"{specification.name}/{simulation}: reusing current "
                "source-signed standard LPO products"
            )
            continue
        dataset = _load_simulation_dataset(
            simulation,
            tracking_figure_test=True,
            include_lifecycle_fields=False,
        )
        test_subset = None
        if test_days is not None:
            dataset, test_subset = select_actual_test_window(
                dataset,
                start=test_start,
                days=test_days,
            )
            progress(
                f"{simulation}: actual-data test subset "
                f"{test_subset['diagnostic_start']} to "
                f"{test_subset['diagnostic_end']}"
            )
        precipitation = precipitation_rate_from_dataset(dataset)
        preprocessing_path = paths["root"] / "preprocessing_metadata.json"
        tracking_metadata_path = (
            paths["root"] / UPSTREAM_TRACKING_METADATA_FILENAME
        )
        if stage in {"all", "preprocess"}:
            progress(
                f"{specification.name}/{simulation}: writing upstream LPT "
                "precipitation inputs"
            )
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
            progress(
                f"{specification.name}/{simulation}: running pinned upstream LPT"
            )
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
            write_upstream_tracking_metadata(
                tracking_metadata_path,
                specification=specification,
                preprocessing_metadata=preprocessing,
                systems_file=systems_file,
            )
        else:
            systems_file = Path(
                (paths["root"] / "upstream_systems_file.txt").read_text().strip()
            )
        if stage == "track":
            dataset.close()
            continue
        if stage in {"all", "postprocess"}:
            require_current_upstream_tracking(
                tracking_metadata_path,
                specification=specification,
                preprocessing_metadata=preprocessing,
                systems_file=systems_file,
            )
            progress(
                f"{specification.name}/{simulation}: standardising tracks and "
                + (
                    "sampling two-dimensional tracking-figure diagnostics"
                    if tracking_figure_test
                    else "sampling tracking and precipitation event diagnostics"
                )
            )
            steps, events = standardise_upstream_tracks(
                systems_file, simulation, specification
            )
            classification = _load_classification(
                simulation,
                precipitation.Time,
                TRACKING_FIGURE_CLASSIFICATION_FIELDS,
            )
            lifecycle, events, lifecycle_summary = postprocess_event_masks(
                upstream_systems_file=systems_file,
                standard_directory=paths["standard"],
                steps=steps,
                events=events,
                precipitation=precipitation,
                classification=classification,
                specification=specification,
                tracking_figure_test=tracking_figure_test,
            )
            metadata = configuration_metadata(
                specification,
                simulation=simulation,
                upstream_source=source,
                test_subset=test_subset,
            )
            write_standard_products(
                paths["standard"],
                metadata=metadata,
                track_steps=steps,
                events=events,
                lifecycle=lifecycle,
                lifecycle_summary=lifecycle_summary,
            )
            write_json_file(
                paths["standard"] / "lpt_cache_manifest.json",
                expected_manifest,
            )
            progress(
                f"{specification.name}/{simulation}: wrote "
                f"{len(events)} retained tracks and {len(lifecycle)} event steps"
            )
        dataset.close()
    if stage in {"all", "postprocess"}:
        selected_simulations = (
            VALIDATION_SIMULATIONS
            if specification.record_subset == "first_100_days"
            else tuple(SIMULATION_DIRS)
        )
        event_files = [
            mode_directory / simulation / "standard" / "lpt_events.csv"
            for simulation in selected_simulations
            if (
                mode_directory / simulation / "standard" / "lpt_events.csv"
            ).is_file()
        ]
        expected_file_count = (
            len(VALIDATION_SIMULATIONS)
            if specification.record_subset == "first_100_days"
            else len(SIMULATION_DIRS)
        )
        if len(event_files) != expected_file_count:
            progress(
                f"{specification.name}: selective postprocessing complete; "
                "cross-simulation population summary is deferred until all "
                "selected standard event products exist"
            )
            return
        combined_events = pd.concat(
            [pd.read_csv(path) for path in event_files],
            ignore_index=True,
        )
        population = lpo_population_summary(combined_events)
        profile_suffix = (
            ""
            if ACTIVE_SIMULATION_PROFILE == "legacy"
            else f".{ACTIVE_SIMULATION_PROFILE}"
        )
        write_csv_atomic(
            population,
            mode_directory / f"lpo_population_summary{profile_suffix}.csv",
            index=False,
        )
        if tracking_figure_test:
            progress(
                f"{specification.name}: wrote LPO population summary; "
                "test mode omits event uncertainty"
            )
        else:
            uncertainty = lpo_event_property_uncertainty(combined_events)
            write_csv_atomic(
                uncertainty,
                mode_directory
                / f"lpo_event_property_uncertainty{profile_suffix}.csv",
                index=False,
            )
            progress(
                f"{specification.name}: wrote LPO population and uncertainty summaries"
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Track MPAS large precipitation objects with the published LPT "
            "implementation and build tracking-validation products."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mode", choices=("publication", "validation"), default="publication"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Run a quick actual-data check of LPO tracking and Figures 1--3. "
            "The scientific tracking criteria are retained, apart from the "
            "explicitly test-only duration cutoff."
        ),
    )
    parser.add_argument(
        "--test-start",
        default=None,
        metavar="TIME",
        help=(
            "First diagnostic time for --test, for example 2000-06-01. "
            "Defaults to the first complete UTC day common to each loaded record."
        ),
    )
    parser.add_argument(
        "--test-days",
        type=int,
        default=DEFAULT_TEST_DAYS,
        metavar="N",
        help=(
            "Physical duration of the actual-data tracking-figure window. "
            f"The minimum is {MINIMUM_TEST_DAYS} days so a complete 72-hour "
            "filter precedes at least five tracking times."
        ),
    )
    parser.add_argument(
        "--test-min-duration-days",
        type=float,
        default=DEFAULT_TEST_MINIMUM_DURATION_DAYS,
        metavar="DAYS",
        help=(
            "Test-only minimum retained track duration (default: one day, "
            "the five-point minimum required by the upstream smoother). This "
            "affects only isolated --test products; publication and "
            "validation remain at seven days."
        ),
    )
    parser.add_argument(
        "--stage",
        choices=("all", "preprocess", "track", "postprocess", "figures"),
        default="all",
    )
    parser.add_argument(
        "--rebuild-cache",
        action="store_true",
        help="Rebuild source-signed LPO products before plotting.",
    )
    parser.add_argument(
        "--simulation",
        choices=(*SIMULATION_DIRS, "all"),
        default="all",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=PUBLICATION_FIGURES + VALIDATION_FIGURES,
    )
    args = parser.parse_args(argv)
    if args.test and args.test_days < MINIMUM_TEST_DAYS:
        parser.error(
            f"--test-days must be at least {MINIMUM_TEST_DAYS} so the "
            "full 72-hour filter precedes the upstream five-point track "
            "smoother"
        )
    if (
        args.test
        and args.test_min_duration_days
        < MINIMUM_UPSTREAM_SMOOTHING_DURATION_DAYS
    ):
        parser.error(
            "--test-min-duration-days must be at least 1 day because the "
            "unmodified upstream LPT spline smoother requires five six-hourly "
            "track points"
        )
    if args.test and args.mode != "publication":
        parser.error("--test uses the publication configuration; omit --mode validation")
    if args.mode == "validation" and args.simulation not in {"all", "control"}:
        parser.error("validation uses the first 100 days from control only")
    if (
        args.test
        and args.simulation != "all"
        and args.stage in {"all", "figures"}
    ):
        parser.error(
            "--test figure generation requires --simulation all; a single "
            "simulation may be used with preprocess, track, or postprocess stages"
        )
    if args.figures:
        args.figures = list(dict.fromkeys(args.figures))
    if args.rebuild_cache and args.stage != "all":
        parser.error("--rebuild-cache requires --stage all")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    progress(
        f"Starting LPO tracking diagnostics in "
        f"{'actual-data test' if args.test else args.mode} mode "
        f"(stage={args.stage})"
    )
    test_minimum_duration_hours = int(
        round(24.0 * args.test_min_duration_days)
    )
    cache_root = lpo_tracking_output_root()
    figure_root = lpo_tracking_figure_root()
    cache_output = (
        actual_test_output_directory(
            cache_root,
            start=args.test_start,
            days=args.test_days,
            minimum_duration_hours=test_minimum_duration_hours,
        )
        if args.test
        else cache_root / args.mode
    )
    figure_output = (
        actual_test_output_directory(
            figure_root,
            start=args.test_start,
            days=args.test_days,
            minimum_duration_hours=test_minimum_duration_hours,
        )
        if args.test
        else figure_root
    )
    cache_output.mkdir(parents=True, exist_ok=True)
    if args.test:
        progress(f"Actual-data LPO tracking test data: {cache_output}")
        progress(f"Actual-data LPO tracking test figures: {figure_output}")
    specs = (
        actual_data_test_run_specifications(args.test_min_duration_days)
        if args.test
        else publication_run_specifications()
        if args.mode == "publication"
        else validation_run_specifications()
    )
    simulations = (
        VALIDATION_SIMULATIONS
        if args.mode == "validation"
        else tuple(SIMULATION_DIRS)
        if args.simulation == "all"
        else (args.simulation,)
    )
    if args.stage == "figures":
        build_tracking_figures_from_saved_outputs(
            cache_output,
            mode=args.mode,
            figures=args.figures,
            figure_output=figure_output,
            tracking_figure_test=args.test,
        )
        progress("Finished LPO figure regeneration from saved products")
        return 0
    source = upstream_source_path()
    progress(f"Verifying pinned upstream LPT checkout: {source}")
    verify_upstream_source(source)
    for specification in specs:
        run_specification(
            specification,
            simulations=simulations,
            output=cache_output,
            source=source,
            stage=args.stage,
            test_start=args.test_start if args.test else None,
            test_days=args.test_days if args.test else None,
            tracking_figure_test=args.test,
            rebuild_cache=args.rebuild_cache,
        )
    if args.mode == "validation" and args.stage in {"all", "postprocess"}:
        progress("Matching validation events to the primary threshold run")
        build_validation_matches(cache_output)
    if args.stage == "all":
        build_tracking_figures_from_saved_outputs(
            cache_output,
            mode=args.mode,
            figures=args.figures,
            figure_output=figure_output,
            tracking_figure_test=args.test,
        )
    progress("LPO tracking diagnostics complete")
    return 0


def read_standard_event_tables(
    mode_directory: str | os.PathLike[str],
    simulations: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Read the project-standard event and lifecycle tables for one run root."""

    mode_directory = Path(mode_directory)
    selected_simulations = tuple(
        SIMULATION_DIRS if simulations is None else simulations
    )
    event_files = [
        mode_directory / simulation / "standard" / "lpt_events.csv"
        for simulation in selected_simulations
        if (
            mode_directory / simulation / "standard" / "lpt_events.csv"
        ).is_file()
    ]
    lifecycle_files = [
        mode_directory
        / simulation
        / "standard"
        / "lpt_event_relative_diagnostics.csv"
        for simulation in selected_simulations
        if (
            mode_directory
            / simulation
            / "standard"
            / "lpt_event_relative_diagnostics.csv"
        ).is_file()
    ]
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
    *, allow_stale_product_version: bool = False
) -> Path:
    """Validate the stable publication interface used by downstream scripts."""

    mode_directory = publication_output_directory()
    missing: list[str] = []
    for simulation in SIMULATION_DIRS:
        standard = mode_directory / "primary" / simulation / "standard"
        metadata = None
        for filename in (
            "lpt_configuration.json",
            "lpt_events.csv",
            "lpt_track_steps.csv",
            "lpt_event_relative_diagnostics.csv",
            "lpt_event_lifecycle_summary.csv",
            "lpt_event_mask_index.csv",
        ):
            path = standard / filename
            if not path.is_file():
                missing.append(str(path))
                continue
            if filename == "lpt_configuration.json":
                metadata = json.loads(path.read_text(encoding="utf-8"))
                if (
                    metadata.get("product_version") != PRODUCT_VERSION
                    and not allow_stale_product_version
                ):
                    raise RuntimeError(
                        f"{path} has product_version "
                        f"{metadata.get('product_version')!r}; expected "
                        f"{PRODUCT_VERSION!r}. Rebuild the LPO publication run."
                    )
                if (
                    metadata.get("product_version") != PRODUCT_VERSION
                    and allow_stale_product_version
                ):
                    progress(
                        "Allowing stale LPO publication product version "
                        f"{metadata.get('product_version')!r} from {path}; "
                        f"current version is {PRODUCT_VERSION!r}"
                    )
                schema = metadata.get("standard_schema")
                if not isinstance(schema, dict):
                    raise RuntimeError(
                        f"{path} does not declare the standard product schema; "
                        "rebuild the LPT publication run"
                    )
                saved_hash = metadata.get("configuration_sha256")
                unhashed = dict(metadata)
                unhashed.pop("configuration_sha256", None)
                if saved_hash != _sha256_json(unhashed):
                    raise RuntimeError(
                        f"{path} has an invalid configuration hash; rebuild "
                        "or restore the standard LPT products"
                    )
        if metadata is None:
            continue
        schema_files = {
            "lpt_track_steps.csv": "track_step_columns",
            "lpt_events.csv": "event_columns",
            "lpt_event_relative_diagnostics.csv": "lifecycle_columns",
            "lpt_event_lifecycle_summary.csv": "lifecycle_summary_columns",
        }
        schema = metadata["standard_schema"]
        for filename, schema_key in schema_files.items():
            path = standard / filename
            if not path.exists():
                continue
            actual_columns = list(pd.read_csv(path, nrows=0).columns)
            if actual_columns != list(schema.get(schema_key, [])):
                raise RuntimeError(
                    f"{path} columns do not match metadata key {schema_key!r}; "
                    "rebuild the standard LPT products"
                )
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:8])
        raise FileNotFoundError(
            "Missing standard LPT publication products. Run "
            "lpo_tracking.py first:\n"
            f"{preview}"
        )
    return mode_directory


def require_actual_test_products(
    *,
    start: str | None = None,
    days: int = DEFAULT_TEST_DAYS,
    minimum_duration_days: float = DEFAULT_TEST_MINIMUM_DURATION_DAYS,
) -> Path:
    """Validate and return the isolated actual-data LPT test products."""

    minimum_duration_hours = int(round(24.0 * minimum_duration_days))
    mode_directory = actual_test_output_directory(
        lpo_tracking_output_root(),
        start=start,
        days=days,
        minimum_duration_hours=minimum_duration_hours,
    )
    missing: list[str] = []
    for simulation in SIMULATION_DIRS:
        standard = mode_directory / "primary" / simulation / "standard"
        metadata_path = standard / "lpt_configuration.json"
        required_paths = (
            metadata_path,
            standard / "lpt_events.csv",
            standard / "lpt_track_steps.csv",
            standard / "lpt_event_relative_diagnostics.csv",
            standard / "lpt_event_lifecycle_summary.csv",
            standard / "lpt_event_mask_index.csv",
        )
        missing.extend(str(path) for path in required_paths if not path.is_file())
        if not metadata_path.is_file():
            continue
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("product_version") != PRODUCT_VERSION:
            raise RuntimeError(
                f"{metadata_path} has product_version "
                f"{metadata.get('product_version')!r}; expected "
                f"{PRODUCT_VERSION!r}. Rerun the LPO tracking test."
            )
        test_subset = metadata.get("test_subset")
        if (
            not isinstance(test_subset, dict)
            or test_subset.get("definitive_publication_product") is not False
            or test_subset.get("figures_produced")
            != list(PUBLICATION_FIGURES)
        ):
            raise RuntimeError(
                f"{metadata_path} predates the tracking-only actual-data test "
                "schema; rerun the LPT test."
            )
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:8])
        raise FileNotFoundError(
            "Missing isolated LPT test products. Run "
            "lpo_tracking.py --test first:\n"
            f"{preview}"
        )
    return mode_directory


def _require_validation_figure_products(output: Path) -> None:
    """Require the complete controlled-sensitivity product matrix."""

    missing: list[str] = []
    for run_name in VALIDATION_RUN_NAMES:
        run_directory = output / run_name
        for simulation in VALIDATION_SIMULATIONS:
            standard = run_directory / simulation / "standard"
            for filename in (
                "lpt_configuration.json",
                "lpt_events.csv",
                "lpt_track_steps.csv",
                "lpt_event_relative_diagnostics.csv",
                "lpt_event_mask_index.csv",
                "lpt_time_longitude_overview.nc",
            ):
                path = standard / filename
                if not path.is_file():
                    missing.append(str(path))
    matching = output / "validation_event_matching.csv"
    if not matching.is_file():
        missing.append(str(matching))
    if missing:
        preview = "\n".join(f"  {path}" for path in missing[:12])
        raise FileNotFoundError(
            "Validation Figure 8 requires the complete threshold, gap, "
            "overlap, merger/split, and matching "
            "products:\n"
            f"{preview}"
        )


def _require_primary_figure_product_versions(primary: Path) -> None:
    """Reject publication/test figures built from stale tracking products."""

    for simulation in SIMULATION_DIRS:
        configuration_path = (
            primary
            / simulation
            / "standard"
            / "lpt_configuration.json"
        )
        if not configuration_path.is_file():
            raise FileNotFoundError(
                f"Missing Figure 1--3 configuration: {configuration_path}"
            )
        metadata = json.loads(
            configuration_path.read_text(encoding="utf-8")
        )
        if metadata.get("product_version") != PRODUCT_VERSION:
            raise RuntimeError(
                f"{configuration_path} has product_version "
                f"{metadata.get('product_version')!r}; expected "
                f"{PRODUCT_VERSION!r}. Rerun LPO tracking and postprocessing "
                "before regenerating figures."
            )


def build_tracking_figures_from_saved_outputs(
    output: str | os.PathLike[str],
    *,
    mode: str,
    figures: Sequence[int] | None = None,
    figure_output: str | os.PathLike[str] | None = None,
    tracking_figure_test: bool = False,
) -> None:
    """Regenerate figures exclusively from saved event-level products."""

    output = Path(output)
    figures = set(
        figures
        or (
            PUBLICATION_FIGURES
            if mode == "publication"
            else VALIDATION_FIGURES
        )
    )
    figure_saver = FigureSaver(
        str(Path(figure_output or output) / "lpo_tracking"),
        enabled=True,
    )

    def save(number: int, stem: str, figure) -> None:
        if figure is None:
            raise RuntimeError(
                f"Figure {number} {stem!r} could not be built from the "
                "required saved products"
            )
        figure_saver.save(figure, stem, number=number)
        plt.close(figure)

    if mode == "validation":
        if 8 in figures:
            _require_validation_figure_products(output)
            progress("Starting lpo_tracking Figure 5: validation sensitivities")
            save(5, "lpo_validation_sensitivity", _plot_validation_figure(output))
            for stem, figure in _plot_validation_track_figures(output):
                save(5, stem, figure)
        return
    primary = output / "primary"
    _require_primary_figure_product_versions(primary)
    events, _lifecycle = read_standard_event_tables(primary)
    if figures.intersection(TRACKING_FIGURES) and events.empty:
        raise RuntimeError(
            f"{primary}: no LPO events are available for requested tracking "
            "figures"
        )
    if 1 in figures and not events.empty:
        progress("Starting Figure 1: representative LPO method")
        save(
            1,
            "lpo_method",
            _plot_method_example(
                primary,
                events,
                tracking_figure_test=tracking_figure_test,
            ),
        )
        progress(
            "Starting Figure 1 envelope gallery: "
            "all retained LPOs in every simulation"
        )
        _write_lpo_envelope_gallery(
            primary,
            events,
            Path(figure_saver.save_loc) / "figure_01_lpo_envelopes",
            metadata_output_directory=(
                output / "figure_products" / "figure_01_lpo_envelopes"
            ),
            tracking_figure_test=tracking_figure_test,
        )
    if 2 in figures and not events.empty:
        progress(
            "Starting Figure 2: "
            + (
                "actual-data subset LPO tracking overview"
                if tracking_figure_test
                else "full-record LPO tracking overview"
            )
        )
        for simulation in SIMULATION_DIRS:
            save(
                2,
                f"lpo_tracking_overview_{simulation}",
                _plot_tracking_overview(
                    primary,
                    events,
                    simulation=simulation,
                    tracking_figure_test=tracking_figure_test,
                ),
            )
            movie_filename = (
                f"02_lpo_tracking_{simulation}_test_period.mp4"
                if tracking_figure_test
                else f"02_lpo_tracking_{simulation}_full_record.mp4"
            )
            _write_tracking_overview_movie(
                primary,
                events,
                Path(figure_saver.save_loc)
                / "figure_02_tracking_movie"
                / movie_filename,
                metadata_path=(
                    output
                    / "figure_products"
                    / "figure_02_tracking_movie"
                    / Path(movie_filename).with_suffix(".json")
                ),
                simulation=simulation,
                tracking_figure_test=tracking_figure_test,
            )
    if 3 in figures and not events.empty:
        progress("Starting Figure 3a: LPO event distributions")
        save(
            3,
            "lpo_event_distributions",
            _plot_event_distributions(
                events,
                tracking_figure_test=tracking_figure_test,
            ),
        )
        progress("Starting Figure 3b: LPO population summaries")
        save(
            3,
            "lpo_population_summary",
            _plot_population_summaries(
                events,
                tracking_figure_test=tracking_figure_test,
            ),
        )
        if not tracking_figure_test:
            progress("Starting Figure 3c: LPO event-property uncertainty")
            save(
                3,
                "lpo_event_property_uncertainty",
                _plot_event_property_uncertainty(events),
            )
    if 4 in figures and not events.empty:
        progress("Starting lpo_tracking Figure 4: all LPO tracks")
        save(
            4,
            "all_lpo_tracks_tropical_map",
            _plot_all_lpo_tracks_map(primary, events),
        )


def _event_product_path(primary: Path, simulation: str, event_id: str) -> Path:
    index = pd.read_csv(
        primary / simulation / "standard" / "lpt_event_mask_index.csv"
    ).set_index("event_id")
    return primary / simulation / "standard" / index.loc[event_id, "mask_file"]


def filtered_precipitation_cmap_norm():
    """Return the common precipitation levels with a blue colour map."""

    _, common_norm = precip_rate_cmap_norm()
    filtered_cmap = plt.get_cmap("Blues").copy()
    filtered_cmap.set_under("white")
    filtered_cmap.set_over("#08306b")
    filtered_norm = BoundaryNorm(
        common_norm.boundaries,
        ncolors=filtered_cmap.N,
    )
    return filtered_cmap, filtered_norm


def _tracking_test_example_events(
    events: pd.DataFrame,
    count: int = 2,
) -> pd.DataFrame:
    """Choose deterministic, non-extreme control events for the test figure."""

    candidates = events[events.simulation == "control"].copy()
    tropical = candidates[
        candidates.primary_tropical_sample.astype(bool)
    ]
    if len(tropical) >= count:
        candidates = tropical
    if len(candidates) < count:
        candidates = events.copy()
    if len(candidates) < count:
        raise RuntimeError(
            f"The tracking test Figure 1 needs {count} retained events; "
            f"only {len(candidates)} are available. Choose a different "
            "--test-start or increase --test-days."
        )
    candidates = candidates.sort_values(
        ["maximum_area_km2", "event_id"]
    ).reset_index(drop=True)
    positions = np.linspace(
        0,
        len(candidates) - 1,
        count + 2,
    )[1:-1]
    indices = np.rint(positions).astype(int)
    if len(set(indices)) != count:
        indices = np.linspace(
            0,
            len(candidates) - 1,
            count,
            dtype=int,
        )
    return candidates.iloc[indices].reset_index(drop=True)


def publication_method_gallery_events(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Return every retained LPO in deterministic gallery order."""

    required = {"simulation", "event_id"}
    missing = sorted(required - set(events.columns))
    if missing:
        raise RuntimeError(
            f"Figure 1 gallery selection lacks event fields: {missing}"
        )
    present_simulations = set(events.simulation.astype(str))
    unknown_simulations = present_simulations - set(SIMULATION_DIRS)
    if unknown_simulations:
        raise RuntimeError(
            "Figure 1 gallery contains unknown simulations: "
            f"{sorted(unknown_simulations)}"
        )
    sort_columns = ["simulation"]
    if "start_time" in events:
        sort_columns.append("start_time")
    sort_columns.append("event_id")
    return events.sort_values(sort_columns).reset_index(drop=True)


def tracking_movie_windows(
    time_values: Sequence,
    *,
    window_days: float = TRACKING_MOVIE_WINDOW_DAYS,
    step_hours: float = TRACKING_MOVIE_FRAME_STEP_HOURS,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Return complete sliding windows spanning a saved tracking record."""

    if window_days <= 0.0 or step_hours <= 0.0:
        raise ValueError("Movie window and frame step must be positive")
    stamps = pd.DatetimeIndex(pd.to_datetime(time_values)).dropna().sort_values()
    if stamps.empty:
        raise RuntimeError("The tracking movie has no valid saved times")
    record_start = pd.Timestamp(stamps[0])
    record_end = pd.Timestamp(stamps[-1])
    window = pd.Timedelta(days=window_days)
    step = pd.Timedelta(hours=step_hours)
    if record_end - record_start <= window:
        return [(record_start, record_end)]
    last_start = record_end - window
    starts = list(pd.date_range(record_start, last_start, freq=step))
    if not starts or starts[-1] != last_start:
        starts.append(last_start)
    return [
        (pd.Timestamp(start), min(pd.Timestamp(start) + window, record_end))
        for start in starts
    ]


def available_method_example_lags(
    available_times: Sequence,
    reference_time,
    requested_lags_hours: Sequence[int] = METHOD_EXAMPLE_LAGS_HOURS,
) -> tuple[int, ...]:
    """Return requested method-panel lags that exist in one event product."""

    available = pd.DatetimeIndex(pd.to_datetime(available_times)).dropna()
    reference = pd.Timestamp(reference_time)
    available_nanoseconds = set(available.asi8)
    retained = tuple(
        int(lag)
        for lag in requested_lags_hours
        if (reference + pd.Timedelta(hours=int(lag))).value
        in available_nanoseconds
    )
    if not retained:
        raise RuntimeError(
            "The LPO method example has no saved time at any requested "
            f"display lag {tuple(requested_lags_hours)}"
        )
    return retained


def _regional_event_field(
    field: xr.DataArray,
    *,
    centre_longitude: float,
    centre_latitude: float,
    longitude_half_width: float = 45.0,
    latitude_half_width: float = 25.0,
) -> xr.DataArray:
    """Extract one seam-safe geographic view around an LPT event."""

    relative_longitude = (
        (
            np.asarray(field.longitude, dtype=float)
            - centre_longitude
            + 180.0
        )
        % 360.0
    ) - 180.0
    longitude_indices = np.flatnonzero(
        np.abs(relative_longitude) <= longitude_half_width
    )
    longitude_indices = longitude_indices[
        np.argsort(relative_longitude[longitude_indices])
    ]
    latitude_indices = np.flatnonzero(
        np.abs(
            np.asarray(field.latitude, dtype=float)
            - centre_latitude
        )
        <= latitude_half_width
    )
    if not len(longitude_indices) or not len(latitude_indices):
        raise RuntimeError(
            "The configured regional LPT test view does not intersect the "
            "saved event grid"
        )
    return field.isel(
        longitude=longitude_indices,
        latitude=latitude_indices,
    ).assign_coords(
        longitude=(
            centre_longitude
            + relative_longitude[longitude_indices]
        )
    )


def _plot_lpo_envelope_check(
    primary: Path,
    chosen: pd.Series,
):
    """Plot one Kerns--Chen Figure 1-style LPO identification check."""

    threshold = PRIMARY_CONFIGURATION.precipitation_threshold_mm_day
    raw_cmap, precipitation_norm = precip_rate_cmap_norm()
    filtered_cmap, filtered_norm = filtered_precipitation_cmap_norm()
    fig = plt.figure(figsize=(12, 6.2), layout="constrained")
    grid = fig.add_gridspec(
        2,
        4,
        width_ratios=(1.0, 1.0, 1.0, 0.045),
        wspace=0.08,
    )
    axes = np.asarray(
        [
            [fig.add_subplot(grid[row, column]) for column in range(3)]
            for row in range(2)
        ]
    )
    raw_colourbar_axis = fig.add_subplot(grid[0, 3])
    filtered_colourbar_axis = fig.add_subplot(grid[1, 3])
    raw_mappable = filtered_mappable = None
    product_path = _event_product_path(
        primary,
        chosen.simulation,
        chosen.event_id,
    )
    with xr.open_dataset(product_path) as dataset:
        required = {
            "unfiltered_72h_precipitation_mm_day",
            "filtered_72h_precipitation_mm_day",
            "lpt_envelope_mask",
        }
        missing = sorted(required - set(dataset.data_vars))
        if missing:
            raise RuntimeError(
                f"{product_path} lacks Figure 1 fields {missing}; rebuild "
                f"standard LPO products with {PRODUCT_VERSION}"
            )
        available = pd.DatetimeIndex(pd.to_datetime(dataset.Time.values))
        if len(available) < 3:
            raise RuntimeError(
                f"{chosen.event_id} has fewer than three LPO times"
            )
        target_times = (
            available[0],
            available[len(available) // 2],
            available[-1],
        )
        track = pd.read_csv(
            primary
            / chosen.simulation
            / "standard"
            / "lpt_event_relative_diagnostics.csv",
            parse_dates=["Time"],
        )
        track = track[track.event_id == chosen.event_id]
        reference = pd.Timestamp(chosen.reference_time)
        reference_step = track.iloc[
            np.argmin(np.abs(track.Time - reference))
        ]
        centre_longitude = float(
            reference_step.precip_weighted_centroid_lon
        )
        centre_latitude = float(reference_step.precip_weighted_centroid_lat)

        for column, time_value in enumerate(target_times):
            unfiltered = dataset[
                "unfiltered_72h_precipitation_mm_day"
            ].sel(Time=time_value).load()
            filtered = dataset[
                "filtered_72h_precipitation_mm_day"
            ].sel(Time=time_value).load()
            envelope = dataset.lpt_envelope_mask.sel(Time=time_value).load()
            current = track.iloc[
                np.argmin(np.abs(track.Time - time_value))
            ]
            regional_envelope = _regional_event_field(
                envelope,
                centre_longitude=centre_longitude,
                centre_latitude=centre_latitude,
            )
            recent = track[
                (track.Time <= time_value)
                & (track.Time >= time_value - pd.Timedelta(days=2))
            ]
            recent_longitude = centre_longitude + (
                (
                    recent.precip_weighted_centroid_lon.to_numpy(dtype=float)
                    - centre_longitude
                    + 180.0
                )
                % 360.0
                - 180.0
            )
            for row, field in enumerate((unfiltered, filtered)):
                axis = axes[row, column]
                regional = _regional_event_field(
                    field,
                    centre_longitude=centre_longitude,
                    centre_latitude=centre_latitude,
                )
                mappable = axis.pcolormesh(
                    regional.longitude,
                    regional.latitude,
                    regional,
                    cmap=raw_cmap if row == 0 else filtered_cmap,
                    norm=precipitation_norm if row == 0 else filtered_norm,
                    shading="auto",
                    rasterized=True,
                )
                if row == 0:
                    raw_mappable = mappable
                else:
                    filtered_mappable = mappable
                axis.contour(
                    regional_envelope.longitude,
                    regional_envelope.latitude,
                    regional_envelope,
                    levels=[0.5],
                    colors=LPO_ENVELOPE_COLOUR,
                    linewidths=2.0,
                )
                axis.plot(
                    recent_longitude,
                    recent.precip_weighted_centroid_lat,
                    color=CENTROID_TRACK_COLOUR,
                    lw=CENTROID_TRACK_LINEWIDTH,
                    solid_capstyle="round",
                    zorder=4,
                )
                centroid_longitude = centre_longitude + (
                    (
                        float(current.precip_weighted_centroid_lon)
                        - centre_longitude
                        + 180.0
                    )
                    % 360.0
                    - 180.0
                )
                centroid_latitude = float(
                    current.precip_weighted_centroid_lat
                )
                axis.scatter(
                    centroid_longitude,
                    centroid_latitude,
                    marker="x",
                    color="black",
                    s=34,
                    linewidths=1.4,
                    zorder=5,
                )
                axis.xaxis.set_major_formatter(
                    FuncFormatter(
                        lambda value, _position: f"{value % 360.0:.0f}"
                    )
                )
                axis.grid(alpha=0.18)
                if row == 0:
                    lag_hours = (
                        pd.Timestamp(time_value) - reference
                    ) / pd.Timedelta(hours=1)
                    axis.set_title(
                        f"{pd.Timestamp(time_value):%Y-%m-%d %HZ} "
                        f"({lag_hours:+.0f} h)"
                    )
                if column == 0:
                    field_label = (
                        "Rain"
                        if row == 0
                        else "Filtered rain"
                    )
                    axis.set_ylabel(f"{field_label}\nLatitude")
                if row == 1:
                    axis.set_xlabel("longitude (°E)")
    if raw_mappable is not None:
        fig.colorbar(
            raw_mappable,
            cax=raw_colourbar_axis,
            label=r"Rain (mm day$^{-1}$)",
        )
    if filtered_mappable is not None:
        fig.colorbar(
            filtered_mappable,
            cax=filtered_colourbar_axis,
            label=r"Filtered rain (mm day$^{-1}$)",
        )
    axes[0, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color=LPO_ENVELOPE_COLOUR,
                lw=2.0,
                label=f"LPO ({threshold:g} mm day$^{{-1}}$)",
            ),
            Line2D(
                [0],
                [0],
                color="black",
                marker="x",
                linestyle="none",
                label="centroid",
            ),
        ],
        loc="upper left",
        fontsize=7,
        frameon=True,
    )
    fig.suptitle(
        "LPO identification: "
        f"{SIMULATION_LABELS[chosen.simulation]} {chosen.event_id}"
    )
    return fig


def _concurrent_lpo_envelopes(
    primary: Path,
    events: pd.DataFrame,
    chosen: pd.Series,
    target_times: Sequence,
) -> dict[pd.Timestamp, list[xr.DataArray]]:
    """Load other retained LPO masks present at the displayed times."""

    targets = pd.DatetimeIndex(pd.to_datetime(target_times))
    output = {pd.Timestamp(time): [] for time in targets}
    others = events[
        (events.simulation == chosen.simulation)
        & (events.event_id != chosen.event_id)
    ].copy()
    if others.empty:
        return output
    others["start_time"] = pd.to_datetime(others.start_time)
    others["end_time"] = pd.to_datetime(others.end_time)
    others = others[
        (others.start_time <= targets.max())
        & (others.end_time >= targets.min())
    ]
    standard = primary / chosen.simulation / "standard"
    mask_index = pd.read_csv(
        standard / "lpt_event_mask_index.csv"
    ).set_index("event_id")
    for other in others.itertuples():
        if other.event_id not in mask_index.index:
            raise RuntimeError(
                f"{standard}: mask index lacks {other.event_id}"
            )
        mask_path = standard / mask_index.loc[other.event_id, "mask_file"]
        with xr.open_dataset(mask_path) as product:
            available = pd.DatetimeIndex(pd.to_datetime(product.Time.values))
            available_values = set(available.asi8)
            for target in targets:
                if target.value not in available_values:
                    continue
                output[pd.Timestamp(target)].append(
                    product.lpt_envelope_mask.sel(Time=target).load()
                )
    return output


def _plot_method_example(
    primary: Path,
    events: pd.DataFrame,
    *,
    tracking_figure_test: bool = False,
    chosen_event_id: str | None = None,
):
    if chosen_event_id is not None:
        candidates = events[events.event_id == chosen_event_id].copy()
        if len(candidates) != 1:
            raise RuntimeError(
                f"Figure 1 event {chosen_event_id!r} occurs "
                f"{len(candidates)} times in the saved event table"
            )
    elif tracking_figure_test:
        if "test_method_example" not in events:
            raise RuntimeError(
                "Tracking-test event products do not identify the saved "
                "method examples; re-run --test postprocessing"
            )
        candidates = events[
            events.test_method_example.astype(bool)
        ].copy()
        complete = candidates[
            candidates.complete_primary_lag_window.astype(bool)
        ]
        if len(complete):
            candidates = complete
    else:
        candidates = events[
            events.primary_tropical_sample.astype(bool)
            & events.complete_primary_lag_window.astype(bool)
        ].copy()
    if candidates.empty:
        raise RuntimeError(
            "Figure 1 has no eligible saved LPT method example"
        )
    # A median-area event with branches/restructuring is representative without
    # selecting the cleanest-looking case.
    candidates["representative_score"] = np.abs(
        candidates.maximum_area_km2 - candidates.maximum_area_km2.median()
    )
    if chosen_event_id is not None:
        chosen = candidates.iloc[0]
    else:
        branched = candidates[
            candidates.maximum_simultaneous_branches > 1
        ]
        chosen = (
            branched.sort_values("representative_score").iloc[0]
            if len(branched)
            else candidates.sort_values("representative_score").iloc[0]
        )
    dataset = xr.open_dataset(
        _event_product_path(primary, chosen.simulation, chosen.event_id)
    )
    reference = pd.Timestamp(chosen.reference_time)
    available = pd.DatetimeIndex(pd.to_datetime(dataset.Time.values))
    displayed_lags = available_method_example_lags(
        available,
        reference,
    )
    target_times = [
        reference + pd.Timedelta(hours=lag) for lag in displayed_lags
    ]
    track_steps = pd.read_csv(
        primary
        / chosen.simulation
        / "standard"
        / "lpt_event_relative_diagnostics.csv",
        parse_dates=["Time"],
    )
    track_steps = track_steps[track_steps.event_id == chosen.event_id]
    reference_step = track_steps.iloc[
        np.argmin(np.abs(track_steps.Time - reference))
    ]
    centre_longitude = float(
        reference_step.precip_weighted_centroid_lon
    )
    centre_latitude = float(
        reference_step.precip_weighted_centroid_lat
    )
    displayed_track = track_steps[
        (track_steps.Time >= min(target_times))
        & (track_steps.Time <= max(target_times))
    ].sort_values("Time")
    displayed_track_longitude = centre_longitude + (
        (
            displayed_track.precip_weighted_centroid_lon.to_numpy(dtype=float)
            - centre_longitude
            + 180.0
        )
        % 360.0
        - 180.0
    )
    concurrent_envelopes = _concurrent_lpo_envelopes(
        primary,
        events,
        chosen,
        target_times,
    )
    fig, axes = plt.subplots(
        3,
        len(target_times),
        figsize=(12, 8),
        constrained_layout=True,
        sharex=True,
        sharey=True,
        squeeze=False,
    )
    raw_mappable = filtered_mappable = None
    raw_cmap, precipitation_norm = precip_rate_cmap_norm()
    filtered_cmap, filtered_norm = filtered_precipitation_cmap_norm()
    regime_inside_cmap, regime_norm = high_cloud_class_cmap_norm(alpha=1.0)
    regime_outside_cmap, _ = high_cloud_class_cmap_norm(alpha=0.28)
    regime_inside_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    regime_outside_cmap.set_bad((1.0, 1.0, 1.0, 0.0))
    for column, time_value in enumerate(target_times):
        raw = dataset.raw_precipitation_mm_day.sel(Time=time_value)
        filtered = dataset.filtered_72h_precipitation_mm_day.sel(Time=time_value)
        mask = dataset.lpt_envelope_mask.sel(Time=time_value)
        if "cloud_regime_context" not in dataset:
            raise RuntimeError(
                f"{dataset.encoding.get('source', 'event product')} lacks "
                "cloud_regime_context; rebuild LPT postprocessing with "
                f"product version {PRODUCT_VERSION}"
            )
        regime = dataset.cloud_regime_context.sel(Time=time_value)
        current = track_steps.iloc[
            np.argmin(np.abs(track_steps.Time - time_value))
        ]
        current_longitude = centre_longitude + (
            (
                float(current.precip_weighted_centroid_lon)
                - centre_longitude
                + 180.0
            )
            % 360.0
            - 180.0
        )
        current_latitude = float(current.precip_weighted_centroid_lat)
        raw = _regional_event_field(
            raw,
            centre_longitude=centre_longitude,
            centre_latitude=centre_latitude,
        )
        filtered = _regional_event_field(
            filtered,
            centre_longitude=centre_longitude,
            centre_latitude=centre_latitude,
        )
        mask = _regional_event_field(
            mask,
            centre_longitude=centre_longitude,
            centre_latitude=centre_latitude,
        )
        regime = _regional_event_field(
            regime,
            centre_longitude=centre_longitude,
            centre_latitude=centre_latitude,
        )
        concurrent_regional = []
        for other_envelope in concurrent_envelopes[pd.Timestamp(time_value)]:
            try:
                regional_other = _regional_event_field(
                    other_envelope,
                    centre_longitude=centre_longitude,
                    centre_latitude=centre_latitude,
                )
            except RuntimeError:
                continue
            if bool(regional_other.any()):
                concurrent_regional.append(regional_other)
        raw_mappable = axes[0, column].pcolormesh(
            raw.longitude,
            raw.latitude,
            raw,
            cmap=raw_cmap,
            norm=precipitation_norm,
            shading="auto",
        )
        filtered_mappable = axes[1, column].pcolormesh(
            filtered.longitude,
            filtered.latitude,
            filtered,
            cmap=filtered_cmap,
            norm=filtered_norm,
            shading="auto",
        )
        axes[2, column].pcolormesh(
            mask.longitude,
            mask.latitude,
            xr.where(mask, np.nan, 1.0),
            cmap=ListedColormap(("0.94",)),
            shading="auto",
        )
        axes[2, column].pcolormesh(
            regime.longitude,
            regime.latitude,
            regime.where((~mask) & (regime > 0)),
            cmap=regime_outside_cmap,
            norm=regime_norm,
            shading="auto",
        )
        axes[2, column].pcolormesh(
            regime.longitude,
            regime.latitude,
            regime.where(mask & (regime > 0)),
            cmap=regime_inside_cmap,
            norm=regime_norm,
            shading="auto",
        )
        for row in range(3):
            for other_envelope in concurrent_regional:
                axes[row, column].contour(
                    other_envelope.longitude,
                    other_envelope.latitude,
                    other_envelope,
                    levels=[0.5],
                    colors=LPO_ENVELOPE_COLOUR,
                    linewidths=1.1,
                    linestyles="--",
                    alpha=0.35,
                    zorder=3.5,
                )
            axes[row, column].contour(
                mask.longitude,
                mask.latitude,
                mask,
                levels=[0.5],
                colors=LPO_ENVELOPE_COLOUR,
                linewidths=2.0,
                zorder=4,
            )
            axes[row, column].plot(
                displayed_track_longitude,
                displayed_track.precip_weighted_centroid_lat,
                color=CENTROID_TRACK_COLOUR,
                lw=CENTROID_TRACK_LINEWIDTH,
                solid_capstyle="round",
                zorder=5,
            )
            axes[row, column].scatter(
                current_longitude,
                current_latitude,
                marker="x",
                color=CENTROID_TRACK_COLOUR,
                s=34,
                linewidths=1.4,
                zorder=6,
            )
            axes[row, column].xaxis.set_major_formatter(
                FuncFormatter(
                    lambda value, _position: f"{value % 360.0:.0f}"
                )
            )
            axes[row, column].grid(alpha=0.18)
            axes[row, column].set_title(
                (
                    f"{(time_value-reference)/pd.Timedelta(hours=1):+.0f} h"
                    if row == 0
                    else ""
                )
            )
    threshold = PRIMARY_CONFIGURATION.precipitation_threshold_mm_day
    for axis, label in zip(
        axes[:, 0],
        (
            "Rain",
            "Filtered rain",
            "High cloud",
        ),
    ):
        axis.set_ylabel(label + "\nLatitude")
    for axis in axes[-1]:
        axis.set_xlabel("Longitude")
    if raw_mappable is not None:
        fig.colorbar(
            raw_mappable,
            ax=axes[0, :],
            pad=0.01,
            label=r"Rain (mm day$^{-1}$)",
        )
    if filtered_mappable is not None:
        fig.colorbar(
            filtered_mappable,
            ax=axes[1, :],
            pad=0.01,
            label=r"Filtered rain (mm day$^{-1}$)",
        )
    axes[2, 0].legend(
        handles=[
            *high_cloud_class_legend_handles()[:3],
            Line2D(
                [0],
                [0],
                color=LPO_ENVELOPE_COLOUR,
                lw=2.0,
                label=f"LPO ({threshold:g} mm day$^{{-1}}$)",
            ),
            Line2D(
                [0],
                [0],
                color=LPO_ENVELOPE_COLOUR,
                lw=1.1,
                ls="--",
                alpha=0.35,
                label="Other concurrent LPOs",
            ),
            Line2D(
                [0],
                [0],
                color=CENTROID_TRACK_COLOUR,
                lw=CENTROID_TRACK_LINEWIDTH,
                label="Centroid path",
            ),
            Line2D(
                [0],
                [0],
                color=CENTROID_TRACK_COLOUR,
                marker="x",
                linestyle="none",
                label="Current centroid",
            ),
        ],
        loc="lower left",
        fontsize=6.5,
        frameon=True,
    )
    title_prefix = (
        "Representative LPO"
        if chosen_event_id is None
        else f"LPO example ({SIMULATION_LABELS[chosen.simulation]})"
    )
    fig.suptitle(f"{title_prefix}: {chosen.event_id}")
    dataset.close()
    return fig


def _write_lpo_envelope_gallery(
    primary: Path,
    events: pd.DataFrame,
    output_directory: Path,
    *,
    metadata_output_directory: Path | None = None,
    tracking_figure_test: bool,
) -> None:
    """Write one Kerns--Chen-style identification figure per retained LPO."""

    selected = publication_method_gallery_events(events)
    output_directory.mkdir(parents=True, exist_ok=True)
    metadata_output_directory = (
        output_directory
        if metadata_output_directory is None
        else metadata_output_directory
    )
    metadata_output_directory.mkdir(parents=True, exist_ok=True)
    selection_columns = [
        column
        for column in (
            "simulation",
            "event_id",
            "start_time",
            "end_time",
            "reference_time",
            "duration_hours",
            "maximum_area_km2",
            "maximum_simultaneous_branches",
        )
        if column in selected
    ]
    selection_table = selected[selection_columns].copy()
    selection_table.insert(
        0,
        "gallery_order",
        np.arange(1, len(selection_table) + 1),
    )
    selection_table["selection_method"] = (
        "all retained LPOs in every available simulation"
    )
    selection_table["product_version"] = PRODUCT_VERSION
    selection_table.to_csv(
        metadata_output_directory / "selected_events.csv",
        index=False,
    )
    for number, event in enumerate(selected.itertuples(), start=1):
        progress(
            "Figure 1 envelope gallery: rendering "
            f"{event.simulation}/{event.event_id} "
            f"({number}/{len(selected)})"
        )
        figure = _plot_method_example(
            primary,
            selected,
            tracking_figure_test=tracking_figure_test,
            chosen_event_id=event.event_id,
        )
        safe_event_id = str(event.event_id).replace("/", "_")
        output_path = output_directory / f"lpo_envelope_{safe_event_id}.png"
        try:
            figure.savefig(output_path, dpi=300)
        finally:
            plt.close(figure)
        progress(f"Done with figure file: {output_path}")


def periodic_track_segments(
    longitude: Sequence[float],
    time: Sequence,
) -> list[tuple[np.ndarray, pd.DatetimeIndex]]:
    """Split a track at periodic seams and inactive family intervals."""

    lon = np.mod(np.asarray(longitude, dtype=float), 360.0)
    stamps = pd.DatetimeIndex(pd.to_datetime(time))
    valid = np.isfinite(lon) & stamps.notna()
    valid_indices = np.flatnonzero(valid)
    if len(valid_indices) == 0:
        raise RuntimeError("Saved LPT track contains no finite points")
    boundaries = (
        np.flatnonzero(
            (np.diff(valid_indices) > 1)
            | (np.abs(np.diff(lon[valid_indices])) > 180.0)
        )
        + 1
    )
    segments = []
    for start, end in zip(
        np.r_[0, boundaries],
        np.r_[boundaries, len(valid_indices)],
    ):
        indices = valid_indices[start:end]
        if len(indices):
            segments.append((lon[indices], stamps[indices]))
    return segments


def tropical_track_segments(
    longitude: Sequence[float],
    latitude: Sequence[float],
    *,
    latitude_limit: float = TROPICAL_CENTROID_LIMIT_DEG,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split visible tropical track segments at gaps and map seams."""

    lon = np.mod(np.asarray(longitude, dtype=float), 360.0)
    lat = np.asarray(latitude, dtype=float)
    if lon.shape != lat.shape:
        raise ValueError("Track longitude and latitude shapes differ")
    valid = (
        np.isfinite(lon)
        & np.isfinite(lat)
        & (np.abs(lat) <= float(latitude_limit))
    )
    valid_indices = np.flatnonzero(valid)
    if not len(valid_indices):
        return []
    boundaries = (
        np.flatnonzero(
            (np.diff(valid_indices) > 1)
            | (np.abs(np.diff(lon[valid_indices])) > 180.0)
        )
        + 1
    )
    segments = []
    for start, end in zip(
        np.r_[0, boundaries],
        np.r_[boundaries, len(valid_indices)],
    ):
        indices = valid_indices[start:end]
        if len(indices):
            segments.append((lon[indices], lat[indices]))
    return segments


def _read_plot_track_steps(standard: Path) -> pd.DataFrame:
    """Read upstream geometric centroids for tracking-overview plots."""

    track_path = standard / "lpt_track_steps.csv"
    if not track_path.is_file():
        raise FileNotFoundError(
            f"Missing standard upstream track steps: {track_path}"
        )
    steps = pd.read_csv(track_path, parse_dates=["Time"])
    required = (
        "event_id",
        "Time",
        "centroid_lon",
        "centroid_lat",
        "area_km2",
    )
    missing = sorted(set(required) - set(steps.columns))
    if missing:
        raise RuntimeError(
            f"{track_path} lacks required plotting fields: {missing}"
        )
    return steps[list(required)].rename(
        columns={
            "centroid_lon": "plot_centroid_lon",
            "centroid_lat": "plot_centroid_lat",
            "area_km2": "plot_area_km2",
        }
    )


def _event_is_highlighted(event) -> bool:
    required = ("primary_tropical_sample", "reference_in_record_subset")
    values = []
    for name in required:
        if not hasattr(event, name) or pd.isna(getattr(event, name)):
            raise RuntimeError(
                f"Saved LPT event lacks required boolean {name!r}"
            )
        values.append(bool(getattr(event, name)))
    return all(values)


def simulation_day_values(
    time_values: Sequence,
    record_start: pd.Timestamp,
) -> np.ndarray:
    """Convert timestamps to elapsed simulation days, numbered from day 1."""

    stamps = pd.DatetimeIndex(pd.to_datetime(time_values))
    return np.asarray(
        1.0 + (stamps - pd.Timestamp(record_start)) / pd.Timedelta(days=1),
        dtype=float,
    )


def _plot_saved_track_panel(
    axis,
    standard: Path,
    events: pd.DataFrame,
    *,
    show_envelope_edges: bool = False,
    annotate_ids: bool = False,
    time_origin: pd.Timestamp | None = None,
    time_bounds: tuple[pd.Timestamp, pd.Timestamp] | None = None,
):
    """Draw one Kerns--Chen-style panel from the stable saved products."""

    with xr.open_dataset(standard / "lpt_time_longitude_overview.nc") as overview:
        variable = "equatorial_filtered_precipitation_mm_day"
        candidate_variable = "equatorial_candidate_lpo_envelope"
        missing_variables = [
            name
            for name in (variable, candidate_variable)
            if name not in overview
        ]
        if missing_variables:
            raise RuntimeError(
                f"{standard}: Figure 2 requires {missing_variables}. Rebuild the "
                f"standard products with {PRODUCT_VERSION}."
            )
        cached_latitude_band = overview.attrs.get(
            "filtered_precipitation_latitude_band"
        )
        if cached_latitude_band != TRACKING_OVERVIEW_LATITUDE_BAND:
            raise RuntimeError(
                f"{standard}: Figure 2 requires "
                f"{TRACKING_OVERVIEW_LATITUDE_BAND!r}, but the saved "
                f"overview records {cached_latitude_band!r}. Rerun LPO "
                "postprocessing; tracking does not need to be repeated."
            )
        field = periodic_longitude_field(
            overview[variable].load()
        )
        candidate_envelopes = periodic_longitude_field(
            overview[candidate_variable].astype(bool).load()
        )
    if time_bounds is not None:
        field = field.sel(Time=slice(*time_bounds))
        candidate_envelopes = candidate_envelopes.sel(
            Time=slice(*time_bounds)
        )
        if field.sizes.get("Time", 0) == 0:
            raise RuntimeError(
                f"{standard}: saved overview does not intersect figure time "
                f"bounds {time_bounds[0]} to {time_bounds[1]}"
            )
    if time_origin is None:
        time_origin = pd.Timestamp(pd.to_datetime(field.Time.values[0]))
    precipitation_cmap, precipitation_norm = (
        filtered_precipitation_cmap_norm()
    )
    mappable = axis.pcolormesh(
        field.longitude,
        simulation_day_values(field.Time.values, time_origin),
        field,
        cmap=precipitation_cmap,
        norm=precipitation_norm,
        shading="auto",
        rasterized=True,
    )
    axis.contour(
        candidate_envelopes.longitude,
        simulation_day_values(candidate_envelopes.Time.values, time_origin),
        candidate_envelopes,
        levels=[0.5],
        colors=LPO_ENVELOPE_COLOUR,
        linewidths=0.55,
        linestyles="--",
        alpha=0.35,
        zorder=2.2,
    )
    steps = _read_plot_track_steps(standard)
    if time_bounds is not None:
        steps = steps[
            pd.to_datetime(steps.Time).between(*time_bounds)
        ].copy()
    mask_index = None
    if show_envelope_edges:
        index_path = standard / "lpt_event_mask_index.csv"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"Envelope boundaries require {index_path}"
            )
        mask_index = pd.read_csv(index_path).set_index("event_id")

    for event in events.itertuples():
        track = (
            steps[steps.event_id == event.event_id]
            .sort_values("Time")
            .reset_index(drop=True)
        )
        if track.empty:
            raise RuntimeError(
                f"{standard}: event {event.event_id} has no lifecycle track "
                "steps"
            )
        highlighted = _event_is_highlighted(event)
        colour = CENTROID_TRACK_COLOUR
        segments = periodic_track_segments(track.plot_centroid_lon, track.Time)
        for lon, stamps in segments:
            axis.plot(
                lon,
                simulation_day_values(stamps, time_origin),
                color=colour,
                lw=2.2 if highlighted else 0.9,
                alpha=1.0 if highlighted else 0.55,
                solid_capstyle="round",
                zorder=3,
            )

        valid_track = track[
            np.isfinite(track.plot_centroid_lon)
            & pd.to_datetime(track.Time, errors="coerce").notna()
            & (track.plot_area_km2 > 0.0)
        ].reset_index(drop=True)
        if valid_track.empty:
            raise RuntimeError(
                f"{standard}: event {event.event_id} has no finite active "
                "track steps"
            )
        if len(valid_track) > 1:
            interval_hours = np.nanmedian(
                np.diff(pd.DatetimeIndex(valid_track.Time))
                / np.timedelta64(1, "h")
            )
            marker_stride = (
                max(1, int(round(24.0 / interval_hours)))
                if np.isfinite(interval_hours) and interval_hours > 0
                else 1
            )
        else:
            marker_stride = 1
        markers = valid_track.iloc[::marker_stride]
        effective_radius_km = np.sqrt(
            np.maximum(markers.plot_area_km2.to_numpy(dtype=float), 0.0)
            / np.pi
        )
        marker_size = np.clip(
            (effective_radius_km / 95.0) ** 2, 5.0, 55.0
        )
        axis.scatter(
            np.mod(markers.plot_centroid_lon, 360.0),
            simulation_day_values(markers.Time, time_origin),
            s=marker_size,
            facecolor=colour,
            edgecolor="white",
            linewidth=0.35,
            zorder=3.5,
        )

        reference_time = getattr(event, "reference_time", None)
        if reference_time is None or pd.isna(reference_time):
            raise RuntimeError(
                f"Saved LPT event {event.event_id} lacks reference_time"
            )
        reference_time = pd.Timestamp(reference_time)
        reference_step = valid_track.iloc[
            np.argmin(np.abs(valid_track.Time - reference_time))
        ]
        axis.scatter(
            [reference_step.plot_centroid_lon % 360.0],
            simulation_day_values([reference_time], time_origin),
            marker="*",
            color="#ffd43b",
            edgecolor="black",
            linewidth=0.65,
            s=70,
            zorder=4,
        )

        if annotate_ids and highlighted:
            short_id = event.event_id.rsplit("_", 1)[-1]
            for row, vertical_alignment in (
                (valid_track.iloc[0], "top"),
                (valid_track.iloc[-1], "bottom"),
            ):
                axis.text(
                    row.plot_centroid_lon % 360.0,
                    simulation_day_values([row.Time], time_origin)[0],
                    short_id,
                    fontsize=7,
                    color="black",
                    fontweight="bold",
                    ha="center",
                    va=vertical_alignment,
                    clip_on=True,
                    zorder=5,
                )

        if mask_index is not None:
            if event.event_id not in mask_index.index:
                raise RuntimeError(
                    f"{standard}: mask index lacks {event.event_id}"
                )
            mask_path = standard / mask_index.loc[event.event_id, "mask_file"]
            with xr.open_dataset(mask_path) as product:
                envelope_mask = product.lpt_envelope_mask
                if time_bounds is not None:
                    envelope_mask = envelope_mask.sel(
                        Time=slice(*time_bounds)
                    )
                envelope = equatorial_envelope_for_plot(envelope_mask)
                axis.contour(
                    envelope.longitude,
                    simulation_day_values(envelope.Time.values, time_origin),
                    envelope,
                    levels=[0.5],
                    colors=LPO_ENVELOPE_COLOUR,
                    linewidths=0.7,
                    alpha=0.9,
                    zorder=2.5,
                )

    axis.set_xlim(0, 360)
    axis.set_xticks(np.arange(0, 361, 60))
    axis.grid(color="black", linestyle="--", linewidth=0.35, alpha=0.28)
    axis.yaxis.set_major_locator(MaxNLocator(nbins=10, integer=True))
    return mappable


def _track_legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=LPO_ENVELOPE_COLOUR,
            alpha=0.35,
            lw=0.8,
            ls="--",
            label="Candidate envelope",
        ),
        Line2D(
            [0],
            [0],
            color=LPO_ENVELOPE_COLOUR,
            lw=1.2,
            label="LPO boundary",
        ),
        Line2D(
            [0],
            [0],
            color=CENTROID_TRACK_COLOUR,
            alpha=0.55,
            lw=0.9,
            label="Other LPOs",
        ),
        Line2D(
            [0],
            [0],
            color=CENTROID_TRACK_COLOUR,
            marker="o",
            markeredgecolor="black",
            markersize=5,
            lw=2.2,
            label="Tropical LPOs",
        ),
        Line2D(
            [0],
            [0],
            color="none",
            marker="*",
            markerfacecolor="#ffd43b",
            markeredgecolor="black",
            markersize=9,
            label="Peak rain",
        ),
    ]


def _plot_tracking_overview(
    primary: Path,
    events: pd.DataFrame,
    *,
    simulation: str,
    day_range: tuple[int, int] | None = None,
    tracking_figure_test: bool = False,
):
    if simulation not in SIMULATION_DIRS:
        raise ValueError(f"Unknown simulation {simulation!r}")
    panel_ranges = (
        (day_range,)
        if day_range is not None
        else (None,)
        if tracking_figure_test
        else ((1, 100), (101, 200))
    )
    fig, axes = plt.subplots(
        1,
        len(panel_ranges),
        figsize=(12, 9) if len(panel_ranges) == 1 else (17, 8.5),
        constrained_layout=True,
        sharex=True,
        squeeze=False,
    )
    standard = primary / simulation / "standard"
    sim_events = events[events.simulation == simulation]
    mappable = None
    for column, panel_range in enumerate(panel_ranges):
        axis = axes[0, column]
        mappable = _plot_saved_track_panel(
            axis,
            standard,
            sim_events,
            show_envelope_edges=True,
            annotate_ids=True,
        )
        axis.set_xlabel("Longitude (°E)")
        axis.set_ylabel("Simulation day")
        if panel_range is not None:
            day_start, day_end = panel_range
            axis.set_ylim(float(day_start), float(day_end + 1))
            ticks = [day_start]
            ticks.extend(
                value
                for value in range(
                    20 * int(np.ceil(day_start / 20.0)),
                    day_end + 1,
                    20,
                )
                if value != day_start
            )
            if ticks[-1] != day_end:
                ticks.append(day_end)
            axis.set_yticks(ticks)
            axis.set_title(f"Simulation days {day_start}–{day_end}")
    axes[0, 0].legend(
        handles=_track_legend_handles(),
        loc="upper left",
        fontsize=8,
        frameon=True,
    )
    if mappable is None:
        raise RuntimeError(f"No Figure 2 panels were drawn for {simulation}")
    fig.colorbar(
        mappable,
        ax=axes,
        pad=0.015,
        label=r"Filtered rain (mm day$^{-1}$)",
    )
    record_label = (
        "Actual-data test subset"
        if tracking_figure_test
        else "Full record"
    )
    fig.suptitle(
        f"{record_label}: {SIMULATION_LABELS[simulation]} "
        "20°S–20°N filtered rain and LPO tracks"
    )
    return fig


def _plot_all_lpo_tracks_map(primary: Path, events: pd.DataFrame):
    """Plot all retained geometric-centroid tracks in the tropical belt."""

    data_crs = ccrs.PlateCarree()
    projection = ccrs.PlateCarree(central_longitude=180.0)
    fig, axes = plt.subplots(
        len(SIMULATION_DIRS),
        1,
        figsize=(13.5, 7.2),
        constrained_layout=True,
        subplot_kw={"projection": projection},
        squeeze=False,
    )
    for row, simulation in enumerate(SIMULATION_DIRS):
        axis = axes[row, 0]
        standard = primary / simulation / "standard"
        steps = _read_plot_track_steps(standard)
        simulation_events = events[events.simulation == simulation]
        visible_count = 0
        for event_id in simulation_events.event_id.astype(str):
            track = steps[steps.event_id.astype(str) == event_id].sort_values(
                "Time"
            )
            if track.empty:
                raise RuntimeError(
                    f"{standard}: event {event_id} has no track steps"
                )
            segments = tropical_track_segments(
                track.plot_centroid_lon,
                track.plot_centroid_lat,
                latitude_limit=TRACK_MAP_LATITUDE_LIMIT_DEG,
            )
            if segments:
                visible_count += 1
            for longitude, latitude in segments:
                axis.plot(
                    longitude,
                    latitude,
                    color=SIMULATION_COLOURS[simulation],
                    lw=0.85,
                    marker=".",
                    markersize=1.8,
                    alpha=0.62,
                    solid_capstyle="round",
                    transform=data_crs,
                    zorder=2,
                )
        axis.set_extent(
            [
                0.001,
                359.999,
                -TRACK_MAP_LATITUDE_LIMIT_DEG,
                TRACK_MAP_LATITUDE_LIMIT_DEG,
            ],
            crs=data_crs,
        )
        axis.set_xticks(np.arange(0.0, 360.0, 60.0), crs=data_crs)
        axis.set_yticks(
            np.arange(
                -TRACK_MAP_LATITUDE_LIMIT_DEG,
                TRACK_MAP_LATITUDE_LIMIT_DEG + 1.0,
                10.0,
            ),
            crs=data_crs,
        )
        axis.xaxis.set_major_formatter(
            LongitudeFormatter(zero_direction_label=True)
        )
        axis.yaxis.set_major_formatter(LatitudeFormatter())
        axis.gridlines(
            crs=data_crs,
            xlocs=np.arange(0.0, 360.0, 60.0),
            ylocs=np.arange(
                -TRACK_MAP_LATITUDE_LIMIT_DEG,
                TRACK_MAP_LATITUDE_LIMIT_DEG + 1.0,
                10.0,
            ),
            color="0.5",
            linewidth=0.35,
            linestyle="--",
            alpha=0.35,
        )
        axis.set_title(
            f"{SIMULATION_LABELS[simulation]}: "
            f"{visible_count} of {len(simulation_events)} tracks intersect "
            "10°S–10°N",
            loc="left",
        )
        if row == len(SIMULATION_DIRS) - 1:
            axis.set_xlabel("Longitude")
    axes[0, 0].legend(
        handles=[
            Line2D(
                [0],
                [0],
                color="0.3",
                lw=1.2,
                label="Retained LPO geometric-centroid track",
            )
        ],
        loc="lower left",
        frameon=True,
        fontsize=8,
    )
    fig.suptitle("Full-record aquaplanet LPO tracks within 10°S–10°N")
    return fig


def _tracking_movie_times(
    primary: Path,
    simulation: str,
) -> pd.DatetimeIndex:
    """Return the saved time interval for one simulation's overview movie."""

    overview_path = (
        primary
        / simulation
        / "standard"
        / "lpt_time_longitude_overview.nc"
    )
    with xr.open_dataset(overview_path) as overview:
        if "Time" not in overview.coords or not overview.Time.size:
            raise RuntimeError(
                f"The tracking movie has no saved times in {overview_path}"
            )
        times = pd.DatetimeIndex(pd.to_datetime(overview.Time.values)).dropna()
    if times.empty:
        raise RuntimeError(
            f"The tracking movie has no valid times in {overview_path}"
        )
    return times.sort_values()


def _write_tracking_overview_movie(
    primary: Path,
    events: pd.DataFrame,
    output_path: Path,
    *,
    metadata_path: Path | None = None,
    simulation: str,
    tracking_figure_test: bool = False,
) -> None:
    """Animate sliding time windows for one simulation's LPT overview."""

    if not animation.writers.is_available("ffmpeg"):
        raise RuntimeError(
            "Figure 2 movie output requires ffmpeg to write its MP4; "
            "install ffmpeg in the diagnostic environment"
        )
    movie_times = _tracking_movie_times(primary, simulation)
    window_days = TRACKING_MOVIE_WINDOW_DAYS
    step_hours = (
        6 if tracking_figure_test else TRACKING_MOVIE_FRAME_STEP_HOURS
    )
    windows = tracking_movie_windows(
        movie_times,
        window_days=window_days,
        step_hours=step_hours,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(
        f"{output_path.stem}.tmp{output_path.suffix}"
    )
    figure, axis = plt.subplots(
        1,
        1,
        figsize=(12, 8),
        constrained_layout=True,
    )
    standard = primary / simulation / "standard"
    mappable = _plot_saved_track_panel(
        axis,
        standard,
        events[events.simulation == simulation],
        show_envelope_edges=True,
        annotate_ids=True,
    )
    axis.set_xlabel("Longitude (°E)")
    axis.set_ylabel("Simulation day")
    axis.legend(
        handles=_track_legend_handles(),
        loc="upper left",
        fontsize=8,
        frameon=True,
    )
    figure.colorbar(
        mappable,
        ax=axis,
        pad=0.015,
        label=r"Filtered rain (mm day$^{-1}$)",
    )
    title = figure.suptitle("")
    writer = animation.FFMpegWriter(
        fps=TRACKING_MOVIE_FRAMES_PER_SECOND,
        codec="libx264",
        bitrate=3000,
        metadata={
            "title": (
                f"Test-period {simulation} MPAS LPO tracks"
                if tracking_figure_test
                else f"Full-record {simulation} MPAS LPO tracks"
            ),
            "artist": "MPAS-A diagnostic workflow",
            "comment": (
                f"{UPSTREAM_CITATION}; project product {PRODUCT_VERSION}"
            ),
        },
        extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"],
    )
    progress(
        "Starting Figure 2 movie: "
        f"{simulation}, "
        f"{len(windows)} frames spanning "
        f"{movie_times[0]:%Y-%m-%d %HZ} to "
        f"{movie_times[-1]:%Y-%m-%d %HZ}"
    )
    try:
        with writer.saving(figure, str(temporary_path), dpi=120):
            for frame_number, (window_start, window_end) in enumerate(
                windows,
                start=1,
            ):
                window_start_day, window_end_day = simulation_day_values(
                    [window_start, window_end],
                    movie_times[0],
                )
                axis.set_ylim(window_start_day, window_end_day)
                title.set_text(
                    f"{SIMULATION_LABELS[simulation]} LPO tracks: "
                    f"simulation days {window_start_day:.0f}–"
                    f"{window_end_day:.0f}"
                )
                writer.grab_frame(facecolor="white")
                if (
                    frame_number == 1
                    or frame_number == len(windows)
                    or frame_number % 10 == 0
                ):
                    progress(
                        "Figure 2 movie: wrote frame "
                        f"{frame_number}/{len(windows)}"
                    )
        temporary_path.replace(output_path)
    except Exception:
        with contextlib.suppress(FileNotFoundError):
            temporary_path.unlink()
        raise
    finally:
        plt.close(figure)
    metadata = {
        "product_version": PRODUCT_VERSION,
        "upstream_repository": UPSTREAM_REPOSITORY,
        "upstream_commit": UPSTREAM_COMMIT,
        "source_figures": [f"02_lpo_tracking_overview_{simulation}.png"],
        "output_file": output_path.name,
        "simulation": simulation,
        "record_start": movie_times[0].isoformat(),
        "record_end": movie_times[-1].isoformat(),
        "window_days": window_days,
        "frame_step_hours": step_hours,
        "test_subset": tracking_figure_test,
        "frames_per_second": TRACKING_MOVIE_FRAMES_PER_SECOND,
        "frame_count": len(windows),
        "rain_field": "equatorial_filtered_precipitation_mm_day",
        "candidate_envelope_field": "equatorial_candidate_lpo_envelope",
        "candidate_centroids_drawn": False,
        "rain_latitude_band": TRACKING_OVERVIEW_LATITUDE_BAND,
        "time_axis": (
            "elapsed simulation days numbered from 1 at the first saved "
            "complete-filter time"
        ),
        "temporal_filter": "complete trailing 72-hour mean",
        "spatial_filter": "Gaussian sigma 5 degrees",
        "precipitation_colormap": "Blues with shared precipitation boundaries",
        "tracking_configuration": PRIMARY_CONFIGURATION.as_dict(),
        "track_figure_method": TRACK_FIGURE_CITATION,
    }
    metadata_path = (
        output_path.with_suffix(".json")
        if metadata_path is None
        else metadata_path
    )
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    progress(f"Done with Figure 2 movie file: {output_path}")


def lpo_population_summary(events: pd.DataFrame) -> pd.DataFrame:
    """Return population statistics for each simulation."""

    selected = events[events.primary_tropical_sample.astype(bool)]
    if "reference_in_record_subset" in selected:
        selected = selected[selected.reference_in_record_subset.astype(bool)]
    rows = []
    distribution_metrics = (
        "duration_hours",
        "mean_area_km2",
        "maximum_area_km2",
        "event_integrated_precipitation",
        "mean_filtered_precip_mm_day",
        "zonal_speed_m_s",
        "total_zonal_displacement_km",
        "maximum_simultaneous_branches",
    )
    for simulation in SIMULATION_DIRS:
        subset = selected[selected.simulation == simulation]
        if len(subset):
            days = float(subset.analysed_record_days.iloc[0])
            domain_area = float(subset.tracking_domain_area_km2.iloc[0])
            event_rate = 100.0 * len(subset) / days
            area_time_coverage = (
                100.0
                * subset.envelope_area_time_km2_h.sum()
                / (domain_area * days * 24.0)
            )
        else:
            days = domain_area = np.nan
            event_rate = area_time_coverage = 0.0
        row = {
            "product_version": PRODUCT_VERSION,
            "simulation": simulation,
            "event_count": int(len(subset)),
            "analysed_record_days": days,
            "tracking_domain_area_km2": domain_area,
            "events_per_100_days": event_rate,
            "envelope_area_time_coverage_percent": area_time_coverage,
            "event_weighting": "each retained tropical LPO has equal weight",
        }
        for metric in distribution_metrics:
            values = subset[metric] if metric in subset else pd.Series(dtype=float)
            row[f"median_{metric}"] = (
                float(values.median()) if len(values) else np.nan
            )
            row[f"mean_{metric}"] = (
                float(values.mean()) if len(values) else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows)


LPO_UNCERTAINTY_METRICS = (
    ("duration_hours", "h"),
    ("maximum_area_km2", "km2"),
    ("event_integrated_precipitation", "km2 mm"),
    ("zonal_speed_m_s", "m s-1"),
    ("total_zonal_displacement_km", "km"),
)
LPO_UNCERTAINTY_COMPARISONS = [
    (
        simulation,
        "control",
        f"{SIMULATION_LABELS[simulation]} - {SIMULATION_LABELS['control']}",
    )
    for simulation in tuple(SIMULATION_DIRS)[1:]
]
LPO_UNCERTAINTY_COMPARISONS.extend(
    (
        "nudged-u",
        simulation,
        f"{SIMULATION_LABELS['nudged-u']} - {SIMULATION_LABELS[simulation]}",
    )
    for simulation in tuple(SIMULATION_DIRS)[2:]
)
LPO_UNCERTAINTY_COMPARISONS = tuple(LPO_UNCERTAINTY_COMPARISONS)


def lpo_event_property_uncertainty(events: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap complete LPO events independently between simulations."""

    selected = _selected_population_events(events)
    rows = []
    for metric_index, (metric, units) in enumerate(LPO_UNCERTAINTY_METRICS):
        for comparison_index, (simulation, reference, label) in enumerate(
            LPO_UNCERTAINTY_COMPARISONS
        ):
            simulation_values = selected.loc[
                selected.simulation == simulation, metric
            ].to_numpy(dtype=float)
            reference_values = selected.loc[
                selected.simulation == reference, metric
            ].to_numpy(dtype=float)
            finite_simulation = int(np.isfinite(simulation_values).sum())
            finite_reference = int(np.isfinite(reference_values).sum())
            if finite_simulation and finite_reference:
                result = independent_bootstrap_difference(
                    simulation_values,
                    reference_values,
                    rng=np.random.default_rng(
                        BOOTSTRAP_SEED
                        + 3000
                        + 10 * metric_index
                        + comparison_index
                    ),
                    n_resamples=BOOTSTRAP_REPLICATES,
                )
                status = "ok"
            else:
                result = {
                    "estimate": np.nan,
                    "interval_low": np.nan,
                    "interval_high": np.nan,
                    "sign_probability": np.nan,
                    "n_simulation": finite_simulation,
                    "n_control": finite_reference,
                }
                status = "insufficient_events"
            rows.append(
                {
                    "product_version": PRODUCT_VERSION,
                    "metric": metric,
                    "units": units,
                    "simulation": simulation,
                    "reference": reference,
                    "comparison_label": label,
                    "mean_response": result["estimate"],
                    "ci_low": result["interval_low"],
                    "ci_high": result["interval_high"],
                    "sign_probability": result["sign_probability"],
                    "n_simulation_events": result["n_simulation"],
                    "n_reference_events": result["n_control"],
                    "status": status,
                    "event_weighting": "each retained tropical LPO has equal weight",
                    "resampling": "complete events sampled independently by simulation",
                }
            )
    return pd.DataFrame(rows)


def _plot_event_property_uncertainty(events: pd.DataFrame):
    """Plot event-property responses and complete-event uncertainty."""

    table = lpo_event_property_uncertainty(events)
    titles = {
        "duration_hours": "Duration response (h)",
        "maximum_area_km2": r"Maximum area response (km$^2$)",
        "event_integrated_precipitation": r"Integrated rain response (km$^2$ mm)",
        "zonal_speed_m_s": r"Zonal speed response (m s$^{-1}$)",
        "total_zonal_displacement_km": "Net zonal displacement response (km)",
    }
    colours = {
        simulation: SIMULATION_COLOURS[simulation]
        for simulation in tuple(SIMULATION_DIRS)[1:]
    }
    figure, axes = plt.subplots(5, 1, figsize=(8.5, 9.0), constrained_layout=True)
    y_positions = np.arange(len(LPO_UNCERTAINTY_COMPARISONS))
    for axis, (metric, _units) in zip(axes, LPO_UNCERTAINTY_METRICS):
        subset = table[table.metric == metric]
        for position, row in enumerate(subset.itertuples(index=False)):
            colour = colours.get(row.simulation, "0.25")
            axis.errorbar(
                row.mean_response,
                position,
                xerr=[[row.mean_response - row.ci_low], [row.ci_high - row.mean_response]],
                marker="o",
                color=colour,
                capsize=2,
            )
        axis.axvline(0.0, color="0.55", linewidth=0.8)
        axis.set_yticks(y_positions)
        axis.set_yticklabels(subset.comparison_label)
        axis.set_xlabel(titles[metric])
        axis.grid(axis="x", alpha=0.18)
    figure.suptitle("Tropical LPO event-property responses with 95% event-bootstrap intervals")
    return figure


def _selected_population_events(events: pd.DataFrame) -> pd.DataFrame:
    """Return the primary tropical event population."""

    selected = events[events.primary_tropical_sample.astype(bool)].copy()
    if "reference_in_record_subset" in selected:
        selected = selected[selected.reference_in_record_subset.astype(bool)]
    required = {
        "mean_filtered_precip_mm_day",
    }
    missing = sorted(required - set(selected.columns))
    if missing:
        raise RuntimeError(
            f"Figure 3 requires LPO product {PRODUCT_VERSION}; missing "
            f"event fields {missing}"
        )
    return selected


def _plot_event_distributions(
    events: pd.DataFrame,
    *,
    tracking_figure_test: bool = False,
):
    """Plot readable event-property CDFs."""

    selected = _selected_population_events(events)
    selected = selected.assign(
        duration_days=selected.duration_hours / 24.0,
        maximum_area_million_km2=selected.maximum_area_km2 / 1.0e6,
        integrated_precip_million_km2_mm=(
            selected.event_integrated_precipitation / 1.0e6
        ),
        zonal_displacement_thousand_km=(
            selected.total_zonal_displacement_km / 1.0e3
        ),
    )
    metrics = [
        ("duration_days", "Duration (days)"),
        (
            "maximum_area_million_km2",
            r"Maximum area ($10^6$ km$^2$)",
        ),
        (
            "mean_filtered_precip_mm_day",
            r"Mean filtered rain (mm day$^{-1}$)",
        ),
        (
            "integrated_precip_million_km2_mm",
            r"Integrated rain ($10^6$ km$^2$ mm)",
        ),
        ("zonal_speed_m_s", r"Zonal speed (m s$^{-1}$)"),
        (
            "zonal_displacement_thousand_km",
            r"Net zonal displacement ($10^3$ km)",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    line_styles = ("-", "--", ":")
    for metric_index, (axis, (column, label)) in enumerate(
        zip(axes.flat, metrics)
    ):
        for simulation_index, simulation in enumerate(SIMULATION_DIRS):
            values = selected.loc[
                selected.simulation == simulation, column
            ].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if not len(values):
                continue
            colour = SIMULATION_COLOURS[simulation]
            legend_label = (
                f"{SIMULATION_LABELS[simulation]} ($n={len(values)}$)"
            )
            ordered = np.sort(values)
            cumulative = (
                100.0
                * np.arange(1, len(ordered) + 1, dtype=float)
                / len(ordered)
            )
            axis.step(
                ordered,
                cumulative,
                where="post",
                color=colour,
                linestyle=line_styles[simulation_index],
                linewidth=2.0,
                label=legend_label,
            )
        axis.set_xlabel(label)
        axis.set_ylabel("Cumulative events (%)")
        axis.set_ylim(0.0, 102.0)
        axis.grid(alpha=0.22)
        axis.margins(x=0.03)
        if metric_index == 0:
            axis.legend(frameon=False, fontsize=7)
    prefix = (
        "Actual-data test (not for scientific inference): "
        if tracking_figure_test
        else ""
    )
    fig.suptitle(f"{prefix}tropical LPO property CDFs")
    return fig


def _plot_population_summaries(
    events: pd.DataFrame,
    *,
    tracking_figure_test: bool = False,
):
    """Plot compact event-rate, coverage and median-property summaries."""

    population = lpo_population_summary(events).set_index("simulation")
    panels = (
        ("events_per_100_days", "Tropical LPOs per 100 days", 1.0),
        (
            "envelope_area_time_coverage_percent",
            "Envelope area-time coverage (%)",
            1.0,
        ),
        ("median_duration_hours", "Median duration (days)", 1.0 / 24.0),
        (
            "median_maximum_area_km2",
            r"Median maximum area ($10^6$ km$^2$)",
            1.0e-6,
        ),
        (
            "median_mean_filtered_precip_mm_day",
            r"Median mean filtered rain (mm day$^{-1}$)",
            1.0,
        ),
        (
            "median_zonal_speed_m_s",
            r"Median zonal speed (m s$^{-1}$)",
            1.0,
        ),
    )
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for axis, (column, title, scale) in zip(axes.flat, panels):
        values = [
            population.loc[simulation, column] * scale
            for simulation in SIMULATION_DIRS
        ]
        axis.bar(
            range(len(SIMULATION_DIRS)),
            values,
            color=[SIMULATION_COLOURS[x] for x in SIMULATION_DIRS],
        )
        axis.axhline(0.0, color="0.3", linewidth=0.7)
        axis.set(
            title=title,
            xticks=range(len(SIMULATION_DIRS)),
            xticklabels=[SIMULATION_LABELS[x] for x in SIMULATION_DIRS],
        )
        axis.grid(axis="y", alpha=0.2)
        for index, simulation in enumerate(SIMULATION_DIRS):
            count = int(population.loc[simulation, "event_count"])
            value = values[index]
            axis.text(
                index,
                value if np.isfinite(value) else 0.0,
                f"n={count}",
                ha="center",
                va="bottom" if not np.isfinite(value) or value >= 0.0 else "top",
                fontsize=7,
            )
    prefix = (
        "Actual-data test (not for scientific inference): "
        if tracking_figure_test
        else ""
    )
    fig.suptitle(f"{prefix}tropical LPO population summaries")
    return fig


def _read_validation_events(output: Path) -> pd.DataFrame:
    """Read the selected event populations from every validation run."""

    rows = []
    for run_name in VALIDATION_RUN_NAMES:
        events, _ = read_standard_event_tables(
            output / run_name,
            simulations=VALIDATION_SIMULATIONS,
        )
        events["run_name"] = run_name
        if "primary_tropical_sample" in events:
            events = events[events.primary_tropical_sample.astype(bool)]
        if "reference_in_record_subset" in events:
            events = events[events.reference_in_record_subset.astype(bool)]
        rows.append(events)
    return pd.concat(rows, ignore_index=True)


def default_event_recovery(
    matching: pd.DataFrame,
    sensitivity_runs: Sequence[str],
) -> pd.DataFrame:
    """Fraction of default events with a one-to-one sensitivity match."""

    rows = []
    selected = matching[matching.sensitivity_run.isin(sensitivity_runs)]
    for sensitivity_run in sensitivity_runs:
        for simulation in VALIDATION_SIMULATIONS:
            subset = selected[
                (selected.sensitivity_run == sensitivity_run)
                & (selected.simulation == simulation)
            ]
            matched = int((subset.match_status == "matched").sum())
            unmatched_default = int(
                (subset.match_status == "unmatched_default").sum()
            )
            denominator = matched + unmatched_default
            rows.append(
                {
                    "sensitivity_run": sensitivity_run,
                    "simulation": simulation,
                    "recovery_fraction": (
                        matched / denominator if denominator else np.nan
                    ),
                }
            )
    return pd.DataFrame(rows)


def _median_iqr(values: pd.Series, scale: float = 1.0) -> tuple[float, ...]:
    """Return a scaled median and interquartile limits."""

    finite = pd.to_numeric(values, errors="coerce").dropna() / scale
    if finite.empty:
        return np.nan, np.nan, np.nan
    lower, median, upper = finite.quantile((0.25, 0.5, 0.75))
    return float(median), float(lower), float(upper)


def validation_sensitivity_summary(
    events: pd.DataFrame,
    matching: pd.DataFrame,
) -> pd.DataFrame:
    """Summarise each validation setting for the compact six-panel figure."""

    sensitivity_runs = tuple(
        name for name in VALIDATION_RUN_NAMES if name != "threshold_12"
    )
    recovery = default_event_recovery(
        matching,
        sensitivity_runs,
    ).set_index(["sensitivity_run", "simulation"])
    rows = []
    position = 0
    for family_index, (family, unit, settings) in enumerate(
        VALIDATION_PARAMETER_FAMILIES
    ):
        for run_name, parameter_label in settings:
            subset = events[
                (events.run_name == run_name)
                & (events.simulation == "control")
            ]
            duration = _median_iqr(subset.duration_hours, scale=24.0)
            speed = _median_iqr(subset.zonal_speed_m_s)
            precipitation = _median_iqr(
                subset.event_integrated_precipitation,
                scale=1.0e9,
            )
            merger_or_split = (
                subset.merger_occurred.astype(bool)
                | subset.split_occurred.astype(bool)
            )
            recovered = (
                1.0
                if run_name == "threshold_12"
                else recovery.loc[(run_name, "control"), "recovery_fraction"]
            )
            rows.append(
                {
                    "family": family,
                    "family_index": family_index,
                    "parameter_unit": unit,
                    "parameter_label": parameter_label,
                    "run_name": run_name,
                    "position": position,
                    "event_count": len(subset),
                    "duration_median_days": duration[0],
                    "duration_q25_days": duration[1],
                    "duration_q75_days": duration[2],
                    "speed_median_m_s": speed[0],
                    "speed_q25_m_s": speed[1],
                    "speed_q75_m_s": speed[2],
                    "recovery_fraction": recovered,
                    "precipitation_median_billion_km2_mm": precipitation[0],
                    "precipitation_q25_billion_km2_mm": precipitation[1],
                    "precipitation_q75_billion_km2_mm": precipitation[2],
                    "merger_split_fraction": (
                        float(merger_or_split.mean())
                        if len(merger_or_split)
                        else np.nan
                    ),
                }
            )
            position += 1
        position += 1
    return pd.DataFrame(rows)


def _plot_iqr_metric(
    axis,
    summary: pd.DataFrame,
    median_column: str,
    lower_column: str,
    upper_column: str,
) -> None:
    for _family_index, group in summary.groupby("family_index", sort=True):
        median = group[median_column].to_numpy(dtype=float)
        lower = group[lower_column].to_numpy(dtype=float)
        upper = group[upper_column].to_numpy(dtype=float)
        axis.errorbar(
            group.position,
            median,
            yerr=np.vstack((median - lower, upper - median)),
            marker="o",
            capsize=3,
            color=SIMULATION_COLOURS["control"],
        )


def _plot_grouped_metric(axis, summary: pd.DataFrame, column: str) -> None:
    """Plot independent lines for each sensitivity family."""

    for _family_index, group in summary.groupby("family_index", sort=True):
        axis.plot(
            group.position,
            group[column],
            marker="o",
            color=SIMULATION_COLOURS["control"],
        )


def _plot_validation_figure(output: Path):
    """Plot the grouped control-only sensitivity summary."""

    events = _read_validation_events(output)
    matching = pd.read_csv(output / "validation_event_matching.csv")
    summary = validation_sensitivity_summary(events, matching)
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    x = summary.position
    _plot_grouped_metric(axes[0, 0], summary, "event_count")
    _plot_iqr_metric(
        axes[0, 1],
        summary,
        "duration_median_days",
        "duration_q25_days",
        "duration_q75_days",
    )
    _plot_iqr_metric(
        axes[0, 2],
        summary,
        "speed_median_m_s",
        "speed_q25_m_s",
        "speed_q75_m_s",
    )
    _plot_grouped_metric(axes[1, 0], summary, "recovery_fraction")
    _plot_iqr_metric(
        axes[1, 1],
        summary,
        "precipitation_median_billion_km2_mm",
        "precipitation_q25_billion_km2_mm",
        "precipitation_q75_billion_km2_mm",
    )
    _plot_grouped_metric(
        axes[1, 2],
        summary,
        "merger_split_fraction",
    )
    titles_and_labels = (
        ("Tropical event count", "Events"),
        ("Median event duration (IQR)", "Duration (days)"),
        ("Median zonal speed (IQR)", r"Speed (m s$^{-1}$)"),
        ("Primary events recovered", "Recovered fraction"),
        (
            "Median integrated precipitation (IQR)",
            r"Precipitation ($10^9$ km$^2$ mm)",
        ),
        ("Events involving merger or split", "Event fraction"),
    )
    tick_labels = summary.parameter_label.tolist()
    family_groups = list(summary.groupby("family_index", sort=True))
    for axis, (title, ylabel) in zip(axes.flat, titles_and_labels):
        axis.set(title=title, ylabel=ylabel)
        axis.set_xticks(x, tick_labels)
        axis.grid(alpha=0.2)
        for group_index, (_family_index, group) in enumerate(family_groups):
            if group_index % 2:
                axis.axvspan(
                    group.position.min() - 0.45,
                    group.position.max() + 0.45,
                    color="0.94",
                    zorder=-2,
                )
            axis.text(
                group.position.mean(),
                -0.16,
                group.family.iloc[0],
                transform=axis.get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
            )
    axes[0, 2].axhline(0.0, color="0.45", lw=0.7)
    axes[1, 0].set_ylim(0.0, 1.05)
    axes[1, 2].set_ylim(0.0, 1.05)
    fig.suptitle("Control LPO tracking sensitivity: first 100 days")
    return fig


def _validation_figure_time_bounds(
    output: Path,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Read the exact calculation period used by validation figures."""

    reference_metadata_path = (
        output
        / "threshold_12"
        / "control"
        / "preprocessing_metadata.json"
    )
    reference_metadata = json.loads(reference_metadata_path.read_text())
    try:
        time_bounds = (
            pd.Timestamp(reference_metadata["reference_start"]),
            pd.Timestamp(reference_metadata["reference_end"]),
        )
    except KeyError as exc:
        raise RuntimeError(
            f"{reference_metadata_path} lacks validation reference bounds"
        ) from exc
    if time_bounds[1] <= time_bounds[0]:
        raise RuntimeError(
            f"{reference_metadata_path} has invalid validation reference "
            f"bounds {time_bounds[0]} to {time_bounds[1]}"
        )
    return time_bounds


def _plot_validation_track_figures(
    output: Path,
) -> list[tuple[str, object]]:
    """Show each control sensitivity using the saved upstream centroids."""

    figures: list[tuple[str, object]] = []
    suffixes = ("threshold", "gap", "overlap", "merge_split")
    time_bounds = _validation_figure_time_bounds(output)
    time_origin = time_bounds[0]
    for suffix, (title, unit, runs) in zip(
        suffixes,
        VALIDATION_PARAMETER_FAMILIES,
    ):
        fig, axes = plt.subplots(
            1,
            len(runs),
            figsize=(4.1 * len(runs), 5.2),
            constrained_layout=True,
            sharex=True,
            sharey=True,
            squeeze=False,
        )
        mappable = None
        for column, (run_name, run_label) in enumerate(runs):
            axis = axes[0, column]
            standard = output / run_name / "control" / "standard"
            events = pd.read_csv(standard / "lpt_events.csv")
            mappable = _plot_saved_track_panel(
                axis,
                standard,
                events,
                show_envelope_edges=True,
                time_origin=time_origin,
                time_bounds=time_bounds,
            )
            axis.set_title(
                run_label if unit == "setting" else f"{run_label} {unit}"
            )
            axis.set_xlabel("Longitude (°E)")
            if column == 0:
                axis.set_ylabel("Simulation day")
        if mappable is None:
            raise RuntimeError(f"No validation track panels were drawn for {suffix}")
        fig.colorbar(
            mappable,
            ax=axes,
            pad=0.015,
            label=r"Filtered rain (mm day$^{-1}$)",
        )
        axes[0, 0].legend(
            handles=_track_legend_handles(),
            loc="upper left",
            fontsize=6.5,
            frameon=True,
        )
        fig.suptitle(
            f"Control {title.lower()} sensitivity: first 100 days\n"
            "Kerns–Chen (2016) Figure 4a presentation"
        )
        figures.append((f"validation_lpo_tracks_{suffix}", fig))
    return figures


if __name__ == "__main__":
    raise SystemExit(main())
