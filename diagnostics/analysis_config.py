"""Fixed definitions for the MPAS-A analysis workflow.

This module contains project-wide definitions only.  Owner scripts retain
their diagnostic thresholds and transformations.
"""

from __future__ import annotations

import os
from pathlib import Path


# The workflow serves these existing archives.  Canonical names remain stable
# in caches.  Labels preserve the wording used by current publication figures.
SIMULATION_ROOT = Path("/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on")
SIMULATION_ARCHIVES = {
    "control": "control_new",
    "nudged-u": "u_nudge",
    "nudged-T": "T_nudge",
}

# All reusable calculation products live below one data-storage directory.
# Diagnostic modules select a logical family.  Only this mapping owns physical
# cache directory names.
CACHE_ROOT = SIMULATION_ROOT / "cache"
CACHE_DIRECTORIES = {
    "block_resampling": "block_resampling",
    "circulation": "circulation",
    "classification": "classification",
    "energy_budget": "energy_budget",
    "high_cloud": "high_cloud",
    "jet_variability": "jet_variability",
    "lpt": "lpt",
    "mean_state": "mean_state",
    "thermal_wind": "thermal_wind",
    "vapour_transport": "vapour_transport",
    "vapour_transport_data": "vapour_transport_data",
    "waves": "waves",
}
SIMULATION_LABELS = {
    "control": "control",
    "nudged-u": "nudged u",
    "nudged-T": "nudged T",
}
SIMULATION_PUBLICATION_LABELS = {
    "control": "control",
    "nudged-u": r"$u$-nudged",
    "nudged-T": r"$T$-nudged",
}
SIMULATION_COLOURS = {
    "control": "#222222",
    "nudged-u": "#1f77b4",
    "nudged-T": "#d62728",
}
SIMULATION_ALIASES = {
    "control": "control",
    "nudged-u": "nudged-u",
    "u-nudged": "nudged-u",
    "nudged u": "nudged-u",
    "nudged-t": "nudged-T",
    "t-nudged": "nudged-T",
    "nudged t": "nudged-T",
}

ANALYSIS_START = "2000-05-05 06:00:00"
ANALYSIS_END = "2000-11-21 00:00:00"
LATLON_PATTERN = "latlon.2000-*.nc"
ZONAL_MEAN_PATTERN = "zm_latlon.2000-*.nc"

DEFAULT_FIGURE_ROOT = Path("/home/cwp29/tmp/mpas/crf_on")
DIAGNOSTIC_OUTPUT_ROOT_ENV = "MPAS_DIAGNOSTIC_OUTPUT_ROOT"
DIAGNOSTIC_FIGURE_ROOT_ENV = "MPAS_DIAGNOSTIC_FIGURE_ROOT"
DIAGNOSTIC_FIGURE_VERSION_ENV = "MPAS_DIAGNOSTIC_FIGURE_VERSION"
DEFAULT_FIGURE_VERSION = "v1"

DEEP_TROPICAL_LAT_BOUNDS = (-10.0, 10.0)
TROPICAL_LAT_BOUNDS = (-20.0, 20.0)
QUICK_TEST_LAT_BOUNDS = (-20.0, 20.0)
QUICK_TEST_LON_BOUNDS = (60.0, 180.0)

# MPAS archive variables are authoritative.  Derived variables require an
# equation and provenance statement in their owner module.
SOURCE_VARIABLES = {
    "relative_humidity": "relhum",
    "precipitable_water": "precipw",
    "cape": "cape",
    "outgoing_longwave_radiation": "olrtoa",
    "pressure": "pressure",
    "geometric_height_interfaces": "zgrid",
}

# Thermodynamic constants use the values already present in the diagnostics.
DRY_AIR_GAS_CONSTANT = 287.0
DRY_AIR_CP = 1004.0
REFERENCE_PRESSURE_PA = 100000.0
GRAVITY = 9.80665

# The control-only autocorrelation review selects these rounded values.  Every
# simulation uses the same value within each diagnostic family.
BLOCK_LENGTH_DAYS = {
    "circulation": 10.0,
    "energy_budget": 10.0,
    "high_cloud": 20.0,
    "jet_variability": 20.0,
    "mean_state_distributions": 20.0,
    "thermal_wind": 20.0,
}
BLOCK_LENGTH_RATIONALE = {
    "circulation": "Control circulation metrics support a 10-day block.",
    "energy_budget": "Control energy-budget metrics support a 10-day block.",
    "high_cloud": "Control high-cloud metrics support a 20-day block.",
    "jet_variability": "Control jet metrics support a 20-day block.",
    "mean_state_distributions": (
        "Control CAPE and precipitable-water distributions support a 20-day block."
    ),
    "thermal_wind": (
        "Control thermodynamic and circulation metrics support a 20-day block."
    ),
}
WAVE_RESPONSE_BLOCK_LENGTH_DAYS = {
    "Eastward_intraseasonal": 20.0,
    "Kelvin": 10.0,
    "MRG": 10.0,
    "Easterly": 20.0,
    "ER": 10.0,
}
WAVE_RESPONSE_BLOCK_LENGTH_RATIONALE = {
    "Eastward_intraseasonal": "Control filtered amplitude supports a 20-day block.",
    "Kelvin": "Control filtered amplitude supports a 10-day block.",
    "MRG": "Control filtered amplitude supports a 10-day block.",
    "Easterly": "Control filtered amplitude supports a 20-day block.",
    "ER": (
        "Control ER amplitude crosses the dependence threshold near the other "
        "10-day wave bands."
    ),
}
BOOTSTRAP_SEED = 20240602
BOOTSTRAP_REPLICATES = 1000
FIELD_BOOTSTRAP_REPLICATES = 600
CONFIDENCE_PERCENTILES = (2.5, 97.5)
FDR_ALPHA = 0.05


def canonical_simulation_name(name: str) -> str:
    """Return the canonical cache name for one simulation name or label."""

    key = str(name).strip().lower()
    try:
        return SIMULATION_ALIASES[key]
    except KeyError as exc:
        raise KeyError(f"Unknown simulation name {name!r}") from exc


def simulation_directory(name: str) -> Path:
    """Return the fixed archive directory for one simulation."""

    return SIMULATION_ROOT / SIMULATION_ARCHIVES[canonical_simulation_name(name)]


def simulation_colour(name: str) -> str:
    """Return the publication colour for one simulation name or label."""

    return SIMULATION_COLOURS[canonical_simulation_name(name)]


def cache_root() -> Path:
    """Return the data-storage root for permanent diagnostic products."""

    return CACHE_ROOT


def cache_directory(family: str, *parts: str) -> Path:
    """Return one centrally defined diagnostic cache path."""

    try:
        directory = CACHE_DIRECTORIES[family]
    except KeyError as exc:
        raise KeyError(f"Unknown cache family {family!r}") from exc
    return CACHE_ROOT.joinpath(directory, *parts)


def figure_root() -> Path:
    """Return the configured figure root."""

    value = os.environ.get(DIAGNOSTIC_OUTPUT_ROOT_ENV)
    value = value or os.environ.get(DIAGNOSTIC_FIGURE_ROOT_ENV)
    return Path(value).expanduser().resolve() if value else DEFAULT_FIGURE_ROOT
