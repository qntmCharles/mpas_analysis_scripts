"""Fixed definitions for the MPAS-A analysis workflow.

This module contains project-wide definitions only.  Owner scripts retain
their diagnostic thresholds and transformations.
"""

from __future__ import annotations

import os
from pathlib import Path


# The workflow serves these archives. Canonical names remain stable in caches.
# The environment selects an ordered profile before diagnostic modules import
# these mappings. Direct diagnostic runs therefore use the same configuration
# as ``run_all_diagnostics.py``.
SIMULATION_ROOT = Path("/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on")
SIMULATION_PROFILE_ENV = "MPAS_SIMULATION_PROFILE"
DEFAULT_SIMULATION_PROFILE = "legacy"
ALL_SIMULATION_ARCHIVES = {
    "control": "control_new",
    "nudged-u": "u_nudge",
    "nudged-T": "T_nudge",
    "nudged-T-from-u": "T_nudge_from_u",
}
# Stable filename tokens for caches that store one simulation per file.
# These values must not depend on plot labels or profile selection.
ALL_SIMULATION_CACHE_NAMES = {
    "control": "control",
    "nudged-u": "nudged_u",
    "nudged-T": "nudged_T",
    "nudged-T-from-u": "nudged_T_from_u",
}
SIMULATION_PROFILES = {
    "legacy": ("control", "nudged-u", "nudged-T"),
    "from-u": ("control", "nudged-u", "nudged-T-from-u"),
    "both": ("control", "nudged-u", "nudged-T", "nudged-T-from-u"),
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
ALL_SIMULATION_LABELS = {
    "control": "control",
    "nudged-u": "U",
    "nudged-T": "T (old)",
    "nudged-T-from-u": "T",
}
ALL_SIMULATION_PUBLICATION_LABELS = dict(ALL_SIMULATION_LABELS)
ALL_SIMULATION_COLOURS = {
    "control": "#222222",
    "nudged-u": "#1f77b4",
    "nudged-T": "#9467bd",
    "nudged-T-from-u": "#d62728",
}
SIMULATION_ALIASES = {
    "control": "control",
    "nudged-u": "nudged-u",
    "u-nudged": "nudged-u",
    "nudged u": "nudged-u",
    "nudged-t": "nudged-T",
    "t-nudged": "nudged-T",
    "nudged t": "nudged-T",
    "nudged-t-from-u": "nudged-T-from-u",
    "t-nudged-from-u": "nudged-T-from-u",
    "nudged t from u": "nudged-T-from-u",
    # Historical display names retain their original archive identities.
    r"$u$-nudged": "nudged-u",
    r"$t$-nudged": "nudged-T",
    r"$t$-nudged from $u$": "nudged-T-from-u",
    "t-nudged (old)": "nudged-T",
    "t-nudged from u": "nudged-T-from-u",
    "t from u": "nudged-T-from-u",
}
SIMULATION_ALIASES.update(
    {
        str(label).strip().lower(): simulation
        for labels in (ALL_SIMULATION_LABELS, ALL_SIMULATION_PUBLICATION_LABELS)
        for simulation, label in labels.items()
    }
)


def simulation_profile_name(value: str | None = None) -> str:
    """Return one validated simulation profile name."""

    profile = value or os.environ.get(
        SIMULATION_PROFILE_ENV, DEFAULT_SIMULATION_PROFILE
    )
    try:
        SIMULATION_PROFILES[profile]
    except KeyError as exc:
        choices = ", ".join(SIMULATION_PROFILES)
        raise ValueError(
            f"Unknown simulation profile {profile!r}; choose one of: {choices}"
        ) from exc
    return profile


ACTIVE_SIMULATION_PROFILE = simulation_profile_name()
ACTIVE_SIMULATION_NAMES = SIMULATION_PROFILES[ACTIVE_SIMULATION_PROFILE]


def _selected(mapping: dict[str, str]) -> dict[str, str]:
    """Return one ordered mapping for the active profile."""

    return {name: mapping[name] for name in ACTIVE_SIMULATION_NAMES}


SIMULATION_ARCHIVES = _selected(ALL_SIMULATION_ARCHIVES)
SIMULATION_LABELS = _selected(ALL_SIMULATION_LABELS)
SIMULATION_PUBLICATION_LABELS = _selected(
    ALL_SIMULATION_PUBLICATION_LABELS
)
SIMULATION_COLOURS = _selected(ALL_SIMULATION_COLOURS)

ANALYSIS_START = "2000-05-05 06:00:00"
ANALYSIS_END = "2000-11-21 00:00:00"
LATLON_PATTERN = "latlon.2000-*.nc"
ZONAL_MEAN_PATTERN = "zm_latlon.2000-*.nc"

# Spin-up uses the same simulation root, archive mapping, and selected profile.
# Numeric spin-up Time counts six-hour samples from SPINUP_FIRST_TIME.
SPINUP_START = "2000-04-25 00:00:00"
SPINUP_END = "2000-05-05 00:00:00"
SPINUP_FIRST_TIME = "2000-04-25 06:00:00"
# An empty subdirectory keeps spin-up products in each simulation's main folder.
SPINUP_SUBDIRECTORY = ""
SPINUP_LATLON_FILENAME = "latlon_spinup.nc"
SPINUP_TEM_FILENAME = "tem_geometric_height_spinup.nc"
# T from U uses the same prescribed spin-up timescale history as nudged U.
SPINUP_NUDGING_TIMESCALE_SOURCES = {"nudged-T-from-u": "nudged-u"}

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


def simulation_cache_name(name: str) -> str:
    """Return the stable filename token for one simulation."""

    return ALL_SIMULATION_CACHE_NAMES[canonical_simulation_name(name)]


def simulation_plot_label(name: str) -> str:
    """Apply current display names while retaining unrelated experiment labels."""

    try:
        return ALL_SIMULATION_LABELS[canonical_simulation_name(name)]
    except KeyError:
        return name


def simulation_cache_label(name: str, *, publication: bool = False) -> str:
    """Preserve historical manifest labels independently of current plot names."""

    try:
        canonical = canonical_simulation_name(name)
    except KeyError:
        return name
    if publication:
        return {
            "control": "control",
            "nudged-u": r"$u$-nudged",
            "nudged-T": r"$T$-nudged",
            "nudged-T-from-u": r"$T$-nudged from $u$",
        }[canonical]
    return ALL_SIMULATION_CACHE_NAMES[canonical].replace("_", " ")


def simulation_directory(name: str) -> Path:
    """Return the fixed archive directory for one simulation."""

    return SIMULATION_ROOT / ALL_SIMULATION_ARCHIVES[
        canonical_simulation_name(name)
    ]


def spinup_directory(name: str) -> Path:
    """Return the explicitly configured spin-up location for one simulation."""

    root = SIMULATION_ROOT / SIMULATION_ARCHIVES[canonical_simulation_name(name)]
    return root / SPINUP_SUBDIRECTORY


def simulation_colour(name: str) -> str:
    """Return the publication colour for one simulation name or label."""

    return ALL_SIMULATION_COLOURS[canonical_simulation_name(name)]


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


def profile_cache_directory(
    family: str,
    *parts: str,
    profile: str | None = None,
) -> Path:
    """Return a cache path for products that combine simulations."""

    selected_profile = simulation_profile_name(profile)
    root = cache_directory(family)
    if selected_profile != DEFAULT_SIMULATION_PROFILE:
        root = root / "profiles" / selected_profile
    return root.joinpath(*parts)


def figure_root() -> Path:
    """Return the configured figure root."""

    value = os.environ.get(DIAGNOSTIC_OUTPUT_ROOT_ENV)
    value = value or os.environ.get(DIAGNOSTIC_FIGURE_ROOT_ENV)
    return Path(value).expanduser().resolve() if value else DEFAULT_FIGURE_ROOT
