"""Recover MPAS Gaussian nudging geometry from restart global attributes.

The MPAS kernel is exp(-(latitude / scale_lat)**2
                         - ((height - centre_z) / scale_z)**2).
Latitude uses radians. Height uses geometric metres at layer midpoints.
The e-fold contour describes relative relaxation strength, not a hard cutoff.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np
import xarray as xr

from diagnostics.analysis_config import (
    ALL_SIMULATION_ARCHIVES,
    ANALYSIS_END,
    ANALYSIS_START,
    canonical_simulation_name,
)


MPAS_SOURCE_COMMIT = "776bd1e87514ac1e19c6a192c0500a88a959f1a6"
MPAS_NUDGING_SOURCE = (
    "https://github.com/qntmCharles/MPAS-Model_aquaplanet/blob/"
    f"{MPAS_SOURCE_COMMIT}/src/core_atmosphere/dynamics/mpas_atm_time_integration.F"
)
NUDGING_CONTOUR_LEVEL = float(np.exp(-1.0))


@dataclass(frozen=True)
class NudgingRegion:
    """Store one enabled kernel and its restart provenance."""

    simulation: str
    variable: str
    centre_z_m: float
    scale_z_m: float
    scale_lat_rad: float
    timescale_days: float
    restart_path: str

    def contour(
        self, level: float = NUDGING_CONTOUR_LEVEL, points: int = 361,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return the analytic contour in absolute latitude and geometric height."""

        if not 0.0 < level < 1.0:
            raise ValueError("Nudging contour strength must lie between zero and one")
        if points < 3:
            raise ValueError("Nudging contours require at least three points")
        radius = np.sqrt(-np.log(level))
        angle = np.linspace(-np.pi / 2.0, np.pi / 2.0, points)
        latitude = np.rad2deg(self.scale_lat_rad) * radius * np.cos(angle)
        height = self.centre_z_m + self.scale_z_m * radius * np.sin(angle)
        return latitude, height

    def report(self) -> dict:
        """Describe the contour extent and local relaxation timescale."""

        return {
            **asdict(self),
            "contour_relative_strength": NUDGING_CONTOUR_LEVEL,
            "maximum_absolute_latitude_degrees": float(np.rad2deg(self.scale_lat_rad)),
            "equatorial_bottom_height_m": self.centre_z_m - self.scale_z_m,
            "equatorial_top_height_m": self.centre_z_m + self.scale_z_m,
            "contour_relaxation_timescale_days": self.timescale_days / NUDGING_CONTOUR_LEVEL,
        }


def _logical(value: object) -> bool:
    text = str(value).strip().upper()
    if text in {"YES", "TRUE", ".TRUE.", "1"}:
        return True
    if text in {"NO", "FALSE", ".FALSE.", "0"}:
        return False
    raise ValueError(f"Unrecognized MPAS logical attribute: {value!r}")


def regions_from_attributes(
    attributes: dict, simulation: str, restart_path: str,
) -> tuple[NudgingRegion, ...]:
    """Read enabled kernels without substituting namelist defaults."""

    regions = []
    for variable in ("u", "theta"):
        prefix = f"config_nudge_{variable}"
        if prefix not in attributes:
            raise ValueError(f"{restart_path}: missing {prefix}")
        if not _logical(attributes[prefix]):
            continue
        try:
            centre, vertical, latitude, timescale = (
                float(attributes[f"{prefix}_{suffix}"])
                for suffix in ("centre_z", "scale_z", "scale_lat", "timescale")
            )
        except KeyError as exc:
            raise ValueError(f"{restart_path}: missing {exc.args[0]}") from exc
        if not np.isfinite([centre, vertical, latitude, timescale]).all():
            raise ValueError(f"{restart_path}: nudging parameters must be finite")
        if min(vertical, latitude, timescale) <= 0.0:
            raise ValueError(f"{restart_path}: nudging scales and timescale must be positive")
        regions.append(NudgingRegion(
            canonical_simulation_name(simulation), variable,
            centre, vertical, latitude, timescale, restart_path,
        ))
    return tuple(regions)


def load_nudging_regions(
    root: Path, simulations: Sequence[str],
    start: str = ANALYSIS_START, end: str = ANALYSIS_END,
) -> dict[str, tuple[NudgingRegion, ...]]:
    """Use the latest restart and check available analysis-window geometry.

    Include the last preceding restart when the analysis starts between files.
    Timescale changes leave relative-strength contours unchanged.
    Use the nearest later snapshot when replacement metadata follows the window.
    """

    first, last = datetime.fromisoformat(start), datetime.fromisoformat(end)
    result = {}
    for simulation in simulations:
        canonical = canonical_simulation_name(simulation)
        directory = Path(root) / ALL_SIMULATION_ARCHIVES[canonical]
        available = []
        for path in sorted(directory.glob("restart.*.nc")):
            timestamp = datetime.strptime(path.name, "restart.%Y-%m-%d_%H.%M.%S.nc")
            available.append((timestamp, path))
        if not available:
            raise FileNotFoundError(f"{directory}: no restart metadata")
        snapshots = [item for item in available if item[0] <= last]
        if not snapshots:
            snapshots = [available[0]]
        preceding = [item for item in snapshots if item[0] <= first]
        selected = ([preceding[-1]] if preceding else []) + [
            item for item in snapshots if item[0] > first
        ]
        geometries = []
        for _timestamp, path in selected:
            # Read only the header of these multi-gigabyte files.
            with xr.open_dataset(path, decode_cf=False) as dataset:
                regions = regions_from_attributes(dataset.attrs, canonical, str(path))
            geometries.append(tuple(
                (r.variable, r.centre_z_m, r.scale_z_m, r.scale_lat_rad)
                for r in regions
            ))
        if any(geometry != geometries[-1] for geometry in geometries):
            raise ValueError(f"{directory}: nudging geometry changes across restart snapshots")
        result[canonical] = regions
    return result


def nudging_region_report(regions: dict[str, tuple[NudgingRegion, ...]]) -> dict:
    """Provide source and coordinate provenance for figures 6 and 10."""

    return {
        "mpas_source": MPAS_NUDGING_SOURCE,
        "mpas_source_commit": MPAS_SOURCE_COMMIT,
        "kernel": "exp(-(latitude_rad/scale_lat_rad)^2 - ((height_m-centre_z_m)/scale_z_m)^2)",
        "contour_definition": "e-fold relative relaxation strength; Gaussian tails extend beyond this contour",
        "pressure_mapping": "interpolate the common geometric-height TEM plotting pressure; no standard atmosphere",
        "timescale_provenance": "timescale_days comes from the latest restart; only kernel geometry is checked across snapshots",
        "metadata_selection": "latest snapshot through the analysis end; nearest later snapshot if earlier metadata is absent",
        "enabled_variables": {
            simulation: [region.variable for region in items]
            for simulation, items in regions.items()
        },
        "regions": [region.report() for items in regions.values() for region in items],
    }
