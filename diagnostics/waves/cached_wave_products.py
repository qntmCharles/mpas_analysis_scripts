"""Read-only access to filtered products from equatorial_wave_diagnostics.py."""

import os

import xarray as xr

from diagnostics.diagnostic_helpers import diagnostic_cache_dir
from diagnostics.waves.equatorial_wave_config import WAVE_LATITUDE_PROJECTION


def filtered_wave_cache_path(
    root_dir, simulation, field_name, wave_name, latitude_tag="m10_10"
):
    projection = WAVE_LATITUDE_PROJECTION[wave_name]
    safe_simulation = simulation.replace(" ", "_")
    cache_dir = diagnostic_cache_dir("wave_cache", root_dir=root_dir)
    return os.path.join(
        cache_dir,
        f"{safe_simulation}_{field_name}_{wave_name}_{latitude_tag}_"
        f"avgLat_{projection}.nc",
    )


def load_existing_filtered_wave(
    root_dir, simulation, field_name, wave_name, latitude_bounds="-10,10"
):
    """Load a validated wave cache without ever creating or modifying it."""
    path = filtered_wave_cache_path(root_dir, simulation, field_name, wave_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing filtered-wave prerequisite {path}. Run "
            "equatorial_wave_diagnostics.py for this field/wave; downstream "
            "diagnostics will not regenerate wave filters."
        )
    with xr.open_dataset(path) as dataset:
        if "wave" not in dataset:
            raise RuntimeError(f"Filtered-wave cache has no 'wave' variable: {path}")
        wave = dataset["wave"]
        expected = {
            "field": field_name,
            "wave_name": wave_name,
            "latitude_projection": WAVE_LATITUDE_PROJECTION[wave_name],
            "filter_lat_bounds": latitude_bounds,
        }
        changed = [
            key for key, value in expected.items() if wave.attrs.get(key) != value
        ]
        if changed:
            raise RuntimeError(
                f"Filtered-wave cache metadata is stale for {path}: {changed}. "
                "Rebuild it explicitly with equatorial_wave_diagnostics.py; "
                "downstream diagnostics will not overwrite it."
            )
        return wave.load()
