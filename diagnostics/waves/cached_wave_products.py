"""Read-only access to filtered products from equatorial_wave_diagnostics.py."""

import os

import xarray as xr

from diagnostics.diagnostic_helpers import diagnostic_cache_dir
from diagnostics.waves.equatorial_wave_config import (
    FILTERED_WAVE_PRODUCT_VERSION,
    WAVE_BOXES,
    WAVE_CONFIG_VERSION,
    WAVE_LATITUDE_PROJECTION,
)
from diagnostics.diagnostic_helpers import jsonable_value
import json


def filtered_wave_cache_path(
    simulation,
    field_name,
    wave_name,
    *,
    test=False,
):
    projection = WAVE_LATITUDE_PROJECTION[wave_name]
    safe_simulation = simulation.replace(" ", "_")
    cache_dir = diagnostic_cache_dir(
        "wave_cache",
        *(("testing",) if test else ()),
    )
    return os.path.join(
        cache_dir,
        f"{safe_simulation}_{field_name}_{wave_name}_m10_10_"
        f"avgLat_{projection}.nc",
    )


def load_existing_filtered_wave(
    simulation,
    field_name,
    wave_name,
    *,
    test=False,
):
    """Load a validated wave cache without ever creating or modifying it."""
    path = filtered_wave_cache_path(
        simulation,
        field_name,
        wave_name,
        test=test,
    )
    if not os.path.exists(path):
        command = (
            "equatorial_wave_diagnostics.py --test"
            if test
            else "equatorial_wave_diagnostics.py"
        )
        raise FileNotFoundError(
            f"Missing filtered-wave prerequisite {path}. Run {command} for "
            "this field/wave; downstream diagnostics will not regenerate "
            "wave filters."
        )
    with xr.open_dataset(path) as dataset:
        if "wave" not in dataset:
            raise RuntimeError(f"Filtered-wave cache has no 'wave' variable: {path}")
        wave = dataset["wave"]
        expected = {
            "field": field_name,
            "wave_name": wave_name,
            "cache_version": FILTERED_WAVE_PRODUCT_VERSION,
            "latitude_projection": WAVE_LATITUDE_PROJECTION[wave_name],
            "filter_lat_bounds": "-10,10",
            "wave_config_version": WAVE_CONFIG_VERSION,
            "wave_box": json.dumps(
                jsonable_value(WAVE_BOXES[wave_name]), sort_keys=True
            ),
        }
        if test:
            expected["test_subset"] = 1
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
