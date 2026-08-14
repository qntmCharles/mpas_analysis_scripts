"""Read-only access to products from ``equatorial_wave_diagnostics.py``."""

import os

import numpy as np
import xarray as xr

from diagnostics.diagnostic_helpers import diagnostic_cache_dir
from diagnostics.waves.equatorial_wave_config import (
    FILTERED_WAVE_PRODUCT_VERSION,
    WAVE_BOXES,
    WAVE_CONFIG_VERSION,
    LPO_LIFECYCLE_FILTERS,
    WAVE_LATITUDE_PROJECTION,
)
from diagnostics.diagnostic_helpers import jsonable_value
import json


LPO_WAVE_SECTION_CACHE_VERSION = (
    "lpo-wave-sections-v12-amplitude-gt-one"
)
LPO_WAVE_SECTION_FILENAME_TEMPLATE = "lpo_wave_sections_{simulation}.nc"
KELVIN_DISPERSION_FILENAME = "kelvin_reference_energy_dispersion.nc"
WAVE_RESPONSE_METRICS_FILENAME = "wave_response_efficiency_metrics.csv"
WAVE_RESPONSE_CONTRASTS_FILENAME = "wave_response_efficiency_contrasts.csv"


def _wave_cache_root(*, test=False):
    return diagnostic_cache_dir(
        "waves",
        *(("testing",) if test else ()),
    )


def wave_product_directory(*, test=False):
    """Return the fixed machine-readable wave-product directory."""

    return os.path.join(_wave_cache_root(test=test), "products")


def wave_response_metrics_product_path(*, test=False):
    """Return the authoritative Wave Figure-11 response-metric table."""

    return os.path.join(
        wave_product_directory(test=test),
        WAVE_RESPONSE_METRICS_FILENAME,
    )


def wave_response_contrasts_product_path(*, test=False):
    """Return the authoritative Wave Figure-11 direct-contrast table."""

    return os.path.join(
        wave_product_directory(test=test),
        WAVE_RESPONSE_CONTRASTS_FILENAME,
    )


def filtered_wave_cache_path(
    simulation,
    field_name,
    wave_name,
    *,
    test=False,
):
    projection = WAVE_LATITUDE_PROJECTION[wave_name]
    safe_simulation = simulation.replace(" ", "_")
    cache_dir = _wave_cache_root(test=test)
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


def lpo_wave_section_cache_path(simulation, *, test=False):
    """Return the compact LPO-following wave-section cache path."""

    safe_simulation = simulation.replace(" ", "_")
    return os.path.join(
        wave_product_directory(test=test),
        LPO_WAVE_SECTION_FILENAME_TEMPLATE.format(
            simulation=safe_simulation
        ),
    )


def load_existing_lpo_wave_sections(
    simulation,
    *,
    tracking_product_version,
    test=False,
):
    """Load validated, prefiltered LPO-section values without filtering."""

    path = lpo_wave_section_cache_path(simulation, test=test)
    if not os.path.exists(path):
        command = (
            "equatorial_wave_diagnostics.py --test --figures 14"
            if test
            else "equatorial_wave_diagnostics.py --figures 14"
        )
        raise FileNotFoundError(
            f"Missing compact LPO wave-section prerequisite {path}. Run "
            f"{command}; lifecycle analysis will not run wave filters."
        )
    with xr.open_dataset(path) as dataset:
        expected = {
            "cache_version": LPO_WAVE_SECTION_CACHE_VERSION,
            "wave_config_version": WAVE_CONFIG_VERSION,
            "tracking_product_version": tracking_product_version,
            "source_simulation": simulation,
            "test_subset": int(test),
        }
        changed = [
            key
            for key, value in expected.items()
            if dataset.attrs.get(key) != value
        ]
        if changed:
            raise RuntimeError(
                f"Compact LPO wave-section metadata is stale for {path}: "
                f"{changed}. Rebuild it explicitly with "
                "equatorial_wave_diagnostics.py --figures 14."
            )
        saved_bands = set(dataset.wave_band.values.astype(str))
        missing_bands = sorted(set(LPO_LIFECYCLE_FILTERS) - saved_bands)
        unknown_bands = sorted(saved_bands - set(WAVE_BOXES))
        if missing_bands or unknown_bands:
            raise RuntimeError(
                f"Compact LPO wave-section cache has missing bands "
                f"{missing_bands} and unknown bands {unknown_bands}"
            )
        required_variables = {
            "filtered_precipitation",
            "filtered_precipitable_water",
            "filtered_olr",
            "filtered_temperature",
            "filtered_upward_pressure_velocity",
            "filtered_specific_humidity",
            "filtered_relative_humidity",
            "filtered_zonal_wind",
            "filtered_static_stability",
            "filtered_native_pressure_hpa",
            "wave_phase_angle_degrees",
            "wave_phase_number",
            "wave_normalized_amplitude",
            "wave_strong_amplitude",
            "random_wave_phase_angle_degrees",
            "random_wave_phase_number",
            "random_wave_normalized_amplitude",
            "random_wave_strong_amplitude",
            "precipitation_wave_rms",
            "peak_event_id",
            "ray_filtered_temperature",
        }
        missing_variables = sorted(
            required_variables - set(dataset.data_vars)
        )
        required_dimensions = {
            "event_step",
            "relative_latitude",
            "relative_longitude",
            "native_level",
            "wave_band",
            "peak_event",
            "ray_height_m",
            "ray_relative_longitude",
            "ray_wave_band",
        }
        missing_dimensions = sorted(
            required_dimensions - set(dataset.sizes)
        )
        if missing_variables or missing_dimensions:
            raise RuntimeError(
                f"Compact LPO wave-section cache {path} lacks variables "
                f"{missing_variables} or dimensions {missing_dimensions}"
            )
        return dataset.load()


def kelvin_dispersion_product_path(*, test=False):
    """Return the saved Figure-13 Kelvin reference-dispersion product."""

    return os.path.join(
        wave_product_directory(test=test),
        KELVIN_DISPERSION_FILENAME,
    )


def load_existing_kelvin_dispersion(*, test=False):
    """Load the validated Figure-13 Kelvin reference without recomputing it."""

    path = kelvin_dispersion_product_path(test=test)
    if not os.path.exists(path):
        command = (
            "equatorial_wave_diagnostics.py --test --figures 13"
            if test
            else "equatorial_wave_diagnostics.py --figures 13"
        )
        raise FileNotFoundError(
            f"Missing Kelvin dispersion prerequisite {path}. Run {command}; "
            "lifecycle analysis will not introduce another dispersion "
            "definition."
        )
    with xr.open_dataset(path) as dataset:
        required = {
            "background_zonal_wind",
            "buoyancy_frequency",
            "static_stability",
            "background_pressure_hpa",
            "horizontal_group_velocity",
            "vertical_group_velocity",
        }
        missing = sorted(required - set(dataset.data_vars))
        if missing:
            raise RuntimeError(
                f"Kelvin dispersion product {path} lacks {missing}"
            )
        if dataset.attrs.get("wave_config_version", WAVE_CONFIG_VERSION) != WAVE_CONFIG_VERSION:
            raise RuntimeError(
                f"Kelvin dispersion product {path} has a stale wave "
                "configuration"
            )
        for attribute in ("primary_k_m-1", "primary_m_m-1"):
            value = dataset.attrs.get(attribute)
            if value is None or not np.isfinite(float(value)):
                raise RuntimeError(
                    f"Kelvin dispersion product {path} lacks finite "
                    f"{attribute!r} metadata"
                )
        return dataset.load()
