"""Compact native LPO profiles for publication overlays.

The lifecycle owner supplies event identities, rain masks, temperature,
stability, and pressure. Native wind and ascent use matching archive samples.
"""

from __future__ import annotations

import dask
import numpy as np
import pandas as pd
import xarray as xr

from diagnostics.analysis_config import (
    ANALYSIS_END, ANALYSIS_START, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED,
    LATLON_PATTERN, cache_directory, simulation_directory,
)
from diagnostics.cache_support import (
    cache_manifest, cache_manifest_matches, load_netcdf, read_json_file,
    write_json_file, write_netcdf_atomic,
)
from diagnostics.convection.lpo_lifecycle_analysis import (
    LIFECYCLE_ANALYSIS_VERSION, LIFECYCLE_CONFIGURATION_FILENAME,
    LIFECYCLE_SECTION_FILENAME, NEAR_PEAK_LAGS_HOURS,
)
from diagnostics.convection.lpo_lifecycle_sections import (
    DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES, PROPERTY_SAMPLING_DEFINITION,
    SECTION_CACHE_VERSION, _default_window_event_cube, event_equal_statistics,
    sample_centroid_latitude_native_sections,
)
from diagnostics.convection.lpo_tracking import PRODUCT_VERSION
from diagnostics.diagnostic_helpers import (
    align_vertical_velocity_to_reference, load_latlon_dataset, progress, vertical_dim,
)


PROFILE_SCHEMA = "lpo-native-environment-profiles-v1-near-peak"
ASCENT_SCHEMA = "lpo-native-ascent-profiles-v1-matched-environment-events"
PROFILE_FIELDS = {
    "temperature": "unfiltered_temperature",
    "static_stability": "unfiltered_static_stability",
    "zonal_wind": "unfiltered_zonal_wind",
    "pressure_hpa": "native_pressure_hpa",
}


def profile_paths(simulation, family="environment"):
    root = cache_directory("lpt", "publication", "primary", simulation, "standard")
    return (root / f"lpo_native_{family}_profiles.nc",
            root / f"lpo_native_{family}_profiles.manifest.json")


def profile_manifest(simulation, family="environment"):
    root = profile_paths(simulation)[0].parent
    ascent = family == "ascent"
    return cache_manifest(
        "diagnostics/convection/lpo_environment_profiles.py", schema=ASCENT_SCHEMA if ascent else PROFILE_SCHEMA,
        configuration={
            "simulation": simulation, "analysis_start": ANALYSIS_START,
            "analysis_end": ANALYSIS_END, "lags_hours": list(NEAR_PEAK_LAGS_HOURS),
            "longitude_half_width_degrees": DEFAULT_LONGITUDE_HALF_WIDTH_DEGREES,
            "sampling": PROPERTY_SAMPLING_DEFINITION,
            "selection": "primary tropical events with every near-peak sample inside the record",
            "weighting": "cosine-weighted raining columns, then equal times and equal complete events",
            **({"ascent": "native w; adjacent interfaces averaged onto pressure midpoints; match environment event identities"}
               if ascent else {"wind": "native uReconstructZonal, without vertical interpolation"}),
            "bootstrap_seed": BOOTSTRAP_SEED + 70, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        },
        dependencies={"section_version": SECTION_CACHE_VERSION, "tracking_version": PRODUCT_VERSION,
                      "lifecycle_version": LIFECYCLE_ANALYSIS_VERSION},
        source_paths=[root / LIFECYCLE_SECTION_FILENAME, root / LIFECYCLE_CONFIGURATION_FILENAME,
                      *(profile_paths(simulation) if ascent else []),
                      *sorted(simulation_directory(simulation).glob(LATLON_PATTERN))],
    )


def select_near_peak_sections(sections):
    """Keep only primary tropical samples in the five-lag near-peak window."""
    selected = (sections.primary_tropical_sample.astype(bool)
                & sections.reference_in_record_subset.astype(bool)
                & sections.lag_hours.isin(NEAR_PEAK_LAGS_HOURS))
    result = sections.isel(event_step=np.flatnonzero(selected.values))
    if result.sizes["event_step"] == 0:
        raise RuntimeError("No primary tropical near-peak LPO samples")
    keys = pd.DataFrame({"event": result.event_id.values.astype(str), "lag": result.lag_hours.values})
    if keys.duplicated().any():
        raise ValueError("Duplicate LPO event/lag samples")
    return result.assign_coords(event_step=np.arange(result.sizes["event_step"]))


def native_event_profiles(sections, *, require_wind=True):
    """Retain the same complete events for every native environmental field."""
    fields = {name: source for name, source in PROFILE_FIELDS.items()
              if require_wind or name != "zonal_wind"}
    cubes = {name: _default_window_event_cube(sections, source, lags=NEAR_PEAK_LAGS_HOURS)
             for name, source in fields.items()}
    complete = xr.ones_like(cubes["pressure_hpa"].event, dtype=bool)
    for cube in cubes.values():
        complete = complete & np.isfinite(cube).all(("lag_hours", "native_level"))
    if not bool(complete.any()):
        raise RuntimeError("No complete near-peak LPO environmental profiles")
    return xr.Dataset({name: cube.where(complete, drop=True).mean("lag_hours", skipna=False)
                       for name, cube in cubes.items()})


def sample_native_field(field, sections, *, name, units, batch_size=8):
    """Use the saved thermodynamic rain mask for a native midpoint field."""
    if field.sizes[vertical_dim(field)] != sections.sizes["native_level"]:
        raise ValueError("Native field and lifecycle profiles use different model levels")
    weight_name = f"{name}_latitude_weight"
    batches = []
    for start in range(0, sections.sizes["event_step"], batch_size):
        part = sections.isel(event_step=slice(start, start + batch_size))
        steps = pd.DataFrame({name: part[name].values for name in (
            "Time", "precip_weighted_centroid_lon", "precip_weighted_centroid_lat",
        )})
        mask = part.property_rain_mask.assign_coords(event_step=np.arange(len(steps)))
        with dask.config.set(scheduler="synchronous"):
            mean, weight = sample_centroid_latitude_native_sections(
                field, steps, horizontal_mask=mask, return_weights=True,
                relative_latitude=sections.relative_latitude.values,
                relative_longitude=sections.relative_longitude.values,
            )
        batches.append(xr.Dataset({
            name: mean, weight_name: weight,
        }).assign_coords(event_step=part.event_step))
    result = xr.concat(batches, dim="event_step")
    result[name].attrs.update(units=units, longitude_sample_weight=weight_name)
    return result


def sample_native_wind(wind, sections, *, batch_size=8):
    return sample_native_field(wind, sections, name="unfiltered_zonal_wind", units="m s-1", batch_size=batch_size)


def sample_native_ascent(history, sections, *, batch_size=8):
    velocity = align_vertical_velocity_to_reference(history.w, history.pressure)
    return sample_native_field(velocity, sections, name="unfiltered_vertical_velocity",
                               units="m s-1", batch_size=batch_size)


def profile_statistics(events):
    """Retain native means and shared independent-event uncertainty draws."""
    names = [name for name in events.data_vars if name != "pressure_hpa"]
    cube = xr.concat([events[name] for name in names], dim=xr.IndexVariable("field", names))
    statistics = event_equal_statistics(
        cube.transpose("event", "field", "native_level"),
        seed=BOOTSTRAP_SEED + 70, n_bootstrap=BOOTSTRAP_REPLICATES,
    )
    output = xr.Dataset({"pressure_hpa": events.pressure_hpa.mean("event")})
    for name in names:
        output[name] = statistics["mean"].sel(field=name, drop=True)
        output[f"{name}_ci_low"] = statistics.ci_low.sel(field=name, drop=True)
        output[f"{name}_ci_high"] = statistics.ci_high.sel(field=name, drop=True)
        output[f"{name}_events"] = events[name]
        output[name].attrs["units"] = {
            "temperature": "K", "static_stability": "s-2", "zonal_wind": "m s-1",
            "vertical_velocity": "m s-1",
        }[name]
    output["pressure_hpa_events"] = events.pressure_hpa
    output.pressure_hpa.attrs["units"] = "hPa"
    output.attrs.update(
        schema=ASCENT_SCHEMA if "vertical_velocity" in events else PROFILE_SCHEMA,
        event_count=events.sizes["event"],
        lag_hours=str(list(NEAR_PEAK_LAGS_HOURS)),
        event_weighting="one equal weight per complete primary tropical event",
        sampling=PROPERTY_SAMPLING_DEFINITION,
        vertical_coordinate="native MPAS levels; own event-mean pressure; no vertical interpolation",
    )
    return output


def read_near_peak_sections(simulation, *, thermodynamics=True):
    """Read current production samples without opening unrelated cached fields."""
    root = profile_paths(simulation)[0].parent
    configuration = read_json_file(root / LIFECYCLE_CONFIGURATION_FILENAME)
    with xr.open_dataset(root / LIFECYCLE_SECTION_FILENAME) as source:
        expected = {"cache_version": SECTION_CACHE_VERSION, "source_simulation": simulation,
                    "lifecycle_analysis_version": LIFECYCLE_ANALYSIS_VERSION,
                    "tracking_product_version": PRODUCT_VERSION, "test_subset": 0}
        if any(source.attrs.get(key) != value for key, value in expected.items()):
            raise RuntimeError(f"Missing or stale production lifecycle sections for {simulation}")
        if (not isinstance(configuration, dict) or configuration.get("test_product", False)
                or configuration.get("lifecycle_analysis_version") != LIFECYCLE_ANALYSIS_VERSION
                or configuration.get("calculation_source_sha256")
                != source.attrs.get("calculation_source_sha256")):
            raise RuntimeError(f"Lifecycle configuration differs from sections for {simulation}")
        names = ["event_id", "Time", "lag_hours", "primary_tropical_sample", "reference_in_record_subset",
                 "precip_weighted_centroid_lon", "precip_weighted_centroid_lat", "property_rain_mask"]
        for name in (PROFILE_FIELDS.values() if thermodynamics else []):
            if name == "unfiltered_zonal_wind":
                continue
            names.extend([name, source[name].attrs["longitude_sample_weight"]])
        selected = select_near_peak_sections(source[names]).load()
    if (bool((selected.Time < np.datetime64(ANALYSIS_START)).any())
            or bool((selected.Time > np.datetime64(ANALYSIS_END)).any())):
        raise ValueError("LPO profiles extend beyond the summary analysis period")
    return selected


def build_profiles(simulation):
    selected = read_near_peak_sections(simulation)
    complete = native_event_profiles(selected, require_wind=False).event.values
    selected = selected.isel(event_step=np.flatnonzero(selected.event_id.isin(complete).values))
    selected = selected.assign_coords(event_step=np.arange(selected.sizes["event_step"]))
    progress(f"{simulation}: sampling native wind for {len(complete)} complete near-peak LPOs")
    history = load_latlon_dataset(
        str(simulation_directory(simulation)), LATLON_PATTERN,
        chunks={"Time": 1, "nVertLevels": 25, "latitude": 120, "longitude": 240},
        chunk_after_open=True,
    )
    try:
        wind = sample_native_wind(history.uReconstructZonal, selected)
    finally:
        history.close()
    return profile_statistics(native_event_profiles(xr.merge([selected, wind], join="exact")))


def build_ascent_profiles(simulation, environment):
    """Match the environment overlay's exact event identities and pressure."""
    selected = read_near_peak_sections(simulation, thermodynamics=False)
    selected = selected.isel(event_step=np.flatnonzero(selected.event_id.isin(environment.event.values).values))
    selected = selected.assign_coords(event_step=np.arange(selected.sizes["event_step"]),
                                      native_level=environment.native_level)
    progress(f"{simulation}: sampling native ascent for {environment.sizes['event']} matched near-peak LPOs")
    history = load_latlon_dataset(
        str(simulation_directory(simulation)), LATLON_PATTERN,
        chunks={"Time": 1, "nVertLevels": 25, "nVertLevelsP1": 26, "latitude": 120, "longitude": 240},
        chunk_after_open=True,
    )
    try:
        sampled = sample_native_ascent(history, selected)
    finally:
        history.close()
    velocity = _default_window_event_cube(
        xr.merge([selected, sampled], join="exact"), "unfiltered_vertical_velocity", lags=NEAR_PEAK_LAGS_HOURS,
    ).sel(event=environment.event)
    if not bool(np.isfinite(velocity).all()):
        raise RuntimeError("Native ascent lacks samples from the matched environment event pool")
    return profile_statistics(xr.Dataset({
        "vertical_velocity": velocity.mean("lag_hours", skipna=False),
        "pressure_hpa": environment.pressure_hpa_events,
    }))


def prepare_ascent_profiles(simulation, *, figures_only=False, rebuild=False):
    environment = prepare_profiles(simulation, figures_only=figures_only)
    path, manifest_path = profile_paths(simulation, "ascent")
    manifest = profile_manifest(simulation, "ascent")
    if not rebuild and path.is_file() and cache_manifest_matches(read_json_file(manifest_path), manifest):
        return load_netcdf(path)
    if figures_only:
        raise RuntimeError(f"Missing or stale LPO ascent product {path}; run the summary without --figures-only")
    profiles = build_ascent_profiles(simulation, environment)
    profiles.attrs["source_simulation"] = simulation
    write_netcdf_atomic(profiles, path)
    write_json_file(manifest_path, manifest)
    return profiles


def prepare_profiles(simulation, *, figures_only=False, rebuild=False):
    """Reuse compact native event products without rerunning LPO tracking."""
    path, manifest_path = profile_paths(simulation)
    manifest = profile_manifest(simulation)
    if not rebuild and path.is_file() and cache_manifest_matches(read_json_file(manifest_path), manifest):
        return load_netcdf(path)
    if figures_only:
        raise RuntimeError(f"Missing or stale LPO profile product {path}; run the summary without --figures-only")
    profiles = build_profiles(simulation)
    profiles.attrs["source_simulation"] = simulation
    write_netcdf_atomic(profiles, path)
    write_json_file(manifest_path, manifest)
    return profiles
