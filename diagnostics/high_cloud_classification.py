#!/usr/bin/env python
# coding: utf-8

"""Shared high-cloud precipitation-regime classification for 0.5 degree data.

``cldfrac`` and model-level pressure define the parent high-cloud geometry.
The pressure of the highest cloudy model level is compared with the standard
ISCCP high-cloud boundary (440 hPa), avoiding a fixed geometric-height cutoff.
Precipitating high cloud is partitioned using three coarse-grid indicators of
cellular convection: heavy total rain, strong resolved ascent within cloud,
and dominance of native MPAS parameterized-convective rain (``rainc``). The
last is used only as a proxy for unresolved horizontal cellularity, not as a
physical convective/stratiform label by itself. Requiring two indicators avoids
allowing any one resolution-sensitive field to determine the class.

The three science classes are mutually exclusive and exhaustive wherever the
surface precipitation components are valid:

* convective precipitating high cloud;
* stratiform precipitating high cloud;
* non-precipitating high cloud.

The unclassified mask is reserved for high-cloud columns with missing or
invalid precipitation-component data. Vertical velocity is the only 3-D field
used beyond cloud fraction and pressure. Hydrometeors remain optional
validation diagnostics, and water vapour is environmental context only.
"""

from dataclasses import asdict, dataclass, replace
import numbers
import warnings

import numpy as np
import xarray as xr

from diagnostic_helpers import finite_or_nan, vertical_dim


HYDROMETEOR_ALIASES = {
    "ice": ("qi", "qice", "q_i", "ice", "cloud_ice"),
    "snow": ("qs", "qsnow", "q_s", "snow"),
    "graupel": ("qg", "qgraup", "qgraupel", "q_g", "graupel"),
    "cloud_water": ("qc", "qcloud", "q_cloud", "q_c", "cloud_water"),
    "rain": ("qr", "qrain", "q_r", "rain_water"),
}

HYDROMETEOR_NEEDED_VARS = sorted(
    {name for aliases in HYDROMETEOR_ALIASES.values() for name in aliases}
)
CLASSIFICATION_SUPPORT_NEEDED_VARS = ["rho", "w", "zgrid"]
CLASSIFICATION_CACHE_VERSION = "high-cloud-precip-regime-0p5deg-v7"


@dataclass(frozen=True)
class HighCloudThresholds:
    """Thresholds for the resolution-appropriate high-cloud regime split."""

    cf_thresh: float = 0.05
    high_cloud_top_pressure_hpa: float = 440.0
    surface_precip_min_mm_day: float = 1.0
    heavy_precip_mm_day: float = 10.0
    strong_updraft_m_s: float = 0.10
    convective_fraction_threshold: float = 0.50
    convective_evidence_required: int = 2


def thresholds_with(**updates):
    return replace(HighCloudThresholds(), **updates)


def classification_cache_metadata(thresholds=None, extra=None):
    thresholds = thresholds or HighCloudThresholds()
    out = {"cache_version": CLASSIFICATION_CACHE_VERSION}
    out.update(asdict(thresholds))
    if extra:
        out.update(extra)
    return out


def netcdf_safe_attr_value(value):
    if isinstance(value, (bool, np.bool_)):
        return int(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def netcdf_safe_attrs(attrs):
    return {key: netcdf_safe_attr_value(value) for key, value in attrs.items()}


def cached_classification_is_current(cached, thresholds=None):
    expected = classification_cache_metadata(thresholds)
    for key, value in expected.items():
        cached_value = cached.attrs.get(key)
        if isinstance(value, float):
            if cached_value is None or not np.isclose(float(cached_value), value):
                return False
        elif isinstance(value, bool):
            if isinstance(cached_value, str):
                cached_bool = cached_value.strip().lower() in {
                    "1", "true", "t", "yes", "y", "on"
                }
            else:
                cached_bool = bool(cached_value)
            if cached_bool != value:
                return False
        elif cached_value != value:
            return False
    return True


def standardize_model_level_field(da, reference=None, name=None):
    out = finite_or_nan(da)
    zdim = vertical_dim(out)
    if zdim != "z":
        out = out.rename({zdim: "z"})
    if reference is not None:
        if out.sizes.get("z") != reference.sizes.get("z"):
            raise ValueError(
                f"{name or da.name or 'field'} has {out.sizes.get('z')} vertical levels, "
                f"but cldfrac has {reference.sizes.get('z')}"
            )
        out = out.assign_coords(z=reference.z)
    return out


def standardize_cloud_fraction(cldfrac):
    cf = standardize_model_level_field(cldfrac, name="cldfrac")
    if "z" not in cf.coords:
        raise ValueError("cldfrac has no geometric-height z coordinate")
    zvals = np.asarray(cf.z.values, dtype=float)
    if zvals.size != cf.sizes["z"] or not np.all(np.isfinite(zvals)):
        raise ValueError("cldfrac z coordinate is not usable")
    if np.nanmax(np.abs(zvals)) < 100.0:
        cf = cf.assign_coords(z=cf.z * 1000.0)
    return cf


def z_km_from_cldfrac(cf):
    zvals = np.asarray(cf.z.values, dtype=float)
    zvals = zvals / 1000.0 if np.nanmax(np.abs(zvals)) > 100.0 else zvals
    return xr.DataArray(zvals, dims=("z",), coords={"z": cf.z}, name="z_km")


def standardize_pressure_hpa(pressure, reference):
    """Return model-level pressure aligned to ``cldfrac`` in hPa."""
    out = standardize_model_level_field(
        pressure, reference=reference, name="pressure"
    )
    finite = np.asarray(out.isel({dim: 0 for dim in out.dims if dim != "z"}).values)
    finite = finite[np.isfinite(finite)]
    if finite.size == 0:
        raise ValueError("pressure has no finite values")
    if np.nanmedian(finite) > 2000.0:
        out = out / 100.0
    out.name = "pressure_hpa"
    out.attrs.update(units="hPa", positive="down")
    return out


def first_dataset_var(ds, aliases):
    return next((name for name in aliases if name in ds.data_vars), None)


def hydrometeor_fields_from_dataset(ds, reference=None, warn_missing=False, strict=False):
    fields = {}
    sources = {}
    for key, aliases in HYDROMETEOR_ALIASES.items():
        source = first_dataset_var(ds, aliases)
        if source is None:
            message = f"Hydrometeor {key!r} is unavailable"
            if strict:
                raise ValueError(message)
            if warn_missing:
                warnings.warn(message + "; treating it as zero.")
            if reference is not None:
                fields[key] = xr.zeros_like(reference)
            sources[key] = ""
            continue
        fields[key] = standardize_model_level_field(
            ds[source], reference=reference, name=source
        )
        sources[key] = source
    return fields, sources


def midlevel_field_from_interfaces(da, reference, name):
    out = finite_or_nan(da)
    zdim = vertical_dim(out)
    if out.sizes[zdim] != reference.sizes["z"] + 1:
        return standardize_model_level_field(out, reference=reference, name=name)
    lower = out.isel({zdim: slice(0, -1)}).rename({zdim: "z"})
    upper = out.isel({zdim: slice(1, None)}).rename({zdim: "z"})
    lower = lower.assign_coords(z=reference.z)
    upper = upper.assign_coords(z=reference.z)
    result = 0.5 * (lower + upper)
    result.name = name
    return result


def layer_thickness_from_dataset(ds, reference):
    source = first_dataset_var(ds, ("zgrid", "z_interface", "height_interface"))
    if source is None:
        raise ValueError("zgrid interface heights are required for condensate paths")
    interfaces = finite_or_nan(ds[source])
    zdim = vertical_dim(interfaces)
    if interfaces.sizes[zdim] != reference.sizes["z"] + 1:
        raise ValueError(
            f"{source} has {interfaces.sizes[zdim]} interfaces; expected "
            f"{reference.sizes['z'] + 1}"
        )
    lower = interfaces.isel({zdim: slice(0, -1)}).rename({zdim: "z"})
    upper = interfaces.isel({zdim: slice(1, None)}).rename({zdim: "z"})
    lower = lower.assign_coords(z=reference.z)
    upper = upper.assign_coords(z=reference.z)
    thickness = np.abs(upper - lower)
    thickness.name = "layer_thickness"
    thickness.attrs["units"] = "m"
    return thickness


def accumulation_rate(accum, dt_hours=6.0):
    accum = finite_or_nan(accum)
    if "Time" not in accum.dims or accum.sizes["Time"] < 2:
        raise ValueError("Accumulated precipitation needs at least two times")
    if np.issubdtype(accum.Time.dtype, np.datetime64):
        dt_days = accum.Time.diff("Time") / np.timedelta64(1, "D")
    elif issubclass(accum.Time.dtype.type, numbers.Integral):
        dt_days = dt_hours / 24.0
    else:
        values = np.asarray(accum.Time.values, dtype=float)
        delta = np.diff(values)
        units = str(accum.Time.attrs.get("units", "")).lower()
        if "hour" in units:
            delta = delta / 24.0
        elif "second" in units:
            delta = delta / 86400.0
        elif "day" not in units:
            delta = np.full(delta.shape, dt_hours / 24.0)
        dt_days = xr.DataArray(delta, dims=("Time",), coords={"Time": accum.Time.isel(Time=slice(1, None))})
    rate = accum.diff("Time") / dt_days
    return rate.clip(min=0.0)


def precip_component_rate_from_dataset(ds, component, dt_hours=6.0):
    specs = {
        "convective": (("rainc",), ("i_rainc",)),
        "grid_scale": (("rainnc",), ("i_rainnc",)),
    }
    accum_aliases, counter_aliases = specs[component]
    accum_name = first_dataset_var(ds, accum_aliases)
    counter_name = first_dataset_var(ds, counter_aliases)
    if accum_name is None or counter_name is None:
        raise ValueError(
            f"MPAS {component} precipitation accumulation/counter is unavailable"
        )
    accum = finite_or_nan(ds[accum_name]) + finite_or_nan(ds[counter_name]) * 100.0
    rate = accumulation_rate(accum, dt_hours=dt_hours)
    rate.name = f"{component}_precip_rate"
    rate.attrs["units"] = "mm day-1"
    return rate


def condensate_path(q, density, layer_thickness, name):
    path = (q * density * layer_thickness).sum(dim="z", skipna=True)
    path.name = name
    path.attrs["units"] = "kg m-2"
    return path


def classify_high_cloud_3d(
    cldfrac,
    pressure=None,
    precip=None,
    convective_precip=None,
    grid_scale_precip=None,
    hydrometeors=None,
    vertical_velocity=None,
    density=None,
    layer_thickness=None,
    thresholds=None,
    hydrometeor_sources=None,
    include_process_diagnostics=False,
):
    """Return an exhaustive coarse-grid precipitation-regime split.

    Vertical velocity supplies one of three deliberately coarse indicators of
    cellular convection. Hydrometeors remain diagnostic only because at
    0.5 degree resolution they describe grid-box-mean cloud physics rather
    than resolved convective-core properties.
    """

    thresholds = thresholds or HighCloudThresholds()
    cf = standardize_cloud_fraction(cldfrac)
    if pressure is None:
        raise ValueError("Model-level pressure is required for high-cloud classification")
    pressure_hpa = standardize_pressure_hpa(pressure, cf)
    if convective_precip is None or grid_scale_precip is None:
        raise ValueError(
            "Both convective_precip and grid_scale_precip rates are required"
        )
    if vertical_velocity is None:
        raise ValueError("Model-level vertical velocity is required for classification")
    convective_precip = finite_or_nan(convective_precip).clip(min=0.0)
    grid_scale_precip = finite_or_nan(grid_scale_precip).clip(min=0.0)
    vertical_velocity = standardize_model_level_field(
        vertical_velocity, reference=cf, name="vertical_velocity"
    )
    if precip is None:
        precip = convective_precip + grid_scale_precip
    precip = finite_or_nan(precip).clip(min=0.0)

    (
        cf,
        pressure_hpa,
        precip,
        convective_precip,
        grid_scale_precip,
        vertical_velocity,
    ) = xr.align(
        cf,
        pressure_hpa,
        precip,
        convective_precip,
        grid_scale_precip,
        vertical_velocity,
        join="inner",
    )

    z_km = z_km_from_cldfrac(cf)
    cloudy = (cf >= thresholds.cf_thresh).fillna(False)
    high_layer = pressure_hpa <= thresholds.high_cloud_top_pressure_hpa
    cloud_top_pressure = pressure_hpa.where(cloudy).min(dim="z", skipna=True)
    cloud_top_pressure.name = "cloud_top_pressure"
    cloud_top_pressure.attrs.update(units="hPa", positive="down")
    high_cloud_fraction = cf.where(high_layer).mean(dim="z", skipna=True)
    high_cloud_fraction.name = "high_cloud_fraction"
    high_cloud_mask = (
        cloud_top_pressure <= thresholds.high_cloud_top_pressure_hpa
    ).fillna(False)
    high_cloud_mask.name = "high_cloud_mask"

    cloud_top_height = z_km.where(cloudy).max(dim="z", skipna=True)
    cloud_base_height = z_km.where(cloudy).min(dim="z", skipna=True)
    cloud_depth = (cloud_top_height - cloud_base_height).where(
        cloud_top_height.notnull()
    )
    for da, name in (
        (cloud_top_height, "cloud_top_height"),
        (cloud_base_height, "cloud_base_height"),
        (cloud_depth, "cloud_depth"),
    ):
        da.name = name
        da.attrs["units"] = "km"

    component_total = convective_precip + grid_scale_precip
    component_total.name = "surface_precip_rate"
    component_total.attrs["units"] = "mm day-1"
    # Mask the denominator before division so lazy evaluation does not still
    # evaluate a divide-by-zero branch inside xr.where.
    convective_fraction = (
        convective_precip / component_total.where(component_total > 0.0)
    ).fillna(0.0).clip(min=0.0, max=1.0)
    convective_fraction.name = "convective_precip_fraction"
    source_dominance = (
        (convective_precip - grid_scale_precip)
        / component_total.where(component_total > 0.0)
    ).fillna(0.0).clip(min=-1.0, max=1.0)
    source_dominance.name = "precip_source_dominance_index"
    source_dominance.attrs.update(
        long_name="(parameterized convective - resolved grid-scale) / total precipitation",
        interpretation="positive values are parameterized-convective dominated",
    )

    valid_cloudy_updraft = (cloudy & vertical_velocity.notnull()).any(dim="z")
    column_max_updraft = vertical_velocity.where(
        cloudy & vertical_velocity.notnull(), other=-np.inf
    ).max(dim="z").where(valid_cloudy_updraft)
    column_max_updraft.name = "column_max_updraft"
    column_max_updraft.attrs.update(
        units="m s-1",
        long_name="maximum resolved vertical velocity within cloudy model levels",
        resolution_caveat="Coarse-grid indicator; not a resolved convective-core velocity",
    )

    valid_precip = convective_precip.notnull() & grid_scale_precip.notnull()
    precipitating = component_total >= thresholds.surface_precip_min_mm_day
    heavy_precip_evidence = (
        component_total >= thresholds.heavy_precip_mm_day
    ).fillna(False)
    strong_updraft_evidence = (
        column_max_updraft >= thresholds.strong_updraft_m_s
    ).fillna(False)
    parameterized_source_evidence = (
        convective_fraction >= thresholds.convective_fraction_threshold
    ).fillna(False)
    convective_evidence_count = (
        heavy_precip_evidence.astype("int8")
        + strong_updraft_evidence.astype("int8")
        + parameterized_source_evidence.astype("int8")
    )
    convective_evidence_count.name = "convective_evidence_count"
    convective_evidence_count.attrs.update(
        long_name="number of coarse-grid convective indicators satisfied",
        indicators=(
            f"rainc + rainnc >= {thresholds.heavy_precip_mm_day:g} mm day-1; "
            f"max(w) over levels with cldfrac >= {thresholds.cf_thresh:g} >= "
            f"{thresholds.strong_updraft_m_s:g} m s-1; rainc / (rainc + rainnc) >= "
            f"{thresholds.convective_fraction_threshold:g}"
        ),
        interpretation=(
            f"convective when count >= {thresholds.convective_evidence_required}; "
            "otherwise precipitating high cloud is stratiform"
        ),
    )
    for evidence, name, meaning in (
        (
            heavy_precip_evidence,
            "heavy_precipitation_evidence",
            f"surface precipitation >= {thresholds.heavy_precip_mm_day:g} mm day-1",
        ),
        (
            strong_updraft_evidence,
            "strong_updraft_evidence",
            f"max(w) over levels with cldfrac >= {thresholds.cf_thresh:g} >= "
            f"{thresholds.strong_updraft_m_s:g} m s-1",
        ),
        (
            parameterized_source_evidence,
            "parameterized_source_evidence",
            f"rainc fraction >= {thresholds.convective_fraction_threshold:g}",
        ),
    ):
        evidence.name = name
        evidence.attrs.update(
            logical_type="bool",
            physical_meaning=meaning,
            role="one of three indicators; never class-defining alone",
        )

    convective = (
        high_cloud_mask
        & valid_precip
        & precipitating
        & (convective_evidence_count >= thresholds.convective_evidence_required)
    ).fillna(False)
    stratiform = (
        high_cloud_mask & valid_precip & precipitating & ~convective
    ).fillna(False)
    nonprecipitating = (
        high_cloud_mask & valid_precip & ~precipitating
    ).fillna(False)
    unclassified = (high_cloud_mask & ~valid_precip).fillna(False)

    convective.name = "convective_precipitating_high_cloud_mask"
    stratiform.name = "stratiform_precipitating_high_cloud_mask"
    nonprecipitating.name = "nonprecipitating_high_cloud_mask"
    unclassified.name = "unclassified_high_cloud_mask"
    precipitating.name = "precipitating_high_cloud_mask"
    regime_code = xr.zeros_like(high_cloud_mask, dtype="int8")
    regime_code = regime_code.where(~convective, 1)
    regime_code = regime_code.where(~stratiform, 2)
    regime_code = regime_code.where(~nonprecipitating, 3)
    regime_code = regime_code.where(~unclassified, 4)
    regime_code.name = "high_cloud_regime_code"
    regime_code.attrs.update(
        flag_values="0,1,2,3,4",
        flag_meanings=(
            "not_high_cloud convective_precipitating stratiform_precipitating "
            "nonprecipitating unclassified"
        ),
    )

    closure_error = precip - component_total
    closure_error.name = "surface_precip_component_closure_error"
    closure_error.attrs["units"] = "mm day-1"

    out = {
        "cloud_top_pressure": cloud_top_pressure,
        "cloud_top_height": cloud_top_height,
        "cloud_base_height": cloud_base_height,
        "cloud_depth": cloud_depth,
        "column_max_updraft": column_max_updraft,
        "high_cloud_fraction": high_cloud_fraction,
        "high_cloud_mask": high_cloud_mask,
        "precipitating_high_cloud_mask": high_cloud_mask & precipitating,
        "convective_precipitating_high_cloud_mask": convective,
        "stratiform_precipitating_high_cloud_mask": stratiform,
        "nonprecipitating_high_cloud_mask": nonprecipitating,
        "unclassified_high_cloud_mask": unclassified,
        "high_cloud_regime_code": regime_code,
        "surface_precip_rate": component_total,
        "convective_precip_rate": convective_precip,
        "grid_scale_precip_rate": grid_scale_precip,
        "convective_precip_fraction": convective_fraction,
        "precip_source_dominance_index": source_dominance,
        "heavy_precipitation_evidence": heavy_precip_evidence,
        "strong_updraft_evidence": strong_updraft_evidence,
        "parameterized_source_evidence": parameterized_source_evidence,
        "convective_evidence_count": convective_evidence_count,
        "surface_precip_component_closure_error": closure_error,
    }

    if include_process_diagnostics:
        hydrometeors = dict(hydrometeors or {})
        missing_hydrometeors = [
            key for key in HYDROMETEOR_ALIASES if key not in hydrometeors
        ]
        if missing_hydrometeors:
            raise ValueError(f"Missing hydrometeor fields: {missing_hydrometeors}")
        if vertical_velocity is None or density is None or layer_thickness is None:
            raise ValueError(
                "vertical velocity, density, and layer thickness are required "
                "for process diagnostics"
            )

        hydrometeors = {
            key: standardize_model_level_field(value, reference=cf, name=key)
            for key, value in hydrometeors.items()
        }
        density = standardize_model_level_field(
            density, reference=cf, name="density"
        )
        layer_thickness = standardize_model_level_field(
            layer_thickness, reference=cf, name="layer_thickness"
        )
        process_inputs = [
            cf,
            vertical_velocity,
            density,
            layer_thickness,
        ] + [hydrometeors[key] for key in HYDROMETEOR_ALIASES]
        process_aligned = xr.align(*process_inputs, join="inner")
        process_cf, vertical_velocity, density, layer_thickness = process_aligned[:4]
        hydrometeors = dict(zip(HYDROMETEOR_ALIASES, process_aligned[4:]))

        ice_phase = (
            hydrometeors["ice"]
            + hydrometeors["snow"]
            + hydrometeors["graupel"]
        )
        liquid_phase = hydrometeors["cloud_water"] + hydrometeors["rain"]
        precipitating_condensate = (
            hydrometeors["rain"]
            + hydrometeors["snow"]
            + hydrometeors["graupel"]
        )
        total_condensate = liquid_phase + ice_phase
        process_z_km = z_km_from_cldfrac(process_cf)
        process_pressure = pressure_hpa.sel(Time=process_cf.Time)
        upper_layer = process_pressure <= thresholds.high_cloud_top_pressure_hpa
        upper_liquid = liquid_phase.where(upper_layer, 0.0)
        upper_ice = ice_phase.where(upper_layer, 0.0)
        upper_total = total_condensate.where(upper_layer, 0.0)
        upper_liquid_path = condensate_path(
            upper_liquid,
            density,
            layer_thickness,
            "upper_liquid_phase_condensate_path",
        )
        upper_ice_path = condensate_path(
            upper_ice,
            density,
            layer_thickness,
            "upper_ice_phase_condensate_path",
        )
        upper_total_path = condensate_path(
            upper_total,
            density,
            layer_thickness,
            "upper_total_condensate_path",
        )
        upper_ice_fraction = (
            upper_ice_path / upper_total_path.where(upper_total_path > 0.0)
        ).clip(min=0.0, max=1.0)
        upper_ice_fraction.name = "upper_condensate_ice_fraction"
        upper_condensate_weighted_height = (
            (upper_total * density * layer_thickness * process_z_km).sum(
                dim="z", skipna=True
            )
            / upper_total_path.where(upper_total_path > 0.0)
        )
        upper_condensate_weighted_height.name = "upper_condensate_weighted_height"
        upper_condensate_weighted_height.attrs["units"] = "km"
        process = {
            "cloud_water_path": condensate_path(
                hydrometeors["cloud_water"], density, layer_thickness, "cloud_water_path"
            ),
            "rain_water_path": condensate_path(
                hydrometeors["rain"], density, layer_thickness, "rain_water_path"
            ),
            "cloud_ice_path": condensate_path(
                hydrometeors["ice"], density, layer_thickness, "cloud_ice_path"
            ),
            "snow_path": condensate_path(
                hydrometeors["snow"], density, layer_thickness, "snow_path"
            ),
            "graupel_path": condensate_path(
                hydrometeors["graupel"], density, layer_thickness, "graupel_path"
            ),
            "liquid_phase_condensate_path": condensate_path(
                liquid_phase, density, layer_thickness, "liquid_phase_condensate_path"
            ),
            "ice_phase_condensate_path": condensate_path(
                ice_phase, density, layer_thickness, "ice_phase_condensate_path"
            ),
            "precipitating_condensate_path": condensate_path(
                precipitating_condensate,
                density,
                layer_thickness,
                "precipitating_condensate_path",
            ),
            "total_condensate_path": condensate_path(
                total_condensate, density, layer_thickness, "total_condensate_path"
            ),
            "upper_liquid_phase_condensate_path": upper_liquid_path,
            "upper_ice_phase_condensate_path": upper_ice_path,
            "upper_total_condensate_path": upper_total_path,
            "upper_condensate_ice_fraction": upper_ice_fraction,
            "upper_condensate_weighted_height": upper_condensate_weighted_height,
        }
        out.update(process)

    for key, da in out.items():
        if key.endswith("_mask"):
            da.attrs["logical_type"] = "bool"
    if hydrometeor_sources is not None:
        source_summary = ",".join(
            f"{key}:{hydrometeor_sources.get(key, '')}"
            for key in HYDROMETEOR_ALIASES
        )
        for da in out.values():
            da.attrs["hydrometeor_sources"] = source_summary
    return out


def classify_high_cloud_from_dataset(
    ds,
    precip=None,
    cldfrac=None,
    pressure=None,
    convective_precip=None,
    grid_scale_precip=None,
    vertical_velocity=None,
    thresholds=None,
    warn_missing=False,
    dt_hours=6.0,
    include_process_diagnostics=False,
):
    if cldfrac is None:
        if "cldfrac" not in ds.data_vars:
            raise ValueError("cldfrac is unavailable for high-cloud classification")
        cldfrac = ds["cldfrac"]
    cf_ref = standardize_cloud_fraction(cldfrac)
    if pressure is None:
        pressure_name = first_dataset_var(ds, ("pressure", "p"))
        if pressure_name is None:
            raise ValueError(
                "Model-level pressure is unavailable for high-cloud classification"
            )
        pressure = ds[pressure_name]
    if convective_precip is None:
        convective_precip = precip_component_rate_from_dataset(
            ds, "convective", dt_hours=dt_hours
        )
    if grid_scale_precip is None:
        grid_scale_precip = precip_component_rate_from_dataset(
            ds, "grid_scale", dt_hours=dt_hours
        )
    if vertical_velocity is None:
        w_name = first_dataset_var(ds, ("w", "vertical_velocity"))
        if w_name is None:
            raise ValueError(
                "Model-level vertical velocity is unavailable for classification"
            )
        vertical_velocity = midlevel_field_from_interfaces(ds[w_name], cf_ref, w_name)
    hydrometeors = None
    sources = None
    density = None
    layer_thickness = None
    if include_process_diagnostics:
        hydrometeors, sources = hydrometeor_fields_from_dataset(
            ds,
            reference=cf_ref,
            warn_missing=warn_missing,
            strict=True,
        )
        rho_name = first_dataset_var(ds, ("rho", "density"))
        if rho_name is None:
            raise ValueError(
                "MPAS rho is required for process validation diagnostics"
            )
        density = standardize_model_level_field(
            ds[rho_name], reference=cf_ref, name=rho_name
        )
        layer_thickness = layer_thickness_from_dataset(ds, cf_ref)
    return classify_high_cloud_3d(
        cldfrac=cf_ref,
        pressure=pressure,
        precip=precip,
        convective_precip=convective_precip,
        grid_scale_precip=grid_scale_precip,
        hydrometeors=hydrometeors,
        vertical_velocity=vertical_velocity,
        density=density,
        layer_thickness=layer_thickness,
        thresholds=thresholds,
        hydrometeor_sources=sources,
        include_process_diagnostics=include_process_diagnostics,
    )


def classification_dataset(classif, thresholds=None, extra_attrs=None):
    variables = {}
    for key, da in classif.items():
        out = da
        if getattr(out, "dtype", None) == bool:
            out = out.astype("int8")
            out.attrs.update(logical_type="bool", true_value=1, false_value=0)
        variables[key] = out
    ds = xr.Dataset(variables)
    attrs = classification_cache_metadata(thresholds)
    attrs.update(
        classification_summary=(
            "High cloud is defined where the highest model level with cldfrac "
            "above threshold has pressure <= 440 hPa. Cloud-top pressure is "
            "retained continuously for QBO-MJO cloud-deepening analysis. "
            "Precipitating high cloud is convective when at least two of three "
            "coarse-grid indicators are present: heavy total rain, strong resolved "
            "ascent within cloudy levels, and parameterized-convective rainc "
            "dominance as a proxy for unresolved cellular structure. Remaining "
            "precipitating high cloud is stratiform. Valid high-cloud columns are "
            "exhaustively partitioned into convective precipitating, stratiform "
            "precipitating, and non-precipitating regimes."
        ),
        resolution_caveat=(
            "Designed for 0.5 degree instantaneous model-state snapshots sampled every "
            "6 hours. Vertical velocity is instantaneous; precipitation rates are derived "
            "from accumulation increments between snapshots. Both are coarse-grid "
            "indicators rather than resolved core values. "
            "rainc/rainnc are model-process diagnostics, and neither source alone "
            "defines physical convective or stratiform precipitation."
        ),
        method_references=(
            "Rossow and Schiffer 1999, BAMS, "
            "doi:10.1175/1520-0477(1999)080<2261:AIUCFI>2.0.CO;2; "
            "Schumacher and Houze 2003, Journal of Climate, "
            "doi:10.1175/1520-0442(2003)016<1739:SRITTA>2.0.CO;2; "
            "Shu et al. 2020, JGR Atmospheres, doi:10.1029/2019JD031510; "
            "Kysely et al. 2016, Climate Dynamics, doi:10.1007/s00382-015-2580-7; "
            "Fowler et al. 2020, GMD, doi:10.5194/gmd-13-2851-2020"
        ),
    )
    if extra_attrs:
        attrs.update(extra_attrs)
    ds.attrs.update(netcdf_safe_attrs(attrs))
    return ds
