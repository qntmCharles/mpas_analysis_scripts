#!/usr/bin/env python
# coding: utf-8

"""Shared, literature-based three-dimensional high-cloud classification.

High cloud retains the established ISCCP-style parent definition (cloud
fraction >= 0.05 and cloud-top pressure <= 440 hPa). Convective cores come
directly from the shared PyFLEXTRKR 225/241-K cold-cloud identification. All
remaining high cloud is stratiform where surface rain is at least 0.5 mm h-1
and non-precipitating otherwise. The native PyFLEXTRKR cold- and warm-shield
masks are retained as scene context, but are not additional cloud classes and
do not alter this exhaustive three-way baseline classification.

The established downstream variable names are retained. In particular,
``convective_precipitating_high_cloud_mask`` is now a legacy name for the
PyFLEXTRKR core/high-cloud intersection; precipitation is not required to
identify that class.
"""

from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr

from diagnostics.diagnostic_helpers import (
    grid_cell_area_km2,
    netcdf_safe_attrs,
)
from diagnostics.convection.pyflextrkr_cloud_identification import (
    PYFLEXTRKR_SCENE_VERSION,
    WARM_SHIELD_DERIVATION_VERSION,
    derive_warm_shield_masks,
    identify_cold_clouds,
    scene_metadata,
)


# Dropping redundant derived display masks does not change any science mask.
# Keep this semantic version stable so downstream PyFLEXTRKR track products,
# which depend only on the unchanged native scene masks, remain reusable.
CLASSIFICATION_CACHE_VERSION = "high-cloud-pyflextrkr-rain-baseline-v11"


@dataclass(frozen=True)
class HighCloudThresholds:
    """Thresholds that directly define the shared cloud classification."""

    cf_thresh: float = 0.05
    high_cloud_top_pressure_hpa: float = 440.0
    # Feng et al. define a precipitation feature using 0.5 mm h-1.
    surface_precip_min_mm_day: float = 12.0


def classification_cache_metadata(thresholds=None, extra=None):
    thresholds = thresholds or HighCloudThresholds()
    out = {
        "cache_version": CLASSIFICATION_CACHE_VERSION,
        **asdict(thresholds),
        **scene_metadata(),
        "surface_precip_threshold_mm_h": (
            thresholds.surface_precip_min_mm_day / 24.0
        ),
        "vertical_precip_attribution_method": "none in baseline classification; surface rain is associated with the high-cloud column",
    }
    if extra:
        out.update(extra)
    return out


def cached_classification_is_current(cached, thresholds=None):
    expected = classification_cache_metadata(thresholds)
    for key, value in expected.items():
        cached_value = cached.attrs.get(key)
        if isinstance(value, float):
            if cached_value is None or not np.isclose(float(cached_value), value):
                return False
        elif cached_value != value:
            return False
    return True


def corrected_cached_warm_shield_masks(cached):
    """Return sound warm masks without invalidating expensive science caches.

    Caches written before ``periodic-connected-components-v1`` can contain a
    validation-only warm-shield error caused by a negative growth sentinel.
    Their core, cold shield, cloud, and precipitation classes remain valid, so
    reconstruct only the two inexpensive warm masks lazily from cached scene
    fields rather than requiring the full 3-D classification to be rebuilt.
    """
    warm_names = (
        "pyflextrkr_warm_cloud_shield_mask",
        "pyflextrkr_core_bearing_warm_shield_mask",
    )
    stored_is_current = all(
        name in cached
        and cached[name].attrs.get("warm_shield_derivation_version")
        == WARM_SHIELD_DERIVATION_VERSION
        for name in warm_names
    )
    if stored_is_current:
        return {name: cached[name] for name in warm_names}, False

    required = (
        "brightness_temperature",
        "pyflextrkr_cold_cloud_shield_label",
        "pyflextrkr_core_mask",
    )
    missing = [name for name in required if name not in cached]
    if missing:
        raise KeyError(
            "Cannot repair legacy warm-shield fields; cached classification "
            f"lacks {missing}"
        )
    warm, core_bearing_warm = derive_warm_shield_masks(
        cached["brightness_temperature"],
        cached["pyflextrkr_cold_cloud_shield_label"],
        cached["pyflextrkr_core_mask"].fillna(False).astype(bool),
    )
    return {
        warm_names[0]: warm,
        warm_names[1]: core_bearing_warm,
    }, True


def standardize_model_level_field(da, reference=None, name=None):
    out = da
    if "nVertLevels" in out.dims:
        out = out.rename({"nVertLevels": "z"})
    elif "z" not in out.dims:
        raise ValueError(f"{name or da.name or 'field'} must use z or nVertLevels")
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
    if zvals.size > 1 and np.any(np.diff(zvals) <= 0.0):
        raise ValueError("cldfrac z coordinate must increase upward")
    return cf


def z_km_from_cldfrac(cf):
    return (cf.z / 1000.0).rename("z_km")


def standardize_pressure_hpa(pressure, reference):
    out = standardize_model_level_field(pressure, reference=reference, name="pressure")
    out = out / 100.0
    out.name = "pressure_hpa"
    out.attrs.update(units="hPa", positive="down")
    return out


def classify_high_cloud_3d(
    cldfrac,
    pressure=None,
    precip=None,
    olr=None,
    pyflextrkr_core_mask=None,
    pyflextrkr_cold_cloud_shield_mask=None,
    pyflextrkr_core_bearing_cold_shield_mask=None,
    pyflextrkr_warm_cloud_shield_mask=None,
    pyflextrkr_core_bearing_warm_shield_mask=None,
    pyflextrkr_cold_cloud_shield_label=None,
    pyflextrkr_valid_tb_mask=None,
    brightness_temperature=None,
    thresholds=None,
):
    """Return the exhaustive PyFLEXTRKR/precipitation high-cloud split."""
    thresholds = thresholds or HighCloudThresholds()
    cf = standardize_cloud_fraction(cldfrac)
    if pressure is None or precip is None:
        raise ValueError("Model-level pressure and total precipitation rate are required")
    pressure_hpa = standardize_pressure_hpa(pressure, cf)
    precip = precip.clip(min=0.0)

    if pyflextrkr_core_mask is None:
        if olr is None:
            raise ValueError("OLR or precomputed PyFLEXTRKR scene masks are required")
        area = grid_cell_area_km2(olr.latitude.values, olr.longitude.values)
        scene = identify_cold_clouds(olr, area)
        brightness_temperature = scene.brightness_temperature
        pyflextrkr_valid_tb_mask = scene.pyflextrkr_valid_tb_mask
        pyflextrkr_core_mask = scene.pyflextrkr_core_mask
        pyflextrkr_cold_cloud_shield_mask = scene.pyflextrkr_cold_cloud_shield_mask
        pyflextrkr_core_bearing_cold_shield_mask = scene.pyflextrkr_core_bearing_cold_shield_mask
        pyflextrkr_warm_cloud_shield_mask = scene.pyflextrkr_warm_cloud_shield_mask
        pyflextrkr_core_bearing_warm_shield_mask = scene.pyflextrkr_core_bearing_warm_shield_mask
        pyflextrkr_cold_cloud_shield_label = scene.pyflextrkr_cold_cloud_shield_label
    if pyflextrkr_valid_tb_mask is None:
        pyflextrkr_valid_tb_mask = xr.ones_like(pyflextrkr_core_mask, dtype=bool)

    align_items = [cf, pressure_hpa, precip, pyflextrkr_core_mask, pyflextrkr_valid_tb_mask]
    names = ["cf", "pressure", "precip", "core", "valid_tb"]
    optional = {
        "shield": pyflextrkr_cold_cloud_shield_mask,
        "core_bearing_shield": pyflextrkr_core_bearing_cold_shield_mask,
        "warm_shield": pyflextrkr_warm_cloud_shield_mask,
        "core_bearing_warm_shield": pyflextrkr_core_bearing_warm_shield_mask,
        "label": pyflextrkr_cold_cloud_shield_label,
        "tb": brightness_temperature,
    }
    for name, da in optional.items():
        if da is not None:
            align_items.append(da)
            names.append(name)
    aligned = xr.align(*align_items, join="inner")
    values = dict(zip(names, aligned))
    cf, pressure_hpa, precip = values["cf"], values["pressure"], values["precip"]
    core = values["core"].fillna(False).astype(bool)
    valid_tb = values["valid_tb"].fillna(False).astype(bool)

    z_km = z_km_from_cldfrac(cf)
    cloudy = (cf >= thresholds.cf_thresh).fillna(False)
    cloud_top_pressure = pressure_hpa.where(cloudy).min("z", skipna=True).rename("cloud_top_pressure")
    cloud_top_pressure.attrs.update(units="hPa", positive="down")
    high_cloud_mask = (cloud_top_pressure <= thresholds.high_cloud_top_pressure_hpa).fillna(False)
    high_cloud_fraction = cf.where(pressure_hpa <= thresholds.high_cloud_top_pressure_hpa).mean("z", skipna=True)
    high_cloud_fraction.name = "high_cloud_fraction"
    cloud_top_height = z_km.where(cloudy).max("z", skipna=True).rename("cloud_top_height")
    cloud_base_height = z_km.where(cloudy).min("z", skipna=True).rename("cloud_base_height")
    cloud_depth = (cloud_top_height - cloud_base_height).rename("cloud_depth")
    for da in (cloud_top_height, cloud_base_height, cloud_depth):
        da.attrs["units"] = "km"

    valid_precip = precip.notnull()
    precipitating = (precip >= thresholds.surface_precip_min_mm_day).fillna(False)

    # Core geometry is the sole convective classifier. Every remaining valid
    # high-cloud column is divided only by the published PF rain threshold.
    # Shield association is retained only as PyFLEXTRKR scene context, not as a
    # class gate or as another high-cloud category.
    convective = (high_cloud_mask & valid_tb & core).fillna(False)
    eligible_remaining = high_cloud_mask & ~convective
    stratiform = (eligible_remaining & valid_precip & precipitating).fillna(False)
    nonprecipitating = (
        eligible_remaining & valid_precip & ~precipitating
    ).fillna(False)
    unclassified = (
        eligible_remaining & ~valid_precip
    ).fillna(False)

    convective.name = "convective_precipitating_high_cloud_mask"
    stratiform.name = "stratiform_precipitating_high_cloud_mask"
    nonprecipitating.name = "nonprecipitating_high_cloud_mask"
    unclassified.name = "unclassified_high_cloud_mask"
    high_cloud_mask.name = "high_cloud_mask"
    regime_code = xr.zeros_like(high_cloud_mask, dtype="int8")
    regime_code = regime_code.where(~convective, 1).where(~stratiform, 2).where(~nonprecipitating, 3).where(~unclassified, 4)
    regime_code.name = "high_cloud_regime_code"
    regime_code.attrs.update(
        flag_values="0,1,2,3,4",
        flag_meanings="not_high_cloud pyflextrkr_convective_core stratiform_precipitating nonprecipitating unclassified",
    )
    out = {
        "cloud_top_pressure": cloud_top_pressure,
        "cloud_top_height": cloud_top_height,
        "cloud_base_height": cloud_base_height,
        "cloud_depth": cloud_depth,
        "high_cloud_fraction": high_cloud_fraction,
        "high_cloud_mask": high_cloud_mask,
        "convective_precipitating_high_cloud_mask": convective,
        "stratiform_precipitating_high_cloud_mask": stratiform,
        "nonprecipitating_high_cloud_mask": nonprecipitating,
        "unclassified_high_cloud_mask": unclassified,
        "high_cloud_regime_code": regime_code,
        "surface_precip_rate": precip,
        "pyflextrkr_valid_tb_mask": valid_tb,
        "pyflextrkr_core_mask": core,
    }
    for key in (
        "shield", "core_bearing_shield", "warm_shield",
        "core_bearing_warm_shield", "label", "tb",
    ):
        if key in values:
            out[{
                "shield": "pyflextrkr_cold_cloud_shield_mask",
                "core_bearing_shield": "pyflextrkr_core_bearing_cold_shield_mask",
                "warm_shield": "pyflextrkr_warm_cloud_shield_mask",
                "core_bearing_warm_shield": "pyflextrkr_core_bearing_warm_shield_mask",
                "label": "pyflextrkr_cold_cloud_shield_label",
                "tb": "brightness_temperature",
            }[key]] = values[key]
    for key, da in out.items():
        if key.endswith("_mask"):
            da.attrs["logical_type"] = "bool"
    convective.attrs.update(
        long_name="high cloud intersecting a PyFLEXTRKR 225-K cold core",
        legacy_name_note="Precipitation is not required despite the retained downstream variable name",
    )
    stratiform.attrs["long_name"] = "rainy high cloud outside the PyFLEXTRKR convective core"
    nonprecipitating.attrs["long_name"] = "non-raining high cloud outside the PyFLEXTRKR convective core"
    return out


def classify_high_cloud_from_dataset(
    ds,
    precip=None,
    cldfrac=None,
    pressure=None,
    olr=None,
    thresholds=None,
):
    cldfrac = ds.cldfrac if cldfrac is None else cldfrac
    cf_ref = standardize_cloud_fraction(cldfrac)
    pressure = ds.pressure if pressure is None else pressure
    if precip is None:
        raise ValueError("Total precipitation rate is required")
    if olr is None:
        for name in ("olrtoa", "olr"):
            if name in ds:
                olr = ds[name]
                break
    return classify_high_cloud_3d(
        cldfrac=cf_ref,
        pressure=pressure,
        precip=precip,
        olr=olr,
        thresholds=thresholds,
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
            "High cloud: cldfrac >= 0.05 with cloud-top pressure <= 440 hPa. "
            "Convective core: intersection with the shared PyFLEXTRKR Tb < 225 K "
            "core grown within a Tb < 241 K cold shield. Every remaining high-"
            "cloud column with valid precipitation is stratiform at rain >= "
            "0.5 mm h-1 and non-precipitating otherwise. No vertical-layer "
            "attribution is applied in this baseline. Native core-bearing "
            "241-K cold- and 261-K warm-shield masks are retained only as "
            "PyFLEXTRKR scene context. They do not define additional cloud "
            "classes or alter Feng MCS acceptance."
        ),
        method_references=(
            "Rossow and Schiffer 1999, BAMS, doi:10.1175/1520-0477(1999)080<2261:AIUCFI>2.0.CO;2; "
            "Feng et al. 2021, Journal of Climate, doi:10.1175/JCLI-D-20-0136.1; "
            "Feng et al. 2023, GMD, doi:10.5194/gmd-16-2753-2023"
        ),
        pyflextrkr_scene_version=PYFLEXTRKR_SCENE_VERSION,
    )
    if extra_attrs:
        attrs.update(extra_attrs)
    ds.attrs.update(netcdf_safe_attrs(attrs))
    return ds
