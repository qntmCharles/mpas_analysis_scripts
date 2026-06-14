#!/usr/bin/env python
# coding: utf-8

"""
Shared 3-D hydrometeor-aware high-cloud classification.

The parent cloud geometry is always defined from ``cldfrac``. Hydrometeors are
then used to separate active deep convection, convectively generated high cloud,
and cirrus-like high cloud. Water vapour can be carried for context elsewhere,
but it is intentionally not used as a class-defining hydrometeor here.
"""

from dataclasses import asdict, dataclass, replace
import warnings

import numpy as np
import xarray as xr
from scipy import ndimage

from diagnostic_helpers import finite_or_nan, vertical_dim


HYDROMETEOR_ALIASES = {
    "ice": ("qi", "qice", "q_i", "ice", "cloud_ice"),
    "snow": ("qs", "qsnow", "q_s", "snow"),
    "graupel": ("qg", "qgraup", "qgraupel", "q_g", "graupel"),
    "cloud_water": ("qc", "qcloud", "q_cloud", "q_c", "cloud_water"),
    "rain": ("qr", "qrain", "q_r", "rain_water"),
}

HYDROMETEOR_NEEDED_VARS = sorted({name for aliases in HYDROMETEOR_ALIASES.values() for name in aliases})
CLASSIFICATION_CACHE_VERSION = "high-cloud-3d-hydrometeor-v1"


@dataclass(frozen=True)
class HighCloudThresholds:
    """Physical thresholds for the shared high-cloud classification."""

    cf_thresh: float = 0.05
    z_high_km: float = 12.0
    low_cloud_min_km: float = 2.0
    low_cloud_max_km: float = 5.0
    mid_cloud_min_km: float = 5.0
    mid_cloud_max_km: float = 10.0
    upper_cloud_min_km: float = 10.0
    max_clear_gap_km: float = 3.0
    deep_min_depth_km: float = 8.0
    surface_precip_deep_mm_day: float = 5.0
    surface_precip_weak_mm_day: float = 1.0
    hydrometeor_present_kg_kg: float = 1.0e-8
    substantial_hydrometeor_kg_kg: float = 1.0e-7
    cirrus_max_depth_km: float = 4.0
    convective_generated_max_distance_km: float = 300.0
    cirrus_min_distance_km: float = 500.0
    require_surface_precip_for_active: bool = True


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
    """Convert Python/numpy scalar attrs to values accepted by netCDF4."""
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
                cached_bool = cached_value.strip().lower() in {"1", "true", "t", "yes", "y", "on"}
            else:
                cached_bool = bool(cached_value)
            if cached_bool != value:
                return False
        elif cached_value != value:
            return False
    return True


def available_cloud_fraction_times(cf):
    if "Time" not in cf.dims:
        return None
    indexers = {dim: 0 for dim in cf.dims if dim != "Time"}
    valid = cf.isel(indexers).notnull().load()
    return valid.Time.values[np.asarray(valid.values, dtype=bool)]


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


def standardize_cloud_fraction(cldfrac, drop_missing_times=True):
    cf = standardize_model_level_field(cldfrac, name="cldfrac")
    if "z" not in cf.coords:
        raise ValueError("cldfrac has no geometric-height z coordinate")
    zvals = np.asarray(cf.z.values, dtype=float)
    if zvals.size != cf.sizes["z"] or not np.all(np.isfinite(zvals)):
        raise ValueError("cldfrac z coordinate is not usable")
    if np.nanmax(np.abs(zvals)) < 100.0:
        cf = cf.assign_coords(z=cf.z * 1000.0)
    if drop_missing_times:
        valid_times = available_cloud_fraction_times(cf)
        if valid_times is not None:
            cf = cf.sel(Time=valid_times)
    return cf


def z_km_from_cldfrac(cf):
    zvals = np.asarray(cf.z.values, dtype=float)
    zvals = zvals / 1000.0 if np.nanmax(np.abs(zvals)) > 100.0 else zvals
    return xr.DataArray(zvals, dims=("z",), coords={"z": cf.z}, name="z_km")


def any_cloud_in_layer(cloudy, z_km, zmin, zmax):
    return cloudy.where((z_km >= zmin) & (z_km < zmax), drop=True).any(dim="z")


def hydrometeor_fields_from_dataset(ds, reference=None, warn_missing=False):
    fields = {}
    sources = {}
    for key, aliases in HYDROMETEOR_ALIASES.items():
        source = next((name for name in aliases if name in ds.data_vars), None)
        if source is None:
            if warn_missing:
                warnings.warn(f"Hydrometeor {key!r} is unavailable; treating it as zero.")
            if reference is not None:
                fields[key] = xr.zeros_like(reference)
            sources[key] = ""
            continue
        try:
            fields[key] = standardize_model_level_field(ds[source], reference=reference, name=source)
            sources[key] = source
        except Exception as exc:
            if warn_missing:
                warnings.warn(f"Could not use hydrometeor {source!r}: {exc}; treating it as zero.")
            if reference is not None:
                fields[key] = xr.zeros_like(reference)
            sources[key] = ""
    return fields, sources


def grid_spacing_km_from_coords(lat, lon):
    lat = np.asarray(lat, dtype=float)
    lon = np.asarray(lon, dtype=float)
    dlat = float(np.nanmedian(np.abs(np.diff(lat)))) if lat.size > 1 else 1.0
    dlon = float(np.nanmedian(np.abs(np.diff(lon)))) if lon.size > 1 else 1.0
    dy = max(111.2 * dlat, 1.0)
    dx = max(111.2 * np.cos(np.deg2rad(float(np.nanmean(lat)))) * dlon, 1.0)
    return dy, dx


def time_chunked(da, size=1):
    if "Time" in da.dims and hasattr(da.data, "chunks"):
        return da.chunk({"Time": size})
    return da


def _periodic_distance_to_mask_km(mask2d, dy_km, dx_km):
    mask2d = np.asarray(mask2d, dtype=bool)
    if not np.any(mask2d):
        return np.full(mask2d.shape, np.nan, dtype=np.float32)
    nlon = mask2d.shape[1]
    tiled_clear = np.concatenate([~mask2d, ~mask2d, ~mask2d], axis=1)
    dist = ndimage.distance_transform_edt(tiled_clear, sampling=(dy_km, dx_km))
    return dist[:, nlon:2 * nlon].astype(np.float32)


def _object_min_distance_mask(labels, nlabels, distance, threshold_km, require_finite=True):
    out = np.zeros(labels.shape, dtype=bool)
    if nlabels == 0:
        return out
    if not np.any(np.isfinite(distance)):
        return out if require_finite else labels > 0
    indices = np.arange(1, nlabels + 1)
    mins = ndimage.minimum(distance, labels=labels, index=indices)
    mins = np.asarray(mins, dtype=float)
    ok = np.isfinite(mins) & (mins <= threshold_km)
    table = np.zeros(nlabels + 1, dtype=bool)
    table[indices[ok]] = True
    return table[labels]


def _object_far_distance_mask(labels, nlabels, distance, threshold_km, no_active_is_far=True):
    out = np.zeros(labels.shape, dtype=bool)
    if nlabels == 0:
        return out
    if not np.any(np.isfinite(distance)):
        return (labels > 0) if no_active_is_far else out
    indices = np.arange(1, nlabels + 1)
    mins = ndimage.minimum(distance, labels=labels, index=indices)
    mins = np.asarray(mins, dtype=float)
    ok = np.isfinite(mins) & (mins >= threshold_km)
    table = np.zeros(nlabels + 1, dtype=bool)
    table[indices[ok]] = True
    return table[labels]


def _object_classifier_numpy(
    cloudy3d,
    high2d,
    active_candidate2d,
    upper_ice_rich2d,
    weak_precip_hydro2d,
    shallow2d,
    precip_weak2d,
    dy_km,
    dx_km,
    convective_generated_max_distance_km,
    cirrus_min_distance_km,
):
    cloudy3d = np.asarray(cloudy3d, dtype=bool)
    high2d = np.asarray(high2d, dtype=bool)
    active = np.asarray(active_candidate2d, dtype=bool) & high2d
    upper_ice_rich2d = np.asarray(upper_ice_rich2d, dtype=bool)
    weak_precip_hydro2d = np.asarray(weak_precip_hydro2d, dtype=bool)
    shallow2d = np.asarray(shallow2d, dtype=bool)
    precip_weak2d = np.asarray(precip_weak2d, dtype=bool)

    distance = _periodic_distance_to_mask_km(active, dy_km, dx_km)

    high_labels, n_high = ndimage.label(high2d, structure=np.ones((3, 3), dtype=bool))
    high_connected = np.zeros(high2d.shape, dtype=bool)
    if n_high > 0 and np.any(active):
        active_labels = np.unique(high_labels[active])
        active_labels = active_labels[active_labels > 0]
        table = np.zeros(n_high + 1, dtype=bool)
        table[active_labels] = True
        high_connected = table[high_labels]

    near_active_object = _object_min_distance_mask(
        high_labels,
        n_high,
        distance,
        convective_generated_max_distance_km,
        require_finite=True,
    )
    far_from_active_object = _object_far_distance_mask(
        high_labels,
        n_high,
        distance,
        cirrus_min_distance_km,
        no_active_is_far=True,
    )

    same_3d_active = np.zeros(high2d.shape, dtype=bool)
    if np.any(active) and np.any(cloudy3d):
        labels3d, n3d = ndimage.label(cloudy3d, structure=np.ones((3, 3, 3), dtype=bool))
        if n3d > 0:
            active_voxel_labels = np.unique(labels3d[:, active])
            active_voxel_labels = active_voxel_labels[active_voxel_labels > 0]
            table3d = np.zeros(n3d + 1, dtype=bool)
            table3d[active_voxel_labels] = True
            same_3d_active = np.any(table3d[labels3d], axis=0) & high2d

    convective_generated = (
        high2d
        & ~active
        & upper_ice_rich2d
        & (high_connected | same_3d_active | near_active_object)
    )
    cirrus_like = (
        high2d
        & ~active
        & ~convective_generated
        & upper_ice_rich2d
        & shallow2d
        & weak_precip_hydro2d
        & precip_weak2d
        & far_from_active_object
    )
    unclassified = high2d & ~(active | convective_generated | cirrus_like)
    return (
        active,
        convective_generated,
        cirrus_like,
        unclassified,
        high_connected,
        same_3d_active,
        distance,
    )


def classify_high_cloud_3d(
    cldfrac,
    precip=None,
    hydrometeors=None,
    thresholds=None,
    hydrometeor_sources=None,
):
    """
    Return high-cloud masks using shared hydrometeor-aware definitions.

    The three science classes are mutually exclusive. Compatibility aliases are
    also returned so older diagnostics can read the shared cache without keeping
    their own classification logic.
    """
    thresholds = thresholds or HighCloudThresholds()
    cf = standardize_cloud_fraction(cldfrac)
    pr = precip
    if pr is None:
        pr = xr.zeros_like(cf.isel(z=0, drop=True))
        pr.name = "precip"
    hydrometeors = dict(hydrometeors or {})

    for key in HYDROMETEOR_ALIASES:
        if key not in hydrometeors:
            hydrometeors[key] = xr.zeros_like(cf)
            if hydrometeor_sources is not None:
                hydrometeor_sources.setdefault(key, "")
        else:
            hydrometeors[key] = standardize_model_level_field(
                hydrometeors[key],
                reference=cf,
                name=key,
            )

    inputs = [cf, finite_or_nan(pr)] + [hydrometeors[key] for key in HYDROMETEOR_ALIASES]
    aligned = xr.align(*inputs, join="inner")
    cf = aligned[0]
    pr = aligned[1]
    hydrometeors = dict(zip(HYDROMETEOR_ALIASES, aligned[2:]))
    z_km = z_km_from_cldfrac(cf)

    cloudy = (cf >= thresholds.cf_thresh).fillna(False)
    cloud_top_height = z_km.where(cloudy).max(dim="z", skipna=True)
    cloud_top_height.name = "cloud_top_height"
    cloud_top_height.attrs["units"] = "km"
    cloud_base_height = z_km.where(cloudy).min(dim="z", skipna=True)
    cloud_base_height.name = "cloud_base_height"
    cloud_base_height.attrs["units"] = "km"
    cloud_depth = (cloud_top_height - cloud_base_height).where(cloud_top_height.notnull())
    cloud_depth.name = "cloud_depth"
    cloud_depth.attrs["units"] = "km"

    high_layer = z_km >= thresholds.z_high_km
    high_cloud_fraction = cf.where(high_layer).mean(dim="z", skipna=True)
    high_cloud_fraction.name = "high_cloud_fraction"
    high_cloud_mask = (cloudy.where(high_layer, False).any(dim="z") | (high_cloud_fraction >= thresholds.cf_thresh)).fillna(False)
    high_cloud_mask.name = "high_cloud_mask"

    low_cloud = any_cloud_in_layer(
        cloudy,
        z_km,
        thresholds.low_cloud_min_km,
        thresholds.low_cloud_max_km,
    )
    mid_cloud = any_cloud_in_layer(
        cloudy,
        z_km,
        thresholds.mid_cloud_min_km,
        thresholds.mid_cloud_max_km,
    )
    upper_cloud = any_cloud_in_layer(
        cloudy,
        z_km,
        thresholds.upper_cloud_min_km,
        max(float(z_km.max()), thresholds.z_high_km + 0.1),
    )

    gap_ok = xr.ones_like(high_cloud_mask, dtype=bool)
    if thresholds.max_clear_gap_km is not None and thresholds.max_clear_gap_km > 0:
        for z0 in np.arange(
            thresholds.low_cloud_min_km,
            thresholds.z_high_km,
            thresholds.max_clear_gap_km,
        ):
            z1 = min(z0 + thresholds.max_clear_gap_km, max(float(z_km.max()), thresholds.z_high_km))
            if z1 > z0:
                gap_ok = gap_ok & any_cloud_in_layer(cloudy, z_km, z0, z1 + 1.0e-6)

    column_connected = (high_cloud_mask & low_cloud & mid_cloud & upper_cloud & gap_ok).fillna(False)
    column_connected.name = "column_connected_mask"

    ice_phase_condensate = hydrometeors["ice"] + hydrometeors["snow"] + hydrometeors["graupel"]
    precipitating_condensate = hydrometeors["rain"] + hydrometeors["snow"] + hydrometeors["graupel"]
    total_condensate = (
        hydrometeors["cloud_water"]
        + hydrometeors["rain"]
        + hydrometeors["ice"]
        + hydrometeors["snow"]
        + hydrometeors["graupel"]
    )
    rain_or_graupel = hydrometeors["rain"] + hydrometeors["graupel"]

    upper_ice_rich = (
        ice_phase_condensate.where(high_layer).max(dim="z", skipna=True)
        >= thresholds.hydrometeor_present_kg_kg
    ).fillna(False)
    precip_hydro_column = (
        precipitating_condensate.max(dim="z", skipna=True)
        >= thresholds.substantial_hydrometeor_kg_kg
    ).fillna(False)
    rain_graupel_weak = (
        rain_or_graupel.max(dim="z", skipna=True)
        < thresholds.substantial_hydrometeor_kg_kg
    ).fillna(True)
    surface_precip_ok = (pr >= thresholds.surface_precip_deep_mm_day).fillna(False)
    if not thresholds.require_surface_precip_for_active:
        surface_precip_ok = xr.ones_like(high_cloud_mask, dtype=bool)

    active_candidate = (
        high_cloud_mask
        & column_connected
        & (cloud_depth >= thresholds.deep_min_depth_km)
        & upper_ice_rich
        & precip_hydro_column
        & surface_precip_ok
    ).fillna(False)
    active_candidate.name = "active_deep_convection_candidate_mask"

    shallow = (cloud_depth <= thresholds.cirrus_max_depth_km).fillna(False)
    precip_weak = (pr <= thresholds.surface_precip_weak_mm_day).fillna(False)

    # The object/distance classifier needs full horizontal scenes, so keep the
    # non-core Time dimension small. Otherwise xarray/dask can build one very
    # large first task and sit at 0% progress for hours before any write lands.
    classifier_inputs = [
        time_chunked(da)
        for da in (
            cloudy,
            high_cloud_mask,
            active_candidate,
            upper_ice_rich,
            rain_graupel_weak,
            shallow,
            precip_weak,
        )
    ]

    dy_km, dx_km = grid_spacing_km_from_coords(cf.latitude.values, cf.longitude.values)
    outputs = xr.apply_ufunc(
        _object_classifier_numpy,
        *classifier_inputs,
        kwargs=dict(
            dy_km=dy_km,
            dx_km=dx_km,
            convective_generated_max_distance_km=thresholds.convective_generated_max_distance_km,
            cirrus_min_distance_km=thresholds.cirrus_min_distance_km,
        ),
        input_core_dims=[
            ["z", "latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
        ],
        output_core_dims=[
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
        ],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[bool, bool, bool, bool, bool, bool, np.float32],
        dask_gufunc_kwargs={"allow_rechunk": True},
    )

    (
        active_deep_convection,
        convectively_generated,
        cirrus_like,
        unclassified,
        high_object_connected_to_active,
        same_3d_object_as_active,
        distance_to_active,
    ) = outputs
    active_deep_convection.name = "active_deep_convection_high_cloud_mask"
    convectively_generated.name = "convectively_generated_high_cloud_mask"
    cirrus_like.name = "cirrus_like_high_cloud_mask"
    unclassified.name = "unclassified_high_cloud_mask"
    high_object_connected_to_active.name = "high_cloud_object_connected_to_active_mask"
    same_3d_object_as_active.name = "same_3d_cloud_object_as_active_mask"
    distance_to_active.name = "distance_to_active_convection_km"
    distance_to_active.attrs["units"] = "km"

    nonprecipitating = (high_cloud_mask & precip_weak & rain_graupel_weak).fillna(False)
    nonprecipitating.name = "nonprecipitating_high_cloud_mask"

    out = {
        "cloud_top_height": cloud_top_height,
        "cloud_base_height": cloud_base_height,
        "cloud_depth": cloud_depth,
        "high_cloud_fraction": high_cloud_fraction,
        "high_cloud_mask": high_cloud_mask,
        "column_connected_mask": column_connected,
        "active_deep_convection_candidate_mask": active_candidate,
        "active_deep_convection_high_cloud_mask": active_deep_convection,
        "convectively_generated_high_cloud_mask": convectively_generated,
        "cirrus_like_high_cloud_mask": cirrus_like,
        "unclassified_high_cloud_mask": unclassified,
        "nonprecipitating_high_cloud_mask": nonprecipitating,
        "high_cloud_object_connected_to_active_mask": high_object_connected_to_active,
        "same_3d_cloud_object_as_active_mask": same_3d_object_as_active,
        "distance_to_active_convection_km": distance_to_active,
        "total_condensate_column_sum": total_condensate.sum(dim="z", skipna=True),
        "precipitating_condensate_column_sum": precipitating_condensate.sum(dim="z", skipna=True),
        "ice_phase_condensate_column_sum": ice_phase_condensate.sum(dim="z", skipna=True),
    }
    for key in ("total_condensate_column_sum", "precipitating_condensate_column_sum", "ice_phase_condensate_column_sum"):
        out[key].attrs.update(
            units="kg kg-1 level-sum",
            caveat="Unweighted vertical sum used as a hydrometeor-content diagnostic, not a mass path.",
        )
    for key, da in out.items():
        if key.endswith("_mask"):
            da.attrs["logical_type"] = "bool"
    if hydrometeor_sources is not None:
        for key, source in hydrometeor_sources.items():
            out_key = f"hydrometeor_source_{key}"
            for da in out.values():
                da.attrs[out_key] = source
    return out


def classify_high_cloud_from_dataset(ds, precip=None, cldfrac=None, thresholds=None, warn_missing=False):
    if cldfrac is None:
        if "cldfrac" not in ds.data_vars:
            raise ValueError("cldfrac is unavailable for high-cloud classification")
        cldfrac = ds["cldfrac"]
    cf_ref = standardize_cloud_fraction(cldfrac)
    hydrometeors, sources = hydrometeor_fields_from_dataset(
        ds,
        reference=cf_ref,
        warn_missing=warn_missing,
    )
    return classify_high_cloud_3d(
        cf_ref,
        precip=precip,
        hydrometeors=hydrometeors,
        thresholds=thresholds,
        hydrometeor_sources=sources,
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
        dict(
            classification_summary=(
                "Parent high cloud from cldfrac; active deep convection requires deep "
                "vertical cloud plus precipitating hydrometeors and surface rain; "
                "convectively generated high cloud is ice-rich high cloud connected "
                "to active convection by a high-cloud/3-D cloud object or object "
                "distance; cirrus-like high cloud is shallow, weakly precipitating, "
                "and object-separated from active convection."
            )
        )
    )
    if extra_attrs:
        attrs.update(extra_attrs)
    ds.attrs.update(netcdf_safe_attrs(attrs))
    return ds
