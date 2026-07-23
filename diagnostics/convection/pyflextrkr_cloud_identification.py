#!/usr/bin/env python
# coding: utf-8

"""Shared PyFLEXTRKR cold-core/cold-shield identification.

The scene segmentation follows the 50-km configuration in Feng et al.
(2021, J. Climate, doi:10.1175/JCLI-D-20-0136.1): OLR-derived brightness
temperature cores colder than 225 K are spread to the connected 241 K cold
cloud shield.  The product deliberately contains *all* detected shields;
the lifetime-dependent precipitation-feature criteria used to identify MCSs
are applied later by the tracking diagnostic.
"""

from dataclasses import asdict, dataclass

import numpy as np
import xarray as xr


PYFLEXTRKR_SCENE_VERSION = "feng-2021-50km-core-cold-warm-hierarchy-v2"
WARM_SHIELD_DERIVATION_VERSION = "periodic-connected-components-v1"


@dataclass(frozen=True)
class PyFLEXTRKRSceneThresholds:
    core_tb_k: float = 225.0
    cold_tb_k: float = 241.0
    warm_tb_k: float = 261.0
    cloud_tb_k: float = 293.0
    min_valid_tb_k: float = 160.0
    max_valid_tb_k: float = 330.0
    smoothing_cells: int = 5
    min_cold_core_pixels: int = 1


def scene_metadata(thresholds=None):
    thresholds = thresholds or PyFLEXTRKRSceneThresholds()
    return {
        "pyflextrkr_scene_version": PYFLEXTRKR_SCENE_VERSION,
        **{f"pyflextrkr_{key}": value for key, value in asdict(thresholds).items()},
        "pyflextrkr_scene_method": "OLR-to-Tb; detect Tb < 225 K and spread to connected Tb < 241 K; retain a separate connected Tb < 261 K warm-shield extension",
        "pyflextrkr_scene_reference": "Feng et al. 2021, Journal of Climate, doi:10.1175/JCLI-D-20-0136.1",
    }


def merge_periodic_longitude_labels(labels):
    """Merge 8-connected labels that meet across the periodic longitude seam."""
    # PyFLEXTRKR's growth helper uses -1 for excluded pixels.  Negative labels
    # must never reach the remapping array below: NumPy would interpret them as
    # indices from the end and could turn excluded pixels into a valid object.
    labels = np.where(np.asarray(labels) > 0, labels, 0).astype(np.int32)
    max_label = int(labels.max(initial=0))
    if max_label == 0 or labels.shape[1] < 2:
        return labels
    parent = np.arange(max_label + 1, dtype=np.int32)

    def find(value):
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(a, b):
        if a <= 0 or b <= 0:
            return
        ra, rb = find(int(a)), find(int(b))
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    nlat = labels.shape[0]
    for j in range(nlat):
        for jj in range(max(0, j - 1), min(nlat, j + 2)):
            union(labels[j, 0], labels[jj, -1])

    roots = np.array([find(i) for i in range(max_label + 1)], dtype=np.int32)
    positive_roots = sorted(root for root in np.unique(roots[1:]) if root > 0)
    remap = np.zeros(max_label + 1, dtype=np.int32)
    for new_label, root in enumerate(positive_roots, start=1):
        remap[roots == root] = new_label
    return remap[labels]


def olr_to_brightness_temperature(olr, thresholds=None):
    """Use PyFLEXTRKR's published OLR-to-Tb conversion and validity range."""
    from pyflextrkr.idclouds_tbpf import olr_to_tb

    thresholds = thresholds or PyFLEXTRKRSceneThresholds()
    values = np.asarray(olr, dtype=float)
    tb = np.full(values.shape, np.nan, dtype=float)
    valid_olr = np.isfinite(values) & (values > 0.0)
    if np.any(valid_olr):
        with np.errstate(invalid="ignore", divide="ignore"):
            tb[valid_olr] = olr_to_tb(values[valid_olr])
    valid_tb = (
        np.isfinite(tb)
        & (tb >= thresholds.min_valid_tb_k)
        & (tb <= thresholds.max_valid_tb_k)
    )
    return np.where(valid_tb, tb, np.nan)


def _labels_associated_with_core(labels, core):
    """Return all labelled pixels belonging to a label touching a core pixel."""
    associated_labels = set(int(value) for value in labels[core & (labels > 0)])
    for j, i in np.argwhere(core & (labels == 0)):
        neighbours = np.take(
            labels[max(0, j - 1):min(labels.shape[0], j + 2), :],
            [(i - 1) % labels.shape[1], i, (i + 1) % labels.shape[1]],
            axis=1,
        )
        associated_labels.update(
            int(value) for value in np.ravel(neighbours) if value > 0
        )
    associated = core.copy()
    if associated_labels:
        associated |= np.isin(labels, list(associated_labels))
    return associated


def _warm_shield_masks_numpy(
    brightness_temperature,
    cold_shield_labels,
    core_mask,
    warm_tb_k,
):
    """Grow cold shields through their connected sub-``warm_tb_k`` regions."""
    from scipy import ndimage

    tb = np.asarray(brightness_temperature, dtype=float)
    cold_labels = np.asarray(cold_shield_labels, dtype=np.int32)
    core = np.asarray(core_mask, dtype=bool)
    warm_candidate = np.isfinite(tb) & (tb < float(warm_tb_k))

    # Only candidate components containing a retained cold shield belong to a
    # warm extension.  Labelling the Boolean candidate region directly avoids
    # PyFLEXTRKR's -1 growth sentinel and also handles connectivity across the
    # periodic longitude seam before selecting associated components.
    component_labels, _ = ndimage.label(
        warm_candidate,
        structure=np.ones((3, 3), dtype=np.int8),
    )
    component_labels = merge_periodic_longitude_labels(component_labels)
    warm_shield = _labels_associated_with_core(
        component_labels,
        cold_labels > 0,
    ) & warm_candidate
    core_bearing_warm_shield = _labels_associated_with_core(
        component_labels,
        core,
    ) & warm_candidate
    return warm_shield, core_bearing_warm_shield


def derive_warm_shield_masks(
    brightness_temperature,
    cold_shield_labels,
    core_mask,
    thresholds=None,
):
    """Derive corrected warm-shield masks from scene fields, including lazily."""
    thresholds = thresholds or PyFLEXTRKRSceneThresholds()
    brightness_temperature = brightness_temperature.transpose(
        ..., "latitude", "longitude"
    )
    cold_shield_labels = cold_shield_labels.transpose(
        ..., "latitude", "longitude"
    )
    core_mask = core_mask.transpose(..., "latitude", "longitude")
    warm_shield, core_bearing_warm_shield = xr.apply_ufunc(
        _warm_shield_masks_numpy,
        brightness_temperature,
        cold_shield_labels,
        core_mask,
        input_core_dims=[
            ["latitude", "longitude"],
            ["latitude", "longitude"],
            ["latitude", "longitude"],
        ],
        output_core_dims=[
            ["latitude", "longitude"],
            ["latitude", "longitude"],
        ],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[bool, bool],
        kwargs={"warm_tb_k": thresholds.warm_tb_k},
    )
    warm_shield.name = "pyflextrkr_warm_cloud_shield_mask"
    core_bearing_warm_shield.name = (
        "pyflextrkr_core_bearing_warm_shield_mask"
    )
    attrs = {
        **scene_metadata(thresholds),
        "logical_type": "bool",
        "warm_shield_derivation_version": WARM_SHIELD_DERIVATION_VERSION,
    }
    warm_shield.attrs.update(attrs)
    core_bearing_warm_shield.attrs.update(attrs)
    return warm_shield, core_bearing_warm_shield


def identify_scene(olr, cell_area_km2, thresholds=None):
    """Return native core/cold shield plus an optional connected warm extension."""
    from pyflextrkr.label_and_grow_cold_clouds import label_and_grow_cold_clouds

    thresholds = thresholds or PyFLEXTRKRSceneThresholds()
    tb = olr_to_brightness_temperature(olr, thresholds)
    valid = np.isfinite(tb)
    empty_labels = np.zeros(tb.shape, dtype=np.int32)
    if not np.any(valid):
        return (
            tb.astype(np.float32), valid, valid.copy(), valid.copy(),
            valid.copy(), valid.copy(), valid.copy(), empty_labels
        )

    area = np.asarray(cell_area_km2, dtype=float)
    usable_area = area[valid & np.isfinite(area) & (area > 0.0)]
    pixel_area = float(np.nanmedian(usable_area)) if usable_area.size else 1.0
    # This is the convention used internally by the existing PyFLEXTRKR call.
    pixel_radius_km = float(np.sqrt(max(pixel_area, 1.0)))
    tb_thresholds = np.asarray(
        [
            thresholds.core_tb_k,
            thresholds.cold_tb_k,
            thresholds.warm_tb_k,
            thresholds.cloud_tb_k,
        ],
        dtype=float,
    )
    result = label_and_grow_cold_clouds(
        tb,
        pixel_radius_km,
        tb_thresholds,
        0.0,  # retain every core/shield; MCS area filtering occurs after tracking
        thresholds.min_cold_core_pixels,
        thresholds.smoothing_cells,
        0,
    )
    labels = np.asarray(result["final_convcold_cloudnumber"], dtype=np.int32)
    labels = merge_periodic_longitude_labels(np.where(valid, labels, 0))
    cold_shield = labels > 0
    # PyFLEXTRKR cloud type 1 is the cold-core portion of a retained object.
    cloud_type = np.asarray(result["final_cloudtype"], dtype=np.int16)
    # ``final_cloudtype`` is PyFLEXTRKR's native unsmoothed pixel-type output.
    # Keep its 225-K core pixels even when the 5-cell smoothing used to seed
    # shield growth removes a one-cell minimum in very coarse data.
    core = valid & (cloud_type == 1)
    core_bearing_shield = _labels_associated_with_core(labels, core)

    warm_shield, core_bearing_warm_shield = _warm_shield_masks_numpy(
        tb,
        labels,
        core,
        thresholds.warm_tb_k,
    )
    return (
        tb.astype(np.float32), valid, core, cold_shield, core_bearing_shield,
        warm_shield, core_bearing_warm_shield, labels
    )


def identify_cold_clouds(olr, cell_area_km2, thresholds=None):
    """Apply :func:`identify_scene` to every scene of an xarray OLR field."""
    thresholds = thresholds or PyFLEXTRKRSceneThresholds()
    olr = olr.transpose(..., "latitude", "longitude")
    cell_area_km2 = cell_area_km2.sel(
        latitude=olr.latitude, longitude=olr.longitude
    ).transpose("latitude", "longitude")
    (
        tb, valid, core, shield, core_bearing_shield,
        warm_shield, core_bearing_warm_shield, labels,
    ) = xr.apply_ufunc(
        identify_scene,
        olr,
        cell_area_km2,
        input_core_dims=[
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
            ["latitude", "longitude"],
        ],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"allow_rechunk": True},
        output_dtypes=[np.float32, bool, bool, bool, bool, bool, bool, np.int32],
        kwargs={"thresholds": thresholds},
    )
    variables = {
        "brightness_temperature": tb,
        "pyflextrkr_valid_tb_mask": valid,
        "pyflextrkr_core_mask": core,
        "pyflextrkr_cold_cloud_shield_mask": shield,
        "pyflextrkr_core_bearing_cold_shield_mask": core_bearing_shield,
        "pyflextrkr_warm_cloud_shield_mask": warm_shield,
        "pyflextrkr_core_bearing_warm_shield_mask": core_bearing_warm_shield,
        "pyflextrkr_cold_cloud_shield_label": labels,
    }
    attrs = scene_metadata(thresholds)
    for name, da in variables.items():
        da.name = name
        da.attrs.update(attrs)
    variables["brightness_temperature"].attrs.update(units="K")
    variables["pyflextrkr_cold_cloud_shield_label"].attrs.update(
        background_label=0, label_scope="unique within each scene"
    )
    for name in (
        "pyflextrkr_valid_tb_mask",
        "pyflextrkr_core_mask",
        "pyflextrkr_cold_cloud_shield_mask",
        "pyflextrkr_core_bearing_cold_shield_mask",
        "pyflextrkr_warm_cloud_shield_mask",
        "pyflextrkr_core_bearing_warm_shield_mask",
    ):
        variables[name].attrs["logical_type"] = "bool"
    return xr.Dataset(variables, attrs=attrs)
