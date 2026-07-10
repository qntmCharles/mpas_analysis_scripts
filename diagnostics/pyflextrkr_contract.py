"""Shared producer/consumer contract for MPAS PyFLEXTRKR products."""

from __future__ import annotations

import hashlib
import json
import os

import numpy as np
import pandas as pd


PYFLEXTRKR_PRODUCT_VERSION = "mpas-organised-convective-envelope-pyflextrkr-v9"
PYFLEXTRKR_METADATA_FILENAME = "pyflextrkr_product_metadata.json"
PYFLEXTRKR_TRACK_MANIFEST_FILENAME = "organised_track_manifest.csv"
PRIMARY_TRACK_SOURCE = "organised_convective_system_mask"
PRIMARY_ANALYSIS_ROLE = "primary_persistence_and_composites"
TRACKED_SOURCES = (PRIMARY_TRACK_SOURCE,)
MIN_ORGANISED_TRACK_OBSERVATIONS = 4
MIN_ORGANISED_TRACK_DURATION_HOURS = 24.0
ORGANISED_TRACK_ACCEPTANCE_CRITERIA = "duration_and_embedded_classifier_core"
ORGANISED_TRACK_EMBEDDED_CORE_REQUIREMENT = "any_positive_area_during_track"
ORGANISED_TRACK_OBJECT_GEOMETRY = "pyflextrkr_olr_to_tb_detect_and_spread"
ORGANISED_ENVELOPE_MIN_RAW_AREA_KM2 = 5000.0


def epoch_seconds(time_values):
    stamps = pd.to_datetime(np.asarray(time_values))
    return np.asarray(
        (stamps - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1),
        dtype=np.int64,
    )


def sha256_file(path, block_size=1024 * 1024):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(block_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def sha256_label_sequence(labels_by_time):
    digest = hashlib.sha256()
    for labels in labels_by_time:
        values = np.ascontiguousarray(labels, dtype=np.int32)
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        digest.update(values.tobytes())
    return digest.hexdigest()


def time_signature(time_values):
    seconds = epoch_seconds(time_values)
    digest = hashlib.sha256(np.ascontiguousarray(seconds).tobytes()).hexdigest()
    return {
        "time_count": int(seconds.size),
        "first_basetime": int(seconds[0]) if seconds.size else None,
        "last_basetime": int(seconds[-1]) if seconds.size else None,
        "basetimes_sha256": digest,
    }


def product_fingerprint(metadata):
    keys = (
        "product_version",
        "source",
        "simulation",
        "analysis_role",
        "classification_cache_version",
        "time_count",
        "first_basetime",
        "last_basetime",
        "basetimes_sha256",
        "cloudid_labels_sha256",
        "tracknumbers_sha256",
        "track_min_overlap_fraction",
        "track_min_area_km2",
        "min_track_observations",
        "min_track_duration_hours",
        "track_acceptance_criteria",
        "embedded_convective_core_requirement",
        "object_geometry",
        "envelope_min_raw_area_km2",
        "pyflextrkr_tb_core_k",
        "pyflextrkr_tb_cold_k",
        "pyflextrkr_tb_warm_k",
        "pyflextrkr_tb_cloud_k",
        "pyflextrkr_tb_smoothing_cells",
        "pyflextrkr_min_cold_core_pixels",
        "pyflextrkr_warm_anvil_expansion",
        "track_manifest_sha256",
        "pyflextrkr_nmaxlinks",
    )
    payload = {key: metadata.get(key) for key in keys}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def metadata_path(work_path):
    return os.path.join(work_path, PYFLEXTRKR_METADATA_FILENAME)


def track_manifest_path(work_path):
    return os.path.join(work_path, PYFLEXTRKR_TRACK_MANIFEST_FILENAME)


def read_track_manifest(work_path):
    path = track_manifest_path(work_path)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


def manifest_accepted_mask(manifest):
    values = manifest["accepted"]
    if values.dtype.kind == "b":
        return values.astype(bool)
    normalized = values.astype(str).str.strip().str.lower()
    invalid = ~normalized.isin({"true", "false", "1", "0"})
    if invalid.any():
        raise ValueError("organised-track manifest contains invalid accepted values")
    return normalized.isin({"true", "1"})


def read_product_metadata(work_path):
    path = metadata_path(work_path)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_product_metadata(work_path, metadata):
    os.makedirs(work_path, exist_ok=True)
    payload = dict(metadata)
    payload["product_fingerprint"] = product_fingerprint(payload)
    path = metadata_path(work_path)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(tmp_path, path)
    return payload
