"""Shared cache validation, loading, materialisation, and atomic writes."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from pathlib import Path

import dask
import numpy as np
import pandas as pd
import xarray as xr

from diagnostics.analysis_config import simulation_plot_label


def relabel_cached_simulations(product):
    """Apply current plot labels to a compact cached table or dataset."""

    def display(value):
        if not isinstance(value, str):
            return value
        for separator in (" - ", " − "):
            if separator in value:
                return separator.join(simulation_plot_label(part) for part in value.split(separator))
        return simulation_plot_label(value)

    names = ("simulation", "simulation_a", "simulation_b", "source_simulation", "comparison")
    if isinstance(product, pd.DataFrame):
        product = product.copy()
        for name in names:
            if name in product.columns:
                product[name] = product[name].map(display)
        return product
    coordinates = {}
    for name in names:
        if name in product.coords:
            coordinate = product.coords[name]
            values = np.asarray(coordinate.values)
            renamed = np.asarray([display(value) for value in values.ravel()]).reshape(values.shape)
            coordinates[name] = coordinate.copy(data=renamed)
    return product.assign_coords(coordinates) if coordinates else product


def jsonable_value(value):
    """Convert numpy and pandas values into stable JSON values."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (list, tuple, np.ndarray)):
        return [jsonable_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable_value(item) for key, item in value.items()}
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(value))
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def read_json_file(path):
    """Read JSON metadata and return ``None`` for a missing file."""

    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json_file(path, payload):
    """Write JSON metadata through an atomic same-directory replacement."""

    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(jsonable_value(payload), handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def source_file_signature(paths):
    """Return a stable size and modification-time source signature."""

    records = []
    for value in sorted({os.path.abspath(os.fspath(path)) for path in paths}):
        path = Path(value)
        if not path.is_file():
            raise FileNotFoundError(f"Missing cache source file: {path}")
        stat = path.stat()
        records.append(
            {"path": value, "size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)}
        )
    if not records:
        raise ValueError("Cache source-file collection is empty")
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":"))
    return {"sha256": hashlib.sha256(encoded.encode()).hexdigest(), "files": records}


def cache_manifest(
    producer,
    *,
    schema,
    configuration=None,
    dependencies=None,
    source_paths=None,
):
    """Build one manifest from scientific settings and dependencies."""

    manifest = {
        "producer": str(producer),
        "schema": str(schema),
        "configuration": jsonable_value(configuration or {}),
        "dependencies": jsonable_value(dependencies or {}),
    }
    if source_paths is not None:
        manifest["sources"] = source_file_signature(source_paths)
    encoded = json.dumps(manifest, sort_keys=True, separators=(",", ":"))
    manifest["manifest_sha256"] = hashlib.sha256(encoded.encode()).hexdigest()
    return manifest


def cache_manifest_matches(saved, expected, *, compare_sources=True):
    """Return whether cache identities match at the requested validation level."""

    if not isinstance(saved, dict) or not isinstance(expected, dict):
        return False
    if compare_sources:
        return jsonable_value(saved) == jsonable_value(expected)
    identity_keys = ("producer", "schema", "configuration", "dependencies")
    return all(
        jsonable_value(saved.get(key)) == jsonable_value(expected.get(key))
        for key in identity_keys
    ) and isinstance(saved.get("manifest_sha256"), str)


def metadata_reader(path_resolver):
    """Bind a cache-name path resolver to the shared JSON reader."""

    return lambda name: read_json_file(path_resolver(name))


def metadata_writer(path_resolver, enabled=True):
    """Bind a cache-name path resolver to the shared JSON writer."""

    if not enabled:
        return lambda name, payload: None
    return lambda name, payload: write_json_file(path_resolver(name), payload)


def netcdf_safe_attribute(value):
    """Convert metadata to values accepted by NetCDF backends."""

    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, bool):
        return np.int8(value)
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(jsonable_value(value), sort_keys=True)
    if isinstance(value, (list, tuple, np.ndarray)):
        array = np.asarray(value)
        if array.dtype.kind == "b":
            return array.astype(np.int8)
        if array.dtype.kind in "SUif":
            return array
        return json.dumps(jsonable_value(value), sort_keys=True)
    if isinstance(value, (str, int, float)):
        return value
    return str(value)


def netcdf_safe_attrs(attrs):
    """Return metadata containing only NetCDF-compatible values."""

    return {key: netcdf_safe_attribute(value) for key, value in attrs.items()}


def netcdf_safe_dataset(dataset):
    """Return a shallow copy with NetCDF-compatible metadata."""

    output = dataset.copy(deep=False)
    output.attrs = netcdf_safe_attrs(dataset.attrs)
    for name in output.variables:
        output[name].attrs = netcdf_safe_attrs(dataset[name].attrs)
    return output


def materialize(value, *, scheduler="threads", num_workers=2):
    """Finish a deferred calculation before an output file is opened."""

    settings = {"scheduler": scheduler}
    if scheduler in {"threads", "processes"} and num_workers is not None:
        settings["num_workers"] = int(num_workers)
    with dask.config.set(settings):
        return value.load()


def load_netcdf(path, **open_kwargs):
    """Load one NetCDF file into memory and close the file immediately."""

    with xr.open_dataset(path, **open_kwargs) as dataset:
        return dataset.load()


def write_netcdf_atomic(
    dataset,
    path,
    *,
    encoding=None,
    unlimited_dims=None,
    materialize_first=True,
    scheduler="threads",
    num_workers=2,
):
    """Write one complete NetCDF file without exposing partial output."""

    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if materialize_first:
        dataset = materialize(
            dataset,
            scheduler=scheduler,
            num_workers=num_workers,
        )
    dataset = netcdf_safe_dataset(dataset)
    temporary = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        dataset.to_netcdf(
            temporary,
            encoding=encoding,
            unlimited_dims=unlimited_dims,
        )
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return dataset


def write_csv_atomic(table, path, **to_csv_kwargs):
    """Write one complete CSV file without exposing partial output."""

    path = os.fspath(path)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = f"{path}.{uuid.uuid4().hex}.tmp"
    try:
        table.to_csv(temporary, **to_csv_kwargs)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
