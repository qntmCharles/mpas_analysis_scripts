"""Read versioned upstream TEM fields before downstream cache construction."""

from pathlib import Path
import glob
import os

import numpy as np
import xarray as xr

from diagnostics.analysis_config import SIMULATION_ROOT
from diagnostics.cache_support import read_json_file, source_file_signature
from diagnostics.diagnostic_helpers import load_latlon_dataset
from diagnostics.tem_core import (
    TEM_FORMULATION, FLUX_TERMS, HEIGHT_PATTERN, PRESSURE_PATTERN, PRESSURE_REQUIRED, TEM_VERSION,
    log_pressure_height,
)


def tem_archive_root(formulation=TEM_FORMULATION, root=None):
    if formulation not in {"log-pressure", "geometric-height"}:
        raise ValueError(f"Unknown TEM formulation: {formulation}")
    root = root or (os.environ.get("MPAS_TEM_HEIGHT_ROOT") if formulation == TEM_FORMULATION else None) or os.environ.get("MPAS_TEM_ARCHIVE_ROOT")
    return Path(root) if root else SIMULATION_ROOT


def tem_archive_pattern(directory, formulation=TEM_FORMULATION, root=None):
    """Use companion filenames, preserving explicit legacy archive roots."""
    if formulation == "log-pressure":
        return PRESSURE_PATTERN
    archive_root = tem_archive_root(formulation, root)
    if any((archive_root / directory).glob(HEIGHT_PATTERN)):
        return HEIGHT_PATTERN
    explicit_root = root or os.environ.get("MPAS_TEM_HEIGHT_ROOT") or os.environ.get("MPAS_TEM_ARCHIVE_ROOT")
    if explicit_root and archive_root != SIMULATION_ROOT:
        return "zm_latlon.2000-*.nc"
    return HEIGHT_PATTERN


def tem_source_paths(directory, formulation=TEM_FORMULATION, root=None, *, pattern=None):
    """Select production TEM files unless the caller supplies an explicit pattern."""
    explicit_pattern = pattern is not None
    pattern = pattern or tem_archive_pattern(directory, formulation, root)
    files = sorted(glob.glob(str(tem_archive_root(formulation, root) / directory / pattern)))
    if not explicit_pattern:
        # Spin-up archives use a separate time axis and never enter production
        # loading, validation, or downstream cache source signatures.
        files = [path for path in files if "_spinup" not in Path(path).stem]
    return [item for path in files for item in (path, path + ".metadata.json")]


def validate_tem_metadata(directory, formulation=TEM_FORMULATION, root=None, *, pattern=None):
    paths = tem_source_paths(directory, formulation, root, pattern=pattern)
    if not paths:
        raise FileNotFoundError(f"No stored {formulation} TEM fields in {tem_archive_root(formulation, root) / directory}. "
                                "Run compute_tem_latlon_to_zm.py first.")
    for path in paths[::2]:
        metadata = read_json_file(path + ".metadata.json")
        if not metadata or metadata.get("schema") != TEM_VERSION:
            raise ValueError(f"Missing or stale TEM completion metadata: {path}")
        if metadata.get("configuration", {}).get("formulation") != formulation:
            raise ValueError(f"TEM formulation differs: {path}")
        if metadata.get("configuration", {}).get("output_dtype") not in {"float32", "float64"}:
            raise ValueError(f"Stored TEM lacks a valid storage precision: {path}")
        if formulation == "log-pressure" and metadata.get("configuration", {}).get("output_dtype") != "float64":
            raise ValueError(f"Pressure TEM requires float64 storage: {path}; rerun the producer")
        if metadata.get("configuration", {}).get("hemispheric_fold", False):
            raise ValueError(f"Stored TEM fields require unfolded hemispheres: {path}")
        sources = metadata.get("sources", {})
        source_paths = [item["path"] for item in sources.get("files", [])]
        if not source_paths or source_file_signature(source_paths) != sources:
            raise ValueError(f"Stored TEM input provenance differs: {path}; rerun the producer")
        with xr.open_dataset(path, chunks=None) as ds:
            # Legacy in-place native archives preserve the original global attributes.
            companion = Path(path).name.startswith("tem_")
            if (companion or "manifest_sha256" in ds.attrs) and ds.attrs.get("manifest_sha256") != metadata.get("manifest_sha256"):
                raise ValueError(f"Stored TEM and completion metadata signatures differ: {path}")
            required = set(PRESSURE_REQUIRED)
            if formulation == "geometric-height":
                required.add("density_wind_covariance")
            if not required.issubset(ds):
                raise ValueError(f"Stored {formulation} TEM lacks required fields: {path}")
            if formulation == "log-pressure" and ds.pres.attrs.get("units") != "hPa":
                raise ValueError(f"Stored TEM pressure coordinate requires hPa: {path}")
            if ds.attrs.get("tem_formulation", formulation) != formulation:
                raise ValueError(f"Stored TEM global coordinate formulation differs: {path}")
            if ds.attrs.get("hemispheric_fold", "disabled") != "disabled":
                raise ValueError(f"Stored TEM requires unfolded hemispheres: {path}")
            if ds.pressure.dtype != np.dtype(metadata["configuration"]["output_dtype"]):
                raise ValueError(f"Stored TEM precision differs for pressure: {path}")
            for name in required - {"pressure", "valid_zonal_sample"}:
                if ds[name].attrs.get("tem_version") != TEM_VERSION:
                    raise ValueError(f"Stored TEM scientific version differs for {name}: {path}")
                if ds[name].attrs.get("tem_formulation") != formulation:
                    raise ValueError(f"Stored TEM coordinate formulation differs for {name}: {path}")
                if ds[name].attrs.get("hemispheric_fold", "disabled") != "disabled":
                    raise ValueError(f"Stored TEM field {name} requires unfolded hemispheres: {path}")
                if ds[name].dtype != np.dtype(metadata["configuration"]["output_dtype"]):
                    raise ValueError(f"Stored TEM precision differs for {name}: {path}")
            units = {"EPFd": "ms-2", "EPFd_y": "ms-2", "EPFd_z": "ms-2",
                     **{name: "ms-2" for name in (*FLUX_TERMS, "EPFd_momentum", "EPFd_heat")},
                     "vres": "ms-1", "wres": "ms-1", "u_bar": "ms-1",
                     "v_bar": "ms-1", "w_bar": "ms-1", "pressure": "Pa",
                     "rho_zm": "kgm-3", "Fy": "kgs-2", "Fz": "kgs-2",
                     "theta_bar": "K", "t": "K", "u_z": "s-1", "theta_z": "Km-1",
                     "absolute_vorticity": "s-1", "uv": "m2s-2", "uw": "m2s-2",
                     "vtheta": "Kms-1"}
            for name, expected in units.items():
                if ds[name].attrs.get("units", "").replace(" ", "") != expected:
                    raise ValueError(f"Stored TEM units differ for {name}: {path}")
    return paths


def load_stored_tem(directory, *, formulation=TEM_FORMULATION, root=None, start, end):
    paths = validate_tem_metadata(directory, formulation, root)
    pattern = tem_archive_pattern(directory, formulation, root)
    dataset = load_latlon_dataset(
        tem_archive_root(formulation, root) / directory, pattern,
        # TEM variables store complete latitude rows on the native vertical grid.
        # Full latitude rows avoid repeated HDF5 data-sieve reads.
        chunks={"Time": 16, "latitude": -1}, direct_chunks=True,
        chunk_after_open=True, start=start, end=end, source_paths=paths[::2],
    )
    # In-place native archives do not carry the TEM producer's global attributes.
    dataset.attrs.update(tem_version=TEM_VERSION, tem_formulation=formulation,
                         hemispheric_fold="disabled")
    return dataset


def height_coordinate_tem(dataset):
    """Expose stored TEM on its physical height coordinate without recalculation."""
    if dataset.EPFd.attrs.get("tem_version") != TEM_VERSION:
        raise ValueError("TEM plotting requires the current stored scientific version")
    if dataset.EPFd.attrs.get("hemispheric_fold", "disabled") != "disabled":
        raise ValueError("TEM plotting requires unfolded stored fields")
    formulation = dataset.EPFd.attrs.get("tem_formulation")
    if formulation not in {"log-pressure", "geometric-height"}:
        raise ValueError("TEM plotting requires a declared coordinate formulation")
    if dataset.attrs.get("tem_formulation", formulation) != formulation:
        raise ValueError("TEM global and field coordinate formulations differ")
    if formulation == "log-pressure":
        height = log_pressure_height(dataset.pres)
        dataset = dataset.rename(pres="z").assign_coords(z=height).sortby("z")
    dataset = dataset.assign(uReconstructZonal=dataset.u_bar,
                             uReconstructMeridional=dataset.v_bar)
    dataset.attrs["tem_formulation"] = formulation
    dataset.z.attrs.update(
        units="m", long_name=("log-pressure height" if formulation == "log-pressure"
                              else "geometric height"),
    )
    return dataset
