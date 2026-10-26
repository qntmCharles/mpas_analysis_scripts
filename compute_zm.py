#!/usr/bin/env python
"""Generate unfolded zonal means with verified in-file isobaric coordinates.

Use the central simulation profile and spin-up filename by default.
Reuse the existing pressure-repair validation and native-height TEM helpers.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import dask
import numpy as np
import xarray as xr

import compute_tem_latlon_to_zm as tem
import fix_control_isobaric_levels as repair
from diagnostics import analysis_config as config
from diagnostics.archive_coordinates import ISOBARIC_DIM_LEVEL_VAR, standardize_latlon_dataset
from diagnostics.cache_support import (
    cache_manifest, cache_manifest_matches, read_json_file, write_json_file, write_netcdf_atomic,
)

ZM_VERSION = "zm-v2-verified-isobaric"
MODEL_VARIABLES = (
    "uReconstructZonal", "uReconstructMeridional", "theta", "relhum", "vorticity",
    "pressure", "temperature", "ke", "theta_nudge", "u_nudge", "cldfrac", "rho",
    "qv", "qc", "qi", "qr", "qs", "qg",
)


def verified_pressure_levels(path):
    """Resolve missing or misleading labels through the existing native-field checks."""
    settings = SimpleNamespace(
        time_slice=slice(None), lat_slice=slice(None), lon_slice=slice(None),
        detect_time_count=2, detect_lat_count=4, detect_lon_count=4,
        detect_vars=None, old_levels_hpa=None, target_levels_hpa=None, extrapolate=False,
    )
    diagnosis = repair.detect_latlon_pressure_levels(path, settings)
    if diagnosis is None:
        raise ValueError(f"Cannot verify isobaric levels in {path}")
    candidates, scores = diagnosis
    aggregate = repair.aggregate_candidate_scores(scores)
    ranked = sorted(((name, score) for name, score in aggregate.items() if np.isfinite(score["rms"])),
                    key=lambda item: item[1]["rms"])
    if not ranked:
        raise ValueError(f"No valid native-field comparison for isobaric levels in {path}")
    name, best = ranked[0]
    if len(ranked) > 1 and not repair.materially_worse_mapping(ranked[1][1], best):
        raise ValueError(f"Ambiguous isobaric pressure levels in {path}; inspect the existing pressure-repair diagnostic")
    for variable, choices in scores.items():
        selected = choices[name]
        finite = [score for score in choices.values() if np.isfinite(score["rms"])]
        if not finite or repair.materially_worse_mapping(selected, min(finite, key=lambda score: score["rms"])):
            raise ValueError(f"Isobaric pressure axes disagree for {variable} in {path}")
    return dict(candidates)[name]


def zonal_dataset(source, levels):
    """Retain native height, numeric time, and the verified stored pressure grid."""
    for dim, name in ISOBARIC_DIM_LEVEL_VAR.items():
        if dim in source.dims:
            if source.sizes[dim] != len(levels):
                raise ValueError(f"{dim} has a different isobaric grid; repair the source before averaging")
            source = source.assign({name: xr.DataArray(levels, dims=dim)})
    names = [name for name in (*repair.ZM_ISOBARIC_SPECS, *MODEL_VARIABLES)
             if name in source and "longitude" in source[name].dims]
    if not names:
        raise ValueError("No supported longitude-dependent fields are available")
    z_full = tem.zfull_from_latlon(source)
    height = xr.DataArray(0.5 * (z_full[:-1] + z_full[1:]), dims="z",
                          attrs=dict(units="m", long_name="Geometric layer midpoint height at the TEM reference column"))
    if "nVertLevels" in source.dims:
        source = source.rename(nVertLevels="z")
    if "z" in source.dims:
        if source.sizes["z"] != height.size:
            raise ValueError("Native model levels and zgrid midpoints differ")
        source = source.assign_coords(z=height)
    grid = standardize_latlon_dataset(source, needed_vars=names)
    result = xr.Dataset({name: tem.finite_or_nan(grid[name]).mean("longitude", skipna=True, keep_attrs=True)
                         for name in names})
    if "w" in source:
        result["w"] = tem.midlevel_vertical_velocity(source, height).mean("longitude", skipna=True, keep_attrs=True)
    result = result.assign_coords({tem.time_dim(source): source[tem.time_dim(source)]})
    result.attrs.update(source.attrs, zm_version=ZM_VERSION, hemispheric_fold="none",
                        pressure_coordinate="stored isobaric levels verified against native-column pressure interpolation",
                        zonal_mean="instantaneous longitude mean; mask nonfinite and MPAS fill values")
    return result


def producer_manifest(path):
    return cache_manifest("compute_zm", schema=ZM_VERSION, source_paths=[path],
                          configuration=dict(hemispheric_fold=False, model_variables=MODEL_VARIABLES,
                                             pressure_validation="existing native-field candidate comparison",
                                             detection_sample=[2, 4, 4]))


def write_zonal_mean(path, args):
    output = (args.output_dir or path.parent) / f"zm_{path.name}"
    metadata = Path(str(output) + ".metadata.json")
    manifest = producer_manifest(path)
    if args.resume and output.exists() and cache_manifest_matches(read_json_file(metadata), manifest):
        with tem.open_dataset(output) as stored:
            if stored.attrs.get("manifest_sha256") == manifest["manifest_sha256"]:
                print(f"Reusing completed zonal means: {output}", flush=True)
                return
    if output.exists() and not (args.overwrite or args.dry_run):
        raise FileExistsError(f"{output} already exists; use --overwrite to replace an old zonal-mean file")
    levels = verified_pressure_levels(path)
    if args.dry_run:
        print(f"Verified {path} -> {output}: {repair.format_levels(levels)} hPa", flush=True)
        return
    with tem.open_dataset(path) as source:
        source = source.chunk(tem.chunks_for(source, args))
        result = zonal_dataset(source, levels)
        result.attrs["manifest_sha256"] = manifest["manifest_sha256"]
        encoding = {name: dict(dtype="float64", zlib=True, complevel=1) for name in result.data_vars}
        with dask.config.set(scheduler="single-threaded"):
            write_netcdf_atomic(result, output, encoding=encoding, materialize_first=False)
    write_json_file(metadata, manifest)
    print(f"Stored zonal means: {output}", flush=True)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", nargs="?", type=Path, help="Optional source directory; default: configured simulation profile")
    parser.add_argument("--latlon-pattern", default=config.SPINUP_LATLON_FILENAME, help="Source filename or glob; default: configured spin-up file")
    parser.add_argument("--output-dir", type=Path, help="Output directory for a single source directory; default: beside each latlon file")
    parser.add_argument("--resume", action="store_true", help="Reuse completed products with current source provenance")
    parser.add_argument("--overwrite", action="store_true", help="Replace existing zonal-mean files")
    parser.add_argument("--dry-run", action="store_true", help="Validate pressure levels and show outputs without writing")
    parser.add_argument("--time-chunk", type=int, default=1)
    parser.add_argument("--latitude-chunk", type=int, default=30)
    parser.add_argument("--longitude-chunk", type=int, default=180)
    args = parser.parse_args(argv)
    if args.output_dir and not args.data_dir:
        parser.error("--output-dir requires one data_dir to prevent simulation filename collisions")
    if min(args.time_chunk, args.latitude_chunk, args.longitude_chunk) < 1:
        parser.error("Chunk sizes must be positive")
    return args


def main(argv=None):
    args = parse_args(argv)
    directories = ([args.data_dir] if args.data_dir else
                   [config.spinup_directory(name) for name in config.SIMULATION_ARCHIVES])
    paths = []
    for directory in directories:
        matches = sorted(directory.glob(args.latlon_pattern))
        if not matches:
            print(f"No source matches {directory / args.latlon_pattern}", flush=True)
        paths.extend(matches)
    if not paths:
        raise FileNotFoundError("No configured latlon sources are available")
    for path in paths:
        write_zonal_mean(path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
