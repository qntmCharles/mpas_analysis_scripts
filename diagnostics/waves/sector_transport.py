"""Shared native-height spectral covariances for the retained wave figures.

The symmetric time-local covariance allocation is the inverse-transform form
of the existing weighted co-spectrum. It permits the same instantaneous TEM
operator, pressure sampling and boundary geometry at every spectral sector.
"""

from __future__ import annotations

import glob
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import xarray as xr
from scipy import fft

from diagnostics.analysis_config import simulation_cache_name
from diagnostics.cache_support import (cache_manifest, materialize, write_netcdf_atomic)
from diagnostics.diagnostic_helpers import progress
from diagnostics.mean_state import circulation_mechanism as mechanism
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.tem_core import EARTH_RADIUS, TEM_VERSION, epfd_flux_terms, flux_divergence
from diagnostics.waves.regional_epfd_spectrum import (control_region_weights, mechanism_latitude_indices,
    regional_native_weights, forcing_covariance_weights)
from diagnostics.waves.regional_ep_flux_sectors import (SECTOR_ANALYSIS_VERSION, SECTORS, LOW_SECTORS,
    positive_spatial_masks, FREQUENCY_CUTOFF, WAVENUMBER_CUTOFF)

COVARIANCE_VERSION = "spectral-sector-covariances-v1-symmetric-native-mass"
LATITUDE_LIMIT = 40.
RAW_FIELDS = ("u", "v", "w", "theta", "density")


def pathway_height_slice(pressure, top_hpa=50.):
    """Bracket the displayed pressure domain at every time and keep derivative halos."""
    values = pressure.transpose("Time", "z", "latitude").values
    levels = np.flatnonzero(np.any(np.isfinite(values) & (values >= top_hpa*100.), axis=(0, 2)))
    if not levels.size:
        raise ValueError("Native pressure does not cover the pathway plotting domain")
    # The first level above the pressure boundary needs its central neighbour.
    return slice(0, min(pressure.sizes["z"], int(levels[-1])+3))

def allocate_covariances(spatial, masks, longitude_count, modes):
    """Allocate instantaneous zonal mass covariances using self-adjoint filters.

    Input axes: time, point, field=(u,rho*v,rho*w,theta), positive spatial mode.
    Output axes: sector, time, point, covariance=(uv,uw,vtheta), with mass density
    retained. Half of each cross-frequency interaction goes to each partner.
    Sector sums recover the instantaneous unfiltered covariance, not only its
    time mean. Time-varying coefficients therefore retain exact spectral sums.
    """
    spatial = np.asarray(spatial, dtype=np.complex128)
    transformed = fft.fft(spatial, axis=0, workers=2)
    multiplicity = np.where((np.asarray(modes) == 0) | (2*np.asarray(modes) == longitude_count), 1., 2.)/longitude_count**2
    output = np.empty((len(masks), *spatial.shape[:2], 3))
    for index, mask in enumerate(masks):
        if not np.any(mask):
            output[index] = 0.
            continue
        filtered = fft.ifft(transformed*mask[:, None, None, :], axis=0, workers=2)
        for pair, (left, right) in enumerate(((0, 1), (0, 2), (1, 3))):
            cross = .5*(filtered[:, :, left]*spatial[:, :, right].conj()
                + spatial[:, :, left]*filtered[:, :, right].conj()).real
            output[index, :, :, pair] = np.einsum("tpk,k->tp", cross, multiplicity)
    return output

def native_sector_fields(covariances, background):
    """Apply the shared instantaneous HPE(z) operator before every reduction."""
    density = background.rho_zm
    covariance = xr.Dataset({name: xr.DataArray(covariances[..., index], dims=("Time", "z", "latitude"),
        coords={dim: background[dim] for dim in ("Time", "z", "latitude")})/density
        for index, name in enumerate(("uv", "uw", "vtheta"))})
    fluxes = epfd_flux_terms(covariance, background)
    result = flux_divergence(fluxes, density, "z", background.z.values)
    # Preserve additive means when stability masks affect one divergence stencil.
    valid = np.isfinite(result.EPFd)
    for name in ("EPFd_y", "EPFd_z"):
        result[name] = result[name].where(valid)
    result["ep_flux_meridional_normalized"] = fluxes.momentum_meridional+fluxes.heat_meridional
    result["ep_flux_vertical_normalized"] = fluxes.momentum_vertical+fluxes.heat_vertical
    normalizer = density*EARTH_RADIUS*np.cos(np.deg2rad(background.latitude))
    result["Fy"] = normalizer*result.ep_flux_meridional_normalized
    result["Fz"] = normalizer*result.ep_flux_vertical_normalized
    return result

def _regional_stencil(background, q):
    """Retain all nonzero adjoint weights, including derivative neighbours."""
    weights = forcing_covariance_weights(background, q)
    active = np.any(weights != 0, axis=(0, 1))
    rows = np.flatnonzero(active.any(axis=1))
    if rows.size == 0:
        raise ValueError("The mechanism box has no active spectral stencil")
    return slice(max(0, int(rows.min())-1), min(background.sizes["z"], int(rows.max())+2))

def _store_frame(low_stage, high_stage, index, frame, region_z, region_latitude, low_count, *, include_wind_variance=False):
    u, v, w, theta, rho = [frame[name].isel(Time=0).transpose("z", "latitude", "longitude").values.astype(float)
        for name in RAW_FIELDS]
    if not all(np.isfinite(value).all() for value in (u, v, w, theta, rho)) or not (rho > 0).all():
        raise ValueError("Spectral transport requires complete native longitude and time records")
    fields = np.stack([u, rho*v, rho*w, theta]+([v] if include_wind_variance else []), axis=2)
    fields -= fields.mean(axis=-1, keepdims=True)
    spatial = fft.rfft(fields[..., ::-1], axis=-1, workers=2)
    spatial[..., 0] = 0.
    low_stage[index] = spatial[..., :low_count]
    if high_stage is not None:
        high_stage[index] = spatial[region_z][:, region_latitude, :, low_count:]

def _spectral_covariance_pass(low_stage, high_stage, names, definitions, region_z, region_columns, nlon):
    """Keep low modes throughout the pathway domain and all modes near the box."""
    nt, nz, nlat, _, low_count = low_stage.shape
    pathway_names = list(LOW_SECTORS)+[item["name"] for item in definitions]
    selected = [names.index(name) for name in pathway_names]
    pathway = np.empty((len(selected), nt, nz, nlat, 3))
    region = np.zeros((len(names), nt, region_z.stop-region_z.start, len(region_columns), 3))
    _, low_masks = positive_spatial_masks(nt, nlon, np.arange(low_count), definitions)
    position = {int(column): index for index, column in enumerate(region_columns)}
    for level in range(nz):
        # One height plane bounds memory and avoids rereading raw fields.
        plane = np.array(low_stage[:, level])
        for left in range(0, nlat, 24):
            right = min(nlat, left+24)
            regional_block = region_z.start <= level < region_z.stop and any(column in position for column in range(left, right))
            indices = list(range(len(names))) if regional_block else selected
            allocated = allocate_covariances(plane[:, left:right], low_masks[indices], nlon, np.arange(low_count))
            pathway[:, :, level, left:right] = allocated[[indices.index(index) for index in selected]]
            if regional_block:
                for column in range(left, right):
                    if column in position:
                        region[:, :, level-region_z.start, position[column]] = allocated[:, :, column-left]
        progress(f"Sector time spectra: low modes, native height {level+1}/{nz}")
    if high_stage is not None:
        modes = np.arange(low_count, nlon//2+1)
        _, high_masks = positive_spatial_masks(nt, nlon, modes, definitions)
        for level in range(region.shape[2]):
            for left in range(0, len(region_columns), 8):
                right = min(len(region_columns), left+8)
                plane = np.array(high_stage[:, level, left:right])
                region[:, :, level, left:right] += allocate_covariances(plane, high_masks, nlon, modes)
            progress(f"Sector time spectra: remaining modes, regional height {level+1}/{region.shape[2]}")
    return pathway_names, pathway, region

def compute_covariance_cache(dataset, upstream, fixed_weights, raw_reader, analysis, definitions, *, scratch_root=None,
        covariance_path=None, covariance_signature=None):
    """Build all pathways and sector budgets in one sequential native-data pass."""
    from diagnostics.waves.ep_flux_decomposition import validate_spectral_grid
    if covariance_path is None:
        raise ValueError("A destination path is required for the covariance cache")
    validate_spectral_grid(dataset)
    if (not np.array_equal(dataset.Time, upstream.Time)
            or int(analysis.attrs["time_sample_count"]) != dataset.sizes["Time"]
            or int(analysis.attrs["longitude_sample_count"]) != dataset.sizes["longitude"]):
        raise ValueError("Stored TEM, regional spectra and native transport require the same complete record")
    latitude = upstream.latitude.where(abs(upstream.latitude) <= LATITUDE_LIMIT+1., drop=True)
    progress("Sector transport: loading instantaneous height-TEM coefficients; no raw histories yet")
    background = materialize(upstream[["pressure", "rho_zm", "u_z", "theta_z", "absolute_vorticity", "u_bar", "t"]].sel(latitude=latitude))
    background = background.isel(z=pathway_height_slice(background.pressure))
    region_columns = mechanism_latitude_indices(latitude)
    local = background.isel(latitude=region_columns)
    # This is the existing Control-weighted projection, not a new regional mean.
    local["EPFd"] = materialize(upstream.EPFd.sel(z=local.z, latitude=local.latitude))
    q, reference = regional_native_weights(local, fixed_weights)
    section = _regional_stencil(local, q)
    local, q = local.isel(z=section), q.isel(z=section)
    nt, nlon = dataset.sizes["Time"], dataset.sizes["longitude"]
    low_count = min(20, nlon//2)+1
    names, _ = positive_spatial_masks(nt, nlon, np.arange(low_count), definitions)
    low_shape = (nt, background.sizes["z"], latitude.size, 4, low_count)
    high_shape = (nt, local.sizes["z"], local.sizes["latitude"], 4, nlon//2+1-low_count)
    required = (np.prod(low_shape)+np.prod(high_shape))*16
    pathway_names = list(LOW_SECTORS)+[item["name"] for item in definitions]
    checkpoint = None
    if covariance_path is not None and Path(covariance_path).exists():
        with xr.open_dataset(covariance_path) as candidate:
            compatible = (candidate.attrs.get("product_version") == COVARIANCE_VERSION
                and candidate.attrs.get("manifest_sha256") == covariance_signature
                and list(candidate.pathway.values) == pathway_names
                and list(candidate.sector.values) == names
                and np.array_equal(candidate.Time, dataset.Time)
                and np.array_equal(candidate.z, background.z)
                and np.array_equal(candidate.latitude, latitude)
                and np.array_equal(candidate.region_z, local.z)
                and np.array_equal(candidate.region_latitude, local.latitude))
            if compatible:
                progress("Sector transport: reusing cached time-local spectral covariances; raw histories remain closed")
                checkpoint = candidate.load()
    if checkpoint is not None:
        pathway_covariance = checkpoint.pathway_covariance.values
        regional_covariance = checkpoint.regional_covariance.values
    else:
        root = Path(scratch_root or os.environ.get("MPAS_EPFD_SCRATCH_ROOT") or Path(__file__).resolve().parents[2]/".epfd_scratch")
        root.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(root).free < required+1024**3:
            raise OSError(f"Sector transport needs {required/1024**3:.1f} GiB temporary space under {root}")
        progress(f"Sector transport CALCULATION: one native pass; {background.sizes['z']} native heights; all {nt} times; "
            f"{required/1024**3:.1f} GiB temporary Fourier storage")
        with TemporaryDirectory(prefix="sector-transport-", dir=root) as directory:
            low = np.memmap(Path(directory)/"low.dat", dtype="complex128", mode="w+", shape=low_shape)
            high = (np.memmap(Path(directory)/"high.dat", dtype="complex128", mode="w+", shape=high_shape)
                if high_shape[-1] else None)
            try:
                frames = iter(raw_reader.iter_frames(latitude.values, output_height=background.z.values))
                for index in range(nt):
                    started = perf_counter()
                    progress(f"Sector native read {index+1}/{nt}: {dataset.Time.values[index]}")
                    frame = next(frames)
                    if not np.array_equal(frame.Time, dataset.Time.isel(Time=slice(index, index+1))):
                        raise ValueError("Spectral transport native and stored TEM timestamps differ")
                    _store_frame(low, high, index, frame, section, region_columns, low_count)
                    del frame
                    progress(f"Sector native sample {index+1}/{nt}: read and longitude transform complete ({perf_counter()-started:.1f} s)")
                if next(frames, None) is not None:
                    raise ValueError("The native transport reader supplies extra time samples")
                pathway_names, pathway_covariance, regional_covariance = _spectral_covariance_pass(
                    low, high, names, definitions, section, region_columns, nlon)
            finally:
                low._mmap.close()
                if high is not None:
                    high._mmap.close()
        if covariance_path is not None:
            checkpoint = xr.Dataset(dict(
                pathway_covariance=(("pathway", "Time", "z", "latitude", "covariance"), pathway_covariance),
                regional_covariance=(("sector", "Time", "region_z", "region_latitude", "covariance"), regional_covariance)),
                coords=dict(pathway=pathway_names, sector=names, Time=dataset.Time, z=background.z,
                    latitude=latitude, region_z=local.z.values, region_latitude=local.latitude.values,
                    covariance=["uv", "uw", "vtheta"]),
                attrs=dict(product_version=COVARIANCE_VERSION, manifest_sha256=covariance_signature,
                    covariance_units="mass-bearing covariance; divide by instantaneous rho_zm before TEM",
                    subset_definitions=json.dumps(definitions, sort_keys=True)))
            encoding = {name: dict(zlib=True, complevel=2, shuffle=True,
                chunksizes=(1, min(nt, 32), 1, min(field.shape[-2], 60), 3)) for name, field in checkpoint.items()}
    recovered = sum(float((native_sector_fields(regional_covariance[index], local).EPFd*q).sum(
        ("z", "latitude")).mean("Time")) for index, name in enumerate(names) if name in SECTORS)
    np.testing.assert_allclose(recovered, reference, atol=1e-11, rtol=2e-6)
    progress(f"Sector covariances: saving validated samples: {covariance_path}")
    write_netcdf_atomic(checkpoint, covariance_path, encoding=encoding)
    return checkpoint


def load_or_build_covariances(*, simulation, archive_dir, control_archive_dir, cache_dir, pattern,
        start, end, definitions, analysis, tem_root=None, figures_only=False, rebuild=False, scratch_root=None):
    """Reuse time-resolved covariances without retaining obsolete pathway plots."""
    from diagnostics.waves.ep_flux_decomposition import _open_epfd_raw_archive
    path = Path(cache_dir)/"epfd"/f"sector_covariances_{simulation_cache_name(simulation)}.nc"
    sources = sorted(glob.glob(os.path.join(archive_dir, pattern)))+tem_source_paths(Path(archive_dir).name, root=tem_root)
    sources += tem_source_paths(Path(control_archive_dir).name, root=tem_root)
    def manifest(saved_definitions):
        return cache_manifest(__name__, schema=COVARIANCE_VERSION,
            configuration=dict(simulation=simulation,start=str(start),end=str(end),definitions=saved_definitions,
                latitude_limit=LATITUDE_LIMIT,frequency_cutoff=FREQUENCY_CUTOFF,wavenumber_cutoff=WAVENUMBER_CUTOFF,
                pathway_pressure_domain_hpa=[50.,1000.],derivative_halo=1,
                region_latitude=mechanism.REGION_LATITUDE,region_pressure=mechanism.REGION_PRESSURE),
            dependencies=dict(tem_version=TEM_VERSION,sector_version=SECTOR_ANALYSIS_VERSION),source_paths=sources)
    if path.exists() and not rebuild:
        with xr.open_dataset(path) as product:
            saved = json.loads(product.attrs["subset_definitions"])
            selected = {item["name"]:item for item in saved}
            compatible = (product.attrs.get("product_version") == COVARIANCE_VERSION
                and product.attrs.get("manifest_sha256") == manifest(saved)["manifest_sha256"]
                and all(all(selected.get(item["name"], {}).get(key) == value
                    for key, value in item.items()) for item in definitions))
            if compatible:
                return path
    if figures_only:
        raise FileNotFoundError(f"Missing or stale covariance samples: {path}; calculate without --figures-only")
    upstream = load_stored_tem(Path(archive_dir).name, root=tem_root, start=start, end=end)
    control = load_stored_tem(Path(control_archive_dir).name, root=tem_root, start=start, end=end)
    try:
        box = materialize(control[["rho_zm","pressure"]].isel(latitude=mechanism_latitude_indices(control.latitude)))
        fixed = control_region_weights(box)
        dataset, reader = _open_epfd_raw_archive(archive_dir,pattern,start,end,upstream.z,requested_fields=RAW_FIELDS)
        try:
            compute_covariance_cache(dataset,upstream,fixed,reader,analysis,definitions,
                scratch_root=scratch_root,covariance_path=path,covariance_signature=manifest(definitions)["manifest_sha256"])
        finally:
            dataset.close()
    finally:
        control.close()
        upstream.close()
    return path
