"""Filtered pressure-work energy flux on native geometric-height levels."""
from __future__ import annotations

import glob
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr
from scipy import fft

from diagnostics.analysis_config import simulation_cache_name
from diagnostics.cache_support import cache_manifest, materialize, read_json_file, write_json_file, write_netcdf_atomic
from diagnostics.diagnostic_helpers import progress
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.tem_core import TEM_VERSION
from diagnostics.waves.publication_wave_forcing import _cached_product, _native_profile_stage, _read_staged_height
from diagnostics.waves.regional_epfd_spectrum import control_region_weights, mechanism_latitude_indices
from diagnostics.waves.sector_wind_variance import SECTOR_NAMES, variance_masks
from diagnostics.waves.wave_uncertainty import PressureSampler, native_height_slice

VERSION = "wave-pressure-work-v1-filter-both-fields"
PRESSURES = np.arange(70., 351., 10.)
LATITUDES = np.r_[.25, np.arange(.5, 25.1, .5)]


def load_energy_flux_height_profile(*, control_archive_dir, start, end, tem_root=None):
    """Define one geometric-height mapping for both simulations from the Control TEM archive."""
    upstream = load_stored_tem(Path(control_archive_dir).name, root=tem_root, start=start, end=end)
    try:
        pressure = upstream.pressure.sel(latitude=slice(-25.,25.))
        latitude_weights = np.cos(np.deg2rad(pressure.latitude))
        profile = materialize(pressure.mean("Time").weighted(latitude_weights).mean("latitude"))
    finally:
        upstream.close()
    if not np.all(np.isfinite(profile)) or not np.all(profile.values>0.) or not np.all(np.diff(profile.values)<0.):
        raise ValueError("The Control reference pressure must decrease with geometric height")
    return xr.DataArray(profile.z.values,dims="pres",coords=dict(pres=profile.values/100.),
        attrs=dict(units="m",definition="Control time and area mean pressure–height profile over 25S–25N")).sortby("pres")


def filtered_pressure_work(spatial, masks, longitude_count):
    """Return <p_b*v_b> and <p_b*w_b>, filtering both partners identically.

    Input axes are time, latitude, field=(pressure,v,w), nonnegative zonal mode.
    Output axes are sector, time, latitude, component=(meridional,upward).
    Values use Pa m/s = W/m²; density does not multiply pressure work.
    """
    nt, nlat, fields, modes = spatial.shape
    if fields != 3 or modes != longitude_count//2+1 or masks.shape[1:] != (nt, modes):
        raise ValueError("Pressure-work records and spectral masks differ")
    transformed = fft.fft(np.ascontiguousarray(spatial.transpose(1, 2, 3, 0)), axis=-1, workers=2)
    weights = np.full(modes, 2./longitude_count**2)
    weights[0] *= .5
    if longitude_count % 2 == 0:
        weights[-1] *= .5
    output = np.empty((len(masks), nt, nlat, 2))
    for band, mask in enumerate(masks):
        filtered = fft.ifft(transformed*mask.T[None, None], axis=-1, workers=2)
        for component in range(2):
            cross = (filtered[:, 0]*filtered[:, component+1].conj()).real
            output[band, :, :, component] = np.einsum("lkt,k->tl", cross, weights)
    return output


def compute_energy_flux(stage, background, longitude_count, fixed_weights, directory, signature, *,
        pressures=PRESSURES, latitudes=LATITUDES, latitude_indices=None):
    """Sample instantaneous fluxes before hemisphere pooling and time averaging."""
    nt, nz, nlat = [background.sizes[name] for name in ("Time", "z", "latitude")]
    shape = (nz, 3, nt, nlat, 2)
    path, state_path = directory/"energy_flux.dat", directory/"energy-progress.json"
    state = read_json_file(state_path) or {}
    resume = (path.exists() and state.get("shape") == list(shape)
        and state.get("manifest_sha256") == signature)
    values = np.memmap(path, dtype="float64", mode="r+" if resume else "w+", shape=shape)
    count = int(state.get("complete_heights", 0)) if resume else 0
    masks = variance_masks(nt, longitude_count)
    try:
        for level in range(count, nz):
            started = perf_counter()
            plane = _read_staged_height(stage, level)
            if latitude_indices is not None:
                plane = plane[:, latitude_indices]
            for left in range(0, nlat, 8):
                values[level, :, :, left:left+8] = filtered_pressure_work(plane[:, left:left+8], masks, longitude_count)
            del plane
            values.flush()
            write_json_file(state_path, dict(shape=list(shape), complete_heights=level+1, manifest_sha256=signature))
            progress(f"Wave-energy flux: height {level+1}/{nz} complete ({perf_counter()-started:.1f} s)")
        flux = xr.DataArray(np.asarray(values), dims=("z", "sector", "Time", "latitude", "component"),
            coords=dict(z=background.z, sector=list(SECTOR_NAMES), Time=background.Time,
                latitude=background.latitude, component=["meridional", "upward"]))
        sampler = PressureSampler(background, pressures, latitudes, fixed_weights)
        physical = []
        for band in SECTOR_NAMES:
            local = flux.sel(sector=band, drop=True)
            meridional = local.sel(component="meridional", drop=True)
            upward = local.sel(component="upward", drop=True)
            physical.append(xr.Dataset({name: sampler.samples(field, parity).mean("Time")
                for name, field, parity in (("poleward", meridional, "odd"), ("upward", upward, "even"))}))
        sector = xr.IndexVariable("sector", list(SECTOR_NAMES))
        physical = xr.concat(physical, dim=sector)
        product = xr.Dataset(dict(energy_flux_poleward=physical.poleward, energy_flux_upward=physical.upward))
        for name in ("energy_flux_poleward", "energy_flux_upward"):
            product[name].attrs.update(units="W m-2", definition="zonal mean of identically band-filtered pressure and velocity")
        return product.load()
    finally:
        values._mmap.close()


def load_or_build_energy_flux(*, simulation, archive_dir, control_archive_dir, cache_dir, pattern,
        start, end, tem_root=None, figures_only=False, rebuild=False, scratch_root=None):
    """Reuse the complete-record native reader and the maintained wave cache tree."""
    path = Path(cache_dir)/"epfd"/f"wave_energy_flux_{simulation_cache_name(simulation)}.nc"
    sources = sorted(glob.glob(str(Path(archive_dir)/pattern)))
    sources += tem_source_paths(Path(archive_dir).name, root=tem_root)+tem_source_paths(Path(control_archive_dir).name, root=tem_root)
    manifest = cache_manifest(__name__, schema=VERSION,
        configuration=dict(simulation=simulation, start=str(start), end=str(end), pressures=PRESSURES,
            latitudes=LATITUDES, latitude_limit=26., latitude_stride_outside_box=4,
            bands=SECTOR_NAMES, stage_dtype="complex64", quantity="filtered pressure-work energy flux"),
        dependencies=dict(tem_version=TEM_VERSION), source_paths=sources)
    if not rebuild:
        cached = _cached_product(path, manifest, figures_only)
        if cached is not None:
            progress(f"{simulation}: using cached wave-energy flux: {path}")
            return cached[["energy_flux_poleward", "energy_flux_upward"]]
    if figures_only:
        raise FileNotFoundError(f"Missing wave-energy flux cache: {path}; calculate Figure 21 without --figures-only")
    upstream = load_stored_tem(Path(archive_dir).name, root=tem_root, start=start, end=end)
    control = load_stored_tem(Path(control_archive_dir).name, root=tem_root, start=start, end=end)
    stage = None
    try:
        fixed = control_region_weights(materialize(control[["rho_zm", "pressure"]].isel(
            latitude=mechanism_latitude_indices(control.latitude))))
        background = materialize(upstream[["pressure"]].sel(latitude=slice(-26., 26.)))
        positive = background.latitude.values[background.latitude.values > 0]
        retained = np.unique(np.r_[positive[::4], positive[-1], positive[(positive >= 3.5) & (positive <= 16.5)]])
        background = background.sel(latitude=np.sort(np.r_[-retained, retained]))
        background = background.isel(z=native_height_slice(background.pressure, bottom=350., top=70.))
        progress(f"{simulation}: pressure-work energy flux; read pressure, v, and w only")
        stage, nlon, directory = _native_profile_stage(archive_dir, pattern, start, end, upstream, background,
            manifest, scratch_root or Path(__file__).resolve().parents[2]/".epfd_scratch"/"wave-energy-flux",
            fields=("pressure", "v", "w"), stage_dtype="complex64")
        product = compute_energy_flux(stage, background, nlon, fixed, directory, manifest["manifest_sha256"])
        product.attrs.update(product_version=VERSION, simulation=simulation, manifest_sha256=manifest["manifest_sha256"],
            cache_manifest=json.dumps(manifest, sort_keys=True),
            definition="pressure-work energy flux; zonal mean; native geometric height; filter both factors before multiplication",
            hemisphere_weighting="simultaneous hemispheres; fixed Control mass; poleward meridional sign")
        write_netcdf_atomic(product, path)
        stage._mmap.close(); stage = None
        for filename in ("fourier.dat", "progress.json", "energy_flux.dat", "energy-progress.json"):
            (directory/filename).unlink(missing_ok=True)
        for filename in directory.glob("reader*.json"):
            filename.unlink()
        directory.rmdir()
        progress(f"{simulation}: wave-energy flux cache complete: {path}")
        return product
    finally:
        if stage is not None:
            stage._mmap.close()
        upstream.close(); control.close()
