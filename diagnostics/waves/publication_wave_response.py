"""Native equatorial sampling and the 400 hPa extension for Figure 22."""
from __future__ import annotations

import glob
import json
from pathlib import Path

import numpy as np
import xarray as xr

from diagnostics.analysis_config import simulation_cache_name
from diagnostics.cache_support import cache_manifest, materialize, write_netcdf_atomic
from diagnostics.diagnostic_helpers import progress
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.tem_core import TEM_VERSION
from diagnostics.waves.publication_wave_forcing import (
    BACKGROUND_FIELDS, _cached_product, _native_profile_stage)
from diagnostics.waves.regional_epfd_spectrum import control_region_weights, mechanism_latitude_indices
from diagnostics.waves.sector_transport import native_sector_fields
from diagnostics.waves.sector_wind_variance import compute_variance_sections
from diagnostics.waves.wave_band_response import BANDS, PRESSURES as BAND_PRESSURES, LATITUDES as BAND_LATITUDES, _high_covariances
from diagnostics.waves.wave_energy_flux import compute_energy_flux
from diagnostics.waves.wave_uncertainty import PressureSampler, native_height_slice

VERSION = "publication-wave-response-v1-native-equator-400hpa"
PRESSURES = np.unique(np.r_[BAND_PRESSURES, np.arange(360., 401., 10.)])
LATITUDES = np.unique(np.r_[0., .25, .5, .75, BAND_LATITUDES[BAND_LATITUDES <= 25.]])
LOWER_PRESSURES = np.arange(350., 401., 10.)


def forcing_cache_path(cache_dir, simulation):
    return Path(cache_dir)/"epfd"/f"publication_band_forcing_{simulation_cache_name(simulation)}.nc"


def variance_cache_path(cache_dir, simulation):
    return Path(cache_dir)/"epfd"/f"publication_wave_variance_{simulation_cache_name(simulation)}.nc"


def _save(product, path, manifest):
    product.attrs.update(product_version=VERSION, manifest_sha256=manifest["manifest_sha256"],
        cache_manifest=json.dumps(manifest, sort_keys=True),
        equator="interpolate each instantaneous pressure-sampled scalar across signed native latitude; one equatorial value")
    write_netcdf_atomic(product, path, encoding={name:dict(zlib=True, complevel=2) for name in product.data_vars})
    return product


def _remove_completed_stage(stage, directory):
    stage._mmap.close()
    for name in ("fourier.dat", "progress.json", "high_covariance.dat", "high_covariance.json",
            "variance.dat", "variance-progress.json", "energy_flux.dat", "energy-progress.json"):
        (directory/name).unlink(missing_ok=True)
    for path in directory.glob("reader*.json"):
        path.unlink()
    directory.rmdir()


def _forcing(*, simulation, upstream, fixed, covariance_path, band_path, path, manifest,
        archive_dir, pattern, start, end, scratch_root, analysis):
    """Reuse native covariances; read only heights missing below the old domain."""
    with xr.open_dataset(covariance_path) as low, xr.open_dataset(band_path) as old:
        background = materialize(upstream[list(BACKGROUND_FIELDS)].sel(
            latitude=low.latitude.where(abs(low.latitude) <= 26., drop=True)))
        background = background.isel(z=native_height_slice(background.pressure, bottom=400., top=70.))
        common = background.z.where(background.z.isin(old.z), drop=True)
        high = old.high_wavenumber_covariance.sel(z=common, latitude=background.latitude).load()
        missing = background.z.where(~background.z.isin(common), drop=True)
        stage = None
        try:
            if missing.size:
                stage_background = background.sel(z=missing)
                stage, nlon, directory = _native_profile_stage(archive_dir, pattern, start, end,
                    upstream, stage_background, manifest, scratch_root/"forcing", stage_dtype="complex64")
                extra = _high_covariances(stage, stage_background, nlon, directory, manifest["manifest_sha256"])
                extra = xr.DataArray(extra, dims=("Time", "z", "latitude", "covariance"),
                    coords=dict(Time=background.Time, z=missing, latitude=background.latitude, covariance=high.covariance))
                high = xr.concat([high, extra], dim="z").sortby("z")
            sampler = PressureSampler(background, PRESSURES, LATITUDES, fixed)
            fields = []
            for band in BANDS:
                values = high if band == "high_wavenumber" else materialize(low.pathway_covariance.sel(
                    pathway=band, z=background.z, latitude=background.latitude))
                native = native_sector_fields(values.transpose("Time", "z", "latitude", "covariance").values, background)
                fields.append(sampler.samples(native.EPFd)*86400.)
            forcing = xr.concat(fields, dim=xr.IndexVariable("sector", list(BANDS))).expand_dims(component=["total"])
            forcing.attrs.update(units="m s-1 day-1",
                definition="instantaneous native TEM divergence; signed-latitude interpolation at the equator")
            product = xr.Dataset(dict(forcing=forcing, zonal_wind=sampler.samples(background.u_bar).mean("Time")))
            # Keep the original regional budget as an independent closure check.
            weights = fixed.sum("hemisphere").rename(region_pres="pres", region_latitude="absolute_latitude")
            for band, names in zip(BANDS, (("low_eastward",), ("low_westward",),
                    ("high_wavenumber_eastward", "high_wavenumber_westward"))):
                regional = (forcing.sel(sector=band, component="total", pres=weights.pres,
                    absolute_latitude=weights.absolute_latitude)*weights).sum(("pres", "absolute_latitude")).mean("Time")
                expected = analysis.sector_forcing.sel(comparison=simulation, sector=list(names)).sum("sector").sel(component="total")
                np.testing.assert_allclose(regional, expected, atol=2e-7, rtol=2e-6)
            _save(product, path, manifest)
            if stage is not None:
                _remove_completed_stage(stage, directory)
                stage = None
            return product
        finally:
            if stage is not None:
                stage._mmap.close()


def _variance(*, upstream, fixed, path, manifest, archive_dir, pattern, start, end, scratch_root):
    # Preserve the earlier variance stencil, with native neighbours for the equator.
    positive = upstream.latitude.values[upstream.latitude.values > 0.]
    latitude = positive[positive <= 46.]
    latitude = np.unique(np.r_[latitude[::4], latitude[(latitude >= 3.5) & (latitude <= 16.5)],
        positive[positive <= 1.26]])
    latitude = latitude[latitude <= 27.]
    background = materialize(upstream[["pressure", "u_bar"]].sel(latitude=np.sort(np.r_[-latitude, latitude])))
    background = background.isel(z=native_height_slice(background.pressure, bottom=400., top=70.))
    stage, nlon, directory = _native_profile_stage(archive_dir, pattern, start, end, upstream,
        background, manifest, scratch_root/"variance", fields=("u", "v"), stage_dtype="complex64")
    try:
        product = compute_variance_sections(stage, background, nlon, fixed, directory,
            pressures=PRESSURES, latitudes=LATITUDES)[["horizontal_wind_variance", "horizontal_wind_variance_samples"]]
        _save(product, path, manifest)
    except BaseException:
        stage._mmap.close()
        raise
    _remove_completed_stage(stage, directory)
    return product


def _lower_activity(*, upstream, fixed, path, manifest, archive_dir, pattern, start, end, scratch_root):
    # Retain the original energy-flux latitude stencil.
    positive = upstream.latitude.values[upstream.latitude.values > 0.]
    energy_latitude = positive[positive <= 26.]
    energy_latitude = np.unique(np.r_[energy_latitude[::4], energy_latitude[-1],
        energy_latitude[(energy_latitude >= 3.5) & (energy_latitude <= 16.5)]])
    energy_latitude = np.sort(np.r_[-energy_latitude, energy_latitude])
    background = materialize(upstream[["pressure", "u_bar"]].sel(latitude=energy_latitude))
    background = background.isel(z=native_height_slice(background.pressure, bottom=400., top=350.))
    stage, nlon, directory = _native_profile_stage(archive_dir, pattern, start, end, upstream,
        background, manifest, scratch_root/"lower", fields=("pressure", "v", "w"), stage_dtype="complex64")
    try:
        product = compute_energy_flux(stage, background, nlon, fixed, directory,
            manifest["manifest_sha256"], pressures=LOWER_PRESSURES)
        _save(product, path, manifest)
    except BaseException:
        stage._mmap.close()
        raise
    _remove_completed_stage(stage, directory)
    return product


def load_or_build_publication_response(*, simulation, archive_dir, control_archive_dir, cache_dir, pattern,
        start, end, covariance_path, analysis, base_energy,
        tem_root=None, figures_only=False, rebuild=False):
    """Add actual equatorial samples and lower levels through maintained native readers."""
    name = simulation_cache_name(simulation)
    root = Path(cache_dir)/"epfd"
    band_path = root/f"wave_band_response_{name}.nc"
    sources = sorted(glob.glob(str(Path(archive_dir)/pattern)))+[Path(covariance_path), band_path]
    sources += tem_source_paths(Path(archive_dir).name, root=tem_root)
    sources += tem_source_paths(Path(control_archive_dir).name, root=tem_root)
    products, manifests, paths = {}, {}, {}
    for task in ("forcing", "variance", "lower"):
        paths[task] = forcing_cache_path(cache_dir, simulation) if task == "forcing" else root/f"publication_wave_{task}_{name}.nc"
        manifests[task] = cache_manifest(__name__, schema=VERSION, configuration=dict(
            task=task, simulation=simulation, start=str(start), end=str(end), pressures=PRESSURES,
            latitudes=LATITUDES, stage_dtype="complex64", equator="signed native interpolation before time averaging"),
            dependencies=dict(tem_version=TEM_VERSION), source_paths=sources)
        variables = {"forcing": ("zonal_wind",), "variance": ("horizontal_wind_variance",)}
        products[task] = None if rebuild else _cached_product(paths[task], manifests[task], figures_only,
            variables=variables.get(task) if figures_only else None)
    if any(value is None for value in products.values()):
        if figures_only:
            raise FileNotFoundError("Calculate Figure 22 without --figures-only to add the equator and 400 hPa")
        upstream = load_stored_tem(Path(archive_dir).name, root=tem_root, start=start, end=end)
        control = load_stored_tem(Path(control_archive_dir).name, root=tem_root, start=start, end=end)
        try:
            fixed = control_region_weights(materialize(control[["rho_zm", "pressure"]].isel(
                latitude=mechanism_latitude_indices(control.latitude))))
            scratch = Path(__file__).resolve().parents[2]/".epfd_scratch"/"publication-wave-response"
            for task, builder in (("forcing", _forcing), ("variance", _variance), ("lower", _lower_activity)):
                if products[task] is not None:
                    continue
                progress(f"{simulation}: Figure 22 native {task} calculation")
                arguments = dict(upstream=upstream, fixed=fixed, path=paths[task], manifest=manifests[task],
                    archive_dir=archive_dir, pattern=pattern, start=start, end=end, scratch_root=scratch)
                if task == "forcing":
                    arguments.update(simulation=simulation, covariance_path=covariance_path, band_path=band_path, analysis=analysis)
                products[task] = builder(**arguments)
        finally:
            upstream.close()
            control.close()
    lower = products["lower"]
    energy_names = list(base_energy.data_vars)
    lower_energy = lower[energy_names].sel(absolute_latitude=base_energy.absolute_latitude)
    energy = xr.concat([base_energy.where(base_energy.pres < 350., drop=True), lower_energy], dim="pres").sortby("pres")
    result = products["forcing"].copy(deep=False)
    if "forcing" in result:
        result.forcing.attrs.update(units="m s-1 day-1")
    result.zonal_wind.attrs.update(units="m s-1", definition="time and longitude mean zonal velocity")
    for variable in products["variance"].data_vars:
        result[variable] = products["variance"][variable]
    for variable in energy_names:
        result[variable] = energy[variable].interp(pres=result.pres, absolute_latitude=result.absolute_latitude)
        if not bool(np.isfinite(result[variable].sel(absolute_latitude=slice(.25,25.))).all()):
            raise ValueError(f"Figure 22 {variable} must cover the vector domain through 400 hPa")
    if not bool(np.isfinite(result.horizontal_wind_variance).all()):
        raise ValueError("Figure 22 variance must cover the equator and 70–400 hPa")
    return result
