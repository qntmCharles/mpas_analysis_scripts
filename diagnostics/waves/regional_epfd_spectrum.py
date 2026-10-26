"""Additive signed EP-forcing spectra in the circulation mechanism box.

Use the native HPE(z) operator with instantaneous coefficients. Transfer the
regional quadrature through the interpolation and finite-difference operators
before computing symmetric co-spectra. This retains exact regional forcing
without a stationary-background approximation, detrending, or taper correction.
"""

from __future__ import annotations

import glob
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from time import perf_counter

import numpy as np
import xarray as xr
from scipy import fft

from diagnostics.analysis_config import simulation_cache_name
from diagnostics.cache_support import (
    cache_manifest, cache_manifest_matches, load_netcdf, materialize,
    read_json_file, write_json_file, write_netcdf_atomic,
)
from diagnostics.diagnostic_helpers import progress
from diagnostics.mean_state import circulation_mechanism as mechanism
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.tem_core import EARTH_RADIUS, FLUX_TERMS, TEM_FORMULATION, TEM_VERSION, differentiate_wrt
from diagnostics.tem_native import PREPARED_FIELDS
from diagnostics.waves.equatorial_wave_config import ALL_CACHED_FILTERS, WAVE_BOXES, WAVE_CONFIG_VERSION
from diagnostics.waves.wave_metrics import configured_wave_filter_mask

SPECTRUM_VERSION = "regional-epfd-spectrum-v1-native-instantaneous"
PHASE_SPEED_BIN_WIDTH = 5.
PHASE_SPEED_DISPLAY_EDGES = np.arange(-35.,36.,PHASE_SPEED_BIN_WIDTH)
PHASE_SPEED_DISPLAY_CENTRES = .5*(PHASE_SPEED_DISPLAY_EDGES[:-1]+PHASE_SPEED_DISPLAY_EDGES[1:])
COEFFICIENT_FIELDS = ("rho_zm", "u_z", "theta_z", "absolute_vorticity")
REFERENCE_FIELDS = ("pressure", "EPFd", *FLUX_TERMS, *COEFFICIENT_FIELDS)


def mechanism_latitude_indices(latitude):
    """Include interpolation brackets and one native derivative neighbour on each side."""
    latitude = np.asarray(latitude)
    positive = np.flatnonzero(latitude >= 0)
    absolute = latitude[positive]
    if absolute.size < 3 or absolute[0] > mechanism.REGION_LATITUDE[0] or absolute[-1] < mechanism.REGION_LATITUDE[1]:
        raise ValueError("Native latitudes must bracket the complete mechanism box")
    begin = max(0, np.searchsorted(absolute, mechanism.REGION_LATITUDE[0])-2)
    stop = min(absolute.size, np.searchsorted(absolute, mechanism.REGION_LATITUDE[1], side="right")+2)
    north = positive[begin:stop]
    return np.unique(np.r_[north, latitude.size-1-north])


def control_region_weights(control):
    """Use the mechanism diagnostic's mean control mass and exact box quadrature."""
    sample = mechanism.sample_mechanism_field
    rho = sample(control.rho_zm, control.pressure)
    height = sample(xr.zeros_like(control.pressure)+control.z, control.pressure)
    p_pa = height.region_pres*100.
    minus_z_p = -height.assign_coords(region_pres=p_pa).differentiate("region_pres", edge_order=2)
    minus_z_p = minus_z_p.assign_coords(region_pres=height.region_pres)
    mass = (rho*minus_z_p*np.cos(np.deg2rad(rho.region_latitude))).mean("Time", skipna=True)
    quadrature = mechanism._quadrature(p_pa.values)[:, None]*mechanism._quadrature(np.deg2rad(mass.region_latitude))
    weights = mass*xr.DataArray(quadrature, dims=("region_pres", "region_latitude"))
    if not np.isfinite(weights).all() or not (weights > 0).all():
        raise ValueError("The control mechanism box needs positive finite mass weights everywhere")
    weights /= weights.sum(("region_pres", "region_latitude"))
    weights /= weights.sizes["hemisphere"]
    return weights.transpose("hemisphere", "region_pres", "region_latitude")


def _interpolation_matrix(source, target):
    source, target = np.asarray(source), np.asarray(target)
    if source.size < 2 or np.any(np.diff(source) <= 0) or target.min() < source[0] or target.max() > source[-1]:
        raise ValueError("Interpolation requires ordered source coordinates bracketing the fixed box")
    left = np.clip(np.searchsorted(source, target, side="right")-1, 0, source.size-2)
    fraction = (target-source[left])/(source[left+1]-source[left])
    matrix = np.zeros((target.size, source.size))
    matrix[np.arange(target.size), left] = 1.-fraction
    matrix[np.arange(target.size), left+1] += fraction
    return matrix


def regional_native_weights(archive, fixed_weights):
    """Transpose the existing pressure/latitude sampling onto native grid points.

    Time masks reproduce pointwise skipna means on the mechanism grid. The
    returned weights precede the final ordinary time mean, so they include
    T/N_valid wherever samples are missing.
    """
    regional = mechanism.sample_mechanism_field(archive.EPFd, archive.pressure).transpose(
        "Time", "hemisphere", "region_pres", "region_latitude")
    valid = np.isfinite(regional.values)
    counts = valid.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("Every fixed mechanism grid point needs finite EP forcing samples")
    time_count = archive.sizes["Time"]
    target_weights = valid*fixed_weights.values[None]*time_count/counts[None]
    pressure = archive.pressure.transpose("Time", "z", "latitude").values/100.
    latitude = archive.latitude.values
    native = np.zeros_like(pressure, dtype=float)
    for hemisphere, sign in enumerate((-1., 1.)):
        columns = np.flatnonzero(sign*latitude >= 0)
        columns = columns[np.argsort(np.abs(latitude[columns]))]
        latitude_matrix = _interpolation_matrix(np.abs(latitude[columns]), mechanism.REGION_LATITUDES)
        column_weights = np.einsum("tpl,lj->tpj", target_weights[:, hemisphere], latitude_matrix)
        for local, column in enumerate(columns):
            if not np.any(column_weights[:, :, local]):
                continue
            for time in range(time_count):
                if not np.any(column_weights[time, :, local]):
                    continue
                p = pressure[time, :, column]
                levels = np.flatnonzero(np.isfinite(p) & (p > 0))
                levels = levels[np.argsort(p[levels])]
                matrix = _interpolation_matrix(p[levels], mechanism.REGION_PRESSURES)
                native[time, levels, column] += column_weights[time, :, local] @ matrix
    q = xr.DataArray(native, dims=("Time", "z", "latitude"),
                     coords={name: archive[name] for name in ("Time", "z", "latitude")})
    reference = float((regional.mean("Time", skipna=True)*fixed_weights).sum())
    reconstructed = float((xr.where(q != 0, archive.EPFd, 0.)*q).sum(("z", "latitude")).mean("Time"))
    np.testing.assert_allclose(reconstructed, reference, rtol=1e-10, atol=1e-13,
                               err_msg="Native quadrature differs from the mechanism region sampling")
    return q, reference


def forcing_covariance_weights(archive, region_weights):
    """Apply the transpose of the shared native divergence to regional weights."""
    z, latitude = archive.z.values, archive.latitude.values
    dz = differentiate_wrt(xr.DataArray(np.eye(z.size), dims=("z", "source_z"), coords={"z": z}), "z", z).values
    phi = np.deg2rad(latitude)
    dy = differentiate_wrt(xr.DataArray(np.eye(phi.size), dims=("latitude", "source_latitude"),
                                       coords={"latitude": latitude}), "latitude", phi).values
    q = region_weights.transpose("Time", "z", "latitude").values
    rho = archive.rho_zm.transpose("Time", "z", "latitude").values
    if np.any((q != 0) & (~np.isfinite(rho) | (rho <= 0))):
        raise ValueError("Finite positive density is required wherever regional weights are nonzero")
    q_over_rho = np.divide(q, rho, out=np.zeros_like(q), where=q != 0)
    cos2 = np.cos(phi)**2
    meridional = np.einsum("ij,tzi->tzj", dy, q_over_rho/(EARTH_RADIUS*cos2)) * cos2
    vertical = np.einsum("ij,til->tjl", dz, q_over_rho)
    shear = archive.u_z.transpose("Time", "z", "latitude").values
    stability = archive.theta_z.transpose("Time", "z", "latitude").values
    vorticity = archive.absolute_vorticity.transpose("Time", "z", "latitude").values
    heat_y = np.divide(meridional*shear, stability, out=np.zeros_like(q), where=meridional != 0)
    heat_z = np.divide(vertical*vorticity, stability, out=np.zeros_like(q), where=vertical != 0)
    weights = np.stack([-meridional, -vertical, heat_y, heat_z])
    if not np.isfinite(weights).all():
        raise ValueError("The forcing spectrum needs finite coefficients throughout the native derivative stencil")
    return weights


def signed_frequency_spectrum(positive_spatial, longitude_count):
    """Fold joint Fourier partners into positive frequency and signed eastward k."""
    time_count = positive_spatial.shape[-2]
    wavenumber = np.rint(fft.fftshift(fft.fftfreq(longitude_count)*longitude_count)).astype(int)
    frequency_index = np.arange(time_count//2+1)
    spectrum = np.empty((*positive_spatial.shape[:-2], frequency_index.size, longitude_count))
    for column, k in enumerate(wavenumber):
        times = frequency_index if k >= 0 else (-frequency_index) % time_count
        spectrum[..., :, column] = positive_spatial[..., times, abs(k)]
    weights = np.full(frequency_index.size, 2.)
    weights[0] = 1.
    if time_count % 2 == 0:
        weights[-1] = 1.
    spectrum *= weights[:, None]/(time_count*longitude_count)**2
    return wavenumber, fft.rfftfreq(time_count, d=.25), spectrum


def point_forcing_cospectra(spatial_coefficients, weights):
    """Symmetric co-spectra retain coefficient modulation without an arbitrary factor ordering."""
    coefficients = np.asarray(spatial_coefficients, dtype=np.complex128)  # field, time, k>=0
    transformed = fft.fft(coefficients, axis=1, workers=2)
    result = []
    for term, (left, right) in enumerate(((0, 1), (0, 2), (1, 3), (1, 3))):
        weighted_right = fft.fft(weights[term, :, None]*coefficients[right], axis=0, workers=2)
        weighted_left = fft.fft(weights[term, :, None]*coefficients[left], axis=0, workers=2)
        result.append(.5*(transformed[left]*weighted_right.conj()
                           + weighted_left*transformed[right].conj()).real)
    return np.stack(result)


def validate_regional_spectrum(product):
    if product.attrs.get("product_version") != SPECTRUM_VERSION or product.attrs.get("tem_formulation") != TEM_FORMULATION:
        raise ValueError("Regional forcing spectrum requires the current height-TEM method")
    required = {"forcing", "forcing_by_flux", "regional_epfd", "regional_flux_forcing", "closure_error", "filter_mask"}
    if not required.issubset(product):
        raise ValueError("Regional forcing spectrum is incomplete or nonfinite")
    for name in ("forcing", "forcing_by_flux", "regional_epfd", "regional_flux_forcing", "closure_error"):
        if not np.isfinite(product[name]).all():
            raise ValueError(f"Regional forcing spectrum contains nonfinite {name}")
    time_count = int(product.attrs.get("time_sample_count", 0))
    longitude_count = int(product.attrs.get("longitude_sample_count", 0))
    if time_count < 3 or longitude_count < 32:
        raise ValueError("Regional forcing spectrum lacks complete sampling metadata")
    expected_k = np.rint(fft.fftshift(fft.fftfreq(longitude_count)*longitude_count)).astype(int)
    if (not np.array_equal(product.wavenumber, expected_k)
            or not np.array_equal(product.frequency, fft.rfftfreq(time_count, d=.25))):
        raise ValueError("Regional forcing spectrum omits resolved Fourier bins")
    if list(product.flux_term.values) != list(FLUX_TERMS):
        raise ValueError("Regional spectrum flux terms differ")
    if list(product.wave_type.values) != list(ALL_CACHED_FILTERS):
        raise ValueError("Regional spectrum filter boundaries differ")
    np.testing.assert_allclose(product.forcing_by_flux.sum("flux_term"), product.forcing, rtol=1e-10, atol=1e-13)
    np.testing.assert_allclose(product.forcing.sum(), product.regional_epfd, rtol=1e-8, atol=1e-12,
                               err_msg="Spectral sum differs from stored regional EP forcing")
    np.testing.assert_allclose(product.forcing_by_flux.sum(("frequency", "wavenumber")),
                               product.regional_flux_forcing, rtol=1e-8, atol=1e-12)
    np.testing.assert_allclose(product.closure_error, product.forcing.sum()-product.regional_epfd,
                               rtol=1e-8, atol=1e-15)


def compute_regional_spectrum(dataset, upstream, fixed_weights, raw_reader, *, scratch_root=None):
    """Read only the box's native columns; retain all spatial and temporal modes."""
    if upstream.attrs.get("tem_formulation") != TEM_FORMULATION:
        raise ValueError("Regional spectra require stored geometric-height TEM")
    from diagnostics.waves.ep_flux_decomposition import validate_spectral_grid
    validate_spectral_grid(dataset)
    indices = mechanism_latitude_indices(upstream.latitude)
    archive = materialize(upstream[list(REFERENCE_FIELDS)].isel(latitude=indices))
    xr.align(archive, dataset.sel(latitude=archive.latitude), join="exact")
    q, reference = regional_native_weights(archive, fixed_weights)
    weights = forcing_covariance_weights(archive, q)
    point_z, point_lat = np.nonzero(np.any(weights != 0, axis=(0, 1)))
    if not point_z.size:
        raise ValueError("The fixed mechanism box contains no native forcing stencil")
    height_start, height_stop = int(point_z.min()), int(point_z.max())+1
    height = archive.z.isel(z=slice(height_start, height_stop))
    local_z = point_z-height_start
    point_weights = weights[:, :, point_z, point_lat]
    time_count, longitude_count = dataset.sizes["Time"], dataset.sizes["longitude"]
    mode_count = longitude_count//2+1
    shape = (point_z.size, time_count, 4, mode_count)
    stage_bytes = int(np.prod(shape))*np.dtype(np.complex128).itemsize
    root = Path(scratch_root or os.environ.get("MPAS_EPFD_SCRATCH_ROOT")
                or Path(__file__).resolve().parents[2]/".epfd_scratch")
    root.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(root).free < stage_bytes+512*1024**2:
        raise OSError(f"Regional EP spectrum needs {stage_bytes/1024**3:.1f} GiB temporary disk space under {root}")
    progress(f"Regional EP spectrum: {mechanism.REGION_DESCRIPTION}; {point_z.size} native stencil points; "
             f"{height.size} output heights; all {mode_count} spatial Fourier modes")
    progress(f"Regional EP spectrum: one native pass; {stage_bytes/1024**3:.2f} GiB temporary Fourier storage")
    total = np.zeros((len(FLUX_TERMS), time_count, mode_count))
    with TemporaryDirectory(prefix="regional-epfd-", dir=root) as temporary:
        stage = np.memmap(Path(temporary)/"spatial_modes.dat", dtype=np.complex128, mode="w+", shape=shape)
        try:
            frames = iter(raw_reader.iter_frames(archive.latitude.values, output_height=height.values))
            for time in range(time_count):
                started = perf_counter()
                progress(f"Regional EP native read: {time+1}/{time_count}; {dataset.Time.values[time]}")
                try:
                    frame = next(frames)
                except StopIteration as exc:
                    raise ValueError("Regional spectrum source ends before the requested record") from exc
                if not np.array_equal(frame.Time, dataset.Time.isel(Time=slice(time, time+1))):
                    raise ValueError("Regional spectrum source and TEM times differ")
                if not np.array_equal(frame.z, height) or not np.array_equal(frame.latitude, archive.latitude):
                    raise ValueError("Regional spectrum native reader returns a different grid")
                read_seconds = perf_counter()-started
                def point_field(name):
                    return frame[name].isel(Time=0).transpose("z", "latitude", "longitude").values[local_z, point_lat].astype(float)
                inputs = {name: point_field(name) for name in PREPARED_FIELDS}
                finite = np.logical_and.reduce([np.isfinite(values).all(axis=-1) for values in inputs.values()])
                finite &= (inputs["density"] > 0).all(axis=-1)
                if np.any(~finite & np.any(point_weights[:, time] != 0, axis=0)):
                    raise ValueError("Missing native eddy samples enter the regional forcing stencil")
                fields = np.stack([inputs["u"], inputs["density"]*inputs["v"],
                                   inputs["density"]*inputs["w"], inputs["theta"]])
                fields[:, ~finite] = 0.
                fields -= fields.mean(axis=-1, keepdims=True)
                spectrum = fft.rfft(fields[..., ::-1], axis=-1, workers=2)
                spectrum[..., 0] = 0.  # zonal-mean flow never contributes to the eddy flux
                stage[:, time] = spectrum.transpose(1, 0, 2)
                del frame, fields, inputs, spectrum
                progress(f"Regional EP sample {time+1}/{time_count}: read {read_seconds:.1f} s; "
                         f"Fourier calculation/store {perf_counter()-started-read_seconds:.1f} s")
            for point in range(point_z.size):
                total += point_forcing_cospectra(np.asarray(stage[point]).transpose(1, 0, 2), point_weights[:, :, point])
                if point == 0 or (point+1) % 25 == 0 or point+1 == point_z.size:
                    progress(f"Regional EP time spectra: {point+1}/{point_z.size} native points complete")
        finally:
            stage._mmap.close()
    k, frequency, contribution = signed_frequency_spectrum(total, longitude_count)
    component_reference = [float((xr.where(q != 0, archive[name], 0.)*q).sum(("z", "latitude")).mean("Time"))
                           for name in FLUX_TERMS]
    masks = np.stack([configured_wave_filter_mask(time_count, longitude_count, name)[k % longitude_count].T
                      for name in ALL_CACHED_FILTERS])
    product = xr.Dataset({
        "forcing_by_flux": (("flux_term", "frequency", "wavenumber"), contribution),
        "regional_epfd": xr.DataArray(reference),
        "regional_flux_forcing": ("flux_term", component_reference),
        "filter_mask": (("wave_type", "frequency", "wavenumber"), masks.astype("int8")),
    }, coords=dict(flux_term=list(FLUX_TERMS), frequency=frequency, wavenumber=k,
                   wave_type=list(ALL_CACHED_FILTERS)))
    product["forcing"] = product.forcing_by_flux.sum("flux_term")
    product["closure_error"] = product.forcing.sum()-product.regional_epfd
    for name in ("forcing", "forcing_by_flux", "regional_epfd", "regional_flux_forcing", "closure_error"):
        product[name].attrs.update(units="m s-2", sign_convention="eastward-positive acceleration")
    product.forcing.attrs["normalization"] = "acceleration contribution per discrete (k,frequency) bin; sum all bins"
    product.frequency.attrs.update(units="cycles day-1", long_name="ground-relative frequency")
    product.wavenumber.attrs.update(units="1", sign_convention="positive eastward; spatial Nyquist direction unresolved")
    product.attrs.update(
        product_version=SPECTRUM_VERSION, tem_formulation=TEM_FORMULATION, upstream_tem_version=TEM_VERSION,
        region=mechanism.REGION_DESCRIPTION, region_weighting="fixed mean control mass; equal hemisphere weights",
        temporal_preprocessing="none; stationary modes and temporal Nyquist retained",
        spectral_method="symmetric co-spectra with exact instantaneous regional/divergence weights",
        coefficient_treatment="instantaneous; no mean-background approximation",
        parity_treatment="native hemispheres before regional averaging; filter masks are footprints only",
        analysis_start=str(dataset.Time.values[0]), analysis_end=str(dataset.Time.values[-1]),
        time_sample_count=time_count, longitude_sample_count=longitude_count,
        reference="https://doi.org/10.1175/JAS-D-20-0249.1",
    )
    validate_regional_spectrum(product)
    progress(f"Regional EP spectrum complete: sum {float(product.forcing.sum())*86400:.6g} m/s/day; "
             f"closure error {float(product.closure_error)*86400:.3g} m/s/day")
    return product


def load_or_build_regional_spectrum(*, simulation, archive_dir, control_archive_dir, cache_dir, pattern,
                                    start, end, tem_root=None, figures_only=False, rebuild=False,
                                    tem_loader=load_stored_tem, raw_loader=None, scratch_root=None):
    """Cache the complete signed spectrum without rebuilding existing band products."""
    from diagnostics.waves.ep_flux_decomposition import _open_epfd_raw_archive
    path = Path(cache_dir)/"epfd"/f"regional_forcing_spectrum_{simulation_cache_name(simulation)}.nc"
    metadata = str(path)+".metadata.json"
    upstream_sources = tem_source_paths(Path(archive_dir).name, root=tem_root)
    control_sources = tem_source_paths(Path(control_archive_dir).name, root=tem_root)
    manifest = cache_manifest(__name__, schema=SPECTRUM_VERSION,
        configuration=dict(simulation=simulation, start=str(start), end=str(end),
                           region_latitude=mechanism.REGION_LATITUDE, region_pressure=mechanism.REGION_PRESSURE,
                           region_latitude_grid=mechanism.REGION_LATITUDES, region_pressure_grid=mechanism.REGION_PRESSURES,
                           region_weighting="fixed mean control mass; equal hemisphere weights",
                           control_archive=str(Path(control_archive_dir).resolve()),
                           preprocessing="none", filter_boxes=WAVE_BOXES),
        dependencies=dict(tem_version=TEM_VERSION, wave_config_version=WAVE_CONFIG_VERSION),
        source_paths=sorted(glob.glob(os.path.join(archive_dir, pattern)))+upstream_sources+control_sources)
    if not rebuild and path.exists() and cache_manifest_matches(read_json_file(metadata), manifest):
        try:
            product = load_netcdf(path)
            if product.attrs.get("manifest_sha256") != manifest["manifest_sha256"]:
                raise ValueError("Spectrum and metadata signatures differ")
            validate_regional_spectrum(product)
        except (OSError, ValueError, KeyError, AssertionError) as exc:
            progress(f"Ignoring invalid regional forcing spectrum {path}: {exc}")
        else:
            progress(f"{simulation}: using cached regional forcing spectrum: {path}")
            return product
    if figures_only:
        raise RuntimeError(f"Missing or stale regional forcing spectrum: {path}. Run Figure 17 without --figures-only.")
    progress(f"{simulation}: calculating the mechanism-box EP forcing spectrum")
    control = tem_loader(Path(control_archive_dir).name, formulation=TEM_FORMULATION, root=tem_root, start=start, end=end)
    upstream = None
    try:
        indices = mechanism_latitude_indices(control.latitude)
        control_fields = materialize(control[["pressure", "rho_zm"]].isel(latitude=indices))
        fixed_weights = control_region_weights(control_fields)
        upstream = (control if Path(archive_dir).resolve() == Path(control_archive_dir).resolve() else
                    tem_loader(Path(archive_dir).name, formulation=TEM_FORMULATION, root=tem_root, start=start, end=end))
        raw, reader = (raw_loader or _open_epfd_raw_archive)(archive_dir, pattern, start, end, upstream.z)
        try:
            product = compute_regional_spectrum(raw, upstream, fixed_weights, reader, scratch_root=scratch_root)
        finally:
            raw.close()
    finally:
        if upstream is not None and upstream is not control:
            upstream.close()
        control.close()
    product.attrs.update(simulation=simulation, manifest_sha256=manifest["manifest_sha256"])
    write_netcdf_atomic(product, path)
    write_json_file(metadata, manifest)
    progress(f"{simulation}: saved regional forcing spectrum: {path}")
    return product






def _forcing_values(forcing):
    """Require a complete ordered positive-frequency forcing spectrum."""
    field = forcing.transpose("frequency", "wavenumber")
    frequency = np.asarray(field.frequency, dtype=float)
    values = np.asarray(field, dtype=float)
    if (frequency.size < 2 or frequency[0] != 0 or not np.all(np.diff(frequency) > 0)
            or not np.isfinite(values).all()):
        raise ValueError("Forcing spectra require complete finite positive-frequency bins, including zero")
    return field, frequency, values


def signed_phase_speed_histogram(forcing, *, time_count, longitude_count, bin_width=PHASE_SPEED_BIN_WIDTH):
    """Sum signed forcing by ground-relative phase speed without renormalization."""
    if not np.isfinite(bin_width) or bin_width <= 0:
        raise ValueError("Phase-speed bins require a positive finite width")
    field, frequency, values = _forcing_values(forcing)
    k = np.asarray(field.wavenumber)
    stationary = np.broadcast_to(frequency[:, None] == 0, values.shape)
    unresolved = np.broadcast_to(k[None, :] == 0, values.shape).copy()
    if longitude_count % 2 == 0:
        unresolved |= np.broadcast_to(2*abs(k[None, :]) == longitude_count, values.shape)
    if time_count % 2 == 0:
        nyquist_frequency = (time_count//2)*frequency[1]
        unresolved |= np.broadcast_to(np.isclose(frequency[:, None], nyquist_frequency, rtol=1e-12, atol=1e-12), values.shape)
    unresolved &= ~stationary
    travelling = ~(stationary | unresolved)
    speed = np.divide(2*np.pi*EARTH_RADIUS/86400.*frequency[:, None], k[None, :],
                      out=np.zeros_like(values, dtype=float), where=k[None, :] != 0)
    extent = max(1, int(np.ceil(np.max(abs(speed[travelling]), initial=0.)/bin_width)))
    edges = np.arange(-extent, extent+1)*bin_width
    summed, _ = np.histogram(speed[travelling], bins=edges, weights=values[travelling])
    product = xr.Dataset({
        "forcing": ("phase_speed", summed),
        "phase_speed_bounds": (("phase_speed", "bound"), np.column_stack([edges[:-1], edges[1:]])),
        "stationary_forcing": float(values[stationary].sum()),
        "unresolved_forcing": float(values[unresolved].sum()),
        "total_forcing": float(values.sum()),
    }, coords={"phase_speed": (edges[:-1]+edges[1:])/2., "bound": [0, 1]})
    product.attrs.update(bin_width_ms=float(bin_width), propagation="negative westward; positive eastward",
                         unresolved="nonstationary k=0 and spatial/temporal Nyquist modes")
    product.phase_speed.attrs.update(units="m s-1", bounds="phase_speed_bounds")
    for name in ("forcing", "stationary_forcing", "unresolved_forcing", "total_forcing"):
        product[name].attrs.update(field.attrs)
    np.testing.assert_allclose(product.forcing.sum()+product.stationary_forcing+product.unresolved_forcing,
                               product.total_forcing, rtol=1e-11, atol=1e-12)
    return product
