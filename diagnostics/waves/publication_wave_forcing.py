"""Publication budgets and full-TEM wave transport in the mechanism box.

The vertical profiles use the same instantaneous native-height divergence
as the regional spectrum. Pressure and latitude sampling follow the
mechanism diagnostic. Neither mean coefficients nor momentum-only divergence
replace the full TEM operator.
"""

from __future__ import annotations

import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from time import perf_counter

import numpy as np
import xarray as xr
from scipy import fft
from scipy.sparse import coo_matrix

from diagnostics.analysis_config import simulation_cache_name, simulation_colour
from diagnostics.cache_support import (cache_manifest, cache_manifest_matches, materialize,
    read_json_file, write_json_file, write_netcdf_atomic)
from diagnostics.diagnostic_helpers import progress
from diagnostics.mean_state import circulation_mechanism as mechanism
from diagnostics.mean_state.circulation_bootstrap import hemisphere_samples
from diagnostics.tem_core import EARTH_RADIUS, TEM_FORMULATION, TEM_VERSION, epfd_flux_terms, flux_divergence
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.waves.regional_ep_flux_sectors import (
    FREQUENCY_CUTOFF, WAVENUMBER_CUTOFF, LOW_SECTORS, positive_spatial_masks, subset_mask,
)
from diagnostics.waves.regional_epfd_spectrum import (
    _interpolation_matrix, control_region_weights, mechanism_latitude_indices, signed_phase_speed_histogram,
)
from diagnostics.waves.sector_transport import RAW_FIELDS, native_sector_fields, _store_frame

PROFILE_VERSION = "publication-vertical-forcing-v2-1ms-tem-transport"
PUBLICATION_VERSION = "publication-wave-mechanisms-v2-aligned-tem-profiles"
PHASE_BIN_WIDTH = 1.
EULERIAN_FLOW_LEVEL = .2
PRESSURES = np.unique(np.r_[np.geomspace(50., 500., 61), mechanism.REGION_PRESSURES])
FAMILIES = ("low_frequency", "high_wavenumber")
BACKGROUND_FIELDS = ("pressure", "EPFd", "rho_zm", "u_z", "theta_z", "absolute_vorticity", "u_bar", "v_bar")

def publication_windows(analysis):
    """Select each direction's strongest negative slow meridional response bin."""
    result = []
    for sector in LOW_SECTORS:
        histogram = signed_phase_speed_histogram(
            analysis.spectrum.sel(comparison="u_minus_control", component="meridional").where(
                analysis.sector_mask.sel(sector=sector), 0.),
            time_count=analysis.attrs["time_sample_count"], longitude_count=analysis.attrs["longitude_sample_count"])
        eligible = ((histogram.phase_speed > 0) if sector.endswith("eastward") else (histogram.phase_speed < 0))
        eligible &= abs(histogram.phase_speed) < 20
        chosen = histogram.where(eligible, drop=True)
        best = int(chosen.forcing.argmin())
        if float(chosen.forcing.isel(phase_speed=best)) >= 0:
            raise ValueError(f"No negative slow meridional forcing response in {sector}")
        left, right = chosen.phase_speed_bounds.values[best]
        result.append(dict(name=sector+"_dominant_speed", parent=sector,
            speed_min=float(left), speed_max=float(right)))
    return result


def _sample_profile(field, pressure, targets):
    return mechanism._on_pressure(hemisphere_samples(field, "even"), hemisphere_samples(pressure, "even"),
        targets, "pres").interp(absolute_latitude=mechanism.REGION_LATITUDES).rename(
            absolute_latitude="region_latitude").transpose("Time", "hemisphere", "pres", "region_latitude")


def profile_latitude_weights(control, fixed_weights, pressures=PRESSURES):
    """Extend Control mass weighting vertically; preserve every original box weight."""
    density = _sample_profile(control.rho_zm, control.pressure, pressures)
    height = _sample_profile(xr.zeros_like(control.pressure)+control.z, control.pressure, pressures)
    minus_z_p = -height.assign_coords(pres=pressures*100.).differentiate("pres", edge_order=2)
    minus_z_p = minus_z_p.assign_coords(pres=pressures)
    mass = (density*minus_z_p).mean("Time")*np.cos(np.deg2rad(density.region_latitude))
    mass *= xr.DataArray(mechanism._quadrature(np.deg2rad(mechanism.REGION_LATITUDES)), dims="region_latitude")
    mass /= mass.sum("region_latitude")
    mass /= mass.sizes["hemisphere"]
    original = fixed_weights.rename(region_pres="pres")
    original = original/original.sum(("hemisphere", "region_latitude"))
    mass.loc[dict(pres=original.pres)] = original
    if not np.isfinite(mass).all() or not (mass > 0).all():
        raise ValueError("Publication profiles require positive finite Control mass weights")
    return mass.transpose("hemisphere", "pres", "region_latitude")


def _profile_native_latitude_weights(background, weights):
    """Transpose exact latitude interpolation and pointwise valid-time means."""
    sampled = _sample_profile(background.EPFd, background.pressure, weights.pres.values)
    valid = np.isfinite(sampled.values)
    counts = valid.sum(axis=0)
    if np.any(counts == 0):
        raise ValueError("A publication profile point has no valid forcing samples")
    target = valid*weights.values[None]*background.sizes["Time"]/counts[None]
    native = np.zeros((background.sizes["Time"], weights.sizes["pres"], background.sizes["latitude"]))
    for hemisphere, sign in enumerate((-1., 1.)):
        columns = np.flatnonzero(sign*background.latitude.values >= 0)
        columns = columns[np.argsort(abs(background.latitude.values[columns]))]
        matrix = _interpolation_matrix(abs(background.latitude.values[columns]), mechanism.REGION_LATITUDES)
        native[:, :, columns] = np.einsum("tpl,lj->tpj", target[:, hemisphere], matrix)
    return native


def phase_bin_allocation(time_count, longitude_count, bin_width=PHASE_BIN_WIDTH):
    """Map the existing signed sectors onto complete-record positive-k storage."""
    modes = np.arange(longitude_count//2+1)
    names, masks = positive_spatial_masks(time_count, longitude_count, modes)
    family_masks = np.stack([masks[[names.index(name) for name in selected]].any(axis=0)
        for selected in (LOW_SECTORS, ("high_wavenumber_eastward", "high_wavenumber_westward"))])
    frequency = fft.fftfreq(time_count, d=.25)
    speed = np.divide(2*np.pi*EARTH_RADIUS/86400.*frequency[:, None], modes[None],
        out=np.zeros((time_count, modes.size)), where=modes[None] != 0)
    extent = max(1, int(np.ceil(np.max(abs(speed)[family_masks.any(axis=0)], initial=0.)/bin_width))+1)
    edges = np.arange(-extent, extent+1)*bin_width
    indices = np.searchsorted(edges, speed, side="right")-1
    allocation = np.full(speed.shape, -1, dtype=int)
    for family, mask in enumerate(family_masks):
        allocation[mask] = family*(edges.size-1)+indices[mask]
    return edges, allocation


def _read_staged_height(spatial, level):
    """Read disjoint time records concurrently, avoiding serial NFS page faults."""
    if (not isinstance(spatial, np.memmap) or not spatial.flags.c_contiguous
            or isinstance(spatial.base, np.memmap)):
        return np.array(spatial[:, level])
    output = np.empty((spatial.shape[0], *spatial.shape[2:]), dtype=spatial.dtype)
    descriptor = os.open(spatial.filename, os.O_RDONLY)
    def read_range(bounds):
        for time in range(*bounds):
            destination = output[time].view("uint8").ravel()
            offset = spatial.offset+time*spatial.strides[0]+level*spatial.strides[1]
            complete = 0
            while complete < destination.size:
                count = os.preadv(descriptor, [destination[complete:]], offset+complete)
                if count <= 0:
                    raise OSError("The staged Fourier file ends inside a height record")
                complete += count
    workers = min(4, spatial.shape[0])
    edges = np.linspace(0, spatial.shape[0], workers+1, dtype=int)
    try:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            list(executor.map(read_range, zip(edges[:-1], edges[1:])))
    finally:
        os.close(descriptor)
    return output


def compute_vertical_profiles(spatial, background, latitude_weights, longitude_count):
    """Apply native divergence before exact pressure/latitude/time reduction.

    Storage axes are time, height, latitude, field=(u,rho*v,rho*w,theta), mode.
    The existing self-adjoint filters allocate time-local mass covariances.
    Each bin then uses the shared TEM operators, including instantaneous density
    and heat-flux coefficients. A sparse interpolation matrix performs only
    the final pressure, latitude and time reduction. Complete-record Fourier
    transforms occur once per spatial block and feed every speed mask.
    """
    nt, nz, nlat = [background.sizes[name] for name in ("Time", "z", "latitude")]
    if spatial.shape != (nt, nz, nlat, 4, longitude_count//2+1):
        raise ValueError("Publication Fourier records and TEM coordinates differ")
    edges, allocation = phase_bin_allocation(nt, longitude_count)
    bins = edges.size-1
    groups = np.unique(allocation[allocation >= 0])
    masks = [(allocation == group) for group in groups]
    modes = [np.flatnonzero(mask.any(axis=0)) for mask in masks]
    pressure = background.pressure.transpose("Time", "z", "latitude").values/100.
    native_latitude = _profile_native_latitude_weights(background, latitude_weights)
    forcing = np.zeros((latitude_weights.sizes["pres"], len(FAMILIES), bins))
    flux = np.zeros_like(forcing)
    meridional = np.zeros_like(forcing)
    tem_transport = np.zeros_like(forcing)
    row_indices, column_indices, projection_values = [], [], []
    for column in range(nlat):
        if not np.any(native_latitude[:, :, column]):
            continue
        projection = np.stack([_interpolation_matrix(pressure[t, ::-1, column], latitude_weights.pres.values)[:, ::-1]
            for t in range(nt)])
        q = projection*native_latitude[:, :, column, None]
        times, targets, levels = np.nonzero(q)
        row_indices.append(targets)
        column_indices.append((times*nz+levels)*nlat+column)
        projection_values.append(q[times, targets, levels]/nt)
    projection = coo_matrix((np.concatenate(projection_values),
        (np.concatenate(row_indices), np.concatenate(column_indices))),
        shape=(latitude_weights.sizes["pres"], nt*nz*nlat)).tocsr()
    active_points = projection.getnnz(axis=0) > 0
    del row_indices, column_indices, projection_values
    def project(field):
        values = field.transpose("Time", "z", "latitude").values.ravel()
        if not np.isfinite(values[active_points]).all():
            raise ValueError("A full-TEM field has invalid samples inside the active profile projection")
        return projection @ np.where(np.isfinite(values), values, 0.)
    # Three height planes preserve the original derivative stencil. Storage
    # stays bounded when the number of phase-speed bins increases fivefold.
    planes = {}
    multiplicity = np.where((np.arange(longitude_count//2+1) == 0) |
        (2*np.arange(longitude_count//2+1) == longitude_count), 1., 2.)/longitude_count**2

    def covariance_plane(level):
        started = perf_counter()
        progress(f"Publication spectra: calculating Fourier covariances for height {level+1}/{nz}")
        plane = _read_staged_height(spatial, level)
        result = np.empty((groups.size, nt, nlat, 3))
        for left in range(0, nlat, 8):
            right = min(nlat, left+8)
            # Keep each complete time series contiguous through both FFTs.
            block = np.ascontiguousarray(np.moveaxis(plane[:, left:right], 0, -1))
            transformed = fft.fft(block, axis=-1, workers=2)
            for index, (mask, selected_modes) in enumerate(zip(masks, modes)):
                original = block[:, :, selected_modes, :]
                filtered = fft.ifft(transformed[:, :, selected_modes, :]*mask[:, selected_modes].T[None, None],
                    axis=-1, workers=2)
                for pair, (a, b) in enumerate(((0, 1), (0, 2), (1, 3))):
                    cross = .5*(filtered[:, a]*original[:, b].conj()
                        + original[:, a]*filtered[:, b].conj()).real
                    result[index, :, left:right, pair] = np.einsum("lkt,k->tl", cross, multiplicity[selected_modes], optimize=True)
        progress(f"Publication spectra: height {level+1}/{nz} Fourier covariances complete ({perf_counter()-started:.1f} s)")
        return result

    for level in range(nz):
        start = min(max(0, level-1), nz-3)
        stencil = range(start, start+3)
        for previous in list(planes):
            if previous not in stencil:
                del planes[previous]
        for needed in stencil:
            if needed not in planes:
                planes[needed] = covariance_plane(needed)
        local = background.isel(z=slice(start, start+3))
        columns = ((np.arange(nt)[:, None]*nz+level)*nlat+np.arange(nlat)).ravel()
        local_projection = projection[:, columns]
        active = active_points[columns]
        def project_level(field):
            values = field.isel(z=level-start).transpose("group", "Time", "latitude").values.reshape(-1, nt*nlat)
            if not np.isfinite(values[:, active]).all():
                raise ValueError("A full-TEM field has invalid samples inside the active profile projection")
            return (local_projection @ np.where(np.isfinite(values), values, 0.).T)
        for first in range(0, groups.size, 12):
            last = min(groups.size, first+12)
            values = np.stack([planes[z][first:last] for z in stencil], axis=2)
            covariance = xr.Dataset({name: xr.DataArray(values[..., index],
                dims=("group", "Time", "z", "latitude"),
                coords={name: local[name] for name in ("Time", "z", "latitude")})/local.rho_zm
                for index, name in enumerate(("uv", "uw", "vtheta"))})
            terms = epfd_flux_terms(covariance, local)
            fields = flux_divergence(terms, local.rho_zm, "z", local.z.values)
            valid = np.isfinite(fields.EPFd)
            reduced = [project_level(field) for field in (
                fields.EPFd_z.where(valid), fields.EPFd_y.where(valid),
                covariance.uw*local.rho_zm,
                -local.rho_zm*(terms.momentum_vertical+terms.heat_vertical))]
            for offset, group in enumerate(groups[first:last]):
                family, speed_bin = divmod(int(group), bins)
                for output, value in zip((forcing, meridional, flux, tem_transport), reduced):
                    output[:, family, speed_bin] += value[:, offset]
        progress(f"Publication 1 m/s full-TEM spectra: height {level+1}/{nz} complete")
    product = xr.Dataset(dict(vertical_forcing=(("pres", "family", "phase_speed"), forcing),
        meridional_forcing_by_speed=(("pres", "family", "phase_speed"), meridional),
        vertical_momentum_flux=(("pres", "family", "phase_speed"), flux),
        vertical_tem_transport=(("pres", "family", "phase_speed"), tem_transport),
        phase_speed_bounds=(("phase_speed", "bound"), np.column_stack((edges[:-1], edges[1:])))),
        coords=dict(pres=latitude_weights.pres, family=list(FAMILIES), phase_speed=.5*(edges[:-1]+edges[1:])))
    product["zonal_wind"] = ("pres", project(background.u_bar))
    product["equivalent_wind"] = ("pres", project(background.u_bar/np.cos(np.deg2rad(background.latitude))))
    winds = _sample_profile(background.u_bar/np.cos(np.deg2rad(background.latitude)),
        background.pressure, latitude_weights.pres.values).mean("Time")
    product["equivalent_wind_min"] = winds.min(("hemisphere", "region_latitude"))
    product["equivalent_wind_max"] = winds.max(("hemisphere", "region_latitude"))
    product.vertical_forcing.attrs.update(units="m s-2", operator="full momentum_vertical + heat_vertical HPE(z)")
    product.vertical_momentum_flux.attrs.update(units="Pa", definition="u'*(rho*w)'; momentum only; positive upward eastward")
    product.vertical_tem_transport.attrs.update(units="Pa",
        definition="rho*(uw - absolute_vorticity*vtheta/theta_z); upward eastward TEM momentum transport")
    product.meridional_forcing_by_speed.attrs.update(units="m s-2", operator="full momentum_meridional + heat_meridional HPE(z)")
    product.pres.attrs["units"] = "hPa"
    product.phase_speed.attrs["units"] = "m s-1"
    for name in ("zonal_wind", "equivalent_wind", "equivalent_wind_min", "equivalent_wind_max"):
        product[name].attrs["units"] = "m s-1"
    product.attrs.update(product_version=PROFILE_VERSION, tem_formulation=TEM_FORMULATION,
        time_sample_count=nt, longitude_sample_count=longitude_count, bin_width_ms=PHASE_BIN_WIDTH,
        latitude_average="5–15 degrees absolute latitude; fixed Control mass; equal hemispheres",
        spectral_allocation="symmetric time-local allocation; instantaneous coefficients; divergence before averaging",
        phase_speed_convention="c_eq=2*pi*a*f/(86400*k); equivalent wind = u_bar/cos(latitude)")
    return product


def validate_profile_budget(product, analysis, fixed_weights, simulation):
    """Check each signed vertical speed bin against the original regional budget once."""
    pressure_weights = fixed_weights.sum(("hemisphere", "region_latitude")).rename(region_pres="pres")
    for component, variable, output in (("vertical", "vertical_forcing", "regional_vertical_forcing"),
            ("meridional", "meridional_forcing_by_speed", "regional_meridional_forcing_by_speed")):
        regional = (product[variable].sel(pres=pressure_weights.pres)*pressure_weights).sum("pres")*86400.
        for family, names in zip(FAMILIES, (LOW_SECTORS, ("high_wavenumber_eastward", "high_wavenumber_westward"))):
            mask = analysis.sector_mask.sel(sector=list(names)).any("sector")
            expected = signed_phase_speed_histogram(analysis.spectrum.sel(comparison=simulation, component=component).where(mask, 0.),
                time_count=analysis.attrs["time_sample_count"], longitude_count=analysis.attrs["longitude_sample_count"],
                bin_width=product.attrs["bin_width_ms"])
            actual = regional.sel(family=family).reindex(phase_speed=expected.phase_speed, fill_value=0.)
            np.testing.assert_allclose(actual, expected.forcing, atol=2e-7, rtol=2e-6,
                err_msg=f"{family} {component} profiles differ from the original signed regional forcing bins")
        product[output] = regional/86400.


def build_meridional_sections(covariances, background, analysis, windows, fixed_weights, simulation, pressures=PRESSURES, *, retain_samples=False):
    """Reuse saved instantaneous mass covariances, including the TEM heat term."""
    saved = json.loads(covariances.attrs["subset_definitions"])
    forcings, transports, tem_transports, integrated_transports, regional = [], [], [], [], []
    forcing_samples, integrated_samples = [], []
    latitude = np.unique(np.r_[background.latitude.values[background.latitude.values >= 0], mechanism.REGION_LATITUDES])
    point_weights = fixed_weights.rename(region_pres="pres", region_latitude="absolute_latitude")
    hemisphere_weights = point_weights/point_weights.sum("hemisphere")
    hemisphere_weights = hemisphere_weights.interp(
        pres=xr.DataArray(np.clip(pressures, *mechanism.REGION_PRESSURE), dims="pres"),
        absolute_latitude=xr.DataArray(np.clip(latitude, *mechanism.REGION_LATITUDE), dims="absolute_latitude")
    ).assign_coords(pres=pressures, absolute_latitude=latitude)
    def sample(field, parity):
        return mechanism._on_pressure(hemisphere_samples(field, parity),
            hemisphere_samples(background.pressure, "even"), pressures, "pres").interp(absolute_latitude=latitude)
    def fold(field, parity):
        # Interpolate each time and hemisphere before averaging. Local Control
        # mass weights preserve the original equal-hemisphere regional integral.
        sampled = sample(field, parity).mean("Time")
        return (sampled*hemisphere_weights).sum("hemisphere", skipna=False)
    sampled_height = sample(xr.zeros_like(background.pressure)+background.z, "even")
    # Integrate in geometric height between the actual pressure surfaces.
    # This Jacobian retains each instantaneous hemisphere before averaging.
    height_jacobian = -sampled_height.differentiate("pres", edge_order=2)
    region_weight = xr.DataArray(np.zeros((len(pressures), len(latitude))), dims=("pres", "absolute_latitude"),
        coords=dict(pres=pressures, absolute_latitude=latitude))
    region_weight.loc[dict(pres=point_weights.pres, absolute_latitude=point_weights.absolute_latitude)] = point_weights.sum("hemisphere")
    for definition in windows:
        name = definition["name"]
        progress(f"{simulation}: full meridional forcing and momentum overlay for {name}")
        previous = next((item for item in saved if item["name"] == name), {})
        if any(previous.get(key) != definition[key] for key in ("parent", "speed_min", "speed_max")):
            raise ValueError(f"Saved pathway {name} does not match the meridional response window; rebuild its covariance cache")
        covariance = materialize(covariances.pathway_covariance.sel(pathway=name,
            covariance=["uv", "uw", "vtheta"]).transpose("Time", "z", "latitude", "covariance"))
        fields = native_sector_fields(covariance.values, background)
        instantaneous = (sample(fields.EPFd_y, "even")*hemisphere_weights).sum("hemisphere", skipna=False)
        forcings.append(instantaneous.mean("Time"))
        if retain_samples:
            forcing_samples.append(instantaneous)
        transports.append(fold(covariance.sel(covariance="uv", drop=True), "odd"))
        geometry = np.cos(np.deg2rad(background.latitude))**2
        tem = -background.rho_zm*geometry*fields.ep_flux_meridional_normalized
        tem_transports.append(fold(tem, "odd"))
        integrand = sample(tem, "odd")*height_jacobian*hemisphere_weights
        integrated = integrand.sum("hemisphere", skipna=False).sel(
            pres=slice(*mechanism.REGION_PRESSURE)).integrate("pres")
        integrated_transports.append(integrated.mean("Time"))
        if retain_samples:
            integrated_samples.append(integrated)
        actual = (mechanism.sample_mechanism_field(fields.EPFd_y, background.pressure).mean("Time")*fixed_weights).sum()
        mask = subset_mask(analysis.wavenumber, analysis.frequency, definition,
            time_count=analysis.attrs["time_sample_count"], longitude_count=analysis.attrs["longitude_sample_count"])
        expected = analysis.spectrum.sel(comparison=simulation, component="meridional").where(mask, 0.).sum()/86400.
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-12,
            err_msg=f"{name} full meridional TEM section differs from its regional spectrum")
        np.testing.assert_allclose((forcings[-1]*region_weight).sum(), actual, rtol=2e-6, atol=2e-12,
            err_msg=f"{name} displayed hemispheric section loses the regional mass-weighted budget")
        regional.append(actual)
    subset = xr.IndexVariable("subset", [item["name"] for item in windows])
    product = xr.Dataset(dict(meridional_forcing=xr.concat(forcings, dim=subset),
        meridional_momentum_flux=xr.concat(transports, dim=subset),
        meridional_tem_transport=xr.concat(tem_transports, dim=subset),
        integrated_meridional_tem_transport=xr.concat(integrated_transports, dim=subset),
        regional_meridional_forcing=xr.concat(regional, dim=subset)))
    if retain_samples:
        product["meridional_forcing_samples"] = xr.concat(forcing_samples, dim=subset)
        product["integrated_meridional_tem_transport_samples"] = xr.concat(integrated_samples, dim=subset)
    product["poleward_mean_flow"] = fold(background.v_bar, "odd")
    product["equivalent_wind_section"] = fold(background.u_bar/np.cos(np.deg2rad(background.latitude)), "even")
    product["region_weight"] = region_weight
    product.meridional_forcing.attrs.update(units="m s-2", operator="full momentum_meridional + heat_meridional HPE(z)")
    product.meridional_momentum_flux.attrs.update(units="Pa", definition="u'*(rho*v_poleward)'; momentum only; negative equatorward")
    product.meridional_tem_transport.attrs.update(units="Pa",
        definition="rho*cos(latitude)^2*(uv-u_z*vtheta/theta_z); poleward eastward TEM momentum transport")
    product.integrated_meridional_tem_transport.attrs.update(units="N m-1",
        definition="integral of meridional_tem_transport dz between instantaneous 170 and 100 hPa surfaces")
    product.equivalent_wind_section.attrs.update(units="m s-1", definition="Control-weighted mean of u_bar/cos(latitude)")
    product.poleward_mean_flow.attrs.update(units="m s-1", definition="Eulerian mean v_bar; SH sign reversed; positive poleward")
    return product


def _store_profile_frame(stage, index, frame, fields, *, include_wind_variance=False):
    if tuple(fields) == RAW_FIELDS:
        _store_frame(stage, None, index, frame, None, None, stage.shape[-1],include_wind_variance=include_wind_variance)
    else:
        values = np.stack([frame[name].isel(Time=0).transpose("z", "latitude", "longitude").values
            for name in fields]).astype(float)
        if not np.isfinite(values).all():
            raise ValueError("Spectral diagnostics require complete finite native field records")
        values -= values.mean(axis=-1, keepdims=True)
        transformed = fft.rfft(values[..., ::-1], axis=-1, workers=2)
        transformed[..., 0] = 0.
        stage[index] = transformed.transpose(1, 2, 0, 3)


def _native_profile_stage(archive_dir, pattern, start, end, upstream, background, manifest, scratch_root, *, fields=RAW_FIELDS,
        stage_dtype="complex128", include_wind_variance=False):
    """Read complete native frames once; retain a resumable sequential Fourier stage."""
    from diagnostics.waves.ep_flux_decomposition import _open_epfd_raw_archive
    root = Path(scratch_root or os.environ.get("MPAS_EPFD_SCRATCH_ROOT") or Path(__file__).resolve().parents[2]/".epfd_scratch")
    prefix, task_label = {
        RAW_FIELDS: ("publication-profiles-", "Publication profiles"),
        ("u", "v"): ("sector-wind-variance-", "Horizontal-wind variance"),
        ("pressure", "v", "w"): ("wave-energy-flux-", "Wave-energy flux"),
    }.get(tuple(fields), ("native-spectral-fields-", "Native spectral fields"))
    directory = root/(prefix+simulation_cache_name(manifest["configuration"]["simulation"]))
    directory.mkdir(parents=True, exist_ok=True)
    state_path, stage_path = directory/"progress.json", directory/"fourier.dat"
    state = read_json_file(state_path) or {}
    resumed = state.get("manifest_sha256") == manifest["manifest_sha256"] and stage_path.exists()
    count = int(state.get("complete_samples", 0)) if resumed else 0
    nt = background.sizes["Time"]
    if not 0 <= count <= nt:
        raise ValueError("Invalid sample count in the resumable publication Fourier stage")
    # Open only unread dates when a previous calculation supplies complete frames.
    remaining_start = str(background.Time.values[min(count, nt-1)])
    dataset, reader = _open_epfd_raw_archive(archive_dir, pattern, remaining_start, end, upstream.z, requested_fields=fields)
    count_fields = 4 if tuple(fields) == RAW_FIELDS else len(fields)
    if include_wind_variance:
        if tuple(fields)!=RAW_FIELDS:
            raise ValueError("Combined variance and covariance reads require the complete native field set")
        count_fields+=1
    shape = (nt, background.sizes["z"], background.sizes["latitude"], count_fields, dataset.sizes["longitude"]//2+1)
    dtype=np.dtype(stage_dtype)
    required = int(np.prod(shape))*dtype.itemsize
    if resumed and (state.get("shape") != list(shape) or stage_path.stat().st_size != required):
        dataset.close()
        raise ValueError("The saved publication Fourier stage has incompatible dimensions")
    if not resumed and shutil.disk_usage(directory).free < required+1024**3:
        dataset.close()
        raise OSError(f"Publication profiles need {required/1024**3:.1f} GiB under {directory}")
    stage = np.memmap(stage_path, dtype=dtype, mode="r+" if resumed else "w+", shape=shape)
    previous_ranges = state.get("ranges") if resumed else None
    state = dict(manifest_sha256=manifest["manifest_sha256"], complete_samples=count, shape=list(shape))
    write_json_file(state_path, state)
    def terminate(signum, frame):
        raise SystemExit(128+signum)
    old_handler = signal.signal(signal.SIGTERM, terminate)
    try:
        completed = count
        if count < nt and previous_ranges:
            for left, right in zip(previous_ranges[:-1], previous_ranges[1:]):
                previous = read_json_file(directory/f"reader-{left}.json") or {}
                if previous.get("manifest_sha256") == manifest["manifest_sha256"] and previous.get("right") == right:
                    completed += min(right, max(left, int(previous.get("complete_samples", left))))-left
        progress(f"{task_label} CALCULATION: {nt-completed}/{nt} native frames remain; "
            f"{shape[1]} heights; {shape[2]} latitudes; {required/1024**3:.1f} GiB sequential Fourier stage")
        workers = max(1, int(os.environ.get("MPAS_SECTOR_NATIVE_READERS", "4")))
        if count < nt and workers > 1:
            edges = previous_ranges or np.linspace(count, nt, min(workers, nt-count)+1, dtype=int).tolist()
            state["ranges"] = edges
            write_json_file(state_path, state)
            configuration = dict(archive_dir=str(archive_dir), pattern=pattern, stage_path=str(stage_path),
                shape=list(shape), z=upstream.z.values.tolist(), height=background.z.values.tolist(),
                latitude=background.latitude.values.tolist(), times=[str(value) for value in background.Time.values],
                manifest_sha256=manifest["manifest_sha256"], fields=list(fields), task_label=task_label, stage_dtype=dtype.name,
                include_wind_variance=include_wind_variance)
            config_path = directory/"readers.json"
            write_json_file(config_path, configuration)
            progress(f"{task_label}: {len(edges)-1} independent native readers; one full-record temporal transform")
            processes = []
            try:
                for left, right in zip(edges[:-1], edges[1:]):
                    processes.append(subprocess.Popen([sys.executable, "-m", "diagnostics.waves.publication_wave_forcing",
                        "--native-worker", str(config_path), str(left), str(right)], cwd=Path(__file__).resolve().parents[2]))
                for process in processes:
                    status = process.wait()
                    if status:
                        raise RuntimeError(f"A publication native reader exits with status {status}; checkpoints remain reusable")
            finally:
                for process in processes:
                    if process.poll() is None:
                        process.terminate()
                for process in processes:
                    process.wait()
            state["complete_samples"] = nt
            write_json_file(state_path, state)
        elif count < nt:
            frames = iter(reader.iter_frames(background.latitude.values, output_height=background.z.values))
            for index in range(count, nt):
                started = perf_counter()
                progress(f"{task_label} native read {index+1}/{nt}: {background.Time.values[index]}")
                frame = next(frames)
                if not np.array_equal(frame.Time, background.Time.isel(Time=slice(index, index+1))):
                    raise ValueError("Publication native frames and TEM timestamps differ")
                _store_profile_frame(stage, index, frame, fields,include_wind_variance=include_wind_variance)
                if (index+1) % 8 == 0 or index+1 == nt:
                    stage.flush()
                    state["complete_samples"] = index+1
                    write_json_file(state_path, state)
                progress(f"{task_label} native sample {index+1}/{nt}: read and transform complete ({perf_counter()-started:.1f} s)")
        return stage, dataset.sizes["longitude"], directory
    except BaseException:
        stage._mmap.close()
        raise
    finally:
        dataset.close()
        signal.signal(signal.SIGTERM, old_handler)


def _native_profile_worker(configuration_path, left, right):
    """Fill one disjoint frame range using the existing native-reader conventions."""
    from diagnostics.waves.ep_flux_decomposition import _open_epfd_raw_archive
    config = read_json_file(configuration_path)
    progress_path = Path(configuration_path).parent/f"reader-{left}.json"
    previous = read_json_file(progress_path) or {}
    valid = previous.get("manifest_sha256") == config["manifest_sha256"] and previous.get("right") == right
    count = int(previous.get("complete_samples", left)) if valid else left
    if not left <= count <= right:
        raise ValueError("A publication native-reader checkpoint has an invalid frame count")
    if count == right:
        progress(f"Publication reader {left+1}–{right}: complete staged frames reused")
        return
    height = xr.DataArray(config["z"], dims="z", coords={"z": config["z"]})
    fields = tuple(config.get("fields", RAW_FIELDS))
    task_label = config.get("task_label", "Publication")
    dataset, reader = _open_epfd_raw_archive(config["archive_dir"], config["pattern"], config["times"][count],
        config["times"][right-1], height, requested_fields=fields)
    stage = np.memmap(config["stage_path"], dtype=config.get("stage_dtype","complex128"), mode="r+", shape=tuple(config["shape"]))
    state = dict(manifest_sha256=config["manifest_sha256"], right=right, complete_samples=count)
    try:
        progress(f"{task_label} reader {left+1}–{right}: starts at native sample {count+1}")
        complete = count
        for index, frame in enumerate(reader.iter_frames(config["latitude"], output_height=config["height"]), start=count):
            started = perf_counter()
            if index >= right or not np.array_equal(frame.Time, dataset.Time.isel(Time=slice(index-count, index-count+1))):
                raise ValueError("A publication native reader returns an incorrect timestamp")
            _store_profile_frame(stage, index, frame, fields,include_wind_variance=config.get("include_wind_variance",False))
            complete = index+1
            if complete % 8 == 0 or complete == right:
                stage.flush()
                state["complete_samples"] = complete
                write_json_file(progress_path, state)
            if index == count or complete % 10 == 0 or complete == right:
                progress(f"{task_label} reader {left+1}–{right}: native sample {complete} complete; longitude transform {perf_counter()-started:.1f} s")
        if complete != right:
            raise ValueError("A publication native reader ends before its complete frame range")
    finally:
        stage._mmap.close()
        dataset.close()


def _cached_product(path, manifest, figures_only, *, variables=None):
    if not path.exists():
        return None
    with xr.open_dataset(path) as stored:
        previous = json.loads(stored.attrs.get("cache_manifest", "{}"))
        if (stored.attrs.get("manifest_sha256") == previous.get("manifest_sha256")
                and cache_manifest_matches(previous, manifest, compare_sources=not figures_only)):
            return (stored[list(variables)] if variables is not None else stored).load()
    return None


def load_or_build_publication(*, simulation, archive_dir, control_archive_dir, cache_dir, pattern, start, end,
        analysis, tem_root=None, figures_only=False, rebuild=False, scratch_root=None):
    """Reuse compact publication data; perform expensive closure checks only at construction."""
    import fcntl
    windows = publication_windows(analysis)
    cache = Path(cache_dir)/"epfd"
    covariance_path = (cache/f"sector_covariances_{simulation_cache_name(simulation)}.nc").resolve()
    profile_path = cache/f"publication_vertical_forcing_{simulation_cache_name(simulation)}.nc"
    path = cache/f"publication_wave_mechanisms_{simulation_cache_name(simulation)}.nc"
    sources = sorted(glob.glob(str(Path(archive_dir)/pattern)))+tem_source_paths(Path(archive_dir).name, root=tem_root)
    sources += tem_source_paths(Path(control_archive_dir).name, root=tem_root)
    configuration = dict(simulation=simulation, start=str(start), end=str(end), pressure_levels_hpa=PRESSURES.tolist(),
        region_latitude=mechanism.REGION_LATITUDE, region_pressure=mechanism.REGION_PRESSURE,
        bin_width_ms=PHASE_BIN_WIDTH, frequency_cutoff=FREQUENCY_CUTOFF, wavenumber_cutoff=WAVENUMBER_CUTOFF)
    profile_manifest = cache_manifest(__name__, schema=PROFILE_VERSION, configuration=configuration,
        dependencies=dict(tem_version=TEM_VERSION), source_paths=sources)
    manifest = cache_manifest(__name__, schema=PUBLICATION_VERSION, configuration=dict(configuration, windows=windows),
        dependencies=dict(tem_version=TEM_VERSION, profile_version=PROFILE_VERSION), source_paths=sources+[covariance_path])
    if not rebuild:
        cached = _cached_product(path, manifest, figures_only)
        if cached is not None:
            progress(f"{simulation}: using full-TEM publication mechanism cache: {path}")
            return cached
    if figures_only:
        raise FileNotFoundError(f"Missing full-TEM publication product: {path}; calculate Figure 18 without --figures-only")
    if not covariance_path.exists():
        raise FileNotFoundError(f"Missing time-local covariance cache: {covariance_path}; calculate Figures 16–20 first")
    # A lock prevents two invocations from writing the same resumable stage.
    cache.mkdir(parents=True, exist_ok=True)
    with (cache/f".publication_{simulation_cache_name(simulation)}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        upstream = load_stored_tem(Path(archive_dir).name, formulation=TEM_FORMULATION, root=tem_root, start=start, end=end)
        control = load_stored_tem(Path(control_archive_dir).name, formulation=TEM_FORMULATION, root=tem_root, start=start, end=end)
        try:
            control_box = materialize(control[["rho_zm", "pressure"]].isel(latitude=mechanism_latitude_indices(control.latitude)))
            fixed = control_region_weights(control_box)
            latitude_weights = profile_latitude_weights(control_box, fixed)
            # Existing covariance caches supply the selected meridional windows.
            with xr.open_dataset(covariance_path) as covariances:
                progress(f"{simulation}: loading stored TEM coefficients for the saved meridional covariance grid")
                background = materialize(upstream[list(BACKGROUND_FIELDS)].sel(z=covariances.z, latitude=covariances.latitude))
                progress(f"{simulation}: stored TEM coefficients loaded; reusing saved meridional covariances")
                sections = build_meridional_sections(covariances, background, analysis, windows, fixed, simulation)
            profile = None if rebuild else _cached_product(profile_path, profile_manifest, False)
            if profile is None:
                local = background.isel(latitude=mechanism_latitude_indices(background.latitude))
                pressure = local.pressure.transpose("Time", "z", "latitude").values
                rows = np.flatnonzero(np.any((pressure >= 5000.) & (pressure <= 50000.), axis=(0, 2)))
                local = local.isel(z=slice(max(0, int(rows.min())-2), min(local.sizes["z"], int(rows.max())+3)))
                stage, nlon, directory = _native_profile_stage(archive_dir, pattern, start, end, upstream, local,
                    profile_manifest, scratch_root)
                try:
                    profile = compute_vertical_profiles(stage, local, latitude_weights, nlon)
                    validate_profile_budget(profile, analysis, fixed, simulation)
                    profile.attrs.update(cache_manifest=json.dumps(profile_manifest, sort_keys=True),
                        manifest_sha256=profile_manifest["manifest_sha256"])
                    write_netcdf_atomic(profile, profile_path)
                finally:
                    stage._mmap.close()
                # Release only this calculation's complete temporary Fourier records.
                (directory/"fourier.dat").unlink()
                (directory/"progress.json").unlink()
                for entry in directory.glob("reader*.json"):
                    entry.unlink()
                directory.rmdir()
            product = xr.merge([profile, sections], compat="no_conflicts")
            product.attrs.update(product_version=PUBLICATION_VERSION, simulation=simulation,
                windows=json.dumps(windows, sort_keys=True), cache_manifest=json.dumps(manifest, sort_keys=True),
                manifest_sha256=manifest["manifest_sha256"])
            write_netcdf_atomic(product, path)
            progress(f"{simulation}: full-TEM publication mechanisms saved: {path}")
            return product
        finally:
            upstream.close()
            control.close()


def plot_publication_mechanisms(products, forcing_confidence=None, transport_confidence=None, circulation_guides=None):
    """Pair full-TEM forcing maps with aligned, signed TEM transport profiles."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import NullFormatter
    from diagnostics.diagnostic_helpers import add_symmetric_colorbar, difference_cmap, nice_levels
    from diagnostics.waves.wave_response_plotting import (
        plot_circulation_guides, circulation_guide_handles, equatorial_display_field, plot_region_box)

    control, nudged = xr.align(products["control"], products["nudged-u"], join="exact")
    change = nudged-control
    windows = json.loads(control.attrs["windows"])
    fig = plt.figure(figsize=(8, 8))
    grid = fig.add_gridspec(4, 2, height_ratios=(1., .065, 1.28, .065),
        left=.085, right=.98, bottom=.16, top=.90, wspace=.25, hspace=.60)
    vertical_levels = (nice_levels(-.03, .03, n=25), nice_levels(-.01, .01, n=21))
    meridional_levels = nice_levels(-.4, .4, n=21)
    pressure_axes, lower_images = [], []

    def shading(axis, x, field, levels, filled_contours=False):
        cmap = difference_cmap(levels, cmap="RdBu_r", white_zero_boundary=True)
        if filled_contours:
            field = equatorial_display_field(field)
            return axis.contourf(field.absolute_latitude, field.pres, field.transpose("pres", ...), levels=levels,
                cmap=cmap, norm=BoundaryNorm(levels, cmap.N), extend="both")
        return axis.pcolormesh(x, field.pres, field.transpose("pres", ...), shading="auto",
            cmap=cmap, norm=BoundaryNorm(levels, cmap.N), rasterized=True)

    def label_critical_contours(axis, contours, positions):
        # Place one label per speed and simulation, away from edges and other labels.
        locations = []
        for segments in contours.allsegs:
            visible = [segment[(segment[:, 0] > 1.) & (segment[:, 0] < 24.) &
                (segment[:, 1] > 60.) & (segment[:, 1] < 450.)] for segment in segments]
            visible = [segment for segment in visible if len(segment)]
            if not visible:
                continue
            points = np.concatenate(visible)
            normalized = np.column_stack((points[:, 0]/25., np.log(points[:, 1]/50.)/np.log(10.)))
            score = np.minimum(normalized, 1.-normalized).min(axis=1)
            for position in positions:
                score = np.minimum(score, np.linalg.norm(normalized-position, axis=1))
            index = int(np.argmax(score))
            locations.append(points[index])
            positions.append(normalized[index])
        if locations:
            axis.clabel(contours, manual=locations, fmt=lambda speed: f"{speed:g}",
                fontsize=5.5, inline=True, inline_spacing=3)

    def pressure_guides(axis):
        for pressure in mechanism.REGION_PRESSURE:
            axis.axhline(pressure, color=mechanism.REGION_BOX_STYLE["edgecolor"], lw=.7, ls="--")

    def regional_label(axis, text):
        axis.text(.02, .98, text, va="top", fontsize=6, transform=axis.transAxes,
            bbox=dict(facecolor="white", edgecolor="none", alpha=.85))

    runs = ((control, "control", "--"), (nudged, "nudged-u", "-"))
    bounds = control.phase_speed_bounds
    for column, (family, interval) in enumerate(zip(FAMILIES, ((10., 25.), (-10., 10.)))):
        group = grid[0, column].subgridspec(1, 2, width_ratios=(3.4, 1.), wspace=.06)
        axis = fig.add_subplot(group[0, 0])
        profile = fig.add_subplot(group[0, 1], sharey=axis)
        pressure_axes.extend((axis, profile))
        forcing = change.vertical_forcing.sel(family=family)*86400.
        display_width = 2 if column == 0 else 1
        if display_width == 2:
            # Align zero with a bin edge, retaining separate propagation signs.
            first = 2*np.floor(float(bounds.min())/2)
            last = 2*np.ceil(float(bounds.max())/2)
            forcing = forcing.reindex(phase_speed=np.arange(first+.5, last, 1.), fill_value=0.).coarsen(phase_speed=2).sum()
        # Normalize only the displayed spectrum; profiles and budgets retain signed sums.
        image = shading(axis, forcing.phase_speed, forcing/display_width, vertical_levels[column])
        selected = (bounds.isel(bound=0) >= interval[0]) & (bounds.isel(bound=1) <= interval[1])
        for product, name, style in runs:
            colour = simulation_colour(name)
            axis.fill_betweenx(product.pres, product.equivalent_wind_min, product.equivalent_wind_max,
                color=colour, alpha=.09, linewidth=0, zorder=2)
            axis.plot(product.equivalent_wind, product.pres, color=colour, ls=style, lw=1.2, zorder=5)
            transport = product.vertical_tem_transport.sel(family=family).where(selected, 0.).sum("phase_speed")
            profile.plot(1000.*transport, product.pres, color=colour, ls=style, lw=1.1)
        for speed in interval:
            axis.axvline(speed, color=".35", lw=.7, ls=":")
        pressure_guides(axis)
        pressure_guides(profile)
        regional = change.regional_vertical_forcing.sel(family=family)*86400.
        band_vertical = float(regional.where(selected, 0.).sum())
        band_meridional = float(change.regional_meridional_forcing_by_speed.sel(family=family).where(selected, 0.).sum())*86400.
        regional_label(axis, f"Regional ΔA$_z$ = {float(regional.sum()):+.3f} m/s/day\n"
            f"Band: ΔA$_z$ = {band_vertical:+.3f}; ΔA$_y$ = {band_meridional:+.3f}")
        axis.set_xlim(-40, 40)
        axis.set_xticks(np.arange(-40, 41, 10))
        axis.set_xlabel("Ground-relative phase speed (m/s)")
        axis.set_ylabel("Pressure (hPa)")
        axis.set_title(f"({'ab'[column]}) "+("Low-frequency" if column == 0 else "High-wavenumber")+
            " vertical transmission", fontsize=7.5)
        profile.set_title(f"{interval[0]:g} to {interval[1]:g} m/s", fontsize=6.5)
        profile.axvline(0, color=".6", lw=.6)
        profile.set_xlabel("Upward TEM\ntransport (mPa)",fontsize=6.5,labelpad=5)
        profile.tick_params(labelleft=False)
        profile.tick_params(axis="x", labelsize=6)
        profile.ticklabel_format(axis="x",style="plain",useOffset=False)
        profile.xaxis.get_major_locator().set_params(nbins=3)
        add_symmetric_colorbar(fig, image, [axis, profile], vertical_levels[column],
            "Vertical EPFd change [(m/s/day)/(m/s)]", location="bottom", extend="both",
            ticks_at_boundaries=True, nticks=4, cax=fig.add_subplot(grid[1, column]))

    for column, definition in enumerate(windows):
        group = grid[2, column].subgridspec(2, 1, height_ratios=(3.2, 1.), hspace=.12)
        axis = fig.add_subplot(group[0, 0])
        profile = fig.add_subplot(group[1, 0], sharex=axis)
        pressure_axes.append(axis)
        name = definition["name"]
        forcing = change.meridional_forcing.sel(subset=name)*86400.
        lower_images.append(shading(axis, forcing.absolute_latitude, forcing, meridional_levels, filled_contours=True))
        if forcing_confidence is not None:
            from diagnostics.waves.wave_response_plotting import hatch_insignificant
            mask=forcing_confidence.significant.sel(subset=name).rename(
                section_pres="pres",section_latitude="absolute_latitude")
            hatch_insignificant(axis,mask,forcing,contrast_shading=(forcing,lower_images[-1]))
        edges = [definition["speed_min"], definition["speed_max"]]
        axis.set_xlim(0, 25)
        axis.set_yscale("log")
        axis.set_ylim(500, 50)
        contour_label_positions = []
        for product, simulation, style in runs:
            wind = product.equivalent_wind_section.transpose("pres", "absolute_latitude")
            levels = [level for level in edges if float(wind.min()) < level < float(wind.max())]
            if levels:
                contours = axis.contour(wind.absolute_latitude, wind.pres, wind, levels=levels,
                    colors=[simulation_colour(simulation)], linestyles=style, linewidths=1.1)
                label_critical_contours(axis, contours, contour_label_positions)
            transport = product.integrated_meridional_tem_transport.sel(subset=name)
            profile.plot(transport.absolute_latitude, transport, color=simulation_colour(simulation), ls=style, lw=1.1)
            if transport_confidence is not None:
                bounds=transport_confidence["control_interval" if simulation=="control" else "nudged_u_interval"].sel(
                    subset=name)
                profile.fill_between(bounds.section_latitude,bounds.sel(bound="lower"),bounds.sel(bound="upper"),
                    color=simulation_colour(simulation),alpha=.16,linewidth=0)
        circulation = change.poleward_mean_flow.transpose("pres", "absolute_latitude")
        axis.contour(circulation.absolute_latitude, circulation.pres, circulation, levels=[EULERIAN_FLOW_LEVEL],
            colors="#238443", linewidths=1., linestyles="solid")
        plot_circulation_guides(axis,circulation_guides)
        plot_region_box(axis)
        regional = float(change.regional_meridional_forcing.sel(subset=name))*86400.
        regional_label(axis, f"Regional ΔA$_y$ = {regional:+.3f} m/s/day")
        direction = "eastward" if definition["parent"].endswith("eastward") else "westward"
        axis.set_title(f"({'cd'[column]}) Slow {direction}: {edges[0]:g} to {edges[1]:g} m/s", fontsize=7.5)
        axis.set_xlim(0, 25)
        axis.set_xticks([0, 5, 10, 15, 20, 25])
        axis.set_ylabel("Pressure (hPa)")
        axis.tick_params(labelbottom=False)
        for latitude in mechanism.REGION_LATITUDE:
            profile.axvline(latitude, color=mechanism.REGION_BOX_STYLE["edgecolor"], lw=.7, ls="--")
        profile.axhline(0, color=".6", lw=.6)
        profile.text(.98, 1.04, f"{edges[0]:g} to {edges[1]:g} m/s; 100–170 hPa",
            ha="right", va="bottom", fontsize=5.5, transform=profile.transAxes, clip_on=False)
        profile.set_xlabel(r"Latitude ($^\circ$)")
        profile.set_ylabel("Poleward momentum\ntransport (N/m)", fontsize=6)
        profile.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2), useMathText=True)
        profile.yaxis.get_major_locator().set_params(nbins=3)
        profile.grid(axis="x", alpha=.15)
    for axis in pressure_axes:
        axis.set_yscale("log")
        axis.set_ylim(500, 50)
        axis.set_yticks([500, 300, 200, 170, 100, 70, 50], labels=["500", "300", "200", "170", "100", "70", "50"])
        axis.yaxis.set_minor_formatter(NullFormatter())
    add_symmetric_colorbar(fig, lower_images[0], None, meridional_levels,
        "Full meridional EPFd change (m/s/day)", location="bottom", extend="both",
        ticks_at_boundaries=True, nticks=4, cax=fig.add_subplot(grid[3, :]))
    handles = [Line2D([], [], color=simulation_colour(name), ls=style, lw=2.2, label=label)
        for name, label, style in (("control", "Control", "--"), ("nudged-u", "U", "-"))]
    handles += [Patch(facecolor=simulation_colour(name), alpha=.09, label=f"{label} wind range (5–15°)")
        for name, label in (("control", "Control"), ("nudged-u", "U"))]
    handles.append(Patch(facecolor="none", label="Regional boundaries", **mechanism.REGION_BOX_STYLE))
    handles.append(Line2D([], [], color="#238443", lw=1.,
        label=rf"Poleward $\Delta\bar v = +{EULERIAN_FLOW_LEVEL:g}$ m/s"))
    if circulation_guides is not None:
        handles.extend(circulation_guide_handles())
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(.5, .045), ncol=3, frameon=False, fontsize=6.5)
    fig.text(.5, .008, "Upper panels show descriptive means without confidence intervals.\n"
        "Positive transport carries eastward momentum upward in upper profiles and poleward in lower profiles.",
        ha="center", va="bottom", fontsize=6)
    fig.suptitle("U − Control: wave forcing and TEM transport\n"
        "Lower maps: nonsignificance hatching; lower profiles: 95% intervals; 40-day blocks", fontsize=8.5)
    return fig


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-worker", nargs=3, required=True, metavar=("CONFIG", "LEFT", "RIGHT"))
    arguments = parser.parse_args()
    configuration, left, right = arguments.native_worker
    _native_profile_worker(configuration, int(left), int(right))
