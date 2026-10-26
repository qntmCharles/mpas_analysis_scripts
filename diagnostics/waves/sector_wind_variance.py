"""Absolute horizontal-wind variance in the existing broad spectral sectors."""

from __future__ import annotations

import glob
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr
from scipy import fft

from diagnostics.analysis_config import simulation_cache_name, simulation_colour
from diagnostics.cache_support import cache_manifest, materialize, read_json_file, write_json_file, write_netcdf_atomic
from diagnostics.diagnostic_helpers import progress
from diagnostics.mean_state import circulation_mechanism as mechanism
from diagnostics.mean_state.circulation_bootstrap import hemisphere_samples
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.tem_core import TEM_FORMULATION, TEM_VERSION
from diagnostics.waves.publication_wave_forcing import (
    _cached_product, _native_profile_stage, _read_staged_height,
)
from diagnostics.waves.regional_epfd_spectrum import control_region_weights, mechanism_latitude_indices
from diagnostics.waves.regional_ep_flux_sectors import (
    FREQUENCY_CUTOFF, WAVENUMBER_CUTOFF, positive_spatial_masks,
)

VARIANCE_VERSION = "sector-horizontal-wind-variance-v2-time-samples"
SECTOR_NAMES = ("low_eastward", "low_westward", "high_wavenumber")
LATITUDE_LIMIT = 45.
PLOT_LATITUDE_LIMIT = 45.
VARIANCE_PRESSURES = np.unique(np.r_[np.arange(100., 351., 10.), mechanism.REGION_PRESSURES])
def variance_masks(time_count, longitude_count):
    names, masks = positive_spatial_masks(time_count, longitude_count, np.arange(longitude_count//2+1))
    return np.stack([masks[names.index("low_eastward")], masks[names.index("low_westward")],
        masks[names.index("high_wavenumber_eastward")] | masks[names.index("high_wavenumber_westward")]])


def filtered_wind_variance(spatial, masks, longitude_count):
    """Square filtered velocities; use exact longitude Parseval multiplicities.

    Input axes: time, latitude, wind=(u,v), nonnegative spatial mode.
    Output axes: sector, time, latitude. Time-local values remain nonnegative.
    """
    nt, nlat, winds, modes = spatial.shape
    if winds != 2 or modes != longitude_count//2+1 or masks.shape[1:] != (nt, modes):
        raise ValueError("Horizontal-wind Fourier records and sector masks differ")
    transformed = fft.fft(np.ascontiguousarray(spatial.transpose(1, 2, 3, 0)), axis=-1, workers=2)
    multiplicity = np.full(modes, 2./longitude_count**2)
    multiplicity[0] = 1./longitude_count**2
    if longitude_count % 2 == 0:
        multiplicity[-1] = 1./longitude_count**2
    output = np.empty((len(masks), nt, nlat))
    for index, mask in enumerate(masks):
        filtered = fft.ifft(transformed*mask.T[None, None], axis=-1, workers=2)
        power = filtered.real**2+filtered.imag**2
        output[index] = np.einsum("lvkt,k->tl", power, multiplicity)
    return output


def compute_variance_sections(stage, background, longitude_count, fixed_weights, directory=None, *,
        pressures=VARIANCE_PRESSURES, wind_indices=(0,1), latitudes=None):
    """Retain native-time variance until pressure sampling and Control weighting."""
    nt, nz, nlat = [background.sizes[name] for name in ("Time", "z", "latitude")]
    pressures=np.asarray(pressures)
    if stage.shape[:3]!=(nt,nz,nlat) or stage.shape[-1]!=longitude_count//2+1 or len(wind_indices)!=2:
        raise ValueError("Wind-variance stage and stored TEM coordinates differ")
    masks = variance_masks(nt, longitude_count)
    shape = (nz, len(SECTOR_NAMES), nt, nlat)
    state_path = Path(directory)/"variance-progress.json" if directory else None
    previous = read_json_file(state_path) or {} if state_path else {}
    signature = (read_json_file(Path(directory)/"progress.json") or {}).get("manifest_sha256") if directory else None
    count = int(previous.get("complete_heights", 0))
    if directory:
        variance_path = Path(directory)/"variance.dat"
        resume = (variance_path.exists() and previous.get("shape") == list(shape) and 0 <= count <= nz
            and previous.get("manifest_sha256") == signature)
        if not resume:
            count = 0
        values = np.memmap(variance_path, dtype="float64", mode="r+" if resume else "w+", shape=shape)
    else:
        values = np.empty(shape)
    try:
        for level in range(count, nz):
            started = perf_counter()
            plane = _read_staged_height(stage, level)
            if stage.shape[3]!=2 or tuple(wind_indices)!=(0,1):
                plane=plane[:,:,list(wind_indices),:]
            for left in range(0, nlat, 8):
                values[level, :, :, left:left+8] = filtered_wind_variance(plane[:, left:left+8], masks, longitude_count)
            del plane
            if directory:
                values.flush()
                write_json_file(state_path, dict(shape=list(shape), complete_heights=level+1, manifest_sha256=signature))
            progress(f"Wind variance: height {level+1}/{nz} complete ({perf_counter()-started:.1f} s)")
        variance = xr.DataArray(np.asarray(values), dims=("z", "sector", "Time", "latitude"),
            coords=dict(z=background.z, sector=list(SECTOR_NAMES), Time=background.Time, latitude=background.latitude))
        latitude = np.unique(np.r_[background.latitude.values[background.latitude.values >= 0], mechanism.REGION_LATITUDES])
        point = fixed_weights.rename(region_pres="pres", region_latitude="absolute_latitude")
        hemisphere_weights = (point/point.sum("hemisphere")).interp(
            pres=xr.DataArray(np.clip(pressures, *mechanism.REGION_PRESSURE), dims="pres"),
            absolute_latitude=xr.DataArray(np.clip(latitude, *mechanism.REGION_LATITUDE), dims="absolute_latitude")
        ).assign_coords(pres=pressures, absolute_latitude=latitude)
        pressure = hemisphere_samples(background.pressure, "even")
        def fold(field):
            sampled = mechanism._on_pressure(hemisphere_samples(field, "even"), pressure,
                pressures, "pres").interp(absolute_latitude=latitude)
            return (sampled*hemisphere_weights).sum("hemisphere", skipna=False)
        if latitudes is None:
            samples, winds = fold(variance), fold(background.u_bar)
        else:
            from diagnostics.waves.wave_uncertainty import PressureSampler
            sampler = PressureSampler(background, pressures, latitudes, fixed_weights)
            samples = xr.concat([sampler.samples(variance.sel(sector=sector, drop=True))
                for sector in SECTOR_NAMES], dim=xr.IndexVariable("sector", list(SECTOR_NAMES)))
            winds = sampler.samples(background.u_bar)
        product = xr.Dataset(dict(horizontal_wind_variance=samples.mean("Time"), zonal_wind=winds.mean("Time"),
            horizontal_wind_variance_samples=samples, zonal_wind_samples=winds))
        product.horizontal_wind_variance.attrs.update(units="m2 s-2", definition="zonal mean of filtered u squared plus filtered v squared")
        product.zonal_wind.attrs.update(units="m s-1", definition="time-mean Eulerian zonal velocity; no cosine conversion")
        product.attrs.update(product_version=VARIANCE_VERSION, tem_formulation=TEM_FORMULATION,
            time_sample_count=nt, longitude_sample_count=longitude_count,
            spectral_preprocessing="zonal-mean removal; no detrending, tapering, or amplitude normalization",
            pressure_sampling="instantaneous pressure projection before time averaging",
            hemisphere_weighting="fixed Control mass, matching publication mechanism maps")
        return product.load()
    finally:
        if isinstance(values, np.memmap):
            values._mmap.close()


def load_or_build_wind_variance(*, simulation, archive_dir, control_archive_dir, cache_dir, pattern,
        start, end, tem_root=None, figures_only=False, rebuild=False, scratch_root=None, mean_fields_only=False):
    cache = Path(cache_dir)/"epfd"
    path = cache/f"sector_wind_variance_{simulation_cache_name(simulation)}.nc"
    sources = sorted(glob.glob(str(Path(archive_dir)/pattern)))+tem_source_paths(Path(archive_dir).name, root=tem_root)
    sources += tem_source_paths(Path(control_archive_dir).name, root=tem_root)
    manifest = cache_manifest(__name__, schema=VARIANCE_VERSION,
        configuration=dict(simulation=simulation, start=str(start), end=str(end), latitude_limit=LATITUDE_LIMIT,
            pressure_levels_hpa=VARIANCE_PRESSURES.tolist(), latitude_stride_outside_box=4,
            frequency_cutoff=FREQUENCY_CUTOFF, wavenumber_cutoff=WAVENUMBER_CUTOFF,
            quantity="filtered_u_squared_plus_filtered_v_squared"),
        dependencies=dict(tem_version=TEM_VERSION), source_paths=sources)
    if not rebuild:
        cached = _cached_product(path, manifest, figures_only,
            variables=("horizontal_wind_variance","zonal_wind") if mean_fields_only else None)
        if cached is not None:
            progress(f"{simulation}: using cached horizontal-wind variance: {path}")
            return cached
    if figures_only:
        raise FileNotFoundError(f"Missing horizontal-wind variance cache: {path}; calculate Figure 19 without --figures-only")
    import fcntl
    cache.mkdir(parents=True, exist_ok=True)
    with (cache/f".wind_variance_{simulation_cache_name(simulation)}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        upstream = load_stored_tem(Path(archive_dir).name, formulation=TEM_FORMULATION, root=tem_root, start=start, end=end)
        control = load_stored_tem(Path(control_archive_dir).name, formulation=TEM_FORMULATION, root=tem_root, start=start, end=end)
        try:
            control_box = materialize(control[["rho_zm", "pressure"]].isel(latitude=mechanism_latitude_indices(control.latitude)))
            fixed = control_region_weights(control_box)
            background = materialize(upstream[["pressure", "u_bar"]].sel(latitude=slice(-LATITUDE_LIMIT-1., LATITUDE_LIMIT+1.)))
            positive = background.latitude.values[background.latitude.values > 0]
            retained = np.unique(np.r_[positive[::4], positive[-1], positive[(positive >= 3.5) & (positive <= 16.5)]])
            background = background.sel(latitude=np.sort(np.r_[-retained, retained]))
            pressure = background.pressure.transpose("Time", "z", "latitude").values
            levels = np.flatnonzero(np.any((pressure >= 10000.) & (pressure <= 35000.), axis=(0, 2)))
            if not levels.size:
                raise ValueError("Stored TEM pressure does not cover 100–350 hPa")
            background = background.isel(z=slice(max(0, int(levels.min())-1), min(background.sizes["z"], int(levels.max())+2)))
            progress(f"{simulation}: horizontal-wind variance calculation; read u and v only; reuse the complete-record native reader")
            stage, nlon, directory = _native_profile_stage(archive_dir, pattern, start, end, upstream,
                background, manifest, scratch_root, fields=("u", "v"))
            try:
                product = compute_variance_sections(stage, background, nlon, fixed, directory)
                product.attrs.update(simulation=simulation, cache_manifest=json.dumps(manifest, sort_keys=True),
                    manifest_sha256=manifest["manifest_sha256"])
                write_netcdf_atomic(product, path)
            finally:
                stage._mmap.close()
            # Remove only this successfully completed calculation's scratch records.
            for filename in ("fourier.dat", "progress.json", "variance.dat", "variance-progress.json"):
                (directory/filename).unlink()
            for entry in directory.glob("reader*.json"):
                entry.unlink()
            directory.rmdir()
            progress(f"{simulation}: horizontal-wind variance cache complete: {path}")
            return product
        finally:
            upstream.close()
            control.close()


def plot_wind_variance(products, confidence=None, circulation_guides=None):
    from diagnostics.waves.wave_response_plotting import WIND_LEVELS, plot_circulation_guides, circulation_guide_handles
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D
    from matplotlib.ticker import NullFormatter
    from diagnostics.diagnostic_helpers import add_symmetric_colorbar, difference_cmap, nice_levels

    from diagnostics.waves.wave_response_plotting import equatorial_display_field, plot_region_box

    control, nudged = xr.align(products["control"], products["nudged-u"], join="exact")
    fig, axes = plt.subplots(2, 3, figsize=(8, 6.2), sharex=True, sharey=True, layout="constrained")
    runs = ((control, "control", "Control", "--"), (nudged, "nudged-u", "U", "-"))
    panels = ((control.horizontal_wind_variance, "Control"),
        (nudged.horizontal_wind_variance-control.horizontal_wind_variance, "U − Control"))
    titles = ("Low-frequency eastward", "Low-frequency westward", "High wavenumbers")
    for column, sector in enumerate(SECTOR_NAMES):
        for row, (variance, label) in enumerate(panels):
            axis = axes[row, column]
            field = equatorial_display_field(variance.sel(sector=sector)).transpose("pres", "absolute_latitude")
            maximum = float(abs(field.sel(absolute_latitude=slice(0, PLOT_LATITUDE_LIMIT))).max()) or 1.
            if row == 0:
                levels = nice_levels(0., maximum, n=17)
                colours = plt.get_cmap("YlOrBr")(np.linspace(0., 1., len(levels)-1))
                colours[0] = (1., 1., 1., 1.)
                cmap = ListedColormap(colours)
            else:
                levels = nice_levels(-maximum, maximum, n=17)
                cmap = difference_cmap(levels, cmap="RdBu_r", white_zero_boundary=True)
            image = axis.contourf(field.absolute_latitude, field.pres, field, levels=levels,
                cmap=cmap, norm=BoundaryNorm(levels, cmap.N), extend="max" if row == 0 else "both")
            if row == 1 and confidence is not None:
                from diagnostics.waves.wave_response_plotting import hatch_insignificant
                hatch_insignificant(axis,confidence.significant.sel(sector=sector),variance.sel(sector=sector),contrast_shading=(field,image))
            for background, name, _, style in runs:
                wind = background.zonal_wind.transpose("pres", "absolute_latitude")
                wind_levels=[level for level in WIND_LEVELS[sector] if float(wind.min())<level<float(wind.max())]
                if wind_levels:
                    contours = axis.contour(wind.absolute_latitude, wind.pres, wind,
                        levels=wind_levels, colors=[simulation_colour(name)], linestyles=style, linewidths=.65)
                    axis.clabel(contours, fmt="%g", fontsize=5.5, inline=True)
            plot_circulation_guides(axis,circulation_guides,run=0 if row==0 else 1,reference=None if row==0 else 0)
            plot_region_box(axis)
            axis.set_title(f"({'abcdef'[row*3+column]}) {label}\n{titles[column]}", fontsize=7.5)
            axis.set_yscale("log")
            axis.set_ylim(350, 100)
            axis.set_yticks([350, 300, 250, 200, 170, 150, 100], labels=["350", "300", "250", "200", "170", "150", "100"])
            axis.yaxis.set_minor_formatter(NullFormatter())
            axis.set_xlim(0, PLOT_LATITUDE_LIMIT)
            axis.set_xticks(np.arange(0, PLOT_LATITUDE_LIMIT+1, 10))
            if column == 0:
                axis.set_ylabel("Pressure (hPa)")
            if row == 1:
                axis.set_xlabel(r"Latitude ($^\circ$)")
            colourbar_label = "Horizontal-wind variance (m²/s²)" if row == 0 else "Variance change (m²/s²)"
            add_symmetric_colorbar(fig, image, axis, levels, colourbar_label,
                location="bottom", ticks_at_boundaries=True, nticks=4, shrink=.96, aspect=28)
    handles = [Line2D([], [], color=simulation_colour(name), ls=style, label=f"{label} wind (m/s); signs match wave band")
        for _, name, label, style in runs]
    if circulation_guides is not None:
        handles.extend(circulation_guide_handles())
    fig.legend(handles=handles, loc="outside lower center", ncol=2, frameon=False)
    fig.suptitle("Horizontal-wind variance: Control and U − Control\n"
        "Hatching: pointwise 95% intervals include zero; 40-day blocks", fontsize=8.5)
    return fig
