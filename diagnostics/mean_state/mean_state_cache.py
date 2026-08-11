"""Compact calculation products for ``mean_state_diagnostics.py``."""

from __future__ import annotations

import glob
import os
import sys

import numpy as np
import xarray as xr

_SCRIPTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
if _SCRIPTS_DIR not in sys.path:
    sys.path.append(_SCRIPTS_DIR)

from diagnostics.diagnostic_helpers import (
    LATLON_PATTERN,
    ZONAL_MEAN_PATTERN,
    accumulated_precipitation_components,
    cache_manifest,
    cache_manifest_matches,
    cosine_weighted_power_input,
    diagnostic_cache_dir,
    domain_histograms,
    duration_domain_histograms,
    lat_mean,
    load_latlon_dataset,
    midlevel_height_from_latlon,
    precip_rate_from_accum,
    read_json_file,
    standardize_latlon_dataset,
    write_json_file,
)
from wavenumber_frequency_functions import smooth_wavefreq, spacetime_power


CACHE_ROOT = diagnostic_cache_dir("mean_state_cache")
FAMILY_SCHEMAS = {
    "means": "mean-state-means-v1",
    "rain": "mean-state-rain-v1",
    "distributions": "mean-state-distributions-v1",
    "spectra": "mean-state-spectra-v1",
    "eddy": "mean-state-eddy-v1",
}
FAMILY_FIGURES = {
    "means": {2, 3, 9, 10, 11, 12, 13, 14, 16, 17, 18},
    "rain": {1, 8},
    "distributions": {5, 6},
    "spectra": {7},
    "eddy": {4},
}
FAMILY_FILES = {
    "means": ("means_latlon.nc", "means_zonal.nc"),
    "rain": ("rain.nc",),
    "distributions": ("distributions.nc",),
    "spectra": ("spectra.nc",),
    "eddy": ("eddy.nc",),
}


def requested_families(figures):
    selected = set(figures or range(1, 19))
    return {
        family for family, family_figures in FAMILY_FIGURES.items()
        if selected & family_figures
    }


def _source_paths(root_dir, data_locs, family):
    patterns = [LATLON_PATTERN]
    if family == "means":
        patterns.append(ZONAL_MEAN_PATTERN)
    paths = [
        path
        for data_loc in data_locs
        for pattern in patterns
        for path in sorted(glob.glob(os.path.join(root_dir, data_loc, pattern)))
    ]
    if family == "distributions":
        paths.extend(sorted(glob.glob("/data/phh1-hosted/cwp29/imerg/*3IMERG.*.nc4"))[:300])
    return paths


def family_manifest(root_dir, data_locs, sim_names, family):
    configuration = {
        "family": family,
        "simulations": list(sim_names),
    }
    if family == "means":
        configuration.update({
            "time_reduction": "mean",
            "horizontal_reduction": "cosine-weighted where global",
            "olr_histogram_edges": [100.0, 350.0, 80],
            "isobaric_eke": "resolved-time-longitude covariance",
        })
    elif family == "rain":
        configuration.update({
            "deep_tropical_latitude_bounds": [-10.0, 10.0],
            "rate_units": "mm day-1",
        })
    elif family == "distributions":
        configuration.update({
            "deep_tropical_latitude_bounds": [-10.0, 10.0],
            "precipitation_histogram": {"bins": 300, "range": [0.0, 300.0]},
            "duration_histogram": {
                "threshold": 5.0,
                "bins": 59,
                "range": [1.0, 60.0],
            },
            "imerg_spatial_coarsening": [5, 5],
        })
    elif family == "spectra":
        configuration.update({
            "deep_tropical_latitude_bounds": [-10.0, 10.0],
            "segment_size": 96,
            "overlap": 60,
            "samples_per_day": 4,
            "latitude_aggregation": "cosine-weighted symmetric sum",
        })
    elif family == "eddy":
        configuration.update({
            "pressure_levels_hpa": [100.0, 250.0, 700.0, 850.0],
            "vapour_flux_scale": 1000.0,
            "covariance_precision": "float64",
        })
    return cache_manifest(
        "mean_state_diagnostics.py",
        schema=FAMILY_SCHEMAS[family],
        configuration=configuration,
        source_paths=_source_paths(root_dir, data_locs, family),
    )


def _cache_path(name):
    return os.path.join(CACHE_ROOT, name)


def _manifest_path(family):
    return _cache_path(f"{family}.manifest.json")


def family_is_current(root_dir, data_locs, sim_names, family, rebuild=False):
    if rebuild:
        return False
    if not all(os.path.isfile(_cache_path(name)) for name in FAMILY_FILES[family]):
        return False
    try:
        saved = read_json_file(_manifest_path(family))
        expected = family_manifest(root_dir, data_locs, sim_names, family)
    except (OSError, ValueError):
        return False
    return cache_manifest_matches(saved, expected)


def _save_dataset(dataset, filename):
    path = _cache_path(filename)
    temporary = path + ".tmp"
    dataset.to_netcdf(temporary)
    os.replace(temporary, path)


def _concat_simulations(datasets, sim_names):
    return xr.concat(
        datasets,
        dim=xr.IndexVariable("simulation", sim_names),
        join="outer",
        combine_attrs="drop_conflicts",
    )


def _isobaric_eke(ds):
    u = ds.uzonal_isobaric.astype(np.float64)
    v = ds.umeridional_isobaric.astype(np.float64)
    dims = [dim for dim in ("Time", "longitude") if dim in u.dims]
    u_mean = u.mean(dim=dims, skipna=True)
    v_mean = v.mean(dim=dims, skipna=True)
    eke = 0.5 * (
        (u * u).mean(dim=dims, skipna=True) - u_mean * u_mean
        + (v * v).mean(dim=dims, skipna=True) - v_mean * v_mean
    )
    eke.name = "eke_isobaric"
    return eke.astype(np.float32)


def _build_means(dss, dss_zm, sim_names):
    latlon_products = []
    zonal_products = []
    scalar_names = ("qv", "qc", "qr", "qi", "qs", "qg")
    for ds, zds in zip(dss, dss_zm):
        lat_product = xr.Dataset()
        for name in ("olrtoa", "mslp"):
            lat_product[name] = ds[name].mean("Time", skipna=True).expand_dims(Time=[0])
        for name in scalar_names:
            lat_product[name] = ds[name].mean(("Time", "longitude"), skipna=True)
        lat_product["zmid_latitude"] = midlevel_height_from_latlon(ds)

        zgrid = ds.zgrid
        lower = zgrid.isel(nVertLevelsP1=slice(0, -1)).rename(
            {"nVertLevelsP1": "z"}
        )
        upper = zgrid.isel(nVertLevelsP1=slice(1, None)).rename(
            {"nVertLevelsP1": "z"}
        )
        z_coord = np.arange(lower.sizes["z"])
        zmid = 0.5 * (
            lower.assign_coords(z=z_coord) + upper.assign_coords(z=z_coord)
        )
        weights = np.cos(np.deg2rad(zmid.latitude)).clip(min=0.0)
        lat_product["zgrid_zonal_mean"] = zmid.mean("longitude", skipna=True)
        lat_product["zgrid_global_mean"] = zmid.weighted(weights).mean(
            ("latitude", "longitude"), skipna=True
        )
        lat_product["zgrid_global_std"] = zmid.weighted(weights).std(
            ("latitude", "longitude"), skipna=True
        )
        lat_product["zgrid_global_min"] = zmid.min(("latitude", "longitude"), skipna=True)
        lat_product["zgrid_global_max"] = zmid.max(("latitude", "longitude"), skipna=True)
        lat_product["eke_isobaric"] = _isobaric_eke(ds)

        olr_bins = np.linspace(100.0, 350.0, 80)
        olr_histograms = domain_histograms(ds.olrtoa, bins=olr_bins)
        lat_product["olr_histogram"] = xr.DataArray(
            np.stack([item[0] for item in olr_histograms]),
            dims=("domain", "olr_bin"),
            coords={"domain": ["Global", "10S-10N"], "olr_bin": np.arange(79)},
        )
        lat_product["olr_bin_edges"] = xr.DataArray(
            olr_bins, dims="olr_edge", coords={"olr_edge": np.arange(80)}
        )
        latlon_products.append(lat_product.load())

        zonal_product = zds.mean("Time", skipna=True).expand_dims(Time=[0])
        zonal_products.append(zonal_product.load())

    _save_dataset(_concat_simulations(latlon_products, sim_names), "means_latlon.nc")
    _save_dataset(_concat_simulations(zonal_products, sim_names), "means_zonal.nc")


def _build_rain(dss, sim_names):
    products = []
    for ds in dss:
        precip, _, _ = accumulated_precipitation_components(ds)
        global_accum = lat_mean(precip.mean("longitude"))
        precip_water = lat_mean(ds.precipw.mean("longitude"))
        deep_accum = lat_mean(precip.sel(latitude=slice(-10.0, 10.0)))
        products.append(
            xr.Dataset(
                {
                    "precip_water": precip_water,
                    "precip_rate_global": precip_rate_from_accum(global_accum),
                    "precip_rate_hovmoller": precip_rate_from_accum(deep_accum),
                }
            ).load()
        )
    _save_dataset(_concat_simulations(products, sim_names), "rain.nc")


def _histogram_array(items, bins, name):
    return xr.DataArray(
        np.stack([item[0] for item in items]),
        dims=("domain", f"{name}_bin"),
        coords={"domain": ["Global", "10S-10N"], f"{name}_bin": np.arange(bins)},
    )


def _build_distributions(dss, sim_names):
    products = []
    for ds in dss:
        precip = accumulated_precipitation_components(ds)
        rain_hists = []
        duration_hists = []
        for component in precip:
            rate = precip_rate_from_accum(component)
            rain_hists.append(domain_histograms(rate, bins=300, value_range=(0, 300)))
            duration_hists.append(
                duration_domain_histograms(
                    rate, threshold=5, bins=59, value_range=(1, 60)
                )
            )
        products.append(
            xr.Dataset(
                {
                    "rain_histogram": xr.concat(
                        [_histogram_array(item, 300, "rain") for item in rain_hists],
                        dim=xr.IndexVariable("component", ["total", "convective", "resolved"]),
                    ),
                    "duration_histogram": xr.concat(
                        [_histogram_array(item, 59, "duration") for item in duration_hists],
                        dim=xr.IndexVariable("component", ["total", "convective", "resolved"]),
                    ),
                    "rain_bin_edges": xr.DataArray(np.linspace(0, 300, 301), dims="rain_edge"),
                    "duration_bin_edges": xr.DataArray(np.linspace(1, 60, 60), dims="duration_edge"),
                }
            )
        )

    imerg_paths = sorted(glob.glob("/data/phh1-hosted/cwp29/imerg/*3IMERG.*.nc4"))[:300]
    with xr.open_mfdataset(
        imerg_paths, concat_dim="time", combine="nested", engine="netcdf4"
    ) as imerg_ds:
        from diagnostics.diagnostic_helpers import cosine_weighted_latitude_coarsen

        imerg_pr = cosine_weighted_latitude_coarsen(
            imerg_ds.precipitation, 5, lat_dim="lat"
        ).coarsen(lon=5, boundary="trim").mean()
        imerg_rain = domain_histograms(imerg_pr, bins=300, value_range=(0, 300))
        imerg_duration = duration_domain_histograms(
            imerg_pr, threshold=5, bins=59, value_range=(1, 60)
        )

    combined = _concat_simulations(products, sim_names)
    combined["imerg_rain_histogram"] = _histogram_array(imerg_rain, 300, "rain")
    combined["imerg_duration_histogram"] = _histogram_array(
        imerg_duration, 59, "duration"
    )
    _save_dataset(combined, "distributions.nc")


def _build_spectra(dss, sim_names):
    products = []
    for ds in dss:
        precip, _, _ = accumulated_precipitation_components(ds)
        rate = precip_rate_from_accum(precip)
        pr = rate.sel(latitude=slice(-10.0, 10.0)).rename(
            {"latitude": "lat", "longitude": "lon"}
        )
        pr = cosine_weighted_power_input(pr, lat_dim="lat").load()
        raw = spacetime_power(
            pr,
            lataggreg="sum",
            latitude_bounds=(-10, 10),
            dosymmetries=True,
            segsize=96,
            noverlap=60,
            spd=4,
        )
        sym = raw.sel(component="symmetric", drop=True)
        asym = raw.sel(component="antisymmetric", drop=True)
        products.append(
            xr.Dataset(
                {
                    "symmetric_normalised_power": sym / smooth_wavefreq(sym, freq_ax=1),
                    "antisymmetric_normalised_power": asym / smooth_wavefreq(asym, freq_ax=1),
                }
            ).load()
        )
    _save_dataset(_concat_simulations(products, sim_names), "spectra.nc")


def _build_eddy(dss, sim_names):
    products = []
    pressure_levels = [100, 250, 700, 850]
    for ds in dss:
        u = ds.uzonal_isobaric.sel(pres=pressure_levels).astype(np.float64)
        v = ds.umeridional_isobaric.sel(pres=pressure_levels).astype(np.float64)
        temperature = ds.t_isobaric.sel(pres=pressure_levels).astype(np.float64)
        vapour = ds.qv_isobaric.sel(pres=pressure_levels).astype(np.float64)
        dims = ("longitude", "Time")
        u_mean = u.mean(dims)
        v_mean = v.mean(dims)
        t_mean = temperature.mean(dims)
        q_mean = vapour.mean(dims)
        products.append(
            xr.Dataset(
                {
                    "eke": 0.5 * ((u * u).mean(dims) - u_mean**2 + (v * v).mean(dims) - v_mean**2),
                    "momentum_flux": (u * v).mean(dims) - u_mean * v_mean,
                    "heat_flux": (v * temperature).mean(dims) - v_mean * t_mean,
                    "vapour_flux": 1000.0 * (
                        (v * vapour).mean(dims) - v_mean * q_mean
                    ),
                }
            ).astype(np.float32).load()
        )
    _save_dataset(_concat_simulations(products, sim_names), "eddy.nc")


BUILDERS = {
    "means": _build_means,
    "rain": _build_rain,
    "distributions": _build_distributions,
    "spectra": _build_spectra,
    "eddy": _build_eddy,
}


def prepare_products(
    root_dir,
    data_locs,
    sim_names,
    figures,
    latlon_vars,
    zonal_mean_vars,
    latlon_chunks,
    zm_chunks,
    *,
    figures_only=False,
    rebuild=False,
):
    """Build stale families, then return only compact plotting products."""

    requested = requested_families(figures)
    current = {
        family for family in requested
        if family_is_current(root_dir, data_locs, sim_names, family, rebuild=rebuild)
    }
    stale = requested - current
    if figures_only and stale:
        names = ", ".join(sorted(stale))
        raise RuntimeError(
            f"Figure-only mode requires current mean-state caches for: {names}. "
            "Run without --figures-only to rebuild them."
        )

    if stale:
        os.makedirs(CACHE_ROOT, exist_ok=True)
        need_latlon = bool(stale)
        need_zonal = "means" in stale
        dss = []
        dss_zm = []
        for data_loc in data_locs:
            simulation_dir = os.path.join(root_dir, data_loc)
            if need_latlon:
                ds = load_latlon_dataset(
                    simulation_dir, latlon_pattern=LATLON_PATTERN, chunks=latlon_chunks
                )
                dss.append(standardize_latlon_dataset(ds, latlon_vars))
            if need_zonal:
                zds = load_latlon_dataset(
                    simulation_dir, latlon_pattern=ZONAL_MEAN_PATTERN, chunks=zm_chunks
                )
                dss_zm.append(standardize_latlon_dataset(zds, zonal_mean_vars))

        for family in sorted(stale):
            print(f"Building mean-state cache family: {family}")
            if family == "means":
                BUILDERS[family](dss, dss_zm, sim_names)
            else:
                BUILDERS[family](dss, sim_names)
            write_json_file(
                _manifest_path(family),
                family_manifest(root_dir, data_locs, sim_names, family),
            )
        for dataset in dss + dss_zm:
            dataset.close()

    products = {}
    for family in requested:
        loaded = []
        for filename in FAMILY_FILES[family]:
            with xr.open_dataset(_cache_path(filename)) as cached:
                loaded.append(cached.load())
        products[family] = loaded[0] if len(loaded) == 1 else tuple(loaded)
        print(f"Using current mean-state cache family: {family}")
    return products


def split_simulations(dataset, sim_names):
    return [dataset.sel(simulation=name, drop=True) for name in sim_names]
