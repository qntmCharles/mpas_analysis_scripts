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
    diagnostic_profile_cache_dir,
    domain_histograms,
    duration_domain_histograms,
    lat_mean,
    load_latlon_dataset,
    materialize,
    midlevel_height_from_latlon,
    precip_rate_from_accum,
    progress,
    read_json_file,
    standardize_latlon_dataset,
    time_block_slices,
    write_json_file,
    write_netcdf_atomic,
)
from diagnostics.analysis_config import BLOCK_LENGTH_DAYS, TROPICAL_LAT_BOUNDS
from diagnostics.analysis_config import (
    ALL_SIMULATION_ARCHIVES,
    ALL_SIMULATION_LABELS,
    SIMULATION_ROOT,
    SIMULATION_PROFILES,
    canonical_simulation_name,
    profile_cache_directory,
    simulation_cache_label,
)
from diagnostics.cache_support import relabel_cached_simulations
from wavenumber_frequency_functions import smooth_wavefreq, spacetime_power


CACHE_ROOT = diagnostic_profile_cache_dir("mean_state")
SIMULATION_CACHE_ROOT = diagnostic_cache_dir("mean_state", "simulations")
TEST_TIME_COUNT = None
TEST_FAMILY_TIME_LIMITS = {
    "means": 8,
    "rain": 8,
    "distributions": 40,
    "spectra": 100,
    "eddy": 8,
}
FAMILY_SCHEMAS = {
    "means": "mean-state-means-v1",
    "rain": "mean-state-rain-v2",
    "distributions": "mean-state-distributions-v2",
    "spectra": "mean-state-spectra-v1",
    "eddy": "mean-state-eddy-v1",
}
FAMILY_FIGURES = {
    "means": {2, 3, 9, 10, 11, 12, 13, 14, 16, 17, 18, 20, 21},
    "rain": {1, 8},
    "distributions": {5, 6, 19},
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


def bounded_load(obj):
    """Load one calculation product with bounded Dask concurrency."""

    return materialize(obj)


def configure_test_mode(time_count):
    """Use an isolated cache and a bounded actual-data record."""

    global CACHE_ROOT, SIMULATION_CACHE_ROOT, TEST_TIME_COUNT
    TEST_TIME_COUNT = int(time_count)
    CACHE_ROOT = diagnostic_profile_cache_dir(
        "mean_state", "testing", f"all_{TEST_TIME_COUNT:03d}times"
    )
    SIMULATION_CACHE_ROOT = diagnostic_cache_dir(
        "mean_state",
        "testing",
        f"all_{TEST_TIME_COUNT:03d}times",
        "simulations",
    )


def requested_families(figures):
    selected = set(figures or range(1, 22))
    return {
        family for family, family_figures in FAMILY_FIGURES.items()
        if selected & family_figures
    }


def _family_time_count(family):
    """Return the shortest useful test record for one calculation family."""

    if TEST_TIME_COUNT is None:
        return None
    return min(TEST_TIME_COUNT, TEST_FAMILY_TIME_LIMITS[family])


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
        imerg_limit = 2 if TEST_TIME_COUNT is not None else 300
        paths.extend(
            sorted(glob.glob("/data/phh1-hosted/cwp29/imerg/*3IMERG.*.nc4"))[
                :imerg_limit
            ]
        )
    return paths


def family_manifest(
    root_dir, data_locs, sim_names, family, *, include_sources=True
):
    configuration = {
        "family": family,
        "simulations": [simulation_cache_label(name) for name in sim_names],
    }
    test_time_count = _family_time_count(family)
    if test_time_count is not None:
        configuration["test_source_time_count"] = test_time_count
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
            "hovmoller_fields": ["precipitation_rate", "precipw", "olrtoa"],
            "latitude_aggregation": "cosine-weighted mean",
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
            "convection_pdf": {
                "variables": {"PW": "precipw", "CAPE": "cape"},
                "latitude_bounds": list(TROPICAL_LAT_BOUNDS),
                "block_days": BLOCK_LENGTH_DAYS["mean_state_distributions"],
                "pw_bins": [0.0, 80.0, 121],
                "cape_bins": [0.0, 5000.0, 121],
                "sample_weighting": "equal grid-cell-time samples",
            },
        })
    elif family == "spectra":
        segment_days = 20 if TEST_TIME_COUNT is not None else 96
        overlap_days = 10 if TEST_TIME_COUNT is not None else 60
        configuration.update({
            "deep_tropical_latitude_bounds": [-10.0, 10.0],
            "segment_days": segment_days,
            "overlap_days": overlap_days,
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
        source_paths=(
            _source_paths(root_dir, data_locs, family)
            if include_sources else None
        ),
    )


def simulation_family_manifest(
    root_dir, data_loc, sim_name, family, *, include_sources=True
):
    """Describe one reusable simulation product independently of profiles."""

    return family_manifest(
        root_dir,
        [data_loc],
        [sim_name],
        family,
        include_sources=include_sources,
    )


def _cache_path(name):
    return os.path.join(CACHE_ROOT, name)


def _manifest_path(family):
    return _cache_path(f"{family}.manifest.json")


def _simulation_cache_dir(sim_name):
    return os.path.join(
        SIMULATION_CACHE_ROOT,
        canonical_simulation_name(sim_name),
    )


def _simulation_cache_path(sim_name, filename):
    return os.path.join(_simulation_cache_dir(sim_name), filename)


def _simulation_manifest_path(sim_name, family):
    return _simulation_cache_path(sim_name, f"{family}.manifest.json")


def simulation_family_is_current(
    root_dir,
    data_loc,
    sim_name,
    family,
    *,
    rebuild=False,
    validate_sources=True,
):
    """Return whether one profile-independent simulation family is current."""

    if rebuild:
        return False
    if not all(
        os.path.isfile(_simulation_cache_path(sim_name, filename))
        for filename in FAMILY_FILES[family]
    ):
        return False
    try:
        saved = read_json_file(_simulation_manifest_path(sim_name, family))
        expected = simulation_family_manifest(
            root_dir,
            data_loc,
            sim_name,
            family,
            include_sources=validate_sources,
        )
    except (OSError, ValueError):
        return False
    return cache_manifest_matches(
        saved,
        expected,
        compare_sources=validate_sources,
    )


def family_is_current(
    root_dir, data_locs, sim_names, family, rebuild=False,
    validate_sources=True,
):
    if rebuild:
        return False
    if not all(os.path.isfile(_cache_path(name)) for name in FAMILY_FILES[family]):
        return False
    try:
        saved = read_json_file(_manifest_path(family))
        expected = family_manifest(
            root_dir, data_locs, sim_names, family,
            include_sources=validate_sources,
        )
    except (OSError, ValueError):
        return False
    return cache_manifest_matches(
        saved, expected, compare_sources=validate_sources
    )


def _save_dataset(dataset, filename):
    path = _cache_path(filename)
    _save_dataset_at_path(dataset, path)


def _save_dataset_at_path(dataset, path):
    write_netcdf_atomic(dataset, path)


def _save_simulation_family(sim_name, family, products, manifest):
    if not isinstance(products, tuple):
        products = (products,)
    for dataset, filename in zip(products, FAMILY_FILES[family]):
        _save_dataset_at_path(
            dataset,
            _simulation_cache_path(sim_name, filename),
        )
    write_json_file(_simulation_manifest_path(sim_name, family), manifest)


def _load_simulation_family(sim_name, family):
    loaded = []
    for filename in FAMILY_FILES[family]:
        with xr.open_dataset(_simulation_cache_path(sim_name, filename)) as cached:
            loaded.append(relabel_cached_simulations(cached.load()))
    return loaded[0] if len(loaded) == 1 else tuple(loaded)


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


def _mean_products(ds, zds):
    """Calculate one simulation's compact mean-state products."""

    scalar_names = ("qv", "qc", "qr", "qi", "qs", "qg")
    lat_product = xr.Dataset()
    for name in ("olrtoa", "mslp"):
        lat_product[name] = ds[name].mean("Time", skipna=True).expand_dims(Time=[0])
    for name in scalar_names:
        lat_product[name] = ds[name].mean(("Time", "longitude"), skipna=True)
    lat_product["zmid_latitude"] = midlevel_height_from_latlon(ds)

    zgrid = ds.zgrid
    lower = zgrid.isel(nVertLevelsP1=slice(0, -1)).rename({"nVertLevelsP1": "z"})
    upper = zgrid.isel(nVertLevelsP1=slice(1, None)).rename({"nVertLevelsP1": "z"})
    z_coord = np.arange(lower.sizes["z"])
    zmid = 0.5 * (lower.assign_coords(z=z_coord) + upper.assign_coords(z=z_coord))
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
    lat_product["olr_bin_edges"] = xr.DataArray(olr_bins, dims="olr_edge")
    zonal_product = zds.mean("Time", skipna=True).expand_dims(Time=[0])
    return bounded_load(lat_product), bounded_load(zonal_product)


def _build_means(dss, dss_zm, sim_names):
    products = [_mean_products(ds, zds) for ds, zds in zip(dss, dss_zm)]
    latlon_products = [item[0] for item in products]
    zonal_products = [item[1] for item in products]

    _save_dataset(_concat_simulations(latlon_products, sim_names), "means_latlon.nc")
    _save_dataset(_concat_simulations(zonal_products, sim_names), "means_zonal.nc")


def _build_rain(dss, sim_names):
    products = [_rain_product(ds) for ds in dss]
    _save_dataset(_concat_simulations(products, sim_names), "rain.nc")


def _rain_product(ds):
    precip, _, _ = accumulated_precipitation_components(ds)
    global_accum = lat_mean(precip.mean("longitude"))
    precip_water = lat_mean(ds.precipw.mean("longitude"))
    deep_accum = lat_mean(precip.sel(latitude=slice(-10.0, 10.0)))
    deep_precip_water = lat_mean(
        ds.precipw.sel(latitude=slice(-10.0, 10.0))
    )
    deep_olr = lat_mean(ds.olrtoa.sel(latitude=slice(-10.0, 10.0)))
    return bounded_load(xr.Dataset(
        {
            "precip_water": precip_water,
            "precip_rate_global": precip_rate_from_accum(global_accum),
            "precip_rate_hovmoller": precip_rate_from_accum(deep_accum),
            "precip_water_hovmoller": deep_precip_water,
            "olr_hovmoller": deep_olr,
        }
    ))


def _histogram_array(items, bins, name):
    return xr.DataArray(
        np.stack([item[0] for item in items]),
        dims=("domain", f"{name}_bin"),
        coords={"domain": ["Global", "10S-10N"], f"{name}_bin": np.arange(bins)},
    )


def _blocked_pdf(field, bins):
    field = field.sel(latitude=slice(*TROPICAL_LAT_BOUNDS))
    rows = []
    block_slices = time_block_slices(
        field,
        block_days=BLOCK_LENGTH_DAYS["mean_state_distributions"],
        complete=True,
    )
    if not block_slices and TEST_TIME_COUNT is not None:
        block_slices = [slice(None)]
    if not block_slices:
        raise ValueError(
            "Distribution analysis requires one complete uncertainty block"
        )
    for time_slice in block_slices:
        values = np.asarray(field.isel(Time=time_slice).values, dtype=float).ravel()
        values = values[np.isfinite(values)]
        histogram = np.histogram(values, bins=bins, density=True)[0]
        rows.append(histogram)
    return np.stack(rows)


def _distribution_product(ds):
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
    product = xr.Dataset(
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
    convection_specs = {
        "PW": (ds.precipw, np.linspace(0.0, 80.0, 121)),
        "CAPE": (ds.cape, np.linspace(0.0, 5000.0, 121)),
    }
    for name, (field, edges) in convection_specs.items():
        lower = name.lower()
        product[f"{lower}_pdf_blocks"] = xr.DataArray(
            _blocked_pdf(field, edges),
            dims=("block", f"{lower}_bin"),
            coords={f"{lower}_bin": 0.5 * (edges[:-1] + edges[1:])},
        )
        product[f"{lower}_bin_edges"] = xr.DataArray(edges, dims=f"{lower}_edge")
    return bounded_load(product)


def _build_distributions(dss, sim_names):
    products = [_distribution_product(ds) for ds in dss]
    _save_distributions(products, sim_names)


def _save_distributions(products, sim_names):
    combined = _concat_simulations(products, sim_names)
    combined.update(_imerg_distribution_product())
    _save_dataset(combined, "distributions.nc")


def _imerg_distribution_product():
    """Return the shared observational distribution reference."""

    imerg_limit = 2 if TEST_TIME_COUNT is not None else 300
    imerg_paths = sorted(
        glob.glob("/data/phh1-hosted/cwp29/imerg/*3IMERG.*.nc4")
    )[:imerg_limit]
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
    return xr.Dataset(
        {
            "imerg_rain_histogram": _histogram_array(imerg_rain, 300, "rain"),
            "imerg_duration_histogram": _histogram_array(
                imerg_duration, 59, "duration"
            ),
        }
    )


def _build_spectra(dss, sim_names):
    products = [_spectrum_product(ds) for ds in dss]
    _save_dataset(_concat_simulations(products, sim_names), "spectra.nc")


def _spectrum_product(ds):
    precip, _, _ = accumulated_precipitation_components(ds)
    rate = precip_rate_from_accum(precip)
    pr = rate.sel(latitude=slice(-10.0, 10.0)).rename(
        {"latitude": "lat", "longitude": "lon"}
    )
    pr = bounded_load(cosine_weighted_power_input(pr, lat_dim="lat"))
    segment_days = 20 if TEST_TIME_COUNT is not None else 96
    overlap_days = 10 if TEST_TIME_COUNT is not None else 60
    raw = spacetime_power(
        pr,
        lataggreg="sum",
        latitude_bounds=(-10, 10),
        dosymmetries=True,
        segsize=segment_days,
        noverlap=overlap_days,
        spd=4,
    )
    sym = raw.sel(component="symmetric", drop=True)
    asym = raw.sel(component="antisymmetric", drop=True)
    return bounded_load(xr.Dataset(
        {
            "symmetric_normalised_power": sym / smooth_wavefreq(sym, freq_ax=1),
            "antisymmetric_normalised_power": asym / smooth_wavefreq(asym, freq_ax=1),
        }
    ))


def _build_eddy(dss, sim_names):
    products = [_eddy_product(ds) for ds in dss]
    _save_dataset(_concat_simulations(products, sim_names), "eddy.nc")


def _eddy_product(ds):
    pressure_levels = [100, 250, 700, 850]
    u = ds.uzonal_isobaric.sel(pres=pressure_levels).astype(np.float64)
    v = ds.umeridional_isobaric.sel(pres=pressure_levels).astype(np.float64)
    temperature = ds.t_isobaric.sel(pres=pressure_levels).astype(np.float64)
    vapour = ds.qv_isobaric.sel(pres=pressure_levels).astype(np.float64)
    dims = ("longitude", "Time")
    u_mean = u.mean(dims)
    v_mean = v.mean(dims)
    t_mean = temperature.mean(dims)
    q_mean = vapour.mean(dims)
    return bounded_load(xr.Dataset(
        {
            "eke": 0.5 * ((u * u).mean(dims) - u_mean**2 + (v * v).mean(dims) - v_mean**2),
            "momentum_flux": (u * v).mean(dims) - u_mean * v_mean,
            "heat_flux": (v * temperature).mean(dims) - v_mean * t_mean,
            "vapour_flux": 1000.0 * ((v * vapour).mean(dims) - v_mean * q_mean),
        }
    ).astype(np.float32))


BUILDERS = {
    "means": _build_means,
    "rain": _build_rain,
    "distributions": _build_distributions,
    "spectra": _build_spectra,
    "eddy": _build_eddy,
}

FAMILY_LATLON_VARIABLES = {
    "means": (
        "olrtoa", "mslp", "qv", "qc", "qr", "qi", "qs", "qg", "zgrid",
        "uzonal_isobaric", "umeridional_isobaric",
    ),
    "rain": (
        "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "olrtoa",
    ),
    "distributions": (
        "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "cape",
    ),
    "spectra": ("rainnc", "i_rainnc", "rainc", "i_rainc"),
    "eddy": ("uzonal_isobaric", "umeridional_isobaric", "t_isobaric", "qv_isobaric"),
}


def _calculate_simulation_family(
    family,
    root_dir,
    data_loc,
    sim_name,
    zonal_mean_vars,
    latlon_chunks,
    zm_chunks,
):
    """Calculate one reusable family for one simulation."""

    simulation_dir = os.path.join(root_dir, data_loc)
    progress(f"Calculating {family} products for {sim_name}")
    ds = load_latlon_dataset(
        simulation_dir,
        latlon_pattern=LATLON_PATTERN,
        chunks=latlon_chunks,
    )
    ds = standardize_latlon_dataset(ds, FAMILY_LATLON_VARIABLES[family])
    family_time_count = _family_time_count(family)
    if family_time_count is not None:
        ds = ds.isel(Time=slice(0, family_time_count))
    zds = None
    try:
        if family == "means":
            zds = load_latlon_dataset(
                simulation_dir,
                latlon_pattern=ZONAL_MEAN_PATTERN,
                chunks=zm_chunks,
            )
            zds = standardize_latlon_dataset(zds, zonal_mean_vars)
            if family_time_count is not None:
                zds = zds.isel(Time=slice(0, family_time_count))
            return _mean_products(ds, zds)
        if family == "rain":
            return _rain_product(ds)
        if family == "distributions":
            product = _distribution_product(ds)
            product.update(_imerg_distribution_product())
            return product
        if family == "spectra":
            return _spectrum_product(ds)
        if family == "eddy":
            return _eddy_product(ds)
        raise KeyError(f"Unknown mean-state family {family!r}")
    finally:
        ds.close()
        if zds is not None:
            zds.close()


def _aggregate_cache_root(profile):
    parts = ()
    if TEST_TIME_COUNT is not None:
        parts = ("testing", f"all_{TEST_TIME_COUNT:03d}times")
    return os.fspath(profile_cache_directory("mean_state", *parts, profile=profile))


def _aggregate_family_is_current(profile, family, *, validate_sources=True):
    names = SIMULATION_PROFILES[profile]
    data_locs = [ALL_SIMULATION_ARCHIVES[name] for name in names]
    sim_names = [ALL_SIMULATION_LABELS[name] for name in names]
    root = _aggregate_cache_root(profile)
    if not all(
        os.path.isfile(os.path.join(root, filename))
        for filename in FAMILY_FILES[family]
    ):
        return False
    try:
        saved = read_json_file(os.path.join(root, f"{family}.manifest.json"))
        expected = family_manifest(
            os.fspath(SIMULATION_ROOT),
            data_locs,
            sim_names,
            family,
            include_sources=validate_sources,
        )
    except (OSError, ValueError):
        return False
    return cache_manifest_matches(
        saved,
        expected,
        compare_sources=validate_sources,
    )


def _migrate_simulation_family_from_aggregate(
    root_dir,
    data_loc,
    sim_name,
    family,
    *,
    validate_sources=True,
):
    """Seed one simulation cache from any current aggregate cache."""

    canonical = canonical_simulation_name(sim_name)
    for profile, names in SIMULATION_PROFILES.items():
        if canonical not in names:
            continue
        profile_sim_names = [ALL_SIMULATION_LABELS[name] for name in names]
        if sim_name not in profile_sim_names:
            continue
        if not _aggregate_family_is_current(
            profile,
            family,
            validate_sources=validate_sources,
        ):
            continue
        aggregate_root = _aggregate_cache_root(profile)
        products = []
        for filename in FAMILY_FILES[family]:
            with xr.open_dataset(os.path.join(aggregate_root, filename)) as cached:
                cached = relabel_cached_simulations(cached)
                if "simulation" not in cached.dims:
                    products.append(cached.load())
                else:
                    products.append(
                        cached.sel(simulation=sim_name, drop=True).load()
                    )
        manifest = simulation_family_manifest(
            root_dir,
            data_loc,
            sim_name,
            family,
            include_sources=True,
        )
        _save_simulation_family(
            sim_name,
            family,
            products[0] if len(products) == 1 else tuple(products),
            manifest,
        )
        progress(
            f"Migrated {family} products for {sim_name} from the "
            f"{profile} aggregate cache"
        )
        return True
    return False


def _concat_family_products(products, sim_names, family):
    if family != "distributions":
        return _concat_simulations(products, sim_names)
    observation_names = [
        name for name in products[0].data_vars if name.startswith("imerg_")
    ]
    model_products = [
        product.drop_vars(observation_names, errors="ignore")
        for product in products
    ]
    combined = _concat_simulations(model_products, sim_names)
    for name in observation_names:
        combined[name] = products[0][name]
    return combined


def _assemble_simulation_families(sim_names, family):
    loaded = [_load_simulation_family(name, family) for name in sim_names]
    if len(FAMILY_FILES[family]) == 1:
        return _concat_family_products(loaded, sim_names, family)
    return tuple(
        _concat_family_products(
            [product[index] for product in loaded],
            sim_names,
            family,
        )
        for index in range(len(FAMILY_FILES[family]))
    )


def _build_family_from_archives(
    family,
    root_dir,
    data_locs,
    sim_names,
    zonal_mean_vars,
    latlon_chunks,
    zm_chunks,
):
    """Build one family while opening only one simulation at a time."""

    products = []
    zonal_products = []
    for data_loc, sim_name in zip(data_locs, sim_names):
        simulation_dir = os.path.join(root_dir, data_loc)
        progress(f"Calculating {family} products for {sim_name}")
        ds = load_latlon_dataset(
            simulation_dir, latlon_pattern=LATLON_PATTERN, chunks=latlon_chunks
        )
        ds = standardize_latlon_dataset(ds, FAMILY_LATLON_VARIABLES[family])
        family_time_count = _family_time_count(family)
        if family_time_count is not None:
            ds = ds.isel(Time=slice(0, family_time_count))
        zds = None
        try:
            if family == "means":
                zds = load_latlon_dataset(
                    simulation_dir,
                    latlon_pattern=ZONAL_MEAN_PATTERN,
                    chunks=zm_chunks,
                )
                zds = standardize_latlon_dataset(zds, zonal_mean_vars)
                if family_time_count is not None:
                    zds = zds.isel(Time=slice(0, family_time_count))
                lat_product, zonal_product = _mean_products(ds, zds)
                products.append(lat_product)
                zonal_products.append(zonal_product)
            elif family == "rain":
                products.append(_rain_product(ds))
            elif family == "distributions":
                products.append(_distribution_product(ds))
            elif family == "spectra":
                products.append(_spectrum_product(ds))
            elif family == "eddy":
                products.append(_eddy_product(ds))
            else:
                raise KeyError(f"Unknown mean-state family {family!r}")
        finally:
            ds.close()
            if zds is not None:
                zds.close()

    if family == "means":
        _save_dataset(_concat_simulations(products, sim_names), "means_latlon.nc")
        _save_dataset(_concat_simulations(zonal_products, sim_names), "means_zonal.nc")
    elif family == "distributions":
        _save_distributions(products, sim_names)
    else:
        _save_dataset(_concat_simulations(products, sim_names), FAMILY_FILES[family][0])


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
    """Build only stale simulation products, then assemble the active profile."""

    requested = requested_families(figures)
    missing = []
    for family in sorted(requested):
        for data_loc, sim_name in zip(data_locs, sim_names):
            if simulation_family_is_current(
                root_dir,
                data_loc,
                sim_name,
                family,
                rebuild=rebuild,
                validate_sources=not figures_only,
            ):
                continue
            migrated = False
            if not rebuild:
                migrated = _migrate_simulation_family_from_aggregate(
                    root_dir,
                    data_loc,
                    sim_name,
                    family,
                    validate_sources=not figures_only,
                )
            if not migrated:
                missing.append((family, data_loc, sim_name))

    if figures_only and missing:
        names = ", ".join(
            f"{family}/{sim_name}"
            for family, _data_loc, sim_name in missing
        )
        raise RuntimeError(
            "Figure-only mode requires current mean-state simulation caches "
            f"for: {names}. Run without --figures-only to build them."
        )

    for family, data_loc, sim_name in missing:
        progress(f"Building mean-state cache family: {family}/{sim_name}")
        product = _calculate_simulation_family(
            family,
            root_dir,
            data_loc,
            sim_name,
            zonal_mean_vars,
            latlon_chunks,
            zm_chunks,
        )
        manifest = simulation_family_manifest(
            root_dir,
            data_loc,
            sim_name,
            family,
        )
        _save_simulation_family(sim_name, family, product, manifest)

    products = {}
    for family in requested:
        products[family] = _assemble_simulation_families(
            sim_names,
            family,
        )
        progress(
            f"Using reusable mean-state simulation caches for: {family}"
        )
    return products


def split_simulations(dataset, sim_names):
    return [dataset.sel(simulation=name, drop=True) for name in sim_names]
