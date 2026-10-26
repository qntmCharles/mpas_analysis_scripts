"""Compact regional checks for the circulation workflow's mechanism figure.

EPFd is an acceleration, whereas the divergence theorem applies to EP torque.
Boundary integrals therefore retain the spherical lever arm a*cos(latitude).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from diagnostics.diagnostic_helpers import progress
from diagnostics.mean_state.circulation_bootstrap import hemisphere_samples
from diagnostics.tem_core import EARTH_RADIUS, SCALE_HEIGHT_M

REGION_LATITUDE = (5.0, 15.0)
REGION_PRESSURE = (100.0, 170.0)
REGION_BOX_STYLE = dict(edgecolor="#7b3294", linestyle="--", linewidth=1.1)
REGION_DESCRIPTION = (
    f"{REGION_LATITUDE[0]:g}-{REGION_LATITUDE[1]:g} degrees absolute latitude; "
    f"{REGION_PRESSURE[0]:g}-{REGION_PRESSURE[1]:g} hPa"
)
HC_LATITUDE = (0.0, 25.0)
HC_PRESSURE = (200.0, 700.0)
PRECIP_LATITUDE = (-10.0, 10.0)
REGION_LATITUDES = np.linspace(*REGION_LATITUDE, 61)
REGION_PRESSURES = np.linspace(*REGION_PRESSURE, 81)
BUDGET_TERMS = (
    "a_ep", "a_vres", "a_wres", "unresolved_direct_momentum_residual",
    "a_tendency", "a_density_storage",
)
BOUNDARIES = ("lower", "upper", "equatorward", "poleward")
PRIMITIVES = (*BUDGET_TERMS, "a_ep_meridional", "a_ep_vertical", "vres", "eta", "wres", "u_z")
MMC_DECOMPOSITIONS = {
    "av": ("av_flow", "av_vorticity", "av_interaction"),
    "aw": ("aw_flow", "aw_shear", "aw_interaction"),
}


def annotate_products(dataset):
    """Preserve units after additive temporal reductions."""
    for name in dataset.data_vars:
        if name.startswith("a_") or name == "unresolved_direct_momentum_residual":
            dataset[name].attrs.update(units="m s-2", sign_convention="eastward-positive acceleration")
        elif name.startswith("boundary_") or name.startswith("torque_a_"):
            dataset[name].attrs.update(units="N m", sign_convention="outward-positive EP torque")
    dataset.vres.attrs.update(units="m s-1", sign_convention="poleward-positive after SH sign reversal")
    dataset.eta.attrs.update(units="s-1", definition="f+zeta; SH sign reversed before hemispheric averaging")
    dataset.wres.attrs.update(units="m s-1", sign_convention="upward-positive")
    dataset.u_z.attrs.update(units="s-1", definition="archived native height-coordinate du/dz used in A_wres")
    dataset.mass_weight.attrs.update(units="kg m-2 Pa-1", definition="rho*(-dz/dp)*cos(phi)")
    dataset.torque_normalizer.attrs.update(units="kg m", definition="integral_R rho*a*cos(phi) dV")
    dataset.hc_streamfunction.attrs.update(units="kg s-1", definition="mean native residual streamfunction in each simulation/hemisphere")
    dataset.hc_pressure.attrs.update(units="hPa")
    dataset.precipitation.attrs.update(units="mm day-1", definition="cosine-area-mean precipitation, 10S-10N")
    dataset.region_pres.attrs.update(units="hPa")
    dataset.region_latitude.attrs.update(units="degrees_north_absolute")
    return dataset


def _pressure_column(values, pressure, target):
    """Interpolate valid adjacent levels without extrapolation or gap filling."""
    pressure = np.asarray(pressure)
    valid = np.isfinite(pressure) & (pressure > 0)
    p, values = pressure[valid], np.asarray(values)[valid]
    order = np.argsort(p)
    p, values = p[order], values[order]
    if p.size < 2:
        return np.full(np.asarray(target).shape, np.nan)
    if np.any(np.diff(p) <= 0):
        raise ValueError("Mechanism diagnostics require strictly monotonic pressure columns")
    result = np.interp(target, p, values, left=np.nan, right=np.nan)
    return result


def _on_pressure(field, pressure, target, pressure_dim):
    """Retain time and hemispheres while interpolating archived height columns."""
    if hasattr(field.data, "chunks"):
        field = field.chunk({"z": -1})
        pressure = pressure.chunk({"z": -1})
    result = xr.apply_ufunc(
        _pressure_column, field, pressure / 100.0,
        input_core_dims=[["z"], ["z"]], output_core_dims=[[pressure_dim]],
        kwargs={"target": np.asarray(target)}, vectorize=True,
        dask="parallelized", output_dtypes=[float],
        dask_gufunc_kwargs={"output_sizes": {pressure_dim: len(target)}},
    )
    return result.assign_coords({pressure_dim: target})


def _integral(field, dimension, coordinate):
    return field.assign_coords({dimension: coordinate}).integrate(dimension)


def sample_mechanism_field(field, pressure, parity="even"):
    """Sample one native TEM field on the fixed mechanism grid in both hemispheres."""
    field = hemisphere_samples(field, parity)
    pressure = hemisphere_samples(pressure, "even")
    return _sample_mechanism_grid(field, pressure)


def _sample_mechanism_grid(field, pressure):
    """Restrict native columns before shared pressure and latitude interpolation."""
    latitude = pressure.absolute_latitude.values
    if latitude[0] > REGION_LATITUDE[0] or latitude[-1] < REGION_LATITUDE[1]:
        raise ValueError("The TEM archive does not cover the fixed mechanism latitude box")
    section = slice(max(0, np.searchsorted(latitude, REGION_LATITUDE[0])-1),
                    min(latitude.size, np.searchsorted(latitude, REGION_LATITUDE[1], side="right")+1))
    return _on_pressure(field.isel(absolute_latitude=section), pressure.isel(absolute_latitude=section),
                        REGION_PRESSURES, "region_pres").interp(
        absolute_latitude=REGION_LATITUDES).rename(absolute_latitude="region_latitude")


def regional_tem_samples(diagnostics, archive):
    """Reduce archived TEM samples to small pressure boxes and boundary torques.

    Work in each hemisphere before folding. Pressure-surface slopes contribute
    to fluxes through top/bottom boundaries for geometric-height TEM.
    """
    pressure = hemisphere_samples(archive.pressure, "even")
    density = hemisphere_samples(archive.rho_zm, "even")
    def sample(field):
        return _sample_mechanism_grid(field, pressure)

    values = {}
    for name in PRIMITIVES:
        source = (diagnostics.coriolis_parameter + diagnostics.relative_vorticity
                  if name == "eta" else diagnostics[name])
        parity = "odd" if name in {"eta", "vres"} else "even"
        values[name] = sample(hemisphere_samples(source, parity))
    region = xr.Dataset(values)
    region.update(ep_flux_boundary_samples(
        diagnostics.a_ep_meridional, diagnostics.a_ep_vertical, archive))
    # HC maxima use each hemisphere's own mean streamfunction, never delta Psi.
    hc = hemisphere_samples(diagnostics.residual_streamfunction, "odd").sel(
        absolute_latitude=slice(*HC_LATITUDE))
    region["hc_streamfunction"] = hc.rename(z="hc_z", absolute_latitude="hc_latitude")
    region["hc_pressure"] = (pressure / 100.0).sel(absolute_latitude=slice(*HC_LATITUDE)).rename(
        z="hc_z", absolute_latitude="hc_latitude")
    region.mass_weight.attrs.update(units="kg m-2 Pa-1", definition="rho * (-dz/dp) * cos(phi)")
    for name in region.data_vars:
        if name.startswith(("boundary_", "torque_a_")):
            region[name].attrs.update(units="N m", sign_convention="outward-positive EP torque")
    region.torque_normalizer.attrs.update(units="kg m", definition="integral_R rho*a*cos(phi) dV")
    region.eta.attrs.update(units="s-1", definition="folded control absolute vorticity f+zeta")
    region.attrs.update(
        tem_formulation=archive.attrs["tem_formulation"],
        region=REGION_DESCRIPTION,
        hc_region="maximum |mean Psi*| per hemisphere; 0-25 degrees; 200-700 hPa",
        ep_boundary_weighting="axial torque: integral_R rho*a*cos(phi)*EPFd dV; not unweighted acceleration",
        pressure_surface_flux="cos(phi)*(Fy*z_phi/a-Fz); includes sloping pressure surfaces",
    )
    return region


def ep_flux_boundary_samples(meridional, vertical, archive):
    """Integrate HPE fluxes and acceleration with the same axial-torque geometry.

    Keep hemispheres separate. Pressure faces include the meridional flux
    through their slopes. The returned residual measures native-grid numerical
    differentiation and pressure-interpolation error; no boundary absorbs it.
    """
    pressure = hemisphere_samples(archive.pressure, "even")
    density = hemisphere_samples(archive.rho_zm, "even")
    def sample(field):
        return _sample_mechanism_grid(field, pressure)
    region = xr.Dataset({"a_ep": sample(hemisphere_samples(meridional+vertical, "even")),
        "a_ep_meridional": sample(hemisphere_samples(meridional, "even")),
        "a_ep_vertical": sample(hemisphere_samples(vertical, "even"))})
    rho = sample(density)
    z = xr.zeros_like(archive.pressure) + archive.z
    height = sample(hemisphere_samples(z, "even"))
    phi = np.deg2rad(region.region_latitude)
    p_pa = region.region_pres * 100.0
    if archive.attrs["tem_formulation"] == "log-pressure":
        minus_z_p = xr.zeros_like(height) + SCALE_HEIGHT_M / p_pa
        z_phi = xr.zeros_like(height)
    else:
        minus_z_p = -height.assign_coords(region_pres=p_pa).differentiate("region_pres", edge_order=2)
        minus_z_p = minus_z_p.assign_coords(region_pres=region.region_pres)
        z_phi = height.assign_coords(region_latitude=phi).differentiate("region_latitude", edge_order=2)
        z_phi = z_phi.assign_coords(region_latitude=region.region_latitude)
    cos = np.cos(phi)
    region["mass_weight"] = rho * minus_z_p * cos
    # Remove the common 2*pi*a^2 factor until reporting dimensional torque.
    torque_weight = EARTH_RADIUS * cos * region.mass_weight
    integrate_box = lambda field: _integral(
        _integral(field, "region_pres", p_pa), "region_latitude", phi)
    region["torque_normalizer"] = integrate_box(torque_weight) * (2 * np.pi * EARTH_RADIUS**2)
    fy = sample(hemisphere_samples(archive.Fy, "odd"))
    fz = sample(hemisphere_samples(archive.Fz, "even"))
    through_latitude = cos * fy * minus_z_p / EARTH_RADIUS
    through_pressure_y = cos * fy * z_phi / EARTH_RADIUS
    through_pressure_z = -cos * fz
    through_pressure = through_pressure_y + through_pressure_z
    fluxes = {
        "lower": _integral(through_pressure.isel(region_pres=-1, drop=True), "region_latitude", phi),
        "upper": -_integral(through_pressure.isel(region_pres=0, drop=True), "region_latitude", phi),
        "equatorward": -_integral(through_latitude.isel(region_latitude=0, drop=True), "region_pres", p_pa),
        "poleward": _integral(through_latitude.isel(region_latitude=-1, drop=True), "region_pres", p_pa),
    }
    for name, flux in fluxes.items():
        region[f"boundary_{name}"] = flux * (2 * np.pi * EARTH_RADIUS**2)
    region["boundary_net"] = sum(region[f"boundary_{name}"] for name in BOUNDARIES)
    region["boundary_meridional_pressure_surface"] = _integral(
        through_pressure_y.isel(region_pres=-1, drop=True)
        - through_pressure_y.isel(region_pres=0, drop=True),
        "region_latitude", phi) * (2 * np.pi * EARTH_RADIUS**2)
    region["boundary_meridional"] = (region.boundary_equatorward + region.boundary_poleward
                                     + region.boundary_meridional_pressure_surface)
    region["boundary_vertical"] = _integral(
        through_pressure_z.isel(region_pres=-1, drop=True)
        - through_pressure_z.isel(region_pres=0, drop=True),
        "region_latitude", phi) * (2 * np.pi * EARTH_RADIUS**2)
    for name in ("a_ep", "a_ep_meridional", "a_ep_vertical"):
        region[f"torque_{name}"] = integrate_box(torque_weight * region[name]) * (2 * np.pi * EARTH_RADIUS**2)
    region["boundary_closure_error"] = region.boundary_net - region.torque_a_ep
    region["boundary_meridional_closure_error"] = region.boundary_meridional - region.torque_a_ep_meridional
    region["boundary_vertical_closure_error"] = region.boundary_vertical - region.torque_a_ep_vertical
    for name in region:
        if name.startswith(("boundary_", "torque_a_")):
            region[name].attrs.update(units="N m", sign_convention="outward-positive EP torque")
    region.torque_normalizer.attrs.update(units="kg m", definition="integral_R rho*a*cos(phi) dV")
    return region


def tropical_precipitation(accumulation):
    """Return cosine-area-weighted cumulative precipitation by tropical hemisphere."""
    zonal = accumulation.mean("longitude", skipna=False)
    latitude = np.asarray(zonal.latitude)
    if latitude.min() > PRECIP_LATITUDE[0] or latitude.max() < PRECIP_LATITUDE[1]:
        raise ValueError("Precipitation archive must bracket the full 10S-10N region")
    # Cell-centred archives have no equatorial sample. Interpolate while both
    # sides of the equator remain available, then split into hemispheres.
    targets = np.unique(np.r_[PRECIP_LATITUDE[0],
                             latitude[(latitude > PRECIP_LATITUDE[0]) & (latitude < PRECIP_LATITUDE[1])],
                             0.0, PRECIP_LATITUDE[1]])
    sampled = hemisphere_samples(zonal.interp(latitude=targets), "even")
    phi = np.deg2rad(sampled.absolute_latitude)
    weights = xr.ones_like(sampled) * np.cos(phi)
    return _integral(sampled * weights, "absolute_latitude", phi) / _integral(weights, "absolute_latitude", phi)


def precipitation_window(accumulation, first, last):
    """Use exact accumulated-precipitation endpoints, including MPAS buckets."""
    days = float((np.datetime64(last) - np.datetime64(first)) / np.timedelta64(1, "D"))
    if days <= 0:
        raise ValueError("Precipitation averaging requires a positive-duration interval")
    value = (accumulation.sel(Time=last, drop=True) - accumulation.sel(Time=first, drop=True)) / days
    if not np.isfinite(value.values).all():
        raise ValueError(f"Nonfinite tropical precipitation for interval {first} to {last}; check endpoint data")
    value.attrs.update(units="mm day-1", definition="endpoint accumulation difference / exact duration")
    return value


def _quadrature(coordinate):
    x = np.asarray(coordinate, dtype=float)
    if np.any(np.diff(x) <= 0):
        raise ValueError("Regional quadrature requires increasing coordinates")
    widths = np.diff(x)
    return np.r_[widths[0] / 2, (widths[:-1] + widths[1:]) / 2, widths[-1] / 2]


def _arrays(dataset, blocks=False):
    first = ("block", "hemisphere") if blocks else ("hemisphere",)
    spatial = ("region_pres", "region_latitude", "hc_z", "hc_latitude")
    return {name: np.asarray(field.transpose(*first, *[dim for dim in spatial if dim in field.dims]), dtype=float)
            for name, field in dataset.data_vars.items()}


def _response(run, control, quadrature):
    """Recompute all nonlinear quantities using paired primitive states."""
    eta = control["eta"]
    d_eta = run["eta"] - eta
    d_v = run["vres"] - control["vres"]
    d_w = run["wres"] - control["wres"]
    d_shear = run["u_z"] - control["u_z"]
    d_ep = run["a_ep"] - control["a_ep"]
    valid = np.isfinite(eta) & (np.abs(eta) > 1e-8)
    for name in (*BUDGET_TERMS, "vres", "eta", "wres", "u_z"):
        valid &= np.isfinite(run[name]) & np.isfinite(control[name])
    weights = control["mass_weight"] * quadrature
    weights = np.where(valid & np.isfinite(weights) & (weights > 0), weights, 0.0)
    normalizer = weights.sum(axis=(-2, -1))
    if np.any(normalizer <= 0):
        raise ValueError("The fixed mechanism box has no common valid TEM samples")

    def regional(field, additional_mask=None):
        selected_weights = weights if additional_mask is None else np.where(additional_mask, weights, 0.0)
        selected_normalizer = selected_weights.sum(axis=(-2, -1))
        if np.any(selected_normalizer <= 0):
            raise ValueError("The fixed mechanism box has no nonsingular experiment vorticity samples")
        numerator = np.where(selected_weights > 0, field, 0) * selected_weights
        return float(np.mean(numerator.sum(axis=(-2, -1)) / selected_normalizer))

    output = {name: regional(run[name] - control[name]) for name in
              (*BUDGET_TERMS, "a_ep_meridional", "a_ep_vertical")}
    output["ep_component_closure"] = output["a_ep"] - output["a_ep_meridional"] - output["a_ep_vertical"]
    output["predicted_vres"] = regional(-d_ep / np.where(valid, eta, np.nan))
    experiment_valid = np.isfinite(run["eta"]) & (np.abs(run["eta"]) > 1e-8)
    output["predicted_vres_adjusted"] = regional(
        (-d_ep - control["vres"] * d_eta) / np.where(experiment_valid, run["eta"], np.nan),
        experiment_valid,
    )
    output["observed_vres"] = regional(d_v)
    output["av_flow"] = regional(eta * d_v)
    output["av_vorticity"] = regional(control["vres"] * d_eta)
    output["av_interaction"] = regional(d_eta * d_v)
    output["av_mean_product_remainder"] = output["a_vres"] - sum(output[name] for name in ("av_flow", "av_vorticity", "av_interaction"))
    output["aw_flow"] = regional(-d_w * control["u_z"])
    output["aw_shear"] = regional(-control["wres"] * d_shear)
    output["aw_interaction"] = regional(-d_w * d_shear)
    output["aw_mean_product_remainder"] = output["a_wres"] - sum(output[name] for name in MMC_DECOMPOSITIONS["aw"])
    def hc_maximum(state):
        inside = (state["hc_pressure"] >= HC_PRESSURE[0]) & (state["hc_pressure"] <= HC_PRESSURE[1])
        psi = np.where(inside, np.abs(state["hc_streamfunction"]), np.nan)
        if not np.isfinite(psi).any(axis=(-2, -1)).all():
            raise ValueError("HC strength requires finite native streamfunction samples in both hemispheres")
        return np.nanmax(psi, axis=(-2, -1))
    output["hc_strength"] = float(np.mean(hc_maximum(run) - hc_maximum(control)))
    output["precipitation"] = float(np.mean(run["precipitation"] - control["precipitation"]))
    output["ep_forcing"] = -output["a_ep"]
    torque_denominator = control["torque_normalizer"]
    for name in (*[f"boundary_{side}" for side in (*BOUNDARIES, "net")],
                 "torque_a_ep", "torque_a_ep_meridional", "torque_a_ep_vertical", "boundary_closure_error",
                 "boundary_meridional", "boundary_vertical", "boundary_meridional_pressure_surface",
                 "boundary_meridional_closure_error", "boundary_vertical_closure_error"):
        torque_difference = run[name] - control[name]
        output[f"{name}_dimensional"] = float(np.mean(torque_difference))
        output[name] = float(np.mean(torque_difference / torque_denominator))
    boundary_scale = max(sum(abs(output[f"boundary_{name}"]) for name in BOUNDARIES),
                         abs(output["torque_a_ep"]), 1e-20)
    output["boundary_closure_relative"] = abs(output["boundary_closure_error"]) / boundary_scale
    av = output["a_vres"]
    output["balance_ep_v"] = -output["a_ep"] / av if abs(av) > 1e-12 else np.nan
    output["balance_w_v"] = abs(output["a_wres"] / av) if abs(av) > 1e-12 else np.nan
    output["av_flow_fraction"] = output["av_flow"] / av if abs(av) > 1e-12 else np.nan
    aw = output["a_wres"]
    output["aw_shear_fraction"] = output["aw_shear"] / aw if abs(aw) > 1e-12 else np.nan
    output["budget_closure"] = output["a_tendency"] - sum(output[name] for name in
        ("a_ep", "a_vres", "a_wres", "a_density_storage", "unresolved_direct_momentum_residual"))
    possible = (control["mass_weight"] * quadrature).sum(axis=(-2, -1))
    output["valid_mass_fraction"] = float(np.mean(normalizer / possible))
    output["adjusted_prediction_valid_mass_fraction"] = float(np.mean(
        np.where(experiment_valid, weights, 0.0).sum(axis=(-2, -1)) / possible))
    return output


def _validate_blocks(blocks):
    """Reject positional pairing of blocks with different archived dates."""
    starts, ends = blocks.block_start.values, blocks.block_end.values
    if starts.ndim != 2 or not np.all(starts == starts[:1]) or not np.all(ends == ends[:1]):
        raise ValueError("Mechanism comparisons require identical control/experiment block dates")
    if np.any(np.isnat(starts)) or np.any(np.isnat(ends)):
        raise ValueError("Mechanism blocks contain missing dates")
    if not blocks.attrs.get("test_subset", 0):
        if not np.all(ends - starts == np.timedelta64(10, "D")):
            raise ValueError("Mechanism figure requires non-overlapping 10-day blocks")
    if np.any(starts[:, 1:] < ends[:, :-1]):
        raise ValueError("Mechanism blocks overlap")


def mechanism_analysis(fields, blocks, comparisons, *, resamples, minimum_blocks, seed, percentiles, bootstrap=True,
                       progress_label="Figure 14"):
    """Pair control blocks and preserve the joint variables in every bootstrap draw.

    This new figure uses matched block draws, as requested. Other circulation
    figures retain their independent-simulation bootstrap. Hemispheres receive
    independent draws; simulations share each hemisphere's selected dates.
    Full-run HC markers use maxima of full-run mean Psi*. Bootstrap HC markers
    recompute maxima of each resampled simulation's mean Psi*, not maxima of
    delta Psi or averages of the block maxima.
    """
    _validate_blocks(blocks)
    for label, dataset in (("full-record", fields), ("block", blocks)):
        if not np.isfinite(dataset.precipitation.values).all():
            raise ValueError(f"Nonfinite {label} precipitation in mechanism cache; rebuild with --figures 14")
    q = _quadrature(fields.region_pres * 100)[:, None] * _quadrature(np.deg2rad(fields.region_latitude))[None, :]
    states = [_arrays(fields.isel(simulation=index, drop=True)) for index in range(fields.sizes["simulation"])]
    block_states = [_arrays(blocks.drop_vars(["block_start", "block_end"]).isel(simulation=index, drop=True), blocks=True)
                    for index in range(blocks.sizes["simulation"])]
    nblocks = blocks.sizes["block"]
    if bootstrap and nblocks < minimum_blocks:
        raise ValueError(f"Mechanism uncertainty needs {minimum_blocks} paired blocks; found {nblocks}")
    means = [_response(states[run], states[control], q) for _, run, control, _ in comparisons]
    per_block = []
    for _, run, control, _ in comparisons:
        per_block.append([_response({name: value[index] for name, value in block_states[run].items()},
                                    {name: value[index] for name, value in block_states[control].items()}, q)
                          for index in range(nblocks)])
    intervals = [{} for _ in comparisons]
    if bootstrap:
        rng = np.random.default_rng(seed)
        draws = [[] for _ in comparisons]
        for replicate in range(resamples):
            indices = rng.integers(nblocks, size=(2, nblocks))
            climates = []
            for state in block_states:
                climates.append({name: np.stack([value[indices[h], h].mean(axis=0) for h in range(2)])
                                 for name, value in state.items()})
            for index, (_, run, control, _) in enumerate(comparisons):
                draws[index].append(_response(climates[run], climates[control], q))
            if (replicate + 1) % max(1, resamples // 10) == 0 or replicate + 1 == resamples:
                progress(f"{progress_label}: paired bootstrap {replicate + 1}/{resamples} draws")
        for index in range(len(comparisons)):
            intervals[index] = {name: np.percentile([draw[name] for draw in draws[index]], percentiles)
                                for name in means[index] if all(np.isfinite(draw[name]) for draw in draws[index])}
    return means, per_block, intervals


def mechanism_tables(means, per_block, intervals, comparisons):
    """Export signed diagnostics, balance ratios, and block associations."""
    rows, block_rows, associations = [], [], []
    for index, (_, _, _, label) in enumerate(comparisons):
        label = label.replace("T from U", "T")
        for name, value in means[index].items():
            ci = intervals[index].get(name, [np.nan, np.nan])
            units = ("N m" if name.endswith("_dimensional") else "m s-1" if name.endswith("vres") or name.startswith("predicted_vres_")
                     else "kg s-1" if name == "hc_strength" else "mm day-1" if name == "precipitation"
                     else "1" if name.startswith("balance_") or name in {"valid_mass_fraction", "adjusted_prediction_valid_mass_fraction", "av_flow_fraction", "aw_shear_fraction", "boundary_closure_relative"} else "m s-2")
            rows.append(dict(comparison=label, diagnostic=name, units=units, full_record_mean=value, ci_low=ci[0], ci_high=ci[1]))
        for block, values in enumerate(per_block[index]):
            block_rows.append(dict(comparison=label, block=block, **values))
        hc = np.array([record["hc_strength"] for record in per_block[index]])
        for predictor in ("predicted_vres", "precipitation"):
            x = np.array([record[predictor] for record in per_block[index]])
            correlation = np.corrcoef(x, hc)[0, 1] if np.std(x) > 0 and np.std(hc) > 0 else np.nan
            associations.append(dict(comparison=label, predictor=predictor, n_blocks=len(hc), correlation=correlation,
                                     interpretation="matched-block association; not a causal estimate"))
    return pd.DataFrame(rows), pd.DataFrame(block_rows), pd.DataFrame(associations)
