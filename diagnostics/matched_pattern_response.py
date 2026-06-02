#!/usr/bin/env python
# coding: utf-8

# In[1]:

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
for _rel_path in ("..", "../..", "../../.."):
    _path = os.path.abspath(os.path.join(_THIS_DIR, _rel_path))
    if _path not in sys.path:
        sys.path.append(_path)

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import xarray as xr

from tropopause import tropopause_profile

try:
    from diagnostic_helpers import *
except ModuleNotFoundError:
    from diagnostics.diagnostic_helpers import *


save = True
save_loc = "/home/cwp29/tmp/mpas/crf_on/matched_pattern/"

# Set to [] or None to run all figures. Otherwise list numbered figures.
#
# Figure numbers:
#   1: matched_pattern_temperature_projection
#   2: pattern_scaled_residual_latpres
#   3: pattern_scaled_residual_zonalmean_surface_vars
figures_to_run = []
figsave = FigureSaver(save_loc, enabled=save, figures_to_run=figures_to_run)


def should_run_figure(number):
    return figure_should_run(figures_to_run, number)


def should_run_any(numbers):
    return figure_should_run_any(figures_to_run, numbers)


def run_all_figures():
    return all_figures_requested(figures_to_run)

root_dir = "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on/"
data_locs = ["control", "u_nudge", "T_nudge"]
latlon_file = ["latlon.2000-*.nc", "latlon.2000-*.nc", "latlon.2000-*.nc"]
zm_latlon_file = ["zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc", "zm_latlon.2000-*.nc"]
sim_names = ["control", "nudged u", "nudged T"]

cols = ["k", "b", "r"]

ndays = 200
dt = 6
analysis_end_date = "2000-11-21"
time_format = "time"

needed_vars = [
    "rainnc", "i_rainnc", "rainc", "i_rainc", "precipw", "olrtoa",
    "uzonal_isobaric", "umeridional_isobaric",
    "rh_isobaric", "t_isobaric", "theta", "zgrid", "pressure",
    "w",
]


# In[2]:

dss = []
dss_zm = []

for i, d in enumerate(data_locs):
    ds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks={"Time": -1},
    )
    ds = standardize_latlon_dataset(ds, needed_vars)

    zds = load_latlon_dataset(
        os.path.join(root_dir, d),
        latlon_pattern=zm_latlon_file[i],
        time_format=time_format,
        ndays=ndays,
        end_date=analysis_end_date,
        chunks={"Time": -1},
        prefer_pattern=True,
        add_cldfrac_file=False,
    )
    zds = standardize_latlon_dataset(zds, needed_vars)

    print(ds)
    print(zds)
    dss.append(ds)
    dss_zm.append(zds)


# In[3]:

def subset_between(da, dim, bounds):
    """Subset a coordinate range without assuming ascending or descending order."""
    if dim not in da.dims:
        return da
    lo, hi = sorted(bounds)
    coord = da[dim]
    return da.where((coord >= lo) & (coord <= hi), drop=True)


def pressure_layer_weights(pres):
    """
    Positive layer weights proportional to d ln(p).

    Input pressure is in hPa.  The returned DataArray is on dimension "pres" and
    works for either ascending or descending pressure coordinates.
    """
    if isinstance(pres, xr.DataArray):
        pres_values = np.asarray(pres.values, dtype=float)
        pres_coord = pres_values
    else:
        pres_values = np.asarray(pres, dtype=float)
        pres_coord = pres_values

    if np.any(pres_values <= 0):
        raise ValueError("Pressure values must be positive before taking log(p)")

    if pres_values.size == 1:
        weights = np.ones_like(pres_values)
    else:
        lnp = np.log(pres_values)
        bounds = np.empty(pres_values.size + 1)
        bounds[1:-1] = 0.5 * (lnp[:-1] + lnp[1:])
        bounds[0] = lnp[0] - 0.5 * (lnp[1] - lnp[0])
        bounds[-1] = lnp[-1] + 0.5 * (lnp[-1] - lnp[-2])
        weights = np.abs(np.diff(bounds))

    return xr.DataArray(weights, dims=("pres",), coords={"pres": pres_coord}, name="dlnp_weight")


def lat_weights(latitude):
    """Cosine-latitude area weights on dimension "latitude"."""
    if isinstance(latitude, xr.DataArray):
        lat_values = np.asarray(latitude.values, dtype=float)
    else:
        lat_values = np.asarray(latitude, dtype=float)
    return xr.DataArray(
        np.cos(np.deg2rad(lat_values)),
        dims=("latitude",),
        coords={"latitude": lat_values},
        name="coslat_weight",
    )


def weighted_inner_product(a, b, lat_dim="latitude", pres_dim="pres", use_pressure_weights=True):
    """
    Weighted inner product with finite-value masking.

    Fields are aligned with an inner join, non-finite points are excluded, and
    the product is weighted by cos(latitude) plus d ln(p) if a pressure dimension
    is present.  The sum is taken over whichever dimensions are present.
    """
    aa, bb = xr.align(a, b, join="inner")
    valid = np.isfinite(aa) & np.isfinite(bb)
    product = (aa * bb).where(valid)

    weight = xr.ones_like(product)
    if lat_dim in product.dims:
        lw = lat_weights(product[lat_dim])
        if lat_dim != "latitude":
            lw = lw.rename({"latitude": lat_dim})
        weight = weight * lw
    if use_pressure_weights and pres_dim in product.dims:
        pw = pressure_layer_weights(product[pres_dim])
        if pres_dim != "pres":
            pw = pw.rename({"pres": pres_dim})
        weight = weight * pw

    return (product * weight).sum(dim=list(product.dims), skipna=True)


def matched_pattern_projection(target, basis, lat_bounds=(-30, 30), p_bounds=(70, 200)):
    """
    Fit target ~= alpha * basis over a pressure-latitude domain.

    Interpretation:
      high pattern correlation plus low residual_fraction means the T-nudged
      pattern is mostly the same as the u-nudged pattern, but weaker.
      low correlation or high residual_fraction means the two nudging methods
      produce structurally different balanced states.
    """
    target_box = subset_between(target, "latitude", lat_bounds)
    basis_box = subset_between(basis, "latitude", lat_bounds)
    target_box = subset_between(target_box, "pres", p_bounds)
    basis_box = subset_between(basis_box, "pres", p_bounds)
    target_box, basis_box = xr.align(target_box, basis_box, join="inner")

    common_valid = np.isfinite(target_box) & np.isfinite(basis_box)
    target_box = target_box.where(common_valid)
    basis_box = basis_box.where(common_valid)

    numerator = weighted_inner_product(target_box, basis_box)
    denominator = weighted_inner_product(basis_box, basis_box)
    target_norm2 = weighted_inner_product(target_box, target_box)

    den_val = denominator.values.item()
    target_norm2_val = target_norm2.values.item()
    if not np.isfinite(den_val) or abs(den_val) < 1e-30:
        alpha = np.nan
    else:
        alpha = numerator.values.item() / den_val

    projection = alpha * basis_box
    residual = target_box - projection

    residual_norm2 = weighted_inner_product(residual, residual)
    basis_norm2 = weighted_inner_product(basis_box, basis_box)
    corr_den = np.sqrt(target_norm2.values.item() * basis_norm2.values.item())
    pattern_corr = numerator.values.item() / corr_den if corr_den > 0 else np.nan

    if target_norm2_val > 0:
        residual_ratio = residual_norm2.values.item() / target_norm2_val
        residual_fraction = np.sqrt(residual_ratio)
        explained_fraction = 1.0 - residual_ratio
    else:
        residual_fraction = np.nan
        explained_fraction = np.nan

    print("\nMatched-pattern projection")
    print(f"  latitude bounds: {lat_bounds[0]} to {lat_bounds[1]} deg")
    print(f"  pressure bounds: {p_bounds[0]} to {p_bounds[1]} hPa")
    print(f"  alpha_pattern = {alpha:.3f}")
    print(f"  weighted pattern correlation = {pattern_corr:.3f}")
    print(f"  residual_fraction = {residual_fraction:.3f}")
    print(f"  explained_fraction = {explained_fraction:.3f}")
    print("  Interpretation: high correlation and low residual_fraction imply a")
    print("  mostly shared thermal pattern with different amplitude; low correlation")
    print("  or high residual_fraction implies structurally different balanced states.")

    return {
        "alpha": alpha,
        "projection": projection,
        "residual": residual,
        "pattern_corr": pattern_corr,
        "residual_fraction": residual_fraction,
        "explained_fraction": explained_fraction,
        "target": target_box,
        "basis": basis_box,
    }


def weighted_mean_latpres(da, lat_bounds=(-10, 10), p_bounds=(70, 150)):
    """Area and dlnp weighted mean for pressure-latitude fields."""
    x = subset_between(da, "latitude", lat_bounds)
    x = subset_between(x, "pres", p_bounds)
    valid = np.isfinite(x)
    weights = xr.ones_like(x)
    weights = weights * lat_weights(x.latitude)
    weights = weights * pressure_layer_weights(x.pres)
    return (x.where(valid) * weights.where(valid)).sum(dim=list(x.dims), skipna=True) / weights.where(valid).sum(dim=list(x.dims), skipna=True)


def compute_cpts(zm_datasets):
    """Cold-point tropopause height and pressure profiles for plot overlays."""
    cpts = []
    for zds in zm_datasets:
        required = {"theta", "pressure"}
        if not required.issubset(set(zds.data_vars)) or "z" not in zds.coords:
            return None

        theta = hemispheric_fold(zds.theta.mean(dim="Time"), parity="even")
        pressure = hemispheric_fold(zds.pressure.mean(dim="Time"), parity="even")
        T = theta * (pressure / 100000.0) ** (287.0 / 1004.0)
        T = model_lat_field(T)
        pressure = model_lat_field(pressure)
        z = height_from_zm_latlon(zds).values

        ztps = []
        ptps = []
        for j in range(len(zds.latitude.values)):
            _, ptp, ztp, _ = tropopause_profile(T.values[:, j], pressure.values[:, j], z)
            ztps.append(ztp)
            ptps.append(ptp)
        cpts.append([ztps, ptps])
    return np.array(cpts)


def overlay_cpt_pressure(ax, run_index=1):
    """Overlay cold-point tropopause pressure if it was computed successfully."""
    if CPTs is not None and len(CPTs) > run_index:
        ax.plot(
            dss_zm[run_index].latitude.values,
            CPTs[run_index][1] / 100.0,
            color="k",
            lw=1,
            ls=":",
        )


def anomaly_latpres(var, run_index, control_index=0):
    """Time-mean zonal-mean pressure-latitude anomaly on the run grid."""
    if var == "w":
        field = isobaric_vertical_velocity_from_zm(dss_zm[run_index])
        control_field = isobaric_vertical_velocity_from_zm(dss_zm[control_index])
    else:
        parity = latpres_var_hemispheric_parity(var)
        field = isobaric_lat_mean(dss_zm[run_index][var], hemispheric_parity=parity)
        control_field = isobaric_lat_mean(dss_zm[control_index][var], hemispheric_parity=parity)
    control = control_field.interp(
        pres=field.pres,
        latitude=field.latitude,
        kwargs={"fill_value": np.nan},
    )
    return field - control


def anomaly_surface_map(var, run_index, control_index=0):
    """Time-mean lat-lon anomaly on the run grid."""
    if var == "precip_rate":
        run_field = precip_rate_from_accum(total_precip_accum(dss[run_index]), dt=dt).mean(dim="Time")
        control_field = precip_rate_from_accum(total_precip_accum(dss[control_index]), dt=dt).mean(dim="Time")
    else:
        run_field = finite_or_nan(dss[run_index][var]).mean(dim="Time")
        control_field = finite_or_nan(dss[control_index][var]).mean(dim="Time")
    run_field = hemispheric_fold(run_field, parity="even")
    control_field = hemispheric_fold(control_field, parity="even")

    control_on_run = control_field.interp(
        latitude=run_field.latitude,
        longitude=run_field.longitude,
        kwargs={"fill_value": np.nan},
    )
    return run_field - control_on_run


def latpres_var_hemispheric_parity(var):
    if var in ("umeridional_isobaric", "v_isobaric", "v"):
        return "odd"
    return "even"


def max_abs_for_levels(fields, default=8.0, percentile=99.0):
    vals = []
    for field in fields:
        arr = np.asarray(field.values).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            vals.append(np.nanpercentile(np.abs(arr), percentile))
    if not vals:
        return default
    return max(default, float(np.ceil(max(vals))))


# In[4]:

print("\nComputing pressure-latitude temperature anomalies")
T_control = isobaric_lat_mean(dss_zm[0].t_isobaric, hemispheric_parity="even")
T_u = isobaric_lat_mean(dss_zm[1].t_isobaric, hemispheric_parity="even")
T_T = isobaric_lat_mean(dss_zm[2].t_isobaric, hemispheric_parity="even")

dT_u = T_u - T_control.interp(pres=T_u.pres, latitude=T_u.latitude, kwargs={"fill_value": np.nan})
dT_T = T_T - T_control.interp(pres=T_T.pres, latitude=T_T.latitude, kwargs={"fill_value": np.nan})
dT_T_on_u = dT_T.interp(pres=dT_u.pres, latitude=dT_u.latitude, kwargs={"fill_value": np.nan})

alpha_TTL = np.nan
try:
    dT_TTL_u = weighted_mean_latpres(dT_u, lat_bounds=(-10, 10), p_bounds=(70, 150)).values.item()
    dT_TTL_T = weighted_mean_latpres(dT_T_on_u, lat_bounds=(-10, 10), p_bounds=(70, 150)).values.item()
    alpha_TTL = dT_TTL_u / dT_TTL_T if np.isfinite(dT_TTL_T) and abs(dT_TTL_T) > 1e-12 else np.nan
    print("\nScalar TTL temperature scaling")
    print(f"  u-nudged mean dT_TTL = {dT_TTL_u:.3f} K")
    print(f"  T-nudged mean dT_TTL = {dT_TTL_T:.3f} K")
    print(f"  alpha_TTL = {alpha_TTL:.3f}")
except Exception as exc:
    print(f"Skipping scalar alpha_TTL calculation: {exc}")

CPTs = compute_cpts(dss_zm)


# In[5]:

domains = {
    "TTL_70_200hPa_30S30N": dict(lat_bounds=(-30, 30), p_bounds=(70, 200)),
    "UTLS_50_300hPa_40S40N": dict(lat_bounds=(-40, 40), p_bounds=(50, 300)),
    "DeepTropics_70_200hPa_10S10N": dict(lat_bounds=(-10, 10), p_bounds=(70, 200)),
}

projection_results = {}
metric_rows = []

for domain_name, domain in domains.items():
    result = matched_pattern_projection(dT_u, dT_T_on_u, **domain)
    projection_results[domain_name] = result
    metric_rows.append(
        {
            "domain": domain_name,
            "lat_min": domain["lat_bounds"][0],
            "lat_max": domain["lat_bounds"][1],
            "p_min_hPa": min(domain["p_bounds"]),
            "p_max_hPa": max(domain["p_bounds"]),
            "alpha_pattern": result["alpha"],
            "weighted_pattern_corr": result["pattern_corr"],
            "residual_fraction": result["residual_fraction"],
            "explained_fraction": result["explained_fraction"],
        }
    )

projection_metrics = pd.DataFrame(metric_rows)
print("\nMatched-pattern projection metrics:")
print(projection_metrics.to_string(index=False))
if save:
    projection_metrics.to_csv(save_loc + "matched_pattern_projection_metrics.csv", index=False)


# In[6]:

# What to look for: if the middle panel resembles the left panel and the
# residual is small, the T-nudged thermal response is mostly the same pattern
# but weaker. If the residual retains coherent UTLS structures, the u-nudged
# run is dynamically distinct.

for domain_name, result in projection_results.items():
    if not should_run_figure(1):
        continue
    domain = domains[domain_name]
    alpha_pattern = result["alpha"]

    plot_lat_bounds = (-60, 60)
    plot_p_bounds = (50, 300)
    target_plot = subset_between(subset_between(dT_u, "latitude", plot_lat_bounds), "pres", plot_p_bounds)
    basis_plot = subset_between(subset_between(dT_T_on_u, "latitude", plot_lat_bounds), "pres", plot_p_bounds)
    target_plot, basis_plot = xr.align(target_plot, basis_plot, join="inner")
    projection_plot = alpha_pattern * basis_plot
    residual_plot = target_plot - projection_plot

    max_abs = max_abs_for_levels([target_plot, projection_plot, residual_plot], default=8.0)
    levels = symmetric_levels(max_abs, ndiv=16)

    fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True, sharex=True, sharey=True)
    im0 = ax[0].contourf(target_plot.latitude, target_plot.pres, target_plot, levels=levels, cmap="bwr", extend="both")
    im1 = ax[1].contourf(projection_plot.latitude, projection_plot.pres, projection_plot, levels=levels, cmap="bwr", extend="both")
    im2 = ax[2].contourf(residual_plot.latitude, residual_plot.pres, residual_plot, levels=levels, cmap="bwr", extend="both")

    titles = [
        r"$\Delta T_u$",
        rf"$\alpha_{{pattern}}\Delta T_T$",
        r"$\Delta T_u-\alpha_{pattern}\Delta T_T$",
    ]
    labels = [
        f"{sim_names[1]} - {sim_names[0]}",
        f"{alpha_pattern:.2f} x ({sim_names[2]} - {sim_names[0]})",
        "pattern residual",
    ]
    for j, a in enumerate(ax):
        set_latpres_axis(a, xlabel=True, ylabel=(j == 0), xlim=plot_lat_bounds, ylim=(max(plot_p_bounds), min(plot_p_bounds)))
        overlay_cpt_pressure(a, run_index=1)
        a.set_title(titles[j])
        annotate_panel_label(a, labels[j])

    fig.suptitle(
        f"{domain_name}: alpha={alpha_pattern:.2f}, "
        f"corr={result['pattern_corr']:.2f}, "
        f"resid_frac={result['residual_fraction']:.2f}"
    )
    add_symmetric_colorbar(fig, im2, ax, levels, label="Temperature anomaly (K)")

    figsave.save(fig, f"matched_pattern_temperature_projection_{domain_name}", number=1)
    print(f"Done with figure: comp_matched_pattern_temperature_projection_{domain_name}")


# In[7]:

# What to look for: if residuals remain large after pattern-based thermal
# scaling, then the u-nudged response is not explainable as a scaled version of
# the T-nudged thermal pattern.

latpres_specs = [
    ("t_isobaric", r"T anomaly (K)", 8.0),
    ("uzonal_isobaric", r"u anomaly (m s$^{-1}$)", 8.0),
    ("rh_isobaric", r"RH anomaly", 10.0),
    ("w", r"w anomaly (m s$^{-1}$)", 0.03),
]

for domain_name, result in projection_results.items():
    if not should_run_figure(2):
        continue
    alpha_pattern = result["alpha"]

    for var, label, max_abs in latpres_specs:
        if any(var not in zds.data_vars for zds in dss_zm):
            print(f"Skipping pattern-scaled residual for {var}: not present in all zm datasets")
            continue

        dXu = anomaly_latpres(var, run_index=1)
        dXT = anomaly_latpres(var, run_index=2)
        dXT_on_u = dXT.interp(pres=dXu.pres, latitude=dXu.latitude, kwargs={"fill_value": np.nan})
        residual = dXu - alpha_pattern * dXT_on_u
        projection = alpha_pattern * dXT_on_u
        levels = symmetric_levels(max_abs, ndiv=16)

        fig, ax = plt.subplots(1, 3, figsize=(11, 3.3), constrained_layout=True, sharex=True, sharey=True)
        im0 = ax[0].contourf(dXu.latitude, dXu.pres, dXu, levels=levels, cmap="bwr", extend="both")
        im1 = ax[1].contourf(projection.latitude, projection.pres, projection, levels=levels, cmap="bwr", extend="both")
        im2 = ax[2].contourf(residual.latitude, residual.pres, residual, levels=levels, cmap="bwr", extend="both")

        titles = [
            r"$\Delta X_u$",
            rf"$\alpha_{{pattern}}\Delta X_T$",
            r"$\Delta X_u-\alpha_{pattern}\Delta X_T$",
        ]
        panel_labels = [
            f"{sim_names[1]} - {sim_names[0]}",
            f"{alpha_pattern:.2f} x ({sim_names[2]} - {sim_names[0]})",
            "pattern residual",
        ]
        for j, a in enumerate(ax):
            set_latpres_axis(a, xlabel=True, ylabel=(j == 0), xlim=(-60, 60), ylim=(1000, 50))
            overlay_cpt_pressure(a, run_index=1)
            a.set_title(titles[j])
            annotate_panel_label(a, panel_labels[j])

        fig.suptitle(f"Pattern-scaled residual for {var}; {domain_name}; alpha={alpha_pattern:.2f}")
        add_symmetric_colorbar(fig, im2, ax, levels, label=label)

        figsave.save(fig, f"pattern_scaled_residual_{var}_{domain_name}", number=2)
        print(f"Done with figure: comp_pattern_scaled_residual_{var}_{domain_name}")


# In[8]:

alpha_rows = []
for domain_name, result in projection_results.items():
    row = {
        "domain": domain_name,
        "alpha_pattern": result["alpha"],
    }
    if "alpha_TTL" in globals() and np.isfinite(alpha_TTL):
        row["alpha_TTL"] = alpha_TTL
        row["alpha_pattern_minus_alpha_TTL"] = result["alpha"] - alpha_TTL
        row["alpha_pattern_over_alpha_TTL"] = result["alpha"] / alpha_TTL if abs(alpha_TTL) > 1e-12 else np.nan
    alpha_rows.append(row)

alpha_comparison = pd.DataFrame(alpha_rows)
print("\nScalar-alpha versus pattern-alpha comparison:")
print(alpha_comparison.to_string(index=False))
if save:
    alpha_comparison.to_csv(save_loc + "thermal_scaling_alpha_comparison.csv", index=False)


# In[9]:

def plot_pattern_scaled_surface_zonalmean_residuals(alpha_pattern, domain_name):
    """
    Zonal-mean residuals for noisy 2-D surface/column fields.

    What to look for: broad latitude-band residuals are more reliable than
    individual longitude features for noisy surface fields.
    """
    surface_specs = [
        ("olrtoa", r"OLR residual (W m$^{-2}$)"),
        ("precipw", r"PW residual (kg m$^{-2}$)"),
        ("precip_rate", r"Precip residual (mm day$^{-1}$)"),
    ]

    fig, ax = plt.subplots(1, len(surface_specs), figsize=(12, 3.4), constrained_layout=True)
    if len(surface_specs) == 1:
        ax = [ax]

    for j, (var, ylabel) in enumerate(surface_specs):
        dXu = anomaly_surface_map(var, run_index=1)
        dXT = anomaly_surface_map(var, run_index=2)
        dXT_on_u = dXT.interp(latitude=dXu.latitude, longitude=dXu.longitude, kwargs={"fill_value": np.nan})
        projection = alpha_pattern * dXT_on_u
        residual = dXu - projection

        dXu_zm = dXu.mean(dim="longitude", skipna=True)
        projection_zm = projection.mean(dim="longitude", skipna=True)
        residual_zm = residual.mean(dim="longitude", skipna=True)

        ax[j].plot(dXu_zm.latitude, dXu_zm, color=cols[1], lw=1.5, label=f"{sim_names[1]} - {sim_names[0]}")
        ax[j].plot(projection_zm.latitude, projection_zm, color=cols[2], lw=1.5, label=rf"$\alpha_{{pattern}}\Delta X_T$")
        ax[j].plot(residual_zm.latitude, residual_zm, color=cols[0], lw=1.8, label="pattern residual")
        ax[j].axhline(0, color="0.7", lw=0.7)
        ax[j].axvline(0, color="0.7", lw=0.7)
        ax[j].set_xlim(-60, 60)
        ax[j].set_xlabel("Latitude")
        ax[j].set_ylabel(ylabel)
        ax[j].set_title(var)
        ax[j].grid(alpha=0.2)
        if j == 0:
            ax[j].legend(loc="best", fontsize=8)

    fig.suptitle(f"Pattern-scaled zonal-mean surface residuals; {domain_name}; alpha={alpha_pattern:.2f}")
    figsave.save(fig, "pattern_scaled_residual_zonalmean_surface_vars", number=3)
    print("Done with figure: comp_pattern_scaled_residual_zonalmean_surface_vars")


primary_domain = "TTL_70_200hPa_30S30N"
if should_run_figure(3) and primary_domain in projection_results and np.isfinite(projection_results[primary_domain]["alpha"]):
    plot_pattern_scaled_surface_zonalmean_residuals(projection_results[primary_domain]["alpha"], primary_domain)
elif should_run_figure(3):
    print(f"Skipping surface zonal-mean residuals: no finite alpha for {primary_domain}")

print("Matched-pattern thermal response diagnostics complete.")
