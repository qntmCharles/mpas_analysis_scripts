"""Scientific TEM operations, independent of archive readers and plotting.

Pressure: Andrews, Holton and Leovy (1987), section 3.5, with rho0=p/(g*H).
Pahlavan et al. (2021) uses these equations for its wave decomposition.
H defines the log-pressure coordinate; it is not an amplitude multiplier.
Height: Hardiman et al. (2010), HPE(z), with the northward heat-flux
streamfunction gauge psi/(a*cos(phi)) = <(rho*v)'theta'> / theta_z.
The height momentum budget also requires -d<rho'u'>/dt / <rho>.
Both formulations use instantaneous longitude means and unfolded hemispheres.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

R = 8.31447215
CP_DRY_AIR = 1005.7
M_AIR = 0.0289644
RD = R / M_AIR
KAPPA = RD / CP_DRY_AIR
EARTH_RADIUS = 6371229.0
OMEGA = 2.0 * np.pi / ((23 * 60 + 56) * 60 + 4.1)
GRAVITY = 9.80665
P0_PA = 100000.0
SCALE_HEIGHT_M = 7000.0
MINIMUM_THETA_GRADIENT = 1e-6
TEM_FORMULATION = "geometric-height"
TEM_VERSION = "tem-v2-conservative-unfolded"
PRESSURE_PATTERN = "tem_log_pressure*.nc"
HEIGHT_PATTERN = "tem_geometric_height*.nc"
ISOBARIC_FIELDS = (
    "uzonal_isobaric", "umeridional_isobaric", "t_isobaric", "omg_isobaric",
)
FLUX_TERMS = (
    "momentum_meridional", "momentum_vertical", "heat_meridional", "heat_vertical",
)
PRESSURE_REQUIRED = (
    "EPFd", "EPFd_y", "EPFd_z", "EPFd_momentum", "EPFd_heat",
    *FLUX_TERMS, "u_bar", "v_bar", "w_bar", "theta_bar", "t",
    "u_z", "theta_z", "absolute_vorticity", "rho_zm", "vres", "wres", "Fy", "Fz",
    "uv", "uw", "vtheta", "valid_zonal_sample", "pressure",
)


def finite_or_nan(field):
    return field.where(np.isfinite(field) & (abs(field) < 1e30))


def differentiate_wrt(da, dim, coordinate_values, edge_order=2):
    """Differentiate on physical, potentially nonuniform coordinates."""
    coord = np.asarray(coordinate_values, dtype=np.float64)
    if dim not in da.dims or coord.shape != (da.sizes[dim],):
        raise ValueError(f"Invalid derivative coordinate for {dim}")
    if coord.size <= edge_order or not np.isfinite(coord).all():
        raise ValueError(f"Need at least {edge_order + 1} finite points along {dim}")
    if not (np.all(np.diff(coord) > 0) or np.all(np.diff(coord) < 0)):
        raise ValueError(f"Derivative coordinate {dim} must be strictly monotonic")
    original = da[dim]
    if hasattr(da.data, "chunks"):
        da = da.chunk({dim: -1})
    return (da.assign_coords({dim: coord}).differentiate(dim, edge_order=edge_order)
            .assign_coords({dim: original}).transpose(*da.dims))


def log_pressure_height(pressure_hpa):
    return -SCALE_HEIGHT_M * np.log(np.asarray(pressure_hpa, dtype=float) / 1000.0)


def background_coefficients(background, dim="pres", height=None):
    height = log_pressure_height(background.pres) if height is None else height
    phi = np.deg2rad(np.asarray(background.latitude.values, dtype=float))
    latitude = xr.DataArray(phi, dims="latitude", coords={"latitude": background.latitude})
    u_z = differentiate_wrt(background.u, dim, height)
    theta_z = differentiate_wrt(background.theta, dim, height)
    u_phi = differentiate_wrt(background.u, "latitude", phi)
    return xr.Dataset({
        "u_z": u_z,
        "theta_z": theta_z.where(theta_z > MINIMUM_THETA_GRADIENT),
        "absolute_vorticity": 2 * OMEGA * np.sin(latitude)
        - (u_phi - np.tan(latitude) * background.u) / EARTH_RADIUS,
    })


def epfd_flux_terms(covariances, coefficients):
    """Return density-normalized fluxes before divergence."""
    q = covariances.vtheta / coefficients.theta_z
    return xr.Dataset({
        "momentum_meridional": -covariances.uv,
        "momentum_vertical": -covariances.uw,
        "heat_meridional": coefficients.u_z * q,
        "heat_vertical": coefficients.absolute_vorticity * q,
    })


def flux_divergence(fluxes, density, dim, height):
    """Differentiate density-bearing fluxes with the spherical metric."""
    phi = np.deg2rad(np.asarray(fluxes.latitude.values, dtype=float))
    cos2 = xr.DataArray(np.cos(phi)**2, dims="latitude", coords={"latitude": fluxes.latitude})
    result = {}
    for name in FLUX_TERMS:
        if name.endswith("meridional"):
            value = differentiate_wrt(density * cos2 * fluxes[name], "latitude", phi)
            value = value / (EARTH_RADIUS * density * cos2)
        else:
            value = differentiate_wrt(density * fluxes[name], dim, height) / density
        result[name] = value.transpose(*fluxes[name].dims)
    result["EPFd_y"] = result["momentum_meridional"] + result["heat_meridional"]
    result["EPFd_z"] = result["momentum_vertical"] + result["heat_vertical"]
    result["EPFd_momentum"] = result["momentum_meridional"] + result["momentum_vertical"]
    result["EPFd_heat"] = result["heat_meridional"] + result["heat_vertical"]
    result["EPFd"] = result["EPFd_y"] + result["EPFd_z"]
    return xr.Dataset(result)


def epfd_from_flux_terms(fluxes):
    density = fluxes.pres.astype(float)*100/(GRAVITY*SCALE_HEIGHT_M)
    result = flux_divergence(fluxes, density, "pres", log_pressure_height(fluxes.pres))
    for field in result.data_vars.values():
        field.attrs.update(units="m s-2", reference="https://doi.org/10.1175/JAS-D-20-0249.1")
    return result


def residual_velocities(v_mean, w_mean, q, density, dim, height):
    phi = np.deg2rad(np.asarray(q.latitude.values, dtype=float))
    cos = xr.DataArray(np.cos(phi), dims="latitude", coords={"latitude": q.latitude})
    mass_q = density * q
    return xr.Dataset({
        "vres": v_mean - differentiate_wrt(mass_q, dim, height) / density,
        "wres": w_mean + differentiate_wrt(cos * mass_q, "latitude", phi)
        / (EARTH_RADIUS * cos * density),
    })


def _complete_samples(fields):
    valid = xr.ones_like(fields[0], dtype=bool)
    for field in fields:
        valid = valid & np.isfinite(field)
    return valid.all("longitude")


def _assemble(background, covariances, density, dim, height, formulation):
    coefficients = background_coefficients(background, dim, height)
    fluxes = epfd_flux_terms(covariances, coefficients)
    out = flux_divergence(fluxes, density, dim, height)
    out.update(residual_velocities(background.v, background.w,
                                  covariances.vtheta / coefficients.theta_z, density, dim, height))
    out.update(coefficients)
    out.update(covariances)
    phi = xr.DataArray(np.deg2rad(background.latitude.astype(float)), dims="latitude",
                       coords={"latitude": background.latitude})
    out["Fy"] = density * EARTH_RADIUS * np.cos(phi) * (
        fluxes.momentum_meridional + fluxes.heat_meridional)
    out["Fz"] = density * EARTH_RADIUS * np.cos(phi) * (
        fluxes.momentum_vertical + fluxes.heat_vertical)
    out["rho_zm"] = density.broadcast_like(background.u)
    for name in ("u", "v", "w", "theta"):
        out[f"{name}_bar"] = background[name]
    out["t"] = background.temperature
    for name in out:
        out[name] = finite_or_nan(out[name]).transpose(*background.u.dims)
    out.attrs.update(
        tem_version=TEM_VERSION, tem_formulation=formulation, hemispheric_fold="disabled",
        heat_streamfunction="northward heat-flux gauge; Q=<v'theta'>/theta_z in log-pressure",
        stability_minimum_k_m=MINIMUM_THETA_GRADIENT,
        scale_height_m=SCALE_HEIGHT_M, gravity_m_s2=GRAVITY,
        reference_pressure_pa=P0_PA,
    )
    units = {**{n: "m s-2" for n in (*FLUX_TERMS, "EPFd", "EPFd_y", "EPFd_z", "EPFd_momentum", "EPFd_heat")},
             **{n: "m s-1" for n in ("u_bar", "v_bar", "w_bar", "vres", "wres")},
             "theta_bar": "K", "t": "K", "rho_zm": "kg m-3", "Fy": "kg s-2", "Fz": "kg s-2",
             "u_z": "s-1", "theta_z": "K m-1", "absolute_vorticity": "s-1",
             "uv": "m2 s-2", "uw": "m2 s-2", "vtheta": "K m s-1"}
    for name, field in out.data_vars.items():
        field.attrs.update(units=units.get(name, "1"), tem_version=TEM_VERSION,
                           tem_formulation=formulation, hemispheric_fold="disabled")
    out.EPFd.attrs["long_name"] = "Acceleration due to EP flux convergence"
    out.rho_zm.attrs["long_name"] = ("Log-pressure reference density p/(g H)" if formulation == "log-pressure"
                                      else "Zonal mean mass density")
    return out


def compute_pressure_tem(dataset):
    """Compute unfiltered instantaneous TEM on archived isobaric surfaces."""
    dataset = dataset.sortby("pres")
    p = dataset.pres.astype(float) * 100.0
    fields = [finite_or_nan(dataset.uzonal_isobaric).astype(float),
              finite_or_nan(dataset.umeridional_isobaric).astype(float),
              -SCALE_HEIGHT_M * finite_or_nan(dataset.omg_isobaric).astype(float) / p,
              finite_or_nan(dataset.t_isobaric).astype(float) * (P0_PA / p)**KAPPA]
    valid = _complete_samples(fields)
    u, v, w, theta = [field.where(valid) for field in fields]
    means = [field.mean("longitude", skipna=False) for field in (u, v, w, theta)]
    ub, vb, wb, tb = means
    background = xr.Dataset(dict(u=ub, v=vb, w=wb, theta=tb,
                                  temperature=tb * (p / P0_PA)**KAPPA))
    cov = xr.Dataset(dict(uv=((u-ub)*(v-vb)).mean("longitude", skipna=False),
                          uw=((u-ub)*(w-wb)).mean("longitude", skipna=False),
                          vtheta=((v-vb)*(theta-tb)).mean("longitude", skipna=False)))
    out = _assemble(background, cov, p/(GRAVITY*SCALE_HEIGHT_M), "pres",
                    log_pressure_height(dataset.pres), "log-pressure")
    out["valid_zonal_sample"] = valid.astype("int8")
    out.valid_zonal_sample.attrs.update(units="1", long_name="Complete finite longitude circle", tem_version=TEM_VERSION)
    out["pressure"] = p.broadcast_like(ub).transpose(*ub.dims)
    out.pressure.attrs["units"] = "Pa"
    out.pres.attrs.update(units="hPa", long_name="pressure")
    return out


def compute_height_tem(u, v, w, theta, density, pressure):
    """HPE(z) mass-flux TEM with the northward heat-flux streamfunction gauge."""
    fields = [finite_or_nan(field).astype(float) for field in (u, v, w, theta, density, pressure)]
    valid = _complete_samples(fields) & (fields[4] > 0).all("longitude")
    u, v, w, theta, rho, pressure = [field.where(valid) for field in fields]
    mean = lambda field: field.mean("longitude", skipna=False)
    rb, ub, tb = mean(rho), mean(u), mean(theta)
    rv, rw = mean(rho*v), mean(rho*w)
    background = xr.Dataset(dict(u=ub, v=rv/rb, w=rw/rb, theta=tb,
        temperature=mean(theta * (pressure / P0_PA)**KAPPA)))
    cov = xr.Dataset(dict(uv=mean((rho*v-rv)*(u-ub))/rb,
                          uw=mean((rho*w-rw)*(u-ub))/rb,
                          vtheta=mean((rho*v-rv)*(theta-tb))/rb))
    out = _assemble(background, cov, rb, "z", u.z.values, "geometric-height")
    out["density_wind_covariance"] = mean((rho-rb)*(u-ub))
    out.density_wind_covariance.attrs.update(units="kg m-2 s-1", tem_version=TEM_VERSION,
                                            tem_formulation="geometric-height")
    out["pressure"] = mean(pressure)
    out.pressure.attrs["units"] = "Pa"
    out["valid_zonal_sample"] = valid.astype("int8")
    out.valid_zonal_sample.attrs.update(units="1", long_name="Complete finite longitude circle", tem_version=TEM_VERSION)
    # Preserve familiar covariance names, but identify their revised definitions.
    for name, original in (("upvp", "uv"), ("upwp", "uw"), ("vpthp", "vtheta")):
        out[name] = out[original].copy()
        out[name].attrs["definition"] = "density-normalized mass-flux covariance; HPE(z)"
    out["vptp"] = mean((rho*v-rv) * (theta*(pressure/P0_PA)**KAPPA-background.temperature))/rb
    out.vptp.attrs.update(units="K m s-1", tem_version=TEM_VERSION, tem_formulation="geometric-height")
    out.attrs.update(
        reference="https://doi.org/10.1175/2010JAS3355.1",
        heat_streamfunction="psi/(a*cos(phi)) = <(rho*v)'theta'> / theta_z",
        momentum_storage="a_density_storage = -d<rho'u'>/dt / <rho>",
        dynamical_approximation="hydrostatic shallow-atmosphere HPE(z)",
    )
    return out
