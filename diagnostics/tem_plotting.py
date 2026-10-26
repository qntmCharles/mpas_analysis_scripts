"""Notebook TEM plots using the circulation owner's data and paper conventions.

EP arrows use Jucker (2021), section 3 and Table 2.
Residual arrows use Hollowed et al. (2025), Appendix C, equation C12.
The plotting multiplier is common to both vector components.
"""

import numpy as np
import xarray as xr

from diagnostics.tem_archive import height_coordinate_tem, load_stored_tem
from diagnostics.tem_core import TEM_FORMULATION


SECTION_VARIABLES = {
    "u_bar": (1.0, r"Zonal wind (m s$^{-1}$)"),
    "t": (1.0, "Temperature (K)"),
    "a_ep": (86400.0, r"EPFd (m s$^{-1}$ day$^{-1}$)"),
    "a_ep_meridional": (86400.0, r"Meridional EPFd (m s$^{-1}$ day$^{-1}$)"),
    "a_ep_vertical": (86400.0, r"Vertical EPFd (m s$^{-1}$ day$^{-1}$)"),
    "vres": (1.0, r"$v^*$ (m s$^{-1}$)"),
    "wres": (1000.0, r"$w^*$ (mm s$^{-1}$)"),
    "v_bar": (1.0, r"Mean meridional wind (m s$^{-1}$)"),
    "w_bar": (1000.0, r"Mean vertical wind (mm s$^{-1}$)"),
    "residual_streamfunction": (1e-9, r"Residual streamfunction ($10^9$ kg s$^{-1}$)"),
}


def folded_tem_fields(stored, *, include_guides=True):
    """Compute plotting diagnostics before folding each completed field by parity."""
    from diagnostics.mean_state import circulation_diagnostics as circulation

    if stored.EPFd.attrs.get("tem_formulation") != TEM_FORMULATION:
        raise ValueError("TEM figures require geometric-height fields")
    height = height_coordinate_tem(stored)
    instantaneous = circulation.tem_instantaneous_diagnostics(height)
    parities = {
        "pressure_hpa": "even", "a_ep": "even", "a_ep_meridional": "even",
        "a_ep_vertical": "even", "vres": "odd", "wres": "even",
        "ep_flux_meridional_normalized": "odd", "ep_flux_vertical_normalized": "even",
        "residual_streamfunction": "odd", "residual_streamfunction_phi_gradient": "even",
        "residual_streamfunction_z_gradient": "odd",
    }
    fields = {}
    for name, parity in parities.items():
        field = instantaneous[name]
        if name != "pressure_hpa":
            field = circulation.mask_spatial_derivative_boundaries(field)
        fields[name] = circulation.fold_to_absolute(field, parity)
    for name, parity in (("u_bar", "even"), ("t", "even"),
                         ("v_bar", "odd"), ("w_bar", "even")):
        fields[name] = circulation.fold_to_absolute(height[name], parity)
    result = xr.Dataset(fields, attrs=instantaneous.attrs)
    result.pressure_hpa.attrs["tem_formulation"] = height.attrs["tem_formulation"]
    if include_guides:
        # Calculate each hemisphere's WMO tropopause before folding the guide.
        mean = height.mean("Time", skipna=True,
                           keep_attrs=True)
        pressure = mean.pressure.rename(latitude="absolute_latitude") / 100.0
        pressure.attrs["tem_formulation"] = height.attrs["tem_formulation"]
        temperature = circulation._publication_pressure_field(
            mean.t.rename(latitude="absolute_latitude"), np.geomspace(1.0, 1000.0, 161), pressure,
        ).rename(absolute_latitude="latitude")
        tropopause = circulation.tropopause_pressure(temperature)
        result["tropopause_pressure"] = circulation.fold_to_absolute(tropopause, "even")
        result["guide_zonal_wind"] = circulation.fold_to_absolute(mean.u_bar, "even")
    return result


def load_folded_tem(directory, *, start, end, formulation=TEM_FORMULATION):
    """Read validated simulation-local fields for a notebook's declared period."""
    if formulation != TEM_FORMULATION:
        raise ValueError("TEM figures require geometric-height fields")
    stored = load_stored_tem(directory, formulation=formulation, start=start, end=end)
    try:
        result = folded_tem_fields(stored)
    except Exception:
        stored.close()
        raise
    result.set_close(stored.close)
    return result


def plot_tem_field(axis, fields, variable, *, levels, latitude_limit=60.0):
    """Plot one folded snapshot or mean with pressure units and circulation guides."""
    from diagnostics.mean_state import circulation_diagnostics as circulation

    scale, label = SECTION_VARIABLES[variable]
    field = (fields[variable] * scale).transpose("z", "absolute_latitude")
    pressure = fields.pressure_hpa.transpose("z", "absolute_latitude")
    horizontal, vertical = circulation._tem_contour_coordinates(field, pressure)
    image = axis.contourf(horizontal, vertical, field, levels=levels,
                          cmap="RdBu_r", extend="both")
    circulation._set_latpres_axis(axis, ylabel=True)
    axis.set_xlim(0.0, latitude_limit)
    axis.set_xticks(np.arange(0.0, latitude_limit + 1.0, 10.0))
    axis.set_yticks([1000, 700, 500, 300, 200, 100, 50],
                   labels=["1000", "700", "500", "300", "200", "100", "50"])
    axis.set_title(label)
    if "tropopause_pressure" in fields:
        wind = circulation._publication_pressure_field(
            fields.guide_zonal_wind, circulation._publication_pressure_levels(), pressure,
        )
        guides = xr.Dataset({"zonal_wind": wind,
                             "tropopause_pressure": fields.tropopause_pressure})
        circulation._plot_publication_jet_and_tropopause(
            axis, guides.expand_dims(simulation=["simulation"]), 0, None,
            circulation._publication_pressure_levels(), stj_levels=(40.0,),
        )
    return image


def draw_tem_vectors(axis, fields, *, kind="ep", reference=None, latitude_limit=60.0):
    """Draw paper vectors after layout; return a reference for subsequent frames."""
    from diagnostics.mean_state import circulation_diagnostics as circulation

    names = {
        "ep": ("ep_flux_meridional_normalized", "ep_flux_vertical_normalized"),
        "circulation": ("residual_streamfunction_phi_gradient", "residual_streamfunction_z_gradient"),
    }[kind]
    pressure = fields.pressure_hpa.transpose("z", "absolute_latitude")
    # The full-domain vector grid does not require a tropopause mask.
    tropopause = fields.get("tropopause_pressure", xr.full_like(fields.absolute_latitude, 150.0))
    grid = circulation._publication_vector_grid(
        fields[names[0]], fields[names[1]], pressure, tropopause, "full",
        latitude_limit_degrees=latitude_limit, kind=kind,
    )
    axis.figure.canvas.draw()
    display = circulation._display_vector_grid(grid, axis)
    if reference is None:
        reference = circulation._publication_vector_reference([display])
    circulation._plot_publication_vector_arrows(
        axis, display, reference, [], pressure, minimum_reference_fraction=0.0,
    )
    return reference
