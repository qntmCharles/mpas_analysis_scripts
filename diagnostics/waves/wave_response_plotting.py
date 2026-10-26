"""Plot pooled wave-response fields and pointwise bootstrap support."""

from __future__ import annotations

from typing import Any

import numpy as np
import xarray as xr
from matplotlib.legend_handler import HandlerBase
from diagnostics.mean_state import circulation_mechanism as mechanism


TROPOPAUSE_COLOURS = ("#16843b", "#16843b")
U_ZONAL_WIND_COLOUR = "#16843b"


def _constrained_wave_figure(rows, height, *, sidebar=False):
    """Reserve a bottom row for vector calibration and the shared legend."""
    import matplotlib.pyplot as plt
    figure=plt.figure(figsize=(8,height),layout="constrained")
    figure.get_layout_engine().set(w_pad=.035,h_pad=.035,wspace=.02,hspace=.025)
    outer=figure.add_gridspec(2,1,height_ratios=[rows*2.1,.62 if sidebar else .72])
    panels=(outer[0].subgridspec(rows,4,width_ratios=[1,1,1,.43],height_ratios=[1,1])
        if sidebar else outer[0].subgridspec(rows,3))
    axes=np.empty((rows,3),dtype=object)
    first=None
    for row in range(rows):
        for column in range(3):
            axis=figure.add_subplot(panels[row,column],sharex=first,sharey=first)
            if first is None:
                first=axis
            axis.tick_params(labelleft=column==0,labelbottom=row==rows-1)
            axes[row,column]=axis
    if sidebar:
        colourbar_axes=[]
        for row in range(rows):
            bar=panels[row,3].subgridspec(1,2,width_ratios=[.2,.8])
            colourbar_axes.append(figure.add_subplot(bar[0,0]))
        colourbar_axis=tuple(colourbar_axes)
        footer=outer[1].subgridspec(1,2,width_ratios=[1.25,1.75])
        key_axis,legend_axis=[figure.add_subplot(footer[0,column]) for column in range(2)]
    else:
        footer=outer[1].subgridspec(1,2,width_ratios=[1,2.8])
        key_axis,legend_axis=[figure.add_subplot(footer[0,column]) for column in range(2)]
        colourbar_axis=None
    for axis in (key_axis,legend_axis):
        axis.set_axis_off()
    return figure,axes,key_axis,legend_axis,colourbar_axis


def _bottom_wave_legend(figure,axis,handles,handler_map=None,*,columns=2,left_align=False):
    return figure.legend(handles=handles,handler_map=handler_map,
        loc="center left" if left_align else "center",bbox_to_anchor=(0. if left_align else .5,.5),
        bbox_transform=axis.transAxes,ncol=columns,frameon=False,fontsize=6.5)


class _EnergyFluxLegendHandler(HandlerBase):
    """Draw the plotted quiver geometry, colours, and outline in the legend."""

    def create_artists(self, legend, original, xdescent, ydescent, width, height, fontsize, transform):
        from matplotlib.patches import Polygon
        shaft = original.width*original.axes.get_window_extent().width/original.figure.dpi*72.
        x, y = original._h_arrows(np.array([width/shaft]))
        vertices = np.column_stack((x[0]*shaft+width/2.-xdescent, y[0]*shaft+height/2.-ydescent))
        arrow = Polygon(vertices, facecolor=original.get_facecolor()[0], edgecolor=original.get_edgecolor()[0],
            linewidth=original.get_linewidth()[0], transform=transform)
        arrow.set_gid("energy_flux_legend_arrow")
        return [arrow]


def load_circulation_guides(*, start, end):
    """Reuse current circulation caches for the same analysis period."""
    import json
    from diagnostics.mean_state import circulation_diagnostics as circulation

    fields = []
    for name in ("control", "nudged-u"):
        path = circulation.cache_path("circulation_fields.nc", circulation.simulation_cache_root(name))
        with xr.open_dataset(path) as cached:
            configuration = json.loads(cached.attrs["cache_manifest"])["configuration"]
            if cached.attrs.get("cache_schema") != circulation.CACHE_SCHEMA["circulation"]:
                raise ValueError(f"Rebuild the circulation guide cache: {path}")
            for key, expected in (("analysis_start", start), ("analysis_end", end)):
                if np.datetime64(configuration[key]) != np.datetime64(expected):
                    raise ValueError(f"Circulation guides use a different analysis period: {path}")
            fields.append(cached[["zonal_wind", "tropopause_pressure"]].load().assign_coords(simulation=[name]))
    return xr.concat(fields, dim="simulation", combine_attrs="drop_conflicts")


def plot_circulation_guides(axis, guides, *, run=1, reference=0, show_tropopause=True):
    """Match the circulation publication's 40 m/s STJ and tropopause guides."""
    if guides is None:
        return
    from diagnostics.mean_state import circulation_diagnostics as circulation

    first_line = len(axis.lines)
    circulation._plot_publication_jet_and_tropopause(
        axis, guides, run, reference, circulation._publication_pressure_levels(),
        stj_levels=(40.,), tropopause_colours=TROPOPAUSE_COLOURS, show_tropopause=show_tropopause,
    )
    if run == 0 and reference is None:
        for line in axis.lines[first_line:]:
            line.set_linestyle("--")
            line.set_linewidth(.8)
            line.set_alpha(.7)


def circulation_guide_handles():
    """Keep one guide legend across wave figures."""
    from diagnostics.mean_state import circulation_diagnostics as circulation

    handles, _ = circulation._publication_jet_tropopause_handles(
        stj_levels=(40.,), tropopause_colours=TROPOPAUSE_COLOURS,
    )
    for handle, label in zip(handles, ("STJ: 40 m/s", "U tropopause", "Control tropopause")):
        handle.set_label(label)
    return handles


def equatorial_display_field(field: xr.DataArray) -> xr.DataArray:
    """Extend folded scalar shading to zero using the nearest positive-latitude sample.

    Preserve an existing finite equatorial value. This changes display coordinates
    only; it provides neither an equatorial measurement nor a significance test.
    """
    positive = field.where(field.absolute_latitude > 0., drop=True)
    boundary = positive.isel(absolute_latitude=0, drop=True).expand_dims(absolute_latitude=[0.])
    if 0. in field.absolute_latitude:
        boundary = field.sel(absolute_latitude=[0.]).fillna(boundary)
    return xr.concat([boundary, positive], dim="absolute_latitude").transpose(*field.dims)


def plot_region_box(axis):
    """Reuse the circulation publication's regional bounds and box style."""
    from matplotlib.patches import Rectangle
    x0, x1 = mechanism.REGION_LATITUDE
    p0, p1 = mechanism.REGION_PRESSURE
    return axis.add_patch(Rectangle((x0, p0), x1-x0, p1-p0,
        fill=False, zorder=8, **mechanism.REGION_BOX_STYLE))


def _nearest_grid(field: xr.DataArray, pressure: np.ndarray, latitude: np.ndarray) -> np.ndarray:
    return field.sortby("pres").interp(
        pres=xr.DataArray(pressure, dims="pres"),
        absolute_latitude=xr.DataArray(latitude, dims="absolute_latitude"),
        method="nearest",
    ).transpose("pres", "absolute_latitude").values


def _time_mean_if_present(field: xr.DataArray) -> xr.DataArray:
    return field.mean("Time") if "Time" in field.dims else field


def _significance_plot_grid(significant, estimate, *, pressure="pres", latitude="absolute_latitude", extend_equator=True):
    """Sample support on regular display cells; do not infer equatorial support."""
    p0, p1 = float(estimate[pressure].min()), float(estimate[pressure].max())
    available = np.isfinite(estimate).any(pressure)
    if not bool(available.any()):
        return None
    x0 = float(estimate[latitude].where(available, drop=True).min())
    x1 = float(estimate[latitude].max())
    if extend_equator:
        estimate = equatorial_display_field(estimate)
        significant = equatorial_display_field(significant.fillna(False))
    pressure_centres = np.geomspace(p0, p1, 193)
    latitude_centres = (np.arange(.625, x1, 1.25) if extend_equator
        else np.arange(x0, x1+.125, .25))
    if latitude_centres.size == 0:
        return None
    valid = _nearest_grid(np.isfinite(estimate).astype(float), pressure_centres, latitude_centres) > 0.5
    supported = _nearest_grid(significant.fillna(False).astype(float), pressure_centres, latitude_centres) > 0.5
    if extend_equator:
        supported[:, latitude_centres - .625 < x0] = False
        latitude_edges = np.r_[latitude_centres - .625, latitude_centres[-1] + .625]
    else:
        latitude_edges = np.r_[x0, (latitude_centres[:-1]+latitude_centres[1:])/2, x1]
    logp = np.log(pressure_centres)
    log_edges = np.r_[logp[0] - (logp[1] - logp[0]) / 2,
                      (logp[:-1] + logp[1:]) / 2,
                      logp[-1] + (logp[-1] - logp[-2]) / 2]
    pressure_edges = np.exp(log_edges)
    return dict(pressure=pressure_centres, latitude=latitude_centres, pressure_edges=pressure_edges,
        latitude_edges=latitude_edges, valid=valid, significant=supported,
        estimate=_nearest_grid(estimate,pressure_centres,latitude_centres))


def hatch_insignificant(
    axis: Any,
    significant: xr.DataArray,
    estimate: xr.DataArray,
    *,
    pressure: str = "pres",
    latitude: str = "absolute_latitude",
    contrast_shading: tuple[xr.DataArray, Any] | None = None,
    extend_equator: bool = True,
) -> Any:
    """Hatch finite nonsignificant cells on a regular log-pressure grid."""
    grid=_significance_plot_grid(significant,estimate,pressure=pressure,latitude=latitude,extend_equator=extend_equator)
    if grid is None:
        return None
    insignificant = grid["valid"] & ~grid["significant"]
    layers = [(insignificant, "0.35", 0.4, 0.25)]
    if contrast_shading is not None:
        shade, image = contrast_shading
        values = _nearest_grid(shade, grid["pressure"], grid["latitude"])
        rgba = image.to_rgba(values)
        alpha = rgba[..., 3:4]
        if image.get_alpha() is not None:
            alpha = alpha * image.get_alpha()
        rgb = rgba[..., :3] * alpha + (1.0 - alpha)
        dark = (rgb @ np.array([0.2126, 0.7152, 0.0722])) < 0.6
        layers = [(insignificant & ~dark, "0.25", 0.65, 0.30),
                  (insignificant & dark, "white", 0.85, 0.30)]
    artists = []
    for mask, colour, opacity, line_width in layers:
        artists.append(axis.pcolor(
            grid["latitude_edges"], grid["pressure_edges"],
            np.ma.masked_where(~mask, np.ones(mask.shape)), shading="flat",
            facecolor="none", edgecolor=colour, hatch="//", hatch_linewidth=line_width,
            linewidth=0, alpha=opacity, zorder=2.5,
        ))
    return artists[0]


def _significant_forcing_outlines(axis,significant,estimate,colours):
    """Lightly smooth display boundaries from the unchanged significance mask.

    Gaussian widths are 0.4 degrees in latitude and 0.02 in log pressure.
    Use the full source domain before the axis crop, as with forcing hatching.
    """
    from scipy.ndimage import gaussian_filter
    grid=_significance_plot_grid(significant,estimate)
    if grid is None:
        return []
    # Subdivide cells before smoothing to avoid retaining the original stair steps.
    latitude_edges=grid["latitude_edges"]
    logp_edges=np.log(grid["pressure_edges"])
    rows,columns=grid["valid"].shape
    latitude_step=(latitude_edges[-1]-latitude_edges[0])/(10*columns)
    logp_step=(logp_edges[-1]-logp_edges[0])/(2*rows)
    latitude=latitude_edges[0]+(np.arange(10*columns)+.5)*latitude_step
    pressure=np.exp(logp_edges[0]+(np.arange(2*rows)+.5)*logp_step)
    valid=grid["valid"].repeat(2,axis=0).repeat(10,axis=1)
    artists=[]
    for sign,colour in colours:
        region=grid["valid"] & grid["significant"] & ((grid["estimate"]>0) if sign=="positive" else (grid["estimate"]<0))
        if not region.any():
            continue
        smooth=gaussian_filter(region.astype(float).repeat(2,axis=0).repeat(10,axis=1),
            sigma=(.02/logp_step,.4/latitude_step),mode="constant",cval=0.)
        if smooth.max()<=.5:
            continue
        artist=axis.contour(latitude,pressure,np.ma.masked_where(~valid,smooth),
            levels=[.5],colors=colour,linewidths=1.,corner_mask=False,zorder=6)
        artist.set_gid("significant_epfd_"+sign)
        artists.append(artist)
    return artists


SECTORS=("low_eastward","low_westward","high_wavenumber")
WIND_LEVELS={"low_eastward":[0,5,10,20,30],"low_westward":[-30,-20,-10,-5,0],
             "high_wavenumber":[-30,-20,-10,-5,0,5,10,20,30]}
VARIANCE_LEVELS={"low_eastward":(-12.,-4.,4.,12.),"low_westward":(-12.,-4.,4.,12.),
                 "high_wavenumber":(-1.,-.2,.2,1.)}


def variance_change_contours(axis,change,*,sector):
    """Show signed variance changes with one line width."""
    field=equatorial_display_field(change.sel(sector=sector))
    for positive,colour,style in ((False,"#276419","--"),(True,"#8e0152","-")):
        levels=[value for value in VARIANCE_LEVELS[sector]
                if (value>0)==positive and float(field.min())<value<float(field.max())]
        if levels:
            contour=axis.contour(field.absolute_latitude,field.pres,field.transpose("pres","absolute_latitude"),
                levels=levels,colors=colour,linestyles=style,linewidths=.7,zorder=5.5)
            axis.clabel(contour,fmt=lambda value:f"{value:+g}",fontsize=5.5,inline_spacing=2)


def _draw_ep_vectors(figure,requests,key_axis=None):
    """Use Jucker geometry and one tropical reference for every requested panel."""
    from matplotlib.patches import Rectangle
    from matplotlib.quiver import Quiver
    from diagnostics.mean_state import circulation_diagnostics as circulation
    pressures=np.geomspace(80.,340.,6)
    latitudes=np.arange(2.,20.,4.)
    # Resolve constrained layout before calibrating arrows against the final axes.
    figure.canvas.draw()
    figure.set_layout_engine("none")
    displays=[]
    for axis,product,sector,draw in requests:
        grid=xr.Dataset({name:circulation._publication_pressure_field(
            product[variable].sel(sector=sector),pressures).interp(absolute_latitude=latitudes)
            for name,variable in (("horizontal","ep_flux_horizontal"),("upward","ep_flux_upward"))},
            attrs=dict(vector_kind="ep",latitude_span_degrees=40.,pressure_span_log=np.log(20.)))
        displays.append(circulation._display_vector_grid(grid,axis))
    reference=circulation._publication_vector_reference(displays)
    reference_length=.18*min(min(axis.get_window_extent().size/figure.dpi) for axis,_,_,_ in requests)
    add_key=True
    for (axis,product,_,draw),grid in zip(requests,displays):
        if draw:
            first_collection=len(axis.collections)
            circulation._plot_publication_vector_arrows(axis,grid,reference,[],product.pres,
                add_key=add_key and key_axis is None,minimum_reference_fraction=0.,reference_length_inches=reference_length,
                key_offset=(-.75,.38))
            clip=Rectangle((0.,0.),20.,1.,transform=axis.get_xaxis_transform())
            for artist in axis.collections[first_collection:]:
                if isinstance(artist,Quiver) and artist.angles=="xy":
                    artist.set_clip_path(clip)
            add_key=False
    if key_axis is not None:
        example=circulation._quiver_example_value(.5*reference)
        exponent=int(np.floor(np.log10(example)))
        magnitude=example/10.**exponent
        width,height=key_axis.get_window_extent().size/figure.dpi
        displacement=example*reference_length/reference
        keys=circulation._add_component_quiver_keys(key_axis,displacement/width,displacement/height,
            rf"$\mathcal{{F}}_x={magnitude:g}$",rf"$\mathcal{{F}}_y={magnitude:g}$",width=.0045,
            common_units=rf"$10^{{{exponent}}}$ m$^3$ in",offset=(-.38,0.))
        for arrow in keys:
            arrow.set_edgecolor("white")
            arrow.set_linewidth(.35)


def _pressure_axis(axis,*,ylabel=False,xlabel=False,latitude_limit=40,bottom_pressure=350):
    from matplotlib.ticker import NullFormatter
    axis.set(xlim=(0,latitude_limit),ylim=(bottom_pressure,70),yscale="log")
    ticks=[400,300,200,170,100,70] if bottom_pressure==400 else [350,250,170,100,70]
    axis.set_yticks(ticks,labels=[str(value) for value in ticks])
    axis.set_xticks(np.arange(0,latitude_limit+1,5 if latitude_limit==25 else 10))
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.tick_params(labelsize=6.5)
    if ylabel:
        axis.set_ylabel("Pressure (hPa)",fontsize=7)
    if xlabel:
        axis.set_xlabel(r"Latitude ($^\circ$)",fontsize=7)


def _filled_pressure_map(axis,field,*,cmap,norm,extend="neither",extend_equator=True,intervals=32):
    """Interpolate contour boundaries between samples without spatial smoothing."""
    if extend_equator:
        field = equatorial_display_field(field)
    return axis.contourf(
        field.absolute_latitude,field.pres,field.transpose("pres","absolute_latitude"),
        levels=np.linspace(norm.vmin,norm.vmax,intervals+1),cmap=cmap,norm=norm,extend=extend,zorder=1,
    )


def plot_band_response(control,experiment,forcing_confidence,variance_change=None,circulation_guides=None):
    """Decompose forcing changes by direction, with absolute U EP flux above."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from diagnostics.analysis_config import simulation_colour

    components=("total","meridional","vertical")
    labels=("Low-frequency eastward","Low-frequency westward","High wavenumbers")
    winds=[(equatorial_display_field(_time_mean_if_present(control.zonal_wind)),simulation_colour("control"),"--"),
        (equatorial_display_field(_time_mean_if_present(experiment.zonal_wind)),U_ZONAL_WIND_COLOUR,"-")]
    figure,axes,key_axis,legend_axis,_=_constrained_wave_figure(3,7.)
    for row,component in enumerate(components):
        maps=[forcing_confidence.estimate.sel(sector=sector,component=component) for sector in SECTORS]
        limit=max(float(np.nanquantile(np.abs(np.stack([field.values for field in maps])),.98)),1e-12)
        for column,(sector,label,field) in enumerate(zip(SECTORS,labels,maps)):
            axis=axes[row,column]
            image=_filled_pressure_map(axis,field,cmap="RdBu_r",
                norm=TwoSlopeNorm(vmin=-limit,vcenter=0.,vmax=limit),extend="both")
            image.set_gid("forcing_"+component)
            for wind,colour,style in winds:
                levels=[value for value in WIND_LEVELS[sector] if float(wind.min())<value<float(wind.max())]
                if levels:
                    contours=axis.contour(wind.absolute_latitude,wind.pres,wind.transpose("pres","absolute_latitude"),
                        levels=levels,colors=colour,linewidths=.55,zorder=4,linestyles=style)
                    axis.clabel(contours,fmt="%g",fontsize=5.5)
            significant=forcing_confidence.significant.sel(sector=sector,component=component)
            hatch_insignificant(axis,significant,field,contrast_shading=(field,image))
            if variance_change is not None:
                variance_change_contours(axis,variance_change,sector=sector)
            plot_region_box(axis)
            _pressure_axis(axis,xlabel=row==2)
            if row==0:
                axis.set_title(label.replace("frequency ","frequency\n"),fontsize=7.5,pad=5)
            if column==0:
                axis.set_ylabel(f"{component.capitalize()} filtered EPFd anomaly\nPressure (hPa)",fontsize=7)
        colourbar=figure.colorbar(image,ax=axes[row,:].tolist(),location="right",shrink=.82,
            pad=.02,label="Filtered EPFd anomaly (m/s/day)",extend="both",fraction=.035)
        ticks=MaxNLocator(nbins=5).tick_values(image.norm.vmin,image.norm.vmax)
        colourbar.set_ticks(ticks[(ticks>=image.norm.vmin)&(ticks<=image.norm.vmax)])
        colourbar.minorticks_off()
        colourbar.ax.tick_params(labelsize=6)
        colourbar.ax.yaxis.label.set_size(6.5)
    handles=[Patch(facecolor="white",edgecolor=".35",hatch="//",label="Filtered EPFd: 95% interval includes zero"),
        Line2D([],[],color="#8e0152",lw=.7,label=r"Positive $\Delta\overline{u'^2+v'^2}$"),
        Line2D([],[],color="#276419",ls="--",lw=.7,label=r"Negative $\Delta\overline{u'^2+v'^2}$"),
        Line2D([],[],color=simulation_colour("control"),ls="--",lw=.55,label="Control zonal wind (m/s)"),
        Line2D([],[],color=U_ZONAL_WIND_COLOUR,lw=.55,label="U zonal wind (m/s)")]
    _bottom_wave_legend(figure,legend_axis,handles)
    requests=[(axis,product,sector,draw) for axis,sector in zip(axes[0],SECTORS)
        for product,draw in ((experiment,True),(control,False))]
    _draw_ep_vectors(figure,requests,key_axis=key_axis)
    return figure


def _energy_vector_offsets(poleward, upward, pressure, height_profile, distance_per_flux):
    """Convert physical flux components to endpoints using one reference pressure–height profile."""
    from diagnostics.tem_core import EARTH_RADIUS
    logarithmic_pressure = np.log(height_profile.pres.values)
    heights = height_profile.values
    initial_height = np.interp(np.log(pressure),logarithmic_pressure,heights)
    endpoint_height = initial_height+distance_per_flux*np.asarray(upward)
    if np.any((endpoint_height<heights.min()) | (endpoint_height>heights.max())):
        raise ValueError("Energy-vector endpoints exceed the reference pressure–height profile")
    endpoint_pressure = np.exp(np.interp(endpoint_height,heights[::-1],logarithmic_pressure[::-1]))
    return np.rad2deg(distance_per_flux*np.asarray(poleward)/EARTH_RADIUS),endpoint_pressure-pressure


def _draw_energy_vectors(figure, axes, products, *, key_axis, height_profile, distance_per_flux,
        reference_values, anomaly=False, integrated_key=False):
    """Display physical wave-energy flux with one fixed scale and the axes' geometric exaggeration."""
    from matplotlib.patches import Rectangle
    # Freeze the resolved layout so later legend drawing preserves vector geometry.
    figure.canvas.draw()
    figure.set_layout_engine("none")
    bottom,top=axes[0,0].get_ylim()
    pressure_span=np.log(bottom/top)
    pressures, latitudes = np.geomspace(top+10.,bottom-20.,6), np.arange(2.,25.,4.)
    plotted = []
    for row, product in enumerate(products):
        for column, sector in enumerate(SECTORS):
            axis = axes[row,column]
            field = product.sel(sector=sector).interp(pres=pressures,absolute_latitude=latitudes)
            latitude,pressure = np.meshgrid(latitudes,pressures)
            # One physical length multiplier acts on both components before coordinate conversion.
            horizontal,vertical = _energy_vector_offsets(
                field.energy_flux_poleward.transpose("pres","absolute_latitude"),
                field.energy_flux_upward.transpose("pres","absolute_latitude"),pressure,height_profile,distance_per_flux)
            arrows = axis.quiver(latitude,pressure,horizontal,vertical,
                angles="xy",scale_units="xy",scale=1.,pivot="mid",color=".08",edgecolor="white",linewidth=.2,
                width=.006,headwidth=3.8,headlength=4.5,minlength=0,zorder=7)
            arrows.set_clip_path(Rectangle((0.,0.),25.,1.,transform=axis.get_xaxis_transform()))
            plotted.append(arrows)
    width,height = axes[0,0].get_window_extent().size/figure.dpi
    key_width,key_height = key_axis.get_window_extent().size/figure.dpi
    reference_pressure = 170.
    horizontal,vertical = _energy_vector_offsets(reference_values[0],reference_values[1],
        reference_pressure,height_profile,distance_per_flux)
    lengths = (horizontal*width/25.,-np.log1p(vertical/reference_pressure)*height/pressure_span)
    prefix = r"\Delta " if anomaly else ""
    symbols = (prefix+r"\overline{p'_bv'_b}",prefix+r"\overline{p'_bw'_b}")
    sidebar = key_height>key_width
    if integrated_key:
        key_axis.text(.45,.93,"Filtered wave-energy flux anomaly",transform=key_axis.transAxes,
            ha="left",va="center",fontsize=6.5)
        positions=((.495,.66,.58,.66),(.495,.39,.58,.39))
    else:
        key_axis.add_patch(Rectangle((0.,.04),1.,.86,transform=key_axis.transAxes,
            facecolor=".94",edgecolor=".65",linewidth=.4))
        positions = ((.5,.78,.5,.65),(.5,.43,.5,.30)) if sidebar else ((.25,.73,.25,.42),(.75,.73,.75,.42))
    for vertical,(symbol,value,physical_length,position) in enumerate(zip(symbols,reference_values,lengths,positions)):
        x,y,text_x,text_y = position
        length = physical_length/(key_height if vertical else key_width)
        arrow = key_axis.quiver([x],[y],[0. if vertical else length],[length if vertical else 0.],
            transform=key_axis.transAxes,angles="uv",pivot="mid",scale_units="height" if vertical else "width",
            scale=1.,width=.006*width/key_width,headwidth=3.8,headlength=4.5,minlength=0,
            color=".08",edgecolor="white",linewidth=.2,zorder=9,clip_on=False)
        arrow.set_gid("vertical_reference_vector" if vertical else "horizontal_reference_vector")
        arrow.set_label(f"{symbol}={value:.17g} W m-2")
        label=rf"${symbol}={value:g}"+(r"\;\mathrm{W\,m^{-2}}$" if integrated_key else "$")
        key_axis.text(text_x,text_y,label,transform=key_axis.transAxes,
            ha="left" if integrated_key else "center",va="center",fontsize=6.5 if integrated_key else 5.5)
    if integrated_key:
        key_axis.text(.58,.04,"Reference at 170 hPa",transform=key_axis.transAxes,
            ha="left",va="center",fontsize=5.5)
    else:
        key_axis.text(.5,.10 if sidebar else .25,r"W m$^{-2}$",transform=key_axis.transAxes,
            ha="center",va="center",fontsize=5.)
        key_axis.text(.5,.205 if sidebar else .11,"Vertical key: 170 hPa",transform=key_axis.transAxes,
            ha="center",va="center",fontsize=4.5)
    return plotted


def plot_wave_energy_flux(control,experiment,energy,variance,forcing_confidence,*,height_profile,circulation_guides=None):
    """Compare Control and U-minus-Control variance and filtered wave-energy flux."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize, TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MaxNLocator
    from diagnostics.analysis_config import simulation_colour

    figure,axes,key_axis,legend_axis,_=_constrained_wave_figure(2,5.5)
    raw=variance["control"].sel(absolute_latitude=slice(0,25))
    difference=(variance["nudged-u"]-variance["control"]).sel(absolute_latitude=slice(0,25))
    winds=[(equatorial_display_field(_time_mean_if_present(control.zonal_wind)),simulation_colour("control"),"--"),
        (equatorial_display_field(_time_mean_if_present(experiment.zonal_wind)),U_ZONAL_WIND_COLOUR,"-")]
    colours=(("positive","#b35806"),("negative","#542788"))
    for row,field in enumerate((raw,difference)):
        for column,sector in enumerate(SECTORS):
            axis=axes[row,column]
            local=field.sel(sector=sector)
            limit=max(float(np.nanquantile(abs(local.values),.98)),1e-12)
            norm=Normalize(0,limit) if row==0 else TwoSlopeNorm(vmin=-limit,vcenter=0.,vmax=limit)
            image=_filled_pressure_map(axis,local,cmap="YlOrBr" if row==0 else "RdBu_r",
                norm=norm,extend="max" if row==0 else "both")
            image.set_gid("control_variance" if row==0 else "variance_anomaly")
            for wind,colour,style in winds:
                local_wind=wind.sel(absolute_latitude=slice(0,25))
                levels=[value for value in WIND_LEVELS[sector] if float(local_wind.min())<value<float(local_wind.max())]
                if levels:
                    lines=axis.contour(local_wind.absolute_latitude,local_wind.pres,
                        local_wind.transpose("pres","absolute_latitude"),levels=levels,
                        colors=colour,linewidths=.55,linestyles=style,zorder=4)
                    axis.clabel(lines,fmt="%g",fontsize=5.5)
            estimate=forcing_confidence.estimate.sel(sector=sector,component="total")
            significant=forcing_confidence.significant.sel(sector=sector,component="total")
            _significant_forcing_outlines(axis,significant,estimate,colours)
            plot_region_box(axis)
            _pressure_axis(axis,xlabel=row==1,latitude_limit=25)
            if row==0:
                axis.set_title(("Low-frequency\neastward","Low-frequency\nwestward","High wavenumbers")[column],fontsize=7.5)
            if column==0:
                axis.set_ylabel(("Control","U − Control")[row]+"\nPressure (hPa)",fontsize=7)
            colourbar=figure.colorbar(image,ax=axis,location="bottom",pad=.035,fraction=.055,aspect=30,
                label=(r"Filtered $\overline{u'^2+v'^2}$ (m$^2$ s$^{-2}$)",
                    r"Filtered $\Delta\overline{u'^2+v'^2}$ (m$^2$ s$^{-2}$)")[row])
            ticks=MaxNLocator(nbins=4).tick_values(norm.vmin,norm.vmax)
            colourbar.set_ticks(ticks[(ticks>=norm.vmin)&(ticks<=norm.vmax)])
            colourbar.minorticks_off()
            colourbar.ax.tick_params(labelsize=6)
            colourbar.ax.xaxis.label.set_size(6.5)
    arrows=_draw_energy_vectors(figure,axes,[energy["control"],energy["nudged-u"]-energy["control"]],
        key_axis=key_axis,height_profile=height_profile,distance_per_flux=2000.,reference_values=(100.,.5))
    arrows[0].set_label("Filtered wave-energy flux: Control (top)\n"+
        r"U − Control (bottom); $(\overline{p'_bv'_b},\overline{p'_bw'_b})$")
    handles=[Line2D([],[],color=simulation_colour("control"),ls="--",lw=.55,label="Control zonal wind (m/s)"),
        Line2D([],[],color=U_ZONAL_WIND_COLOUR,lw=.55,label="U zonal wind (m/s)"),
        Line2D([],[],color=colours[0][1],lw=1.,label="Positive filtered ΔEPFd: significant region"),
        Line2D([],[],color=colours[1][1],lw=1.,label="Negative filtered ΔEPFd: significant region"),arrows[0]]
    _bottom_wave_legend(figure,legend_axis,handles,handler_map={type(arrows[0]):_EnergyFluxLegendHandler()})
    return figure


def plot_wave_publication_response(control, experiment, energy, variance, forcing_confidence, variance_confidence,
        *,height_profile,circulation_guides=None):
    """Compare forcing and filtered variance anomalies, with pressure-work energy flux."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import FuncFormatter
    from diagnostics.analysis_config import simulation_colour

    figure,axes,key_axis,legend_axis,colourbar_axes=_constrained_wave_figure(2,3.8,sidebar=True)
    forcing_colourbar_axis,variance_colourbar_axis=colourbar_axes
    forcing = forcing_confidence.estimate.sel(component="total",absolute_latitude=slice(0,25),pres=slice(70,400))
    reference = variance["control"].sel(absolute_latitude=slice(0,25),pres=slice(70,400))
    difference = (variance["nudged-u"]-variance["control"]).sel(absolute_latitude=slice(0,25),pres=slice(70,400))
    forcing_limit = .25
    reference_maxima = {sector:max(float(reference.sel(sector=sector).max()),1e-12) for sector in SECTORS}
    reference_scale = xr.DataArray([reference_maxima[sector] for sector in SECTORS],dims="sector",
        coords=dict(sector=list(SECTORS)))
    variance_percent = 100.*difference/reference_scale
    winds = [(_time_mean_if_present(control.zonal_wind),simulation_colour("control"),"--"),
        (_time_mean_if_present(experiment.zonal_wind),U_ZONAL_WIND_COLOUR,"-")]
    for row,field in enumerate((forcing,variance_percent)):
        for column,sector in enumerate(SECTORS):
            axis = axes[row,column]
            local_field = field.sel(sector=sector)
            limit = forcing_limit if row==0 else 10.
            norm = TwoSlopeNorm(vmin=-limit,vcenter=0.,vmax=limit)
            image = _filled_pressure_map(axis,local_field,cmap="RdBu_r" if row==0 else "PuOr",norm=norm,extend="both",
                extend_equator=False,intervals=10)
            image.set_gid("forcing_anomaly" if row==0 else "variance_anomaly")
            significant = (forcing_confidence.significant.sel(sector=sector,component="total") if row==0
                else variance_confidence.significant.sel(sector=sector))
            hatch_insignificant(axis,significant,local_field,contrast_shading=(local_field,image),extend_equator=False)
            for wind,colour,style in winds:
                local = wind.sel(absolute_latitude=slice(0,25))
                levels = [value for value in WIND_LEVELS[sector] if float(local.min())<value<float(local.max())]
                if levels:
                    lines = axis.contour(local.absolute_latitude,local.pres,local.transpose("pres","absolute_latitude"),
                        levels=levels,colors=colour,linewidths=.55,linestyles=style,zorder=4)
                    axis.clabel(lines,fmt="%g",fontsize=5.5)
            plot_region_box(axis)
            _pressure_axis(axis,xlabel=row==1,latitude_limit=25,bottom_pressure=400)
            panel_label=axis.text(0.,1.02,f"({'abcdef'[3*row+column]})",transform=axis.transAxes,
                ha="left",va="bottom",fontsize=7.5,zorder=9,clip_on=False)
            panel_label.set_gid("panel_label")
            if row==0:
                title=("Low-frequency eastward","Low-frequency westward","High wavenumbers")[column]
                axis.set_title(title,fontsize=7.5)
            else:
                colour_interval=.02*reference_maxima[sector]
                axis.text(1.,1.02,rf"Contour interval: {colour_interval:.3g} m$^2$ s$^{{-2}}$",
                    transform=axis.transAxes,ha="right",va="bottom",fontsize=5.5,color=".2",clip_on=False)
            if column==0:
                axis.set_ylabel("Pressure (hPa)",fontsize=7)
            if row==0 and column==2:
                colourbar=figure.colorbar(image,cax=forcing_colourbar_axis,
                    label="Filtered $\\Delta\\mathrm{EPFd}$\n"+r"(m s$^{-1}$ day$^{-1}$)")
                colourbar.set_ticks([-limit,0.,limit])
                colourbar.formatter=FuncFormatter(lambda value,position:f"{value:.2g}")
                colourbar.update_ticks()
                colourbar.minorticks_off()
                colourbar.ax.tick_params(labelsize=6)
                colourbar.ax.xaxis.label.set_size(6.5)
                colourbar.ax.yaxis.label.set_size(6.5)
    variance_colourbar=figure.colorbar(image,cax=variance_colourbar_axis,
        label="Filtered $\\Delta\\overline{u'^2+v'^2}$\n(% Control maximum)")
    variance_colourbar.set_ticks([-10.,-5.,0.,5.,10.])
    variance_colourbar.formatter=FuncFormatter(lambda value,position:f"{value:g}%")
    variance_colourbar.update_ticks()
    variance_colourbar.minorticks_off()
    variance_colourbar.ax.tick_params(labelsize=6)
    variance_colourbar.ax.yaxis.label.set_size(6.5)
    _draw_energy_vectors(figure,axes[1:2],[energy["nudged-u"]-energy["control"]],
        key_axis=key_axis,height_profile=height_profile,distance_per_flux=12000.,reference_values=(25.,.05),
        anomaly=True,integrated_key=True)
    # Match colourbar lengths to the panel frames after the layout resolves.
    for row,colourbar_axis in enumerate(colourbar_axes):
        bar_position=colourbar_axis.get_position()
        panel_position=axes[row,2].get_position()
        colourbar_axis.set_position([bar_position.x0,panel_position.y0,bar_position.width,panel_position.height])
    handles = [Line2D([],[],color=simulation_colour("control"),ls="--",lw=.55,label="Control zonal wind (m/s)"),
        Line2D([],[],color=U_ZONAL_WIND_COLOUR,lw=.55,label="U zonal wind (m/s)"),
        Patch(facecolor="none",edgecolor=".35",hatch="//",label="95% interval includes zero"),
        Patch(facecolor="none",label="Mechanism box",**mechanism.REGION_BOX_STYLE)]
    _bottom_wave_legend(figure,legend_axis,handles,columns=2,left_align=True)
    return figure
