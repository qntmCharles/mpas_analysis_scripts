"""Disjoint spectral forcing sectors in the fixed circulation mechanism box.

All reductions use cached signed co-spectra and instantaneous coefficients.
Broad sectors partition the regional forcing without identifying individual modes.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy import fft

from diagnostics.tem_core import EARTH_RADIUS, FLUX_TERMS, GRAVITY, OMEGA
from diagnostics.waves.regional_epfd_spectrum import (
    signed_phase_speed_histogram, validate_regional_spectrum,
    PHASE_SPEED_DISPLAY_EDGES, PHASE_SPEED_DISPLAY_CENTRES,
)

SECTOR_ANALYSIS_VERSION = "regional-ep-sectors-v1-symmetric-allocation"
FREQUENCY_CUTOFF = .4
WAVENUMBER_CUTOFF = 20
SECTORS = (
    "low_eastward", "low_westward", "high_frequency_eastward", "high_frequency_westward",
    "high_wavenumber_eastward", "high_wavenumber_westward", "stationary", "unresolved",
)
LOW_SECTORS = SECTORS[:2]
COMPONENT_HATCHES = (("meridional", "/"*3), ("vertical", "\\"*3))
NET_EPFD_COLOUR = "#d62728"
PHASE_SPEED_PLOT_EDGES = PHASE_SPEED_DISPLAY_EDGES[abs(PHASE_SPEED_DISPLAY_EDGES)<=25.]
PHASE_SPEED_PLOT_CENTRES = PHASE_SPEED_DISPLAY_CENTRES[abs(PHASE_SPEED_DISPLAY_CENTRES)<25.]
SECTOR_LABELS = dict(zip(SECTORS, (
    "Low-frequency eastward", "Low-frequency westward", "Higher-frequency eastward",
    "Higher-frequency westward", "High-wavenumber eastward", "High-wavenumber westward",
    "Stationary", "Direction unresolved",
)))
COMPARISONS = ("control", "nudged-u", "nudged-T-from-u", "u_minus_control", "t_minus_control")
COMPARISON_LABELS = ("Control", "U", "T", "U − Control", "T − Control")
COMPARISON_COLOURS = ("0.25", "#2166ac", "#b2182b", "#2166ac", "#b2182b")


def sector_masks(wavenumber, frequency, *, time_count, longitude_count):
    """Partition every resolved bin once, including direction-degenerate modes."""
    k, f = np.asarray(wavenumber), np.asarray(frequency)
    shape = (f.size, k.size)
    stationary = np.broadcast_to(f[:, None] == 0, shape)
    unresolved = np.broadcast_to(k[None, :] == 0, shape).copy()
    if longitude_count % 2 == 0:
        unresolved |= np.broadcast_to(2*abs(k[None, :]) == longitude_count, shape)
    if time_count % 2 == 0:
        unresolved |= np.broadcast_to(np.isclose(f[:, None], 2.), shape)
    unresolved &= ~stationary
    travelling = ~(stationary | unresolved)
    low_k = abs(k)[None, :] <= WAVENUMBER_CUTOFF
    low_f = f[:, None] <= FREQUENCY_CUTOFF + 1e-12
    east, west = k[None, :] > 0, k[None, :] < 0
    values = np.stack([
        travelling & low_k & low_f & east, travelling & low_k & low_f & west,
        travelling & low_k & ~low_f & east, travelling & low_k & ~low_f & west,
        travelling & ~low_k & east, travelling & ~low_k & west, stationary, unresolved,
    ])
    if not np.all(values.sum(axis=0) == 1):
        raise AssertionError("The spectral sectors must partition every Fourier bin")
    return xr.DataArray(values, dims=("sector", "frequency", "wavenumber"),
        coords=dict(sector=list(SECTORS), frequency=f, wavenumber=k))


def comparison_fields(values):
    """Use the same absolute values in each response subtraction."""
    names = [name for name in COMPARISONS[:3] if name in values.simulation]
    labels = names+["u_minus_control" if name == "nudged-u" else "t_minus_control" for name in names[1:]]
    return xr.concat([values.sel(simulation=name, drop=True) for name in names] + [
        values.sel(simulation=name, drop=True)-values.sel(simulation="control", drop=True)
        for name in names[1:]], dim=xr.IndexVariable("comparison", labels))


def forcing_components(by_flux):
    meridional = by_flux.sel(flux_term=[FLUX_TERMS[0], FLUX_TERMS[2]]).sum("flux_term")
    vertical = by_flux.sel(flux_term=[FLUX_TERMS[1], FLUX_TERMS[3]]).sum("flux_term")
    return xr.concat([meridional+vertical, meridional, vertical],
        dim=xr.IndexVariable("component", ["total", "meridional", "vertical"]))


def build_sector_analysis(products):
    """Reduce existing spectra and retain exact directional and sector sums."""
    control = products["control"]
    for product in products.values():
        validate_regional_spectrum(product)
        xr.align(control.forcing, product.forcing, join="exact")
        xr.testing.assert_equal(control.filter_mask, product.filter_mask)
    masks = sector_masks(control.wavenumber, control.frequency,
        time_count=int(control.attrs["time_sample_count"]), longitude_count=int(control.attrs["longitude_sample_count"]))
    names = [name for name in COMPARISONS[:3] if name in products]
    absolute = xr.concat([products[name].forcing_by_flux for name in names],
        dim=xr.IndexVariable("simulation", names))
    spectra = forcing_components(comparison_fields(absolute))*86400.
    sector_forcing = xr.concat([spectra.where(mask, 0.).sum(("frequency", "wavenumber"))
        for mask in masks], dim=masks.sector)
    result = xr.Dataset(dict(sector_forcing=sector_forcing,
        total_forcing=spectra.sum(("frequency", "wavenumber")), sector_mask=masks,
        filter_mask=control.filter_mask, spectrum=spectra))
    for name in ("sector_forcing", "total_forcing", "spectrum"):
        result[name].attrs["units"] = "m s-1 day-1"
    result.attrs.update(product_version=SECTOR_ANALYSIS_VERSION, frequency_cutoff_cpd=FREQUENCY_CUTOFF,
        wavenumber_cutoff=WAVENUMBER_CUTOFF, region="5–15 degrees absolute latitude; 100–170 hPa",
        weighting="fixed mean Control mass; equal hemisphere weights; divergence before regional averaging",
        time_sample_count=int(control.attrs["time_sample_count"]),
        longitude_sample_count=int(control.attrs["longitude_sample_count"]),
        unresolved="nonstationary k=0, temporal Nyquist, and spatial Nyquist; no propagation direction assigned")
    validate_sector_analysis(result)
    return result


def validate_sector_analysis(product):
    np.testing.assert_allclose(product.sector_forcing.sum("sector"), product.total_forcing, atol=1e-11, rtol=1e-10)
    for name in ("sector_forcing", "total_forcing", "spectrum"):
        field = product[name]
        np.testing.assert_allclose(field.sel(component="total"), field.sel(component="meridional")+
            field.sel(component="vertical"), atol=1e-11, rtol=1e-10)


def subset_mask(wavenumber, frequency, definition, *, time_count, longitude_count):
    selected = sector_masks(wavenumber, frequency, time_count=time_count,
        longitude_count=longitude_count).sel(sector=definition["parent"], drop=True)
    k, f = np.asarray(wavenumber), np.asarray(frequency)
    speed = np.divide(2*np.pi*EARTH_RADIUS*f[:, None]/86400., k[None, :],
        out=np.zeros((f.size, k.size)), where=k[None, :] != 0)
    selected = selected & (speed >= definition["speed_min"]) & (speed < definition["speed_max"])
    return selected




def positive_spatial_masks(time_count, longitude_count, spatial_modes, definitions=()):
    """Map the signed spectrum masks to the reversed-longitude rFFT storage."""
    k = np.rint(fft.fftshift(fft.fftfreq(longitude_count)*longitude_count)).astype(int)
    f = fft.rfftfreq(time_count, d=.25)
    base = sector_masks(k, f, time_count=time_count, longitude_count=longitude_count)
    masks = [mask.values for mask in base]
    names = list(SECTORS)
    for definition in definitions:
        masks.append(subset_mask(k, f, definition, time_count=time_count, longitude_count=longitude_count).values)
        names.append(definition["name"])
    # Non-Nyquist spatial modes have two real Fourier partners. At f=0 both
    # signed-k entries describe the same retained coefficient and agree.
    indices = np.arange(time_count)
    signed_index = np.minimum(indices, (-indices) % time_count)
    signed_modes = np.where(indices <= time_count//2, 1, -1)[:, None]*np.asarray(spatial_modes)[None, :]
    if longitude_count % 2 == 0:
        signed_modes[:, np.asarray(spatial_modes) == longitude_count//2] = -longitude_count//2
    columns = np.searchsorted(k, signed_modes)
    result = np.stack([mask[signed_index[:, None], columns] for mask in masks])
    return names, result


def _phase_histogram(product,names,comparison,component):
    mask=product.sector_mask.sel(sector=list(names)).any("sector")
    histogram=signed_phase_speed_histogram(
        product.spectrum.sel(comparison=comparison,component=component).where(mask,0.),
        time_count=product.attrs["time_sample_count"],longitude_count=product.attrs["longitude_sample_count"])
    return histogram.forcing.sel(phase_speed=PHASE_SPEED_PLOT_CENTRES)


def _phase_intervals(axis,x,bounds,*,colour=".15"):
    values=bounds.transpose("bound","phase_speed").values
    axis.vlines(x,values[0],values[1],color=colour,lw=.65,zorder=5)
    axis.plot(np.column_stack((x,x)).T,values,"_",color=colour,ms=2.5,mew=.65,zorder=5)


def _panel_label(axis,letter):
    axis.text(0.,1.02,f"({letter})",transform=axis.transAxes,ha="left",va="bottom",fontsize=7.5,clip_on=False)


def _plot_sector_forcing_summary(product, confidence=None, phase_confidence=None):
    """Show signed flux components and net forcing without changing weights."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from diagnostics.analysis_config import simulation_colour

    groups = (
        ("Low-frequency eastward", ("low_eastward",)),
        ("Low-frequency westward", ("low_westward",)),
        ("High wavenumbers", ("high_wavenumber_eastward", "high_wavenumber_westward")),
        ("Other", ("high_frequency_eastward", "high_frequency_westward", "stationary", "unresolved")),
    )
    field = xr.concat([
        product.sector_forcing.sel(sector=list(names)).sum("sector", skipna=False)
        for _, names in groups
    ]+[product.total_forcing], dim=xr.IndexVariable("row", [label for label, _ in groups]+["Total"]))
    components, hatches = zip(*COMPONENT_HATCHES)
    fig = plt.figure(figsize=(8, 7.2), layout="constrained")
    fig.get_layout_engine().set(w_pad=.035,h_pad=.035,wspace=.025,hspace=.035)
    grid = fig.add_gridspec(3,3,width_ratios=(1.,1.,.045),height_ratios=(1.,1.05,1.))
    first = fig.add_subplot(grid[0, 0])
    axes = (first, fig.add_subplot(grid[0, 1], sharey=first))
    y = .8*np.arange(field.sizes["row"])
    panels = (
        ("Control and U", (("control", -.17), ("nudged-u", .17)), .30),
        ("U − Control", (("u_minus_control", 0.),), .50),
    )
    for letter, axis, (title, comparisons, height) in zip("ab", axes, panels):
        for comparison, offset in comparisons:
            values = field.sel(comparison=comparison)
            colour = simulation_colour("nudged-u" if comparison == "u_minus_control" else comparison)
            positive, negative = np.zeros(y.size), np.zeros(y.size)
            for component, hatch in zip(components, hatches):
                contribution = values.sel(component=component).values
                # Opposite signs start at zero; like signs stack together.
                left = np.where(contribution >= 0, positive, negative)
                axis.barh(y+offset, contribution, left=left, height=height, facecolor=colour,
                    hatch=hatch, edgecolor="white", linewidth=.5, zorder=2)
                positive += np.maximum(contribution, 0.)
                negative += np.minimum(contribution, 0.)
            axis.scatter(values.sel(component="total"), y+offset, color=NET_EPFD_COLOUR, s=23,
                edgecolor="white", linewidth=.4, zorder=6)
            if confidence is not None:
                interval = (confidence.pointwise_interval if comparison == "u_minus_control" else
                    confidence["control_interval" if comparison == "control" else "nudged_u_interval"])
                bounds = interval.sel(component="total").transpose("bound","row").values
                axis.hlines(y+offset,bounds[0],bounds[1],color=NET_EPFD_COLOUR,lw=.9,zorder=4)
                axis.plot(bounds.T,np.column_stack((y+offset,y+offset)),"|",color=NET_EPFD_COLOUR,ms=4,zorder=4)
        axis.axvline(0, color=".35", lw=.8)
        axis.axhline(y[-1]-.4, color=".7", lw=.7)
        _panel_label(axis,letter)
        axis.set_title(title,fontsize=7.5)
        axis.set_xlabel(r"EPFd (m s$^{-1}$ day$^{-1}$)")
        axis.grid(axis="x", alpha=.2)
        axis.set_axisbelow(True)
        axis.margins(x=.1, y=.04)
    row_labels=[str(value).replace("Low-frequency ","Low-frequency\n") for value in field.row.values]
    axes[0].set_yticks(y, labels=row_labels)
    axes[0].get_yticklabels()[-1].set_weight("bold")
    axes[0].invert_yaxis()
    axes[1].tick_params(labelleft=False)
    spectral_axes=np.array([fig.add_subplot(grid[1,column]) for column in range(2)])
    spectral_bar=fig.add_subplot(grid[1,2])
    spectral_handles=plot_epfd_spectrum(product,figure=fig,axes=spectral_axes,colourbar_axis=spectral_bar)
    for column,names in enumerate((LOW_SECTORS,("high_wavenumber_eastward","high_wavenumber_westward"))):
        axis=fig.add_subplot(grid[2,column])
        family=("low_frequency","high_wavenumber")[column]
        if phase_confidence is not None:
            bounds=phase_confidence.pointwise_interval.sel(family=family,component="total",
                phase_speed=PHASE_SPEED_PLOT_CENTRES)
            significant=np.isfinite(bounds).all("bound") & ((bounds.sel(bound="lower")>0.) | (bounds.sel(bound="upper")<0.))
            for left,right,speed,selected in zip(PHASE_SPEED_PLOT_EDGES[:-1],PHASE_SPEED_PLOT_EDGES[1:],
                    PHASE_SPEED_PLOT_CENTRES,significant.values):
                if selected:
                    highlight=axis.axvspan(left,right,facecolor=".9",edgecolor="none",zorder=0)
                    highlight.set_gid(f"phase_significant_{family}_{speed:g}")
        for comparison,offset in (("control",-1.1),("nudged-u",1.1)):
            x=PHASE_SPEED_PLOT_CENTRES+offset
            positive,negative=np.zeros(x.size),np.zeros(x.size)
            for component,hatch in zip(components,hatches):
                values=_phase_histogram(product,names,comparison,component).values
                bars=axis.bar(x,values,bottom=np.where(values>=0.,positive,negative),width=2.,
                    facecolor=simulation_colour(comparison),hatch=hatch,edgecolor="white",linewidth=.5,zorder=3)
                for bar in bars:
                    bar.set_gid(f"phase_component_{family}_{comparison}_{component}")
                positive+=np.maximum(values,0.)
                negative+=np.minimum(values,0.)
            net=_phase_histogram(product,names,comparison,"total")
            points=axis.scatter(x,net,color=NET_EPFD_COLOUR,s=11,edgecolor="white",linewidth=.3,zorder=6)
            points.set_gid(f"phase_net_{family}_{comparison}")
            if phase_confidence is not None:
                interval=phase_confidence["control_interval" if comparison=="control" else "nudged_u_interval"]
                _phase_intervals(axis,x,interval.sel(family=family,component="total",phase_speed=PHASE_SPEED_PLOT_CENTRES),
                    colour=NET_EPFD_COLOUR)
        axis.set_xlim(PHASE_SPEED_PLOT_EDGES[[0,-1]])
        axis.set_xticks(np.arange(-25.,26.,5.))
        axis.axhline(0.,color=".35",lw=.8)
        axis.axvline(0.,color=".6",lw=.6)
        _panel_label(axis,"ef"[column])
        axis.set_title("Low-frequency waves" if column==0 else "High-wavenumber waves",fontsize=7.5)
        axis.set_xlabel(r"Ground-relative phase speed (m s$^{-1}$)")
        axis.set_ylabel("EPFd in 5 m s$^{-1}$ bins\n(m s$^{-1}$ day$^{-1}$)")
        axis.grid(axis="y",alpha=.2)
        axis.set_axisbelow(True)
        axis.margins(y=.12)
    handles=[Patch(facecolor=simulation_colour(name),label=label)
        for name,label in (("control","Control"),("nudged-u","U"))]
    handles += [Patch(facecolor=".4",edgecolor="white",hatch=hatch,label=label)
        for hatch,label in zip(hatches,("Meridional EPFd","Vertical EPFd"))]
    handles.append(Line2D([],[],marker="o",color=NET_EPFD_COLOUR,markerfacecolor=NET_EPFD_COLOUR,
        markeredgecolor="white",markeredgewidth=.3,lw=.65,label="Net EPFd (95% interval)"))
    if phase_confidence is not None:
        handles.append(Patch(facecolor=".9",edgecolor="none",label=r"$\Delta$EPFd: 95% interval excludes 0"))
    fig.legend(handles=handles+spectral_handles,loc="outside lower center",ncol=4,frameon=False,fontsize=6.)
    return fig


def plot_forcing_comparison(product,confidence=None,phase_confidence=None):
    return _plot_sector_forcing_summary(product,confidence=confidence,phase_confidence=phase_confidence)


def plot_phase_speed_anomaly(product,confidence):
    """Retain signed U-minus-Control phase-speed budgets with uncertainty on net EPFd."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from diagnostics.analysis_config import simulation_colour
    figure,axes=plt.subplots(1,2,figsize=(8,3.1),layout="constrained")
    figure.get_layout_engine().set(w_pad=.035,h_pad=.035,wspace=.025,hspace=.025)
    for column,(axis,names) in enumerate(zip(axes,(LOW_SECTORS,("high_wavenumber_eastward","high_wavenumber_westward")))):
        positive,negative=np.zeros(PHASE_SPEED_PLOT_CENTRES.size),np.zeros(PHASE_SPEED_PLOT_CENTRES.size)
        for component,hatch in COMPONENT_HATCHES:
            values=_phase_histogram(product,names,"u_minus_control",component).values
            axis.bar(PHASE_SPEED_PLOT_CENTRES,values,bottom=np.where(values>=0,positive,negative),width=4.4,
                facecolor=simulation_colour("nudged-u"),hatch=hatch,edgecolor="white",linewidth=.5,zorder=2)
            positive+=np.maximum(values,0.)
            negative+=np.minimum(values,0.)
        net=_phase_histogram(product,names,"u_minus_control","total")
        axis.scatter(PHASE_SPEED_PLOT_CENTRES,net,color=NET_EPFD_COLOUR,s=16,edgecolor="white",linewidth=.3,zorder=6)
        family=("low_frequency","high_wavenumber")[column]
        _phase_intervals(axis,PHASE_SPEED_PLOT_CENTRES,confidence.pointwise_interval.sel(
            family=family,component="total",phase_speed=PHASE_SPEED_PLOT_CENTRES))
        axis.axhline(0.,color=".35",lw=.7)
        axis.axvline(0.,color=".6",lw=.6)
        axis.set_xlim(PHASE_SPEED_PLOT_EDGES[[0,-1]])
        axis.set_xticks(np.arange(-25.,26.,5.))
        axis.set_xlabel(r"Ground-relative phase speed (m s$^{-1}$)")
        axis.set_ylabel("$\\Delta$EPFd in 5 m s$^{-1}$ bins\n(m s$^{-1}$ day$^{-1}$)")
        axis.set_title(("Low-frequency waves","High-wavenumber waves")[column],fontsize=7.5)
        _panel_label(axis,"ab"[column])
        axis.grid(axis="y",alpha=.2)
        axis.set_axisbelow(True)
        axis.margins(y=.12)
    handles=[Patch(facecolor=simulation_colour("nudged-u"),edgecolor="white",hatch=hatch,label=label)
        for (_,hatch),label in zip(COMPONENT_HATCHES,("Meridional ΔEPFd","Vertical ΔEPFd"))]
    handles.append(Line2D([],[],marker="o",color=".15",markerfacecolor=NET_EPFD_COLOUR,
        markeredgecolor="white",markeredgewidth=.3,lw=.65,label="Net ΔEPFd (95% interval)"))
    figure.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False,fontsize=6.5)
    return figure


def display_epfd_spectral_density(spectrum):
    """Average neighbouring bins without mixing the signed broad wavenumber bands."""
    spacing = np.diff(spectrum.frequency.values)
    if not spacing.size or not np.allclose(spacing,spacing[0]) or spacing[0] <= 0:
        raise ValueError("The EPFd spectrum needs a uniform positive frequency spacing")
    width = float(spacing[0])
    group = max(1,int(round(.025/width)))
    selected = spectrum.sel(wavenumber=slice(-60,60)).where(
        (spectrum.frequency>0.) & (spectrum.frequency<=1.),drop=True)
    density = (selected/width).coarsen(frequency=group,boundary="pad").mean()
    pieces=[]
    for left,right in ((-60,-21),(-20,-1),(0,0),(1,20),(21,60)):
        part=density.sel(wavenumber=slice(left,right))
        count=part.sizes["wavenumber"]
        if not count:
            continue
        bin_width=np.minimum(2,count-np.arange(0,count,2))
        part=part.coarsen(wavenumber=2,boundary="pad").mean()
        pieces.append(part.assign_coords(wavenumber_width=("wavenumber",bin_width)))
    density=xr.concat(pieces,dim="wavenumber")
    density.attrs.update(frequency_bin_width_cpd=width*group,
        units="m s-1 day-1 (cycles day-1)-1", definition="EPFd contribution per unit frequency and unit zonal wavenumber")
    return density


def equatorial_dispersion_curves(wavenumber, equivalent_depth=50.):
    """Solve the resting equatorial shallow-water dispersion relations of Matsuno."""
    k=np.asarray(wavenumber,dtype=float)
    if equivalent_depth <= 0:
        raise ValueError("Equivalent depth must be positive")
    c=np.sqrt(GRAVITY*equivalent_depth)
    beta=2.*OMEGA/EARTH_RADIUS
    dimensional_k=k/EARTH_RADIUS
    scaled_k=dimensional_k*np.sqrt(c/beta)
    # For n=1, W^3 - (K^2+3) W - K = 0; all three roots are real.
    amplitude=np.sqrt((scaled_k**2+3.)/3.)
    angle=np.arccos(np.clip(scaled_k/(2.*amplitude**3),-1.,1.))/3.
    roots=np.sort(2.*amplitude[None,:]*np.cos(angle[None,:]-2.*np.pi*np.arange(3)[:,None]/3.),axis=0)
    conversion=86400./(2.*np.pi)
    return xr.Dataset(dict(
        kelvin=("wavenumber",np.where(k>0,c*dimensional_k,np.nan)*conversion),
        rossby=("wavenumber",np.where(k<0,roots[1]*np.sqrt(beta*c),np.nan)*conversion),
        mixed_rossby_gravity=("wavenumber",.5*(c*dimensional_k+np.sqrt((c*dimensional_k)**2+4.*beta*c))*conversion),
        inertia_gravity=("wavenumber",roots[2]*np.sqrt(beta*c)*conversion)),
        coords=dict(wavenumber=k),attrs=dict(equivalent_depth_m=equivalent_depth,mean_zonal_wind_ms=0.))


def plot_epfd_spectrum(product,*,figure,axes,colourbar_axis):
    """Embed regional EPFd spectra, broad wave bands, and the displayed phase-speed bin edges."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch, Rectangle
    from matplotlib import patheffects
    from matplotlib.ticker import FuncFormatter

    comparisons=("control","nudged-u")
    density=display_epfd_spectral_density(product.spectrum.sel(component="total",comparison=list(comparisons)))
    maximum=max(float(abs(density).max()),1e-12)
    decade=10.**np.floor(np.log10(maximum))
    limit=np.ceil(maximum/decade)*decade
    colourmap=plt.get_cmap("RdBu_r",11)
    norm=SymLogNorm(linthresh=1e-4,linscale=.5,vmin=-limit,vmax=limit,base=10)
    colours=("#7b3294","#d95f02","#008837")
    # Half-bin edges include k=20, while k=0 remains outside travelling-wave boxes.
    high=float(abs(product.wavenumber).max())+.5
    regions=((-20.5,-.5,FREQUENCY_CUTOFF,colours[0]),(.5,20.5,FREQUENCY_CUTOFF,colours[1]),
        (-high,-20.5,2.,colours[2]),(20.5,high,2.,colours[2]))
    curves=equatorial_dispersion_curves(np.linspace(-60.,60.,1201),equivalent_depth=50.)
    modes=(("kelvin","Kelvin","-"),("rossby","Equatorial Rossby (n=1)","--"),
        ("mixed_rossby_gravity","Mixed Rossby–gravity (n=0)","-."),
        ("inertia_gravity","Inertia–gravity (n=1)",":"))
    for index,(axis,comparison,title) in enumerate(zip(axes,comparisons,("Control","U"))):
        field=density.sel(comparison=comparison)
        k_edges=np.r_[field.wavenumber-field.wavenumber_width/2.,
            float(field.wavenumber[-1]+field.wavenumber_width[-1]/2.)]
        frequency=field.frequency.values
        f_edges=np.r_[frequency-density.attrs["frequency_bin_width_cpd"]/2.,
            frequency[-1]+density.attrs["frequency_bin_width_cpd"]/2.]
        image=axis.pcolormesh(k_edges,f_edges,field.transpose("frequency","wavenumber"),
            cmap=colourmap,norm=norm,shading="auto",rasterized=True)
        image.set_gid("epfd_spectral_density_"+comparison)
        for speed in PHASE_SPEED_PLOT_EDGES:
            if speed==0.:
                continue
            direction=np.sign(speed)
            guide_k=np.array([0.,60.*direction])
            guide_frequency=86400.*guide_k*speed/(2.*np.pi*EARTH_RADIUS)
            guide,=axis.plot(guide_k,guide_frequency,color=".55",ls=":",lw=.45,zorder=3)
            guide.set_gid(f"phase_speed_edge_{speed:g}")
            if abs(speed) in (5.,10.,25.):
                edge_k=min(57.,2.*np.pi*EARTH_RADIUS/(86400.*abs(speed)))
                edge_frequency=86400.*edge_k*abs(speed)/(2.*np.pi*EARTH_RADIUS)
                axis.text(direction*edge_k,min(.97,edge_frequency),f"{speed:g}",
                    ha="right" if speed>0 else "left",va="top",fontsize=5.,color=".35",clip_on=True,
                    path_effects=[patheffects.withStroke(linewidth=1.1,foreground="white")])
        for left,right,top,colour in regions:
            box=Rectangle((left,0.),right-left,top,fill=False,edgecolor=colour,lw=1.,zorder=4,
                path_effects=[patheffects.withStroke(linewidth=1.8,foreground="white")])
            box.set_gid("pahlavan_filter_box")
            axis.add_patch(box)
        for name,label,style in modes:
            line,=axis.plot(curves.wavenumber,curves[name],color=".12",ls=style,lw=.7,zorder=5,
                path_effects=[patheffects.withStroke(linewidth=1.3,foreground="white")])
            line.set_gid("equatorial_dispersion_"+name)
        axis.set_xlim(-60,60)
        axis.set_ylim(0.,1.)
        axis.set_xticks([-60,-40,-20,0,20,40,60])
        axis.set_yticks([0.,.2,.4,.6,.8,1.])
        _panel_label(axis,"cd"[index])
        axis.set_title(title,fontsize=7.5)
        axis.set_xlabel(r"Zonal wavenumber $k$")
    axes[0].set_ylabel("Ground-relative frequency\n(cycles day$^{-1}$)")
    handles=[Patch(facecolor="none",edgecolor=colour,lw=1.,label=label) for colour,label in zip(colours,
        ("Low-frequency westward","Low-frequency eastward","High wavenumbers"))]
    handles.extend(Line2D([],[],color=".12",ls=style,lw=.7,label=label) for name,label,style in modes)
    handles.append(Line2D([],[],color=".55",ls=":",lw=.45,label=r"Phase-speed edges (m s$^{-1}$)"))
    colourbar=figure.colorbar(image,cax=colourbar_axis)
    colourbar.set_ticks([-limit,-1e-2,-1e-4,0.,1e-4,1e-2,limit])
    def spectral_tick(value,position):
        if value==0.:
            return "0"
        if np.isclose(abs(value),limit,rtol=1e-12,atol=0.):
            return f"{value:g}"
        sign="-" if value<0. else ""
        return rf"${sign}10^{{{int(round(np.log10(abs(value))))}}}$"
    colourbar.ax.yaxis.set_major_formatter(FuncFormatter(spectral_tick))
    colourbar.minorticks_off()
    colourbar.set_label("EPFd spectral density\n"+r"[m s$^{-1}$ day$^{-1}$ (cycles day$^{-1}$)$^{-1}$]",fontsize=6.)
    colourbar.ax.tick_params(labelsize=5.5)
    return handles
