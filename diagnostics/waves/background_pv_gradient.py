"""Isentropic dry Ertel PV gradient on the unfiltered mean state's native height grid."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xarray as xr

from diagnostics.analysis_config import simulation_cache_name
from diagnostics.cache_support import cache_manifest, materialize, source_file_signature, write_netcdf_atomic
from diagnostics.diagnostic_helpers import progress
from diagnostics.mean_state.circulation_bootstrap import hemisphere_samples
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.tem_core import EARTH_RADIUS, GRAVITY, OMEGA, TEM_VERSION
from diagnostics.waves.publication_wave_response import variance_cache_path
from diagnostics.waves.regional_epfd_spectrum import control_region_weights, mechanism_latitude_indices

VERSION = "background-isentropic-pv-gradient-v1-native-height"
FOLD_LATITUDES = np.arange(0., 30.01, .25)
SMOOTHING = ((1., 500.), (1.5, 750.))  # Gaussian sigma: latitude degrees, height metres.
MINIMUM_MINUS_THETA_P = 1.e-4  # K Pa-1; exclude neutral and unstable stratification.
GRADIENT_CONVERSION = 1.e12  # SI PV per metre -> PVU per 1000 km.

CAPTION = (
    "Figure 24. Isentropic gradient of dry Ertel PV of the mean state. "
    "Panels show Control, U, and U minus Control. "
    "Unfiltered zonal means of zonal wind, dry potential temperature, and pressure are time averaged on native geometric heights. "
    "The sample and fixed Control hemisphere weights match Figure 22. "
    "All derivatives and differences use native heights; no pressure-level interpolation enters the PV calculation. "
    "For any field A, A_p=A_z/p_z and A_y|p=A_y|z-A_z*p_y|z/p_z. "
    "For each simulation, zeta_bg=-(a*cos(phi))^-1*d(U*cos(phi))/dphi|p, "
    "P_bg=-g*((f+zeta_bg)*theta_p+U_p*theta_y|p), and G_theta=(P_bg)_y|p-(P_bg)_p*theta_y|p/theta_p. "
    "Equivalently, G_theta=(P_bg)_y|z-(P_bg)_z*theta_y|z/theta_z. "
    "Here y=a*phi, f=2*Omega*sin(phi), pressure uses Pa, height uses metres, and phi uses radians. "
    "Calculate on signed latitudes before hemisphere pooling; the northward gradient receives no Southern Hemisphere sign reversal. "
    "This diagnostic differs from the time mean of instantaneous PV. "
    "Gaussian smoothing of all three mean fields uses sigma=1 degree and 500 m in both simulations before differentiation. "
    "Distance and quadrature weights retain the nonuniform native height grid. "
    "A sensitivity calculation uses 1.5 degrees and 750 m. Derivatives retain neighbours outside the displayed domain. "
    "Mask theta_p >= -10^-4 K/Pa; do not clip the denominator. "
    "One PVU is 10^-6 K m^2 kg^-1 s^-1; multiply the SI gradient per metre by 10^12 for PVU per 1000 km. "
    "Mean pressure supplies only the display coordinates: each absolute panel uses its own mean pressure; "
    "the difference at fixed height uses Control mean pressure. "
    "Absolute panels share a colour scale; the difference scale centres on zero. Black solid contours mark zero gradient or anomaly. "
    "Colour scales are logarithmic beyond +/-0.1 for G_theta and +/-0.02 for Delta G_theta (PVU per 1000 km). "
    "Both scales are linear within those bounds. "
    "Black dashed Control and green solid U contours show mean zonal wind in m/s. "
    "Brown contours show U minus Control low-frequency westward meridional EPFd anomalies "
    "at +/-0.1 and +/-0.2 m/s/day; positive contours are solid and negative contours are dashed. "
    "These use the existing native-height wave-band calculation and its cached mean anomaly. "
    "EPFd contours retain the cache's 70–350 hPa coverage. "
    "The purple dashed mechanism box spans 5–15 degrees and 100–170 hPa. "
    "Spatial correspondence tests a changed Rossby-wave propagation environment; it does not establish propagation or causation. "
    "Background: Wirth (2020), https://doi.org/10.5194/wcd-1-111-2020."
)


def _gaussian_matrix(coordinate, sigma):
    """Smooth using physical distances and integration weights on the original nodes."""
    coordinate=np.asarray(coordinate,dtype=float)
    if sigma==0.:
        return np.eye(coordinate.size)
    if sigma<0. or np.any(np.diff(coordinate)<=0.):
        raise ValueError("Smoothing requires ordered physical coordinates and a nonnegative width")
    spacing=np.gradient(coordinate)
    distance=coordinate[None,:]-coordinate[:,None]
    kernel=np.exp(-.5*(distance/sigma)**2)*spacing[None,:]
    kernel[abs(distance)>4.*sigma]=0.
    return kernel/kernel.sum(axis=1,keepdims=True)


def isentropic_pv_gradient(mean_state, *, sigma_latitude=0., sigma_height=0.,
        minimum_minus_theta_p=MINIMUM_MINUS_THETA_P):
    """Transform native-height derivatives to fixed pressure and mean isentropes."""
    state=mean_state.transpose("z","latitude").copy(deep=True)
    height,latitude=state.z.values,state.latitude.values
    for coordinate in (height,latitude):
        if coordinate.size<5 or np.any(np.diff(coordinate)<=0.):
            raise ValueError("PV derivatives require five ordered native heights and signed latitudes")
    kz=_gaussian_matrix(height,sigma_height)
    ky=_gaussian_matrix(latitude,sigma_latitude)
    for name in state.data_vars:
        if not np.isfinite(state[name]).all():
            raise ValueError("Mean fields require finite values throughout the derivative halo")
        state[name].data=kz@state[name].values.astype(float)@ky.T
    angular=state.assign_coords(latitude=np.deg2rad(latitude))
    u,theta,p=angular.zonal_wind,angular.potential_temperature,angular.mean_pressure
    cosine=np.cos(angular.latitude)
    u_z=u.differentiate("z",edge_order=2)
    theta_z=theta.differentiate("z",edge_order=2)
    p_z=p.differentiate("z",edge_order=2)
    if not bool((p_z<0.).all()):
        raise ValueError("Mean pressure must decrease with geometric height")
    p_y=p.differentiate("latitude",edge_order=2)/EARTH_RADIUS
    theta_y_z=theta.differentiate("latitude",edge_order=2)/EARTH_RADIUS
    theta_p=theta_z/p_z
    theta_y_p=theta_y_z-theta_z*p_y/p_z
    zeta_z=-(u*cosine).differentiate("latitude",edge_order=2)/(EARTH_RADIUS*cosine)
    zeta_p=zeta_z+u_z*p_y/p_z
    pv=-GRAVITY*((2.*OMEGA*np.sin(angular.latitude)+zeta_p)*theta_p+u_z/p_z*theta_y_p)
    pv_z=pv.differentiate("z",edge_order=2)
    pv_y_p=pv.differentiate("latitude",edge_order=2)/EARTH_RADIUS-pv_z*p_y/p_z
    valid=np.isfinite(theta_p)&(theta_p < -minimum_minus_theta_p)
    gradient=pv_y_p-pv_z/p_z*theta_y_p/theta_p.where(valid)
    result=xr.Dataset(dict(zonal_wind=u,potential_temperature=theta,mean_pressure=p,
        relative_vorticity=zeta_p,theta_p=theta_p,pv=pv,
        isentropic_pv_gradient=gradient.where(valid)*GRADIENT_CONVERSION,
        valid_stratification=valid)).assign_coords(latitude=latitude).transpose("z","latitude")
    result.pv.attrs.update(units="K m2 kg-1 s-1",definition="dry hydrostatic Ertel PV of the mean state")
    result.isentropic_pv_gradient.attrs.update(units="PVU per 1000 km",
        definition="(P_bg)_y|p - (P_bg)_p * theta_y|p / theta_p",si_conversion=GRADIENT_CONVERSION)
    result.theta_p.attrs.update(units="K Pa-1",valid_condition=f"theta_p < {-minimum_minus_theta_p:g} K Pa-1")
    result.mean_pressure.attrs.update(units="Pa")
    result.z.attrs.update(units="m")
    result.attrs.update(sigma_latitude_degrees=sigma_latitude,sigma_height_metres=sigma_height)
    return result


def fold_background_field(field, fixed_weights, reference_pressure, latitudes=FOLD_LATITUDES):
    """Fold only the completed derivative; use Figure 22's fixed relative hemisphere weights."""
    sides=hemisphere_samples(field,"even").interp(absolute_latitude=latitudes)
    reference=hemisphere_samples(reference_pressure,"even").interp(absolute_latitude=latitudes).mean("hemisphere",skipna=False)/100.
    point=fixed_weights.rename(region_pres="pres",region_latitude="absolute_latitude")
    relative=point/point.sum("hemisphere")
    weights=relative.interp(pres=reference.clip(float(point.pres.min()),float(point.pres.max())),
        absolute_latitude=xr.DataArray(np.clip(latitudes,float(point.absolute_latitude.min()),
            float(point.absolute_latitude.max())),dims="absolute_latitude"))
    weights=weights.assign_coords(absolute_latitude=latitudes).drop_vars("pres")
    folded=(sides*weights).sum("hemisphere",skipna=False)
    if 0. in latitudes:
        folded.loc[dict(absolute_latitude=0.)]=field.interp(latitude=0.).drop_vars("latitude")
    folded.attrs.update(field.attrs)
    return folded


def _load_analysis_times(cache_dir, start, end):
    """Keep Figure 22's sample as the background PV cache reference."""
    times=[]
    for name in ("control","nudged-u"):
        path=variance_cache_path(cache_dir,name)
        with xr.open_dataset(path) as cached:
            configuration=json.loads(cached.attrs["cache_manifest"])["configuration"]
            for key,value in (("start",start),("end",end)):
                if np.datetime64(configuration[key])!=np.datetime64(value):
                    raise ValueError(f"Figure 22 variance uses a different analysis period: {path}")
            times.append(cached.Time.load())
    np.testing.assert_array_equal(times[0].values,times[1].values)
    return times[0]


def load_westward_meridional_epfd_anomaly(cache_dir, start, end, times):
    """Reuse Figure 20's meridional U-minus-Control estimate with its source validation."""
    root=Path(cache_dir)/"epfd"
    sources=[root/f"wave_band_response_{simulation_cache_name(name)}.nc" for name in ("control","nudged-u")]
    for path in sources:
        with xr.open_dataset(path) as cached:
            configuration=json.loads(cached.attrs["cache_manifest"])["configuration"]
            for key,value in (("start",start),("end",end)):
                if np.datetime64(configuration[key])!=np.datetime64(value):
                    raise ValueError(f"Wave-band EPFd uses a different analysis period: {path}")
            np.testing.assert_array_equal(cached.Time.values,times.values)
            if cached.forcing.attrs.get("units","").replace(" ","")!="ms-1day-1":
                raise ValueError(f"Wave-band EPFd requires m/s/day: {path}")
    path=root/"wave_confidence_wave_band_forcing.nc"
    with xr.open_dataset(path) as cached:
        manifest=json.loads(cached.attrs["cache_manifest"])
        if (manifest["configuration"].get("name")!="wave_band_forcing"
                or manifest.get("sources")!=source_file_signature(sources)):
            raise ValueError(f"Rebuild the wave-band EPFd mean and confidence cache: {path}")
        result=cached.estimate.sel(sector="low_westward",component="meridional",drop=True).load()
    result.attrs.update(units="m s-1 day-1",
        definition="U minus Control low-frequency westward meridional EPFd",
        source_cache=str(path))
    return result


def load_or_build_background_pv(*, cache_dir, archive_root, archives, start, end,
        tem_root=None, figures_only=False, rebuild=False):
    """Cache the native-height mean-state diagnostic without changing any wave calculation."""
    times=_load_analysis_times(cache_dir,start,end)
    products,manifests,paths={},{},{}
    for name in ("control","nudged-u"):
        paths[name]=Path(cache_dir)/"epfd"/f"background_pv_gradient_{simulation_cache_name(name)}.nc"
        sources=tem_source_paths(archives[name],root=tem_root)+tem_source_paths(archives["control"],root=tem_root)
        sources+=[variance_cache_path(cache_dir,name)]
        manifests[name]=cache_manifest(__name__,schema=VERSION,configuration=dict(simulation=name,
            start=str(start),end=str(end),signed_latitude_halo_degrees=35.,native_height=True,
            smoothing=SMOOTHING,minimum_minus_theta_p=MINIMUM_MINUS_THETA_P,
            folding="Figure 22 fixed Control hemisphere weights; no gradient sign reversal"),
            dependencies=dict(tem_version=TEM_VERSION),source_paths=sources)
        products[name]=None
        if paths[name].exists() and not rebuild:
            with xr.open_dataset(paths[name]) as cached:
                if cached.attrs.get("manifest_sha256")==manifests[name]["manifest_sha256"]:
                    np.testing.assert_array_equal(cached.Time.values,times.values)
                    products[name]=cached.load()
        if figures_only and products[name] is None:
            raise FileNotFoundError("Build Figure 24 without --figures-only to calculate background PV gradients")
    if any(value is None for value in products.values()):
        control=load_stored_tem(archives["control"],root=tem_root,start=start,end=end)
        try:
            fixed=control_region_weights(materialize(control[["rho_zm","pressure"]].isel(
                latitude=mechanism_latitude_indices(control.latitude))))
            reference_pressure=materialize(control.pressure.sel(latitude=slice(-35.,35.)).mean("Time",skipna=False))
        finally:
            control.close()
        for name in products:
            if products[name] is not None:
                continue
            progress(f"{name}: calculate unfiltered native-height mean state for background PV")
            upstream=load_stored_tem(archives[name],root=tem_root,start=start,end=end)
            try:
                np.testing.assert_array_equal(upstream.Time.values,times.values)
                mean=materialize(upstream[["u_bar","theta_bar","pressure"]].sel(
                    latitude=slice(-35.,35.)).mean("Time",skipna=False)).rename(
                        u_bar="zonal_wind",theta_bar="potential_temperature",pressure="mean_pressure")
            finally:
                upstream.close()
            cases=[]
            for index,(sigma_lat,sigma_z) in enumerate(SMOOTHING):
                state=isentropic_pv_gradient(mean,sigma_latitude=sigma_lat,sigma_height=sigma_z)
                for source,target in (("isentropic_pv_gradient","folded_gradient"),("zonal_wind","folded_zonal_wind"),
                        ("mean_pressure","folded_pressure")):
                    state[target]=fold_background_field(state[source],fixed,reference_pressure)
                cases.append(state.expand_dims(smoothing=["standard" if index==0 else "sensitivity"]))
            result=xr.concat(cases,dim="smoothing")
            for source in mean:
                result["unsmoothed_"+source]=mean[source]
            result=result.assign_coords(Time=times)
            result.attrs.update(product_version=VERSION,manifest_sha256=manifests[name]["manifest_sha256"],
                cache_manifest=json.dumps(manifests[name],sort_keys=True),
                definition="Isentropic gradient of dry Ertel PV of the mean state",
                sample_count=times.size,calculation_coordinate="native geometric height; no pressure-level remapping",
                stratification_threshold_K_Pa=MINIMUM_MINUS_THETA_P)
            write_netcdf_atomic(result,paths[name])
            products[name]=result
            progress(f"{name}: cached native-height PV gradient and smoothing sensitivity")
    xr.align(products["control"].folded_gradient,products["nudged-u"].folded_gradient,join="exact")
    return products,load_westward_meridional_epfd_anomaly(cache_dir,start,end,times)


def plot_background_pv(products, meridional_epfd_anomaly):
    """Use mean pressure only to display the native-height scalar fields."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import SymLogNorm
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import MaxNLocator
    from diagnostics.mean_state.circulation_mechanism import REGION_BOX_STYLE
    from diagnostics.waves.wave_response_plotting import _pressure_axis, plot_region_box, U_ZONAL_WIND_COLOUR

    states=[products[name].sel(smoothing="standard",drop=True,absolute_latitude=slice(0.,25.))
        for name in ("control","nudged-u")]
    fields=[state.folded_gradient for state in states]
    fields.append(fields[1]-fields[0])
    pressures=[state.folded_pressure/100. for state in states]
    pressures.append(pressures[0])
    local=[field.where((pressure>=70.)&(pressure<=400.)) for field,pressure in zip(fields,pressures)]
    def levels_for(values,linear_width):
        maximum=max(float(np.nanmax(abs(value))) for value in values)
        maximum=max(maximum,1.e-6)
        ticks=MaxNLocator(nbins=4,symmetric=True).tick_values(-maximum,maximum)
        limit=float(max(abs(ticks)))
        positive=np.unique(np.r_[linear_width*np.array([1.,5.,20.]),limit])
        positive=positive[positive<=limit]
        levels=np.r_[-positive[::-1],0.,positive]
        return levels,levels
    absolute_levels,absolute_ticks=levels_for(local[:2],.1)
    difference_levels,difference_ticks=levels_for(local[2:],.02)
    figure=plt.figure(figsize=(8.,3.25),layout="constrained")
    figure.get_layout_engine().set(w_pad=.025,h_pad=.025,wspace=.025,hspace=.025)
    grid=figure.add_gridspec(1,5,width_ratios=(1.,1.,.045,1.,.045))
    axes=[figure.add_subplot(grid[0,column]) for column in (0,1,3)]
    bar_axes=[figure.add_subplot(grid[0,column]) for column in (2,4)]
    images=[]
    wind_levels=[-10.,10.,20.,30.,40.]
    epfd_colour="#8c510a"
    for column,(axis,field,pressure,title) in enumerate(zip(axes,fields,pressures,("Control","U","U − Control"))):
        levels=absolute_levels if column<2 else difference_levels
        latitude=np.broadcast_to(field.absolute_latitude,field.shape)
        image=axis.contourf(latitude,pressure,field,levels=levels,cmap="RdBu_r",
            norm=SymLogNorm(linthresh=.1 if column<2 else .02,linscale=.5,
                vmin=levels[0],vmax=levels[-1]),extend="both")
        images.append(image)
        if float(local[column].min())<0.<float(local[column].max()):
            axis.contour(latitude,pressure,field,levels=[0.],colors="black",linewidths=.9)
        if column<2:
            wind=states[column].folded_zonal_wind
            contours=axis.contour(latitude,pressure,wind,levels=wind_levels,
                colors="black" if column==0 else U_ZONAL_WIND_COLOUR,
                linestyles="--" if column==0 else "-",linewidths=.7)
            axis.clabel(contours,fmt="%g",fontsize=5.5,inline_spacing=2)
        else:
            epfd=meridional_epfd_anomaly.sel(pres=slice(70.,400.),absolute_latitude=slice(0.,25.))
            levels=[-.2,-.1,.1,.2]
            contours=axis.contour(epfd.absolute_latitude,epfd.pres,epfd.transpose("pres","absolute_latitude"),
                levels=levels,colors=epfd_colour,linewidths=.9,
                linestyles=["--" if level<0. else "-" for level in levels])
            axis.clabel(contours,fmt=lambda value:f"{value:+g}",fontsize=5.5,inline_spacing=2)
        plot_region_box(axis)
        _pressure_axis(axis,ylabel=column==0,xlabel=True,latitude_limit=25,bottom_pressure=400)
        if column:
            axis.tick_params(labelleft=False)
        axis.set_title(title,fontsize=8.)
        axis.text(0.,1.02,f"({'abc'[column]})",transform=axis.transAxes,ha="left",va="bottom",fontsize=8.)
    for image,bar_axis,ticks,label in ((images[0],bar_axes[0],absolute_ticks,r"$G_\theta$"),
            (images[2],bar_axes[1],difference_ticks,r"$\Delta G_\theta$")):
        bar=figure.colorbar(image,cax=bar_axis,ticks=ticks)
        bar.set_label(label+"\n(PVU per 1000 km)",fontsize=6.5)
        bar.ax.tick_params(labelsize=6.)
    handles=[Line2D([],[],color="black",ls="--",lw=.7,label=r"Control zonal wind (m s$^{-1}$)"),
        Line2D([],[],color=U_ZONAL_WIND_COLOUR,lw=.7,label=r"U zonal wind (m s$^{-1}$)"),
        Line2D([],[],color="black",lw=.9,label="Zero gradient or anomaly"),
        Line2D([],[],color=epfd_colour,lw=.9,
            label=r"Filtered low-frequency westward meridional $\Delta$EPFd (m s$^{-1}$ day$^{-1}$)"),
        Patch(fill=False,**REGION_BOX_STYLE,label="Mechanism box")]
    figure.legend(handles=handles,loc="outside lower center",ncol=3,frameon=False,fontsize=6.)
    return figure
