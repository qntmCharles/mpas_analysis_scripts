"""Retained wave-forcing figures and their shared sampling products."""
from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import xarray as xr
from matplotlib import rc_context

from diagnostics.analysis_config import simulation_cache_name
from diagnostics.cache_support import cache_manifest, materialize, write_netcdf_atomic
from diagnostics.diagnostic_helpers import progress
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.tem_core import FLUX_TERMS, TEM_VERSION
from diagnostics.mean_state import circulation_mechanism as mechanism
from diagnostics.waves.regional_epfd_spectrum import (
    control_region_weights, mechanism_latitude_indices, regional_native_weights, load_or_build_regional_spectrum)
from diagnostics.waves.regional_ep_flux_sectors import build_sector_analysis, SECTORS, plot_forcing_comparison, plot_phase_speed_anomaly
from diagnostics.waves.sector_transport import native_sector_fields, load_or_build_covariances
from diagnostics.waves.publication_wave_forcing import (
    BACKGROUND_FIELDS, publication_windows, build_meridional_sections,
    load_or_build_publication, plot_publication_mechanisms)
from diagnostics.waves.sector_wind_variance import load_or_build_wind_variance, plot_wind_variance
from diagnostics.waves.wave_band_response import load_or_build_band_response
from diagnostics.waves.wave_energy_flux import load_or_build_energy_flux, load_energy_flux_height_profile
from diagnostics.waves.publication_wave_response import load_or_build_publication_response, forcing_cache_path, variance_cache_path
from diagnostics.waves.wave_response_plotting import (
    hatch_insignificant, plot_band_response, plot_wave_energy_flux, plot_wave_publication_response, equatorial_display_field, plot_region_box,
    load_circulation_guides, plot_circulation_guides, circulation_guide_handles)
from diagnostics.waves.wave_uncertainty import (
    PressureSampler, native_height_slice, mean_response_interval, absolute_mean_interval,
    BLOCK_DAYS, REPLICATES, SEED)
from diagnostics.waves.phase_speed_uncertainty import load_or_build_phase_speed_samples
from diagnostics.waves.background_pv_gradient import (
    CAPTION as BACKGROUND_PV_CAPTION, load_or_build_background_pv, plot_background_pv)

SAMPLE_VERSION = "wave-forcing-samples-v1-pooled-time"
PRESSURES = np.unique(np.r_[np.arange(100., 351., 10.), mechanism.REGION_PRESSURES])
LATITUDES = np.arange(0., 40.1, 1.)
GROUPS = (
    ("Low-frequency eastward", ("low_eastward",)),
    ("Low-frequency westward", ("low_westward",)),
    ("High wavenumbers", ("high_wavenumber_eastward", "high_wavenumber_westward")),
    ("Other", ("high_frequency_eastward", "high_frequency_westward", "stationary", "unresolved")),
    ("Total", SECTORS),
)
FIGURES = {16: "tem_flux_decomposition", 17: "regional_wave_forcing", 18: "wave_transport_and_forcing",
    19: "horizontal_wind_variance", 20: "wave_band_ep_flux_and_forcing", 21: "wave_energy_flux_and_variance",
    22: "wave_forcing_variance_and_energy_flux", 23: "phase_speed_forcing_anomaly",
    24: "background_isentropic_pv_gradient"}
UNCERTAINTY_TEXT = ("Hatching marks finite changes whose pointwise 95% intervals include zero. "
    "Intervals use 999 independent-run circular moving-block bootstrap draws with 40-day blocks. "
    "Each sample pools simultaneous hemispheres before resampling.")
ENERGY_VECTOR_TEXT = (
    "Arrows show filtered wave-energy flux (overline(p'_b v'_b), overline(p'_b w'_b)), in W m^-2. "
    "Both pressure and velocity receive band b's filter at native geometric height before multiplication. "
    "Overlines denote zonal and time means; v is positive poleward after hemisphere pooling. "
    "One positive factor C converts both physical components to geometric displacements. "
    "For latitude phi in radians, delta_phi=C*overline(p'_b v'_b)/a, where a is Earth's radius. "
    "The vertical endpoint satisfies z_ref(p_end)=z_ref(p)+C*overline(p'_b w'_b). "
    "The inverse Control time and area mean pressure profile over 25S–25N defines z_ref(p). "
    "Interpolation is linear in ln(p); the logarithmic pressure axes supply the corresponding vertical exaggeration. "
    "Vertical vector length varies with the reference profile's local pressure–height gradient. "
    "The vertical reference arrow uses 170 hPa. "
    "No spherical transport weighting, percentile normalization, or separate band multiplier applies. ")
CAPTIONS = {
16: "Figure 16. U minus Control EPFd, decomposed by flux and direction. "
    "Rows show meridional and vertical divergence. Columns show total, momentum-flux, and heat-flux contributions. "
    "All derivatives use native geometric height and instantaneous TEM coefficients before pressure sampling. "
    "Contours show Control zonal wind at 5, 10, 20, and 30 m/s. Positive forcing accelerates eastward flow. ",
17: "Figure 17. Regional EPFd in 5–15 degrees absolute latitude and 100–170 hPa. "
    "Top panels show signed meridional and vertical contributions, with net EPFd as red circles. "
    "Error bars give pointwise 95% intervals for net forcing with 40-day blocks. "
    "Middle panels show Control and U wavenumber–frequency EPFd spectral density with Pahlavan filter boxes. "
    "Grey rays mark phase-speed bin edges from -25 to +25 m/s in 5 m/s steps: f=86400*k*c/(2*pi*a). "
    "Example equatorial-wave curves use equivalent depth 50 m and zero mean zonal wind. "
    "Lower panels show paired Control/U bars in 5 m/s phase-speed bins over -25 to +25 m/s. "
    "Like signs stack together; opposite signs start separately from zero. "
    "Red circles show each run's net EPFd; error bars give pointwise 95% intervals. "
    "Grey columns identify bins where the pointwise 95% interval for net U minus Control EPFd excludes zero. "
    "Intervals use 999 circular moving-block bootstrap draws with 40-day blocks, after simultaneous hemisphere pooling. "
    "The phase-speed intervals condition on the complete-record filters; spectral shading shows descriptive means. "
    "Opposite diagonal hatch directions identify meridional and vertical EPFd. Broad-sector budgets retain the complete spectrum. "
    "Low-frequency waves use 1 <= |k| <= 20 and 0 < f <= 0.4 cycles/day. "
    "High wavenumbers use |k| > 20; Other includes higher-frequency, stationary, and unresolved contributions. ",
18: "Figure 18. U minus Control EPFd and signed TEM momentum transport. "
    "Upper panels retain descriptive time-mean phase-speed spectra and vertical transport profiles. "
    "Upper transport profiles use mPa; lower integrated transport profiles use N/m. "
    "Lower maps show meridional forcing in the 5 to 10 and -10 to -5 m/s intervals. "
    "Shaded profile bands give pointwise 95% intervals for each run with 40-day blocks. "
    "These intervals condition on phase-speed windows selected from the mean response. "
    "Transport includes the TEM heat-flux correction. Purple dashed lines mark the regional integration boundaries. ",
19: "Figure 19. Horizontal-wind variance, defined as the zonal mean of filtered u-prime squared plus v-prime squared. "
    "Columns show low-frequency eastward, low-frequency westward, and high-wavenumber waves. "
    "Upper panels show Control; lower panels show U minus Control. "
    "Wind contours show Control and U with signs appropriate to each wave band. "
    "The display covers 0–45 degrees and 100–350 hPa. "
    "All sectors retain absolute variance, without density weighting or a TEM heat-flux correction. ",
20: "Figure 20. U minus Control filtered EPFd by wave band. "
    "Rows show total, meridional, and vertical forcing changes. "
    "Columns show low-frequency eastward, low-frequency westward, and high-wavenumber waves. "
    "Positive forcing accelerates eastward flow. "
    "Low-frequency bands use 1 <= |k| <= 20 and 0 < f <= 0.4 cycles/day. "
    "The high-wavenumber band uses |k| > 20 and combines both directions. "
    "Top-row arrows show absolute U EP flux, using the Jucker (2021) circulation transformation within 0–20°. "
    "Variance contours show changes in the zonal and time mean of filtered u′² + v′² (m²/s²). "
    "Purple solid contours show positive variance changes; green dashed contours show negative changes. "
    "Black dashed Control and green solid U zonal wind contours use signs appropriate to each wave band. "
    "Purple dashed boxes mark 5–15° absolute latitude and 100–170 hPa. "
    "Shading extends the nearest scalar sample to the equator; no equatorial significance is inferred. ",
21: "Figure 21. Filtered horizontal variance and wave-energy flux. "
    "The upper row shows Control; the lower row shows U minus Control. "
    "Columns show low-frequency eastward, low-frequency westward, and high-wavenumber waves. "
    "Filtered horizontal variance is the zonal and time mean of u′² + v′²; variance colour scales differ between bands. "
    +ENERGY_VECTOR_TEXT+
    "C=2000 m/(W m^-2), shared by both rows and all wave bands. "
    "Orange and purple outlines enclose significant positive and negative U-minus-Control total EPFd, respectively. "
    "Outlines use lightly smoothed boundaries of the same significance mask as Figure 20's total-forcing row. "
    "The 95% intervals use 999 independent-run circular moving-block bootstrap draws with 40-day blocks. "
    "Each sample pools simultaneous hemispheres before resampling. "
    "Variance and energy-flux fields show descriptive means. "
    "Black dashed Control and green solid U contours show zonal wind. "
    "Purple dashed boxes mark 5–15° absolute latitude and 100–170 hPa. "
    "Shading extends the nearest scalar sample to the equator; no equatorial significance is inferred. ",
22: "Figure 22. U minus Control filtered EPFd (upper row) and filtered horizontal variance (lower row). "
    "Columns show low-frequency eastward, low-frequency westward, and high-wavenumber waves. "
    "Low-frequency bands use 1 <= |k| <= 20 and 0 < f <= 0.4 cycles/day. "
    "The high-wavenumber band uses |k| > 20 and combines both directions. "
    "Positive EPFd accelerates eastward zonal flow. "
    "Filtered horizontal variance is the zonal and time mean of band-filtered u′² + v′². "
    "The shared variance colourbar shows 100*Delta(overline(u′²+v′²))/max_Control(overline(u′²+v′²)), from -10% to +10%. "
    "Each band uses its maximum Control mean variance within 0–25° and 70–400 hPa. "
    "Small labels above the lower panels give the variance contour interval in physical units. "
    "Variance contour levels are zero and ±1, ±2, ±3, ±4, and ±5 times the labelled interval. "
    "Filtered EPFd shares a -0.25 to +0.25 m s^-1 day^-1 colour scale, with 0.05 m s^-1 day^-1 intervals. "
    "EPFd uses red–blue shading; brown shows variance decreases and purple shows variance increases. "
    "Lower-row arrows show U-minus-Control wave-energy flux. "
    +ENERGY_VECTOR_TEXT+
    "C=12000 m/(W m^-2), shared by all wave bands. "
    "Figure 22 uses six times Figure 21's arrow magnification. "
    "Hatching marks filtered EPFd and variance anomalies whose pointwise 95% intervals include zero. "
    "Energy-flux arrows show means without confidence intervals. "
    "Black dashed Control and green solid U contours show zonal wind. "
    "The purple dashed mechanism box spans 5–15° absolute latitude and 100–170 hPa. "
    "Equatorial scalars interpolate between signed native latitudes at each instantaneous pressure surface before time averaging. "
    "Equatorial filtered EPFd and variance use the same bootstrap as other locations. ",
23: "Figure 23. U minus Control regional EPFd in 5 m/s ground-relative phase-speed bins. "
    "Panels show low-frequency waves (1 <= |k| <= 20, 0 < f <= 0.4 cycles/day) and high-wavenumber waves (|k| > 20). "
    "The region spans 5–15° absolute latitude and 100–170 hPa, with fixed Control mass weights and simultaneous hemisphere pooling. "
    "Opposite diagonal hatch directions identify meridional and vertical EPFd changes; opposite signs start separately from zero. "
    "Red dots show net changes, with pointwise 95% intervals from 999 independent-run circular moving-block draws with 40-day blocks. "
    "Intervals condition on the complete-record filters and use the same time draw across components and bins. "
    "Displayed bins span -25 to +25 m/s; Figure 17's broad-sector budgets retain the complete phase-speed range. ",
}
CAPTIONS[17]+=("Spectral panels retain |k| <= 60 and 0 < f <= 1 cycle/day. "
    "Display averaging uses approximately 0.025 cycles/day and neighbouring wavenumbers within each broad band. "
    "The shared symmetric logarithmic scale is linear below 10^-4 m s^-1 day^-1 (cycles day^-1)^-1. "
    "Positive spectral density accelerates eastward flow. Positive k denotes eastward phase propagation. "
    "Wave curves show Kelvin, n=1 equatorial Rossby, n=0 mixed Rossby–gravity, and n=1 inertia–gravity branches. "
    "References: Pahlavan et al. (2021), https://doi.org/10.1175/JAS-D-20-0249.1; "
    "Matsuno (1966), https://doi.org/10.2151/jmsj1965.44.1_25. ")
CAPTIONS[24]=BACKGROUND_PV_CAPTION
for _number in (16,18,19):
    CAPTIONS[_number]+=("Latitude–pressure panels reuse the circulation publication guides and cached mean fields. "
        "Teal contours locate the 40 m/s subtropical jet in the displayed simulation, or U for difference maps. "
        "Green lines show the tropopause: solid U and dashed Control. "
        "Absolute panels show their own tropopause; difference panels show both. ")


def _cached(path, signature):
    if Path(path).exists():
        with xr.open_dataset(path) as product:
            if product.attrs.get("manifest_sha256") == signature:
                return product.load()
    return None


def load_or_build_samples(*, simulation, archive_dir, control_archive_dir, cache_dir, analysis, covariance_path,
        start, end, tem_root=None, figures_only=False, rebuild=False, output_path=None):
    """Retain instantaneous samples for the existing regional and transport definitions."""
    path = Path(output_path or Path(cache_dir)/"epfd"/f"wave_forcing_samples_{simulation_cache_name(simulation)}.nc")
    sources = [Path(covariance_path).resolve()]+tem_source_paths(Path(archive_dir).name, root=tem_root)
    sources += tem_source_paths(Path(control_archive_dir).name, root=tem_root)
    windows = publication_windows(analysis)
    manifest = cache_manifest(__name__, schema=SAMPLE_VERSION,
        configuration=dict(simulation=simulation,start=str(start),end=str(end),windows=windows,
            pressures=PRESSURES,latitudes=LATITUDES,hemispheres="simultaneous fixed-Control mass weighting"),
        dependencies=dict(tem_version=TEM_VERSION), source_paths=sources)
    if not rebuild:
        cached = _cached(path,manifest["manifest_sha256"])
        if cached is not None:
            return cached
    if figures_only:
        raise FileNotFoundError(f"Missing uncertainty samples: {path}; calculate without --figures-only")
    upstream = load_stored_tem(Path(archive_dir).name,root=tem_root,start=start,end=end)
    control = load_stored_tem(Path(control_archive_dir).name,root=tem_root,start=start,end=end)
    try:
        fixed = control_region_weights(materialize(control[["rho_zm","pressure"]].isel(
            latitude=mechanism_latitude_indices(control.latitude))))
        with xr.open_dataset(covariance_path) as covariances:
            progress(f"{simulation}: retaining simultaneous hemisphere samples for confidence intervals")
            background = materialize(upstream[list(BACKGROUND_FIELDS)].sel(z=covariances.z,latitude=covariances.latitude))
            sections = build_meridional_sections(covariances,background,analysis,windows,fixed,simulation,retain_samples=True)
            product = sections[["meridional_forcing_samples","integrated_meridional_tem_transport_samples"]].rename(
                meridional_forcing_samples="section_forcing",integrated_meridional_tem_transport_samples="section_transport",
                pres="section_pres",absolute_latitude="section_latitude")
            rows = native_height_slice(background.pressure)
            local = background.isel(z=rows)
            sampler = PressureSampler(local,PRESSURES,LATITUDES,fixed)
            native_terms = materialize(upstream[list(FLUX_TERMS)].sel(z=local.z,latitude=local.latitude))
            product["decomposition"] = xr.concat([sampler.samples(native_terms[name])*86400. for name in FLUX_TERMS],
                dim=xr.IndexVariable("flux_term",list(FLUX_TERMS)))
            product["zonal_wind"] = sampler.samples(local.u_bar).mean("Time")
            regional_background = materialize(upstream[list(BACKGROUND_FIELDS)].sel(
                z=covariances.region_z.values,latitude=covariances.region_latitude.values))
            q, _ = regional_native_weights(regional_background,fixed)
            sector_samples = []
            for name in SECTORS:
                covariance = materialize(covariances.regional_covariance.sel(sector=name).transpose(
                    "Time","region_z","region_latitude","covariance"))
                fields = native_sector_fields(covariance.values,regional_background)
                sector_samples.append(xr.concat([(xr.where(q!=0,fields[field],0.)*q).sum(("z","latitude"))*86400.
                    for field in ("EPFd","EPFd_y","EPFd_z")],dim=product.component))
            sector = xr.concat(sector_samples,dim=xr.IndexVariable("sector",list(SECTORS)))
            product["regional_forcing"] = xr.concat([sector.sel(sector=list(names)).sum("sector")
                for _,names in GROUPS],dim=xr.IndexVariable("row",[label for label,_ in GROUPS]))
            for label,names in GROUPS:
                expected = analysis.sector_forcing.sel(comparison=simulation,sector=list(names)).sum("sector")
                np.testing.assert_allclose(product.regional_forcing.sel(row=label).mean("Time"),expected,atol=2e-7,rtol=2e-6)
        product.attrs.update(product_version=SAMPLE_VERSION,manifest_sha256=manifest["manifest_sha256"],
            cache_manifest=json.dumps(manifest,sort_keys=True),windows=json.dumps(windows),
            hemisphere_sampling="simultaneous pooled observations; no independent hemisphere replicates")
        encoding = {name:dict(zlib=True,complevel=2) for name in product.data_vars}
        write_netcdf_atomic(product,path,encoding=encoding)
        return product
    finally:
        upstream.close()
        control.close()


def confidence_for(name, control, experiment, *, cache_dir, sources, absolute=False, figures_only=False):
    """Cache bootstrap products separately from expensive native-data calculations."""
    path = Path(cache_dir)/"epfd"/f"wave_confidence_{name}.nc"
    manifest = cache_manifest(__name__,schema="wave-confidence-v2-pointwise-one-block",
        configuration=dict(name=name,block_days=BLOCK_DAYS,replicates=REPLICATES,seed=SEED,absolute=absolute),
        dependencies={},source_paths=sources)
    result = _cached(path,manifest["manifest_sha256"])
    if result is not None:
        return result
    if figures_only:
        raise FileNotFoundError(f"Missing confidence cache: {path}; calculate without --figures-only")
    progress(f"Confidence intervals: {name}; {REPLICATES} draws, one {BLOCK_DAYS:g}-day block length")
    result = mean_response_interval(control,experiment)
    if absolute:
        for label,field in (("control",control),("nudged_u",experiment)):
            result[label+"_interval"] = absolute_mean_interval(field).pointwise_interval
    result.attrs.update(manifest_sha256=manifest["manifest_sha256"],cache_manifest=json.dumps(manifest,sort_keys=True))
    write_netcdf_atomic(result,path)
    return result



def _map(axis, field, mask, wind, *, title, limit=None):
    from matplotlib.colors import BoundaryNorm
    from matplotlib.ticker import NullFormatter
    from diagnostics.diagnostic_helpers import difference_cmap,nice_levels
    estimate = field
    field = equatorial_display_field(field).transpose("pres","absolute_latitude")
    maximum = float(abs(field).quantile(.98)) if limit is None else limit
    levels = nice_levels(-maximum,maximum,n=21)
    cmap = difference_cmap(levels,cmap="RdBu_r",white_zero_boundary=True)
    image = axis.contourf(field.absolute_latitude,field.pres,field,levels=levels,
        cmap=cmap,norm=BoundaryNorm(levels,cmap.N),extend="both")
    contours=axis.contour(wind.absolute_latitude,wind.pres,wind.transpose("pres","absolute_latitude"),
        levels=[5,10,20,30],colors=".3",linewidths=.6)
    axis.clabel(contours,fmt="%g",fontsize=5.5)
    hatch_insignificant(axis,mask,estimate,contrast_shading=(field,image))
    plot_region_box(axis)
    axis.set(xlim=(0,40),ylim=(350,100),yscale="log",title=title,xlabel=r"Latitude ($^\circ$)",ylabel="Pressure (hPa)")
    axis.set_yticks([350,300,250,200,170,150,100],labels=["350","300","250","200","170","150","100"])
    axis.yaxis.set_minor_formatter(NullFormatter())
    axis.title.set_fontsize(7.5)
    return image


def decomposition_fields(field):
    rows=[]
    for direction in ("meridional","vertical"):
        momentum=field.sel(flux_term="momentum_"+direction,drop=True)
        heat=field.sel(flux_term="heat_"+direction,drop=True)
        rows.extend((momentum+heat,momentum,heat))
    return xr.concat(rows,dim=xr.IndexVariable("term",["meridional_total","meridional_momentum","meridional_heat",
        "vertical_total","vertical_momentum","vertical_heat"]))


def plot_decomposition(samples,confidence,circulation_guides=None):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,3,figsize=(8,5.5),sharex=True,sharey=True,layout="constrained")
    for index,axis in enumerate(axes.flat):
        term=confidence.term.values[index]
        image=_map(axis,confidence.estimate.sel(term=term),confidence.significant.sel(term=term),samples.zonal_wind,
            title=f"({'abcdef'[index]}) "+term.replace("_"," ").capitalize())
        plot_circulation_guides(axis,circulation_guides)
        fig.colorbar(image,ax=axis,location="bottom",label="EPFd change (m/s/day)",shrink=.95,pad=.1)
        if index%3:
            axis.set_ylabel("")
        if index<3:
            axis.set_xlabel("")
    if circulation_guides is not None:
        fig.legend(handles=circulation_guide_handles(),loc="outside lower center",ncol=3,frameon=False)
    fig.suptitle("U − Control: momentum-flux and heat-flux contributions to EPFd\n"
        "Hatching: pointwise 95% intervals include zero; 40-day blocks",fontsize=8.5)
    return fig



@rc_context({"font.size":7,"axes.labelsize":7,"axes.titlesize":8,"xtick.labelsize":6.5,
    "ytick.labelsize":6.5,"legend.fontsize":6.5,"figure.titlesize":9})
def run_wave_forcing_figures(*,cache_dir,output_dir,archive_root,archives,pattern,start,end,selected,
        save_figure,should_build,figures_only=False,rebuild=False,tem_root=None):
    """Use the existing figure dispatcher, cache family, and numbered figure saver."""
    wanted=[number for number,stem in FIGURES.items() if (selected is None or number in selected) and should_build(number,stem)]
    if not wanted:
        return
    def background_pv_figure():
        products,meridional_epfd=load_or_build_background_pv(cache_dir=cache_dir,archive_root=archive_root,
            archives=archives,start=start,end=end,tem_root=tem_root,figures_only=figures_only,rebuild=rebuild)
        return plot_background_pv(products,meridional_epfd)
    if 24 in wanted and 22 not in wanted:
        import matplotlib.pyplot as plt
        figure=background_pv_figure()
        save_figure(24,figure,FIGURES[24])
        (Path(output_dir)/f"24_{FIGURES[24]}.caption.txt").write_text(CAPTIONS[24]+"\n")
        plt.close(figure)
        wanted.remove(24)
        if not wanted:
            return
    guides=load_circulation_guides(start=start,end=end) if set(wanted)&{16,18,19,20,21,22} else None
    runs=("control","nudged-u")
    spectra={name:load_or_build_regional_spectrum(simulation=name,archive_dir=Path(archive_root)/archives[name],
        control_archive_dir=Path(archive_root)/archives["control"],cache_dir=cache_dir,pattern=pattern,start=start,end=end,
        tem_root=tem_root,figures_only=figures_only,rebuild=rebuild) for name in runs}
    analysis=build_sector_analysis(spectra)
    windows=publication_windows(analysis) if set(wanted)-{23} else None
    samples={}
    covariance_paths={}
    for name in (runs if set(wanted)-{23} else ()):
        arguments=dict(simulation=name,archive_dir=Path(archive_root)/archives[name],
            control_archive_dir=Path(archive_root)/archives["control"],cache_dir=cache_dir,start=start,end=end,tem_root=tem_root,
            figures_only=figures_only,rebuild=rebuild)
        covariance=load_or_build_covariances(**arguments,pattern=pattern,definitions=windows,analysis=analysis)
        covariance_paths[name]=covariance
        if set(wanted)&{16,17,18}:
            samples[name]=load_or_build_samples(**arguments,analysis=analysis,covariance_path=covariance)
    sources=[Path(cache_dir)/"epfd"/f"wave_forcing_samples_{simulation_cache_name(name)}.nc" for name in runs]
    confidence={}
    phase_confidence=None
    if set(wanted)&{17,23}:
        phase_products={name:load_or_build_phase_speed_samples(simulation=name,
            archive_dir=Path(archive_root)/archives[name],control_archive_dir=Path(archive_root)/archives["control"],
            cache_dir=cache_dir,pattern=pattern,start=start,end=end,analysis=analysis,
            tem_root=tem_root,figures_only=figures_only,rebuild=rebuild) for name in runs}
        phase_sources=[Path(cache_dir)/"epfd"/f"phase_speed_forcing_samples_{simulation_cache_name(name)}.nc" for name in runs]
        phase_confidence=confidence_for("phase_speed_forcing",*[phase_products[name].forcing for name in runs],
            cache_dir=cache_dir,sources=phase_sources,absolute=True,figures_only=figures_only)
    fields={16:("decomposition",True),17:("regional_forcing",False),18:("section_forcing",False)}
    for number,(field,decompose) in fields.items():
        if number in wanted:
            c,u=[decomposition_fields(samples[name][field]) if decompose else samples[name][field] for name in runs]
            confidence[number]=confidence_for(field,c,u,cache_dir=cache_dir,sources=sources,
                absolute=number==17,figures_only=figures_only)
    figures={}
    if 23 in wanted:
        figures[23]=plot_phase_speed_anomaly(analysis,phase_confidence)
    if 16 in wanted:
        figures[16]=plot_decomposition(samples["control"],confidence[16],circulation_guides=guides)
    if 17 in wanted:
        figures[17]=plot_forcing_comparison(analysis,confidence=confidence[17],phase_confidence=phase_confidence)
    if 18 in wanted:
        products={name:load_or_build_publication(simulation=name,archive_dir=Path(archive_root)/archives[name],
            control_archive_dir=Path(archive_root)/archives["control"],cache_dir=cache_dir,pattern=pattern,start=start,end=end,
            analysis=analysis,tem_root=tem_root,figures_only=figures_only,rebuild=rebuild) for name in runs}
        transport=confidence_for("section_transport",samples["control"].section_transport,samples["nudged-u"].section_transport,
            cache_dir=cache_dir,sources=sources,absolute=True,figures_only=figures_only)
        figures[18]=plot_publication_mechanisms(products,forcing_confidence=confidence[18],transport_confidence=transport,
            circulation_guides=guides)
    if set(wanted)&{19,20,21,22}:
        products={name:load_or_build_wind_variance(simulation=name,archive_dir=Path(archive_root)/archives[name],
            control_archive_dir=Path(archive_root)/archives["control"],cache_dir=cache_dir,pattern=pattern,start=start,end=end,
            tem_root=tem_root,figures_only=figures_only,rebuild=rebuild,mean_fields_only=19 not in wanted) for name in runs}
        if 19 in wanted:
            variance_sources=[Path(cache_dir)/"epfd"/f"sector_wind_variance_{simulation_cache_name(name)}.nc" for name in runs]
            variance=confidence_for("horizontal_wind_variance",*[products[name].horizontal_wind_variance_samples for name in runs],
                cache_dir=cache_dir,sources=variance_sources,figures_only=figures_only)
            figures[19]=plot_wind_variance(products,confidence=variance,circulation_guides=guides)
    if set(wanted)&{20,21,22}:
        bands={name:load_or_build_band_response(simulation=name,archive_dir=Path(archive_root)/archives[name],
            control_archive_dir=Path(archive_root)/archives["control"],cache_dir=cache_dir,pattern=pattern,start=start,end=end,
            analysis=analysis,covariance_path=covariance_paths[name],tem_root=tem_root,
            figures_only=figures_only,rebuild=rebuild,require_vectors=20 in wanted,
            mean_fields_only=figures_only or not (set(wanted)&{20,21})) for name in runs}
        band_sources=[Path(cache_dir)/"epfd"/f"wave_band_response_{simulation_cache_name(name)}.nc" for name in runs]
        if set(wanted)&{20,21}:
            band_samples=[None,None] if figures_only else [bands[name].forcing for name in runs]
            band_confidence=confidence_for("wave_band_forcing",*band_samples,
                cache_dir=cache_dir,sources=band_sources,figures_only=figures_only)
        variance_fields={}
        for name in runs:
            lower=products[name].horizontal_wind_variance.interp(absolute_latitude=bands["control"].absolute_latitude)
            upper=bands[name].upper_horizontal_wind_variance.rename(variance_pres="pres")
            variance_fields[name]=xr.concat([upper.where(upper.pres<100.,drop=True),lower],dim="pres").sortby("pres")
        if 20 in wanted:
            figures[20]=plot_band_response(bands["control"],bands["nudged-u"],band_confidence,
                variance_fields["nudged-u"]-variance_fields["control"],circulation_guides=guides)
        if set(wanted)&{21,22}:
            height_profile=load_energy_flux_height_profile(control_archive_dir=Path(archive_root)/archives["control"],
                start=start,end=end,tem_root=tem_root)
            energy={name:load_or_build_energy_flux(simulation=name,archive_dir=Path(archive_root)/archives[name],
                control_archive_dir=Path(archive_root)/archives["control"],cache_dir=cache_dir,pattern=pattern,
                start=start,end=end,tem_root=tem_root,figures_only=figures_only,rebuild=rebuild) for name in runs}
            if 21 in wanted:
                figures[21]=plot_wave_energy_flux(bands["control"],bands["nudged-u"],energy,variance_fields,
                    band_confidence,height_profile=height_profile,circulation_guides=guides)
            if 22 in wanted:
                publication={name:load_or_build_publication_response(simulation=name,
                    archive_dir=Path(archive_root)/archives[name],control_archive_dir=Path(archive_root)/archives["control"],
                    cache_dir=cache_dir,pattern=pattern,start=start,end=end,covariance_path=covariance_paths[name],
                    analysis=analysis,base_energy=energy[name],
                    tem_root=tem_root,figures_only=figures_only,rebuild=rebuild) for name in runs}
                publication_sources=[forcing_cache_path(cache_dir,name) for name in runs]
                publication_samples=[None,None] if figures_only else [publication[name].forcing for name in runs]
                publication_confidence=confidence_for("publication_wave_forcing",
                    *publication_samples,cache_dir=cache_dir,
                    sources=publication_sources,figures_only=figures_only)
                variance_samples=[None,None] if figures_only else [publication[name].horizontal_wind_variance_samples for name in runs]
                publication_variance_confidence=confidence_for("publication_wave_variance",*variance_samples,
                    cache_dir=cache_dir,sources=[variance_cache_path(cache_dir,name) for name in runs],figures_only=figures_only)
                publication_energy={name:publication[name][["energy_flux_poleward","energy_flux_upward"]] for name in runs}
                figures[22]=plot_wave_publication_response(publication["control"],publication["nudged-u"],
                    publication_energy,{name:publication[name].horizontal_wind_variance for name in runs},
                    publication_confidence,publication_variance_confidence,height_profile=height_profile,circulation_guides=guides)
    if 24 in wanted:
        figures[24]=background_pv_figure()
    import matplotlib.pyplot as plt
    for number,figure in figures.items():
        if not np.isclose(figure.get_figwidth(),8.):
            raise ValueError(f"Figure {number} must have an eight-inch canvas before vector calibration")
        save_figure(number,figure,FIGURES[number])
        caption=CAPTIONS[number]+(UNCERTAINTY_TEXT if number not in (17,21,23,24) else "")
        (Path(output_dir)/f"{number:02d}_{FIGURES[number]}.caption.txt").write_text(caption+"\n")
        plt.close(figure)
