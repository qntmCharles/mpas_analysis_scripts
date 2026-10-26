"""Time-local regional EPFd in fixed phase-speed bins for block resampling."""
from __future__ import annotations

import glob
import json
from pathlib import Path
from time import perf_counter

import numpy as np
import xarray as xr
from scipy import fft

from diagnostics.analysis_config import simulation_cache_name
from diagnostics.cache_support import cache_manifest, materialize, write_netcdf_atomic
from diagnostics.diagnostic_helpers import progress
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.tem_core import FLUX_TERMS, TEM_VERSION
from diagnostics.waves.publication_wave_forcing import (
    FAMILIES, _cached_product, _native_profile_stage, _read_staged_height, phase_bin_allocation)
from diagnostics.waves.regional_ep_flux_sectors import forcing_components
from diagnostics.waves.regional_epfd_spectrum import (
    REFERENCE_FIELDS, control_region_weights, forcing_covariance_weights, mechanism_latitude_indices,
    regional_native_weights, signed_phase_speed_histogram,
    PHASE_SPEED_DISPLAY_EDGES as PHASE_EDGES, PHASE_SPEED_DISPLAY_CENTRES as PHASE_CENTRES)

VERSION = "regional-phase-speed-samples-v1-symmetric-native"


def phase_bin_forcing(spatial, weights, longitude_count, groups):
    """Apply the existing symmetric covariance allocation to instantaneous regional TEM weights.

    Spatial axes are time, point, field=(u,rho*v,rho*w,theta), positive zonal mode.
    Weights have axes flux term, time, point. Each group supplies its complete-record filter.
    """
    source=np.asarray(spatial,dtype=np.complex128)
    transformed=fft.fft(source,axis=0,workers=2)
    multiplicity=np.full(source.shape[-1],2./longitude_count**2)
    multiplicity[0]*=.5
    if longitude_count%2==0:
        multiplicity[-1]*=.5
    result=np.zeros((len(groups),source.shape[0],len(FLUX_TERMS)))
    for group,mask in enumerate(groups):
        modes=np.flatnonzero(mask.any(axis=0))
        if not modes.size:
            continue
        original=source[...,modes]
        filtered=fft.ifft(transformed[...,modes]*mask[:,None,None,modes],axis=0,workers=2)
        for term,(left,right) in enumerate(((0,1),(0,2),(1,3),(1,3))):
            cross=.5*(filtered[:,:,left]*original[:,:,right].conj()
                +original[:,:,left]*filtered[:,:,right].conj()).real
            covariance=np.einsum("tpk,k->tp",cross,multiplicity[modes])
            result[group,:,term]=np.einsum("tp,tp->t",covariance,weights[term])
    return result


def compute_phase_speed_samples(stage, background, weights, longitude_count, directory, signature):
    edges,allocation=phase_bin_allocation(background.sizes["Time"],longitude_count,bin_width=5.)
    centres=.5*(edges[:-1]+edges[1:])
    indices=[int(np.flatnonzero(np.isclose(centres,value))[0]) for value in PHASE_CENTRES]
    groups=[allocation==family*(edges.size-1)+index for family in range(2) for index in indices]
    checkpoint=directory/"phase-speed-progress.nc"
    totals=np.zeros((len(groups),background.sizes["Time"],len(FLUX_TERMS)))
    start=0
    if checkpoint.exists():
        with xr.open_dataset(checkpoint) as previous:
            if previous.attrs.get("manifest_sha256")==signature:
                totals=previous.forcing_by_flux.values.copy()
                start=int(previous.attrs["complete_heights"])
    for level in range(start,background.sizes["z"]):
        started=perf_counter()
        columns=np.flatnonzero(np.any(weights[:,:,level]!=0.,axis=(0,1)))
        if columns.size:
            plane=_read_staged_height(stage,level)
            for left in range(0,columns.size,4):
                selected=columns[left:left+4]
                totals+=phase_bin_forcing(plane[:,selected],weights[:,:,level,selected],longitude_count,groups)
            del plane
        saved=xr.Dataset(dict(forcing_by_flux=(("group","Time","flux_term"),totals)),
            attrs=dict(manifest_sha256=signature,complete_heights=level+1))
        write_netcdf_atomic(saved,checkpoint)
        progress(f"Phase-speed uncertainty: height {level+1}/{background.sizes['z']} complete ({perf_counter()-started:.1f} s)")
    terms=xr.DataArray(totals.reshape(2,PHASE_CENTRES.size,background.sizes["Time"],len(FLUX_TERMS))*86400.,
        dims=("family","phase_speed","Time","flux_term"),coords=dict(family=list(FAMILIES),
            phase_speed=PHASE_CENTRES,Time=background.Time,flux_term=list(FLUX_TERMS)))
    forcing=forcing_components(terms).transpose("Time","family","phase_speed","component")
    forcing.attrs.update(units="m s-1 day-1",definition="time-local symmetric spectral allocation; instantaneous native TEM coefficients")
    return xr.Dataset(dict(forcing=forcing,
        phase_speed_bounds=(("phase_speed","bound"),np.column_stack((PHASE_EDGES[:-1],PHASE_EDGES[1:])))))


def load_or_build_phase_speed_samples(*, simulation, archive_dir, control_archive_dir, cache_dir, pattern,
        start, end, analysis, tem_root=None, figures_only=False, rebuild=False):
    path=Path(cache_dir)/"epfd"/f"phase_speed_forcing_samples_{simulation_cache_name(simulation)}.nc"
    sources=sorted(glob.glob(str(Path(archive_dir)/pattern)))
    sources+=tem_source_paths(Path(archive_dir).name,root=tem_root)+tem_source_paths(Path(control_archive_dir).name,root=tem_root)
    manifest=cache_manifest(__name__,schema=VERSION,configuration=dict(simulation=simulation,start=str(start),end=str(end),
        edges=PHASE_EDGES,families=FAMILIES,stage_dtype="complex64"),dependencies=dict(tem_version=TEM_VERSION),source_paths=sources)
    cached=None if rebuild else _cached_product(path,manifest,figures_only)
    if cached is not None:
        progress(f"{simulation}: using cached phase-speed uncertainty samples: {path}")
        return cached
    if figures_only:
        raise FileNotFoundError(f"Calculate Figures 17/23 without --figures-only to create {path}")
    upstream=load_stored_tem(Path(archive_dir).name,root=tem_root,start=start,end=end)
    control=load_stored_tem(Path(control_archive_dir).name,root=tem_root,start=start,end=end)
    stage=None
    try:
        fixed=control_region_weights(materialize(control[["rho_zm","pressure"]].isel(
            latitude=mechanism_latitude_indices(control.latitude))))
        archive=materialize(upstream[list(REFERENCE_FIELDS)].isel(latitude=mechanism_latitude_indices(upstream.latitude)))
        regional,_=regional_native_weights(archive,fixed)
        weights=forcing_covariance_weights(archive,regional)
        active=np.flatnonzero(np.any(weights!=0.,axis=(0,1,3)))
        section=slice(int(active.min()),int(active.max())+1)
        background=archive.isel(z=section)
        weights=weights[:,:,section]
        scratch=Path(__file__).resolve().parents[2]/".epfd_scratch"/"phase-speed-uncertainty"
        stage,nlon,directory=_native_profile_stage(archive_dir,pattern,start,end,upstream,background,manifest,scratch,
            stage_dtype="complex64")
        product=compute_phase_speed_samples(stage,background,weights,nlon,directory,manifest["manifest_sha256"])
        for family,names in zip(FAMILIES,(("low_eastward","low_westward"),
                ("high_wavenumber_eastward","high_wavenumber_westward"))):
            mask=analysis.sector_mask.sel(sector=list(names)).any("sector")
            for component in ("meridional","vertical","total"):
                histogram=signed_phase_speed_histogram(analysis.spectrum.sel(comparison=simulation,component=component).where(mask,0.),
                    time_count=background.sizes["Time"],longitude_count=nlon)
                np.testing.assert_allclose(product.forcing.sel(family=family,component=component).mean("Time"),
                    histogram.forcing.sel(phase_speed=PHASE_CENTRES),rtol=3e-5,atol=2e-7,
                    err_msg="Time-local phase bins must recover the existing regional spectrum")
        product.attrs.update(product_version=VERSION,manifest_sha256=manifest["manifest_sha256"],
            cache_manifest=json.dumps(manifest,sort_keys=True))
        write_netcdf_atomic(product,path)
        stage._mmap.close()
        stage=None
        for item in directory.iterdir():
            item.unlink()
        directory.rmdir()
        progress(f"{simulation}: phase-speed uncertainty samples complete: {path}")
        return product
    finally:
        if stage is not None:
            stage._mmap.close()
        upstream.close()
        control.close()
