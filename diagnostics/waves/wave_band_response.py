"""Native TEM forcing, EP flux, and wind variance for three broad spectral bands."""
from __future__ import annotations

import glob
import json
from pathlib import Path
import numpy as np
import xarray as xr

from diagnostics.analysis_config import simulation_cache_name
from diagnostics.cache_support import cache_manifest, materialize, read_json_file, write_json_file, write_netcdf_atomic
from diagnostics.diagnostic_helpers import progress
from diagnostics.mean_state import circulation_mechanism as mechanism
from diagnostics.mean_state import circulation_diagnostics as circulation
from diagnostics.tem_archive import load_stored_tem, tem_source_paths
from diagnostics.tem_core import TEM_VERSION
from diagnostics.waves.publication_wave_forcing import BACKGROUND_FIELDS, _native_profile_stage, _read_staged_height
from diagnostics.waves.regional_epfd_spectrum import control_region_weights, mechanism_latitude_indices
from diagnostics.waves.sector_transport import native_sector_fields
from diagnostics.waves.wave_uncertainty import PressureSampler, native_height_slice
from diagnostics.waves.sector_wind_variance import compute_variance_sections

VERSION="wave-band-response-v1-native-pooled"
VECTOR_VERSION="jucker2021-circulation-v1"
BANDS=("low_eastward","low_westward","high_wavenumber")
PRESSURES=np.unique(np.r_[np.arange(70.,351.,10.),mechanism.REGION_PRESSURES])
LATITUDES=np.unique(np.r_[np.arange(1.,41.),mechanism.REGION_LATITUDES])


def jucker_native_flux(fields,background):
    """Apply the circulation pressure Jacobian before sampling or averaging."""
    horizontal,upward=circulation._publication_vector_components(
        fields.ep_flux_meridional_normalized.rename(latitude="absolute_latitude"),
        fields.ep_flux_vertical_normalized.rename(latitude="absolute_latitude"),
        (background.pressure/100.).rename(latitude="absolute_latitude"),kind="ep",latitude_span=40.)
    return xr.Dataset(dict(horizontal=horizontal.rename(absolute_latitude="latitude"),
        upward=upward.rename(absolute_latitude="latitude")))


def high_wavenumber_covariance(spatial,longitude_count):
    """Exclude stationary, temporal Nyquist, and spatial Nyquist contributions exactly.

    Removing the temporal mean and alternating mean applies the original high-k
    mask without a complete temporal FFT. Symmetric covariance allocation retains
    the same interaction terms as every existing broad-sector budget.
    """
    stop=spatial.shape[-1]-(longitude_count%2==0)
    high=np.asarray(spatial[...,21:stop],dtype=np.complex128)
    if high.shape[-1]==0:
        return np.zeros((*spatial.shape[:2],3))
    filtered=high-high.mean(axis=0,keepdims=True)
    if high.shape[0]%2==0:
        alternating=np.where(np.arange(high.shape[0])%2, -1.,1.)[:,None,None,None]
        filtered-=alternating*(high*alternating).mean(axis=0,keepdims=True)
    output=np.empty((*spatial.shape[:2],3))
    for index,(left,right) in enumerate(((0,1),(0,2),(1,3))):
        cross=.5*(filtered[:,:,left]*high[:,:,right].conj()+high[:,:,left]*filtered[:,:,right].conj()).real
        output[...,index]=cross.sum(axis=-1)*(2./longitude_count**2)
    return output


def _high_covariances(stage,background,nlon,directory,signature):
    shape=(background.sizes["Time"],background.sizes["z"],background.sizes["latitude"],3)
    path=directory/"high_covariance.dat"
    state_path=directory/"high_covariance.json"
    state=read_json_file(state_path) or {}
    resumed=(path.exists() and state.get("manifest_sha256")==signature and state.get("shape")==list(shape))
    values=np.memmap(path,dtype="float64",mode="r+" if resumed else "w+",shape=shape)
    count=int(state.get("complete_heights",0)) if resumed else 0
    try:
        for level in range(count,shape[1]):
            plane=_read_staged_height(stage,level)
            for left in range(0,shape[2],8):
                values[:,level,left:left+8]=high_wavenumber_covariance(plane[:,left:left+8],nlon)
            del plane
            values.flush()
            write_json_file(state_path,dict(shape=list(shape),complete_heights=level+1,manifest_sha256=signature))
            progress(f"Broad-band response: high-wavenumber covariances, height {level+1}/{shape[1]}")
        return np.array(values)
    finally:
        values._mmap.close()


def load_or_build_band_response(*,simulation,archive_dir,control_archive_dir,cache_dir,pattern,start,end,
        analysis,covariance_path,tem_root=None,figures_only=False,rebuild=False,scratch_root=None,output_path=None,
        require_vectors=False,mean_fields_only=False):
    """Reuse low-frequency covariances and read missing high-wavenumber maps once."""
    path=Path(output_path or Path(cache_dir)/"epfd"/f"wave_band_response_{simulation_cache_name(simulation)}.nc")
    sources=sorted(glob.glob(str(Path(archive_dir)/pattern)))+[Path(covariance_path).resolve()]
    sources+=tem_source_paths(Path(archive_dir).name,root=tem_root)+tem_source_paths(Path(control_archive_dir).name,root=tem_root)
    manifest=cache_manifest(__name__,schema=VERSION,configuration=dict(simulation=simulation,start=str(start),end=str(end),
        pressures=PRESSURES,latitudes=LATITUDES,bands=BANDS,stage_dtype="complex64",cutoff_wavenumber=20,cutoff_frequency=.4),
        dependencies=dict(tem_version=TEM_VERSION),source_paths=sources)
    cached=None
    if path.exists() and not rebuild:
        with xr.open_dataset(path) as previous:
            saved=json.loads(previous.attrs.get("cache_manifest","{}"))
            before=dict(saved.get("configuration",{})); before.pop("pressures",None)
            after=dict(manifest["configuration"]); after.pop("pressures",None)
            reusable=(previous.attrs.get("manifest_sha256")==saved.get("manifest_sha256")
                and before==after and all(saved.get(key)==manifest.get(key)
                    for key in ("producer","schema","dependencies","sources")))
            if reusable:
                if (previous.attrs.get("manifest_sha256")==manifest["manifest_sha256"]
                        and "upper_horizontal_wind_variance" in previous
                        and (not require_vectors or (previous.attrs.get("vector_version")==VECTOR_VERSION
                            and all(name in previous for name in ("ep_flux_horizontal","ep_flux_upward"))))):
                    progress(f"{simulation}: using cached broad-band response: {path}")
                    names=["zonal_wind","upper_horizontal_wind_variance"] if mean_fields_only else list(previous.data_vars)
                    if require_vectors:
                        names=list(dict.fromkeys(names+["ep_flux_horizontal","ep_flux_upward"]))
                    else:
                        names=[name for name in names if name not in ("ep_flux_horizontal","ep_flux_upward")]
                    return previous[names].load()
                cached=previous.load()
    if figures_only:
        raise FileNotFoundError(f"Missing broad-band forcing, variance, or required Jucker vectors: {path}; calculate without --figures-only")
    upstream=load_stored_tem(Path(archive_dir).name,root=tem_root,start=start,end=end)
    control=load_stored_tem(Path(control_archive_dir).name,root=tem_root,start=start,end=end)
    stage=None
    try:
        fixed=control_region_weights(materialize(control[["rho_zm","pressure"]].isel(
            latitude=mechanism_latitude_indices(control.latitude))))
        with xr.open_dataset(covariance_path) as covariances:
            background=materialize(upstream[list(BACKGROUND_FIELDS)].sel(z=covariances.z,latitude=covariances.latitude))
            background=background.isel(z=native_height_slice(background.pressure,top=float(PRESSURES.min())))
            sampler=PressureSampler(background,PRESSURES,LATITUDES,fixed)
            high=None
            upper_variance=None
            if cached is not None and "high_wavenumber_covariance" in cached:
                high=cached.high_wavenumber_covariance
                if "upper_horizontal_wind_variance" in cached:
                    upper_variance=cached.upper_horizontal_wind_variance
            missing=background.z if high is None else background.z.where(~background.z.isin(high.z),drop=True)
            if missing.size or upper_variance is None:
                upper_slice=native_height_slice(background.pressure,bottom=100.,top=float(PRESSURES.min()))
                stage_background=background if high is None else background.isel(z=upper_slice)
                scratch_root=scratch_root or Path(__file__).resolve().parents[2]/".epfd_scratch"/"wave-band-response"/"native"
                stage,nlon,directory=_native_profile_stage(archive_dir,pattern,start,end,upstream,stage_background,manifest,
                    scratch_root,stage_dtype="complex64",include_wind_variance=True)
                calculated=_high_covariances(stage,stage_background,nlon,directory,manifest["manifest_sha256"])
                calculated=xr.DataArray(calculated,dims=("Time","z","latitude","covariance"),
                    coords=dict(Time=stage_background.Time,z=stage_background.z,latitude=stage_background.latitude,
                        covariance=["uv","uw","vtheta"]))
                if high is None:
                    high=calculated
                else:
                    overlap=high.z.where(high.z.isin(calculated.z),drop=True)
                    np.testing.assert_allclose(calculated.sel(z=overlap),high.sel(z=overlap),atol=2e-9,rtol=2e-6,
                        err_msg="Extended native covariances differ from the existing calculation")
                    high=xr.concat([high,calculated.sel(z=missing)],dim="z").sortby("z")
                if upper_variance is None:
                    variance_slice=native_height_slice(stage_background.pressure,bottom=100.,top=float(PRESSURES.min()))
                    variance=compute_variance_sections(stage[:,variance_slice],stage_background.isel(z=variance_slice),
                        nlon,fixed,pressures=np.arange(float(PRESSURES.min()),101.,10.),wind_indices=(0,4))
                    upper_variance=variance.horizontal_wind_variance.interp(absolute_latitude=LATITUDES).rename(pres="variance_pres")
                    variance_path=Path(cache_dir)/"epfd"/f"sector_wind_variance_{simulation_cache_name(simulation)}.nc"
                    if variance_path.exists():
                        with xr.open_dataset(variance_path) as previous_variance:
                            check_latitudes=np.arange(5.,16.)
                            old=previous_variance.horizontal_wind_variance.sel(pres=100.).interp(absolute_latitude=check_latitudes)
                            new=upper_variance.sel(variance_pres=100.).interp(absolute_latitude=check_latitudes)
                            np.testing.assert_allclose(new,old,atol=2e-6,rtol=2e-5,
                                err_msg="Extended variance differs from the existing 100 hPa calculation")
            high=high.sel(z=background.z).transpose("Time","z","latitude","covariance")
            forcing,vectors=[],[]
            def retain(values):
                fields=native_sector_fields(values,background)
                forcing.append(xr.concat([sampler.samples(fields[name])*86400. for name in ("EPFd","EPFd_y","EPFd_z")],
                    dim=xr.IndexVariable("component",["total","meridional","vertical"])))
                if require_vectors:
                    transformed=jucker_native_flux(fields,background)
                    vectors.append(xr.Dataset({name:sampler.samples(transformed[name],parity).mean("Time")
                        for name,parity in (("horizontal","odd"),("upward","even"))}))
            for band in BANDS[:2]:
                progress(f"{simulation}: reusing native covariance samples for {band}")
                values=materialize(covariances.pathway_covariance.sel(pathway=band,z=background.z).transpose(
                    "Time","z","latitude","covariance"))
                retain(values.values)
            retain(high.values)
        sector=xr.IndexVariable("sector",list(BANDS))
        product=xr.Dataset(dict(forcing=xr.concat(forcing,dim=sector),zonal_wind=sampler.samples(background.u_bar).mean("Time")))
        if cached is not None:
            common=cached.pres.where(cached.pres.isin(product.pres),drop=True)
            np.testing.assert_allclose(product.forcing.sel(pres=common),cached.forcing.sel(pres=common),
                atol=2e-7,rtol=2e-6,equal_nan=True,err_msg="Extended maps change the existing forcing calculation")
        if upper_variance is not None:
            product["upper_horizontal_wind_variance"]=upper_variance
        if require_vectors:
            transformed=xr.concat(vectors,dim=sector)
            product["ep_flux_horizontal"]=transformed.horizontal
            product["ep_flux_upward"]=transformed.upward
            for name in ("ep_flux_horizontal","ep_flux_upward"):
                product[name].attrs.update(units="m3",definition="Jucker (2021) circulation transform before final panel dimensions")
            product.attrs["vector_version"]=VECTOR_VERSION
        product["high_wavenumber_covariance"]=high
        product.high_wavenumber_covariance.attrs["definition"]="mass-bearing native covariances for reusable high-wavenumber TEM forcing"
        # The displayed pressure grid retains every original regional quadrature point.
        weights=fixed.sum("hemisphere").rename(region_pres="pres",region_latitude="absolute_latitude")
        expected_names=(("low_eastward",),("low_westward",),("high_wavenumber_eastward","high_wavenumber_westward"))
        for band,names in zip(BANDS,expected_names):
            regional=(product.forcing.sel(sector=band,pres=weights.pres,absolute_latitude=weights.absolute_latitude)*weights).sum(("pres","absolute_latitude")).mean("Time")
            expected=analysis.sector_forcing.sel(comparison=simulation,sector=list(names)).sum("sector")
            np.testing.assert_allclose(regional,expected,atol=2e-7,rtol=2e-6,
                err_msg=f"{band} map differs from the original regional forcing budget")
        np.testing.assert_allclose(product.forcing.sel(component="total"),
            product.forcing.sel(component="meridional")+product.forcing.sel(component="vertical"),atol=1e-9,rtol=1e-9)
        product.forcing.attrs.update(units="m s-1 day-1",definition="instantaneous native TEM divergence before pressure sampling")
        product.attrs.update(product_version=VERSION,manifest_sha256=manifest["manifest_sha256"],cache_manifest=json.dumps(manifest,sort_keys=True))
        write_netcdf_atomic(product,path,encoding={name:dict(zlib=True,complevel=2) for name in product.data_vars})
        if stage is not None:
            stage._mmap.close();stage=None
            for filename in ("fourier.dat","progress.json","high_covariance.dat","high_covariance.json"):
                (directory/filename).unlink(missing_ok=True)
            for item in directory.glob("reader*.json"):item.unlink()
            directory.rmdir()
        progress(f"{simulation}: complete broad-band forcing and variance cache: {path}")
        return product
    finally:
        if stage is not None:stage._mmap.close()
        upstream.close();control.close()
