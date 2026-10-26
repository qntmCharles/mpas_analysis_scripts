"""Sampling uncertainty for combined-hemisphere wave diagnostics.

Circular moving-block resampling preserves simultaneous spatial and variable
covariance. Control and experiment draws remain independent. The hemisphere
average precedes resampling; hemispheres never supply independent replicates.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

BLOCK_DAYS = 40.
REPLICATES = 999
SEED = 20240602


def block_weights(count, block_samples, replicates=REPLICATES, seed=SEED):
    """Return normalized time counts for circular moving-block bootstrap draws."""
    if count < 4 or not 1 <= block_samples < count:
        raise ValueError("Block length must be positive and shorter than the record")
    rng = np.random.default_rng(seed)
    starts = rng.integers(count, size=(replicates, int(np.ceil(count/block_samples))))
    indices = (starts[..., None]+np.arange(block_samples)) % count
    indices = indices.reshape(replicates, -1)[:, :count]
    weights = np.zeros((replicates, count))
    for index, draw in enumerate(indices):
        weights[index] = np.bincount(draw, minlength=count)/count
    return weights


def mean_response_interval(control, experiment, *, block_days=BLOCK_DAYS, replicates=REPLICATES,
        seed=SEED, samples_per_day=4., chunk_size=1024):
    """Estimate independent-run mean changes and pointwise percentile intervals."""
    xr.align(control, experiment, join="exact")
    dimensions=[name for name in control.dims if name!="Time"]
    c,u=[field.transpose("Time",*dimensions).values.reshape(field.sizes["Time"],-1).astype(float)
        for field in (control,experiment)]
    if c.shape!=u.shape:
        raise ValueError("Bootstrap requires aligned complete observations")
    complete=np.isfinite(c).all(axis=0)&np.isfinite(u).all(axis=0)
    absent=np.isnan(c).all(axis=0)&np.isnan(u).all(axis=0)
    if not np.all(complete|absent) or not complete.any():
        raise ValueError("Bootstrap requires finite time series; wholly absent cells remain masked")
    full_points=c.shape[1]
    c,u=c[:,complete],u[:,complete]
    count,points=c.shape
    estimate=u.mean(axis=0)-c.mean(axis=0)
    cw=block_weights(count,int(round(block_days*samples_per_day)),replicates,seed)
    uw=block_weights(count,int(round(block_days*samples_per_day)),replicates,seed+1)
    interval=np.empty((2,points))
    for left in range(0,points,chunk_size):
        right=min(points,left+chunk_size)
        draws=uw@u[:,left:right]-cw@c[:,left:right]
        interval[:,left:right]=np.percentile(draws,[2.5,97.5],axis=0)
    shape=tuple(control.sizes[name] for name in dimensions)
    coords={name:control[name] for name in dimensions}
    def expand(values):
        output=np.full((*values.shape[:-1],full_points),np.nan)
        output[...,complete]=values
        return output.reshape((*values.shape[:-1],*shape))
    bounds=xr.DataArray(expand(interval),dims=("bound",*dimensions),coords=dict(bound=["lower","upper"],**coords))
    result=xr.Dataset(dict(estimate=xr.DataArray(expand(estimate),dims=dimensions,coords=coords),
        pointwise_interval=bounds,significant=(bounds.sel(bound="lower")>0)|(bounds.sel(bound="upper")<0)))
    result.attrs.update(method="independent-run circular moving-block bootstrap; simultaneous hemisphere average",
        replicates=replicates,seed=seed,confidence_percent=95.,block_days=float(block_days),sample_count=count,
        inference="pointwise percentile intervals; no simultaneous field or multiple-block test",
        block_choice="wave figures use 40days to retain slow subtropical westward-variance variability",
        limitation="sampling uncertainty conditional on the finite record and fixed spectral filters")
    return result


def absolute_mean_interval(field, **kwargs):
    """Estimate one run's mean uncertainty against an exact zero reference."""
    return mean_response_interval(xr.zeros_like(field).where(field.notnull()),field,**kwargs)


class PressureSampler:
    """Interpolate instantaneous native fields before pooling and time averaging."""

    def __init__(self, background, pressures, latitudes, fixed_weights):
        from diagnostics.mean_state.circulation_bootstrap import hemisphere_samples
        self.background = background
        self.pressures, self.latitudes = np.asarray(pressures), np.asarray(latitudes)
        pressure = background.pressure.transpose("Time","z","latitude").values/100.
        if not np.isfinite(pressure).all() or not np.all(np.diff(pressure,axis=1)<0):
            raise ValueError("Native pressure must be complete and decrease with height")
        if np.any(self.pressures[None,:,None]>pressure[:,:1]) or np.any(self.pressures[None,:,None]<pressure[:,-1:]):
            raise ValueError("Native heights do not bracket the requested pressure range")
        self.left = np.stack([(pressure>value).sum(axis=1)-1 for value in self.pressures],axis=1)
        self.left = np.clip(self.left,0,pressure.shape[1]-2)
        low, high = [np.take_along_axis(pressure,self.left+offset,axis=1) for offset in (0,1)]
        self.fraction = (self.pressures[None,:,None]-low)/(high-low)
        point = fixed_weights.rename(region_pres="pres",region_latitude="absolute_latitude")
        bounds_p = (float(point.pres.min()),float(point.pres.max()))
        bounds_lat = (float(point.absolute_latitude.min()),float(point.absolute_latitude.max()))
        weight = point/point.sum("hemisphere")
        self.weights = weight.interp(pres=xr.DataArray(np.clip(self.pressures,*bounds_p),dims="pres"),
            absolute_latitude=xr.DataArray(np.clip(self.latitudes,*bounds_lat),dims="absolute_latitude")).assign_coords(
                pres=self.pressures,absolute_latitude=self.latitudes)
        self._hemisphere_samples = hemisphere_samples

    def samples(self, field, parity="even"):
        values = field.transpose("Time","z","latitude").values
        low, high = [np.take_along_axis(values,self.left+offset,axis=1) for offset in (0,1)]
        sampled = xr.DataArray(low+self.fraction*(high-low),dims=("Time","pres","latitude"),
            coords=dict(Time=self.background.Time,pres=self.pressures,latitude=self.background.latitude))
        sides = self._hemisphere_samples(sampled,parity).interp(absolute_latitude=self.latitudes)
        pooled = (sides*self.weights).sum("hemisphere",skipna=False)
        if 0. in self.latitudes:
            # Sample the unique equator across signed native latitudes, before pooling.
            # Poleward components have odd reflection parity and vanish at the fold.
            equator = sampled.interp(latitude=0.).drop_vars("latitude")
            if parity == "odd":
                equator = xr.zeros_like(equator)
            pooled.loc[dict(absolute_latitude=0.)] = equator
        return pooled


def native_height_slice(pressure, bottom=350., top=100.):
    """Retain all heights needed for interpolation and native derivatives."""
    values = pressure.transpose("Time","z","latitude").values/100.
    low = int((values>bottom).sum(axis=1).min())-2
    high = int((values>top).sum(axis=1).max())+2
    return slice(max(0,low),min(values.shape[1],high))
