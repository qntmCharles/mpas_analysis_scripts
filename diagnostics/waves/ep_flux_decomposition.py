"""Shared native archive access for the retained wave-forcing calculations."""
from __future__ import annotations
import glob
import os
import numpy as np
import xarray as xr
from diagnostics.tem_native import NATIVE_FIELDS, WATER_SPECIES, zfull_from_latlon
from diagnostics.diagnostic_helpers import latitude_is_hemispherically_symmetric, standardize_latlon_dataset

def validate_spectral_grid(dataset):
    """Require complete time sampling and a uniform periodic longitude grid."""
    lon = np.asarray(dataset.longitude, dtype=float)
    lat = np.asarray(dataset.latitude, dtype=float)
    height = np.asarray(dataset.z, dtype=float)
    if lon.size < 32 or not np.allclose(np.diff(lon), 360.0 / lon.size, atol=1e-4):
        raise ValueError(
            "EPFd requires a complete uniform longitude grid with wavenumber 16 resolved"
        )
    if (
        lat.size < 3
        or np.any(np.diff(lat) <= 0)
        or not latitude_is_hemispherically_symmetric(lat)
    ):
        raise ValueError("EPFd requires ordered paired latitudes")
    if height.size < 3 or not np.isfinite(height).all() or np.any(np.diff(height) <= 0):
        raise ValueError(
            "EPFd requires at least three increasing geometric-height levels"
        )
    time = np.asarray(dataset.Time)
    if time.size < 3 or not np.all(np.diff(time) == np.timedelta64(6, "h")):
        raise ValueError("EPFd requires a complete six-hourly record")

def _open_epfd_raw_source(path):
    """Bypass full-chunk caching for narrow reads of uncompressed raw fields."""
    from diagnostics.tem_native import open_native_dataset
    return open_native_dataset(path)

def _open_epfd_raw_archive(archive_dir, pattern, start, end, z_coord, *, requested_fields=None):
    """Open archive metadata and expose one sequential native-frame iterator."""
    from types import SimpleNamespace
    from diagnostics.tem_native import iter_native_frames

    paths = sorted(glob.glob(os.path.join(archive_dir, pattern)))
    if not paths:
        raise FileNotFoundError(f"No files match {pattern!r} in {archive_dir}")
    sources, selected = [], []
    def close_sources():
        for source in sources:
            source.close()
    try:
        for path in paths:
            raw = _open_epfd_raw_source(path)
            sources.append(raw)
            names = [*NATIVE_FIELDS, *[name for name in WATER_SPECIES if name in raw]]
            source = standardize_latlon_dataset(raw, names).sel(Time=slice(start, end))
            if "zgrid" in raw:
                source["zgrid"] = xr.DataArray(zfull_from_latlon(raw), dims="nVertLevelsP1")
            if source.sizes["Time"]:
                selected.append(source)
        if not selected:
            raise ValueError("EPFd raw archive has no samples in the requested period")
        expected = np.arange(np.datetime64(start), np.datetime64(end)+np.timedelta64(6, "h"), np.timedelta64(6, "h")).astype("datetime64[ns]")
        if not np.array_equal(np.concatenate([source.Time.values for source in selected]), expected):
            raise ValueError("Input does not have the expected complete 6-hourly Time axis")
        for source in selected[1:]:
            for coordinate in ("latitude", "longitude"):
                if not np.array_equal(source[coordinate], selected[0][coordinate]):
                    raise ValueError(f"Native files disagree on {coordinate}")
        dataset = xr.Dataset(coords=dict(Time=expected, z=z_coord, latitude=selected[0].latitude,
                                        longitude=selected[0].longitude))
        dataset.attrs["density_convention"] = ("dry density times (1 + all six water mixing ratios)"
                                               if any(name in selected[0] for name in WATER_SPECIES)
                                               else "supplied dry density; no water species")
        dataset.set_close(close_sources)
        def frames(latitude, output_height=None):
            for source in selected:
                options = {} if requested_fields is None else {"requested_fields": requested_fields}
                # The global reference interface column is already fixed above.
                # All remaining preparation acts independently in each vertical column.
                selected_source = source.sel(latitude=slice(float(np.min(latitude)), float(np.max(latitude))))
                yield from iter_native_frames(selected_source, z_coord, latitude, output_height=output_height, **options)
        return dataset, SimpleNamespace(iter_frames=frames)
    except Exception:
        close_sources()
        raise
