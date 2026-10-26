#!/usr/bin/env python
"""Describe native six-hour spin-up responses against matched control histories.

Reuse Figure 15's fixed regions, signed HC index, and exact precipitation
intervals. Use regional A_EP to avoid defining a spin-up climatology.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_THIS_DIR = Path(__file__).resolve().parent
for _parent in (_THIS_DIR.parents[1], _THIS_DIR.parents[2]):
    if str(_parent) not in sys.path:
        sys.path.append(str(_parent))

import numpy as np
import pandas as pd
import xarray as xr
from matplotlib import pyplot as plt

from diagnostics.diagnostic_helpers import (
    FigureSaver, cache_manifest, cache_manifest_matches, diagnostic_cache_dir,
    diagnostic_output_dir, load_netcdf, progress, read_json_file,
    simulation_colour, total_precip_accum, write_csv_atomic, write_json_file,
    write_netcdf_atomic,
)
from diagnostics.cache_support import materialize
from diagnostics import analysis_config as config
from diagnostics.mean_state import circulation_diagnostics as circulation
from diagnostics.mean_state import circulation_lead_lag as timing
from diagnostics.mean_state import circulation_mechanism as mechanism
from diagnostics.mean_state.nudging_region import regions_from_attributes
from diagnostics.tem_archive import height_coordinate_tem, validate_tem_metadata
from diagnostics.tem_core import TEM_VERSION

SCHEMA = "circulation-spinup-v2-native-six-hour-control-response"
INDICES = ("zonal_wind", "a_ep", "hc_streamfunction", "precipitation")
UNITS = ("m s-1", "m s-2", "kg s-1", "mm day-1")
LABELS = (r"$\Delta u$ (m s$^{-1}$)", r"$\Delta A_{\rm EP}$ (m s$^{-2}$)",
          r"$\Delta\Psi_{\rm HC}$ (kg s$^{-1}$)", r"$\Delta P$ (mm day$^{-1}$)")
SIX_HOURS = np.timedelta64(6, "h")
ONE_DAY = np.timedelta64(1, "D")


def configured_inputs():
    """Use the shared archive mapping and controller-selected simulation profile."""
    inputs, experiments = {}, {}
    for simulation, directory in config.SIMULATION_ARCHIVES.items():
        name = circulation.COMPARISON_NAMES[simulation][0]
        archive_root = config.SIMULATION_ROOT / directory
        root = config.spinup_directory(simulation)
        inputs[name] = dict(tem=str(root / config.SPINUP_TEM_FILENAME),
                            latlon=str(root / config.SPINUP_LATLON_FILENAME),
                            first_time=config.SPINUP_FIRST_TIME,
                            directory=str(archive_root), simulation=simulation)
        if simulation != "control":
            experiments[name] = simulation
    return inputs, experiments


def archive_times(dataset, first_time=None):
    """Decode explicit six-hour sample indices without guessing their origin."""
    values = np.asarray(dataset.Time)
    if not np.issubdtype(values.dtype, np.datetime64):
        if first_time is None or not np.array_equal(values, np.arange(len(values))):
            raise ValueError("Numeric Time requires first_time and contiguous sample indices 0, 1, ...")
        values = pd.Timestamp(first_time).to_datetime64() + np.arange(len(values)) * SIX_HOURS
        dataset = dataset.assign_coords(Time=values.astype("datetime64[ns]"))
    if len(values) < 2 or not np.all(np.diff(values) == SIX_HOURS):
        raise ValueError("Spin-up archives require contiguous six-hourly samples")
    return dataset


def complete_intervals(tem_times, surface_times, start, end):
    """Retain native samples with exact preceding precipitation endpoints."""
    first, last = (pd.Timestamp(value).to_datetime64().astype("datetime64[ns]") for value in (start, end))
    if last <= first or (last - first) % SIX_HOURS != np.timedelta64(0, "ns"):
        raise ValueError("--start/--end must delimit a positive whole number of six-hour intervals")
    tem_times, surface_times = map(np.asarray, (tem_times, surface_times))
    lower = max(tem_times[0] - SIX_HOURS, surface_times[0])
    upper = min(tem_times[-1], surface_times[-1])
    intervals = []
    for interval_start in np.arange(first, last, SIX_HOURS):
        interval_end = interval_start + SIX_HOURS
        if interval_start < lower or interval_end > upper:
            continue
        if interval_end not in tem_times or not np.isin([interval_start, interval_end], surface_times).all():
            raise ValueError(f"Missing TEM sample or precipitation endpoints within {interval_start} to {interval_end}")
        intervals.append((interval_start, interval_end))
    if not intervals:
        raise ValueError("No native spin-up samples have matching precipitation endpoints")
    if intervals[0][0] != first or intervals[-1][1] != last:
        progress(f"Available six-hour intervals cover {intervals[0][0]} to {intervals[-1][1]}; edge intervals lack coverage")
    return intervals


def build_native_indices(spec, start, end):
    """Reduce stored TEM at each timestamp and preceding six-hour precipitation."""
    tem_path, surface_path = (Path(spec[key]) for key in ("tem", "latlon"))
    validate_tem_metadata(".", "geometric-height", tem_path.parent, pattern=tem_path.name)
    with xr.open_dataset(tem_path, chunks={"Time": 4}) as source, xr.open_dataset(
        surface_path, chunks={"Time": 1},
    ) as surface_source:
        archive = height_coordinate_tem(archive_times(source, spec.get("first_time")))
        surface = archive_times(surface_source, spec.get("surface_first_time", spec.get("first_time")))
        intervals = complete_intervals(archive.Time.values, surface.Time.values, start, end)
        archive = archive.sel(Time=[last for _, last in intervals])
        endpoints = np.r_[intervals[0][0], [last for _, last in intervals]]
        accumulation = materialize(mechanism.tropical_precipitation(
            total_precip_accum(surface.sel(Time=endpoints))), num_workers=1)
        diagnostics = circulation.tem_instantaneous_diagnostics(archive)
        results = []
        for offset in range(0, len(intervals), 4):
            batch = intervals[offset:offset + 4]
            progress(f"{tem_path.name}: native spin-up indices, {batch[0][1]} to {batch[-1][1]}")
            times = [last for _, last in batch]
            state, raw = diagnostics.sel(Time=times), archive.sel(Time=times)
            region = mechanism.regional_tem_samples(state, raw)
            compact = timing.daily_tem_state(region, state, raw, average_time=False)[
                ["a_ep", "mass_weight", "hc_streamfunction"]]
            # Use the same fixed UTLS box and native-column pressure remapping.
            wind = mechanism.hemisphere_samples(raw.uReconstructZonal, "even")
            pressure = mechanism.hemisphere_samples(raw.pressure, "even")
            wind = mechanism._on_pressure(wind, pressure, mechanism.REGION_PRESSURES, "region_pres")
            compact["zonal_wind"] = wind.interp(absolute_latitude=mechanism.REGION_LATITUDES).rename(
                absolute_latitude="region_latitude")
            compact["precipitation"] = xr.concat([
                mechanism.precipitation_window(accumulation, first, last).expand_dims(Time=[last])
                for first, last in batch], dim="Time")
            compact = materialize(compact, num_workers=1)
            if not all(np.isfinite(field.values).all() for field in compact.data_vars.values()):
                raise ValueError(f"Nonfinite spin-up indices for {times}; check native archive coverage")
            compact["interval_start"] = ("Time", np.array([first for first, _ in batch], dtype="datetime64[ns]"))
            compact["interval_end"] = ("Time", np.array(times, dtype="datetime64[ns]"))
            results.append(compact)
    result = xr.concat(results, dim="Time", join="exact", combine_attrs="drop_conflicts")
    result.attrs.update(analysis_period="spin-up", ep_region=mechanism.REGION_DESCRIPTION,
                        hc_definition="Figure 15 fixed signed 5-15 degrees, 150-850 hPa",
                        sampling_definition="native TEM timestamps; preceding six-hour precipitation intervals")
    return result


def prepare_indices(name, spec, args):
    """Reuse one source-validated spin-up cache per history."""
    paths = [spec["tem"], spec["tem"] + ".metadata.json", spec["latlon"]]
    manifest = cache_manifest(
        "mean_state.circulation_spinup", schema=SCHEMA,
        configuration=dict(start=args.start, end=args.end, first_time=spec.get("first_time"),
                           surface_first_time=spec.get("surface_first_time", spec.get("first_time")),
                           ep_latitude=mechanism.REGION_LATITUDES.tolist(),
                           ep_pressure=mechanism.REGION_PRESSURES.tolist(),
                           hc_latitude=timing.HC_LATITUDES.tolist(), hc_pressure=timing.HC_PRESSURES.tolist(),
                           precipitation_latitude=timing.TROPICAL_LATITUDE,
                           sampling="native six-hour samples; exact preceding precipitation endpoints",
                           tem_version=TEM_VERSION), source_paths=paths,
    )
    root = args.cache_root / name
    path, metadata = root / "circulation_spinup_6hourly.nc", root / "spinup.manifest.json"
    if not args.rebuild_cache and path.is_file() and cache_manifest_matches(read_json_file(metadata), manifest):
        result = load_netcdf(path)
        if result.attrs.get("manifest_sha256") == manifest["manifest_sha256"]:
            progress(f"{name}: reuse native spin-up cache")
            return result
    if args.figures_only:
        raise FileNotFoundError(f"No current spin-up cache for {name}: {path}")
    result = build_native_indices(spec, args.start, args.end)
    result.attrs.update(cache_schema=SCHEMA, manifest_sha256=manifest["manifest_sha256"])
    write_netcdf_atomic(result, path)
    write_json_file(metadata, manifest)
    return result


def control_response(run, control):
    """Subtract matched control samples using common instantaneous mass weights."""
    times = np.intersect1d(run.Time.values, control.Time.values)
    if not len(times):
        raise ValueError("Experiment and control have no matching spin-up samples")
    run, control = xr.align(run.sel(Time=times), control.sel(Time=times), join="exact")
    if not (np.array_equal(run.interval_start, control.interval_start)
            and np.array_equal(run.interval_end, control.interval_end)):
        raise ValueError("Experiment/control six-hour endpoints differ")
    if (not np.all(np.diff(times) == SIX_HOURS)
            or not np.array_equal(control.interval_end, times)
            or not np.all(control.interval_end.values - control.interval_start.values == SIX_HOURS)):
        raise ValueError("Control-relative indices require contiguous native six-hour intervals")
    quadrature = xr.DataArray(
        mechanism._quadrature(control.region_pres.values * 100)[:, None]
        * mechanism._quadrature(np.deg2rad(control.region_latitude.values))[None, :],
        dims=("region_pres", "region_latitude"),
        coords=dict(region_pres=control.region_pres, region_latitude=control.region_latitude),
    )
    weights = control.mass_weight * quadrature
    if not np.isfinite(weights.values).all() or not (weights.values > 0).all():
        raise ValueError("Control regional mass weights must be positive and finite")
    dimensions = ("region_pres", "region_latitude")
    fields = {}
    for name in INDICES:
        difference = run[name] - control[name]
        if name in {"zonal_wind", "a_ep"}:
            difference = (difference * weights).sum(dimensions, skipna=False) / weights.sum(dimensions, skipna=False)
        fields[name] = difference.mean("hemisphere", skipna=False)
        if not np.isfinite(fields[name].values).all():
            raise ValueError(f"Nonfinite control-relative {name}")
    result = xr.Dataset(fields)
    result["interval_start"], result["interval_end"] = control.interval_start, control.interval_end
    for name, units in zip(INDICES, UNITS):
        result[name].attrs.update(units=units, anomaly_reference="same-time corresponding control spin-up")
    result.precipitation.attrs["definition"] = "preceding six-hour accumulation difference / exact duration"
    result.attrs.update(anomaly_reference="same-time corresponding control spin-up",
                        regional_weighting="instantaneous control rho*(-dz/dp)*cos(phi); equal hemisphere means")
    return result


def native_timescale(table, experiment, response):
    """Select peak nudging timescales at the exact native sample timestamps."""
    records = table.set_index("Time")[experiment]
    if records.index.has_duplicates or np.isnan(records).any() or (records <= 0).any():
        raise ValueError("Forcing records require unique dates and positive timescales; infinity denotes disabled nudging")
    selected = records.reindex(pd.DatetimeIndex(response.Time.values)).to_numpy()
    if np.isnan(selected).any():
        raise ValueError(f"Forcing CSV lacks exact six-hourly {experiment} values for retained samples")
    return xr.DataArray(selected, dims="Time", coords=dict(Time=response.Time),
                        attrs=dict(units="day", definition="native peak nudging timescale; infinity denotes disabled nudging"))


def restart_timescale(spec, response):
    """Read each completed run's recorded settings over its actual interval."""
    dates = response.Time.values
    timescales = np.full(len(dates), np.nan)
    paths = []
    root = Path(spec["directory"])
    for path in sorted(root.glob("restart.*.nc")):
        last = pd.to_datetime(path.name, format="restart.%Y-%m-%d_%H.%M.%S.nc").to_datetime64()
        if last < dates[0]:
            continue
        with xr.open_dataset(path, decode_cf=False) as source:
            first = pd.Timestamp(source.attrs["config_start_time"].replace("_", " ")).to_datetime64()
            selected = (dates > first) & (dates <= last)
            if not selected.any():
                continue
            regions = regions_from_attributes(source.attrs, spec["simulation"], str(path))
        variable = "u" if spec["simulation"] == "nudged-u" else "theta"
        timescale = next((region.timescale_days for region in regions if region.variable == variable), np.inf)
        if np.any(~np.isnan(timescales[selected]) & (timescales[selected] != timescale)):
            raise ValueError(f"Conflicting restart nudging settings for {spec['simulation']}")
        timescales[selected] = timescale
        paths.append(str(path))
    result = xr.DataArray(timescales, dims="Time", coords=dict(Time=response.Time),
                          attrs=dict(units="day", definition="restart-recorded peak nudging timescale; infinity denotes disabled nudging",
                                     source_paths=paths))
    if np.isnan(result).any():
        progress(f"{spec['simulation']}: restart settings lack complete forcing coverage; unobserved timescale remains missing")
    return result


def sustained_onsets(response, thresholds, sustain_days):
    """Require consecutive same-sign exceedances; report boundary censoring."""
    if not np.isfinite(sustain_days) or sustain_days < 0.5:
        raise ValueError("Onset detection requires at least two sustained six-hour samples (0.5 days)")
    sustain_samples = int(np.ceil(sustain_days * (ONE_DAY / SIX_HOURS)))
    rows = []
    for name, threshold in zip(INDICES, thresholds):
        if not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("Onset thresholds must be positive finite physical-unit amplitudes")
        values = response[name].values
        row = dict(index=name, units=response[name].attrs["units"], threshold=threshold,
                   sustain_days=sustain_days, sustain_samples=sustain_samples,
                   onset_time=pd.NaT, confirmation_time=pd.NaT,
                   direction=0, status="no sustained crossing")
        for first in range(len(values) - sustain_samples + 1):
            window = values[first:first + sustain_samples]
            direction = 1 if np.all(window > threshold) else -1 if np.all(window < -threshold) else 0
            if direction:
                row.update(onset_time=response.Time.values[first],
                           confirmation_time=response.Time.values[first + sustain_samples - 1],
                           direction=direction, status="at or before first available sample" if first == 0 else "sustained crossing")
                break
        rows.append(row)
    return pd.DataFrame(rows)


def plot_responses(responses, onsets, experiments=None):
    """Show forcing and unsmoothed native responses separately for U and T."""
    if experiments is None:
        _, experiments = configured_inputs()
    fig, axes = plt.subplots(5, len(responses), figsize=(6 * len(responses), 11),
                             sharex="col", squeeze=False, constrained_layout=True)
    for column, (experiment, response) in enumerate(responses.items()):
        colour = simulation_colour(experiments[experiment])
        dates = pd.DatetimeIndex(response.Time.values)
        axes[0, column].step(dates, response.nudging_timescale, where="pre", color=colour)
        if np.isnan(response.nudging_timescale).any():
            axes[0, column].text(0.5, 0.5, "Incomplete restart forcing metadata", ha="center", transform=axes[0, column].transAxes)
        if np.isinf(response.nudging_timescale).any():
            axes[0, column].text(0.5, 0.9, "Nudging off: infinite timescale", ha="center", transform=axes[0, column].transAxes)
        label = config.ALL_SIMULATION_LABELS[experiments[experiment]]
        axes[0, column].set(title=f"{label} − control", ylabel=r"Nudging timescale $\tau$ (days)")
        for number, (name, label) in enumerate(zip(INDICES, LABELS), start=1):
            axis = axes[number, column]
            axis.axhline(0, color="0.6", linewidth=0.7)
            axis.plot(dates, response[name], color=colour, marker=".", linewidth=1.2)
            axis.set_ylabel(label)
            if experiment in onsets:
                record = onsets[experiment].set_index("index").loc[name]
                for sign in (-1, 1):
                    axis.axhline(sign * record.threshold, color="0.75", linestyle=":", linewidth=0.7)
                if record.status == "sustained crossing":
                    axis.axvline(pd.Timestamp(record.onset_time), color=colour, linestyle="--", linewidth=0.8)
        axes[-1, column].set_xlabel("Spin-up date (native six-hour samples; P over preceding interval)")
    fig.suptitle("Transient response ordering: same-time control anomalies")
    return fig


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start", default=config.SPINUP_START, help="Override the configured first six-hour interval start")
    parser.add_argument("--end", default=config.SPINUP_END, help="Override the configured last six-hour interval end")
    parser.add_argument("--forcing-csv", type=Path, help="Optional exact six-hourly nudging timescales in days; otherwise use restart settings")
    parser.add_argument("--thresholds", type=float, nargs=4, metavar=("U", "A_EP", "PSI_HC", "P"),
                        help="Optional physical-unit onset amplitudes, in m/s, m/s2, kg/s, mm/day")
    parser.add_argument("--sustain-days", type=float, default=3,
                        help="Same-sign persistence in days, rounded up to six-hour samples (minimum 0.5)")
    parser.add_argument("--cache-root", type=Path, default=Path(diagnostic_cache_dir("circulation", "spinup", "simulations")))
    parser.add_argument("--output-dir", default=diagnostic_output_dir("/home/cwp29/tmp/mpas/crf_on/final/circulation_spinup"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--figures-only", action="store_true")
    mode.add_argument("--rebuild-cache", action="store_true")
    args = parser.parse_args(argv)
    if (not np.isfinite(args.sustain_days) or args.sustain_days < 0.5
            or (args.thresholds and not all(np.isfinite(value) and value > 0 for value in args.thresholds))):
        parser.error("Use at least two sustained six-hour samples (0.5 days) and positive finite onset thresholds")
    return args


def main(argv=None):
    args = parse_args(argv)
    inputs, experiments = configured_inputs()
    histories = {name: prepare_indices(name, spec, args) for name, spec in inputs.items()}
    forcing = pd.read_csv(args.forcing_csv, parse_dates=["Time"]) if args.forcing_csv else None
    responses, onsets = {}, {}
    saver = FigureSaver(args.output_dir)
    output = Path(saver.save_loc)
    notes = [
        "Spin-up timing only. No equilibrium amplitudes, climatological statistics, or causal inference.",
        f"Wind and A_EP use {mechanism.REGION_DESCRIPTION}; instantaneous corresponding-control mass weights.",
        "HC uses Figure 15's signed 5-15 degree, 150-850 hPa pressure-area mean; reverse the SH sign.",
        "Precipitation uses Figure 15's 10S-10N cosine-area mean and bucket-corrected exact six-hour endpoints.",
        "Subtract the corresponding control at matching timestamps; never centre on a spin-up time mean.",
        "Plot unsmoothed native six-hour samples. Nudging timescales describe relaxation at the kernel centre, in days.",
        "Use restart-recorded settings within each completed run; no undocumented interpolation or ramp is inferred.",
        "Six-hour sampling limits ordering resolution. P covers the preceding six-hour interval; TEM is instantaneous.",
        "Missing edge coverage remains unobserved.",
    ]
    for experiment, simulation in experiments.items():
        response = control_response(histories[experiment], histories["control"])
        forcing_simulation = config.SPINUP_NUDGING_TIMESCALE_SOURCES.get(simulation, simulation)
        forcing_experiment = circulation.COMPARISON_NAMES[forcing_simulation][0]
        response["nudging_timescale"] = (native_timescale(forcing, forcing_experiment, response) if forcing is not None
                                          else restart_timescale(inputs[forcing_experiment], response))
        response.nudging_timescale.attrs["source_simulation"] = forcing_simulation
        if forcing_simulation != simulation:
            notes.append(f"{simulation}: use the {forcing_simulation} timescale history, as configured.")
        responses[experiment] = response
        write_csv_atomic(response.to_dataframe().reset_index(), output / f"spinup_{experiment}_6hourly_responses.csv", index=False)
        notes.append(f"{experiment}: reference control; coverage {response.interval_start.values[0]} to {response.interval_end.values[-1]}.")
        if args.thresholds:
            onsets[experiment] = sustained_onsets(response, args.thresholds, args.sustain_days)
            write_csv_atomic(onsets[experiment], output / f"spinup_{experiment}_onsets.csv", index=False)
            pair = onsets[experiment].set_index("index").loc[["hc_streamfunction", "precipitation"]]
            ordering = "HC/P ordering remains unresolved"
            if (pair.status == "sustained crossing").all():
                hc, precipitation = pd.to_datetime(pair.onset_time)
                ordering = "HC and P cross at the same six-hour sample" if hc == precipitation else "HC crosses before P" if hc < precipitation else "P crosses before HC"
            notes.append(f"{experiment.upper()}: {ordering}; this describes threshold crossings, not causality.")
    notes.append(f"Onsets require {int(np.ceil(args.sustain_days * 4))} consecutive same-sign six-hour exceedances ({args.sustain_days:g}-day interval coverage).")
    notes.append("A crossing in the first available sample is left-censored; no onset is assigned to one noisy timestep.")
    if not args.thresholds:
        notes.append("No automatic onsets: supply --thresholds to define descriptive physical-unit amplitude criteria.")
    write_json_file(output / "spinup_analysis.json", dict(inputs=inputs, start=args.start, end=args.end,
                    forcing_csv=str(args.forcing_csv.resolve()) if args.forcing_csv else None,
                    forcing_sources={name: response.nudging_timescale.attrs.get("source_paths", []) for name, response in responses.items()},
                    forcing_simulations={name: response.nudging_timescale.attrs["source_simulation"] for name, response in responses.items()},
                    sampling_hours=6,
                    thresholds=args.thresholds,
                    sustain_days=args.sustain_days, notes=notes))
    figure = plot_responses(responses, onsets, experiments)
    saver.save(figure, "spinup_response_ordering", number=1)
    plt.close(figure)
    progress("\n".join(notes))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
