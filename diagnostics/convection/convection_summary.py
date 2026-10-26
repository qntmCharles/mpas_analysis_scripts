#!/usr/bin/env python3
"""Render headline convection and mean environmental responses."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

_ANALYSIS_DIR = Path(__file__).resolve().parents[2]
if str(_ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(_ANALYSIS_DIR))

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt, ticker
from matplotlib.lines import Line2D

from diagnostics.analysis_config import (
    ANALYSIS_END, ANALYSIS_START,
    ALL_SIMULATION_PUBLICATION_LABELS, simulation_colour,
)
from diagnostics.cache_support import write_csv_atomic, write_json_file
from diagnostics.convection.convection_summary_products import (
    EXPERIMENTS, SIMULATIONS, prepare_inputs, selected_rain_matched,
)
from diagnostics.diagnostic_helpers import (
    FigureSaver, diagnostic_output_dir, progress,
)
from diagnostics.mean_state.publication_profiles import (
    ENVIRONMENT_SPECS, PROFILE_CAPTION, PROFILE_VARIANTS, environment_profile_figure,
)
from diagnostics.ttl import TTL_BASE_PRESSURE_HPA, TTL_TOP_PRESSURE_HPA


STEMS = {
    1: "convection_precipitation_moisture_responses",
    2: "mean_environment_utls_profiles",
}
SURFACE_SPECS = (
    ("precipitation", "Precipitation rate", r"mm day$^{-1}$"),
    ("olr", "OLR", r"W m$^{-2}$"),
    ("precipitable_water", "Precipitable water", r"kg m$^{-2}$"),
)

def experiment_handles():
    return [Line2D(
        [0], [0], color=simulation_colour(sim), lw=1.7,
        label=f"{ALL_SIMULATION_PUBLICATION_LABELS[sim]} − control",
    ) for sim in EXPERIMENTS]


def pressure_axis(axis, *, ylabel=False, lower=400.0):
    axis.axhspan(TTL_TOP_PRESSURE_HPA, TTL_BASE_PRESSURE_HPA,
                color="0.8", alpha=0.25, zorder=0)
    ticks = [value for value in [1000, 850, 700, 500, 400, 300, 200, 150, 100, 70, 50] if value <= lower]
    axis.set_yscale("log")
    axis.set_ylim(lower, 50.0)
    axis.yaxis.set_major_locator(ticker.FixedLocator(ticks))
    axis.yaxis.set_major_formatter(ticker.FixedFormatter([str(v) for v in ticks]))
    axis.yaxis.set_minor_formatter(ticker.NullFormatter())
    axis.grid(alpha=0.17)
    if ylabel:
        axis.set_ylabel("Pressure (hPa)")
    axis.axvline(0, color="0.55", lw=0.7, zorder=1)


def surface_axis(axis, products, field, title, units, panel):
    for sim in EXPERIMENTS:
        response = products["surface"][sim].sel(field=field)
        latitude = response.absolute_latitude
        colour = simulation_colour(sim)
        axis.plot(latitude, response.response, color=colour, lw=1.7)
        axis.fill_between(latitude, response.ci_low, response.ci_high,
                          color=colour, alpha=0.13, lw=0)
    axis.axhline(0, color="0.55", lw=0.7)
    axis.axvspan(0, 10, color="0.8", alpha=0.12, zorder=0)
    axis.set_xlim(0, 40)
    axis.set_xticks([0, 10, 20, 30, 40])
    axis.set_xlabel("Latitude (°)")
    axis.set_ylabel(f"Difference ({units})")
    axis.set_title(f"({panel}) {title}", loc="left")
    axis.grid(alpha=0.17)


def cloud_contribution_axis(axis, products):
    pressure = products["cloud_profiles"].sel(latitude_domain="tropical", simulation="control").pressure_hpa
    for sim in EXPERIMENTS:
        for contribution, style in (("total", "-"), ("nonprecipitating", "--"), ("deep_convective", ":")):
            values = products["cloud_response"][sim].sel(contribution=contribution)
            colour = simulation_colour(sim)
            axis.fill_betweenx(pressure, 100 * values.ci_low, 100 * values.ci_high,
                               color=colour, alpha=0.10, lw=0)
            axis.plot(100 * values.response, pressure, color=colour, ls=style, lw=1.6)
    pressure_axis(axis, ylabel=True)
    axis.set_xlabel("Cloud fraction difference (% pts)")
    axis.set_title("(d) Cloud fraction profiles", loc="left")
    axis.legend(handles=[Line2D([0], [0], color="0.25", ls=style, label=label)
                         for label, style in (("Total", "-"), ("Non-precipitating", "--"),
                                              ("Deep-convective", ":"))],
                fontsize=7, frameon=False, loc="lower right")


def rain_matched_axis(axis, products):
    for sim in EXPERIMENTS:
        rows = selected_rain_matched(products["rain_matched"], sim)
        if rows.empty:
            raise ValueError(f"No supported rain-matched cloud-top bins for {sim}")
        x = rows.rain_bin_centre_mm_h.to_numpy(float)
        colour = simulation_colour(sim)
        axis.plot(x, rows.value, color=colour, lw=1.7, marker="o", ms=3)
        axis.fill_between(x, rows.ci_low, rows.ci_high, color=colour, alpha=0.13, lw=0)
    axis.axhline(0, color="0.55", lw=0.7)
    axis.set_xscale("log")
    axis.set_xlim(0.5, 11.5)
    axis.set_xticks([0.5, 1, 2, 4, 10.8], ["0.5", "1", "2", "4", "8+"])
    axis.xaxis.set_minor_formatter(ticker.NullFormatter())
    axis.invert_yaxis()
    axis.set_xlabel(r"Precipitation rate (mm h$^{-1}$)")
    axis.set_ylabel("Cloud-top pressure difference (hPa)")
    axis.set_title("(e) Cloud depth at matched precip. rate", loc="left")
    axis.grid(alpha=0.17)


def convection_figure(products):
    figure = plt.figure(figsize=(8.0, 6.4), layout="constrained")
    rows = figure.add_gridspec(2, 1, height_ratios=[1.0, 1.25])
    latitude = rows[0].subgridspec(1, 3)
    convection = rows[1].subgridspec(1, 2, width_ratios=[1.15, 1.0])
    for column, spec in enumerate(SURFACE_SPECS):
        surface_axis(figure.add_subplot(latitude[0, column]), products, *spec, chr(97 + column))
    cloud_contribution_axis(figure.add_subplot(convection[0, 0]), products)
    rain_matched_axis(figure.add_subplot(convection[0, 1]), products)
    figure.legend(handles=experiment_handles(), loc="outside lower center", ncol=2,
                  frameon=False, fontsize=8, title="Bands: 95% intervals", title_fontsize=7)
    return figure


def environment_figure(products, *, include_lpo=False, lower=850.0):
    return environment_profile_figure(products, include_lpo=include_lpo, lower=lower)


CAPTIONS = {
    1: """Nudging responses in convection, precipitation, and column moisture.
Blue denotes U minus control; red denotes T minus control.
Full-record means cover 5 May–21 November 2000 without event conditioning.
Panels (a–c) show zonal precipitation rate, outgoing longwave radiation (OLR), and precipitable water (PW).
These latitude profiles average paired hemispheres; light shading marks 10 degrees S–10 degrees N.
Rainfall uses bucket-corrected accumulation and six-hour intervals ending at each archived timestamp.
The first interval includes its preceding accumulation endpoint.
OLR and PW use the same interval-ending times.
Bands show marginal 95% intervals from independent complete 20-day blocks within each simulation.
These intervals do not provide field-wide significance tests.

Panel (d) shows domain-mean cloud-fraction contributions from high-cloud columns over 20 degrees S–20 degrees N.
Solid curves show total high cloud; dashed curves show non-precipitating cloud; dotted curves show deep-convective cloud.
Total includes the stratiform contribution, so the two displayed components do not sum to total.
A high-cloud column has cloud fraction >=0.05 and cloud-top pressure <=440 hPa.
Deep-convective columns contain an OLR-derived PyFLEXTRKR cold-core pixel (Tb <225 K).
Non-precipitating columns contain no cold-core pixel and have surface rain <0.5 mm/hour.
All contributions use the complete tropical area-time denominator, with zero contribution outside the selected class.
Differences use common native heights and the control mean pressure for display.
Gray shading marks the 150–70 hPa TTL reference.
Cloud bands resample complete 20-day numerator and denominator blocks jointly across all classes within each climate.

Panel (e) compares total high-cloud tops within identical bins of surface precipitation rate.
It includes bins >=0.5 mm/hour with at least four complete blocks per simulation.
The open upper bin represents rainfall >=8 mm/hour.
The inverted pressure-response axis places higher cloud tops upward.
Its intervals retain the existing high-cloud owner's independent 20-day estimates.
OLR responds to atmospheric temperature and cloud radiation as well as convective activity.

The later circulation figures diagnose the momentum and moisture-transport pathways.
""",
    2: PROFILE_CAPTION,
}


def export_tables(products, target):
    def save_table(table, name):
        write_csv_atomic(table, target / name, index=False)

    if "surface" in products:
        rows = [data.to_dataframe().reset_index().assign(simulation=sim)
                for sim, data in products["surface"].items()]
        baseline = products["surface_absolute"]["control"].surface_mean.to_dataframe(
            name="control_mean"
        ).reset_index()
        table = pd.concat(rows, ignore_index=True).merge(baseline, on=["field", "absolute_latitude"])
        table["units"] = table.field.map({
            "precipitation": "mm day-1", "olr": "W m-2", "precipitable_water": "kg m-2",
        })
        save_table(table, "surface_latitude_responses.csv")
    if "cloud_budget" in products:
        save_table(products["cloud_budget"], "high_cloud_coverage_summary.csv")
        save_table(pd.concat([
            selected_rain_matched(products["rain_matched"], sim) for sim in EXPERIMENTS
        ], ignore_index=True), "rain_matched_cloud_depth_summary.csv")
        cloud = products["cloud_profiles"].sel(latitude_domain="tropical")
        rows = []
        for sim in EXPERIMENTS:
            table = (100 * products["cloud_response"][sim]).to_dataframe().reset_index()
            table["control_pressure_hpa"] = np.tile(
                cloud.pressure_hpa.sel(simulation="control").values,
                products["cloud_response"][sim].sizes["contribution"],
            )
            rows.append(table.rename(columns={"z": "height_m"}).assign(
                simulation=sim, units="percentage points"
            ))
        save_table(pd.concat(rows, ignore_index=True), "vertical_cloud_fraction_responses.csv")
    if "environment" in products:
        rows = []
        for sim, variables in products["environment"].items():
            for name, data in variables.items():
                if name not in {spec[0] for spec in ENVIRONMENT_SPECS}:
                    continue
                pressure = products["environment_absolute"][sim].pressure_hpa
                table = data.to_dataframe().reset_index()
                table["pressure_hpa"] = pressure.values
                table["units"] = {"temperature": "K", "static_stability": "s-2", "zonal_wind": "m s-1"}[name]
                rows.append(table.rename(columns={"z": "height_m"}).assign(simulation=sim, metric=name))
        save_table(pd.concat(rows, ignore_index=True), "tropical_environment_profiles.csv")
    if "environment_lpo" in products:
        rows = []
        for sim, profile in products["environment_lpo"].items():
            for name, _title, _units, _scale in ENVIRONMENT_SPECS:
                table = profile[[name, f"{name}_ci_low", f"{name}_ci_high", "pressure_hpa"]].to_dataframe().reset_index()
                table = table.rename(columns={name: "mean", f"{name}_ci_low": "ci_low", f"{name}_ci_high": "ci_high"})
                rows.append(table.assign(simulation=sim, metric=name, event_count=profile.attrs["event_count"],
                                         units=profile[name].attrs["units"]))
        save_table(pd.concat(rows, ignore_index=True), "lpo_conditioned_environment_profiles.csv")
    if "ascent_intervals" in products:
        rows = []
        for sim, interval in products["ascent_intervals"].items():
            table = interval.to_dataframe().reset_index()
            table["pressure_hpa"] = products["ascent_absolute"][sim].pressure_hpa.values
            rows.append(table.rename(columns={"z": "height_m"}).assign(
                simulation=sim, metric="vertical_velocity", units="m s-1",
            ))
        save_table(pd.concat(rows, ignore_index=True), "tropical_ascent_profiles.csv")
    if "ascent_lpo_absolute" in products:
        rows = []
        for sim, profile in products["ascent_lpo_absolute"].items():
            table = profile[["vertical_velocity", "vertical_velocity_ci_low", "vertical_velocity_ci_high",
                             "pressure_hpa"]].to_dataframe().reset_index()
            table = table.rename(columns={"vertical_velocity": "mean", "vertical_velocity_ci_low": "ci_low",
                                          "vertical_velocity_ci_high": "ci_high"})
            rows.append(table.assign(simulation=sim, event_count=profile.attrs["event_count"], units="m s-1"))
        save_table(pd.concat(rows, ignore_index=True), "lpo_conditioned_ascent_profiles.csv")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--figures", type=int, nargs="+", choices=(1, 2), default=[1, 2])
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--figures-only", action="store_true",
                       help="Read current products without opening raw histories.")
    modes.add_argument("--rebuild-cache", action="store_true",
                       help="Rebuild compact mean-state and LPO profile products.")
    parser.add_argument("--output-dir", type=Path,
                        help="Override the configured figure output directory.")
    parser.add_argument("--version", help="Override the configured figure release name.")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    products = prepare_inputs(args.figures, figures_only=args.figures_only, rebuild=args.rebuild_cache)
    output = args.output_dir or Path(diagnostic_output_dir(
        "/home/cwp29/tmp/mpas/crf_on/final/convection_summary/"
    ))
    saver = FigureSaver(output, figures_to_run=args.figures, version=args.version)
    target = Path(saver.save_loc)
    with plt.rc_context({"font.size": 8, "axes.titlesize": 8.5,
                         "axes.labelsize": 8, "xtick.labelsize": 7,
                         "ytick.labelsize": 7, "pdf.fonttype": 42, "ps.fonttype": 42,
                         "hatch.color": "0.65", "hatch.linewidth": 0.3}):
        for number in args.figures:
            variants = PROFILE_VARIANTS if number == 2 else (("", False, 850.0, CAPTIONS[number]),)
            for suffix, include_lpo, lower, caption in variants:
                figure = (convection_figure(products) if number == 1
                          else environment_figure(products, include_lpo=include_lpo, lower=lower))
                path = Path(saver.save(figure, STEMS[number] + suffix, number=number))
                figure.savefig(path.with_suffix(".pdf"))
                path.with_name(path.stem + "_caption.txt").write_text(caption, encoding="utf-8")
                plt.close(figure)
    export_tables(products, target)
    write_json_file(target / "summary_provenance.json", {
        "producer": str(Path(__file__).relative_to(_ANALYSIS_DIR)),
        "simulations": list(SIMULATIONS), "figures": args.figures,
        "analysis_start": ANALYSIS_START, "analysis_end": ANALYSIS_END,
        "sources": products["provenance"],
        "profile_variants": [
            {"suffix": suffix, "include_lpo": include_lpo, "lower_pressure_hpa": lower,
             "upper_pressure_hpa": 50.0}
            for suffix, include_lpo, lower, _caption in PROFILE_VARIANTS
        ] if 2 in args.figures else [],
    })
    progress(f"Convection publication summaries ready in {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
