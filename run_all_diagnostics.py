#!/usr/bin/env python3
"""Run the MPAS-A diagnostic scripts one at a time.

The diagnostics in the family subdirectories below ``diagnostics/`` execute
their work at module top level, so this helper intentionally launches each file
in a fresh Python process.  That keeps memory use, warnings, and failures
isolated between diagnostics.
"""

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ANALYSIS_DIR = Path(__file__).resolve().parent
DIAGNOSTICS_DIR = ANALYSIS_DIR / "diagnostics"
DEFAULT_LOG_DIR = ANALYSIS_DIR / "diagnostic_logs"
DEFAULT_SIMULATION_ROOT = Path(
    os.environ.get(
        "MPAS_SIMULATION_ROOT",
        "/store/atmos-adk33/cwp29/mpas-a/data/trop_ref/crf_on",
    )
)
DEFAULT_DIAGNOSTIC_CACHE_ROOT = Path(
    os.environ.get("MPAS_DIAGNOSTIC_CACHE_ROOT", str(DEFAULT_SIMULATION_ROOT))
)
DIAGNOSTIC_OUTPUT_ROOT_ENV = "MPAS_DIAGNOSTIC_OUTPUT_ROOT"
MODULE_SEARCH_DIRS = [
    ANALYSIS_DIR,
    ANALYSIS_DIR.parent,
]

DEFAULT_SCRIPT_ORDER = [
    "mean_state/mean_state_diagnostics.py",
    "mean_state/energy_budget.py",
    "mean_state/vapour_transport_budget.py",
    "mean_state/thermal_wind_scaled_response.py",
    "mean_state/matched_pattern_response.py",
    "mean_state/jet_variability_diagnostics.py",
    "convection/refined_high_cloud_classification_diagnostics.py",
    "convection/lpt_precipitation_envelope_diagnostics.py",
    "waves/equatorial_wave_diagnostics.py",
    "convection/convective_pathway_diagnostics.py",
    "convection/convective_lifecycle_composite_diagnostics.py",
    "significance/significance_bootstrap_diagnostics.py",
]

QC_CAPABLE_SCRIPTS = {
    "convection/refined_high_cloud_classification_diagnostics.py",
    "convection/lpt_precipitation_envelope_diagnostics.py",
}

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all configured diagnostic scripts sequentially.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List the diagnostics that would be run, then exit.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands without running them.",
    )
    parser.add_argument(
        "--only",
        nargs="+",
        metavar="NAME",
        help=(
            "Run only these diagnostics. Names may be stems such as "
            "'mean_state_diagnostics', filenames, or family-relative paths "
            "such as 'mean_state/mean_state_diagnostics.py'."
        ),
    )
    parser.add_argument(
        "--skip",
        nargs="+",
        metavar="NAME",
        default=[],
        help="Skip these diagnostics, using the same name format as --only.",
    )
    parser.add_argument(
        "--continue-on-error",
        "-k",
        action="store_true",
        help="Continue with later diagnostics if one fails.",
    )
    parser.add_argument(
        "--qc",
        action="store_true",
        help=(
            "Request validation figures from selected diagnostics that support "
            "--qc (currently refined high-cloud classification and convective "
            "LPT precipitation-envelope tracking)."
        ),
    )
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python executable to use for each diagnostic.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=DEFAULT_LOG_DIR,
        help="Directory for per-diagnostic combined stdout/stderr logs.",
    )
    parser.add_argument(
        "--no-log",
        action="store_true",
        help="Disable per-diagnostic log files.",
    )
    parser.add_argument(
        "--figure-root",
        "--figure-dir",
        dest="figure_root",
        type=Path,
        help=(
            "Override the diagnostics' default figure/output root. Each script "
            "keeps its usual relative subdirectory under this root."
        ),
    )
    return parser.parse_args()


def normalise_name(name: str) -> str:
    """Normalise a diagnostic stem, filename, or family-relative path."""
    normalised = Path(name).as_posix().removeprefix("diagnostics/")
    return normalised if normalised.endswith(".py") else f"{normalised}.py"


def discover_diagnostic_scripts() -> list[str]:
    """Return the fixed workflow, failing if a configured script is missing."""
    missing = [
        script for script in DEFAULT_SCRIPT_ORDER
        if not (DIAGNOSTICS_DIR / script).is_file()
    ]
    if missing:
        raise SystemExit(f"Configured diagnostic scripts are missing: {', '.join(missing)}")
    return list(DEFAULT_SCRIPT_ORDER)


def resolve_script_name(name: str, available: list[str]) -> str:
    """Resolve a family-relative path or an unambiguous basename/stem."""
    normalised = normalise_name(name)
    if normalised in available:
        return normalised

    basename = Path(normalised).name
    matches = [script for script in available if Path(script).name == basename]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        choices = ", ".join(matches)
        raise SystemExit(f"Ambiguous diagnostic {name!r}; choose one of: {choices}")
    return normalised


def selected_scripts(args: argparse.Namespace) -> list[str]:
    diagnostic_scripts = discover_diagnostic_scripts()
    available = set(diagnostic_scripts)
    selected = (
        [resolve_script_name(name, diagnostic_scripts) for name in args.only]
        if args.only
        else list(diagnostic_scripts)
    )
    skipped = {resolve_script_name(name, diagnostic_scripts) for name in args.skip}

    unknown = sorted((set(selected) | skipped) - available)
    if unknown:
        valid = ", ".join(diagnostic_scripts)
        raise SystemExit(f"Unknown diagnostic(s): {', '.join(unknown)}\nValid diagnostics: {valid}")

    return [script for script in selected if script not in skipped]


def print_script_list(scripts: list[str]) -> None:
    for index, script in enumerate(scripts, start=1):
        print(f"{index:2d}. {script}")


def child_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("MPAS_SIMULATION_ROOT", str(DEFAULT_SIMULATION_ROOT))
    env.setdefault("MPAS_DIAGNOSTIC_CACHE_ROOT", str(DEFAULT_DIAGNOSTIC_CACHE_ROOT))
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.getuid()}"))
    env.setdefault("XDG_CACHE_HOME", str(DEFAULT_DIAGNOSTIC_CACHE_ROOT / f"xdg-cache-{os.getuid()}"))
    if args.figure_root:
        env[DIAGNOSTIC_OUTPUT_ROOT_ENV] = str(args.figure_root.expanduser().resolve())

    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    Path(env["XDG_CACHE_HOME"]).mkdir(parents=True, exist_ok=True)

    pythonpath = [str(path) for path in MODULE_SEARCH_DIRS]
    existing_pythonpath = env.get("PYTHONPATH")
    if existing_pythonpath:
        for path in existing_pythonpath.split(os.pathsep):
            if path and path not in pythonpath:
                pythonpath.append(path)

    env["PYTHONPATH"] = os.pathsep.join(pythonpath)
    return env


def run_script(script: str, args: argparse.Namespace) -> int:
    script_path = DIAGNOSTICS_DIR / script
    command = [args.python, "-u", str(script_path)]
    if args.qc and script in QC_CAPABLE_SCRIPTS:
        command.append("--qc")
    command_text = shlex.join(command)
    env = child_environment(args)

    start = datetime.now()
    print(f"\n[{start:%Y-%m-%d %H:%M:%S}] Starting {script}", flush=True)
    print(f"Command: {command_text}", flush=True)

    log_file = None
    if args.log_dir and not args.no_log:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_name = script.removesuffix(".py").replace("/", "__")
        log_path = args.log_dir / f"{log_name}.log"
        log_file = log_path.open("w", encoding="utf-8")
        log_file.write(f"[{start:%Y-%m-%d %H:%M:%S}] Starting {script}\n")
        log_file.write(f"Command: {command_text}\n\n")
        log_file.flush()
        print(f"Log: {log_path}", flush=True)

    return_code = 1
    try:
        process = subprocess.Popen(
            command,
            cwd=ANALYSIS_DIR,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            if log_file:
                log_file.write(line)

        return_code = process.wait()
    finally:
        end = datetime.now()
        elapsed = end - start
        status = "finished" if return_code == 0 else f"failed with exit code {return_code}"
        end_line = f"[{end:%Y-%m-%d %H:%M:%S}] {script} {status} after {elapsed}"
        print(end_line, flush=True)
        if log_file:
            log_file.write(f"\n{end_line}\n")
            log_file.flush()
            log_file.close()

    return return_code


def main() -> int:
    args = parse_args()
    scripts = selected_scripts(args)

    if args.list:
        print_script_list(scripts)
        return 0

    if not scripts:
        print("No diagnostics selected.")
        return 0

    print("Diagnostics to run:")
    print_script_list(scripts)
    if args.no_log:
        print("Logging disabled (--no-log).")
    elif args.log_dir:
        print(f"Logs will be written to: {args.log_dir}")
    if args.figure_root:
        print(f"Figures/output will be written under: {args.figure_root.expanduser().resolve()}")

    if args.dry_run:
        print("\nDry run only; no diagnostics were executed.")
        for script in scripts:
            command_parts = [args.python, "-u", str(DIAGNOSTICS_DIR / script)]
            if args.qc and script in QC_CAPABLE_SCRIPTS:
                command_parts.append("--qc")
            command = shlex.join(command_parts)
            if args.no_log or not args.log_dir:
                print(command)
            else:
                log_name = script.removesuffix(".py").replace("/", "__")
                log_path = args.log_dir / f"{log_name}.log"
                print(f"{command}  # log: {log_path}")
        return 0

    failures: list[tuple[str, int]] = []
    for script in scripts:
        return_code = run_script(script, args)
        if return_code != 0:
            failures.append((script, return_code))
            if not args.continue_on_error:
                break

    if failures:
        print("\nDiagnostic failures:")
        for script, return_code in failures:
            print(f"  {script}: exit code {return_code}")
        return failures[0][1]

    print("\nAll selected diagnostics completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
