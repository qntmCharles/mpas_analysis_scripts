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
import queue
import shlex
import shutil
import socket
import subprocess
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from diagnostics.analysis_config import (
    DEFAULT_FIGURE_ROOT,
    DEFAULT_SIMULATION_PROFILE,
    DIAGNOSTIC_FIGURE_VERSION_ENV,
    DIAGNOSTIC_OUTPUT_ROOT_ENV,
    SIMULATION_PROFILE_ENV,
    SIMULATION_PROFILES,
    SIMULATION_ROOT,
)


ANALYSIS_DIR = Path(__file__).resolve().parent
DIAGNOSTICS_DIR = ANALYSIS_DIR / "diagnostics"
DEFAULT_LOG_DIR = ANALYSIS_DIR / "diagnostic_logs"
LOG_HEARTBEAT_SECONDS = 300
DEFAULT_SIMULATION_ROOT = SIMULATION_ROOT
MODULE_SEARCH_DIRS = [
    ANALYSIS_DIR,
    ANALYSIS_DIR.parent,
]

DEFAULT_SCRIPT_ORDER = [
    "mean_state/mean_state_diagnostics.py",
    "mean_state/energy_budget.py",
    "mean_state/vapour_transport_budget.py",
    "mean_state/thermal_wind.py",
    "mean_state/circulation_diagnostics.py",
    "mean_state/jet_variability_diagnostics.py",
    "convection/refined_high_cloud_classification_diagnostics.py",
    "convection/lpo_tracking.py",
    "waves/equatorial_wave_diagnostics.py",
    "convection/lpo_lifecycle_analysis.py",
]

OPTIONAL_SCRIPT_ORDER = [
    "mean_state/circulation_spinup.py",
    "convection/convection_summary.py",
]

# Complete publication workflow for the convection/LPO analysis.  The ordering
# is a dependency contract, not just a preferred presentation order:
#
# * tracking consumes the shared high-cloud classification;
# * wave Figures 14 and 15 consume the saved primary LPO tracks;
# * lifecycle consumes wave Figures 13 and 14 products.
CONVECTION_WORKFLOW_ORDER = [
    "convection/refined_high_cloud_classification_diagnostics.py",
    "convection/lpo_tracking.py",
    "waves/equatorial_wave_diagnostics.py",
    "convection/lpo_lifecycle_analysis.py",
]

CONVECTION_ARGUMENTS_BY_SCRIPT = {
    "waves/equatorial_wave_diagnostics.py": (
        "--figures",
        "11",
        "13",
        "14",
        "15",
    ),
}

# These scripts own the reusable scientific caches in the convection workflow.
# Classification and tracking use their own validated product contracts.
CONVECTION_REBUILD_ARGUMENTS_BY_SCRIPT = {
    "convection/refined_high_cloud_classification_diagnostics.py": (
        "--rebuild-derived-cache",
    ),
    "convection/lpo_tracking.py": ("--rebuild-cache",),
    "waves/equatorial_wave_diagnostics.py": ("--rebuild-cache",),
    "convection/lpo_lifecycle_analysis.py": ("--rebuild-cache",),
}

CONVECTION_REBUILD_ENV_BY_SCRIPT = {}

FIGURES_ONLY_ARGUMENTS_BY_SCRIPT = {
    "mean_state/mean_state_diagnostics.py": ("--figures-only",),
    "mean_state/energy_budget.py": ("--figures-only",),
    "mean_state/vapour_transport_budget.py": ("--figures-only",),
    "mean_state/thermal_wind.py": ("--figures-only",),
    "mean_state/circulation_diagnostics.py": ("--figures-only",),
    "mean_state/circulation_spinup.py": ("--figures-only",),
    "mean_state/jet_variability_diagnostics.py": ("--figures-only",),
    "convection/refined_high_cloud_classification_diagnostics.py": (
        "--figures-only",
    ),
    "convection/lpo_tracking.py": ("--stage", "figures"),
    "waves/equatorial_wave_diagnostics.py": ("--figures-only",),
    "convection/lpo_lifecycle_analysis.py": ("--figures-only",),
    "convection/convection_summary.py": ("--figures-only",),
}

REBUILD_ARGUMENTS_BY_SCRIPT = {
    "mean_state/mean_state_diagnostics.py": ("--rebuild-cache",),
    "mean_state/energy_budget.py": ("--rebuild-cache",),
    "mean_state/vapour_transport_budget.py": ("--rebuild-cache",),
    "mean_state/thermal_wind.py": ("--rebuild-cache",),
    "mean_state/circulation_diagnostics.py": ("--rebuild-cache",),
    "mean_state/circulation_spinup.py": ("--rebuild-cache",),
    "mean_state/jet_variability_diagnostics.py": ("--rebuild-cache",),
    "convection/refined_high_cloud_classification_diagnostics.py": (
        "--rebuild-derived-cache",
    ),
    "convection/lpo_tracking.py": ("--rebuild-cache",),
    "waves/equatorial_wave_diagnostics.py": ("--rebuild-cache",),
    "convection/lpo_lifecycle_analysis.py": ("--rebuild-cache",),
    "convection/convection_summary.py": ("--rebuild-cache",),
}

QC_ARGUMENTS_BY_SCRIPT = {
    "convection/refined_high_cloud_classification_diagnostics.py": ("--qc",),
    "convection/lpo_tracking.py": (
        "--mode",
        "validation",
    ),
    "convection/lpo_lifecycle_analysis.py": (
        "--mode",
        "validation",
    ),
}

# Each owner uses the shortest actual-data subset that still satisfies its
# calculation geometry.  Mean-state spectra require 96 precipitation samples.
TEST_ARGUMENTS_BY_SCRIPT = {
    "mean_state/mean_state_diagnostics.py": ("--test", "--test-times", "100"),
    "mean_state/energy_budget.py": ("--test", "--test-times", "8"),
    "mean_state/vapour_transport_budget.py": ("--test", "--test-times", "5"),
    "mean_state/thermal_wind.py": ("--test", "--test-times", "8"),
    "mean_state/circulation_diagnostics.py": ("--test", "--test-times", "8"),
    "mean_state/jet_variability_diagnostics.py": ("--test", "--test-times", "16"),
    "convection/refined_high_cloud_classification_diagnostics.py": (
        "--test",
        "--simulation",
        "all",
        "--test-times",
        "16",
    ),
    "convection/lpo_tracking.py": ("--test", "--test-days", "10"),
    "waves/equatorial_wave_diagnostics.py": ("--test", "--test-days", "21"),
    "convection/lpo_lifecycle_analysis.py": ("--test", "--test-days", "10"),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
        "--profile",
        choices=tuple(SIMULATION_PROFILES),
        default=DEFAULT_SIMULATION_PROFILE,
        help="Select the simulation set used by every diagnostic.",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help=(
            "Build every selected diagnostic from isolated, bounded subsets "
            "of the selected simulations. Formal uncertainty is omitted."
        ),
    )
    parser.add_argument(
        "--convection-workflow",
        action="store_true",
        help=(
            "Run the complete publication convection/LPO workflow in dependency "
            "order, including the required wave and lifecycle products."
        ),
    )
    parser.add_argument(
        "--rebuild-convection-caches",
        action="store_true",
        help=(
            "With --convection-workflow, rebuild high-cloud, LPO, wave, and "
            "lifecycle products."
        ),
    )
    cache_mode = parser.add_mutually_exclusive_group()
    cache_mode.add_argument(
        "--figures-only",
        action="store_true",
        help=(
            "Render only from current calculation products. Selected scripts "
            "without figure-only support cause an error."
        ),
    )
    cache_mode.add_argument(
        "--rebuild-caches",
        action="store_true",
        help="Rebuild reusable calculation products before rendering.",
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
            "validation mode (currently refined high-cloud classification, "
            "LPO tracking, and LPO lifecycle enrichment)."
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
    parser.add_argument(
        "--figure-version",
        metavar="NAME",
        help=(
            "Place diagnostic figure directories beneath one named release "
            "parent instead of their default v1 directories."
        ),
    )
    parser.add_argument(
        "--notify-email",
        metavar="ADDRESS",
        help="Send progress, failure, and final workflow messages by email.",
    )
    parser.add_argument(
        "--notify-on",
        choices=("failure", "always"),
        default="always",
        help="Select failure-only email or progress and final-summary email.",
    )
    args = parser.parse_args(argv)
    if (
        args.figure_version
        and (
            args.figure_version in {".", ".."}
            or Path(args.figure_version).name != args.figure_version
        )
    ):
        parser.error("--figure-version must be one directory name")
    if args.rebuild_convection_caches and not args.convection_workflow:
        parser.error("--rebuild-convection-caches requires --convection-workflow")
    if args.rebuild_convection_caches and (
        args.rebuild_caches or args.figures_only
    ):
        parser.error("Choose one cache mode")
    if args.convection_workflow and args.only:
        parser.error("--convection-workflow cannot be combined with --only")
    if args.convection_workflow and args.skip:
        parser.error("--convection-workflow cannot be combined with --skip")
    if args.convection_workflow and args.qc:
        parser.error("--convection-workflow cannot be combined with --qc")
    if args.convection_workflow and args.continue_on_error:
        parser.error(
            "--convection-workflow is dependency ordered and cannot continue "
            "after a failed prerequisite"
        )
    if args.test and (
        args.convection_workflow
        or args.rebuild_convection_caches
        or args.rebuild_caches
        or args.figures_only
        or args.qc
    ):
        parser.error("--test cannot be combined with workflow or cache modes")
    return args


def normalise_name(name: str) -> str:
    """Normalise a diagnostic stem, filename, or family-relative path."""
    normalised = Path(name).as_posix().removeprefix("diagnostics/")
    return normalised if normalised.endswith(".py") else f"{normalised}.py"


def discover_diagnostic_scripts() -> list[str]:
    """Return configured default and optional diagnostics."""
    configured = [*DEFAULT_SCRIPT_ORDER, *OPTIONAL_SCRIPT_ORDER]
    missing = [
        script for script in configured
        if not (DIAGNOSTICS_DIR / script).is_file()
    ]
    if missing:
        raise SystemExit(f"Configured diagnostic scripts are missing: {', '.join(missing)}")
    return configured


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
    if args.convection_workflow:
        selected = list(CONVECTION_WORKFLOW_ORDER)
    elif args.only:
        selected = [
            resolve_script_name(name, diagnostic_scripts) for name in args.only
        ]
    else:
        selected = list(DEFAULT_SCRIPT_ORDER)
    skipped = {resolve_script_name(name, diagnostic_scripts) for name in args.skip}

    unknown = sorted((set(selected) | skipped) - available)
    if unknown:
        valid = ", ".join(diagnostic_scripts)
        raise SystemExit(f"Unknown diagnostic(s): {', '.join(unknown)}\nValid diagnostics: {valid}")

    return [script for script in selected if script not in skipped]


def print_script_list(scripts: list[str]) -> None:
    for index, script in enumerate(scripts, start=1):
        print(f"{index:2d}. {script}")


def local_mail_command() -> str | None:
    """Return an installed command that accepts ``mail -s subject address``."""

    return shutil.which("mail") or shutil.which("mailx")


def diagnostic_log_path(script: str, args: argparse.Namespace) -> Path | None:
    if args.no_log or not args.log_dir:
        return None
    log_name = script.removesuffix(".py").replace("/", "__")
    return args.log_dir / f"{log_name}.log"


def send_run_notification(
    args: argparse.Namespace,
    scripts: list[str],
    failures: list[tuple[str, int]],
    started: datetime,
) -> bool:
    """Send one run summary without changing the diagnostic exit status."""

    if not args.notify_email:
        return True
    if not failures and args.notify_on != "always":
        return True

    mail_command = local_mail_command()
    if mail_command is None:
        print("Notification failed: neither mail nor mailx is installed.", file=sys.stderr)
        return False

    ended = datetime.now()
    status = "FAILED" if failures else "COMPLETED"
    subject = f"MPAS diagnostics {status} on {socket.gethostname()}"
    lines = [
        f"Status: {status}",
        f"Host: {socket.gethostname()}",
        f"Started: {started:%Y-%m-%d %H:%M:%S}",
        f"Ended: {ended:%Y-%m-%d %H:%M:%S}",
        f"Elapsed: {ended - started}",
        "",
        "Selected diagnostics:",
        *(f"  {script}" for script in scripts),
    ]
    if failures:
        lines.extend(("", "Failures:"))
        for script, return_code in failures:
            log_path = diagnostic_log_path(script, args)
            log_text = str(log_path) if log_path else "logging disabled"
            lines.append(f"  {script}: exit code {return_code}; log: {log_text}")

    try:
        result = subprocess.run(
            [mail_command, "-s", subject, args.notify_email],
            input="\n".join(lines) + "\n",
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        print(f"Notification failed: {exc}", file=sys.stderr)
        return False
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        print(f"Notification failed: {detail}", file=sys.stderr)
        return False

    print(f"Notification sent to {args.notify_email}.")
    return True


def send_progress_notification(
    args: argparse.Namespace,
    scripts: list[str],
    completed: list[str],
    script: str,
    return_code: int,
    started: datetime,
) -> bool:
    """Send immediate progress or failure information for one diagnostic."""

    if not args.notify_email:
        return True
    if return_code == 0 and args.notify_on != "always":
        return True
    mail_command = local_mail_command()
    if mail_command is None:
        print("Notification failed: neither mail nor mailx is installed.", file=sys.stderr)
        return False
    status = "COMPLETED" if return_code == 0 else "FAILED"
    remaining = [item for item in scripts if item not in completed and item != script]
    log_path = diagnostic_log_path(script, args)
    lines = [
        f"Status: {status}",
        f"Diagnostic: {script}",
        f"Host: {socket.gethostname()}",
        f"Workflow elapsed: {datetime.now() - started}",
        f"Completed diagnostics: {len(completed)} of {len(scripts)}",
        f"Exit code: {return_code}",
        f"Log: {log_path if log_path else 'logging disabled'}",
        "",
        "Remaining diagnostics:",
        *(f"  {item}" for item in remaining),
    ]
    subject = f"MPAS diagnostic {status}: {Path(script).stem}"
    result = subprocess.run(
        [mail_command, "-s", subject, args.notify_email],
        input="\n".join(lines) + "\n",
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or f"exit code {result.returncode}"
        print(f"Notification failed: {detail}", file=sys.stderr)
        return False
    print(f"Progress notification sent to {args.notify_email}.")
    return True


def child_environment(args: argparse.Namespace) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env.pop("MPAS_SIMULATION_ROOT", None)
    env.pop("MPAS_DIAGNOSTIC_CACHE_ROOT", None)
    env[SIMULATION_PROFILE_ENV] = args.profile
    env.setdefault("MPLCONFIGDIR", str(Path("/tmp") / f"matplotlib-{os.getuid()}"))
    env.setdefault("XDG_CACHE_HOME", str(Path("/tmp") / f"mpas-xdg-cache-{os.getuid()}"))
    # These diagnostics run serially.  MPICH otherwise selects an unavailable
    # OFI network provider on restricted hosts.
    env.setdefault("FI_PROVIDER", "shm")
    if args.figure_root:
        env[DIAGNOSTIC_OUTPUT_ROOT_ENV] = str(args.figure_root.expanduser().resolve())
    if args.figure_version:
        env[DIAGNOSTIC_FIGURE_VERSION_ENV] = args.figure_version

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


def script_arguments(script: str, args: argparse.Namespace) -> list[str]:
    """Return diagnostic-specific arguments for one selected workflow."""

    script_args: list[str] = []
    if args.test:
        script_args.extend(TEST_ARGUMENTS_BY_SCRIPT[script])
    if args.qc and script in QC_ARGUMENTS_BY_SCRIPT:
        script_args.extend(QC_ARGUMENTS_BY_SCRIPT[script])
    if args.convection_workflow:
        script_args.extend(CONVECTION_ARGUMENTS_BY_SCRIPT.get(script, ()))
    if args.rebuild_convection_caches:
        script_args.extend(
            CONVECTION_REBUILD_ARGUMENTS_BY_SCRIPT.get(script, ())
        )
    if args.figures_only:
        script_args.extend(FIGURES_ONLY_ARGUMENTS_BY_SCRIPT.get(script, ()))
    if args.rebuild_caches:
        script_args.extend(REBUILD_ARGUMENTS_BY_SCRIPT.get(script, ()))
    return script_args


def script_environment_overrides(
    script: str,
    args: argparse.Namespace,
) -> dict[str, str]:
    """Return explicit environment overrides for one diagnostic process."""

    overrides = {SIMULATION_PROFILE_ENV: args.profile}
    if args.rebuild_convection_caches:
        overrides.update(CONVECTION_REBUILD_ENV_BY_SCRIPT.get(script, {}))
    return overrides


def script_command(script: str, args: argparse.Namespace) -> list[str]:
    command = [args.python, "-u", str(DIAGNOSTICS_DIR / script)]
    command.extend(script_arguments(script, args))
    return command


def display_command(script: str, args: argparse.Namespace) -> str:
    """Return a shell-safe command display, including environment overrides."""

    command = script_command(script, args)
    overrides = script_environment_overrides(script, args)
    if overrides:
        command = [
            "env",
            *(f"{name}={value}" for name, value in sorted(overrides.items())),
            *command,
        ]
    return shlex.join(command)


def process_memory_summary(process_id: int) -> str:
    """Return the available Linux memory counters for one diagnostic process."""

    try:
        status = Path(f"/proc/{process_id}/status").read_text(encoding="utf-8")
    except OSError:
        return "memory counters unavailable"

    counters = {}
    for line in status.splitlines():
        name, separator, value = line.partition(":")
        if separator and name in {"VmRSS", "VmHWM", "VmSwap"}:
            counters[name] = value.strip()
    return ", ".join(
        f"{label}={counters[key]}"
        for key, label in (("VmRSS", "RSS"), ("VmHWM", "peak"), ("VmSwap", "swap"))
        if key in counters
    ) or "memory counters unavailable"


def _read_process_output(stream, output_queue):
    """Read child output without blocking workflow heartbeat messages."""

    try:
        for line in stream:
            output_queue.put(line)
    finally:
        output_queue.put(None)


def run_script(script: str, args: argparse.Namespace) -> int:
    command = script_command(script, args)
    command_text = display_command(script, args)
    env = child_environment(args)
    env.update(script_environment_overrides(script, args))

    start = datetime.now()
    print(f"\n[{start:%Y-%m-%d %H:%M:%S}] Starting {script}", flush=True)
    print(f"Command: {command_text}", flush=True)

    log_file = None
    if args.log_dir and not args.no_log:
        args.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = diagnostic_log_path(script, args)
        assert log_path is not None
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
        output_queue = queue.Queue()
        reader = threading.Thread(
            target=_read_process_output,
            args=(process.stdout, output_queue),
            daemon=True,
        )
        reader.start()

        while True:
            try:
                line = output_queue.get(timeout=LOG_HEARTBEAT_SECONDS)
            except queue.Empty:
                now = datetime.now()
                heartbeat = (
                    f"[{now:%Y-%m-%d %H:%M:%S}] {script} is still running "
                    f"after {now - start}; {process_memory_summary(process.pid)}"
                )
                print(heartbeat, flush=True)
                if log_file:
                    log_file.write(heartbeat + "\n")
                    log_file.flush()
                continue

            if line is None:
                break
            print(line, end="", flush=True)
            if log_file:
                log_file.write(line)
                log_file.flush()

        return_code = process.wait()
        reader.join(timeout=1.0)
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
    run_started = datetime.now()

    if args.figures_only:
        unsupported = [
            script for script in scripts
            if script not in FIGURES_ONLY_ARGUMENTS_BY_SCRIPT
        ]
        if unsupported:
            names = ", ".join(unsupported)
            print(
                f"Figure-only mode is unavailable for: {names}",
                file=sys.stderr,
            )
            return 2

    if args.list:
        print_script_list(scripts)
        return 0

    if not scripts:
        print("No diagnostics selected.")
        return 0

    print("Diagnostics to run:")
    print_script_list(scripts)
    print(f"Simulation profile: {args.profile}")
    if args.convection_workflow:
        print("Mode: complete dependency-ordered convection workflow.")
    if args.rebuild_convection_caches:
        print(
            "Cache policy: rebuild high-cloud, LPO, wave, and lifecycle products."
        )
    if args.figures_only:
        print("Cache policy: render only from current scientific products.")
    if args.rebuild_caches:
        print("Cache policy: rebuild supported reusable scientific products.")
    if args.no_log:
        print("Logging disabled (--no-log).")
    elif args.log_dir:
        print(f"Logs will be written to: {args.log_dir}")
    if args.figure_root:
        print(f"Figures/output will be written under: {args.figure_root.expanduser().resolve()}")
    if args.figure_version:
        figure_root = (
            args.figure_root.expanduser().resolve()
            if args.figure_root
            else DEFAULT_FIGURE_ROOT
        )
        print(
            "Figure release will be written under: "
            f"{figure_root / 'final' / args.figure_version}"
        )
    if args.notify_email:
        if local_mail_command() is None:
            print(
                "Cannot enable email notifications because mail and mailx are absent.",
                file=sys.stderr,
            )
            return 2
        print(f"Email notifications will be sent to: {args.notify_email}")

    if args.dry_run:
        print("\nDry run only; no diagnostics were executed.")
        for script in scripts:
            command = display_command(script, args)
            if args.no_log or not args.log_dir:
                print(command)
            else:
                log_name = script.removesuffix(".py").replace("/", "__")
                log_path = args.log_dir / f"{log_name}.log"
                print(f"{command}  # log: {log_path}")
        return 0

    failures: list[tuple[str, int]] = []
    completed: list[str] = []
    for script in scripts:
        try:
            return_code = run_script(script, args)
        except Exception:
            traceback.print_exc()
            return_code = 1
        if return_code != 0:
            failures.append((script, return_code))
            send_progress_notification(
                args, scripts, completed, script, return_code, run_started
            )
            if not args.continue_on_error:
                break
        else:
            completed.append(script)
            send_progress_notification(
                args, scripts, completed, script, return_code, run_started
            )

    if failures:
        print("\nDiagnostic failures:")
        for script, return_code in failures:
            print(f"  {script}: exit code {return_code}")
        send_run_notification(args, scripts, failures, run_started)
        return failures[0][1]

    print("\nAll selected diagnostics completed successfully.")
    send_run_notification(args, scripts, failures, run_started)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
