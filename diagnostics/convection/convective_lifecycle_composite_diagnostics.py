#!/usr/bin/env python3
"""Regenerate LPT-envelope convective lifecycle figures.

The tracking diagnostic owns event identification and writes the stable
event-level products.  This downstream entry point deliberately performs no
tracking and never reads upstream LPT internals: it rebuilds publication
Figures 4--7 exclusively from the standard LPT event tables and masks.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
for _parent in (_THIS_DIR.parent, _THIS_DIR.parent.parent, _THIS_DIR.parent.parent.parent):
    if str(_parent) not in sys.path:
        sys.path.append(str(_parent))

from diagnostics.convection.lpt_precipitation_envelope_diagnostics import (
    LIFECYCLE_FIGURES,
    build_figures_from_saved_outputs,
    lpt_output_root,
    require_publication_products,
)


def parse_command_line(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build fixed-lag, event-equal cloud lifecycle figures for tracked "
            "large-scale precipitation envelopes."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--lpt-output",
        default=str(lpt_output_root()),
        help="Root containing the LPT publication/ and validation/ directories.",
    )
    parser.add_argument(
        "--figures",
        nargs="+",
        type=int,
        choices=LIFECYCLE_FIGURES,
        default=list(LIFECYCLE_FIGURES),
        help="Publication lifecycle figures to regenerate.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_command_line(argv)
    publication = require_publication_products(args.lpt_output)
    build_figures_from_saved_outputs(
        publication,
        mode="publication",
        figures=args.figures,
    )
    print(
        "Finished LPT envelope lifecycle figures from saved event-level products: "
        f"{publication / 'figures'}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
