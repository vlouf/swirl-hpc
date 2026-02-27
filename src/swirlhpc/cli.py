"""
Command-line interface for SWIRL HPC.

Usage:
    swirl-run --radars 2 3 4 --start 2025-10-01 --end 2025-10-07 --config my_config.toml
    swirl-run --radars 2 --start 2025-10-16
"""

from __future__ import annotations

import argparse
import sys
from datetime import date, datetime
from pathlib import Path

from swirlhpc.pipeline import run


def _parse_date(s: str) -> date:
    """Parse a date string in YYYY-MM-DD or YYYYMMDD format."""
    for fmt in ("%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(s, fmt).date()
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(f"Cannot parse date: {s!r}. Use YYYY-MM-DD or YYYYMMDD.")


def _parse_datetime(s: str) -> datetime:
    """Parse a datetime string in ISO-ish formats."""
    for fmt in (
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %H:%M",
        "%Y%m%d_%H%M%S",
        "%Y%m%d_%H%M",
        "%Y%m%dT%H%M%S",
        "%Y%m%dT%H%M",
    ):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Cannot parse datetime: {s!r}. "
        "Use YYYY-MM-DDTHH:MM:SS, YYYY-MM-DDTHH:MM, or YYYYMMDD_HHMM."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run the SWIRL radar wind retrieval pipeline on AURA archive data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single radar, single timestep
  swirl-run --radars 2 --time 2026-01-01T12:30

  # Single radar, single day
  swirl-run --radars 2 --start 2025-10-16

  # Multi-Doppler region "503", date range
  swirl-run --radars 2 49 68 --region 503 --start 2016-12-27 --end 2016-12-30

  # With custom config
  swirl-run --radars 2 --start 2025-10-16 --config my_config.toml
""",
    )
    parser.add_argument(
        "--radars", "-r",
        nargs="+", type=int, required=True,
        help="Radar ID(s) to process. Multiple IDs trigger multi-Doppler retrieval.",
    )
    parser.add_argument(
        "--region", "-R",
        type=str, default=None,
        help="Name for the multi-Doppler region (e.g. 503). Used in output filenames. "
             "If not set, defaults to sorted radar IDs joined by underscore.",
    )

    # Time selection: either --time (single timestep) or --start [--end] (date range)
    time_group = parser.add_mutually_exclusive_group(required=True)
    time_group.add_argument(
        "--time", "-t",
        type=_parse_datetime, default=None,
        help="Single timestep to process (e.g. 2026-01-01T12:30).",
    )
    time_group.add_argument(
        "--start", "-s",
        type=_parse_date, default=None,
        help="Start date for full-day or date-range mode (YYYY-MM-DD).",
    )

    parser.add_argument(
        "--end", "-e",
        type=_parse_date, default=None,
        help="End date (inclusive). Defaults to start date. Only used with --start.",
    )
    parser.add_argument(
        "--config", "-c",
        type=Path, default=None,
        help="Path to a TOML configuration file.",
    )

    args = parser.parse_args()

    if args.time is not None:
        start = args.time   # datetime → single-timestep mode
        end = None
    else:
        start = args.start  # date → date-range mode
        end = args.end

    results = run(
        radar_ids=args.radars,
        start_date=start,
        end_date=end,
        config=args.config,
        region_name=args.region,
    )

    # Exit code: 0 if anything was processed, 1 if nothing
    sys.exit(0 if results["timesteps_processed"] > 0 else 1)


if __name__ == "__main__":
    main()