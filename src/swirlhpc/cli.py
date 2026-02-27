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


def main():
    parser = argparse.ArgumentParser(
        description="Run the SWIRL radar wind retrieval pipeline on AURA archive data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single radar, single day
  swirl-run --radars 2 --start 2025-10-16

  # Multi-Doppler region, date range
  swirl-run --radars 2 3 4 --start 2025-10-01 --end 2025-10-07

  # With custom config
  swirl-run --radars 2 --start 2025-10-16 --config my_config.toml
""",
    )
    parser.add_argument(
        "--radars",
        "-r",
        nargs="+",
        type=int,
        required=True,
        help="Radar ID(s) to process. Multiple IDs trigger multi-Doppler retrieval.",
    )
    parser.add_argument(
        "--start",
        "-s",
        type=_parse_date,
        required=True,
        help="Start date (YYYY-MM-DD or YYYYMMDD).",
    )
    parser.add_argument(
        "--end",
        "-e",
        type=_parse_date,
        default=None,
        help="End date (inclusive). Defaults to start date.",
    )
    parser.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to a TOML configuration file.",
    )

    args = parser.parse_args()

    results = run(
        radar_ids=args.radars,
        start_date=args.start,
        end_date=args.end,
        config=args.config,
    )

    # Exit code: 0 if anything was processed, 1 if nothing
    sys.exit(0 if results["timesteps_processed"] > 0 else 1)


if __name__ == "__main__":
    main()
