#!/usr/bin/env python3

import argparse
import datetime as dt
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_REMOTE_USER = "macro"
DEFAULT_REMOTE_BASE = "/mnt/imagesbucket/reduced"
DEFAULT_LOCAL_BASE = "~/gdrive/Shared drives/MACRO-Be/resources/Data/raw_data"
DEFAULT_PATTERN = "*rg_*.fz"


def parse_date(date_str: str) -> dt.date:
    try:
        return dt.datetime.strptime(date_str, "%Y-%m-%d").date()
    except ValueError as e:
        raise argparse.ArgumentTypeError(
            f"Invalid date '{date_str}'. Expected format: YYYY-MM-DD"
        ) from e


def daterange(start: dt.date, end: dt.date):
    current = start
    while current <= end:
        yield current
        current += dt.timedelta(days=1)


def run_scp_for_day(
    day: dt.date,
    remote_user: str,
    remote_base: str,
    local_base: Path,
    pattern: str,
    dry_run: bool = False,
) -> int:
    day_str = day.isoformat()
    remote_glob = f"{remote_user}:{remote_base}/{day_str}/{pattern}"
    local_dir = local_base / day_str
    local_dir.mkdir(parents=True, exist_ok=True)

    cmd = ["scp", remote_glob, str(local_dir)]

    print(f"\n[{day_str}]")
    print("Remote:", remote_glob)
    print("Local: ", local_dir)

    if dry_run:
        print("Dry run:", " ".join(cmd))
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True)

    if result.returncode == 0:
        print("Copied successfully.")
        return 0

    stderr = (result.stderr or "").strip()

    # Common "no files matched" behavior from remote shell/scp
    no_match_markers = [
        "No such file or directory",
        "not found",
        "No match",
        "No matches found",
        "cannot stat",
    ]

    if any(marker.lower() in stderr.lower() for marker in no_match_markers):
        print("No matching files found; skipping.")
        return 1

    print("scp failed.")
    if stderr:
        print(stderr)
    return 2


def main():
    parser = argparse.ArgumentParser(
        description="Copy MACRO reduced data for one date or a date range."
    )

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--date",
        type=parse_date,
        help="Single date in YYYY-MM-DD format.",
    )
    group.add_argument(
        "--start-date",
        type=parse_date,
        help="Start date in YYYY-MM-DD format.",
    )

    parser.add_argument(
        "--end-date",
        type=parse_date,
        help="End date in YYYY-MM-DD format. Required if --start-date is used.",
    )
    parser.add_argument(
        "--remote-user",
        default=DEFAULT_REMOTE_USER,
        help=f"Remote user/host. Default: {DEFAULT_REMOTE_USER}",
    )
    parser.add_argument(
        "--remote-base",
        default=DEFAULT_REMOTE_BASE,
        help=f"Remote base directory. Default: {DEFAULT_REMOTE_BASE}",
    )
    parser.add_argument(
        "--local-base",
        default=DEFAULT_LOCAL_BASE,
        help=f"Local base directory. Default: {DEFAULT_LOCAL_BASE}",
    )
    parser.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"Filename glob pattern. Default: {DEFAULT_PATTERN}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without running them.",
    )

    args = parser.parse_args()

    if args.start_date and not args.end_date:
        parser.error("--end-date is required when using --start-date")

    if args.end_date and not args.start_date:
        parser.error("--start-date is required when using --end-date")

    if args.start_date and args.end_date and args.end_date < args.start_date:
        parser.error("--end-date must be on or after --start-date")

    local_base = Path(os.path.expanduser(args.local_base))

    if args.date:
        days = [args.date]
    else:
        days = list(daterange(args.start_date, args.end_date))

    copied = 0
    skipped = 0
    failed = 0

    for day in days:
        status = run_scp_for_day(
            day=day,
            remote_user=args.remote_user,
            remote_base=args.remote_base,
            local_base=local_base,
            pattern=args.pattern,
            dry_run=args.dry_run,
        )
        if status == 0:
            copied += 1
        elif status == 1:
            skipped += 1
        else:
            failed += 1

    print("\nSummary")
    print(f"  Days processed: {len(days)}")
    print(f"  Copied:         {copied}")
    print(f"  Skipped:        {skipped}")
    print(f"  Failed:         {failed}")

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()