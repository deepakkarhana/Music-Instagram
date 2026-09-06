"""STEP 1 - Build the song catalog.

Run this from the project root:

    python scripts/01_harvest_itunes.py

It searches the iTunes API for every seed term in `src/catalog/seeds.py` and
writes the results to `data/raw/itunes_catalog.csv`.

Takes roughly 5-8 minutes (we pause between requests to be polite to Apple).
You can stop it any time with Ctrl+C and run it again later - it picks up
exactly where it left off.

Useful flags:

    --limit 50      fewer results per search (good for a quick 1-minute test)
    --sleep 1.5     shorter pause between requests (be careful, may get blocked)
    --out PATH      write somewhere else
"""

import argparse
import os
import sys

# Make `src` importable when running this file directly from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.console import setup as _console_setup  # noqa: E402

_console_setup()

from src.catalog.itunes import harvest  # noqa: E402
from src.catalog.seeds import all_seeds  # noqa: E402

DEFAULT_OUT = os.path.join("data", "raw", "itunes_catalog.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=DEFAULT_OUT,
                        help=f"output CSV path (default: {DEFAULT_OUT})")
    parser.add_argument("--limit", type=int, default=200,
                        help="results per search, max 200 (default: 200)")
    parser.add_argument("--sleep", type=float, default=3.0,
                        help="seconds to wait between requests (default: 3.0)")
    parser.add_argument("--quick", action="store_true",
                        help="quick test: only the first 5 searches, 25 results each")
    args = parser.parse_args()

    seeds = all_seeds()
    limit = args.limit

    if args.quick:
        seeds = seeds[:5]
        limit = 25
        print(">> QUICK TEST MODE: 5 searches only\n")

    print(f"Seeds to search : {len(seeds)}")
    print(f"Results each    : {limit}")
    print(f"Writing to      : {args.out}")
    print(f"Estimated time  : {len(seeds) * args.sleep / 60:.1f} minutes\n")

    total = harvest(seeds, args.out, limit=limit, sleep_seconds=args.sleep)

    print(f"\nDone. {total:,} unique tracks in {args.out}")
    print("Next: run  python scripts/02_inspect_catalog.py  to look at what you got.")


if __name__ == "__main__":
    main()
