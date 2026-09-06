"""STEP 2 - Look at the catalog you just built.

    python scripts/02_inspect_catalog.py

Prints a summary: how many tracks, which genres, which decades, and a few
example rows. Standard library only - no pandas needed.

WHY BOTHER?
-----------
Always look at your data before you model it. Half of all machine learning
bugs are actually data bugs, and they are much cheaper to find now than in
Week 8. Things worth noticing here:

  - Is the Indian/English split roughly what you expected?
  - Are the genres sensible, or is one search term dominating everything?
  - Are there tracks with a missing year, or a suspiciously short duration?
"""

import argparse
import collections
import csv
import os
import sys

# Make `src` importable when running this file directly from the project root.
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.console import setup as _console_setup  # noqa: E402

_console_setup()

DEFAULT_IN = os.path.join("data", "raw", "itunes_catalog.csv")


def bar(count, total, width=28):
    """Draw a tiny text bar chart, because numbers alone are hard to feel."""
    filled = int(round(width * count / total)) if total else 0
    return "#" * filled + "." * (width - filled)


def show_top(title, counter, total, top_n=12):
    print(f"\n{title}")
    print("-" * 64)
    for name, count in counter.most_common(top_n):
        label = (name or "(blank)")[:26].ljust(26)
        print(f"  {label} {count:>6,}  {bar(count, total)}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file", default=DEFAULT_IN, help=f"CSV to inspect (default: {DEFAULT_IN})")
    args = parser.parse_args()

    if not os.path.exists(args.file):
        print(f"No catalog found at {args.file}")
        print("Run this first:  python scripts/01_harvest_itunes.py")
        sys.exit(1)

    rows = []
    with open(args.file, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            rows.append(row)

    if not rows:
        print("The catalog file is empty.")
        sys.exit(1)

    total = len(rows)

    lanes = collections.Counter(r["lane"] for r in rows)
    genres = collections.Counter(r["genre"] for r in rows)
    artists = collections.Counter(r["artist_name"] for r in rows)
    decades = collections.Counter(
        (r["release_year"][:3] + "0s") if len(r["release_year"]) == 4 else "unknown"
        for r in rows
    )

    unique_ids = len({r["track_id"] for r in rows})
    with_preview = sum(1 for r in rows if r["preview_url"])
    explicit = sum(1 for r in rows if r["explicit"] == "explicit")

    print("=" * 64)
    print(f"CATALOG: {args.file}")
    print("=" * 64)
    print(f"  rows                  {total:>8,}")
    print(f"  unique track ids      {unique_ids:>8,}")
    print(f"  unique artists        {len(artists):>8,}")
    print(f"  with preview audio    {with_preview:>8,}   ({100*with_preview/total:.1f}%)")
    print(f"  marked explicit       {explicit:>8,}   ({100*explicit/total:.1f}%)")

    if unique_ids != total:
        print(f"\n  !! {total - unique_ids} duplicate track ids - the harvester should")
        print("     have removed these. Worth investigating.")
    if with_preview != total:
        print(f"\n  !! {total - with_preview} rows have no preview URL and cannot be embedded.")

    show_top("BY LANE", lanes, total, top_n=5)
    show_top("BY GENRE", genres, total)
    show_top("BY DECADE", decades, total, top_n=10)
    show_top("TOP ARTISTS", artists, total, top_n=10)

    print("\nFIVE EXAMPLE TRACKS")
    print("-" * 64)
    step = max(1, total // 5)
    for row in rows[::step][:5]:
        year = row["release_year"] or "????"
        print(f"  {row['track_name'][:34]:<34} | {row['artist_name'][:20]:<20} | {year}")
        print(f"    {row['genre']} / {row['lane']}")

    print("\nWhat to check before moving on:")
    print("  1. Does the lane split look reasonable to you?")
    print("  2. Any genre swallowing everything? Add more seeds to balance it.")
    print("  3. Open a few preview_url links in a browser - do they play?")


if __name__ == "__main__":
    main()
