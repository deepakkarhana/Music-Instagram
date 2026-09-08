"""STEP 7 - Check the photos you collected are usable.

Run this from the project root, after putting photos in `data/photos/`:

    python scripts/07_check_photos.py

WHAT IT CHECKS, AND WHY EACH ONE BITES LATER
--------------------------------------------
**Format.** Phones save HEIC by default on iPhone, and most Python image
libraries cannot read it. Finding that out now costs a minute; finding out in
Week 4, halfway through a batch job, costs an evening.

**Size.** The vision model resizes everything to roughly 384x384 anyway, so a
12-megapixel photo is 30x more data than it can use - slow to load, no better
for it. Very small images are the opposite problem: upscaling invents detail
that was never there.

**Duplicates.** Camera rolls are full of near-identical bursts. Three shots of
the same moment count as one photo for evaluation purposes, and quietly
inflate any accuracy number computed over them.

**Corruption.** A file that will not open is better discovered in a listing
than in a stack trace.

It also reports how many photos you have relative to the target, because 30
covering many vibes beats 100 of the same holiday.
"""

import argparse
import hashlib
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import console
from src.vibe import taxonomy

PHOTO_DIR = os.path.join("data", "photos")

# What the vision model in Week 4 can actually read.
READABLE = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}
# Common phone formats that need converting first.
NEEDS_CONVERTING = {".heic", ".heif", ".dng", ".raw", ".cr2", ".nef", ".arw"}

MIN_SIDE = 384        # below this, the model is upscaling invented detail
LARGE_PIXELS = 12_000_000  # above this, we are wasting load time
TARGET = 20


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", default=PHOTO_DIR, help="folder holding the photos")
    args = parser.parse_args()

    console.setup()

    if not os.path.isdir(args.dir):
        os.makedirs(args.dir, exist_ok=True)
        print(f"Created {args.dir}")
        print()
        print("It is empty. Put 20-30 photos you would actually post in there,")
        print("then run this again. See docs/week-03.md for what to aim for.")
        return 0

    files = sorted(
        os.path.join(args.dir, f)
        for f in os.listdir(args.dir)
        if os.path.isfile(os.path.join(args.dir, f))
        and not f.startswith(".")
        and f != "manifest.csv"  # the attribution record, not a photo
    )

    if not files:
        print(f"{args.dir} is empty.")
        print("Put 20-30 photos in there - see docs/week-03.md.")
        return 0

    readable, convert, unknown, broken = [], [], [], []
    small, huge = [], []
    digests = {}

    for path in files:
        extension = os.path.splitext(path)[1].lower()
        if extension in NEEDS_CONVERTING:
            convert.append(path)
            continue
        if extension not in READABLE:
            unknown.append(path)
            continue

        try:
            from PIL import Image

            with Image.open(path) as image:
                width, height = image.size
                image.verify()  # catches truncated files without decoding it all
        except Exception as error:
            broken.append((path, type(error).__name__))
            continue

        readable.append((path, width, height))
        if min(width, height) < MIN_SIDE:
            small.append((path, width, height))
        if width * height > LARGE_PIXELS:
            huge.append((path, width, height))

        # Content hash, so renamed copies of one photo still collide.
        with open(path, "rb") as handle:
            digests.setdefault(hashlib.md5(handle.read()).hexdigest(), []).append(path)

    print("=" * 70)
    print(f"PHOTOS IN {args.dir}")
    print("=" * 70)
    print(f"  files found     {len(files)}")
    print(f"  usable          {len(readable)}")
    if convert:
        print(f"  need converting {len(convert)}")
    if unknown:
        print(f"  not images      {len(unknown)}")
    if broken:
        print(f"  unreadable      {len(broken)}")

    if convert:
        print("\nNEEDS CONVERTING - most Python image libraries cannot read these")
        for path in convert[:10]:
            print(f"  {os.path.basename(path)}")
        print("  On iPhone: Settings > Camera > Formats > Most Compatible,")
        print("  or open them in Photos and export as JPEG.")

    if broken:
        print("\nUNREADABLE - probably truncated or corrupt")
        for path, error in broken[:10]:
            print(f"  {os.path.basename(path)}  ({error})")

    if small:
        print(f"\nTOO SMALL - under {MIN_SIDE}px on the short side, the model upscales")
        for path, width, height in small[:10]:
            print(f"  {os.path.basename(path)}  {width}x{height}")

    if huge:
        print("\nVERY LARGE - fine, but slow to load for no benefit")
        for path, width, height in huge[:5]:
            print(f"  {os.path.basename(path)}  {width}x{height}")

    duplicates = {d: paths for d, paths in digests.items() if len(paths) > 1}
    if duplicates:
        print("\nEXACT DUPLICATES - same file content, counts as one photo")
        for paths in list(duplicates.values())[:5]:
            print("  " + " = ".join(os.path.basename(p) for p in paths))

    unique = len(readable) - sum(len(p) - 1 for p in duplicates.values())

    print("\n" + "=" * 70)
    print(f"Usable and unique: {unique} (target {TARGET}-30)")
    if unique < TARGET:
        print(f"Add {TARGET - unique} more, spread across different situations.")
    else:
        print("Enough to start Week 4.")

    print()
    print("The vocabulary has 40 vibes:")
    for axis, count in taxonomy.summary().items():
        labels = ", ".join(v.label for v in taxonomy.by_axis(axis)[:4])
        print(f"  {axis:<10} {count:>2}   {labels}...")
    print()
    print("Coverage against these is measured in Week 4, once a model can read")
    print("a photo and pick vibes from the list. For now, aim for variety by eye.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
