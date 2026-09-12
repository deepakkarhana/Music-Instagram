"""STEP 8 - Fetch freely-licensed development photos from Wikimedia Commons.

Run this from the project root:

    python scripts/08_fetch_dev_photos.py

It searches Commons for each vibe in `src/vibe/taxonomy.py`, using that vibe's
`visual_cues` as the query, and saves the results to `data/photos_dev/` with a
manifest recording every licence and author.

WHY A SEPARATE FOLDER FROM data/photos/
---------------------------------------
These are a **development set**: enough images to build and debug Week 4
without waiting for anyone to empty their camera roll.

They are not a stand-in for real photos, and the difference is not cosmetic.
Commons is documentary and landscape photography - composed, well lit, often
shot on real cameras. A phone camera roll is tilted, badly lit, cluttered,
and full of faces, food and selfies. A system tuned on one and deployed on
the other looks excellent right up until someone uses it.

    data/photos_dev/   build and debug against these
    data/photos/       real photos - the ones any reported number comes from

Never report an accuracy figure computed on this folder.

USEFUL FLAGS
------------
    --per-vibe 3     how many photos to fetch per vibe (default 3)
    --axis scene     only fetch for one axis
    --dry-run        search and report, download nothing
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import console
from src.vibe import photos as photo_source
from src.vibe import taxonomy

OUT_DIR = os.path.join("data", "photos_dev")
MANIFEST = os.path.join("data", "photos_dev", "manifest.csv")

FIELDNAMES = ["file", "vibe", "axis", "title", "author", "licence", "source_page", "query"]


def load_manifest(path):
    """Existing rows, plus how many photos each vibe already has.

    Counting per vibe rather than just "has this vibe been done" is what lets
    you raise --per-vibe later and top up, instead of the rerun deciding
    everything is finished and doing nothing. Titles already downloaded are
    tracked too, so a top-up fetches *new* photos rather than the same ones.
    """
    if not os.path.exists(path):
        return [], {}, set()
    with open(path, "r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    counts = {}
    for row in rows:
        counts[row["vibe"]] = counts.get(row["vibe"], 0) + 1
    titles = {r["title"] for r in rows}
    return rows, counts, titles


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", default=OUT_DIR)
    parser.add_argument("--per-vibe", type=int, default=3)
    parser.add_argument("--axis", default=None, choices=taxonomy.AXES)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    console.setup()
    os.makedirs(args.out, exist_ok=True)
    manifest_path = os.path.join(args.out, "manifest.csv")

    rows, counts, have_titles = load_manifest(manifest_path)
    if rows:
        print(f"Already have {len(rows)} photos across {len(counts)} vibes")
        print(f"Topping up to {args.per_vibe} each")
        print()

    vibes = taxonomy.by_axis(args.axis) if args.axis else taxonomy.VIBES
    todo = [v for v in vibes if counts.get(v.name, 0) < args.per_vibe]

    if not todo:
        print(f"Nothing to do - every vibe already has {args.per_vibe} photos.")
        print(f"\nNext:  python scripts/07_check_photos.py --dir {args.out}")
        return 0

    print(f"Fetching up to {args.per_vibe} photos for each of {len(todo)} vibes")
    print("Source: Wikimedia Commons, CC-BY / CC0 / public domain only\n")

    handle = None
    writer = None
    if not args.dry_run:
        exists = os.path.exists(manifest_path)
        handle = open(manifest_path, "a", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()

    saved = 0
    empty = []

    try:
        for number, vibe in enumerate(todo, start=1):
            # WHERE TO LOOK DEPENDS ON WHAT WE ARE LOOKING FOR.
            #
            # Commons is an encyclopaedia's picture library: excellent for
            # "Indian wedding", useless for "melancholy", because nobody
            # uploads a moody photograph to illustrate an article. Openverse
            # reaches Flickr, where people post photographs for their own sake.
            #
            # And the search terms differ too. Searching "melancholy" finds
            # Durer's engraving; searching "rain drops on a window" finds a
            # photograph that actually feels melancholy. taxonomy.py holds the
            # concrete phrasings for the abstract vibes.
            abstract = vibe.axis in taxonomy.LABEL_UNRELIABLE_AXES
            prefer = "openverse" if abstract else "commons"

            candidates = list(taxonomy.search_terms(vibe))
            if not abstract:
                candidates.append(vibe.visual_cues.split(",")[0].strip())

            found, query, source = [], candidates[0], prefer
            for candidate in candidates:
                found, source = photo_source.search_best(
                    candidate, limit=args.per_vibe * 4, prefer=prefer)
                query = candidate
                if found:
                    break
                photo_source.polite_pause()

            print(f"[{number}/{len(todo)}] {vibe.label:<22} [{source}] \"{query}\"")
            if not found:
                print("      nothing usable found")
                empty.append(vibe.name)
                photo_source.polite_pause()
                continue

            already = counts.get(vibe.name, 0)
            wanted = args.per_vibe - already
            kept = 0
            for item in found:
                if kept >= wanted:
                    break
                if item["title"] in have_titles:
                    continue  # already downloaded on an earlier run
                filename = photo_source.safe_name(
                    vibe.name, already + kept + 1, item["title"])
                target = os.path.join(args.out, filename)

                if args.dry_run:
                    print(f"      would save {filename}  [{item['licence']}]")
                    kept += 1
                    continue

                if not photo_source.download(item["url"], target):
                    continue

                writer.writerow({
                    "file": filename,
                    "vibe": vibe.name,
                    "axis": vibe.axis,
                    "title": item["title"],
                    "author": item["author"],
                    "licence": item["licence"],
                    "source_page": item["page"],
                    "query": query,
                })
                handle.flush()
                have_titles.add(item["title"])
                kept += 1
                saved += 1

            if kept == 0:
                empty.append(vibe.name)
            print(f"      kept {kept}  (vibe now has {already + kept})")
            photo_source.polite_pause()

    except KeyboardInterrupt:
        print("\nStopped by you. Everything downloaded so far is saved.")
    finally:
        if handle:
            handle.close()

    print(f"\nSaved {saved} photos to {args.out}")
    if empty:
        print(f"\n{len(empty)} vibes found nothing usable:")
        print("  " + ", ".join(empty))
        print("\nThat is expected. Commons is documentary photography - it has")
        print("plenty of mountains and temples, and almost no gym selfies or")
        print("cafe table shots. Those gaps are exactly where real photos are")
        print("needed, and they are worth knowing about before Week 4.")

    print(f"\nEvery photo's licence and author is in {manifest_path}")
    print(f"\nNext:  python scripts/07_check_photos.py --dir {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
