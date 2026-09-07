"""STEP 6 - Test the vibe vocabulary against the real catalog.

Run this from the project root:

    python scripts/06_probe_vibes.py

WHAT THIS IS FOR
----------------
`src/vibe/taxonomy.py` is a list of guesses. Somebody sat down and decided
that "melancholy" and "monsoon" are different enough to be separate vibes, and
that "quiet piano, minor key, sparse" is what melancholy sounds like.

Guesses like that are usually wrong somewhere, and the expensive way to find
out is Week 6, after the whole system is built on top of them. The cheap way
is now: fire every vibe at the finished index and look at what comes back.

Three failures matter, and each has a different fix:

**Dead vibes.** The best match scores low, so nothing in the catalog sounds
like this. Either the words are bad, or we genuinely have no such music. Fix
the wording first; if it stays dead, drop the vibe or widen the catalog.

**Twin vibes.** Two vibes retrieve almost the same songs. They are one vibe
wearing two names. Merge them, or sharpen the descriptions until they differ.

**Hub songs.** One track shows up under many unrelated vibes. It sits near the
middle of the space and crowds everything else out - a known problem in this
kind of search, and a Week 10 reranking job rather than a taxonomy fault.

USEFUL FLAGS
------------
    --axis mood      only test one axis
    --show 3         how many songs to print per vibe
    --full           print every vibe, not just the problems
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import console
from src.encode import clap
from src.encode.store import read_catalog
from src.retrieve import index as index_module
from src.vibe import taxonomy

CATALOG_PATH = os.path.join("data", "raw", "itunes_catalog.csv")
INDEX_PATH = os.path.join("data", "interim", "catalog")

# A vibe whose best match scores below this is probably not findable.
DEAD_SCORE = 0.45
# Two vibes sharing more than this fraction of their top-10 are near-twins.
TWIN_OVERLAP = 0.5
# A song appearing in this many vibes' top-10 is acting as a hub.
HUB_APPEARANCES = 5


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("--axis", default=None, choices=taxonomy.AXES)
    parser.add_argument("--show", type=int, default=3, help="songs printed per vibe")
    parser.add_argument("--full", action="store_true", help="print every vibe")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    console.setup()

    if not os.path.exists(args.index + ".faiss"):
        print(f"No index at {args.index}.faiss")
        print("Run this first:  python scripts/04_build_index.py")
        return 1

    index, track_ids = index_module.load(args.index)
    catalog = read_catalog(args.catalog)
    by_id = {str(r["track_id"]): r for r in catalog}

    vibes = taxonomy.by_axis(args.axis) if args.axis else taxonomy.VIBES
    print(f"Vocabulary: {len(taxonomy.VIBES)} vibes {taxonomy.summary()}")
    print(f"Index:      {index.ntotal:,} songs")
    print(f"Testing:    {len(vibes)} vibes\n")

    model, processor, device = clap.load(device=args.device)
    vectors = clap.embed_text(model, processor, [v.music_query for v in vibes], device)

    # Ask for 10 each, so overlap between vibes is measurable.
    scores, rows = index_module.search(index, vectors, k=10)

    top_sets = [set(r for r in row if r >= 0) for row in rows]

    # ---------------------------------------------------------------- report
    if args.full:
        print("=" * 74)
        print("WHAT EACH VIBE RETRIEVES")
        print("=" * 74)
        for i, vibe in enumerate(vibes):
            print(f"\n  [{vibe.axis}] {vibe.label}")
            print(f'      "{vibe.music_query}"')
            for rank in range(min(args.show, len(rows[i]))):
                row = rows[i][rank]
                if row < 0:
                    continue
                track = by_id.get(track_ids[row], {})
                name = (track.get("track_name") or "?")[:40]
                artist = (track.get("artist_name") or "?")[:24]
                print(f"      {scores[i][rank]:.3f}  {name:<40} {artist}")

    print("\n" + "=" * 74)
    print("PROBLEMS FOUND")
    print("=" * 74)

    # 1. dead vibes
    dead = [(v, float(scores[i][0])) for i, v in enumerate(vibes) if scores[i][0] < DEAD_SCORE]
    print(f"\nDEAD VIBES - nothing in the catalog sounds like this (top score < {DEAD_SCORE})")
    if dead:
        for vibe, score in sorted(dead, key=lambda x: x[1]):
            print(f"  {score:.3f}  [{vibe.axis}] {vibe.label}")
            print(f'         "{vibe.music_query}"')
    else:
        print("  none - every vibe finds something")

    # 2. twin vibes
    print(f"\nTWIN VIBES - retrieve the same songs (>{int(TWIN_OVERLAP*100)}% of top-10 shared)")
    twins = []
    for i in range(len(vibes)):
        for j in range(i + 1, len(vibes)):
            shared = len(top_sets[i] & top_sets[j])
            overlap = shared / max(1, min(len(top_sets[i]), len(top_sets[j])))
            if overlap > TWIN_OVERLAP:
                twins.append((overlap, vibes[i], vibes[j]))
    if twins:
        for overlap, a, b in sorted(twins, reverse=True, key=lambda x: x[0]):
            print(f"  {overlap:.0%} shared  {a.label} [{a.axis}]  <->  {b.label} [{b.axis}]")
    else:
        print("  none - every vibe retrieves a distinct set")

    # 3. hub songs
    print(f"\nHUB SONGS - appear under {HUB_APPEARANCES}+ unrelated vibes")
    from collections import Counter

    counts = Counter()
    for s in top_sets:
        counts.update(s)
    hubs = [(n, r) for r, n in counts.items() if n >= HUB_APPEARANCES]
    if hubs:
        for n, row in sorted(hubs, reverse=True)[:10]:
            track = by_id.get(track_ids[row], {})
            name = (track.get("track_name") or "?")[:40]
            artist = (track.get("artist_name") or "?")[:24]
            print(f"  in {n:>2} vibes  {name:<40} {artist}")
    else:
        print("  none - no song dominates")

    # ---------------------------------------------------------------- health
    print("\n" + "=" * 74)
    best = scores[:, 0]
    print(f"Best-match score: mean {best.mean():.3f}, min {best.min():.3f}, max {best.max():.3f}")
    print(f"Dead: {len(dead)}/{len(vibes)}   Twin pairs: {len(twins)}   Hubs: {len(hubs)}")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
