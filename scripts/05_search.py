"""STEP 5 - Search the catalog with a sentence.

Run this from the project root:

    python scripts/05_search.py "wide open mountains at sunset, calm and nostalgic"

Or with no arguments to run the built-in demo, which fires a handful of
photo-style vibe descriptions at the catalog and prints what comes back:

    python scripts/05_search.py

WHY THIS SCRIPT MATTERS MORE THAN IT LOOKS
------------------------------------------
There is no photo here yet. That is deliberate. This is the moment we find out
whether the *core assumption of the whole project* actually holds:

    a sentence describing a mood can retrieve music that matches that mood

If this works, Week 5's demo is mostly plumbing - swap the hand-written
sentence for one a vision model writes about your photo. If it does not work,
we need to know now, in Week 2, not in Week 9.

Read the results honestly. Some will be uncanny. Some will be nonsense. Both
are information.

THE LANGUAGE TEST
-----------------
    python scripts/05_search.py --language-test

CLAP was trained overwhelmingly on English captions. Nobody has told us how it
handles Hindi and Punjabi *vocals*. This mode asks it directly, and prints
which lane each result came from so you can see whether the model is actually
distinguishing Indian music or just guessing.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import console
from src.encode import clap
from src.encode.store import read_catalog
from src.retrieve import index as index_module

CATALOG_PATH = os.path.join("data", "raw", "itunes_catalog.csv")
INDEX_PATH = os.path.join("data", "interim", "catalog")

# Sentences of the kind a vision model will eventually write about a photo.
# Each one targets a different corner of the space, so a catalog that returns
# the same five songs for all of them is obviously broken.
DEMO_QUERIES = [
    "wide open mountains at golden hour, calm, vast and nostalgic acoustic music",
    "old money elegance, vintage wealth, smooth orchestral jazz with strings",
    "rainy café window, soft melancholy, quiet piano and gentle acoustic guitar",
    "high energy gym workout, aggressive hip hop with heavy bass",
    "indian wedding celebration, joyful dhol drums and festive punjabi dance music",
    "late night city drive alone, moody synth, slow electronic",
]

# Same six moods, but phrased to ask for Indian music specifically. If CLAP
# understands Indian audio, these should pull noticeably more 'indian' lane
# results than the English phrasings above.
LANGUAGE_TEST_QUERIES = [
    ("english phrasing", "romantic love song, tender and emotional"),
    ("hindi phrasing", "romantic hindi bollywood love song with soft male vocals"),
    ("punjabi phrasing", "punjabi bhangra dance song with dhol and energetic vocals"),
    ("instrumental phrasing", "indian classical sitar and tabla, meditative and slow"),
]


def show(results, show_lane=False):
    """Print one result list as a small table."""
    if not results:
        print("    (nothing found)")
        return

    for rank, track in enumerate(results, start=1):
        name = (track.get("track_name") or "?")[:42]
        artist = (track.get("artist_name") or "?")[:26]
        year = track.get("release_year") or "----"
        lane = f"  [{track.get('lane', '?')}]" if show_lane else ""
        print(f"    {rank}. {track['score']:.3f}  {name:<42}  {artist:<26} {year}{lane}")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("query", nargs="*", help="the sentence to search with")
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("-k", type=int, default=5, help="how many songs to return")
    parser.add_argument("--language-test", action="store_true",
                        help="check how CLAP handles Indian music")
    parser.add_argument("--device", default=None, help="force 'cpu' or 'cuda'")
    parser.add_argument("--model", default=clap.MODEL_NAME, help="CLAP checkpoint to use")
    args = parser.parse_args()

    console.setup()

    if not os.path.exists(args.index + ".faiss"):
        print(f"No index at {args.index}.faiss")
        print("Run this first:  python scripts/04_build_index.py")
        return 1

    index, track_ids = index_module.load(args.index)
    catalog = read_catalog(args.catalog)
    catalog_by_id = {str(row["track_id"]): row for row in catalog}
    print(f"Index: {index.ntotal:,} songs\n")

    print("Loading CLAP...")
    model, processor, device = clap.load(args.model, device=args.device)
    print(f"Running on {device.upper()}\n")

    def run(text, show_lane=False):
        vector = clap.embed_text(model, processor, [text], device)
        results = index_module.search_by_text(
            index, track_ids, catalog_by_id, vector, k=args.k
        )
        show(results, show_lane=show_lane)

    if args.language_test:
        print("=" * 72)
        print("LANGUAGE TEST - does CLAP actually understand Indian music?")
        print("=" * 72)
        print("\nWatch the [lane] tag. If the Indian-specific queries return")
        print("mostly [indian] tracks, CLAP is hearing the difference. If the")
        print("lanes look random, it is not - and we handle language with")
        print("metadata filters instead.\n")

        for label, query in LANGUAGE_TEST_QUERIES:
            print(f"\n  {label}: \"{query}\"")
            run(query, show_lane=True)
        print()
        return 0

    if args.query:
        query = " ".join(args.query)
        print(f'"{query}"')
        run(query)
        print()
        return 0

    print("=" * 72)
    print("DEMO - six photo-style descriptions, no photos yet")
    print("=" * 72)
    for query in DEMO_QUERIES:
        print(f'\n  "{query}"')
        run(query)
    print("\nTry your own:")
    print('  python scripts/05_search.py "your description here"')
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
