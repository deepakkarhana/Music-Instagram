"""STEP 10 - Give it a photo. Get songs.

    python scripts/10_recommend.py path/to/photo.jpg
    python scripts/10_recommend.py --sample 5          try 5 dev photos
    python scripts/10_recommend.py photo.jpg --combine join

This is the whole project working end to end for the first time:

    photo -> CLIP -> vibes -> CLAP -> songs

Two models that have never met, joined only by a vocabulary of 40 names. CLIP
reads what the photo looks like. CLAP knows what music sounds like. The vibe
name carries meaning from one to the other.

WHAT TO LOOK AT
---------------
The vibes it picked matter as much as the songs. When a recommendation is
wrong, it is usually wrong at the vibe step, not the music step - and you can
see which by reading the vibes it chose. A beach photo tagged "melancholy"
will faithfully return sad music, and the retrieval was not the problem.

Measured on the development photos, the vision step gets the right vibe in its
top 3 about 42% of the time, against 7.5% for guessing. It is much better at
concrete things (occasion 63%, scene 56%) than at abstract ones (mood 23%,
light 17%), which is the single biggest weakness in the pipeline today.
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import console
from src.agent import recommend as recommender
from src.encode import clap
from src.encode.store import read_catalog
from src.retrieve import index as index_module
from src.vibe import taxonomy
from src.vibe import vision

CATALOG_PATH = os.path.join("data", "raw", "itunes_catalog.csv")
INDEX_PATH = os.path.join("data", "interim", "catalog")
DEV_DIR = os.path.join("data", "photos_dev")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("photo", nargs="*", help="photo file(s) to recommend for")
    parser.add_argument("--sample", type=int, default=0,
                        help="instead, try N random development photos")
    parser.add_argument("--dir", default=DEV_DIR, help="folder to sample from")
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("-k", type=int, default=5, help="songs to return")
    parser.add_argument("--combine", default="average", choices=("average", "join"),
                        help="how to merge several vibes into one music query")
    parser.add_argument("--per-artist", type=int, default=1,
                        help="max songs from one artist")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    console.setup()

    if not os.path.exists(args.index + ".faiss"):
        print(f"No index at {args.index}.faiss")
        print("Run:  python scripts/04_build_index.py")
        return 1

    photos = list(args.photo)
    if args.sample:
        pool = [
            os.path.join(args.dir, f)
            for f in sorted(os.listdir(args.dir))
            if os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".webp")
        ]
        random.seed(11)
        photos = random.sample(pool, min(args.sample, len(pool)))

    if not photos:
        print("Give me a photo, or use --sample 5 to try the development set.")
        return 1

    index, track_ids = index_module.load(args.index)
    catalog_by_id = {str(r["track_id"]): r for r in read_catalog(args.catalog)}
    print(f"Catalog: {index.ntotal:,} songs\n")

    print("Loading CLIP (reads photos) ...")
    clip_model, clip_processor, device = vision.load(device=args.device)
    vision.require_working_encoder(clip_model, clip_processor, device)

    print("Loading CLAP (knows music) ...")
    clap_model, clap_processor, _ = clap.load(device=args.device)
    clap.require_working_text_encoder(clap_model, clap_processor, device)
    print(f"Both healthy, running on {device.upper()}\n")

    # Precompute both sides of the vocabulary once.
    vibe_image_vectors = vision.embed_text(
        clip_model, clip_processor, [v.visual_cues for v in taxonomy.VIBES], device
    )
    vibe_text_vectors = recommender.build_vibe_text_vectors(
        clap_model, clap_processor, device
    )

    for path in photos:
        image = vision.load_image(path)
        if image is None:
            print(f"Could not read {path}")
            continue

        print("=" * 72)
        print(os.path.basename(path))
        print("=" * 72)

        picked = recommender.photo_to_vibes(
            clip_model, clip_processor, device, image, vibe_image_vectors
        )
        print("\n  CLIP sees:")
        for vibe, score in picked:
            print(f"    {score:.3f}  [{vibe.axis}] {vibe.label}")

        query_vector, description = recommender.vibes_to_query_vector(
            picked, vibe_text_vectors, combine=args.combine,
            clap_model=clap_model, clap_processor=clap_processor, device=device,
        )
        print(f"\n  Music query ({args.combine}):")
        print(f"    {description}")

        results = recommender.recommend(
            query_vector.reshape(1, -1), index, track_ids, catalog_by_id,
            k=args.k, per_artist=args.per_artist,
        )
        print("\n  Songs:")
        for rank, track in enumerate(results, start=1):
            name = (track.get("track_name") or "?")[:42]
            artist = (track.get("artist_name") or "?")[:24]
            year = track.get("release_year") or "----"
            print(f"    {rank}. {track['score']:.3f}  {name:<42} {artist:<24} {year}")

        if results:
            print(f"\n  Why: {recommender.explain(picked, results[0])}")
        print()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
