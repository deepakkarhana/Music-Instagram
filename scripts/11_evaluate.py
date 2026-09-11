"""STEP 11 - Prove it works, or find out that it does not.

    python scripts/11_evaluate.py                      on the dev photos
    python scripts/11_evaluate.py --dir data/photos    on your own photos

WHAT THIS ANSWERS
-----------------
Up to now the evidence has been "look, the songs seem reasonable". That is not
evidence. This script compares the full pipeline against the simpler things it
is supposed to beat, and reports numbers.

    random      draw songs out of a hat - the floor
    popular     the most prolific artists, ignoring the photo entirely
    keyword     string matching on titles and genres, no models at all
    OURS        photo -> CLIP -> vibe -> CLAP -> songs
    oracle      the music half given the CORRECT vibe - a ceiling, not a rival

The keyword baseline is the one that matters. If two neural networks cannot
beat `if "wedding" in genre`, the complexity has not earned itself.

The oracle is not something to beat. The gap between OURS and oracle is
exactly what the vision step is costing, which says where the next week of
work should go.

THREE METRICS, IMPERFECT IN DIFFERENT WAYS
------------------------------------------
    recall@5     how much of the oracle's top 20 a method recovers
    nDCG@5       the same, but rewarding the oracle's favourites first
    lane         for unambiguously Indian vibes, how many songs are Indian
    diversity    distinct artists over songs returned

`lane` is the honest one. That field came from the Week 1 catalogue and **no
model in the pipeline has ever seen it**, so nothing can game it. The other
two treat CLAP's judgement as truth, which is a real weakness and is why
there are three.
"""

import argparse
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import console
from src.agent import recommend as recommender
from src.encode import clap
from src.encode.store import read_catalog
from src.eval import baselines, metrics
from src.retrieve import index as index_module
from src.vibe import taxonomy, vision

CATALOG_PATH = os.path.join("data", "raw", "itunes_catalog.csv")
INDEX_PATH = os.path.join("data", "interim", "catalog")
DEV_DIR = os.path.join("data", "photos_dev")

ORACLE_POOL = 20  # how many oracle songs count as "relevant"
K = 5


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", default=DEV_DIR)
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--index", default=INDEX_PATH)
    parser.add_argument("--limit", type=int, default=0, help="only N photos")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    console.setup()

    manifest = os.path.join(args.dir, "manifest.csv")
    if not os.path.exists(manifest):
        print(f"No manifest.csv in {args.dir}")
        print("This needs photos with known vibes. Either use the development")
        print("set, or label your own photos first (Week 6 annotation tool).")
        return 1

    with open(manifest, "r", encoding="utf-8", newline="") as handle:
        labels = {r["file"]: r["vibe"] for r in csv.DictReader(handle)}

    index, track_ids = index_module.load(args.index)
    catalog_by_id = {str(r["track_id"]): r for r in read_catalog(args.catalog)}
    print(f"Index: {index.ntotal:,} songs")

    print("Loading CLIP ...")
    clip_model, clip_processor, device = vision.load(device=args.device)
    vision.require_working_encoder(clip_model, clip_processor, device)
    print("Loading CLAP ...")
    clap_model, clap_processor, _ = clap.load(device=args.device)
    clap.require_working_text_encoder(clap_model, clap_processor, device)

    vibe_image_vectors = vision.embed_text(
        clip_model, clip_processor, [v.visual_cues for v in taxonomy.VIBES], device)
    vibe_text_vectors = recommender.build_vibe_text_vectors(
        clap_model, clap_processor, device)

    files = sorted(
        f for f in os.listdir(args.dir)
        if f in labels and os.path.splitext(f)[1].lower() in (".jpg", ".jpeg", ".png", ".webp")
    )
    if args.limit:
        files = files[: args.limit]
    print(f"Photos: {len(files)} with known vibes\n")

    # Popularity ignores the photo, so compute it once.
    popular = baselines.popular_songs(track_ids, catalog_by_id, k=K)

    METHODS = ("random", "popular", "keyword", "keyword+", "ours", "oracle")
    rows = {name: [] for name in METHODS}
    vibe_hits = 0

    for number, filename in enumerate(files, start=1):
        gold = taxonomy.get(labels[filename])
        image = vision.load_image(os.path.join(args.dir, filename))
        if image is None:
            continue

        # --- the oracle defines what counts as relevant for this photo -----
        oracle_pool = baselines.oracle_songs(
            gold, vibe_text_vectors, taxonomy.VIBES, index, track_ids, k=ORACLE_POOL)

        # --- our pipeline --------------------------------------------------
        picked = recommender.photo_to_vibes(
            clip_model, clip_processor, device, image, vibe_image_vectors)
        vibe_hits += any(v.name == gold.name for v, _ in picked)

        query_vector, _ = recommender.vibes_to_query_vector(
            picked, vibe_text_vectors,
            clap_model=clap_model, clap_processor=clap_processor, device=device)
        ours = [
            str(t["track_id"]) for t in recommender.recommend(
                query_vector.reshape(1, -1), index, track_ids, catalog_by_id,
                k=K, per_artist=1)
        ]

        methods = {
            "random": baselines.random_songs(track_ids, k=K, seed=number),
            "popular": popular,
            # Two keyword baselines, because the fair comparison matters.
            # "keyword"  gets the vibe OUR pipeline predicted - same information
            #            we had, so it is a like-for-like rival.
            # "keyword+" gets the TRUE vibe, which is oracle-level knowledge it
            #            would never have in practice. It is here because the
            #            first version of this script accidentally gave keyword
            #            the true vibe while making ours guess, and the result
            #            is worth keeping: string matching on metadata is very
            #            strong at language, which is a real finding.
            "keyword": baselines.keyword_songs(
                picked[0][0], track_ids, catalog_by_id, k=K),
            "keyword+": baselines.keyword_songs(gold, track_ids, catalog_by_id, k=K),
            "ours": ours,
            "oracle": oracle_pool[:K],
        }

        # Measure BOTH lane directions. A one-sided check rewards a method that
        # always answers "Indian" - see src/eval/metrics.py for how that bug
        # made the popularity baseline look perfect.
        want = metrics.expected_lane(gold.name)
        for name, ids in methods.items():
            row = {
                "recall": metrics.recall_at_k(ids, oracle_pool, k=K),
                "ndcg": metrics.ndcg_at_k(ids, oracle_pool, k=K),
                "diversity": metrics.artist_diversity(ids, catalog_by_id),
            }
            if want == "indian":
                row["lane_in"] = metrics.lane_precision(ids, catalog_by_id, "indian")
            elif want == "english":
                row["lane_en"] = metrics.lane_precision(ids, catalog_by_id, "english")
            rows[name].append(row)

        if number % 20 == 0:
            print(f"  {number}/{len(files)} photos")

    # ---------------------------------------------------------------- report
    print("\n" + "=" * 74)
    print(f"RESULTS - {len(files)} photos, top-{K} recommendations")
    print("=" * 74)
    print(f"{'method':<11}{'recall@5':>10}{'nDCG@5':>9}{'divers':>8}"
          f"{'IN vibes':>10}{'EN vibes':>10}{'balanced':>10}")
    print("-" * 74)

    indian_count = sum(1 for f in files if taxonomy.get(labels[f]).name in metrics.INDIAN_VIBES)
    english_count = sum(1 for f in files if taxonomy.get(labels[f]).name in metrics.WESTERN_VIBES)

    for name in METHODS:
        stats = metrics.summarise(rows[name])
        lane_in = stats.get("lane_in")
        lane_en = stats.get("lane_en")
        balanced = (lane_in + lane_en) / 2 if lane_in is not None and lane_en is not None else None
        fmt = lambda v: f"{100*v:.0f}%" if v is not None else "-"
        mark = ("  <- ours" if name == "ours"
                else "  (by construction)" if name == "oracle"
                else "  (given the TRUE vibe)" if name == "keyword+" else "")
        print(f"{name:<11}{100*stats['recall']:>9.1f}%{stats['ndcg']:>9.3f}"
              f"{stats['diversity']:>8.2f}{fmt(lane_in):>10}{fmt(lane_en):>10}"
              f"{fmt(balanced):>10}{mark}")

    print("-" * 74)
    print(f"lane checked on {indian_count} Indian-coded and {english_count} "
          f"Western-coded photos; neutral vibes are skipped")
    print("  balanced = mean of the two, so a method that always answers one")
    print("  language scores ~50% no matter which it picks")
    print(f"vision step found the right vibe for {vibe_hits}/{len(files)} "
          f"({100*vibe_hits/max(1,len(files)):.1f}%) photos")

    ours = metrics.summarise(rows["ours"])
    oracle = metrics.summarise(rows["oracle"])
    keyword = metrics.summarise(rows["keyword"])
    print("\nREADING IT")
    if keyword["recall"] > 0:
        print(f"  vs keyword matching : {ours['recall']/max(1e-9, keyword['recall']):.1f}x recall")
    else:
        print("  vs keyword matching : keyword scored 0 - it found nothing relevant")
    print(f"  vs popularity       : {100*ours['recall']:.1f}% against "
          f"{100*metrics.summarise(rows['popular'])['recall']:.1f}%")
    print(f"  share of the oracle recovered : {100*ours['recall']:.1f}%")
    print("    The oracle scores 100% by construction - its top 5 sit inside its")
    print("    own top 20 - so that row is a reference point, not a result. What")
    print("    it means is that if the vision step were perfect we would recover")
    print("    everything; we recover "
          f"{100*ours['recall']:.0f}%, and the shortfall is the vision step's cost.")
    print("=" * 74)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
