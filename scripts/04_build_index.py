"""STEP 4 - Build the search index from the embeddings.

Run this from the project root:

    python scripts/04_build_index.py

It reads every embedding produced by step 3 and packs them into a FAISS index,
which is the thing that answers "which songs are closest to this vector?" in
milliseconds.

This takes a couple of seconds. Rerun it any time you embed more songs.
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import console
from src.encode import clap
from src.encode.store import VectorStore, read_catalog
from src.retrieve import dedup as dedup_module
from src.retrieve import index as index_module

CATALOG_PATH = os.path.join("data", "raw", "itunes_catalog.csv")
STORE_PATH = os.path.join("data", "interim", "clap_audio")
INDEX_PATH = os.path.join("data", "interim", "catalog")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog", default=CATALOG_PATH)
    parser.add_argument("--store", default=STORE_PATH)
    parser.add_argument("--out", default=INDEX_PATH, help="where to save the index")
    parser.add_argument("--dedup-threshold", type=float, default=dedup_module.DEFAULT_THRESHOLD,
                        help="cosine above which two tracks count as the same recording")
    parser.add_argument("--no-dedup", action="store_true",
                        help="keep duplicate recordings in the index")
    args = parser.parse_args()

    console.setup()

    store = VectorStore(args.store, dim=clap.EMBED_DIM)
    track_ids, vectors = store.load()

    if len(track_ids) == 0:
        print("No embeddings found.")
        print("Run this first:  python scripts/03_embed_catalog.py")
        return 1

    print(f"Loaded {len(track_ids):,} embeddings of {vectors.shape[1]} numbers each.")

    # A quick sanity check worth keeping. Every vector should have length 1.
    # If this drifts, cosine similarity silently stops being cosine similarity
    # and every score in the project becomes subtly wrong.
    import numpy as np

    lengths = np.linalg.norm(vectors, axis=1)
    print(f"Vector lengths: min {lengths.min():.4f}, max {lengths.max():.4f} (should be 1.0000)")

    # Drop songs that are the same recording twice. iTunes lists a track on the
    # single, the album and three compilations, so without this a search can
    # return the same song five times.
    if not args.no_dedup:
        keep = dedup_module.find_keepers(vectors, threshold=args.dedup_threshold)
        kept, dropped = dedup_module.summarise(keep)
        print(f"Deduplication: dropped {dropped:,} duplicate recordings "
              f"(cosine >= {args.dedup_threshold}), {kept:,} songs remain.")
        vectors = vectors[keep]
        track_ids = [t for t, k in zip(track_ids, keep) if k]

    index = index_module.build(vectors)
    index_module.save(index, track_ids, args.out)

    size_mb = os.path.getsize(args.out + ".faiss") / (1024 * 1024)
    print(f"\nIndex saved to {args.out}.faiss ({size_mb:.1f} MB), {index.ntotal:,} songs.")

    # Report how much of the catalog actually made it in.
    if os.path.exists(args.catalog):
        catalog = read_catalog(args.catalog)
        coverage = 100.0 * len(track_ids) / len(catalog) if catalog else 0
        print(f"Coverage: {len(track_ids):,} of {len(catalog):,} catalog songs ({coverage:.1f}%).")

    print("\nNext:  python scripts/05_search.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
