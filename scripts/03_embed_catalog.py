"""STEP 3 - Turn every song in the catalog into 512 numbers.

Run this from the project root:

    python scripts/03_embed_catalog.py

For each song it streams the 30-second preview, plays it to CLAP, saves the
resulting embedding, and throws the audio away. Nothing is written to disk
except numbers.

HOW LONG IT TAKES
-----------------
    GPU (Colab T4)   about 20 minutes for 10,000 songs
    CPU (laptop)     several hours

Stop it any time with Ctrl+C. Rerunning resumes from exactly where it stopped,
so doing it in three evening sessions is completely fine.

USEFUL FLAGS
------------
    --quick          only 60 songs, to prove the pipeline works end to end
    --limit 2000     stop after this many new songs this run
    --batch 16       songs processed per batch (lower it if you run out of memory)
    --workers 6      parallel downloads (raise on fast wifi, lower if unstable)
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Make `import src...` work when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import console
from src.encode import audio as audio_module
from src.encode import clap
from src.encode.store import VectorStore, read_catalog

CATALOG_PATH = os.path.join("data", "raw", "itunes_catalog.csv")
STORE_PATH = os.path.join("data", "interim", "clap_audio")


def fetch_and_decode(track):
    """Download one preview and decode it. Runs in a worker thread.

    Downloading waits on the network and decoding is done by C code, so
    several of these genuinely run at the same time - Python's GIL is released
    for both. This is why downloads are parallel but embedding is not.
    """
    wave = audio_module.load(track["preview_url"])
    return track, wave


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--catalog", default=CATALOG_PATH, help="input catalog CSV")
    parser.add_argument("--store", default=STORE_PATH, help="where to save embeddings")
    parser.add_argument("--batch", type=int, default=16, help="songs per batch")
    parser.add_argument("--workers", type=int, default=6, help="parallel downloads")
    parser.add_argument("--limit", type=int, default=0, help="stop after N new songs")
    parser.add_argument("--quick", action="store_true", help="just 60 songs, as a test")
    parser.add_argument("--device", default=None, help="force 'cpu' or 'cuda'")
    parser.add_argument("--model", default=clap.MODEL_NAME, help="CLAP checkpoint to use")
    args = parser.parse_args()

    console.setup()

    if not os.path.exists(args.catalog):
        print(f"No catalog at {args.catalog}")
        print("Run this first:  python scripts/01_harvest_itunes.py")
        return 1

    catalog = read_catalog(args.catalog)
    store = VectorStore(args.store, dim=clap.EMBED_DIM)

    # Skip anything already embedded, and anything already known to be broken.
    skip = store.skip_ids()
    todo = [t for t in catalog if str(t["track_id"]) not in skip and t.get("preview_url")]

    if skip:
        print(f"Resuming: {len(skip):,} songs already handled\n")

    if args.quick:
        todo = todo[:60]
    elif args.limit:
        todo = todo[: args.limit]

    if not todo:
        print(f"Nothing left to do - all {len(catalog):,} songs are embedded.")
        print("\nNext:  python scripts/04_build_index.py")
        return 0

    print(f"Catalog:  {len(catalog):,} songs")
    print(f"To embed: {len(todo):,} songs\n")

    print(f"Loading CLAP ({args.model})...")
    print("The first run downloads about 600 MB. After that it is cached.\n")
    model, processor, device = clap.load(args.model, device=args.device)

    # Before spending hours on this, prove the model can actually tell two
    # different sentences apart. A checkpoint whose text encoder is dead
    # produces embeddings that look fine and retrieve nothing.
    offset = clap.require_working_text_encoder(model, processor, device)
    print(f"Text encoder health check passed (offset {offset:.4f}, want < 0.98)")
    print(f"Running on: {device.upper()}")
    if device == "cpu":
        print("No GPU found. This works, it is just slow - and it resumes,")
        print("so you can stop and restart as often as you like.\n")
    else:
        print()

    embedded = 0
    failed = 0
    started = time.time()

    with store as writer, ThreadPoolExecutor(max_workers=args.workers) as pool:
        try:
            for start in range(0, len(todo), args.batch):
                batch = todo[start : start + args.batch]

                # Download the whole batch in parallel, then embed it in one go.
                for track, wave in pool.map(fetch_and_decode, batch):
                    track_id = str(track["track_id"])

                    if wave is None:
                        writer.append_failed(track_id)
                        failed += 1
                        continue

                    vector = clap.embed_track(model, processor, wave, device)
                    if vector is None:
                        writer.append_failed(track_id)
                        failed += 1
                        continue

                    writer.append(track_id, vector)
                    embedded += 1

                writer.flush()

                done = embedded + failed
                elapsed = time.time() - started
                rate = done / elapsed if elapsed > 0 else 0
                remaining = (len(todo) - done) / rate if rate > 0 else 0
                print(
                    f"  {done:>6,}/{len(todo):,}  "
                    f"embedded {embedded:,}  failed {failed:,}  "
                    f"{rate:.1f} songs/sec  ~{remaining / 60:.0f} min left"
                )

        except KeyboardInterrupt:
            print("\nStopped by you. Everything so far is saved.")
            print("Run the script again to pick up from here.")

    ids, vectors = store.load()
    print(f"\nDone. {len(ids):,} songs embedded, {vectors.shape[1]} numbers each.")
    if failed:
        print(f"{failed:,} previews were dead and got skipped - that is normal.")
    print("\nNext:  python scripts/04_build_index.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
