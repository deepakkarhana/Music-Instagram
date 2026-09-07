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
    --in-order       embed in catalog order (default is a fixed-seed shuffle,
                     so stopping early still leaves a representative sample)
    --mirror PATH    copy results to PATH every 10 batches. Use this on Colab
                     with a Google Drive folder - Colab deletes its own disk
                     when you disconnect, and an hour of work goes with it
"""

import argparse
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor

# Make `import src...` work when running this file directly.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import console
from src.encode import audio as audio_module
from src.encode import clap
from src.encode import store as store_module
from src.encode.store import VectorStore, read_catalog

CATALOG_PATH = os.path.join("data", "raw", "itunes_catalog.csv")
STORE_PATH = os.path.join("data", "interim", "clap_audio")


def fetch_and_decode(track):
    """Download one preview and decode it. Runs in a worker thread.

    Only these two steps belong in a thread. Downloading waits on the network
    and decoding happens inside FFmpeg's C code, and both release Python's
    global interpreter lock, so they genuinely overlap.

    THE MEL STEP DOES NOT BELONG HERE, AND MEASURING SAID SO
    --------------------------------------------------------
    Computing the mel spectrogram was moved into this function on the
    reasoning that it is "just NumPy maths, which releases the GIL". That was
    wrong. Hugging Face's feature extractor runs Python-level loops, so it
    holds the lock the whole time. Measured on 8 songs:

        serial     1.27s
        8 threads  7.84s      <- six times SLOWER

    Threads cannot overlap work that holds the GIL; they only add contention.
    Speeding this up needs separate *processes*, not threads - which is a
    Week 10 problem, not a Week 2 one.
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
    parser.add_argument("--mirror", default=None,
                        help="folder to copy results into periodically (e.g. Google Drive)")
    parser.add_argument("--mirror-every", type=int, default=10,
                        help="copy to --mirror after this many batches")
    parser.add_argument("--fp16", action="store_true",
                        help="16-bit on GPU: ~2x faster, but untested - check your results")
    parser.add_argument("--in-order", action="store_true",
                        help="embed in catalog order instead of shuffled")
    args = parser.parse_args()

    console.setup()

    if not os.path.exists(args.catalog):
        print(f"No catalog at {args.catalog}")
        print("Run this first:  python scripts/01_harvest_itunes.py")
        return 1

    # If a previous run mirrored its work somewhere durable, bring it back
    # before deciding what still needs doing. This is what makes a wiped Colab
    # machine resume instead of starting over.
    if args.mirror:
        restored = store_module.restore_from_mirror(args.store, args.mirror)
        if restored:
            print(f"Restored {restored} files from {args.mirror}")
            print()

    catalog = read_catalog(args.catalog)
    store = VectorStore(args.store, dim=clap.EMBED_DIM)

    # Shuffle with a fixed seed, so that stopping early still leaves a
    # *representative* sample rather than the first N rows.
    #
    # This matters more than it looks. The catalog is ordered by search term,
    # so the first few thousand rows are all Indian-lane tracks. Embedding
    # "the first 2000" would build an index that cannot answer an English
    # query, and the demo would look broken for reasons that have nothing to
    # do with the model. A fixed seed keeps runs reproducible and resumable.
    if not args.in_order:
        random.Random(20260906).shuffle(catalog)

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
    model, processor, device = clap.load(args.model, device=args.device, half=args.fp16)

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

                # Download and decode the batch in parallel - these do release
                # the GIL, so threads genuinely help here.
                results = list(pool.map(fetch_and_decode, batch))

                # Feature extraction stays on the main thread. See the note in
                # fetch_and_decode: threading it made things six times slower.
                usable = []
                for track, wave in results:
                    if wave is None:
                        writer.append_failed(str(track["track_id"]))
                        failed += 1
                        continue
                    inputs, count = clap.prepare_audio(processor, wave)
                    if inputs is None:
                        writer.append_failed(str(track["track_id"]))
                        failed += 1
                        continue
                    usable.append((track, inputs, count))

                # One GPU call for the entire batch.
                if usable:
                    vectors = clap.embed_prepared(
                        model,
                        [i for _, i, _ in usable],
                        [c for _, _, c in usable],
                        device,
                    )
                    for (track, _, _), vector in zip(usable, vectors):
                        writer.append(str(track["track_id"]), vector)
                        embedded += 1

                writer.flush()

                # Copy to durable storage every so often. Not every batch -
                # these are whole-file copies, and Drive is slow.
                batch_number = start // args.batch + 1
                if args.mirror and batch_number % args.mirror_every == 0:
                    store_module.mirror(args.store, args.mirror)

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
