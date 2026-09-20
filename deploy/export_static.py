"""Export everything a browser needs to run this with no server at all.

    python deploy/export_static.py

WHY THIS EXISTS
---------------
Hugging Face now charges for Spaces that run code. Only Static Spaces - plain
files, no Python - stay free. Every other free host worth using gives about
512 MB of memory, and CLIP and CLAP need roughly 2 GB between them.

So the server had to go. Which turned out to be possible, because of something
that was true all along and easy to miss:

**CLAP never runs when someone uploads a photo.**

Look at what a request actually does. The 40 vibes' music descriptions are
fixed, so their CLAP vectors are fixed. Every song's CLAP vector was computed
back in Week 2 and never changes. The only thing that depends on the photo is
CLIP's image encoder.

    fixed, precomputed    40 vibe visual_cues  -> CLIP text vectors
    fixed, precomputed    40 vibe music_query  -> CLAP text vectors
    fixed, precomputed    11,870 songs         -> CLAP audio vectors
    per request           the photo            -> CLIP image vector

Everything after that is dot products. A browser can do dot products.

So this exports the three fixed pieces, and the page runs CLIP's image encoder
in WebAssembly. No server, no hosting cost, ever - and it keeps working if
nobody pays a bill for the next decade, which is the right property for
something on a CV.

WHY int8
--------
The song vectors are 24 MB as float32 and 6 MB as int8. Measured cost of that
quantisation: a maximum cosine error of 0.0015, against score differences
between songs of 0.02 and up. Four times smaller for an error two orders of
magnitude below the signal.

Vectors are unit length, so every value already sits in [-1, 1] and a single
global scale of 127 works - no per-vector scales to store or apply.
"""

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import console
from src.agent import recommend as recommender
from src.encode import clap
from src.encode.store import read_catalog
from src.retrieve import filters, index as index_module
from src.vibe import taxonomy, vision

CATALOG_PATH = os.path.join("data", "raw", "itunes_catalog.csv")
INDEX_PATH = os.path.join("data", "interim", "catalog")
OUT_DIR = os.path.join("app", "static", "data")


def quantise(vectors):
    """Unit-length float32 rows -> int8, one global scale of 127."""
    return np.clip(np.round(np.asarray(vectors, dtype=np.float32) * 127), -127, 127).astype(np.int8)


def main():
    console.setup()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading index ...")
    index, track_ids = index_module.load(INDEX_PATH)
    catalog = {str(r["track_id"]): r for r in read_catalog(CATALOG_PATH)}
    song_vectors = index.reconstruct_n(0, index.ntotal)

    print("Loading CLIP (for the vibe text vectors) ...")
    clip_model, clip_processor, device = vision.load()
    vision.require_working_encoder(clip_model, clip_processor, device)

    print("Loading CLAP (for the vibe music vectors) ...")
    clap_model, clap_processor, _ = clap.load(device=device)
    clap.require_working_text_encoder(clap_model, clap_processor, device)

    vibe_clip = vision.embed_text(
        clip_model, clip_processor, [v.visual_cues for v in taxonomy.VIBES], device)
    vibe_clap = recommender.build_vibe_text_vectors(
        clap_model, clap_processor, device)

    # ---------------------------------------------------------------- songs
    kept_ids, kept_rows = [], []
    for position, track_id in enumerate(track_ids):
        if track_id in catalog and catalog[track_id].get("preview_url"):
            kept_ids.append(track_id)
            kept_rows.append(position)

    vectors = song_vectors[kept_rows]
    quantised = quantise(vectors)

    # Base64 inside JSON, not a raw .i8 file. Hugging Face rejects binary
    # files pushed through ordinary git and asks for Xet or LFS instead, which
    # means another tool to install and another thing to go wrong on a machine
    # that is not mine. Base64 is text, so it travels as plain source.
    #
    # Costs 33% - 6.1 MB becomes 8.1 MB - and gzip over the wire claws most of
    # that back, because base64 of quantised vectors compresses well.
    import base64

    encoded = base64.b64encode(quantised.tobytes()).decode("ascii")
    with open(os.path.join(OUT_DIR, "vectors.json"), "w", encoding="utf-8") as handle:
        json.dump({"rows": len(kept_ids), "dim": int(vectors.shape[1]),
                   "scale": 127, "b64": encoded}, handle, separators=(",", ":"))

    # Short keys: this file is downloaded by every visitor, and "track_name"
    # repeated 11,870 times is a megabyte of the same word.
    meta = [
        {
            "n": catalog[t].get("track_name", ""),
            "a": catalog[t].get("artist_name", ""),
            "y": catalog[t].get("release_year", ""),
            "g": catalog[t].get("genre", ""),
            "l": catalog[t].get("lane", ""),
            "u": catalog[t].get("preview_url", ""),
        }
        for t in kept_ids
    ]
    with open(os.path.join(OUT_DIR, "songs.json"), "w", encoding="utf-8") as handle:
        json.dump(meta, handle, separators=(",", ":"), ensure_ascii=False)

    # ---------------------------------------------------------------- vibes
    vibes = [
        {
            "name": v.name,
            "label": v.label,
            "axis": v.axis,
            "cues": v.visual_cues,
            "music": v.music_query,
            "lane": filters.wanted_lane([(v, 1.0)]) or "",
        }
        for v in taxonomy.VIBES
    ]
    with open(os.path.join(OUT_DIR, "vibes.json"), "w", encoding="utf-8") as handle:
        json.dump({
            "vibes": vibes,
            "axes": list(taxonomy.AXES),
            "dim": int(vectors.shape[1]),
            "songs": len(kept_ids),
            # The vibe vectors stay FULL PRECISION. There are only 40 of
            # them - 80 KB against the song file's 6 MB - and quantising them
            # was measurably wrong: it shifted vibe scores enough to flip
            # close calls, so the browser chose different vibes from the
            # server on 3 of 8 test photos. The songs can absorb int8 because
            # a 0.0015 error sits far below the gaps between them; a vibe
            # decision is often won by less than that.
            "clip": [round(float(x), 6) for x in vibe_clip.flatten()],
            "clap": [round(float(x), 6) for x in vibe_clap.flatten()],
        }, handle, separators=(",", ":"))

    # ------------------------------------------- the page a Static Space serves
    # Hugging Face Static Spaces serve index.html from the repository root, so
    # the page is written there with its data paths adjusted. Keeping the
    # source in app/static/ means the browser and server versions live side by
    # side and can be compared.
    source = os.path.join("app", "static", "standalone.html")
    with open(source, "r", encoding="utf-8") as handle:
        page = handle.read()
    page = page.replace("fetch('data/", "fetch('app/static/data/")
    with open("index.html", "w", encoding="utf-8", newline="\n") as handle:
        handle.write(page)
    print("wrote index.html (the Static Space entry point)")

    # --------------------------------------------------------------- report
    print()
    for name in ("vectors.json", "songs.json", "vibes.json"):
        size = os.path.getsize(os.path.join(OUT_DIR, name)) / 1e6
        print(f"  {name:<14} {size:>6.1f} MB")
    print(f"\n{len(kept_ids):,} songs exported to {OUT_DIR}")
    print("\nThe page loads these plus CLIP's ONNX weights from Hugging Face's")
    print("CDN. After that it runs entirely in the browser - no server, and")
    print("nothing to pay for.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
