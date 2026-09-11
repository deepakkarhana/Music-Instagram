"""The web demo: drop a photo in a browser, get songs you can play.

    python app/main.py
    then open http://127.0.0.1:8000

WHY A REAL SERVER AND NOT A NOTEBOOK
------------------------------------
Everything so far has been scripts. Scripts prove the idea works; they do not
show anyone what it feels like. Dropping your own photo into a page and
hearing the songs it picked is a different kind of evidence - and it is the
thing worth putting in front of someone in an interview.

It is also the first time the project has to behave like software rather than
an experiment: load the models once instead of per request, fail politely on a
file that is not an image, and answer fast enough that a person waits happily.

LOADING ONCE, NOT PER REQUEST
-----------------------------
CLIP and CLAP take roughly half a minute to load and about 2 GB of memory.
Doing that per request would make every upload unusable.

So they are loaded once at startup, along with the FAISS index and the
vocabulary's two sets of vectors. After that a request is: embed one photo,
score it against 40 vibes, average a few vectors, search 11,870 songs. All of
that is milliseconds. The slow part is reading the photo off the wire.

WHY THE PREVIEWS PLAY
---------------------
iTunes preview URLs can be handed straight to an HTML <audio> element, so the
browser streams the 30 seconds directly from Apple. We never host, store or
re-serve any audio - which is the same rule the rest of the project follows,
and the reason this is publishable.
"""

import io
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse

from src.agent import recommend as recommender
from src.encode import clap
from src.encode.store import read_catalog
from src.retrieve import index as index_module
from src.vibe import taxonomy
from src.vibe import vision

CATALOG_PATH = os.path.join("data", "raw", "itunes_catalog.csv")
INDEX_PATH = os.path.join("data", "interim", "catalog")
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

MAX_UPLOAD_BYTES = 15 * 1024 * 1024  # a phone photo is 3-8 MB; 15 is generous

app = FastAPI(title="Music-Instagram", docs_url="/api/docs")

# Everything expensive, loaded once. Populated by `startup`.
STATE = {}


@app.on_event("startup")
def startup():
    """Load the models and index once, before the first request."""
    started = time.time()
    print("Loading index ...")
    index, track_ids = index_module.load(INDEX_PATH)
    catalog_by_id = {str(r["track_id"]): r for r in read_catalog(CATALOG_PATH)}

    print("Loading CLIP (reads photos) ...")
    clip_model, clip_processor, device = vision.load()
    vision.require_working_encoder(clip_model, clip_processor, device)

    print("Loading CLAP (knows music) ...")
    clap_model, clap_processor, _ = clap.load(device=device)
    clap.require_working_text_encoder(clap_model, clap_processor, device)

    print("Embedding the vocabulary ...")
    STATE.update({
        "index": index,
        "track_ids": track_ids,
        "catalog_by_id": catalog_by_id,
        "clip": (clip_model, clip_processor),
        "clap": (clap_model, clap_processor),
        "device": device,
        "vibe_image_vectors": vision.embed_text(
            clip_model, clip_processor,
            [v.visual_cues for v in taxonomy.VIBES], device),
        "vibe_text_vectors": recommender.build_vibe_text_vectors(
            clap_model, clap_processor, device),
    })
    print(f"Ready in {time.time() - started:.0f}s on {device.upper()} - "
          f"{index.ntotal:,} songs, {len(taxonomy.VIBES)} vibes")
    print("Open http://127.0.0.1:8000")


@app.get("/")
def home():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/health")
def health():
    """Enough to tell whether the thing is actually up and loaded."""
    if not STATE:
        return JSONResponse({"ready": False}, status_code=503)
    return {
        "ready": True,
        "songs": STATE["index"].ntotal,
        "vibes": len(taxonomy.VIBES),
        "device": STATE["device"],
    }


@app.post("/api/recommend")
async def recommend_endpoint(photo: UploadFile = File(...), k: int = 5):
    """Take a photo, return the vibes it shows and the songs that fit."""
    if not STATE:
        raise HTTPException(503, "Still starting up - try again in a moment.")

    raw = await photo.read()
    if not raw:
        raise HTTPException(400, "That file was empty.")
    if len(raw) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            413, f"That photo is {len(raw)/1e6:.0f} MB. Please keep it under 15 MB."
        )

    # Reuse the same loader the scripts use, so the browser path and the
    # command-line path cannot drift apart - including the EXIF rotation fix.
    image = vision.load_image(io.BytesIO(raw))
    if image is None:
        raise HTTPException(
            400, "Could not read that as an image. JPEG, PNG or WebP please - "
                 "iPhone HEIC files need converting first."
        )

    started = time.time()
    clip_model, clip_processor = STATE["clip"]
    clap_model, clap_processor = STATE["clap"]
    device = STATE["device"]

    picked = recommender.photo_to_vibes(
        clip_model, clip_processor, device, image, STATE["vibe_image_vectors"]
    )
    query_vector, description = recommender.vibes_to_query_vector(
        picked, STATE["vibe_text_vectors"],
        clap_model=clap_model, clap_processor=clap_processor, device=device,
    )
    results = recommender.recommend(
        query_vector.reshape(1, -1), STATE["index"], STATE["track_ids"],
        STATE["catalog_by_id"], k=max(1, min(k, 10)), per_artist=1,
    )

    return {
        "vibes": [
            {"name": v.name, "label": v.label, "axis": v.axis, "score": round(s, 4)}
            for v, s in picked
        ],
        "query": description,
        "songs": [
            {
                "track_name": t.get("track_name", ""),
                "artist_name": t.get("artist_name", ""),
                "release_year": t.get("release_year", ""),
                "genre": t.get("genre", ""),
                "lane": t.get("lane", ""),
                "preview_url": t.get("preview_url", ""),
                "score": round(t["score"], 4),
            }
            for t in results
        ],
        "why": recommender.explain(picked, results[0]) if results else "",
        "took_ms": int((time.time() - started) * 1000),
    }


if __name__ == "__main__":
    import uvicorn

    if not os.path.exists(INDEX_PATH + ".faiss"):
        print(f"No index at {INDEX_PATH}.faiss")
        print("Run:  python scripts/04_build_index.py")
        raise SystemExit(1)

    print("Starting up - the models take about half a minute to load.\n")
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="warning")
