"""Photo in, songs out. The whole pipeline in one place.

THE CHAIN
---------
    photo
      -> CLIP image encoder                     (what does this look like?)
      -> score against every vibe's visual_cues
      -> pick the strongest vibes
      -> take those vibes' music_query strings   (what should this sound like?)
      -> CLAP text encoder
      -> FAISS search over 11,870 song embeddings
      -> ranked songs

CLIP and CLAP never meet. They were trained by different people, at different
times, on different data, and neither knows the other exists. The vocabulary
in `taxonomy.py` is the entire bridge: a vibe is a name with two descriptions
attached, one that CLIP can read and one that CLAP can.

That is the idea the whole project rests on, and this file is where it either
works or does not.

HOW SEVERAL VIBES BECOME ONE MUSIC QUERY
----------------------------------------
A photo is rarely one vibe. A friend on a beach at sunset is a place, a time
and a mood at once, so we take the best vibe on each axis and have to combine
them into a single search.

Two obvious ways, and they are not equivalent:

**Join the text.** Paste the music queries together and embed the result.
Simple, but CLAP has a token limit and long queries drift toward mush - the
more you add, the more it describes nothing in particular.

**Average the vectors.** Embed each vibe's query separately, then take a
weighted average, weighting by how confident CLIP was about that vibe. The
average of "airy acoustic guitar" and "warm brass, mellow" lands between
them, which is what a photo showing both should retrieve.

`--combine` switches between them, because which one is better is a question
to measure rather than assume.

WHY WEIGHT BY THE CLIP SCORE
----------------------------
If CLIP is 0.31 confident the photo shows mountains and 0.19 confident it is
golden hour, the music should lean toward the mountains. Unweighted averaging
would treat a strong signal and a weak guess as equals, which throws away the
one piece of information we have about how sure the model is.
"""

import numpy as np

from src.encode import clap
from src.retrieve import index as index_module
from src.vibe import taxonomy
from src.vibe import vision


def build_vibe_text_vectors(clap_model, clap_processor, device, vibes=None):
    """CLAP embeddings of every vibe's music_query. Computed once, reused."""
    vibes = vibes or taxonomy.VIBES
    return clap.embed_text(
        clap_model, clap_processor, [v.music_query for v in vibes], device
    )


def photo_to_vibes(clip_model, clip_processor, device, image, vibe_image_vectors,
                   vibes=None, per_axis=True, top_k=3):
    """Decide which vibes a photo shows.

    Returns a list of (vibe, clip_score), strongest first. With `per_axis` it
    takes the best on each axis rather than the top few overall - otherwise a
    beach photo returns "beach", "bright daylight" and "sea" as three near
    duplicates and the music query says one thing three times.
    """
    vibes = vibes or taxonomy.VIBES
    image_vector = vision.embed_images(clip_model, clip_processor, [image], device)
    scores = vision.score_vibes(image_vector, vibe_image_vectors)[0]
    return vision.top_vibes(scores, vibes, k=top_k, per_axis=per_axis)


def vibes_to_query_vector(picked, vibe_text_vectors, vibes=None, combine="average",
                          clap_model=None, clap_processor=None, device=None):
    """Turn the chosen vibes into a single CLAP query vector.

    combine="average"  weighted mean of each vibe's query vector (default)
    combine="join"     paste the query strings together and embed once
    """
    vibes = vibes or taxonomy.VIBES
    order = {v.name: i for i, v in enumerate(vibes)}

    if combine == "join":
        text = ", ".join(vibe.music_query for vibe, _ in picked)
        return clap.embed_text(clap_model, clap_processor, [text], device)[0], text

    # Weight by CLIP's confidence, but only by how much each vibe beat the
    # weakest one. Raw cosine scores sit in a narrow band well above zero, so
    # using them directly would make every vibe roughly equally important.
    raw = np.array([score for _, score in picked], dtype=np.float32)
    weights = raw - raw.min() + 1e-3
    weights = weights / weights.sum()

    stack = np.vstack([vibe_text_vectors[order[vibe.name]] for vibe, _ in picked])
    combined = (stack * weights[:, None]).sum(axis=0, keepdims=True)

    length = np.linalg.norm(combined)
    if length > 0:
        combined = combined / length

    description = " + ".join(f"{v.label} ({w:.0%})" for (v, _), w in zip(picked, weights))
    return combined[0].astype(np.float32), description


def recommend(query_vector, index, track_ids, catalog_by_id, k=5, per_artist=1):
    """Search the catalog and return readable results.

    `per_artist` caps how many songs one artist can contribute. Without it a
    single prolific stock-music label can fill the whole list, which is a bad
    recommendation even when every individual match is correct.
    """
    # Ask for extra, because the per-artist cap will discard some.
    scores, rows = index_module.search(index, query_vector, k=min(k * 8, index.ntotal))

    seen_artists = {}
    results = []
    for score, row in zip(scores[0], rows[0]):
        if row < 0:
            continue
        track = dict(catalog_by_id.get(track_ids[row], {}))
        artist = (track.get("artist_name") or "?").lower()
        if seen_artists.get(artist, 0) >= per_artist:
            continue
        seen_artists[artist] = seen_artists.get(artist, 0) + 1
        track["score"] = float(score)
        results.append(track)
        if len(results) >= k:
            break

    return results


def explain(picked, track):
    """One line saying why this song was chosen.

    Deliberately plain. It names the vibes that drove the query rather than
    inventing a story about the music, because naming the actual cause is
    honest and inventing a rationale is not.
    """
    labels = ", ".join(vibe.label.lower() for vibe, _ in picked[:2])
    name = track.get("track_name") or "this track"
    return f"{name} matched the {labels} in your photo."
