"""Things to beat.

WHY BASELINES ARE THE POINT
---------------------------
"Our system returns reasonable songs" is not a result. Reasonable compared to
what? A recommender that always returns the five most popular songs looks
surprisingly good to a casual eye, and needs no machine learning at all.

A result is: *this method beats these simpler methods by this much, measured
this way.* Without the comparison you have a demo. With it you have evidence,
and the comparison is the first thing anyone competent will ask for.

THE FOUR WE COMPARE AGAINST
---------------------------
**Random.** The floor. If a method cannot beat drawing songs out of a hat,
nothing else about it matters.

**Popularity.** Return songs by the most prolific artists in the catalogue,
ignoring the photo completely. This is the baseline that embarrasses people:
it ignores the input entirely and still scores respectably, because popular
music is broadly agreeable. Any honest evaluation has to include it.

**Keyword matching.** Take the words of the predicted vibe and look for them
in track titles, genres and the search terms that found them. No embeddings,
no models - just string matching, the thing you would build in an afternoon.
**This is the baseline that actually matters.** If a pipeline of two neural
networks cannot beat `if "wedding" in genre`, it has not earned its
complexity.

**Oracle.** Not a baseline to beat - a ceiling. It uses the *correct* vibe
rather than the predicted one, so it shows what the music half would return if
the vision half were perfect. The gap between our system and the oracle is
exactly how much the vision step is costing us, which tells you where to spend
the next week of work.
"""

import random

import numpy as np


def random_songs(track_ids, k=5, seed=0):
    """Draw k songs at random. The floor."""
    rng = random.Random(seed)
    return rng.sample(list(track_ids), min(k, len(track_ids)))


def popular_songs(track_ids, catalog_by_id, k=5):
    """Songs by the artists with most tracks in the catalogue.

    We have no play counts - iTunes does not give them away - so "how many
    tracks did this artist get into our catalogue" stands in for popularity.
    It is a proxy and should be described as one: it rewards prolific
    stock-music labels as much as genuine stars.

    Returns the same list for every photo, which is the entire point.
    """
    from collections import Counter

    counts = Counter(
        (catalog_by_id.get(t, {}).get("artist_name") or "?") for t in track_ids
    )
    ranked = [a for a, _ in counts.most_common()]

    picked, seen = [], set()
    for artist in ranked:
        for track_id in track_ids:
            if catalog_by_id.get(track_id, {}).get("artist_name") == artist:
                if artist in seen:
                    continue
                picked.append(track_id)
                seen.add(artist)
                break
        if len(picked) >= k:
            break
    return picked[:k]


def keyword_songs(vibe, track_ids, catalog_by_id, k=5):
    """Match the vibe's words against track text. No models involved.

    Scores each track by how many distinct vibe words appear in its title,
    genre or the seed term that found it. Ties are broken by track order,
    which is arbitrary but stable.

    This is the honest "do you even need machine learning?" comparison.
    """
    stop = {
        "and", "or", "with", "the", "a", "an", "of", "in", "on", "at", "no",
        "very", "soft", "slow", "fast", "light", "deep", "big", "long", "one",
    }
    words = {
        w.strip(",").lower()
        for w in (vibe.label + " " + vibe.music_query + " " + vibe.visual_cues).split()
        if len(w) > 3 and w.strip(",").lower() not in stop
    }

    scored = []
    for track_id in track_ids:
        track = catalog_by_id.get(track_id, {})
        haystack = " ".join([
            track.get("track_name", ""), track.get("genre", ""),
            track.get("seed_term", ""), track.get("album_name", ""),
        ]).lower()
        hits = sum(1 for w in words if w in haystack)
        if hits:
            scored.append((hits, track_id))

    scored.sort(key=lambda pair: -pair[0])
    return [track_id for _, track_id in scored[:k]]


def oracle_songs(gold_vibe, vibe_text_vectors, vibes, index, track_ids, k=5):
    """What the music half returns when told the right vibe. The ceiling."""
    from src.retrieve import index as index_module

    position = {v.name: i for i, v in enumerate(vibes)}[gold_vibe.name]
    vector = vibe_text_vectors[position].reshape(1, -1)
    _, rows = index_module.search(index, vector, k=k)
    return [track_ids[r] for r in rows[0] if r >= 0]
