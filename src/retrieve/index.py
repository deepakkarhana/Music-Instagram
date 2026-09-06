"""The search index - finding the closest songs, fast.

THE PROBLEM
-----------
We have ~10,000 songs, each a vector of 512 numbers. A photo becomes another
such vector. "Which songs match?" means "which of the 10,000 vectors point
most nearly the same direction as this one?"

You could just compare against all 10,000 one by one. At our size that is
genuinely fine - a few milliseconds. But it grows badly, and real catalogs
have millions of songs, so we use the tool the industry uses: **FAISS**, from
Meta. It is a library for exactly this question.

WHY IndexFlatIP
---------------
"Flat" means it really does check every song - no approximation, no accuracy
lost. It is the right choice below roughly a million vectors, and it removes a
whole class of "is my index tuned correctly?" bugs while we are still figuring
out whether the *idea* works. When the catalog grows, swapping in an
approximate index (IVF, HNSW) is a few lines here and nothing elsewhere.

"IP" means inner product. Because every vector we store has been normalised to
length 1 (see `src/encode/clap.py`), the inner product *is* the cosine
similarity. Scores come back between -1 and 1, where 1 means identical
direction.

WHAT GETS SAVED
---------------
Two files, and neither contains audio:

    catalog.faiss       the index itself
    catalog_ids.json    which track_id each row belongs to

FAISS only knows about row numbers. The ids file is how we turn row 4,182 back
into a song name.
"""

import json
import os

import numpy as np


def build(vectors):
    """Create an exact cosine-similarity index over `vectors`.

    `vectors` must be a (n, dim) float32 array of unit-length rows.
    """
    import faiss

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError(f"expected a 2-D array, got shape {vectors.shape}")

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    return index


def save(index, track_ids, base_path):
    """Write the index and its id list. `base_path` has no extension."""
    import faiss

    folder = os.path.dirname(base_path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    faiss.write_index(index, base_path + ".faiss")
    with open(base_path + "_ids.json", "w", encoding="utf-8") as handle:
        json.dump(list(track_ids), handle)


def load(base_path):
    """Read back an index and its id list.

    Returns
    -------
    (index, list[str])
    """
    import faiss

    index = faiss.read_index(base_path + ".faiss")
    with open(base_path + "_ids.json", "r", encoding="utf-8") as handle:
        track_ids = json.load(handle)
    return index, track_ids


def search(index, query_vectors, k=10):
    """Find the `k` nearest songs for each query vector.

    Returns
    -------
    (scores, rows)
        Both are (n_queries, k) arrays. `scores` are cosine similarities,
        highest first. `rows` are positions in the index - use the id list
        from `load` to turn them into track_ids.
    """
    queries = np.ascontiguousarray(query_vectors, dtype=np.float32)
    if queries.ndim == 1:
        queries = queries.reshape(1, -1)

    # Asking for more neighbours than exist makes FAISS return -1 padding.
    k = min(k, index.ntotal)
    return index.search(queries, k)


def search_by_text(index, track_ids, catalog_by_id, query_vector, k=10, seen_artists=None):
    """Convenience wrapper: search, then return readable rows.

    Returns a list of dicts with the score and the track's catalog metadata,
    ready to print or serve from an API.
    """
    scores, rows = search(index, query_vector, k=k)

    results = []
    for score, row in zip(scores[0], rows[0]):
        if row < 0:  # FAISS padding when k exceeded the catalog size
            continue
        track_id = track_ids[row]
        track = dict(catalog_by_id.get(track_id, {}))
        track["score"] = float(score)
        results.append(track)

    return results
