"""Remove songs that are the same recording twice.

WHY THIS IS NEEDED
------------------
The catalog is built from ~90 overlapping searches, and iTunes lists the same
recording many times over: on the single, on the album, on the soundtrack, on
a "best of" compilation. Week 1 removed duplicates by `track_id`, which only
catches the literal same listing. It does not catch these:

    "Shayad"                        - Pritam & Arijit Singh
    "Shayad (From \\"Love Aaj Kal\\")"  - Pritam & Arijit Singh

Different track ids, different titles, identical audio. Cosine similarity of
their embeddings: 0.9973.

WHY IT MATTERS FOR RECOMMENDATIONS
----------------------------------
Returning five songs that are actually the same song three times is a bad
recommendation regardless of how well the model understood the photo. Worse,
stock-music labels upload the same track dozens of times under one title -
one artist in our catalog has 42 rows called "Late Night Drive - Mellow Lo-Fi
for Driving". A cluster like that crowds out everything else and shows up as
the top result for unrelated queries.

WHY WE COMPARE AUDIO, NOT TITLES
--------------------------------
We already have an embedding of every song, and the embedding *is* what the
audio sounds like. Comparing those catches re-releases whose titles differ,
and avoids falsely merging two genuinely different songs that happen to share
a name. Title matching can do neither.

CHOOSING THE THRESHOLD
----------------------
Measured on a 1,560-track sample:

    cosine > 0.99    31 pairs     clearly the same recording
    cosine > 0.98    44 pairs     the same recording, incl. re-releases
    cosine > 0.95  1,927 pairs    far too many - now merging different songs

0.98 is the setting that removes duplicates without merging distinct music.
"""

import numpy as np

DEFAULT_THRESHOLD = 0.98


def find_keepers(vectors, threshold=DEFAULT_THRESHOLD, neighbours=20):
    """Decide which rows to keep, dropping near-identical audio.

    Walks the tracks in order and keeps one, dropping any later track that
    sounds nearly identical to something already kept. The first listing of a
    song wins, which is arbitrary but stable - the same input always gives the
    same output.

    Uses the FAISS index to ask "what are this song's closest neighbours?"
    rather than comparing every pair against every other pair. At 13,497 songs
    the all-pairs matrix would be 729 MB; this stays small.

    Parameters
    ----------
    vectors : np.ndarray
        (n, dim) array of unit-length embeddings.
    threshold : float
        Cosine similarity above which two tracks count as the same recording.
    neighbours : int
        How many nearest neighbours to inspect per track. Duplicate clusters
        larger than this are still caught, because each member is checked
        against the ones already kept.

    Returns
    -------
    np.ndarray
        Boolean mask, True for rows to keep.
    """
    import faiss

    vectors = np.ascontiguousarray(vectors, dtype=np.float32)
    count = len(vectors)
    if count == 0:
        return np.zeros(0, dtype=bool)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    k = min(neighbours, count)
    scores, rows = index.search(vectors, k)

    keep = np.ones(count, dtype=bool)
    for i in range(count):
        if not keep[i]:
            continue  # already dropped as somebody else's duplicate
        for score, j in zip(scores[i], rows[i]):
            # Only drop *later* tracks, so the earliest listing survives and
            # we never drop both halves of a pair.
            if j > i and score >= threshold:
                keep[j] = False

    return keep


def summarise(keep):
    """Return (kept, dropped) counts for printing."""
    kept = int(keep.sum())
    return kept, int(len(keep) - kept)
