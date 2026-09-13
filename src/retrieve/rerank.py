"""Make the five songs different from each other, not just different artists.

THE PROBLEM THE ARTIST CAP DOES NOT SOLVE
-----------------------------------------
Capping one song per artist already gives a diversity score of 1.00. That
number is comfortable and slightly misleading: five different artists can all
be lo-fi beats. The list looks varied in the metadata and sounds identical.

That matters for a recommender specifically. Someone choosing music for a
story does not want five attempts at the same answer - they want a few real
options. If the top five are near-duplicates, four of them are wasted slots.

MAXIMAL MARGINAL RELEVANCE
--------------------------
The standard fix, and it is simple. Instead of taking the top five by score,
build the list one at a time, each time choosing the song that maximises:

    lambda * (how well it matches the photo)
      - (1 - lambda) * (how similar it is to what we have already picked)

lambda = 1 is the old behaviour. lambda = 0 ignores the photo entirely and
just picks five songs unlike each other. The useful range is 0.6 to 0.8.

WE ALREADY HAVE WHAT THIS NEEDS
-------------------------------
Song-to-song similarity is the dot product of their CLAP embeddings, which the
FAISS index can hand back with `reconstruct`. No extra model, no extra data -
the same vectors that answered "does this song match the photo?" also answer
"do these two songs sound alike?".

THE TRADE-OFF IS REAL AND SHOULD BE MEASURED
--------------------------------------------
Diversity is bought with relevance. Every song chosen for being different is a
song not chosen for being the best match, so agreement with the oracle will
fall. The question is whether the drop is small enough to be worth a list that
is actually useful, and that is a measurement rather than an opinion - see
`scripts/11_evaluate.py`.
"""

import numpy as np

DEFAULT_LAMBDA = 0.7


def song_vectors(index, rows):
    """Get the stored embedding for each index position.

    FAISS keeps the original vectors for a flat index, so this costs nothing
    and needs no separate copy of the embeddings.
    """
    return np.vstack([index.reconstruct(int(r)) for r in rows])


def mmr(candidate_rows, candidate_scores, index, k=5, lam=DEFAULT_LAMBDA):
    """Pick k rows balancing match quality against sounding different.

    Parameters
    ----------
    candidate_rows : sequence[int]
        Index positions, best-matching first.
    candidate_scores : sequence[float]
        Their similarity to the photo's query vector.
    lam : float
        1.0 keeps the original ranking. Lower values push for variety.

    Returns
    -------
    list[int]
        Positions into `candidate_rows`, in the order they should be shown.
    """
    rows = list(candidate_rows)
    if not rows or lam >= 1.0:
        return list(range(min(k, len(rows))))

    vectors = song_vectors(index, rows)
    # Vectors are unit length, so this is cosine similarity between songs.
    similarity = vectors @ vectors.T
    scores = np.asarray(candidate_scores, dtype=np.float32)

    chosen = [0]  # always start with the best match
    while len(chosen) < min(k, len(rows)):
        best_position, best_value = None, -np.inf
        for i in range(len(rows)):
            if i in chosen:
                continue
            # How close is this to the most similar thing already chosen?
            redundancy = float(np.max(similarity[i, chosen]))
            value = lam * scores[i] - (1.0 - lam) * redundancy
            if value > best_value:
                best_position, best_value = i, value
        if best_position is None:
            break
        chosen.append(best_position)

    return chosen


def mean_pairwise_similarity(rows, index):
    """How alike a set of songs sounds. Lower is more varied.

    The honest companion to the artist-diversity number: five artists with a
    mean similarity of 0.95 are five versions of one song.
    """
    if len(rows) < 2:
        return 0.0
    vectors = song_vectors(index, rows)
    similarity = vectors @ vectors.T
    upper = similarity[np.triu_indices(len(rows), k=1)]
    return float(np.mean(upper))
