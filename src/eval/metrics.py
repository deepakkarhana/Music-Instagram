"""How to score a list of recommended songs.

THE PROBLEM WITH SCORING THIS AT ALL
------------------------------------
Normal retrieval evaluation needs a ground truth: for this query, these
documents are correct. We have nothing of the kind. Nobody has written down
which of 11,870 songs belong with a photo of a beach, and there is no single
right answer if they tried - a dozen songs fit, and taste decides among them.

So we measure three different things, each imperfect in a *different* way. A
method that wins on all three is probably good. A method that wins on one is
probably exploiting that metric.

1. ORACLE AGREEMENT
-------------------
The oracle is the music half given the *correct* vibe. It defines a set of
songs that genuinely belong with this photo's vibe, without the vision step
being involved at all.

    recall@k = how much of the oracle's top 20 a method recovers in its top k

This measures the pipeline end to end while isolating the vision step's cost.
Its weakness is obvious and worth stating: it treats CLAP's judgement as
truth. If CLAP is wrong about what a wedding sounds like, the oracle is wrong
too and every method is scored against a wrong answer.

2. BALANCED LANE ACCURACY - the independent one
-----------------------------------------------
Every track carries a `lane` field, `indian` or `english`, recorded in Week 1
from which search found it. **Neither CLIP nor CLAP has ever seen this
field.** It comes from the catalogue, not from any model.

So for photos whose vibe is unambiguously Indian - a wedding, a temple, a
festival - we can ask what fraction of returned songs are actually Indian.

THE FIRST VERSION OF THIS METRIC WAS BROKEN, and the way it broke is worth
keeping. Measured only on Indian vibes, the scores came out:

    random    52%
    popular  100%      <- ignores the photo entirely
    keyword   96%
    ours      81%

The popularity baseline won by ignoring its input. The most prolific artists
in this catalogue happen to be Indian, so "always return popular songs" always
returns Indian songs, and scored perfectly on a metric asking "are these
Indian?".

A one-sided metric rewards bias, not accuracy. The fix is to measure both
directions and average them:

    Indian vibes  -> what fraction of results are indian?
    Western vibes -> what fraction are english?
    score         -> the mean of the two

Now a method that always answers "Indian" scores ~100% on one half and ~0% on
the other, averaging to chance. Only a method that actually reads the photo
can score well on both.

3. DIVERSITY
------------
Five songs by one artist is a bad recommendation even when all five are
correct. Cheap to measure, easy to forget, and it catches the failure where a
method "wins" by returning five copies of the safest possible answer.
"""

import numpy as np

# Vibes whose music should be Indian by any reasonable reading. Kept small on
# purpose: a "romantic" photo could honestly be either, and including it would
# measure nothing but noise.
INDIAN_VIBES = {
    "wedding_indian", "festival_indian", "traditional_indian", "temple",
    "vintage_bollywood",
}

# Vibes whose music is Western-coded. These are the other half of the balanced
# check - without them, a method that always returns Indian music scores
# perfectly. Also kept small and deliberately uncontroversial: old money means
# lounge jazz, Y2K means synth pop, cottagecore means acoustic folk.
WESTERN_VIBES = {"old_money", "y2k", "cottagecore"}


def recall_at_k(method_ids, oracle_ids, k=5):
    """Fraction of a method's top k that the oracle also chose."""
    if not method_ids:
        return 0.0
    top = list(method_ids)[:k]
    hits = sum(1 for t in top if t in set(oracle_ids))
    return hits / len(top)


def ndcg_at_k(method_ids, oracle_ids, k=5):
    """Like recall, but rewards putting the oracle's favourites first.

    A song the oracle ranked 1st is worth more than one it ranked 20th, and a
    hit at position 1 counts more than a hit at position 5. Standard nDCG,
    with the oracle's ranking supplying the relevance grades.
    """
    if not method_ids or not oracle_ids:
        return 0.0

    relevance = {t: len(oracle_ids) - i for i, t in enumerate(oracle_ids)}
    gains = [relevance.get(t, 0) for t in list(method_ids)[:k]]
    discounts = 1.0 / np.log2(np.arange(2, len(gains) + 2))
    actual = float(np.sum(np.array(gains) * discounts))

    best = sorted(relevance.values(), reverse=True)[:k]
    if not best:
        return 0.0
    ideal = float(np.sum(np.array(best) * (1.0 / np.log2(np.arange(2, len(best) + 2)))))
    return actual / ideal if ideal else 0.0


def expected_lane(vibe_name):
    """Which lane this vibe's music should come from, or None if it is neutral.

    Most vibes are neutral - a beach is a beach in any language - and scoring
    them would measure nothing.
    """
    if vibe_name in INDIAN_VIBES:
        return "indian"
    if vibe_name in WESTERN_VIBES:
        return "english"
    return None


def lane_precision(method_ids, catalog_by_id, want="indian"):
    """Fraction of returned songs in the wanted lane.

    The independent check: this field came from the Week 1 catalogue and no
    model in the pipeline has ever seen it. Always pair it with the opposite
    direction - see the module docstring for why a one-sided version is
    trivially gamed.
    """
    if not method_ids:
        return 0.0
    hits = sum(
        1 for t in method_ids
        if catalog_by_id.get(t, {}).get("lane") == want
    )
    return hits / len(method_ids)


def artist_diversity(method_ids, catalog_by_id):
    """Distinct artists divided by songs returned. 1.0 means all different."""
    if not method_ids:
        return 0.0
    artists = {
        (catalog_by_id.get(t, {}).get("artist_name") or "?").lower()
        for t in method_ids
    }
    return len(artists) / len(method_ids)


def summarise(rows):
    """Average every metric across photos. `rows` is a list of dicts.

    Not every row carries every metric - `lane` only applies to photos whose
    vibe is unambiguously Indian - so keys are collected across *all* rows and
    each is averaged only over the rows that have it.

    An earlier version read the key list from `rows[0]` alone. The first photo
    alphabetically was a beach, so `lane` was absent from row zero and silently
    dropped from the report for every method. It printed "-" rather than
    failing, which is exactly the sort of bug that survives review: the output
    looked orderly and a whole metric had quietly vanished.
    """
    if not rows:
        return {}

    keys = []
    for row in rows:
        for key, value in row.items():
            if isinstance(value, (int, float)) and key not in keys:
                keys.append(key)

    summary = {}
    for key in keys:
        values = [r[key] for r in rows if key in r]
        if values:
            summary[key] = float(np.mean(values))
    return summary
