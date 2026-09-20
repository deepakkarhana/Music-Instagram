"""Hard filters, for the things an embedding should not be asked to do.

WHY THIS EXISTS - two weeks arrived at it independently
-------------------------------------------------------
**Week 2**, testing CLAP on Indian music, found it separates South Asian from
Western reliably but cannot tell Punjabi from Hindi from Tamil. A query for
"punjabi bhangra" returned A.R. Rahman (Tamil), Dilbaro (Hindi) and an Urdu
ghazal alongside one actual Punjabi track.

**Week 6**, comparing against baselines, found that plain string matching on
the `genre` and `seed_term` fields beat our embeddings at language: 75%
balanced lane accuracy against our 71%, and 95% when given the true vibe.
Those fields literally contain the words *bollywood* and *punjabi*.

Two different methods, one conclusion: **language is metadata, not acoustics.**
Whether a song is in Hindi is a fact recorded in the catalogue, not something
to be inferred from how it sounds. Asking the embedding to rediscover it is
strictly worse than looking it up.

WHAT THIS DOES AND DOES NOT FILTER
----------------------------------
Only vibes that are unambiguously coded one way. An Indian wedding should
return Indian music; "old money" means lounge jazz. But a beach is a beach in
any language, and a romantic photo could honestly be either - filtering those
would throw away good recommendations to satisfy a rule nobody asked for.

Most photos are therefore unfiltered. That is the intended behaviour, not a
weak implementation.

THE FALLBACK MATTERS
--------------------
If filtering leaves too few songs, the filter is dropped for that request.
Five slightly-wrong-language songs beat two right-language ones, and a demo
that sometimes returns one result looks broken rather than principled.
"""

# Vibes that clearly call for Indian music.
INDIAN_VIBES = {
    "wedding_indian", "festival_indian", "traditional_indian", "temple",
    "vintage_bollywood",
}

# Vibes that clearly call for Western music.
WESTERN_VIBES = {"old_money", "y2k", "cottagecore", "streetwear"}

# Below this many surviving songs, drop the filter rather than return a stub.
MIN_RESULTS = 5


def wanted_lane(picked):
    """Which lane the chosen vibes call for, or None to leave it alone.

    `picked` is the list of (vibe, score) that CLIP produced, strongest first.

    ONLY THE TOP VIBE DECIDES, and the first version of this was wrong about
    that in a way worth recording.

    It also let a coded vibe further down trigger the filter, provided its
    score was at least half the top score. That sounds conservative and is
    not, because **CLIP's cosine scores sit in a narrow band**. On a real
    photo of a concert crowd:

        0.2927  Euphoric            neutral
        0.2675  Party or club       neutral
        0.2128  Night               neutral
        0.2113  Snow                neutral
        0.2021  Vintage Bollywood   indian     <- 69% of the top score

    Fifth place, clearly not what the photo is about, and 0.2021/0.2927 = 0.69
    sailed past the 0.5 threshold. The whole result set was restricted to
    Indian music because of it.

    In a band running 0.20 to 0.29, *everything* is 70-100% of the top. Ratios
    of raw cosine scores carry almost no information - a mistake already noted
    in `vibes_to_query_vector`, where weights are built from differences for
    exactly this reason, and then made again here.

    The top vibe alone is predictable and explicable: if the strongest thing
    CLIP sees is an Indian wedding, restrict to Indian music. Otherwise leave
    the results alone.
    """
    if not picked:
        return None

    top_vibe = picked[0][0]
    if top_vibe.name in INDIAN_VIBES:
        return "indian"
    if top_vibe.name in WESTERN_VIBES:
        return "english"
    return None


def apply_lane(track_ids, catalog_by_id, lane, minimum=MIN_RESULTS):
    """Keep only songs in `lane`, preserving order.

    Returns (kept, was_applied). If filtering would leave fewer than
    `minimum` songs, the original list comes back untouched - see the note in
    the module docstring about why.
    """
    if not lane:
        return list(track_ids), False

    kept = [
        t for t in track_ids
        if catalog_by_id.get(t, {}).get("lane") == lane
    ]
    if len(kept) < minimum:
        return list(track_ids), False
    return kept, True


def describe(lane, applied):
    """One line for the demo, so the filter is visible rather than mysterious."""
    if not lane:
        return ""
    if not applied:
        return f"wanted {lane} music, but too few matched - filter skipped"
    return f"restricted to {lane} music (from the catalogue, not the audio)"
