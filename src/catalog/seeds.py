"""Search terms we use to build the song catalog.

WHY THIS FILE EXISTS
--------------------
The iTunes Search API does not let you say "give me every song". You have to
*search* for something. So to build a catalog of ~50,000 tracks, we run many
different searches and merge the results.

Each seed is a (term, country, lane) triple:

    term    - what we type into the search box
    country - which iTunes storefront to search ("IN" = India, "US" = USA).
              The Indian storefront has far better Bollywood/Punjabi coverage.
    lane    - our own label: "indian" or "english". We keep this because it is
              a useful first guess at the language of the track, and later the
              retrieval system uses language as a hard filter.

The seeds are deliberately a mix of GENRES, MOODS, ERAS and ARTISTS. Mixing
those axes gives much better coverage than any one of them alone.

Feel free to add your own terms - more seeds means a bigger, richer catalog.
"""

# --------------------------------------------------------------------------
# Indian storefront: Hindi, Punjabi, and regional
# --------------------------------------------------------------------------
INDIAN_SEEDS = [
    # -- broad genre nets --
    "bollywood hits",
    "bollywood romantic",
    "bollywood dance",
    "bollywood sad songs",
    "punjabi songs",
    "punjabi bhangra",
    "punjabi romantic",
    "hindi indie",
    "desi hip hop",
    "sufi music",
    "ghazal",
    "indian classical fusion",
    "tamil hits",
    "telugu hits",
    "marathi songs",
    "bengali songs",
    # -- eras (these pull very different sonic worlds) --
    "bollywood 90s",
    "bollywood 2000s",
    "old hindi songs",
    "retro bollywood",
    # -- moods and occasions (closest to how people pick story music) --
    "hindi party songs",
    "hindi workout songs",
    "hindi acoustic",
    "hindi lofi",
    "bollywood monsoon songs",
    "hindi wedding songs",
    "hindi motivational songs",
    "hindi breakup songs",
    "hindi road trip songs",
    "hindi soft songs",
    # -- artists (dense, high-quality clusters) --
    "arijit singh",
    "shreya ghoshal",
    "atif aslam",
    "diljit dosanjh",
    "ap dhillon",
    "sidhu moose wala",
    "pritam",
    "a r rahman",
    "anuv jain",
    "prateek kuhad",
    "the local train",
    "king rapper",
    "badshah",
    "neha kakkar",
    "jubin nautiyal",
    "kishore kumar",
    "lata mangeshkar",
    "shankar mahadevan",
]

# --------------------------------------------------------------------------
# US storefront: English / international
# --------------------------------------------------------------------------
ENGLISH_SEEDS = [
    # -- genre --
    "indie folk",
    "lo-fi hip hop",
    "pop hits",
    "sad indie",
    "edm festival",
    "jazz standards",
    "classical piano",
    "rock anthems",
    "r&b smooth",
    "ambient music",
    "synthwave",
    "country roads",
    "hip hop classics",
    "acoustic guitar",
    "cinematic orchestral",
    "bossa nova",
    "house music",
    "punk rock",
    "soul music",
    "dream pop",
    "shoegaze",
    "afrobeats",
    "k-pop hits",
    "reggae",
    # -- mood and occasion --
    "gym motivation",
    "study focus music",
    "road trip songs",
    "beach vibes",
    "rainy day music",
    "golden hour music",
    "late night drive",
    "coffee shop music",
    "summer party",
    "winter cozy music",
    "epic trailer music",
    "romantic slow songs",
    # -- era --
    "80s hits",
    "90s hits",
    "2000s throwback",
    "60s jazz",
]


def all_seeds():
    """Return every seed as a (term, country, lane) tuple.

    Returns
    -------
    list[tuple[str, str, str]]
        e.g. [("bollywood hits", "IN", "indian"), ("indie folk", "US", "english"), ...]
    """
    seeds = [(term, "IN", "indian") for term in INDIAN_SEEDS]
    seeds += [(term, "US", "english") for term in ENGLISH_SEEDS]
    return seeds


if __name__ == "__main__":
    # Run `python src/catalog/seeds.py` to see what we're about to search for.
    seeds = all_seeds()
    n_in = sum(1 for _, _, lane in seeds if lane == "indian")
    n_en = len(seeds) - n_in
    print(f"{len(seeds)} seed searches ({n_in} indian, {n_en} english)")
    print(f"At 200 results each, that's up to {len(seeds) * 200:,} raw rows")
    print("(expect far fewer after removing duplicates and tracks with no preview)")
