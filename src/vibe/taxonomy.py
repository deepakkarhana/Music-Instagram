"""The vibe vocabulary - the words this project is allowed to think in.

WHAT A "VIBE" IS HERE
---------------------
A vibe is one recognisable kind of story photo, paired with the music that
belongs to it. Each entry holds two very different descriptions:

    visual_cues   what the PHOTO looks like    (for the vision model, Week 4)
    music_query   what the MUSIC sounds like   (for CLAP, working today)

Keeping them separate is the whole design, and Week 2 is why.

THE LESSON THAT SHAPED THIS FILE
--------------------------------
Week 2 measured something that looks obvious in hindsight and was not obvious
at all beforehand:

    "high energy gym workout, aggressive hip hop with heavy bass"
        -> Rainy Day, Bossa Nova, Beach Vibes

CLAP only ever hears audio. **A gym is a place, not a timbre.** The words
"gym" and "workout" describe a room and an activity; they carry no sound.
Rewriting the same intent as sound - *"aggressive rap, hard hitting drums,
angry male rapper shouting"* - retrieved gym music immediately.

So every `music_query` below is written in the language of **sound**:
instruments, tempo, texture, vocal style, production. Never places, never
activities, never aesthetics. If a query mentions a room or a feeling without
naming a sound, it is wrong, and `scripts/06_probe_vibes.py` will expose it.

The `visual_cues` field is the opposite - pure appearance, no music words. It
exists so Week 4's vision model has something concrete to match a photo
against. The vibe name is the join between the two.

WHY A FIXED VOCABULARY AT ALL
-----------------------------
We could let a language model invent a fresh description per photo. We will,
eventually. But a fixed list is what makes the system *measurable*: you cannot
compute accuracy against free text. These names become the labels for the
Week 7 classifier and the categories in the Week 6 evaluation.

AXES, AND WHY PHOTOS GET SEVERAL VIBES
--------------------------------------
A single photo is rarely one thing. A friend on a Goa beach at sunset is a
place, a time of day, and a mood at once. So vibes are grouped on four axes
and a photo may carry one from each:

    scene      where it is            beach, mountains, cafe, city at night
    time       when, and what light   golden hour, monsoon, night
    mood       what it feels like     nostalgic, euphoric, tender
    occasion   what is happening      wedding, workout, party, studying

This is a hypothesis, not a finished answer. It was written before any photos
were collected, and Week 3's job is to test it against real ones.
"""

from collections import namedtuple

Vibe = namedtuple("Vibe", "name label axis visual_cues music_query")

VIBES = [
    # ---------------------------------------------------------------- scene
    Vibe("mountains", "Mountains",
         "scene",
         "wide landscape, peaks and ridgelines, haze, a small figure for scale",
         "airy acoustic guitar, slow tempo, wide reverb, sustained strings, no drums"),
    Vibe("beach", "Beach",
         "scene",
         "sand, sea horizon, bright sky, swimwear, waves",
         "bright ukulele and acoustic guitar, relaxed mid tempo, light shaker percussion"),
    Vibe("cafe", "Cafe",
         "scene",
         "table with cups, window light, indoor plants, laptop or book",
         "soft jazz piano, brushed drums, upright bass, quiet and unhurried"),
    Vibe("city_night", "City at night",
         "scene",
         "streetlights, neon signs, wet tarmac, tall buildings after dark",
         "moody synthesizer pads, slow electronic beat, deep sub bass, sparse"),
    Vibe("road_trip", "Road trip",
         "scene",
         "car window, open highway, dashboard, landscape rushing past",
         "steady driving drums, electric guitar riff, mid tempo rock, warm bass"),
    Vibe("home", "At home",
         "scene",
         "sofa, bed, warm lamps, blankets, indoor domestic clutter",
         "gentle fingerpicked guitar, soft male vocal, slow, very quiet"),
    Vibe("nature_green", "Green nature",
         "scene",
         "forest, fields, trees, heavy greenery, daylight",
         "flute and soft strings, slow, open and spacious, natural reverb"),
    Vibe("temple", "Temple or shrine",
         "scene",
         "carved stone, idols, marigold garlands, incense smoke, bells",
         "indian devotional bhajan, harmonium, group chanting, temple bells"),
    Vibe("street_market", "Street or market",
         "scene",
         "crowded lane, stalls, signage, motorbikes, dense colour",
         "busy percussion, fast tabla, layered brass, energetic and chaotic"),
    Vibe("desert", "Desert or dunes",
         "scene",
         "sand dunes, arid ground, sparse scrub, vast empty sky",
         "sarangi and slow rhythmic percussion, long drone, sparse arrangement"),
    Vibe("snow", "Snow",
         "scene",
         "white ground, bare trees, breath fog, heavy coats",
         "soft piano, high sustained strings, very slow, glassy bell tones"),
    Vibe("gym", "Gym",
         "scene",
         "weights, mirrors, machines, sportswear, harsh indoor light",
         "hard hitting drums, distorted bass, shouted male rap vocals, fast and aggressive"),

    # ----------------------------------------------------------------- time
    Vibe("golden_hour", "Golden hour",
         "time",
         "low warm sun, long shadows, orange and amber cast, lens flare",
         "warm acoustic guitar, slow tempo, soft brass, mellow and unhurried"),
    Vibe("night", "Night",
         "time",
         "darkness, artificial light sources, high contrast, deep shadow",
         "slow synth bass, sparse electronic percussion, dark and spacious"),
    Vibe("monsoon", "Rain and monsoon",
         "time",
         "wet surfaces, raindrops, grey overcast light, umbrellas",
         "quiet piano, soft rain-like texture, slow melancholy strings"),
    Vibe("bright_daylight", "Bright daylight",
         "time",
         "strong sun, saturated colour, hard shadows, blue sky",
         "upbeat acoustic strumming, handclaps, cheerful mid tempo, bright"),

    # ----------------------------------------------------------------- mood
    Vibe("nostalgic", "Nostalgic",
         "mood",
         "faded or grainy tones, old photographs, film look, soft focus",
         "old recording with tape hiss, gentle strings, slow waltz, vintage vocal"),
    Vibe("romantic", "Romantic",
         "mood",
         "two people close together, soft warm light, gentle expressions",
         "tender male and female duet, soft strings, slow ballad tempo"),
    Vibe("melancholy", "Melancholy",
         "mood",
         "muted desaturated colour, solitary figure, downcast posture",
         "slow solo piano, minor key, sparse, quiet and sorrowful"),
    Vibe("euphoric", "Euphoric",
         "mood",
         "arms raised, crowds, motion blur, bright saturated light",
         "soaring synth chords, four on the floor kick, fast tempo, big build"),
    Vibe("confident", "Confident swagger",
         "mood",
         "direct gaze at camera, strong posture, styled outfit, bold colour",
         "heavy trap beat, deep 808 bass, confident rap vocal, mid tempo"),
    Vibe("calm", "Calm",
         "mood",
         "simple composition, soft even light, uncluttered frame, stillness",
         "sustained ambient pads, no percussion, very slow, gentle drone"),
    Vibe("playful", "Playful",
         "mood",
         "laughing faces, silly poses, motion, bright mismatched colour",
         "bouncy plucked strings, quick tempo, whistling, light comedic percussion"),
    Vibe("dramatic", "Dramatic",
         "mood",
         "strong contrast, dark sky, imposing scale, tense composition",
         "full orchestral strings, timpani, slow heavy build, cinematic brass"),

    # ------------------------------------------------------------- occasion
    Vibe("wedding_indian", "Indian wedding",
         "occasion",
         "lehenga and sherwani, marigold decor, mandap, gold jewellery, crowds",
         "dhol drums, shehnai, festive dance rhythm, celebratory group vocals"),
    Vibe("festival_indian", "Indian festival",
         "occasion",
         "colour powder, diyas, lights, fireworks, dense joyful crowds",
         "loud dhol and dholak, fast bhangra rhythm, brass, energetic chorus"),
    Vibe("party", "Party or club",
         "occasion",
         "dim room, coloured lights, drinks, dancing, crowd close together",
         "loud four on the floor beat, sidechained synths, fast dance tempo"),
    Vibe("workout", "Working out",
         "occasion",
         "mid movement, sweat, sportswear, gym or running route",
         "fast aggressive drums, distorted guitar, shouted vocals, driving tempo"),
    Vibe("study", "Studying or working",
         "occasion",
         "books, notes, laptop, desk lamp, focused solitary figure",
         "lo-fi beat, muted piano loop, soft vinyl crackle, slow and repetitive"),
    Vibe("travel", "Travelling",
         "occasion",
         "luggage, airports, trains, maps, unfamiliar streets",
         "rhythmic acoustic guitar, light percussion, optimistic mid tempo"),
    Vibe("graduation", "Graduation or achievement",
         "occasion",
         "gowns and caps, certificates, medals, group celebration",
         "triumphant brass, uplifting strings, steady march rhythm, major key"),
    Vibe("food", "Food",
         "occasion",
         "plated dishes, close crop, rich texture and colour, table setting",
         "light bossa nova guitar, brushed percussion, warm and casual"),
    Vibe("birthday", "Birthday",
         "occasion",
         "cake, candles, balloons, party hats, close friends",
         "cheerful pop with handclaps, bright synths, upbeat and simple"),

    # ---------------------------------------------------- aesthetic (scene)
    Vibe("old_money", "Old money",
         "aesthetic",
         "tailored neutral clothing, heritage interiors, understated luxury",
         "smooth jazz saxophone, upright bass, brushed drums, lounge tempo"),
    Vibe("streetwear", "Streetwear",
         "aesthetic",
         "oversized clothing, sneakers, urban walls, graffiti, concrete",
         "boom bap drums, scratching, sampled soul loop, laid back rap vocal"),
    Vibe("traditional_indian", "Traditional Indian",
         "aesthetic",
         "saree or kurta, ethnic jewellery, henna, traditional textiles",
         "sitar and tabla, classical Indian vocal, slow raga, harmonium drone"),
    Vibe("minimal", "Minimal",
         "aesthetic",
         "plain backgrounds, restrained palette, clean lines, negative space",
         "one repeated marimba figure, dry recording, no reverb, steady pulse"),
    Vibe("vintage_bollywood", "Vintage Bollywood",
         "aesthetic",
         "retro styling, film-grain colour, seventies and eighties fashion",
         "old Hindi film orchestra, female playback vocal, tabla, vintage strings"),
    Vibe("y2k", "Y2K",
         "aesthetic",
         "low-rise fashion, metallic and glossy textures, flash photography",
         "bright synth pop, dance beat, processed vocals, glossy production"),
    Vibe("cottagecore", "Cottagecore",
         "aesthetic",
         "flowers, linen, baking, meadows, soft pastoral light",
         "acoustic folk guitar, mandolin, soft female vocal, gentle waltz"),
]

# TWO ENTRIES WERE FIXED BY MEASUREMENT, NOT BY TASTE
# ---------------------------------------------------
# `scripts/06_probe_vibes.py` fires every query above at the real index and
# reports what fails. The first draft had two problems, and candidate
# rewordings were scored rather than argued about:
#
#   Temple   "harmonium and tabla, devotional chanting"       0.445  dead
#            "indian devotional bhajan, harmonium, group
#             chanting, temple bells"                         0.673  fixed
#
#   Minimal  "single sustained piano notes, wide silence"     60% of its
#            results were identical to Melancholy - one vibe, two names
#            "one repeated marimba figure, dry recording"     0% overlap
#
# The Minimal fix is a happy accident of vocabulary: minimalist *music*
# genuinely is repeating marimba figures, so the honest description of the
# sound is also the one that separates it from "sad piano".


AXES = ("scene", "time", "mood", "occasion", "aesthetic")

# WHICH AXES CAN BE LABELLED BY IMAGE SEARCH, AND WHICH CANNOT
# ------------------------------------------------------------
# The development photos were labelled by searching Wikimedia Commons for each
# vibe. That works when the vibe is a concrete noun and fails badly when it is
# an abstract quality, because Commons search matches *words in titles*.
#
# Reliable - searching the word returns the thing:
#     scene      "Beach of Cape Fiolent"          a beach
#     occasion   "A fancy Indian wedding"         a wedding
#     time       "Louvre at night"                a night scene
#
# Unreliable - searching the word returns something merely NAMED that:
#     mood       "calm"        -> Calm Air, an airline. Photos of a Saab.
#                "melancholy"  -> Durer's Melencolia I engraving
#                "nostalgic"   -> the Istanbul Nostalgic Tram
#                "romantic"    -> Romanticism, the art movement
#                "dramatic"    -> the Academy of Dramatic Arts, a building
#     aesthetic  "old money"   -> literal banknotes and a money box
#                "streetwear"  -> graphics with the word on a background
#                "minimal"     -> minimal surfaces, from mathematics
#                "y2k"         -> the Y2K computer bug
#
# 43 of 118 development photos carry a label of the second kind. Any accuracy
# figure computed over them is measuring word collisions, not vision.
#
# This matters because it corrects an earlier conclusion. Week 4 reported
# "mood 22.7%" and Week 6 built on it to argue CLIP is weak at mood. That is
# not established: a model refusing to call a photograph of an aeroplane
# "calm" is behaving correctly, and would be marked wrong.
#
# Mood remains untested, not disproven. Testing it needs labels a person wrote
# while looking at the photo.
# HOW TO SEARCH FOR A PHOTO THAT HAS A MOOD
# -----------------------------------------
# Searching an image library for "melancholy" returns Durer's engraving and a
# book cover, because search engines match words in titles and "melancholy" is
# mostly used as a name. Searching for "calm" returns an airline.
#
# The fix is to search for the *photographic situation* that carries the mood
# instead of the mood itself. A rain-streaked window is melancholy without ever
# using the word; a crowd with raised hands is euphoric.
#
# This also explains why the first attempt produced photos with no mood at all.
# A person asked to name the mood of those photos declined 83% of the time -
# not out of indecision, but because a photograph of a tram has no mood.
#
# Only the abstract axes need this. "Beach" and "Indian wedding" are perfectly
# good search terms for themselves.
PHOTO_SEARCH_TERMS = {
    # ------------------------------------------------------------- mood
    "nostalgic":  ["faded old family photograph", "vintage film grain street scene"],
    "romantic":   ["couple silhouette at sunset", "couple holding hands close"],
    "melancholy": ["rain drops on a window", "empty bench in fog"],
    "euphoric":   ["concert crowd hands raised", "festival fireworks celebration"],
    "confident":  ["fashion portrait direct gaze", "studio portrait strong pose"],
    "calm":       ["still lake reflection at dawn", "misty quiet morning water"],
    "playful":    ["children laughing playing water", "dog running splashing"],
    "dramatic":   ["storm clouds dramatic light", "silhouette against dramatic sky"],

    # -------------------------------------------------------- aesthetic
    "old_money":        ["vintage library interior leather", "classic tailored suit portrait"],
    "streetwear":       ["street fashion urban outfit", "sneakers against graffiti wall"],
    "traditional_indian": ["saree portrait woman", "mehndi henna decorated hands"],
    "minimal":          ["minimalist white interior", "single object negative space"],
    "vintage_bollywood": ["retro indian film actress", "1970s indian fashion"],
    "y2k":              ["2000s party flash photography", "early 2000s fashion teenagers"],
    "cottagecore":      ["wildflower meadow linen dress", "rustic kitchen baking bread"],
}


def search_terms(vibe):
    """What to type into an image search to find photos of this vibe.

    Falls back to the label, which is right for concrete vibes and wrong for
    abstract ones - hence the table above.
    """
    return PHOTO_SEARCH_TERMS.get(vibe.name, [vibe.label])


LABEL_RELIABLE_AXES = ("scene", "time", "occasion")
LABEL_UNRELIABLE_AXES = ("mood", "aesthetic")


def names():
    """Every vibe name, in order."""
    return [v.name for v in VIBES]


def by_axis(axis):
    """All vibes on one axis."""
    return [v for v in VIBES if v.axis == axis]


def get(name):
    """Look up one vibe by name. Raises KeyError if it does not exist."""
    for vibe in VIBES:
        if vibe.name == name:
            return vibe
    raise KeyError(f"no vibe named {name!r}")


def music_queries():
    """(name, music_query) for every vibe - what gets sent to CLAP."""
    return [(v.name, v.music_query) for v in VIBES]


def summary():
    """Counts per axis, for printing."""
    return {axis: len(by_axis(axis)) for axis in AXES}
