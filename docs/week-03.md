# Week 3 — Decide what words the system is allowed to think in

**Goal:** a fixed vocabulary of "vibes", each proven to retrieve real music,
plus a set of real photos to test it against.

No new model this week. The work is deciding *what the system is allowed to
say*, and then checking that those words actually do something.

---

## Why a vocabulary, and not just free text

The obvious approach is to let a language model write whatever description it
likes for each photo. That works, and we will do it eventually.

But you cannot compute accuracy against free text. If the model writes *"warm
golden evening feelings"* for one photo and *"amber sunset mood"* for another,
there is no way to say whether it got them right, whether it is consistent, or
whether it improved after a change.

A fixed list of names fixes that. The same 40 names become:

- the labels for the classifier we train in **Week 7**
- the categories the evaluation reports in **Week 6**
- the thing a human annotator ticks in **Week 6**

Free text can come later, on top of a measurable base.

---

## The rule this vocabulary is built on

Week 2 measured something that only looks obvious afterwards:

```
"high energy gym workout, aggressive hip hop with heavy bass"
    →  Rainy Day,  Bossa Nova,  Beach Vibes
```

CLAP only ever hears audio. **A gym is a place, not a sound.** "Gym" and
"workout" describe a room and an activity, and neither has a timbre. Rewriting
the same intent in sound — *"aggressive rap, hard hitting drums, angry male
rapper shouting"* — found gym music immediately.

So every vibe carries **two separate descriptions**, and they never mix:

| field | language | used by | example (Gym) |
|---|---|---|---|
| `visual_cues` | appearance only | the vision model, Week 4 | weights, mirrors, sportswear, harsh indoor light |
| `music_query` | sound only | CLAP, working today | hard hitting drums, distorted bass, shouted male rap vocals |

The vibe **name** is the only thing joining them. That join is the entire
architecture in one line: a photo is matched on how it *looks*, and music is
retrieved on how it *sounds*, and the name carries meaning across the gap.

---

## The 40 vibes, on five axes

A photo is rarely one thing. A friend on a Goa beach at sunset is a place, a
time and a mood at once, so vibes sit on axes and a photo can take one from
each.

| axis | count | what it captures | examples |
|---|:--:|---|---|
| `scene` | 12 | where it is | mountains, cafe, city at night, temple, gym |
| `time` | 4 | when, and what light | golden hour, night, monsoon, bright daylight |
| `mood` | 8 | what it feels like | nostalgic, melancholy, euphoric, confident |
| `occasion` | 9 | what is happening | Indian wedding, festival, workout, studying |
| `aesthetic` | 7 | how it is styled | old money, streetwear, traditional Indian, Y2K |

See [`src/vibe/taxonomy.py`](../src/vibe/taxonomy.py).

---

## Testing the vocabulary before trusting it

A list of 40 guesses is worth very little on its own. Somebody decided that
"melancholy" and "minimal" are different, and that temples sound like
harmonium — those are hypotheses, and the expensive time to discover they are
wrong is Week 6, after everything is built on top.

```bash
python scripts/06_probe_vibes.py          # the problems
python scripts/06_probe_vibes.py --full   # every vibe and what it retrieves
```

It fires all 40 queries at the real 11,870-song index and looks for three
specific failures:

**Dead vibes** — the best match scores low, so nothing in the catalog sounds
like this. Either the words are wrong or the music is missing.

**Twin vibes** — two vibes retrieve the same songs. They are one vibe wearing
two names.

**Hub songs** — one track appears under many unrelated vibes. It sits near the
middle of the space and crowds out everything else.

### What the first draft got wrong

Two failures, both fixed by measuring candidate rewordings rather than
arguing about them:

```
Temple    "harmonium and tabla, devotional chanting"           0.445  dead
          "indian devotional bhajan, harmonium, group
           chanting, temple bells"                             0.673  fixed

Minimal   "single sustained piano notes, wide silence"         60% of its
          results were identical to Melancholy
          "one repeated marimba figure, dry recording"         0% overlap
```

The Minimal fix is a pleasing accident. Minimalist *music* genuinely is
repeating marimba figures, so the honest description of the sound turned out
to be the one that separates it from "sad piano". The first attempt described
the *feeling* of minimalism, and feelings are exactly what CLAP cannot hear.

### Where it stands now

```
Dead: 0/40    Twin pairs: 0    Hubs: 2
Best-match score: mean 0.659, min 0.460, max 0.797
```

The two hubs — *Rainy Day* and *Simply Magic*, each surfacing under 6
unrelated vibes — are left alone deliberately. That is a ranking problem, not
a vocabulary problem, and Week 10 fixes it properly with diversity-aware
reranking. Papering over it here would hide it.

---

## Your half: the photos

This is the part that cannot be generated. The vocabulary above was written
from imagination, and imagination is a poor model of what people actually
post.

**Collect 20–30 photos you would genuinely put on a story.** Your own camera
roll is ideal.

Put them in `data/photos/`, then:

```bash
python scripts/07_check_photos.py
```

It reports what you have, flags files that will cause trouble later, and shows
which vibes are covered and which are missing.

### Aim for spread, not volume

Thirty photos across many vibes beats a hundred of the same holiday. Try to
cover:

- outdoors or travel
- a cafe or indoors
- a night out
- gym or sport
- a wedding or festival
- a portrait or outfit
- food
- something rainy or grey

### What makes a photo useful here

- **Real, not stock.** Stock photos are lit and composed unlike anything on a
  story feed, and a model tuned on them will fail on real ones.
- **Some hard cases.** A few genuinely ambiguous photos are worth more than
  twenty obvious ones — they are where Week 6's evaluation gets interesting.
- **Anything you would actually post.** Including the blurry, badly lit, and
  unflattering. Those are the real distribution.

Nothing you add is committed to git — `data/` is ignored, deliberately.

---

## What "done" looks like

- [x] 40 vibes defined across 5 axes
- [x] every vibe retrieves real music (0 dead)
- [x] every vibe retrieves a *distinct* set (0 twins)
- [ ] 20–30 real photos in `data/photos/`
- [ ] every axis has photos covering at least half its vibes

---

**Next week:** a vision model reads a photo and picks vibes from this list —
and because the list is fixed, we can finally measure whether it is right.
