# Week 6 — Prove it, or find out it is not true

**Goal:** stop saying "the songs seem reasonable" and produce numbers.

```bash
python scripts/11_evaluate.py
```

---

## The result

118 photos, top-5 recommendations, 11,870-song catalogue.

| method | recall@5 | nDCG@5 | diversity | IN vibes | EN vibes | **balanced lane** |
|---|---:|---:|---:|---:|---:|---:|
| random | 0.0% | 0.000 | 1.00 | 52% | 56% | 54% |
| popularity | 0.0% | 0.000 | 1.00 | 100% | 0% | 50% |
| keyword | 1.0% | 0.008 | 0.83 | 85% | 64% | 75% |
| keyword⁺ *(true vibe)* | 1.5% | 0.010 | 0.86 | 96% | 93% | **95%** |
| **ours** | **21.2%** | **0.187** | **1.00** | 81% | 60% | 71% |
| oracle *(by construction)* | 100.0% | 1.000 | 0.93 | 88% | 100% | 94% |

**20.8x the keyword baseline.** Random and popularity score exactly zero —
across 118 photos they never once land on a song the oracle would have chosen.

Diversity is a clean 1.00: five songs, five different artists, every time.

---

## Why these baselines

**Random** is the floor. If you cannot beat a hat, nothing else matters.

**Popularity** returns the most prolific artists and ignores the photo
entirely. It is here because it embarrasses people: popular music is broadly
agreeable, so it often looks fine to a casual eye.

**Keyword** is the one that matters. String matching on titles, genres and
search terms — no models, the thing you would build in an afternoon. If two
neural networks cannot beat a genre substring check, the complexity has not
earned itself.

**Oracle** is not a rival. It is the music half given the *correct* vibe, so
the gap shows what the vision step costs.

---

## Three bugs in the evaluation code, all found by the baselines

This is the part worth reading, because none of them would have crashed
anything.

### 1. A whole metric silently vanished

`summarise()` read its list of metrics from `rows[0]`. Not every photo carries
every metric — `lane` only applies to culturally-coded vibes — and the first
photo alphabetically was a *beach*. So `lane` was absent from row zero and
dropped for every method. The report printed a tidy dash and nobody would have
noticed.

### 2. A metric a photo-blind baseline could win

The first version measured lane accuracy only on Indian vibes:

```
random    52%
popular  100%      <- ignores the photo entirely
ours      81%
```

The most prolific artists in this catalogue are Indian, so "always return
popular songs" always returns Indian songs — and scored perfectly on a metric
asking "are these Indian?".

**A one-sided metric rewards bias, not accuracy.** Fixed by measuring both
directions and averaging: Indian-coded vibes should return Indian music,
Western-coded vibes should return English music. Popularity immediately fell
to 50% — exactly chance — because it scores 100% on one half and 0% on the
other.

### 3. A baseline running with information we did not have

`keyword_songs(gold, ...)` handed the keyword baseline the **true** vibe while
our pipeline had to predict its own. Oracle-level knowledge, given to a rival,
by accident.

Fixed by running both: `keyword` gets the vibe *we* predicted (the fair
comparison), `keyword+` gets the true one and is labelled as such. Our margin
went **up**, from 13.9x to 20.8x.

### What this says

All three were caught the same way: **the baselines disagreed with each other
in ways that made no sense.** Random at 52% beside popularity at 100% is
absurd, and the absurdity is what exposed the metric.

Evaluation code fails silently. There is no crash and no obviously wrong
output — just a plausible number that goes straight into a report. Baselines
are what make the silence audible.

---

## What the numbers say to do next

### The vision step is the bottleneck

It finds the right vibe for **60/118 photos (50.8%)**. The oracle recovers
everything by construction; we recover 21.2%. Almost the entire shortfall is
the vision step guessing wrong — and Week 4 already showed where it fails:
**mood 22.7%, light 16.7%**, against occasion 63% and scene 56%.

Mood is the axis that drives music choice hardest. That is the case for
Week 7's trained classifier, now supported by two independent measurements
rather than an assumption.

### Language belongs in a filter, not an embedding

Keyword matching beats us on balanced lane accuracy — **75% against our 71%**,
and 95% when given the true vibe. String matching on `genre` and `seed_term`,
fields that literally contain the words *bollywood* and *punjabi*, gets
language right more often than our embeddings do.

This is the same conclusion Week 2 reached from the other direction: CLAP
separates South Asian from Western music but cannot tell Punjabi from Hindi
from Tamil, so language should be handled by metadata.

Two independent weeks, two different methods, one instruction for Week 10:
**add a metadata language filter.** Not a hunch — a measured 24-point gap with
a known cause.

---

## What this evaluation cannot tell you

The oracle treats CLAP's judgement as truth. If CLAP is wrong about what a
wedding sounds like, every method is being scored against a wrong answer, and
recall@5 measures agreement with a machine rather than correctness.

Balanced lane accuracy is the antidote — that field came from the Week 1
catalogue and **no model in the pipeline has ever seen it** — but it covers
only 24 of 118 photos, and only the language dimension.

The number that would settle it is a person listening and saying whether the
song fits. That is human evaluation, it is the honest gold standard, and it
does not exist yet.

**And every number above is measured on Wikimedia photography** — composed,
well lit, shot on real cameras. Phone photos are none of those things. The
evaluation is not finished until it runs on a real camera roll.

---

## What "done" looks like

- [x] four baselines, including the one that should be hard to beat
- [x] three metrics that fail in different ways
- [x] an independent check no model in the pipeline can game
- [x] the comparison made fair after discovering it was not
- [ ] human preference ratings
- [ ] the same table on real phone photos

---

**Next:** Week 7 trains a classifier on the axis this week identified as the
weakest link — mood — because that is where the measurement says the next
improvement lives.
