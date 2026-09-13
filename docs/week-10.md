# Week 10 — Two corrections the measurements asked for

Week 6 produced a table. This week acts on it. Both changes are **trade-offs
that were measured rather than assumed**, and both are on by default with a
switch to turn them off.

---

## 1. Language is metadata, not acoustics

Two weeks reached this independently.

**Week 2** found CLAP separates South Asian from Western music reliably but
cannot tell Punjabi from Hindi from Tamil — a "punjabi bhangra" query returned
A.R. Rahman (Tamil), *Dilbaro* (Hindi) and an Urdu ghazal.

**Week 6** found plain string matching on `genre` and `seed_term` beat our
embeddings at language: 75% balanced lane accuracy against our 71%.

Whether a song is in Hindi is a **fact recorded in the catalogue**, not
something to infer from how it sounds. `src/retrieve/filters.py` looks it up.

### Measured

| method | recall@5 | IN vibes | EN vibes | balanced |
|---|---:|---:|---:|---:|
| ours | 24.4% | 85% | 68% | 76% |
| ours + language filter | 20.3% | **100%** | 69% | **85%** |

Language accuracy up 9 points, Indian-coded vibes perfect. Recall down 4.

### A hypothesis I had, and it was wrong

I assumed the recall drop was a metric artifact: a filtered method scored
against an *unfiltered* oracle counts every correctly-filtered-out song as a
miss. So the evaluation now scores the filtered method against an oracle
applying the same filter.

**The drop survived** — 20.3% against 24.4%. The trade-off is real. Filtering
discards our best-scoring matches when they are in the wrong lane and replaces
them with lane-correct songs further down the ranking.

### On by default anyway

`recall@5` measures *agreement with CLAP*, and this deliberately overrides
CLAP on the one dimension two separate weeks measured it to be unreliable
about. Lower agreement is what a correction is supposed to produce.

A wedding photo returning Hindi film music is what a person wants. Agreeing
with CLAP is not a goal in itself.

Two details worth keeping: only unambiguous vibes are filtered, because a
beach is a beach in any language; and the filter drops itself when too few
songs survive, because five slightly-wrong-language songs beat two right ones.

---

## 2. Five different artists can all be lo-fi

Artist diversity was already a comfortable 1.00 — and slightly misleading. It
counts names, not sounds.

So `sound_alike` measures what the songs are like **to each other**, using the
same CLAP vectors that answered "does this match the photo?". No new model, no
new data.

### Measured

| method | recall@5 | artists | **samey** |
|---|---:|---:|---:|
| ours | 24.4% | 1.00 | 0.80 |
| ours + language | 20.3% | 1.00 | 0.79 |
| **ours + both** | 18.8% | 1.00 | **0.71** |
| oracle | 100.0% | 0.91 | 0.80 |

The oracle's own results score 0.80 — **the reranker makes results more varied
than the reference it is measured against.**

Maximal Marginal Relevance, λ = 0.7. Higher values (0.85, 0.9) changed nothing
at all on the photos tested; only 0.7 does real work.

---

## A bug worth recording

MMR chooses songs by a blend of relevance and novelty, so the chosen five come
out in MMR order — which put a score of **0.559 above 0.515 and below 0.582**.
A numbered list whose scores go up and down looks broken to anyone reading it;
the selection logic is invisible, only the result is seen.

MMR now decides **which** songs appear; relevance decides **the order they are
shown in**.

---

## And one accusation I withdrew

A rain-on-a-window photo returned **Enter Sandman**, and I blamed the
diversity reranker for reaching too far.

It was already there at λ = 1.0, with no reranking at all. The vision step had
read the photo as *"Rain and monsoon 37% + Road trip 35%"* — and rain on a car
window is a perfectly reasonable road trip. The retrieval did what it was
asked; the reranker had nothing to do with it.

---

## Where the numbers stand

| | Week 6 | now |
|---|---:|---:|
| recall@5 | 21.2% | 24.4% |
| vs keyword baseline | 20.8x | **39.7x** |
| vision step correct | 50.8% | 55.9% |
| balanced language | 71% | **85%** |
| samey | 0.80 | **0.71** |

The demo serves everything on: language filter and diversity reranking.
`--no-language-filter` and `--diversity 1.0` turn them off.

---

**Next:** deploy it publicly, so the whole thing is a link rather than a
localhost port.
