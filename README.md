# Music-Instagram

**Give it a photo. It recommends the songs that fit.**

You are about to post a story. You have the photo — a hike at golden hour, a
café window in the rain, a wedding — and now you are scrolling through a music
library trying to find something that *feels* like the picture. That search is
the annoying part.

This project removes it. Upload a photo, get five songs ranked by how well they
match, each with the exact fifteen seconds to use and one line explaining why.

> There is no "search by song" here. The **photo itself is the query**. Nothing
> is being compared against a song you already picked.

---

## How can a model possibly do that?

Here is the whole idea in four steps.

**1. A computer can turn a photo into a list of numbers.**
Models like CLIP and SigLIP read an image and produce ~512 numbers describing
what is in it. Similar photos get similar numbers. That list is called an
**embedding**.

**2. A computer can turn a song into a list of numbers too.**
A model called **CLAP** listens to audio and produces its own embedding.

**3. But those two sets of numbers live in different worlds.**
An image embedding and an audio embedding cannot be compared directly. They
were built by different models that never met. This is the actual problem.

**4. Text is the bridge.**
CLIP connects *images and text*. CLAP connects *audio and text*. Both touch
text — so we route through it:

```
photo → describe it in words → turn those words into a CLAP embedding
      → find the songs whose audio embedding is closest
```

Because CLAP's text and audio live in the *same* space, a sentence describing a
photo can retrieve music directly. "Wide open mountains at sunset, calm and
nostalgic" lands near airy acoustic folk. No rules written by hand — the
geometry does the work.

Later we train a small model to skip the middle step entirely and go from image
straight to the music space. That is the part that makes this a machine
learning project rather than a wiring job.

📄 **[Full architecture and 12-week plan →](https://claude.ai/code/artifact/68d5933d-4ce0-4f77-8e7a-1b2d25d8d5a1)**

---

## Progress

| Week | Milestone | Status |
|:----:|-----------|:------:|
| 1 | Build the song catalog | ✅ done |
| 2 | Turn every song into an embedding, build the search index | ✅ done |
| 3 | Define the "vibe" vocabulary, collect photos | 🟡 next |
| 4 | Teach a vision model to describe a photo's vibe | ⬜ |
| 5 | **First working demo** — upload a photo, get songs | ⬜ |
| 6 | Measure it properly against baselines | ⬜ |
| 7 | Train the aesthetic classifier | ⬜ |
| 8–9 | Train the image → music bridge | ⬜ |
| 10 | Ranking, variety, and picking the best 15 seconds | ⬜ |
| 11 | The agent layer and explanations | ⬜ |
| 12 | Deploy publicly and write it up | ⬜ |

---

## Try it right now

No installation, no API key, no signup. Python 3 is all you need.

```bash
# a 30-second taste — 5 searches
python scripts/01_harvest_itunes.py --quick

# see what you collected
python scripts/02_inspect_catalog.py
```

Happy with that? Collect the real catalog (about 5–8 minutes):

```bash
python scripts/01_harvest_itunes.py
```

Safe to stop with `Ctrl+C` at any point — rerunning resumes where it left off.

### Then make it searchable

This part needs a few libraries:

```bash
python -m pip install -r requirements.txt
```

```bash
python scripts/03_embed_catalog.py --quick   # 60 songs, proves it works
python scripts/03_embed_catalog.py           # the real run — slow, resumable
python scripts/04_build_index.py             # pack them into a search index
python scripts/05_search.py                  # search with a sentence
```

Then ask it for something:

```bash
python scripts/05_search.py "rainy café window, quiet piano, soft melancholy"
```

No photo involved yet — that arrives in Week 5. This is the step that proves
words can retrieve music, which is the assumption everything else rests on.

Embedding the full catalog takes about **4 hours on a laptop CPU** or
**20 minutes on a free Colab T4** — see
[`notebooks/week02_embed_catalog.ipynb`](notebooks/week02_embed_catalog.ipynb).
Either way it is resumable: stop with `Ctrl+C` and rerun to continue.

---

## Where the songs come from

| Source | What it gives us | Cost |
|--------|------------------|------|
| **iTunes Search API** | 30-second `.m4a` previews, plus artist/genre/year. Excellent Hindi, Punjabi and English coverage | Free, no key |
| **Deezer API** | More previews, fills gaps iTunes misses | Free, no key |
| **MTG-Jamendo** | ~55k Creative Commons tracks with mood tags — our training set | Free |
| **Last.fm** | Crowd tags like *romantic*, *gym*, *monsoon*, *90s* | Free key |

**On the audio:** we stream a preview, compute its embedding, and discard the
file. This repo contains **no audio and never will** — only numbers and
metadata. That keeps it legal to publish and small enough to clone.

---

## Layout

```
Music-Instagram/
├── scripts/     numbered steps — run them in order
├── src/
│   ├── catalog/   fetching songs from the music APIs      ← Week 1
│   ├── encode/    turning audio and images into numbers   ← Week 2
│   ├── vibe/      understanding what a photo feels like   ← Weeks 3–4
│   ├── bridge/    the model we train ourselves            ← Weeks 8–9
│   ├── retrieve/  search, filter, rank                    ← Weeks 2, 10
│   ├── agent/     orchestration and explanations          ← Week 11
│   └── eval/      how we prove any of this works          ← Week 6
├── notebooks/   Google Colab notebooks for the heavy training
├── app/         the web demo
├── docs/        plain-English notes on each week
└── data/        not in git — see data/README.md
```

---

## Honest limitations

- **It is not an Instagram feature and cannot be.** Instagram has no public API
  for story music. This is a standalone web demo — the recommendation engine
  that would sit behind such a feature.
- **Previews are 30 seconds, not whole songs.** Usually the chorus, which is
  what a story uses anyway, but it is a real limitation and we say so.
- **CLAP separates South Asian from Western music, but not Punjabi from Hindi
  from Tamil.** Measured on all 11,870 indexed songs: Indian-specific queries
  return 100% Indian-lane tracks against a 44.4% baseline, and asking for
  sitar and tabla returns actual Ravi Shankar recordings. But a "punjabi"
  query still returns Tamil, Hindi and Urdu tracks, so language is handled by
  metadata filters, not by the embedding. See [docs/week-02.md](docs/week-02.md).
- **Descriptions must be about sound, not scenery.** "Gym workout" means
  nothing to a model that only hears audio; "hard hitting drums, shouted
  vocals" does.

---

*A major project in AI/ML. Built in the open, one week at a time.*
