# Week 2 — Teach the computer to listen

**Goal:** turn every song in the catalog into 512 numbers, and be able to
search those numbers with an English sentence.

By the end of this week you can type *"rainy café window, quiet piano"* and get
back real songs. No photos yet — but the hard half of the problem is solved.

---

## The problem Week 2 actually solves

Week 1 left us with 13,497 songs and a preview URL for each. A computer cannot
compare a sentence to an audio file. They are not the same kind of thing.

So we convert both into the same kind of thing: **a list of 512 numbers**.

That list is called an **embedding**, and the rule that makes it useful is:

> Things that mean similar things get similar numbers.

Once a song and a sentence are both 512 numbers, "which song matches this
sentence?" becomes ordinary arithmetic — and arithmetic is something computers
are extremely good at.

---

## CLAP, and why it is the whole project

Most models embed one kind of thing. CLAP embeds two, and that is the point.

CLAP has an **audio encoder** and a **text encoder**, and they were trained
*together* on hundreds of thousands of (audio clip, caption) pairs with a
single instruction:

> When the caption describes the audio, make the two vectors point the same
> direction. When it does not, push them apart.

Do that a few hundred thousand times and something remarkable falls out. The
two encoders end up sharing one space. A sentence and a song can be compared
directly, even though one is words and the other is sound.

```
"slow acoustic guitar, warm and nostalgic"  →  text encoder  →  [512 numbers]
                                                                     ↕ compare
🎵 an actual acoustic folk track            →  audio encoder →  [512 numbers]
```

Nobody wrote a rule saying "acoustic guitar means nostalgic". The geometry
learned it.

**This is why the project needs no (photo, song) training pairs to get
started.** We are not the first people to solve the hard part. We are
connecting two things other people already solved.

---

## Cosine similarity, in one paragraph

Two vectors point in some directions. The angle between them measures how alike
they are — small angle, similar meaning. The cosine of that angle is a number
from -1 to 1, and it is what we mean by "score".

There is a shortcut. If you first rescale every vector to length exactly 1
(**normalising**), then the cosine is just the dot product: multiply the pairs
of numbers, add them up. That is why `clap.py` normalises everything and why
the index is an *inner product* index. Same maths, far less of it.

---

## What the code does

**`src/encode/audio.py`** — download a preview, decode the `.m4a`, convert it to
48,000 samples per second (what CLAP expects), and cut it into 10-second
windows.

**`src/encode/clap.py`** — load CLAP; embed audio; embed text. `embed_track`
does the full job for one song: window it, embed each window, average, normalise.

**`src/encode/store.py`** — an append-only file that survives being killed
partway through.

**`src/retrieve/index.py`** — the FAISS index and the search call.

```bash
python scripts/03_embed_catalog.py --quick   # 60 songs, proves it works
python scripts/03_embed_catalog.py           # the real run
python scripts/04_build_index.py             # pack them into the index
python scripts/05_search.py                  # search with sentences
```

---

## Why we cut every preview into three 10-second windows

CLAP only looks at 10 seconds at a time. Feed it a 30-second preview and it
silently keeps 10 seconds and discards the rest — and depending on
configuration, it may keep a **random** 10 seconds, which would give different
results every run.

So we do the cutting ourselves. Three windows, embed each, average the three,
normalise the average. Nothing gets discarded, and the result is identical
every time.

This is a small decision that prevents a horrible class of bug: results that
change between runs for no visible reason.

---

## The habit this week: never lose work

Week 1 made the harvester resumable for a five-minute job. That was practice.
This week the job takes **hours**, and Colab will disconnect you partway
through.

`store.py` writes two files that grow together:

```
clap_audio.f32       raw numbers, 512 per song, back to back
clap_audio_ids.csv   one track_id per line, same order
```

Row 7 of the ids file describes numbers 7×512 to 8×512−1 of the vector file.
That is the entire format — no header, no index, nothing to corrupt.

Two details do real work:

- **Numbers are written before the id.** If we die in between, there is an
  orphan vector with no id, which `load()` trims off. The other order would
  leave an id pointing at numbers that were never written — a silent
  misalignment that would poison every search result afterwards.
- **Dead previews get logged too.** Otherwise every rerun retries the same
  broken URLs forever.

Compare that to `np.save`: a `.npy` file keeps its length in a header at the
front, so appending means rewriting the header, and a crash mid-rewrite
corrupts the whole file.

---

## The bug that cost an afternoon, and the one line that would have caught it

We started on `laion/larger_clap_music`, for the obvious reason: a CLAP
fine-tuned on music should beat a general one.

That upload is broken. Its **text projection layer was never trained** - every
bias is exactly 0.0, and the weights sit at a third the spread of a trained
layer. Nothing warns you. It downloads, loads without error, produces
embeddings, builds an index, and answers searches.

But its text encoder returned **almost the same vector for every sentence**.
"aggressive heavy metal" and "quiet sad piano" came back 0.999 similar. So
every query returned the same songs, which looks like *"the AI is not very
good"* rather than *"the model is broken"*. That is what makes this class of
bug dangerous: it produces plausible-looking output forever.

### How it was actually found

Not by reading code. By testing the assumption against known answers:

1. Embed 24 tracks across 6 genres.
2. Ask 5 questions where the correct answer is known (*"smooth jazz
   saxophone"* should return the jazz tracks).
3. Count how often it is right, against random guessing.

The first version scored **1/5, with the same songs winning every query**.

Two "fixes" were tried first and both were wrong in an instructive way.
Subtracting the average vector made the score spread look five times
healthier (0.0365 -> 0.1837) while accuracy stayed at chance. **A number that
looks better is not a fix.** Only the known-answer test could tell the
difference.

### The one line

A broken text encoder is detectable in a single measurement. Embed a few
wildly different sentences and measure the length of their average. Unit
vectors pointing in genuinely different directions partly cancel, so a healthy
encoder gives something well below 1:

```
laion/clap-htsat-unfused    0.67   healthy
laion/larger_clap_general   0.69   healthy
laion/larger_clap_music     0.9997 dead
```

That check is now `self_test()` in `src/encode/clap.py`, and
`require_working_text_encoder()` runs it before the long embedding job starts.
A dead model now stops in two seconds instead of producing four hours of
meaningless numbers.

### What we chose, measured not assumed

| checkpoint | top-1 | mean rank of first correct |
|---|---|---|
| `laion/clap-htsat-unfused` | **3/5** | **2.4** |
| `laion/larger_clap_general` | 1/5 | 2.6 |
| random guessing | - | ~3.5 |

**The lesson worth keeping:** a published, popular, correctly-named model can
be silently broken. Test the assumption on 24 items before trusting it on
13,497.

---

## The open question we test instead of assume

CLAP's training captions were overwhelmingly English. **Nobody has told us
whether it understands Hindi or Punjabi vocals.** Most of our catalog is
Indian music, so this is not a footnote — it could be the difference between a
project that works and one that does not.

So we ask it directly:

```bash
python scripts/05_search.py --language-test
```

This fires four differently-phrased queries at the index and prints which lane
(`indian` / `english`) each result came from.

- If Indian-specific queries pull mostly `indian` tracks, CLAP is genuinely
  hearing the difference.
- If the lanes look random, it is not — and the fix is to describe
  *instrumentation and mood* in the query (*dhol drums, sitar, tabla*) and let
  a metadata filter handle language.

Either answer is fine. Finding out in Week 9 would not be.

**Write down what you observe.** This is the first real experimental result in
the project, and it belongs in the final report.

---

## How long this takes

| Where | 13,497 songs |
|---|---|
| Google Colab (T4 GPU) | ~25 minutes |
| Laptop (CPU) | several hours |

The CPU path genuinely works — it is just slow, and it resumes, so three
evening sessions is a perfectly good way to do it. Start with `--quick` either
way; there is no reason to discover a bug on song 4,000.

---

## What "done" looks like

- [ ] `--quick` run finished without errors
- [ ] full catalog embedded (`04_build_index.py` reports coverage above ~90%)
- [ ] vector lengths report `1.0000` — if not, cosine similarity is broken
- [ ] the six demo queries return sensible, *different* songs from each other
- [ ] you have run `--language-test` and written down what you saw

If all six demo queries return the same songs, something is wrong: most likely
the index was built from a stale or empty store.

---

**Next week:** the vibe vocabulary — deciding what words we will use to
describe a photo, and collecting the photos to test on.
