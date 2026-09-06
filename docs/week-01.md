# Week 1 — Build the song catalog

**Goal:** end the week with a file containing tens of thousands of real songs,
each with a working 30-second audio preview.

Everything later depends on this. No catalog, no project.

---

## Why this is step one

The system recommends songs, so it needs songs to recommend. But not just song
*names* — we need the actual audio, because in Week 2 a model has to *listen*
to each track to understand what it sounds like. A song title tells a model
almost nothing.

That is the constraint that decided our data source. Most music APIs give you
metadata but no audio. The iTunes Search API gives us both, needs no signup,
and its Indian catalog is excellent.

---

## What the code does

Three files, in the order they matter:

**`src/catalog/seeds.py`** — the list of things to search for.
The API has no "give me everything" button; you must search. So we run ~90
different searches spanning genres (*bollywood dance*), moods (*hindi breakup
songs*), eras (*bollywood 90s*) and artists (*arijit singh*). Mixing those axes
gives much wider coverage than any single one.

**`src/catalog/itunes.py`** — the actual fetching.
Calls the API, keeps the fields we care about, drops tracks with no preview
audio, and removes duplicates as it goes.

**`scripts/01_harvest_itunes.py`** — the thing you run.

```bash
python scripts/01_harvest_itunes.py --quick   # fast test
python scripts/01_harvest_itunes.py           # the real run
python scripts/02_inspect_catalog.py          # look at the result
```

---

## Three habits worth stealing from this code

These are not about music. They are how working ML systems get built, and
Week 1 is a cheap place to learn them.

### 1. Make long jobs resumable

`harvest()` reads the CSV it already wrote and skips searches it has already
done. Stop it with `Ctrl+C`, run it again, and it continues.

This looks like over-engineering for a five-minute job. It is not. In Week 2 we
run a job that takes *hours*, and Colab will disconnect partway through. If you
learn the pattern now on a job where failure is cheap, you will write it by
reflex when failure is expensive.

### 2. Fail slowly on purpose

The code sleeps 3 seconds between requests. We could go faster. But APIs block
clients that hammer them, and being blocked costs far more than the minutes we
saved. When you are a guest on someone's server, be a polite one.

### 3. Look at your data before you model it

That is the entire job of `02_inspect_catalog.py`. Run it and ask:

- Is the Indian/English balance what you wanted?
- Is one genre swallowing everything?
- Do a few of those `preview_url` links actually play in a browser?

A surprising share of "the model is broken" turns out to be "the data was
broken and nobody looked." Looking takes two minutes.

---

## Something that already went wrong

The first run printed `Children�s Music` instead of `Children's Music`.

Windows terminals often default to an old text encoding that cannot represent
anything outside basic English. Harmless for a curly apostrophe — but our
catalog is full of Hindi and Punjabi titles, and on those it does not garble
the output, it **crashes the script**.

Fix: `src/console.py` switches output to UTF-8. Every script that prints track
names calls it first.

Worth internalising: this bug appeared in the first five minutes, on the
friendliest possible input. Small encoding problems on small data are large
encoding problems on large data.

---

## Done when

- [ ] `data/raw/itunes_catalog.csv` has 10,000+ unique tracks
- [ ] 100% of rows have a `preview_url`
- [ ] The lane split looks deliberate, not accidental
- [ ] You have opened three preview URLs and heard music

---

## Next

**Week 2 — teach the computer to listen.** Every track gets streamed through
CLAP, turned into 512 numbers, and stored in a searchable index. By the end you
will type *"sad piano"* and get back sad piano, having never once used the word
"sad" as a tag.
