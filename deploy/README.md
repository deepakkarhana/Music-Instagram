# Putting this on the internet

The demo runs on your laptop at `127.0.0.1:8000`. This turns it into a URL
anyone can open — the thing you put in a CV, a LinkedIn post, or a message to
someone who asked what you have been building.

**Target: a free Hugging Face Space.** Free tier, 16 GB RAM, 2 CPU cores, no
card required, and it sleeps when nobody is using it and wakes on the next
visit. Good enough: a request is ~250 ms once the models are warm.

---

## What has to change, and why

### 1. The data cannot come from git

`data/` is gitignored, deliberately — the repo ships code, not 28 MB of
embeddings and a 4 MB catalogue. But the Space needs both to answer a single
request.

Two files are needed at startup:

```
data/raw/itunes_catalog.csv     4 MB    song metadata
data/interim/clap_audio.f32    28 MB    the embeddings
data/interim/clap_audio_ids.csv          which row is which song
```

The index itself is rebuilt from those in about two seconds, so it does not
need uploading.

The Space stores them as a **dataset repository**, which is where Hugging Face
expects data to live. `prepare.py` uploads them.

### 2. Models load once, and the first visit pays for it

CLIP and CLAP are about 2 GB together and take ~30 seconds to load. On a Space
that sleeps, the first visitor after a nap waits for that.

The page already handles this: `/api/health` returns 503 until the models are
ready, and the front end says so rather than appearing broken.

### 3. CPU only

No GPU on the free tier. That is fine — this project has only ever run on CPU,
and a request is ~250 ms. Nothing needs changing.

---

## Steps

**1. Make an account** at [huggingface.co](https://huggingface.co) if you have
not already. You have one — `HF_TOKEN` is already set on your machine.

**2. Upload the data** (once):

```bash
python deploy/prepare.py
```

It creates a private dataset repo called `Deepakkarhana01/music-instagram-data`
and uploads the three files.

**3. Create the Space:**

- [huggingface.co/new-space](https://huggingface.co/new-space)
- Name: `music-instagram`
- SDK: **Docker**
- Hardware: **CPU basic (free)**
- Visibility: **Public**

**4. Push the code:**

```bash
git remote add space https://huggingface.co/spaces/Deepakkarhana01/music-instagram
git push space main
```

**5. Add the token** so the Space can read its private dataset:

Space → **Settings** → **Variables and secrets** → New secret
- Name: `HF_TOKEN`
- Value: your token

The Space rebuilds and goes live at:

```
https://huggingface.co/spaces/Deepakkarhana01/music-instagram
```

---

## Why Docker rather than the Gradio SDK

Gradio would be fewer steps, and would replace the interface with Gradio's own.
The page in `app/static/` is part of the work — the drag-and-drop, the vibe
chips, the playable previews, the visible music query. A Dockerfile keeps
exactly what runs locally.

It also means the Space runs the *same* FastAPI app you have been testing, so
"it worked on my laptop" and "it works in production" are the same claim.

---

## What still costs nothing

No audio is stored or served. Previews stream from Apple straight into the
browser's `<audio>` element, exactly as they do locally. The Space holds code,
28 MB of numbers, and a CSV.
