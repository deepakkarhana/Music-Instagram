# Where this project is

Updated 2026-09-20. Read this first after any break.

## Resume the Claude Code session

```bash
cd C:\Users\ASUS\OneDrive\Desktop\Music-Instagram
claude --resume f062430e-edd2-4bdc-84df-6ba8b5328a72
```

Must be run **from this folder** — sessions are scoped to their directory. Or
run `claude` then `/resume` and pick it from the list.

This file exists so the project is resumable even if that session is gone.

---

## The one thing in progress

Pushing to the Hugging Face Space. Everything is committed and ready; the
command is:

```bash
git push space space-deploy:main --force
```

Then open <https://huggingface.co/spaces/Deepakkarhana01/music-instagram>

Why `space-deploy` rather than `main`: Hugging Face rejects binary files over
plain git, and an early commit on `main` contains `songs.i8`. `space-deploy`
is an orphan branch with one commit and no binary anywhere in its history.

---

## What is built

| Week | What | State |
|---|---|---|
| 1 | iTunes catalogue — 13,497 songs, 100% with audio | done |
| 2 | CLAP embeddings, FAISS index — 11,870 after dedup | done |
| 3 | 40-vibe vocabulary across 5 axes | done |
| 4 | CLIP reads photos | done |
| 5 | Web demo, playable previews | done |
| 6 | Evaluation against baselines | done |
| 7 | Train a mood classifier | **not needed** — CLIP does mood at 67% |
| 10 | Language filter, diversity reranking | done |
| 12 | Browser-only version for a free Static Space | pushing |

## Headline numbers

```
recall@5            24.4%   against 1.0% for keyword matching (39.7x)
random, popularity   0.0%
vision step         55.9%   right vibe
mood accuracy       67%     against 12.5% chance
balanced language   85%     with the metadata filter
```

## Run it

```bash
python app/main.py                  # server version, localhost:8000
python app/main.py                  # then /label to annotate photos
python scripts/11_evaluate.py       # the numbers above
python deploy/export_static.py      # rebuild the browser bundle
```

---

## Things worth not forgetting

**A published model was silently broken.** `laion/larger_clap_music` ships an
untrained text projection — it loads fine and returns the same vector for
every sentence. `self_test()` in `src/encode/clap.py` catches it in one
measurement.

**Two evaluation metrics were wrong before they were right.** Ranking a vibe
against all 40 measures a question the pipeline never asks (real accuracy 49%,
not 20%), and one-sided lane accuracy let a photo-blind baseline score 100%.

**Mood was never weak.** The 22.7% came from labels that were word collisions
— "calm" fetched photos of an airline called Calm Air.

**Only the top vibe decides the language filter.** CLIP scores sit in a narrow
band, so ratio thresholds fire on almost anything: a fifth-place "Vintage
Bollywood" at 69% of the top score once restricted a concert photo to Indian
music.

## Still open

- `data/photos/` is empty — every number is measured on Wikimedia and Flickr
  photography, not a real camera roll
- aesthetic axis unmeasured; its photos are junk (vintage_bollywood fetched
  skunks)
- no human preference ratings — the honest gold standard
- the HF token in the session history should be rotated
