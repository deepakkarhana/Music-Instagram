# data/

This folder is **not tracked by git** (see `.gitignore`).

Everything here is either large (embeddings, FAISS indexes) or not ours to
redistribute (preview audio). The rule for this project:

    fetch preview audio -> compute embedding -> discard the audio

We ship **embeddings and metadata**, never audio files. That keeps the repo
publishable and keeps us inside the terms of the APIs we use.

## Layout

    data/
      raw/        # exactly what the APIs gave us, unmodified
      interim/    # cleaned, deduplicated, merged catalogs
      embeddings/ # .npy arrays (created in Week 2, mostly in Colab)
      index/      # FAISS index files (Week 2)

If you clone this repo fresh, these folders will be empty. Re-create them by
running the scripts in `scripts/` in numerical order.
