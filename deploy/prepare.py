"""Upload the catalogue and embeddings so a deployed Space can read them.

    python deploy/prepare.py

WHY THE DATA IS NOT IN GIT
--------------------------
`data/` is gitignored on purpose. The repository ships code; it does not ship
28 MB of embeddings, and it certainly does not ship audio. That decision was
made in Week 1 and is the reason this project is publishable at all.

But a running Space needs both files to answer a single request. Hugging Face
keeps data in **dataset repositories**, separate from code, which is exactly
the split we already have. This uploads them there.

WHAT GETS UPLOADED, AND WHAT DOES NOT
-------------------------------------
    itunes_catalog.csv     4 MB   song names, artists, preview URLs
    clap_audio.f32        28 MB   the embeddings
    clap_audio_ids.csv            which row belongs to which song

The FAISS index is **not** uploaded. It rebuilds from those files in about two
seconds, so shipping it would be 23 MB to save two seconds.

No audio, as always. The repo, the dataset and the Space together contain zero
seconds of music - previews stream from Apple directly into the listener's
browser.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATASET_REPO = "Deepakkarhana01/music-instagram-data"

FILES = [
    (os.path.join("data", "raw", "itunes_catalog.csv"), "itunes_catalog.csv"),
    (os.path.join("data", "interim", "clap_audio.f32"), "clap_audio.f32"),
    (os.path.join("data", "interim", "clap_audio_ids.csv"), "clap_audio_ids.csv"),
]


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set.")
        print("Set it with:  setx HF_TOKEN hf_your_token_here")
        print("then open a NEW terminal - setx only affects new ones.")
        return 1

    missing = [path for path, _ in FILES if not os.path.exists(path)]
    if missing:
        print("These files are missing:")
        for path in missing:
            print(f"  {path}")
        print("\nRun the pipeline first - see the README.")
        return 1

    from huggingface_hub import HfApi

    api = HfApi(token=token)

    print(f"Creating dataset repo {DATASET_REPO} (private) ...")
    api.create_repo(DATASET_REPO, repo_type="dataset", private=True, exist_ok=True)

    total = 0
    for path, name in FILES:
        size = os.path.getsize(path) / 1e6
        total += size
        print(f"  uploading {name:<24} {size:>6.1f} MB")
        api.upload_file(
            path_or_fileobj=path,
            path_in_repo=name,
            repo_id=DATASET_REPO,
            repo_type="dataset",
        )

    print(f"\nUploaded {total:.0f} MB to {DATASET_REPO}")
    print("\nThe Space reads these at build time. Next steps are in")
    print("deploy/README.md - create the Space, push the code, add HF_TOKEN")
    print("as a secret so the Space can read this private dataset.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
