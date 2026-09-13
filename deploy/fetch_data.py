"""Download the catalogue and embeddings into a freshly built container.

Runs inside the Dockerfile, not on your laptop. `deploy/prepare.py` puts the
files on Hugging Face; this fetches them back.

WHY AT BUILD TIME RATHER THAN ON FIRST REQUEST
----------------------------------------------
A Space on the free tier sleeps when idle and wakes on the next visit. If the
28 MB download happened on first request, the first visitor after every nap
would wait for it on top of the 30 seconds the models already take.

Doing it in the image means the container starts with everything present. The
build is slower once; every visit afterwards is faster.

IF IT FAILS
-----------
Loudly, with the reason. A Space that starts up "successfully" and then answers
every request with an empty catalogue is far harder to diagnose than one that
refuses to build.
"""

import os
import sys

DATASET_REPO = os.environ.get("DATA_REPO", "Deepakkarhana01/music-instagram-data")

FILES = [
    ("itunes_catalog.csv", os.path.join("data", "raw", "itunes_catalog.csv")),
    ("clap_audio.f32", os.path.join("data", "interim", "clap_audio.f32")),
    ("clap_audio_ids.csv", os.path.join("data", "interim", "clap_audio_ids.csv")),
]


def main():
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN is not set in this Space.")
        print("Settings -> Variables and secrets -> New secret -> HF_TOKEN")
        print("The dataset is private, so the Space cannot read it without one.")
        return 1

    from huggingface_hub import hf_hub_download

    for name, target in FILES:
        os.makedirs(os.path.dirname(target), exist_ok=True)
        print(f"fetching {name} ...")
        try:
            path = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=name,
                repo_type="dataset",
                token=token,
            )
        except Exception as error:
            print(f"FAILED on {name}: {type(error).__name__}: {error}")
            print("\nCheck that deploy/prepare.py ran, and that HF_TOKEN can")
            print(f"read {DATASET_REPO}.")
            return 1

        # Copy rather than symlink: the cache directory is not guaranteed to
        # survive into the running container, and a dangling symlink fails at
        # request time rather than build time.
        import shutil

        shutil.copyfile(path, target)
        print(f"  -> {target}  ({os.path.getsize(target)/1e6:.1f} MB)")

    print("\nData ready.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
