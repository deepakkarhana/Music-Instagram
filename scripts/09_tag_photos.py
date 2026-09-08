"""STEP 9 - Look at a photo and decide which vibes it shows.

Run this from the project root:

    python scripts/09_tag_photos.py                     tag the dev photos
    python scripts/09_tag_photos.py --dir data/photos   tag your own
    python scripts/09_tag_photos.py --compare           CLIP vs SigLIP

WHAT THIS DOES
--------------
CLIP puts images and text in one shared space, exactly as CLAP does for audio
and text. So the vision step needs no training at all: embed the photo, embed
every vibe's `visual_cues`, and see which ones point the same way.

This is the last missing piece. After this the whole chain exists:

    photo --CLIP--> vibe --CLAP--> songs

MEASURING IT, FOR FREE
----------------------
The development photos came with labels nobody had to write. Each was fetched
by searching Commons for a specific vibe, so a photo found under "Mountains"
is a mountains photo. `manifest.csv` records which search produced which file.

That gives a real accuracy number today:

    for each photo, does the model rank the vibe it was fetched for highly?

The labels are noisy - Commons search is not perfect, and a mountain photo at
sunset is honestly *both* "mountains" and "golden hour". So treat the number
as a signal, not a grade. A model scoring near chance is broken; one scoring
well is probably working. It is far better than no measurement, which is what
you have without the manifest.

Chance level here is 1/40 for top-1 and 3/40 for top-3, so anything in that
region means the model is not reading the photos at all.
"""

import argparse
import csv
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import console
from src.vibe import taxonomy
from src.vibe import vision

DEV_DIR = os.path.join("data", "photos_dev")


def load_photos(folder):
    """Load every readable image in a folder. Returns (paths, images)."""
    if not os.path.isdir(folder):
        return [], []

    paths, images = [], []
    for name in sorted(os.listdir(folder)):
        path = os.path.join(folder, name)
        if not os.path.isfile(path) or name.startswith("."):
            continue
        if os.path.splitext(name)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            continue
        image = vision.load_image(path)
        if image is not None:
            paths.append(path)
            images.append(image)
    return paths, images


def load_labels(folder):
    """Read manifest.csv, mapping filename -> the vibe it was fetched for."""
    manifest = os.path.join(folder, "manifest.csv")
    if not os.path.exists(manifest):
        return {}
    with open(manifest, "r", encoding="utf-8", newline="") as handle:
        return {row["file"]: row["vibe"] for row in csv.DictReader(handle)}


def evaluate(model_name, paths, images, labels, device_arg, batch=16):
    """Tag every photo and, where labels exist, score the result."""
    print(f"\nLoading {model_name} ...")
    model, processor, device = vision.load(model_name, device=device_arg)
    offset = vision.require_working_encoder(model, processor, device)
    print(f"  text encoder healthy (offset {offset:.4f}, want < 0.98), on {device.upper()}")

    vibes = taxonomy.VIBES
    vibe_vectors = vision.embed_text(
        model, processor, [v.visual_cues for v in vibes], device
    )

    image_vectors = []
    for start in range(0, len(images), batch):
        image_vectors.append(
            vision.embed_images(model, processor, images[start : start + batch], device)
        )
        print(f"  embedded {min(start + batch, len(images))}/{len(images)} photos", end="\r")
    print(" " * 40, end="\r")
    image_vectors = np.vstack(image_vectors)

    scores = vision.score_vibes(image_vectors, vibe_vectors)

    top1 = top3 = top5 = counted = 0
    axis_hits = Counter()
    axis_total = Counter()

    for i, path in enumerate(paths):
        gold = labels.get(os.path.basename(path))
        if not gold:
            continue
        order = np.argsort(-scores[i])
        ranked = [vibes[j].name for j in order]
        rank = ranked.index(gold) + 1
        counted += 1
        top1 += rank == 1
        top3 += rank <= 3
        top5 += rank <= 5
        axis = taxonomy.get(gold).axis
        axis_total[axis] += 1
        axis_hits[axis] += rank <= 3

    return {
        "model": model_name,
        "scores": scores,
        "vibes": vibes,
        "counted": counted,
        "top1": top1,
        "top3": top3,
        "top5": top5,
        "axis_hits": axis_hits,
        "axis_total": axis_total,
    }


def report(result):
    n = result["counted"]
    if not n:
        print("\n  no labelled photos, so no accuracy to report")
        return
    print(f"\n  ACCURACY on {n} labelled photos (chance: top-1 2.5%, top-3 7.5%)")
    print(f"    top-1  {result['top1']:>4}/{n}  {100*result['top1']/n:5.1f}%")
    print(f"    top-3  {result['top3']:>4}/{n}  {100*result['top3']/n:5.1f}%")
    print(f"    top-5  {result['top5']:>4}/{n}  {100*result['top5']/n:5.1f}%")
    print("\n  top-3 accuracy by axis")
    for axis in taxonomy.AXES:
        total = result["axis_total"][axis]
        if total:
            hits = result["axis_hits"][axis]
            print(f"    {axis:<10} {hits:>3}/{total:<3} {100*hits/total:5.1f}%")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--dir", default=DEV_DIR, help="folder of photos to tag")
    parser.add_argument("--model", default=vision.MODEL_NAME)
    parser.add_argument("--compare", action="store_true",
                        help="run CLIP and SigLIP and report which is better")
    parser.add_argument("--show", type=int, default=8, help="photos to print")
    parser.add_argument("--per-axis", action="store_true",
                        help="show the best vibe on each axis, not the top 3 overall")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    console.setup()

    paths, images = load_photos(args.dir)
    if not paths:
        print(f"No readable photos in {args.dir}")
        print("Run:  python scripts/08_fetch_dev_photos.py")
        return 1

    labels = load_labels(args.dir)
    print(f"Photos:  {len(paths)} in {args.dir}")
    print(f"Labelled: {sum(1 for p in paths if os.path.basename(p) in labels)}")
    print(f"Vibes:   {len(taxonomy.VIBES)}")

    models = [vision.CLIP_MODEL, vision.SIGLIP_MODEL] if args.compare else [args.model]
    results = []
    for name in models:
        try:
            results.append(evaluate(name, paths, images, labels, args.device))
            report(results[-1])
        except Exception as error:
            print(f"\n  {name} failed: {type(error).__name__}: {str(error)[:150]}")

    if not results:
        return 1

    if len(results) > 1:
        print("\n" + "=" * 66)
        print("COMPARISON")
        print("=" * 66)
        for r in results:
            n = max(1, r["counted"])
            print(f"  {r['model']:<34} top-1 {100*r['top1']/n:5.1f}%  "
                  f"top-3 {100*r['top3']/n:5.1f}%")
        best = max(results, key=lambda r: r["top3"])
        print(f"\n  Winner by top-3: {best['model']}")
        print("  Set this as MODEL_NAME in src/vibe/vision.py")

    # Show what it actually decided, which is the real sanity check.
    best = max(results, key=lambda r: r["top3"])
    print("\n" + "=" * 66)
    print(f"WHAT {best['model']} SEES")
    print("=" * 66)
    step = max(1, len(paths) // args.show)
    sample = list(range(0, len(paths), step))[: args.show]
    for i in sample:
        gold = labels.get(os.path.basename(paths[i]), "?")
        picked = vision.top_vibes(best["scores"][i], best["vibes"], k=3,
                                  per_axis=args.per_axis)
        print(f"\n  {os.path.basename(paths[i])[:56]}")
        print(f"    fetched as: {gold}")
        for vibe, score in picked:
            mark = " <-" if vibe.name == gold else ""
            print(f"    {score:.3f}  [{vibe.axis}] {vibe.label}{mark}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
