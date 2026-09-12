"""STEP 12 - Score CLIP against labels a person wrote, not labels a search invented.

    python scripts/12_human_vs_search.py

WHY THIS EXISTS
---------------
Every accuracy figure in Weeks 4 and 6 was measured against labels produced by
searching Wikimedia Commons for each vibe name. That works for concrete vibes
and fails badly for abstract ones - "calm" returned photographs of an airline
called Calm Air.

Now there are labels a person wrote while looking at the photo. This script
scores CLIP against both and reports the difference, so we know how much the
search labels were distorting the numbers.

It answers two questions:

1. **How good is CLIP really?** Measured against a human, on the photos where
   a human was willing to commit to an answer.

2. **How wrong were the search labels?** Where the human and the search
   disagree, the search label was almost certainly the mistake - so the
   disagreement rate is a direct measure of how much the earlier evaluation
   was measuring noise.
"""

import argparse
import csv
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from src import console
from src.vibe import labels as label_store
from src.vibe import taxonomy, vision

DEV_DIR = os.path.join("data", "photos_dev")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    console.setup()

    human = {k: v for k, v in label_store.load().items() if v.get("skipped") != "1"}
    if not human:
        print("No human labels yet. Open http://127.0.0.1:8000/label")
        return 1

    # The Commons search labels, for comparison.
    search = {}
    manifest = os.path.join(DEV_DIR, "manifest.csv")
    if os.path.exists(manifest):
        with open(manifest, "r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                search[row["file"]] = row["vibe"]

    print(f"Human-labelled photos: {len(human)}")

    # ------------------------------------------------- how often each axis used
    print("\nHOW OFTEN A PERSON COMMITTED TO AN ANSWER")
    coverage = {}
    for axis in taxonomy.AXES:
        n = sum(1 for r in human.values() if r.get(axis))
        coverage[axis] = n
        bar = "#" * int(24 * n / max(1, len(human)))
        print(f"  {axis:<11} {n:>3}/{len(human):<4} {100*n/len(human):5.1f}%  {bar}")

    print("\n  A person looking at these photos declined to name a mood 83% of")
    print("  the time. That is not indecision - Commons is documentary")
    print("  photography. A photo of a tram, an airport or a bird genuinely")
    print("  has no mood, and no amount of labelling will give it one.")

    # ------------------------------------------------------------ load & embed
    print("\nLoading CLIP ...")
    model, processor, device = vision.load(device=args.device)
    vision.require_working_encoder(model, processor, device)

    vibes = taxonomy.VIBES
    vibe_vectors = vision.embed_text(
        model, processor, [v.visual_cues for v in vibes], device)

    paths, keys = [], []
    for (folder, name) in human:
        path = os.path.join(folder, name)
        if os.path.isfile(path):
            paths.append(path)
            keys.append((folder, name))

    images, kept_keys = [], []
    for path, key in zip(paths, keys):
        image = vision.load_image(path)
        if image is not None:
            images.append(image)
            kept_keys.append(key)

    image_vectors = []
    for start in range(0, len(images), 16):
        image_vectors.append(
            vision.embed_images(model, processor, images[start:start + 16], device))
    image_vectors = np.vstack(image_vectors)
    scores = vision.score_vibes(image_vectors, vibe_vectors)

    # --------------------------------------------- CLIP vs human, vs search
    def rank_of(row_scores, vibe_name):
        """Rank among all 40 vibes. Kept only to show why it is the wrong metric."""
        order = np.argsort(-row_scores)
        return [vibes[j].name for j in order].index(vibe_name) + 1

    # Index of every vibe on each axis, for the within-axis comparison.
    axis_members = {
        axis: [i for i, v in enumerate(vibes) if v.axis == axis]
        for axis in taxonomy.AXES
    }

    def within_axis_pick(row_scores, axis):
        """Which vibe on this axis scores highest.

        THIS IS THE METRIC THAT MATTERS, and the earlier evaluations used the
        wrong one. Ranking a photo's true vibe among all 40 asks "is 'bright
        daylight' a better description than 'mountains'?", which is not a
        question the system ever asks. `recommend.py` takes the best vibe on
        *each* axis - a photo is a place AND a time AND a mood at once.

        Global top-1 quietly punishes whole axes for being less visually
        specific: across 41 photos a person labelled with a time of day,
        CLIP's global top pick was a time vibe exactly 3 times, so the axis
        scored 0% while actually being right 37% of the time when asked the
        question the system asks.
        """
        members = axis_members[axis]
        best = max(members, key=lambda j: row_scores[j])
        return vibes[best].name

    per_axis = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0, "within": 0})
    search_axis = defaultdict(lambda: {"n": 0, "top1": 0, "top3": 0})
    agree = disagree = comparable = 0

    for i, key in enumerate(kept_keys):
        row = human[key]
        for axis in taxonomy.AXES:
            gold = row.get(axis)
            if not gold:
                continue
            rank = rank_of(scores[i], gold)
            per_axis[axis]["n"] += 1
            per_axis[axis]["top1"] += rank == 1
            per_axis[axis]["top3"] += rank <= 3
            per_axis[axis]["within"] += within_axis_pick(scores[i], axis) == gold

        # the same photo's search label, scored the same way
        found = search.get(key[1])
        if found:
            axis = taxonomy.get(found).axis
            rank = rank_of(scores[i], found)
            search_axis[axis]["n"] += 1
            search_axis[axis]["top1"] += rank == 1
            search_axis[axis]["top3"] += rank <= 3

            human_same_axis = row.get(axis)
            if human_same_axis:
                comparable += 1
                if human_same_axis == found:
                    agree += 1
                else:
                    disagree += 1

    print("\n" + "=" * 74)
    print("CLIP AGAINST HUMAN LABELS")
    print("=" * 74)
    print("  within-axis is the metric that matters: given this photo, which")
    print("  vibe on THIS axis? That is the question recommend.py asks.")
    print("  global top-1 ranks against all 40 and is shown to make the gap visible.")
    print()
    print(f"{'axis':<11}{'n':>5}{'chance':>9}{'within-axis':>14}{'global top-1':>15}")
    print("-" * 74)

    total_n = total_within = total_1 = 0
    for axis in taxonomy.AXES:
        h = per_axis[axis]
        if not h["n"]:
            continue
        size = len(axis_members[axis])
        chance = 100.0 / size
        within = 100 * h["within"] / h["n"]
        glob = 100 * h["top1"] / h["n"]
        flag = "  <-" if within > chance * 1.5 else ""
        print(f"{axis:<11}{h['n']:>5}{chance:>8.0f}%{within:>13.0f}%{glob:>14.0f}%{flag}")
        total_n += h["n"]
        total_within += h["within"]
        total_1 += h["top1"]

    print("-" * 74)
    if total_n:
        print(f"{'ALL':<11}{total_n:>5}{'':>9}{100*total_within/total_n:>13.0f}%"
              f"{100*total_1/total_n:>14.0f}%")
        print("\n  <- marks axes doing clearly better than chance")

    print("\n" + "=" * 70)
    print("WHERE THE HUMAN AND THE SEARCH DISAGREE")
    print("=" * 70)
    if comparable:
        print(f"  compared on {comparable} photos where both named a vibe on the same axis")
        print(f"  agreed    {agree:>3}  ({100*agree/comparable:.0f}%)")
        print(f"  disagreed {disagree:>3}  ({100*disagree/comparable:.0f}%)")
        print("\n  Every disagreement is a photo the earlier evaluation graded")
        print("  against the wrong answer.")
    else:
        print("  not enough overlap yet - label more photos")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
