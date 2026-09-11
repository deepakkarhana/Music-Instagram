"""Storing labels a person actually wrote while looking at the photo.

WHY THIS FILE HAD TO EXIST
--------------------------
The development photos were labelled by searching Wikimedia Commons for each
vibe name. For concrete vibes that works - search "beach", get a beach. For
abstract ones it fails completely, because Commons matches words in titles:

    "calm"        -> Calm Air, an airline. Photographs of a Saab.
    "melancholy"  -> Durer's Melencolia I engraving
    "nostalgic"   -> the Istanbul Nostalgic Tram
    "old money"   -> literal banknotes
    "minimal"     -> minimal surfaces, from mathematics

43 of 118 photos carry a label like that, and every accuracy figure computed
over them measured word collisions rather than vision.

So mood and aesthetic need labels from a person. There is no clever way
around it - that is what the annotation page is for.

ONE DESIGN DECISION WORTH DEFENDING
-----------------------------------
The page does **not** pre-select what CLIP guessed.

Pre-filling the model's answer would make labelling perhaps three times
faster, and people overwhelmingly accept a pre-filled answer rather than
re-examining it. The labels would drift toward agreeing with CLIP, and then
CLIP would be evaluated against labels it had partly written.

That failure is invisible: accuracy goes up, the labels look reasonable, and
the number means nothing. Slower and honest beats faster and circular.

THE FORMAT
----------
One row per photo, one column per axis, blank where the axis does not apply.
A photo can legitimately be a beach *and* golden hour *and* calm - which is
exactly why the axes exist.
"""

import csv
import os
from datetime import datetime, timezone

LABELS_PATH = os.path.join("data", "labels.csv")

FIELDNAMES = ["folder", "file", "scene", "time", "mood", "occasion",
              "aesthetic", "skipped", "labelled_at"]


def _key(folder, name):
    """A folder identifier that does not depend on the slash direction.

    `os.path.join` produces backslashes on Windows, but anything typed by hand
    or arriving from a URL uses forward slashes. Keyed on the raw string, the
    same photo saved both ways becomes two separate rows - so a correction
    silently becomes a duplicate instead of an overwrite, and the file quietly
    accumulates contradictory labels for one image.
    """
    return (str(folder).replace("\\", "/").rstrip("/"), str(name))


def load(path=LABELS_PATH):
    """Every label written so far, keyed by (folder, file).

    Later rows win, so re-labelling a photo corrects it rather than adding a
    second opinion.
    """
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8", newline="") as handle:
        return {_key(r["folder"], r["file"]): r for r in csv.DictReader(handle)}


def save(row, path=LABELS_PATH):
    """Append one labelled photo. Re-labelling appends a newer row, and
    `load` keeps the last one, so corrections simply overwrite."""
    folder = os.path.dirname(path)
    if folder:
        os.makedirs(folder, exist_ok=True)

    exists = os.path.exists(path)
    with open(path, "a", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        if not exists:
            writer.writeheader()
        row = dict(row)
        row["labelled_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        writer.writerow({k: row.get(k, "") for k in FIELDNAMES})


def progress(folders, path=LABELS_PATH):
    """How many photos are labelled, skipped and left, across some folders."""
    done = load(path)
    total = labelled = skipped = 0

    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            total += 1
            row = done.get(_key(folder, name))
            if not row:
                continue
            if row.get("skipped") == "1":
                skipped += 1
            else:
                labelled += 1

    return {"total": total, "labelled": labelled, "skipped": skipped,
            "left": total - labelled - skipped}


def next_unlabelled(folders, path=LABELS_PATH):
    """The next photo nobody has looked at yet, as (folder, filename)."""
    done = load(path)
    for folder in folders:
        if not os.path.isdir(folder):
            continue
        for name in sorted(os.listdir(folder)):
            if os.path.splitext(name)[1].lower() not in (".jpg", ".jpeg", ".png", ".webp"):
                continue
            if _key(folder, name) not in done:
                return folder, name
    return None, None


def counts_by_vibe(path=LABELS_PATH):
    """How many photos carry each vibe - for spotting thin classes early.

    A classifier cannot learn a vibe from two examples, and knowing which
    vibes are starved is worth more once fifty photos are done than once all
    of them are.
    """
    from collections import Counter

    counts = Counter()
    for row in load(path).values():
        if row.get("skipped") == "1":
            continue
        for axis in ("scene", "time", "mood", "occasion", "aesthetic"):
            value = row.get(axis)
            if value:
                counts[value] += 1
    return counts
