"""Fetch freely-licensed photos from Wikimedia Commons.

WHY WIKIMEDIA AND NOT PINTEREST
-------------------------------
Pinterest is a wall of images pinned from everywhere else on the internet.
Almost all of it is somebody's copyrighted work, and being visible there gives
nobody the right to reuse it. Scraping it also breaks Pinterest's terms.

"I found it online" is not a licence, and a project you intend to publish and
put on a CV is exactly the wrong place to be casual about it.

Wikimedia Commons is the opposite: every file carries an explicit licence,
most are Creative Commons or public domain, the API needs no key, and each
result comes with the author and licence attached. So we can record where
every photo came from and prove we were allowed to use it.

WHAT THESE PHOTOS ARE FOR, AND WHAT THEY ARE NOT
------------------------------------------------
They are a **development set**: enough images to build and debug Week 4's
vision pipeline without waiting on anyone.

They are **not** a substitute for real photos, and it matters why. Commons is
full of documentary and landscape photography - carefully composed, well lit,
often shot on real cameras. A phone camera roll is not like that. It is
badly lit, tilted, cluttered, full of faces and selfies and food shot from
above.

A system tuned and measured on Commons photos would look excellent and then
fall over on the first story anyone actually posts. So:

    data/photos_dev/   these - for building and debugging
    data/photos/       the user's own - for evaluation, and the number
                       that gets reported

Never report an accuracy figure computed on the development set.

ATTRIBUTION
-----------
Every download is recorded in a manifest CSV with its title, author, licence
and source URL. CC BY and CC BY-SA both *require* attribution - the manifest
is what makes honouring that possible.
"""

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API_URL = "https://commons.wikimedia.org/w/api.php"

# Wikimedia asks that automated clients identify themselves with a contact.
HEADERS = {
    "User-Agent": "Music-Instagram/0.1 (student research project; "
                  "https://github.com/deepakkarhana/Music-Instagram)"
}

# Be polite: Commons is donated infrastructure.
SLEEP_SECONDS = 1.0

# Width to download. The vision model works at roughly 384px, so anything past
# about 1024 is wasted bytes and slower loading.
THUMB_WIDTH = 1024

# Licences we accept. Everything here allows reuse with attribution.
ALLOWED_LICENCE = re.compile(r"(cc[ -]by|cc0|public domain|pdm)", re.IGNORECASE)


def _strip_html(text):
    """Commons returns author fields as HTML links. We want the name."""
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", "", str(text))
    return " ".join(text.split())[:120]


def search(term, limit=10, timeout=45):
    """Search Commons for photos matching `term`.

    Returns a list of dicts with url, title, author, licence and page.
    Empty list on failure - a failed search is not worth crashing over.
    """
    params = {
        "action": "query",
        "format": "json",
        "generator": "search",
        "gsrsearch": term,
        "gsrnamespace": "6",           # namespace 6 is File:
        "gsrlimit": str(limit),
        "prop": "imageinfo",
        "iiprop": "url|extmetadata|size|mime",
        "iiurlwidth": str(THUMB_WIDTH),
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)

    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as error:
        print(f"    search failed for {term!r}: {type(error).__name__}")
        return []

    results = []
    for page in payload.get("query", {}).get("pages", {}).values():
        info = (page.get("imageinfo") or [{}])[0]
        meta = info.get("extmetadata", {})

        # Photographs only. Commons is also full of diagrams, maps and scans.
        if not str(info.get("mime", "")).startswith("image/"):
            continue
        if str(info.get("mime")) not in ("image/jpeg", "image/png", "image/webp"):
            continue

        licence = _strip_html(meta.get("LicenseShortName", {}).get("value"))
        if not ALLOWED_LICENCE.search(licence):
            continue  # skip anything needing more than attribution

        thumb = info.get("thumburl")
        if not thumb:
            continue

        results.append({
            "title": page.get("title", "").replace("File:", ""),
            "author": _strip_html(meta.get("Artist", {}).get("value")),
            "licence": licence,
            "url": thumb,
            "page": info.get("descriptionurl", ""),
            "width": info.get("thumbwidth", 0),
            "height": info.get("thumbheight", 0),
        })

    return results


OPENVERSE_URL = "https://api.openverse.org/v1/images/"


def search_openverse(term, limit=10, timeout=60):
    """Search Openverse, which aggregates Flickr and other CC photo libraries.

    WHY A SECOND SOURCE
    -------------------
    Wikimedia Commons is an encyclopaedia's picture library: documentary and
    reference photographs, carefully catalogued. Excellent for "what does an
    Indian wedding look like", useless for "what does melancholy look like",
    because nobody uploads a moody photograph to illustrate an article.

    Openverse reaches Flickr, which is where people post photographs taken for
    their own sake. The difference is immediate - searching "rain drops on a
    window" returns "Window Pain" and "Here comes the rain again..", which are
    photographs with a mood rather than records of an object.

    Same licensing discipline: commercial use and modification only, with the
    creator and licence recorded for every download.
    """
    params = {
        "q": term,
        "page_size": min(limit, 20),
        "license_type": "commercial,modification",
    }
    url = OPENVERSE_URL + "?" + urllib.parse.urlencode(params)

    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.load(response)
    except Exception as error:
        print(f"    openverse failed for {term!r}: {type(error).__name__}")
        return []

    results = []
    for item in payload.get("results", []):
        link = item.get("url")
        if not link:
            continue
        # Skip anything tiny - it will be upscaled into invented detail later.
        if (item.get("width") or 0) < 600:
            continue

        licence = str(item.get("license", "")).upper()
        version = item.get("license_version") or ""
        results.append({
            "title": (item.get("title") or "untitled")[:90],
            "author": _strip_html(item.get("creator")),
            "licence": f"CC {licence} {version}".strip(),
            "url": link,
            "page": item.get("foreign_landing_url", ""),
            "width": item.get("width", 0),
            "height": item.get("height", 0),
        })
    return results


def search_best(term, limit=10, prefer="commons"):
    """Search one source, falling back to the other when it finds nothing.

    Commons is better for concrete subjects and Openverse for atmospheric
    ones, so the caller picks which to try first and this handles the miss.
    """
    first = search if prefer == "commons" else search_openverse
    second = search_openverse if prefer == "commons" else search

    found = first(term, limit=limit)
    if found:
        return found, prefer
    polite_pause()
    return second(term, limit=limit), ("openverse" if prefer == "commons" else "commons")


def download(url, path, timeout=60):
    """Save one image. Returns True on success."""
    try:
        request = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            data = response.read()
    except Exception:
        return False

    if len(data) < 5_000:  # a few KB means an error page, not a photo
        return False

    with open(path, "wb") as handle:
        handle.write(data)
    return True


def safe_name(vibe_name, index, title):
    """Build a filename that says which vibe a photo was fetched for."""
    stem = re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(title)[0])[:40].strip("_")
    return f"{vibe_name}__{index:02d}__{stem or 'photo'}.jpg"


def polite_pause():
    time.sleep(SLEEP_SECONDS)
