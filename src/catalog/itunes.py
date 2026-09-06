"""Harvest song metadata + 30-second preview URLs from the iTunes Search API.

WHY iTUNES?
-----------
We need two things that are hard to get together for free:

  1. Songs people actually recognise (Bollywood, Punjabi, English chart music)
  2. Real audio, because our model listens to the audio to understand a song

The iTunes Search API gives us both, needs **no API key and no signup**, and
covers the Indian catalog very well. Each result includes a `previewUrl` -
a 30-second `.m4a` clip. Thirty seconds is plenty: it is usually the chorus,
exactly the part someone would put on a story.

WHAT WE DO WITH THE AUDIO
-------------------------
Nothing permanent. Later (Week 2) we stream each preview, turn it into a list
of numbers (an "embedding"), and throw the audio away. We never store or
redistribute audio files.

THIS MODULE USES ONLY THE PYTHON STANDARD LIBRARY.
No `pip install` needed to run it. That is deliberate - it should work on any
Python 3 install, on any machine, on day one.
"""

import csv
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request

# The API endpoint. Documented by Apple; free and public.
API_URL = "https://itunes.apple.com/search"

# Apple rate-limits to roughly 20 requests per minute. We stay well under it.
# Being polite costs us a few minutes and avoids getting temporarily blocked.
DEFAULT_SLEEP_SECONDS = 3.0

# Apple caps `limit` at 200 results per request.
MAX_LIMIT = 200

# Some servers reject requests that do not identify themselves.
HEADERS = {"User-Agent": "Music-Instagram/0.1 (student research project)"}

# The columns we keep. Everything else the API returns, we drop.
FIELDNAMES = [
    "track_id",
    "track_name",
    "artist_name",
    "album_name",
    "genre",
    "release_year",
    "duration_ms",
    "explicit",
    "preview_url",
    "country",
    "lane",
    "seed_term",
]


def search(term, country="IN", limit=MAX_LIMIT, retries=3):
    """Run one search against the iTunes API and return the raw results.

    Parameters
    ----------
    term : str
        What to search for, e.g. "bollywood hits".
    country : str
        Two-letter storefront code. "IN" for India, "US" for the USA.
    limit : int
        How many results to ask for (Apple caps this at 200).
    retries : int
        How many times to retry on a network or rate-limit error.

    Returns
    -------
    list[dict]
        The raw JSON objects Apple sent back. Empty list if the search failed.
    """
    params = {
        "term": term,
        "country": country,
        "media": "music",
        "entity": "song",
        "limit": min(limit, MAX_LIMIT),
    }
    url = API_URL + "?" + urllib.parse.urlencode(params)

    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=30) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return payload.get("results", [])

        except urllib.error.HTTPError as error:
            # 403 usually means "you are going too fast". Back off and retry.
            wait = DEFAULT_SLEEP_SECONDS * (2 ** attempt)
            print(f"    HTTP {error.code} on '{term}' - waiting {wait:.0f}s and retrying")
            time.sleep(wait)

        except Exception as error:  # network hiccup, timeout, bad JSON
            wait = DEFAULT_SLEEP_SECONDS * (2 ** attempt)
            print(f"    {type(error).__name__} on '{term}' - waiting {wait:.0f}s and retrying")
            time.sleep(wait)

    print(f"    GAVE UP on '{term}' after {retries} attempts")
    return []


def normalise(raw, lane, seed_term):
    """Turn one raw iTunes result into a clean row for our catalog.

    Returns None if the track is unusable (no preview audio available).
    """
    preview_url = raw.get("previewUrl")
    if not preview_url:
        # No audio means we cannot embed it, so the track is useless to us.
        return None

    # releaseDate looks like "2019-05-24T07:00:00Z"; we only want the year.
    release_date = raw.get("releaseDate") or ""
    release_year = release_date[:4] if len(release_date) >= 4 else ""

    return {
        "track_id": raw.get("trackId", ""),
        "track_name": raw.get("trackName", ""),
        "artist_name": raw.get("artistName", ""),
        "album_name": raw.get("collectionName", ""),
        "genre": raw.get("primaryGenreName", ""),
        "release_year": release_year,
        "duration_ms": raw.get("trackTimeMillis", ""),
        "explicit": raw.get("trackExplicitness", ""),
        "preview_url": preview_url,
        "country": raw.get("country", ""),
        "lane": lane,
        "seed_term": seed_term,
    }


def _load_existing(out_path):
    """Read a partially-written catalog so we can resume instead of restarting.

    Returns
    -------
    (set, set)
        Track IDs we already have, and seed terms we have already searched.
    """
    seen_ids, done_seeds = set(), set()
    if not os.path.exists(out_path):
        return seen_ids, done_seeds

    with open(out_path, "r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            seen_ids.add(row.get("track_id", ""))
            done_seeds.add(row.get("seed_term", ""))

    return seen_ids, done_seeds


def harvest(seeds, out_path, limit=MAX_LIMIT, sleep_seconds=DEFAULT_SLEEP_SECONDS):
    """Run every seed search and append the results to a CSV file.

    This function is *resumable*. If you stop it half way (or your wifi dies),
    just run it again - it reads what is already in the CSV and skips those
    searches. This habit matters: in Week 2 we do a job that takes hours, and
    a crash at hour three should not cost you hour one.

    Parameters
    ----------
    seeds : list[tuple[str, str, str]]
        (term, country, lane) triples - see `src/catalog/seeds.py`.
    out_path : str
        Where to write the CSV.
    limit : int
        Results per search.
    sleep_seconds : float
        Pause between requests, to stay under Apple's rate limit.

    Returns
    -------
    int
        Total number of unique tracks in the file after this run.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    seen_ids, done_seeds = _load_existing(out_path)
    if done_seeds:
        print(f"Resuming: {len(seen_ids):,} tracks already saved, "
              f"{len(done_seeds)} searches already done\n")

    file_exists = os.path.exists(out_path)
    handle = open(out_path, "a", encoding="utf-8", newline="")
    writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
    if not file_exists:
        writer.writeheader()

    try:
        for index, (term, country, lane) in enumerate(seeds, start=1):
            if term in done_seeds:
                print(f"[{index}/{len(seeds)}] skip (already done): {term}")
                continue

            print(f"[{index}/{len(seeds)}] searching {country}: {term}")
            results = search(term, country=country, limit=limit)

            added = 0
            for raw in results:
                row = normalise(raw, lane=lane, seed_term=term)
                if row is None:
                    continue
                if str(row["track_id"]) in seen_ids:
                    continue  # we already have this song from another search
                seen_ids.add(str(row["track_id"]))
                writer.writerow(row)
                added += 1

            # Flush after every search so a crash never loses more than one search.
            handle.flush()
            print(f"    {len(results)} results, {added} new "
                  f"(catalog now {len(seen_ids):,} tracks)")

            time.sleep(sleep_seconds)

    except KeyboardInterrupt:
        print("\nStopped by you. Progress is saved - run the script again to resume.")

    finally:
        handle.close()

    return len(seen_ids)
