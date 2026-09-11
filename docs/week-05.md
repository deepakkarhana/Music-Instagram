# Week 5 — The demo

**Goal:** drop a photo into a browser and hear the songs it picked.

```bash
python app/main.py
```

Then open <http://127.0.0.1:8000>. First start takes about half a minute
while the two models load.

---

## Why this week matters more than it looks

Everything before this was scripts. Scripts prove the idea works; they do not
show anyone what it *feels* like. Dropping your own photo into a page and
hearing what came back is a different kind of evidence, and it is the thing
worth putting in front of someone in an interview.

It is also the first time the project has to behave like software rather than
an experiment.

---

## What changed to make it a server

**Load once, not per request.** CLIP and CLAP take roughly half a minute and
about 2 GB between them. Loading per request would make every upload
unusable, so they load at startup along with the index and both sets of
vocabulary vectors. After that a request is: embed one photo, score it
against 40 vibes, average a few vectors, search 11,870 songs.

Measured: **1,241 ms** on the first request, **251 ms** after. The slow part
is reading the photo off the wire, not the models.

**Fail politely.** Every wrong input gets a message that says what to do:

```
not an image   400  "Could not read that as an image. JPEG, PNG or WebP
                     please - iPhone HEIC files need converting first."
empty file     400  "That file was empty."
17 MB photo    413  "That photo is 17 MB. Please keep it under 15 MB."
```

**One code path.** The browser and the command line both call
`vision.load_image`, so they cannot drift apart — including the EXIF rotation
fix, which matters far more here because phone uploads are the whole point.

---

## The previews actually play

iTunes preview URLs go straight into an HTML `<audio>` element, so the browser
streams the 30 seconds directly from Apple. **No audio is stored or served by
us** — the same rule the whole project follows, and the reason it is
publishable.

---

## The gym photo, finally

Week 2's worst failure was this:

```
"high energy gym workout, aggressive hip hop with heavy bass"
    →  Rainy Day,  Bossa Nova,  Beach Vibes
```

A photo of a gym, through the finished pipeline:

```
  CLIP sees:  Working out 0.217,  Gym 0.214
  Songs:      0.609  Birthday              The Beatles
              0.604  Rock Anthems          Panos Koutselinis
              0.600  Enter Sandman         Metallica
```

Nothing about the retrieval changed. What changed is that the gym vibe's
`music_query` is written in the language of **sound** — *"hard hitting drums,
distorted bass, shouted male rap vocals"* — instead of the language of rooms.
The Week 3 rule, doing its job.

An Indian wedding photo, same run:

```
  CLIP sees:  Temple or shrine 0.292,  Indian wedding 0.268
  Songs:      0.743  Kabhi Kabhi Mere      Lata Mangeshkar & Mukesh
              0.710  Shanti-Mantra         Ravi Shankar
```

---

## What the page shows, and why

Not just songs. It shows **the vibes it picked and the query it built**,
because when a recommendation is wrong it is almost always wrong at the vibe
step, not the music step. A beach photo tagged "melancholy" will faithfully
return sad music, and the retrieval was never the problem.

Showing the intermediate step turns "the AI got it wrong" into "it thought
this was melancholy, and here is why that is understandable" — which is the
difference between a demo and a diagnosis.

---

## What "done" looks like

- [x] a page you can drop a photo into
- [x] models loaded once, sub-second responses
- [x] previews that play
- [x] every error path returns something a person can act on
- [x] the vibe step visible, not hidden
- [ ] deployed somewhere public (Week 12)

---

**Next:** Week 6 measures all of this properly — against baselines, on real
phone photos, with a human saying whether the songs actually fit.
