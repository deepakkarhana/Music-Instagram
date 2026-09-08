# Week 4 — Teach it to look, and connect both halves

**Goal:** a model reads a photo, picks vibes from the vocabulary, and those
vibes retrieve music. The whole chain, working.

```
photo --CLIP--> vibe --CLAP--> songs
```

---

## The trick is the same one, twice

Week 2 used CLAP because it puts **audio and text** in one shared space, so a
sentence could retrieve music.

CLIP does exactly the same for **images and text**. Embed a photo, embed the
sentence *"sand, sea horizon, bright sky, waves"*, and if the photo is a beach
the two vectors point the same way.

So the vision step needs **no training at all**. Which means the interesting
part is not the models — it is what joins them:

> CLIP and CLAP have never met. Different authors, different data, different
> years, and neither knows the other exists. The 40-name vocabulary is the
> entire bridge. A vibe is a name with two descriptions attached: one CLIP can
> read, one CLAP can.

Each description goes to the model that can actually use it:

| field | goes to | why |
|---|---|---|
| `visual_cues` | CLIP | it was trained on captions describing what is *visible* |
| `music_query` | CLAP | it was trained on captions describing what is *audible* |

Feeding CLIP the vibe *name* would repeat Week 2's gym mistake in the other
direction. "occasion: graduation" is a database label, not something a camera
ever saw.

---

## Measuring it, using labels nobody wrote

The 118 development photos came pre-labelled by accident. Each was fetched by
searching Wikimedia Commons for one specific vibe, and `manifest.csv` records
which search produced which file — so a photo found under "Mountains" is a
mountains photo.

That gives a real accuracy number with no annotation work:

> For each photo, does the model rank the vibe it was fetched for highly?

Chance is **2.5% top-1** and **7.5% top-3**, because there are 40 vibes.

### CLIP vs SigLIP, chosen by measurement

```
openai/clip-vit-base-patch32      top-1  28.0%   top-3  42.4%   <- chosen
google/siglip-base-patch16-224    top-1   8.7%   top-3  18.3%
random guessing                   top-1   2.5%   top-3   7.5%
```

CLIP is **11x chance at top-1** and **5.6x at top-3**.

SigLIP losing is surprising — it usually beats CLIP at zero-shot
classification — so it was worth checking whether the fault was in *our
usage*. SigLIP is known to need `padding="max_length"` where CLIP does not,
and using the wrong one silently degrades it. Tested both:

```
padding=True         shared-offset 0.7554
padding=max_length   shared-offset 0.7699
```

Effectively identical, and both healthy. So SigLIP was handled fairly and
genuinely does worse on this task. Worth the ten minutes: choosing a model
because of a bug in how you called the other one is an easy mistake and an
embarrassing one.

---

## Where it works and where it does not

This is the most useful result of the week.

| axis | top-3 | what it is |
|---|---|---|
| **occasion** | **63.0%** | what is happening — wedding, graduation, party |
| **scene** | **55.6%** | where it is — beach, mountains, cafe |
| aesthetic | 28.6% | how it is styled — old money, Y2K, streetwear |
| mood | 22.7% | what it feels like — nostalgic, euphoric |
| **time** | **16.7%** | light and time of day — golden hour, night |

There is a clear pattern. CLIP is good at **concrete, nameable things** and
weak at **abstract, perceptual ones**. That follows directly from its training
data: image captions on the internet say *"a wedding in Jaipur"*, and almost
never say *"shot in flat overcast light, feels wistful"*.

**This matters more than the headline number**, because mood is arguably the
axis that drives music choice hardest. Whether a photo is melancholy or
euphoric changes the song far more than whether it is a beach or a mountain.

So the weakest link in the pipeline is precisely the link that matters most.
That is what Week 7's trained classifier is for — and this measurement is the
justification for training one, rather than a plan we assumed at the start.

---

## Combining several vibes into one search

A photo is rarely one vibe, so we take the best on each axis. Merging them
into a single CLAP query can be done two ways, and they are not equivalent:

**Join the text** — paste the music queries together and embed once. Simple,
but CLAP has a token limit and long queries drift toward describing nothing in
particular.

**Average the vectors** (default) — embed each vibe separately, then take a
weighted mean. The average of *"airy acoustic guitar"* and *"warm brass,
mellow"* lands between them, which is what a photo showing both should get.

Weights come from CLIP's confidence, but as *differences* rather than raw
scores. Raw cosine values sit in a narrow band well above zero, so using them
directly would make a strong signal and a weak guess almost equally important.

`--combine join` switches, because which is better is a question to measure
later, not to settle by argument now.

---

## One more thing worth having: the per-artist cap

Without it, one prolific stock-music label fills the entire list. Every match
is individually correct and the recommendation is still useless. `--per-artist`
defaults to 1.

---

## It works

```bash
python scripts/10_recommend.py photo.jpg
python scripts/10_recommend.py --sample 5
```

A photo of wedding flower decorations:

```
  CLIP sees:
    0.230  [aesthetic] Vintage Bollywood
    0.228  [mood] Nostalgic
    0.213  [occasion] Birthday

  Songs:
    0.698  Ziddi Dil                    Vishal Dadlani
    0.697  In the Backseat              Arcade Fire
    0.677  Hum Tum (From "Hum Tum")     Alka Yagnik & Babul Supriyo
    0.677  Kyon Ki Itna Pyar            Udit Narayan & Alka Yagnik
```

Marigolds to Bollywood, through two models that have never met.

### And a failure worth keeping

```
  old_money__02__662_feet_in_old_money_geograph_org_uk.jpg
  CLIP sees:  Green nature (46%), Melancholy (29%)
```

CLIP was right and the *label* was wrong. That file is a landscape photograph
whose title puns on "old money" meaning imperial units. The search that
fetched it was fooled by the words; the model looking at the pixels was not.

A reminder that the 42.4% is measured against noisy labels and is therefore a
floor, not a ceiling.

---

## What "done" looks like

- [x] a model reads a photo and picks vibes
- [x] measured against a real baseline, not vibes-in-the-loose-sense
- [x] model chosen by measurement, with the losing model verified as fairly used
- [x] full chain runs: photo → vibes → music query → songs
- [ ] the same numbers on *real* phone photos, not Commons photography

---

**Next:** the demo people can actually use, and then Week 6 measures all of
this properly — against baselines, on real photos, with a human saying whether
the songs fit.
