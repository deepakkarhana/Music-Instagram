"""Read a photo and decide which vibes it shows.

THE SAME TRICK AS CLAP, ON THE OTHER SIDE
-----------------------------------------
Week 2 used CLAP, which puts *audio* and *text* in one shared space, so a
sentence could retrieve music.

CLIP does exactly the same thing for *images* and *text*. Embed a photo, embed
the sentence "sand, sea horizon, bright sky, waves", and if the photo is a
beach the two vectors point the same way.

That is the whole vision step. No training, no labels:

    photo -> CLIP image encoder -> 512 numbers
    every vibe's visual_cues -> CLIP text encoder -> 512 numbers each
    whichever vibes score highest are what the photo shows

And this is where the two halves of the project meet. The vibe we pick here
carries a `music_query` written in the language of sound, which goes to CLAP,
which returns songs. The vibe name is the bridge:

    photo --CLIP--> vibe name --CLAP--> songs

Neither model knows the other exists. The vocabulary in `taxonomy.py` is the
only thing joining them.

WHY visual_cues AND NOT THE VIBE NAME
-------------------------------------
CLIP was trained on image captions, so it understands descriptions of what is
*visible*: "sand, sea horizon, bright sky". It has no idea what "occasion:
graduation" means as a label. Feeding it the vibe name would be the same
mistake Week 2 made when it asked CLAP about a gym.

Each field goes to the model that can read it:

    visual_cues  -> CLIP  (appearance)
    music_query  -> CLAP  (sound)

CHECK THE MODEL BEFORE TRUSTING IT
----------------------------------
Week 2 lost an afternoon to a published checkpoint whose text encoder was
silently dead - it returned nearly the same vector for every sentence, so
every query returned the same results and it looked like a mediocre model
rather than a broken one.

`self_test()` below is the same one-line check that would have caught it:
embed a few unrelated sentences and measure how much they differ. Run it
before any long job.
"""

import numpy as np

# Two candidates. Which one is better is decided by measurement in
# `scripts/09_tag_photos.py --compare`, not by reputation.
CLIP_MODEL = "openai/clip-vit-base-patch32"
SIGLIP_MODEL = "google/siglip-base-patch16-224"

MODEL_NAME = CLIP_MODEL


def pick_device(preferred=None):
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_name=MODEL_NAME, device=None):
    """Load a vision-language model and its processor.

    Works for both CLIP and SigLIP - `AutoModel` picks the right class, and
    both expose `get_image_features` and `get_text_features`.
    """
    import torch
    from transformers import AutoModel, AutoProcessor

    device = pick_device(device)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    return model, processor, device


def _as_tensor(features):
    """Pull the plain tensor out of whatever the model returned.

    transformers 4.x returned a bare tensor; 5.x returns an output object with
    the vector in `.pooler_output`. Supporting both means this runs on a
    laptop and on Colab without pinning a version.
    """
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def _normalise(vectors):
    """Rescale rows to length 1, so a dot product is cosine similarity."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    lengths[lengths == 0] = 1.0
    return (vectors / lengths).astype(np.float32)


def embed_text(model, processor, texts, device):
    """Turn descriptions into unit vectors in the shared image-text space."""
    import torch

    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    inputs = processor(
        text=list(texts), return_tensors="pt", padding=True, truncation=True
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = _as_tensor(model.get_text_features(**inputs))
    return _normalise(features.float().cpu().numpy())


def embed_images(model, processor, images, device):
    """Turn PIL images into unit vectors in the same space."""
    import torch

    if not images:
        return np.zeros((0, 1), dtype=np.float32)

    inputs = processor(images=list(images), return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        features = _as_tensor(model.get_image_features(**inputs))
    return _normalise(features.float().cpu().numpy())


def load_image(path, max_side=768):
    """Open a photo, fix its rotation, and shrink it if it is huge.

    Phones record orientation in EXIF rather than rotating the pixels, so a
    photo taken in portrait often loads sideways. A sideways photo is a
    different photo as far as the model is concerned.
    """
    from PIL import Image, ImageOps

    try:
        image = Image.open(path)
        image = ImageOps.exif_transpose(image)  # apply the rotation flag
        image = image.convert("RGB")
    except Exception:
        return None

    if max(image.size) > max_side:
        scale = max_side / max(image.size)
        new_size = (int(image.width * scale), int(image.height * scale))
        image = image.resize(new_size, Image.LANCZOS)

    return image


_SELF_TEST_PROMPTS = [
    "a snowy mountain range under a clear sky",
    "a plate of food on a restaurant table",
    "a crowded concert with coloured stage lights",
]


def self_test(model, processor, device, threshold=0.98):
    """Check the text encoder actually distinguishes different sentences.

    Unit vectors pointing in genuinely different directions partly cancel when
    averaged, so a healthy encoder gives an average well below length 1. A
    dead one gives ~0.999, because every vector is the same vector.

    Returns (ok, offset). Closer to 1.0 is worse.
    """
    vectors = embed_text(model, processor, _SELF_TEST_PROMPTS, device)
    offset = float(np.linalg.norm(vectors.mean(axis=0)))
    return offset < threshold, offset


def require_working_encoder(model, processor, device):
    """Refuse to proceed on a model that cannot encode text. Raises RuntimeError."""
    ok, offset = self_test(model, processor, device)
    if not ok:
        raise RuntimeError(
            "This vision model's text encoder looks broken. Three unrelated "
            f"sentences produced nearly the same vector (shared-offset {offset:.4f}; "
            "anything above 0.98 is bad). Every photo would get the same vibes. "
            "Try a different checkpoint via --model."
        )
    return offset


def score_vibes(image_vectors, vibe_vectors):
    """Similarity between every photo and every vibe.

    Returns an (n_photos, n_vibes) array of cosine similarities.
    """
    return np.asarray(image_vectors) @ np.asarray(vibe_vectors).T


def top_vibes(scores_row, vibes, k=3, per_axis=False):
    """Turn one photo's score row into a ranked list of (vibe, score).

    With `per_axis`, returns the best vibe on each axis instead of the best
    overall - a photo is usually a scene *and* a mood *and* an occasion, so
    the top 3 overall can easily be three near-identical scenes.
    """
    order = np.argsort(-scores_row)

    if not per_axis:
        return [(vibes[i], float(scores_row[i])) for i in order[:k]]

    best = {}
    for i in order:
        axis = vibes[i].axis
        if axis not in best:
            best[axis] = (vibes[i], float(scores_row[i]))
    return sorted(best.values(), key=lambda pair: -pair[1])
