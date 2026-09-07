"""CLAP - the model that puts music and words in the same space.

THE ONE IDEA THAT MAKES THIS PROJECT WORK
-----------------------------------------
CLAP (Contrastive Language-Audio Pretraining) has two halves:

    an **audio encoder** - listens to a song, outputs 512 numbers
    a **text encoder**   - reads a sentence, outputs 512 numbers

The trick is that both halves were trained *together*, on hundreds of
thousands of (audio clip, caption) pairs, with one instruction: make the two
vectors point the same direction when the caption describes the audio, and
different directions when it does not.

The consequence is the whole project:

    embed_text("warm nostalgic acoustic folk, slow and airy")

...lands in the same 512-dimensional space as the embedding of an actual
acoustic folk song. So we can compare a *sentence* to *audio* directly, by
asking how closely the two vectors point the same way.

That is why we never need paired (photo, song) training data to get a first
version working. Words are the meeting point.

WHY EVERY VECTOR IS NORMALISED
------------------------------
We rescale every vector to length 1. After that, the dot product of two
vectors is exactly the cosine of the angle between them - a similarity score
from -1 (opposite) to 1 (identical). This is what lets us use a plain
inner-product FAISS index and still be doing cosine similarity.

WHICH MODEL, AND A HARD-WON LESSON
----------------------------------
`laion/clap-htsat-unfused` - the original LAION CLAP checkpoint.

We started with `laion/larger_clap_music`, because a checkpoint fine-tuned on
music obviously beats a general one. It does not, because that upload is
broken: its text projection layer was never trained. The weights are still at
their starting values - every bias is exactly 0.0, and the weights have a
third the spread of a trained layer.

Nothing crashes. It loads, it produces embeddings, the index builds, searches
run. But the text encoder returns nearly the *same vector for every sentence*,
so every query returns the same songs. That reads as "the AI is not very good"
rather than "the model is broken", which is exactly why it is dangerous.

Measured on 24 tracks and 5 genre queries where the right answer was known:

    laion/clap-htsat-unfused    3/5 top-1, mean rank 2.4   <- chosen
    laion/larger_clap_general   1/5 top-1, mean rank 2.6
    laion/larger_clap_music     text encoder dead

(random guessing scores about 3.5)

`self_test()` at the bottom of this file exists so this can never happen
silently again. Run it against any new checkpoint before trusting it.

AN OPEN QUESTION WE TEST, NOT ASSUME
------------------------------------
CLAP's training captions were overwhelmingly English. Whether it understands
Hindi and Punjabi *vocals* is genuinely unknown to us. `scripts/05_search.py`
is built to answer that. If the answer is "poorly", the fix is to describe
instrumentation and mood in the query rather than language, and let metadata
filters handle language - but we want to find that out now, not in Week 9.
"""

import numpy as np

# The checkpoint we use. Music-tuned, not the general-audio one.
MODEL_NAME = "laion/clap-htsat-unfused"

# CLAP's output size. Both encoders produce a vector this long.
EMBED_DIM = 512


def pick_device(preferred=None):
    """Return 'cuda' if a GPU is available, otherwise 'cpu'.

    Everything here runs on a CPU. A GPU just makes it roughly 20x faster,
    which is the difference between minutes and an evening.
    """
    import torch

    if preferred:
        return preferred
    return "cuda" if torch.cuda.is_available() else "cpu"


def load(model_name=MODEL_NAME, device=None, half=False):
    """Download (first time) and load CLAP.

    Parameters
    ----------
    half : bool
        Use 16-bit precision on the GPU. Roughly doubles speed and halves
        memory - but see the warning below. Off by default.

    Returns
    -------
    (model, processor, device)
        `model` does the thinking; `processor` converts raw waveforms and raw
        strings into the exact tensor format the model expects.

    WHY HALF PRECISION IS OFF BY DEFAULT
    ------------------------------------
    16-bit floats hold far less detail than 32-bit ones. For most models that
    is a free speed-up, but some produce NaNs or quietly lose accuracy, and we
    have no GPU to test this on. A silent accuracy loss here would be
    especially nasty: the embeddings would still look completely normal.

    Full precision on a Colab T4 embeds the whole catalog in well under an
    hour, which is fast enough. Speed is not worth an untested risk to the
    numbers everything else is built on. Pass --fp16 if you want it and can
    check the results.
    """
    import torch
    from transformers import ClapModel, ClapProcessor

    device = pick_device(device)

    processor = ClapProcessor.from_pretrained(model_name)
    model = ClapModel.from_pretrained(model_name)

    # Evaluation mode: we are using the model, not training it. This switches
    # off dropout and similar training-only behaviour.
    model = model.to(device).eval()

    if half and device == "cuda":
        model = model.half()

    return model, processor, device


def _as_tensor(features):
    """Pull the plain tensor out of whatever the model handed back.

    transformers 4.x returned a bare tensor from `get_audio_features`.
    transformers 5.x returns a `BaseModelOutputWithPooling` object instead,
    with the 512-number shared-space vector in `.pooler_output`.

    Supporting both means this code runs on your laptop and on Colab without
    anyone having to pin a version.
    """
    if hasattr(features, "pooler_output"):
        return features.pooler_output
    return features


def _normalise(vectors):
    """Rescale each row to length 1, so dot product becomes cosine similarity."""
    lengths = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Guard against dividing by zero on an all-silent clip.
    lengths[lengths == 0] = 1.0
    return (vectors / lengths).astype(np.float32)


def embed_audio(model, processor, waveforms, device, sample_rate=48000):
    """Turn a list of waveforms into a (n, 512) array of unit vectors.

    Every waveform must already be mono, float32, and at `sample_rate`.
    See `src/encode/audio.py`, which produces exactly that.
    """
    import torch

    if not waveforms:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    # NOTE: transformers 5.x renamed this argument from `audios` to `audio`.
    # On transformers 4.x the old name is required, so we try the new name and
    # fall back - this keeps working on Colab, which may pin an older version.
    try:
        inputs = processor(
            audio=list(waveforms),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
    except (TypeError, ValueError):
        inputs = processor(
            audios=list(waveforms),
            sampling_rate=sample_rate,
            return_tensors="pt",
        )
    inputs = {key: value.to(device) for key, value in inputs.items()}

    # Match the inputs to whatever precision the model is actually in, rather
    # than assuming. Guessing here is how you get a dtype mismatch crash.
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.float16:
        inputs = {
            key: value.half() if value.dtype == torch.float32 else value
            for key, value in inputs.items()
        }

    # `no_grad` tells PyTorch not to remember how it computed each number.
    # We are not training, so that bookkeeping is pure wasted memory.
    with torch.no_grad():
        features = _as_tensor(model.get_audio_features(**inputs))

    return _normalise(features.float().cpu().numpy())


def embed_text(model, processor, texts, device):
    """Turn a list of sentences into a (n, 512) array of unit vectors.

    These vectors live in the *same space* as the audio ones above. That is
    the entire point of CLAP.
    """
    import torch

    if not texts:
        return np.zeros((0, EMBED_DIM), dtype=np.float32)

    inputs = processor(text=list(texts), return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}

    with torch.no_grad():
        features = _as_tensor(model.get_text_features(**inputs))

    return _normalise(features.float().cpu().numpy())


def embed_track(model, processor, wave, device, sample_rate=48000):
    """Embed one full preview: window it, embed each window, average, normalise.

    Averaging several windows gives a vector describing the *whole* preview
    rather than one arbitrary slice of it. Averaging breaks the unit length,
    so we normalise again afterwards.

    Returns a single (512,) vector, or None if the audio was unusable.
    """
    from src.encode import audio as audio_module

    chunks = audio_module.windows(wave, sample_rate=sample_rate)
    if not chunks:
        return None

    per_window = embed_audio(model, processor, chunks, device, sample_rate)
    averaged = per_window.mean(axis=0, keepdims=True)
    return _normalise(averaged)[0]

def prepare_audio(processor, wave, sample_rate=48000):
    """Do the CPU-side half of embedding one song: windows -> mel spectrograms.

    WHY THIS IS SPLIT OUT
    ---------------------
    Turning a waveform into the mel spectrogram the model expects is pure
    number-crunching on the CPU, and it is not cheap - measured at 0.19s per
    song, against 0.12s to decode the audio in the first place.

    It is split out so the CPU half and the GPU half are separable and can be
    measured independently, which is how the numbers below were found.

    DO NOT run this in a thread pool. It looks like NumPy maths that would
    release the GIL, but Hugging Face's feature extractor runs Python-level
    loops and holds the lock throughout. Measured on 8 songs: 1.27s serial
    against 7.84s across 8 threads - six times slower, because the threads
    only add contention. Parallelising it needs processes, not threads.

    Returns
    -------
    (inputs, window_count)
        `inputs` is None when the audio was unusable.
    """
    from src.encode import audio as audio_module

    chunks = audio_module.windows(wave, sample_rate=sample_rate)
    if not chunks:
        return None, 0

    inputs = processor(audio=chunks, sampling_rate=sample_rate, return_tensors="pt")
    return inputs, len(chunks)


def embed_prepared(model, prepared, counts, device):
    """Run the GPU half on features already extracted by `prepare_audio`.

    `prepared` is a list of processor outputs, one per song; `counts` says how
    many windows each contributed. They are merged into a single forward pass
    and then split back apart per song.
    """
    import torch

    if not prepared:
        return []

    keys = prepared[0].keys()
    merged = {key: torch.cat([item[key] for item in prepared]) for key in keys}
    merged = {key: value.to(device) for key, value in merged.items()}

    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.float16:
        merged = {
            key: value.half() if value.dtype == torch.float32 else value
            for key, value in merged.items()
        }

    with torch.no_grad():
        features = _as_tensor(model.get_audio_features(**merged))

    per_window = _normalise(features.float().cpu().numpy())

    vectors = []
    position = 0
    for count in counts:
        averaged = per_window[position : position + count].mean(axis=0, keepdims=True)
        vectors.append(_normalise(averaged)[0])
        position += count
    return vectors

def embed_tracks(model, processor, waves, device, sample_rate=48000):
    """Embed several previews in ONE forward pass. Same maths as embed_track.

    WHY THIS EXISTS
    ---------------
    `embed_track` handles one song, which means one GPU call per song with
    only 3 windows in it. A GPU is a machine for doing thousands of identical
    sums at once - feeding it three at a time leaves it mostly idle waiting
    for Python to hand it the next scrap of work.

    Measured on a Colab T4: one song at a time ran at 2.3 songs/sec, only
    2.5x a laptop CPU, which is absurd for a GPU. The fix is to gather the
    windows from every song in the batch and send them together.

    This is the single most common performance mistake in beginner ML code,
    and it looks like nothing: the per-song version is correct, readable, and
    slow. Correct and slow is easy to leave in place for weeks.

    Parameters
    ----------
    waves : list[np.ndarray]
        Decoded previews. Each becomes 1-3 windows depending on length, which
        is why we track how many belong to each song.

    Returns
    -------
    list
        One (512,) vector per input wave, or None where the audio was unusable.
        Same order as `waves`.
    """
    from src.encode import audio as audio_module

    chunks = []
    counts = []
    for wave in waves:
        windows = audio_module.windows(wave, sample_rate=sample_rate)
        counts.append(len(windows))
        chunks.extend(windows)

    if not chunks:
        return [None] * len(waves)

    per_window = embed_audio(model, processor, chunks, device, sample_rate)

    # Walk back through, taking each song's share of the windows and averaging
    # them. `counts` is what makes this safe when songs differ in length.
    vectors = []
    position = 0
    for count in counts:
        if count == 0:
            vectors.append(None)
            continue
        averaged = per_window[position : position + count].mean(axis=0, keepdims=True)
        vectors.append(_normalise(averaged)[0])
        position += count

    return vectors

# Sentences chosen to be about as different as music descriptions get.
# If a model cannot tell these apart, it cannot tell anything apart.
_SELF_TEST_PROMPTS = [
    "aggressive heavy metal guitar with screaming vocals",
    "quiet solo piano, sad and slow",
    "energetic punjabi bhangra with dhol drums",
]


def self_test(model, processor, device, threshold=0.98):
    """Check that the text encoder actually distinguishes different sentences.

    WHY THIS EXISTS
    ---------------
    We lost an afternoon to a checkpoint whose text tower returned nearly the
    same vector for every sentence. Nothing crashed. Embeddings were produced,
    the index built, searches ran - and every query returned the same songs,
    which reads as "the AI is bad" rather than "the model is broken".

    The tell is measurable in one line: embed a few wildly different sentences
    and measure the length of their average. Unit vectors pointing in genuinely
    different directions partly cancel, so a healthy encoder gives an average
    noticeably shorter than 1. A broken one gives ~0.9997, because every vector
    is the same vector.

    Returns
    -------
    (ok, offset)
        `ok` is False when the encoder looks degenerate. `offset` is that
        average length - closer to 1.0 is worse.
    """
    vectors = embed_text(model, processor, _SELF_TEST_PROMPTS, device)
    offset = float(np.linalg.norm(vectors.mean(axis=0)))
    return offset < threshold, offset


def require_working_text_encoder(model, processor, device):
    """Refuse to start a long job on a model that cannot encode text.

    Raises RuntimeError with an explanation rather than silently producing
    hours of meaningless numbers.
    """
    ok, offset = self_test(model, processor, device)
    if not ok:
        raise RuntimeError(
            "CLAP's text encoder looks broken on this checkpoint. "
            "Three completely different sentences produced almost the same "
            f"vector (shared-offset {offset:.4f}; anything above 0.98 is bad). "
            "Retrieval would return the same songs for every query, so this "
            "refuses to run. Try a different checkpoint via --model."
        )
    return offset
