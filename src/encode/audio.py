"""Fetch a 30-second preview and turn it into numbers a model can listen to.

WHAT "AUDIO AS NUMBERS" MEANS
----------------------------
Sound is a wave. To store it, a computer measures the height of that wave many
thousands of times per second. Each measurement is one number. "48000 Hz" means
48,000 measurements per second, so a 30-second preview is about 1.44 million
numbers.

That long list of numbers is called a **waveform**. It is what CLAP listens to.

THE FORMAT SURPRISE (worth reading)
-----------------------------------
Apple's documentation and most tutorials call these "preview MP3s". They are
not. Every single one of our 13,497 previews is an **.m4a** file - AAC audio
inside an MP4 container. You can see it in the first bytes of the file:
`ftypM4A`.

This matters because the obvious library, `soundfile`, cannot read AAC at all.
It reads MP3 happily, which is exactly the kind of near-miss that wastes an
afternoon: the download succeeds, the file is clearly audio, and decoding
fails with "Format not recognised".

So we use **PyAV**, which is a Python wrapper around FFmpeg - the program that
can decode essentially any audio format in existence. Crucially, the PyAV
package ships FFmpeg's decoders *inside the wheel*, so `pip install av` is the
whole setup. Nothing to install separately, no PATH to configure.

Lesson worth keeping: check what your data actually is, not what the docs say
it is. One `print(raw[:16])` would have saved the afternoon.

WHY 48000 AND NOT SOMETHING ELSE
--------------------------------
CLAP was trained on audio sampled at 48,000 Hz. Hand it a different rate and
every sound is effectively pitch-shifted, and the model gets confused. The
previews arrive at 44,100 Hz in stereo, so we convert: 48,000 Hz, single
channel. That conversion is called **resampling**, and PyAV does it for us.

WHY WE CUT THE PREVIEW INTO 10-SECOND WINDOWS
---------------------------------------------
CLAP only looks at 10 seconds at a time. Hand it 30 seconds and it quietly
throws most of it away - and in some configurations it keeps a *random* 10
seconds, which would make our results different every run.

So we do it ourselves, explicitly: cut the preview into three 10-second
windows, embed each one, and average the three. Nothing is discarded, and the
result is identical every time we run it.

WE NEVER SAVE THE AUDIO
-----------------------
Everything here happens in memory. The file is downloaded into a variable,
converted to numbers, and dropped. Nothing touches your disk. That is what
keeps this repo legal to publish.
"""

import io
import urllib.error
import urllib.request

import numpy as np

# CLAP's native sample rate. Do not change this - the model was trained on it.
SAMPLE_RATE = 48000

# How much audio CLAP looks at in one go.
WINDOW_SECONDS = 10

# Identify ourselves, same as the harvester does.
HEADERS = {"User-Agent": "Music-Instagram/0.1 (student research project)"}


def fetch(url, timeout=30, retries=2):
    """Download one preview into memory.

    Returns the raw bytes, or None if the download failed. A failure is not a
    crisis - previews expire and get pulled all the time. We skip the track and
    move on.
    """
    for attempt in range(retries + 1):
        try:
            request = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read()
        except (urllib.error.HTTPError, urllib.error.URLError, OSError):
            if attempt == retries:
                return None
    return None


def decode(raw_bytes, target_sr=SAMPLE_RATE):
    """Turn encoded audio bytes into a waveform: mono, float32, at `target_sr`.

    Handles .m4a/AAC (what iTunes actually serves), and also mp3, wav, flac and
    anything else FFmpeg understands - PyAV works it out from the file itself,
    so we never have to care what the extension claims.

    Returns None if the bytes are not decodable audio.
    """
    # Imported here, not at the top of the file, so that simply importing this
    # module does not require the audio libraries to be installed.
    import av

    try:
        with av.open(io.BytesIO(raw_bytes)) as container:
            if not container.streams.audio:
                return None

            # Ask FFmpeg to hand us exactly what we want: one channel, 48 kHz,
            # 32-bit floats. Doing it here means no separate resampling step.
            #   'fltp' = float, planar (channels stored one after another)
            resampler = av.AudioResampler(
                format="fltp", layout="mono", rate=target_sr
            )

            pieces = []
            for frame in container.decode(audio=0):
                for resampled in resampler.resample(frame):
                    pieces.append(resampled.to_ndarray().reshape(-1))

            # The resampler buffers internally. Flushing it (resample(None))
            # releases the last few milliseconds, which would otherwise be lost.
            for resampled in resampler.resample(None):
                pieces.append(resampled.to_ndarray().reshape(-1))

    except Exception:
        # Truncated download, DRM-protected file, dead link that returned an
        # HTML error page instead of audio - all land here. Skip the track.
        return None

    if not pieces:
        return None

    wave = np.concatenate(pieces).astype(np.float32)
    return wave if wave.size else None


def load(url, target_sr=SAMPLE_RATE):
    """Download and decode in one step. Returns a waveform, or None."""
    raw = fetch(url)
    if raw is None:
        return None
    return decode(raw, target_sr=target_sr)


def windows(wave, window_seconds=WINDOW_SECONDS, sample_rate=SAMPLE_RATE):
    """Cut a waveform into fixed-length chunks for CLAP.

    A 30-second preview at 48000 Hz becomes three arrays of 480,000 numbers.

    Two edge cases matter:

    - **Audio shorter than one window.** Some previews are only a few seconds.
      We pad the end with silence so the model still receives a full window.
    - **A leftover tail.** If the last chunk is less than half a window, we drop
      it rather than pad it, because mostly-silence would drag the average
      toward "quiet" and misrepresent the song.

    Returns
    -------
    list[np.ndarray]
        At least one window. Empty only if the waveform itself is empty.
    """
    size = int(window_seconds * sample_rate)
    if wave.size == 0:
        return []

    # Shorter than one window: pad with silence and return the single window.
    if wave.size < size:
        padded = np.zeros(size, dtype=np.float32)
        padded[: wave.size] = wave
        return [padded]

    chunks = []
    for start in range(0, wave.size, size):
        chunk = wave[start : start + size]
        if chunk.size < size:
            # Keep a partial tail only if it is at least half a window long.
            if chunk.size < size // 2:
                break
            padded = np.zeros(size, dtype=np.float32)
            padded[: chunk.size] = chunk
            chunk = padded
        chunks.append(np.asarray(chunk, dtype=np.float32))

    return chunks
