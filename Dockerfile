# The demo, packaged for a free Hugging Face Space.
#
# WHY DOCKER AND NOT THE GRADIO SDK
# ---------------------------------
# Gradio would be fewer steps and would replace the interface with its own.
# The page in app/static/ is part of the work - drag and drop, the vibe chips,
# the playable previews, the visible music query. This keeps exactly what runs
# on a laptop, so "it worked locally" and "it works in production" are the same
# claim about the same code.

FROM python:3.12-slim

# PyAV needs nothing extra - it ships FFmpeg's decoders inside its wheel, which
# is the whole reason it was chosen over soundfile back in Week 2. faiss and
# torch need libgomp, which slim images leave out.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run as user 1000. Writing as root leaves files the app cannot touch.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH" \
    HOME=/home/user \
    PYTHONUNBUFFERED=1 \
    # Model weights are large; cache them inside the writable home directory
    # rather than wherever the library defaults to.
    HF_HOME=/home/user/.cache/huggingface

WORKDIR /home/user/app

# Dependencies first, so editing the app does not reinstall torch every build.
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=user . .

# Fetch the catalogue and embeddings, then build the index. Done at build time
# rather than on first request, so a visitor never waits for a 28 MB download.
RUN python deploy/fetch_data.py && python scripts/04_build_index.py

# Spaces route traffic to 7860.
EXPOSE 7860
CMD ["python", "-m", "uvicorn", "app.main:app", \
     "--host", "0.0.0.0", "--port", "7860", "--log-level", "warning"]
