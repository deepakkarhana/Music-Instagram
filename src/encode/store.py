"""A resumable place to keep embeddings while we compute them.

WHY THIS FILE EXISTS
--------------------
Embedding ten thousand songs takes hours. Wifi drops. Laptops sleep. Colab
disconnects you after 90 minutes of inactivity. If a crash at hour three
means starting again at hour zero, the project stops being fun very quickly.

Week 1 solved the same problem for the harvester by appending to a CSV and
re-reading it on startup. This is the same habit applied to numbers.

HOW IT WORKS
------------
Two files that grow side by side, one row at a time:

    clap_audio.f32       raw float32 numbers, 512 per track, back to back
    clap_audio_ids.csv   one track_id per line, in the same order

Row 7 of the ids file describes numbers 7*512 through 8*512-1 of the vector
file. That is the whole format.

WHY NOT JUST np.save?
---------------------
A `.npy` file stores its length in a header at the front, so appending means
rewriting the header - and a crash mid-rewrite corrupts everything. A plain
stream of float32 has no header, so appending is just "write 512 more numbers
at the end". A crash costs you at most one track, and the file is still
readable. We recover the shape at load time by dividing by 512.

FAILURES ARE RECORDED TOO
-------------------------
If a preview is dead, we log the track_id in a third file. Otherwise every
rerun would retry the same broken URLs forever.
"""

import csv
import os

import numpy as np


class VectorStore:
    """Append-only storage for embeddings, safe to interrupt at any time."""

    def __init__(self, base_path, dim=512):
        """`base_path` is a path without an extension, e.g. 'data/interim/clap_audio'."""
        self.dim = dim
        self.vectors_path = base_path + ".f32"
        self.ids_path = base_path + "_ids.csv"
        self.failed_path = base_path + "_failed.csv"

        folder = os.path.dirname(base_path)
        if folder:
            os.makedirs(folder, exist_ok=True)

    # ---------------------------------------------------------------- reading

    def done_ids(self):
        """Track IDs already embedded successfully."""
        return self._read_id_file(self.ids_path)

    def failed_ids(self):
        """Track IDs we tried and could not embed. We do not retry these."""
        return self._read_id_file(self.failed_path)

    def skip_ids(self):
        """Everything we should not attempt again: done plus failed."""
        return self.done_ids() | self.failed_ids()

    @staticmethod
    def _read_id_file(path):
        if not os.path.exists(path):
            return set()
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return {line.strip() for line in handle if line.strip()}

    def load(self):
        """Read everything back.

        Returns
        -------
        (list[str], np.ndarray)
            Track IDs, and a matching (n, dim) array of embeddings.
        """
        ids = []
        if os.path.exists(self.ids_path):
            with open(self.ids_path, "r", encoding="utf-8", newline="") as handle:
                ids = [line.strip() for line in handle if line.strip()]

        if not os.path.exists(self.vectors_path):
            return ids, np.zeros((0, self.dim), dtype=np.float32)

        flat = np.fromfile(self.vectors_path, dtype=np.float32)
        vectors = flat.reshape(-1, self.dim)

        # If a crash happened mid-write the two files can disagree by one row.
        # Trust the shorter one; a single dropped track is not worth a repair
        # tool, and the next run will simply redo it.
        count = min(len(ids), len(vectors))
        return ids[:count], vectors[:count]

    # ---------------------------------------------------------------- writing

    def open_for_append(self):
        """Open both files for appending. Remember to call `close`."""
        self._vectors_handle = open(self.vectors_path, "ab")
        self._ids_handle = open(self.ids_path, "a", encoding="utf-8", newline="")
        self._failed_handle = open(self.failed_path, "a", encoding="utf-8", newline="")
        return self

    def append(self, track_id, vector):
        """Save one embedding.

        Order matters: write the numbers *first*, then the id. If we crash in
        between, we end up with an extra vector and no id for it, which `load`
        trims off harmlessly. The other order would leave an id pointing at
        numbers that are not there.
        """
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)
        if vector.size != self.dim:
            raise ValueError(f"expected {self.dim} numbers, got {vector.size}")

        vector.tofile(self._vectors_handle)
        self._ids_handle.write(str(track_id) + "\n")

    def append_failed(self, track_id):
        """Record that this track could not be embedded, so we skip it later."""
        self._failed_handle.write(str(track_id) + "\n")

    def flush(self):
        """Push everything to disk, so an abrupt kill loses nothing."""
        for handle in (self._vectors_handle, self._ids_handle, self._failed_handle):
            handle.flush()
            os.fsync(handle.fileno())

    def close(self):
        for handle in (self._vectors_handle, self._ids_handle, self._failed_handle):
            handle.close()

    def __enter__(self):
        return self.open_for_append()

    def __exit__(self, *exc_info):
        try:
            self.flush()
        finally:
            self.close()
        return False


def read_catalog(csv_path):
    """Load the Week 1 catalog CSV into a list of dictionaries."""
    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))
