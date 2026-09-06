"""Make printing non-English text safe on Windows.

THE PROBLEM
-----------
Windows terminals often default to an old text encoding (cp1252) that cannot
represent Hindi, Punjabi, or even a curly apostrophe. When Python tries to
print a track name like "तुम ही हो" it either shows garbage or crashes
with UnicodeEncodeError.

Since our whole catalog is Indian music, this would bite us constantly.

THE FIX
-------
Call `setup()` at the top of any script that prints track names. It switches
stdout to UTF-8 and, if a character still cannot be shown, prints a
replacement character instead of crashing.
"""

import sys


def setup():
    """Switch stdout/stderr to UTF-8 so non-English text prints safely."""
    for stream in (sys.stdout, sys.stderr):
        # `reconfigure` exists on Python 3.7+. Guard anyway, so this never
        # becomes the reason a script fails to start.
        if hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass
