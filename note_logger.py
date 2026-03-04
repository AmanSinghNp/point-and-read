"""
note_logger.py -- Append OCR text to a Markdown notes file with timestamps.
"""

from __future__ import annotations

import os
from datetime import datetime


def default_notes_path() -> str:
    """Return default notes path (Desktop/notes.md), env-overridable."""
    env_path = os.environ.get("POINTREAD_NOTES_PATH", "").strip()
    if env_path:
        return env_path
    desktop = os.path.join(os.path.expanduser("~"), "Desktop")
    return os.path.join(desktop, "notes.md")


def append_markdown_entry(text: str, notes_path: str | None = None) -> str:
    """Append a timestamped markdown section to notes file.

    Args:
        text: OCR text to append.
        notes_path: Optional destination path. Defaults to Desktop/notes.md.

    Returns:
        The notes file path used.
    """
    body = (text or "").strip()
    if not body:
        raise ValueError("No text to append")

    target = notes_path or default_notes_path()
    os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    with open(target, "a", encoding="utf-8") as f:
        f.write(f"\n## {ts}\n\n")
        f.write(body)
        f.write("\n")

    return target

