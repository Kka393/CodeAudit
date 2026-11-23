"""
ml/cleaner.py
=============
REUSED, unmodified in behavior from the existing AICD_MultiView
research project's src/preprocessing/cleaner.py. Only the import
path changed (there was no dependency on the old package layout).

Light, non-destructive cleaning of raw source code prior to
tokenization.
"""

import re
import unicodedata

_EXCESSIVE_BLANK_LINES = re.compile(r"\n{3,}")
_TRAILING_WHITESPACE = re.compile(r"[ \t]+(?=\n)")


def clean_code(source_code: str) -> str:
    """Clean and normalize a raw source-code string. See original
    project docstring for full behavior description; logic is
    unchanged: unicode NFC normalize, strip BOM, normalize line
    endings, strip trailing whitespace, collapse excessive blank
    lines, trim leading/trailing blank lines."""
    if source_code is None:
        return ""
    if not isinstance(source_code, str):
        raise TypeError(f"clean_code expects a str, got {type(source_code)}")
    if source_code.strip() == "":
        return ""

    text = unicodedata.normalize("NFC", source_code)
    if text.startswith("\ufeff"):
        text = text.lstrip("\ufeff")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = _TRAILING_WHITESPACE.sub("", text)
    text = _EXCESSIVE_BLANK_LINES.sub("\n\n", text)
    text = text.strip("\n")
    return text
