from __future__ import annotations

import html
import re
import unicodedata
from typing import Iterable


TURKISH_FOLD_MAP = str.maketrans(
    {
        "ç": "c",
        "Ç": "C",
        "ğ": "g",
        "Ğ": "G",
        "ı": "i",
        "I": "I",
        "İ": "I",
        "ö": "o",
        "Ö": "O",
        "ş": "s",
        "Ş": "S",
        "ü": "u",
        "Ü": "U",
    }
)

TOKEN_RE = re.compile(r"[A-Za-zÇĞİÖŞÜçğıöşü0-9]+(?:['’][A-Za-zÇĞİÖŞÜçğıöşü0-9]+)?", re.UNICODE)


def normalize_unicode(text: str) -> str:
    text = text.replace("\ufeff", "")
    text = unicodedata.normalize("NFC", text)
    replacements = {
        "\u00a0": " ",
        "\u200b": "",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "“": '"',
        "”": '"',
        "‘": "'",
        "’": "'",
        "–": "-",
        "—": "-",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    return text


def strip_markdown_to_text(markdown: str) -> str:
    text = normalize_unicode(markdown)
    text = html.unescape(text)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"`{1,3}[^`]*`{1,3}", " ", text)
    text = re.sub(r"\$\$.*?\$\$", " ", text, flags=re.DOTALL)
    text = re.sub(r"\$[^$]+\$", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"^\s{0,3}#{1,6}\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^\s*[-*+]\s+", " ", text, flags=re.MULTILINE)
    text = re.sub(r"\|", " ", text)
    return normalize_for_eval(text)


def normalize_for_eval(text: str) -> str:
    text = normalize_unicode(text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"([A-Za-zÇĞİÖŞÜçğıöşü])- *\n *([A-Za-zÇĞİÖŞÜçğıöşü])", r"\1\2", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_words(text: str) -> list[str]:
    return TOKEN_RE.findall(normalize_for_eval(text).lower())


def ascii_fold_turkish(text: str) -> str:
    return normalize_unicode(text).translate(TURKISH_FOLD_MAP)


def has_turkish_diacritic(text: str) -> bool:
    return any(ch in "çğıİöşüÇĞİÖŞÜ" for ch in text)


def case_like(candidate: str, original: str) -> str:
    if original.isupper():
        return candidate.upper()
    if original[:1].isupper():
        return candidate[:1].upper() + candidate[1:]
    return candidate


def join_nonempty(parts: Iterable[str], sep: str = "\n") -> str:
    return sep.join(part for part in parts if part)

