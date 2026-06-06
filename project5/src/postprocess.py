from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Mapping

from .text_utils import (
    TOKEN_RE,
    ascii_fold_turkish,
    case_like,
    has_turkish_diacritic,
    normalize_for_eval,
)


COMMON_RESTORATIONS = {
    "Turkiye": "Türkiye",
    "turkiye": "Türkiye",
    "Turkce": "Türkçe",
    "turkce": "Türkçe",
    "cunku": "çünkü",
    "Cunku": "Çünkü",
    "icin": "için",
    "olcumu": "ölçümü",
    "olcum": "ölçüm",
    "universite": "üniversite",
    "Universite": "Üniversite",
}


@dataclass
class LexiconModel:
    total_counts: dict[str, Counter[str]]
    document_counts: dict[str, dict[str, Counter[str]]]

    def candidates(self, folded: str, exclude_document: str | None = None) -> Counter[str]:
        counts = Counter(self.total_counts.get(folded, Counter()))
        if exclude_document is not None:
            for token, count in self.document_counts.get(exclude_document, {}).get(folded, Counter()).items():
                counts[token] -= count
                if counts[token] <= 0:
                    del counts[token]
        return counts


def build_lexicon(reference_by_id: Mapping[str, str]) -> LexiconModel:
    total: dict[str, Counter[str]] = defaultdict(Counter)
    per_doc: dict[str, dict[str, Counter[str]]] = {}
    for doc_id, text in reference_by_id.items():
        doc_counts: dict[str, Counter[str]] = defaultdict(Counter)
        for match in TOKEN_RE.finditer(text):
            token = match.group(0)
            if len(token) < 4 or token.isdigit():
                continue
            folded = ascii_fold_turkish(token).lower()
            total[folded][token] += 1
            doc_counts[folded][token] += 1
        per_doc[doc_id] = doc_counts
    return LexiconModel(dict(total), per_doc)


def normalize_ocr_output(text: str) -> str:
    text = re.sub(r"([A-Za-zÇĞİÖŞÜçğıöşü])- *\n *([A-Za-zÇĞİÖŞÜçğıöşü])", r"\1\2", text)
    text = text.replace("|", "I")
    text = re.sub(r"\brn\b", "m", text)
    return normalize_for_eval(text)


def choose_candidate(token: str, lexicon: LexiconModel | None, exclude_document: str | None) -> str:
    if token in COMMON_RESTORATIONS:
        return COMMON_RESTORATIONS[token]
    if lexicon is None or len(token) < 4 or token.isdigit():
        return token
    folded = ascii_fold_turkish(token).lower()
    candidates = lexicon.candidates(folded, exclude_document)
    if not candidates:
        return token
    candidate, count = candidates.most_common(1)[0]
    current_count = candidates.get(token, 0) + candidates.get(token.lower(), 0)
    if candidate.lower() == token.lower():
        return token
    if not has_turkish_diacritic(candidate):
        return token
    # Conservative threshold to avoid over-correcting names and abbreviations.
    if count < max(2, current_count + 1):
        return token
    return case_like(candidate, token)


def correct_text(text: str, lexicon: LexiconModel | None = None, exclude_document: str | None = None) -> str:
    normalized = normalize_ocr_output(text)

    def replace(match: re.Match[str]) -> str:
        return choose_candidate(match.group(0), lexicon, exclude_document)

    corrected = TOKEN_RE.sub(replace, normalized)
    return normalize_for_eval(corrected)

