from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, TypeVar

from .text_utils import normalize_for_eval, tokenize_words

T = TypeVar("T")


@dataclass(frozen=True)
class TextMetrics:
    cer: float
    wer: float
    ned: float
    char_accuracy: float
    word_accuracy: float


def edit_distance(a: Sequence[T], b: Sequence[T]) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(
                min(
                    previous[j] + 1,
                    current[j - 1] + 1,
                    previous[j - 1] + cost,
                )
            )
        previous = current
    return previous[-1]


def compute_text_metrics(reference: str, hypothesis: str) -> TextMetrics:
    ref = normalize_for_eval(reference)
    hyp = normalize_for_eval(hypothesis)
    char_edits = edit_distance(ref, hyp)
    ref_words = tokenize_words(ref)
    hyp_words = tokenize_words(hyp)
    word_edits = edit_distance(ref_words, hyp_words)
    cer = char_edits / max(len(ref), 1)
    wer = word_edits / max(len(ref_words), 1)
    ned = char_edits / max(max(len(ref), len(hyp)), 1)
    return TextMetrics(
        cer=cer,
        wer=wer,
        ned=ned,
        char_accuracy=max(0.0, 1.0 - cer),
        word_accuracy=max(0.0, 1.0 - wer),
    )

