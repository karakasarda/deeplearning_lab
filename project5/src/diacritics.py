from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

from .metrics import edit_distance
from .text_utils import ascii_fold_turkish, normalize_for_eval


TURKISH_DIACRITICS = set("çğıİöşüÇĞİÖŞÜ")
DIACRITIC_BASES = set("cCgGiIIoOsSuU")
RELEVANT_CHARS = TURKISH_DIACRITICS | DIACRITIC_BASES


@dataclass(frozen=True)
class AlignmentOp:
    reference: Optional[str]
    hypothesis: Optional[str]
    operation: str


def _align_replace(ref: str, hyp: str) -> list[AlignmentOp]:
    if not ref:
        return [AlignmentOp(None, ch, "insertion") for ch in hyp]
    if not hyp:
        return [AlignmentOp(ch, None, "deletion") for ch in ref]

    rows = len(ref) + 1
    cols = len(hyp) + 1
    dp = [[0] * cols for _ in range(rows)]
    for i in range(rows):
        dp[i][0] = i
    for j in range(cols):
        dp[0][j] = j
    for i, rc in enumerate(ref, start=1):
        for j, hc in enumerate(hyp, start=1):
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + (0 if rc == hc else 1),
            )

    ops: list[AlignmentOp] = []
    i, j = len(ref), len(hyp)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            if dp[i][j] == dp[i - 1][j - 1] + cost:
                op = "match" if cost == 0 else "substitution"
                ops.append(AlignmentOp(ref[i - 1], hyp[j - 1], op))
                i -= 1
                j -= 1
                continue
        if i > 0 and dp[i][j] == dp[i - 1][j] + 1:
            ops.append(AlignmentOp(ref[i - 1], None, "deletion"))
            i -= 1
        else:
            ops.append(AlignmentOp(None, hyp[j - 1], "insertion"))
            j -= 1
    ops.reverse()
    return ops


def align_characters(reference: str, hypothesis: str) -> list[AlignmentOp]:
    ref = normalize_for_eval(reference)
    hyp = normalize_for_eval(hypothesis)
    # Full DP is acceptable for OCRTurk page text sizes and gives stable substitutions.
    if len(ref) * len(hyp) <= 10_000_000:
        return _align_replace(ref, hyp)

    # Fallback for unexpectedly long inputs: keep memory bounded, still count large edits.
    import difflib

    ops: list[AlignmentOp] = []
    matcher = difflib.SequenceMatcher(a=ref, b=hyp, autojunk=False)
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            ops.extend(AlignmentOp(a, b, "match") for a, b in zip(ref[i1:i2], hyp[j1:j2]))
        elif tag == "delete":
            ops.extend(AlignmentOp(ch, None, "deletion") for ch in ref[i1:i2])
        elif tag == "insert":
            ops.extend(AlignmentOp(None, ch, "insertion") for ch in hyp[j1:j2])
        else:
            ops.extend(_align_replace(ref[i1:i2], hyp[j1:j2]))
    return ops


def diacritic_pair_label(reference: Optional[str], hypothesis: Optional[str]) -> str:
    ref = reference if reference is not None else "<ins>"
    hyp = hypothesis if hypothesis is not None else "<del>"
    return f"{ref}->{hyp}"


def analyze_diacritics(reference: str, hypothesis: str) -> tuple[list[dict[str, object]], dict[str, object]]:
    ops = align_characters(reference, hypothesis)
    confusion = Counter()
    ref_diacritic_count = 0
    correct_diacritic_count = 0
    base_loss_count = 0
    total_relevant_errors = 0

    for op in ops:
        ref = op.reference
        hyp = op.hypothesis
        ref_is_diacritic = ref in TURKISH_DIACRITICS if ref else False
        relevant = (ref in RELEVANT_CHARS if ref else False) or (hyp in RELEVANT_CHARS if hyp else False)
        if ref_is_diacritic:
            ref_diacritic_count += 1
            if ref == hyp:
                correct_diacritic_count += 1
        if relevant and op.operation != "match":
            total_relevant_errors += 1
            folded_equal = bool(ref and hyp and ascii_fold_turkish(ref).lower() == ascii_fold_turkish(hyp).lower())
            if ref_is_diacritic and folded_equal and ref != hyp:
                base_loss_count += 1
            confusion[(ref or "", hyp or "", op.operation, folded_equal)] += 1

    rows = [
        {
            "reference_char": ref,
            "hypothesis_char": hyp,
            "pair": diacritic_pair_label(ref or None, hyp or None),
            "operation": operation,
            "folded_equal": folded_equal,
            "count": count,
        }
        for (ref, hyp, operation, folded_equal), count in sorted(confusion.items())
    ]
    summary = {
        "reference_diacritic_count": ref_diacritic_count,
        "correct_diacritic_count": correct_diacritic_count,
        "diacritic_error_count": max(ref_diacritic_count - correct_diacritic_count, 0),
        "diacritic_accuracy": correct_diacritic_count / max(ref_diacritic_count, 1),
        "base_loss_count": base_loss_count,
        "relevant_char_error_count": total_relevant_errors,
        "char_edit_distance": edit_distance(normalize_for_eval(reference), normalize_for_eval(hypothesis)),
    }
    return rows, summary

