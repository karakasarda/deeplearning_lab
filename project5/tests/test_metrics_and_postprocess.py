from __future__ import annotations

import unittest

from src.diacritics import analyze_diacritics
from src.metrics import compute_text_metrics
from src.postprocess import build_lexicon, correct_text


class MetricsAndPostprocessTest(unittest.TestCase):
    def test_cer_and_wer_detect_diacritic_loss(self) -> None:
        metrics = compute_text_metrics("Türkçe ölçüm", "Turkce olcum")
        self.assertGreater(metrics.cer, 0)
        self.assertGreater(metrics.wer, 0)

    def test_diacritic_confusion_counts_base_loss(self) -> None:
        rows, summary = analyze_diacritics("çığ öşü", "cig osu")
        self.assertGreater(summary["base_loss_count"], 0)
        pairs = {row["pair"] for row in rows}
        self.assertIn("ç->c", pairs)

    def test_leave_one_out_lexicon_correction(self) -> None:
        lexicon = build_lexicon(
            {
                "data_1": "Türkiye ekonomisi için ölçüm",
                "data_2": "Türkiye üniversite raporu",
            }
        )
        corrected = correct_text("Turkiye ekonomisi icin olcum", lexicon=lexicon, exclude_document="missing")
        self.assertIn("Türkiye", corrected)


if __name__ == "__main__":
    unittest.main()

