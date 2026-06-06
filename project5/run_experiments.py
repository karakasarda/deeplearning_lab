from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path
from typing import Any

from src.config import DEFAULT_DATASET_ROOT, DEFAULT_DPI, DEFAULT_ENGINES, RENDERED_DIR, RESULTS_DIR
from src.data_utils import create_demo_dataset, discover_items, ensure_ocrturk_dataset, load_reference_text, render_pdf_pages
from src.diacritics import analyze_diacritics
from src.metrics import compute_text_metrics
from src.ocr_engines import build_engine
from src.postprocess import build_lexicon, correct_text, normalize_ocr_output
from src.reporting import (
    aggregate_summary,
    ensure_dirs,
    plot_confusion_heatmap,
    plot_metric_bars,
    save_csv,
    save_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run OCRTurk OCR and diacritic post-correction experiments")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--download-data", action="store_true", help="Clone OCRTurk into --dataset-root if missing")
    parser.add_argument("--make-demo", action="store_true", help="Create a tiny synthetic OCRTurk-like dataset")
    parser.add_argument("--engines", nargs="+", default=list(DEFAULT_ENGINES), help="OCR engines: tesseract easyocr paddleocr mock")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of OCRTurk pages")
    parser.add_argument("--dpi", type=int, default=DEFAULT_DPI)
    parser.add_argument("--gpu", action="store_true", help="Use GPU for engines that support it")
    parser.add_argument("--force-render", action="store_true")
    parser.add_argument("--force-ocr", action="store_true")
    parser.add_argument("--skip-render", action="store_true", help="Only valid for engines that do not need images, such as mock")
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--rendered-dir", type=Path, default=RENDERED_DIR)
    return parser.parse_args()


def metric_row(
    item: Any,
    engine: str,
    stage: str,
    status: str,
    reference: str,
    hypothesis: str,
    error: str = "",
) -> tuple[dict[str, object], list[dict[str, object]]]:
    base = {
        "item_id": item.item_id,
        "engine": engine,
        "stage": stage,
        "status": status,
        "document_type": item.document_type,
        "difficulty": item.difficulty,
        "source": item.source,
        "error": error,
    }
    if status != "ok":
        return base, []
    metrics = compute_text_metrics(reference, hypothesis)
    confusion_rows, dia = analyze_diacritics(reference, hypothesis)
    base.update(
        {
            "cer": metrics.cer,
            "wer": metrics.wer,
            "ned": metrics.ned,
            "char_accuracy": metrics.char_accuracy,
            "word_accuracy": metrics.word_accuracy,
            **dia,
        }
    )
    for row in confusion_rows:
        row.update(
            {
                "item_id": item.item_id,
                "engine": engine,
                "stage": stage,
                "document_type": item.document_type,
                "difficulty": item.difficulty,
            }
        )
    return base, confusion_rows


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    ensure_dirs(args.results_dir, args.rendered_dir, args.results_dir / "outputs", args.results_dir / "plots")
    if args.make_demo:
        create_demo_dataset(args.dataset_root)
    if args.download_data:
        ensure_ocrturk_dataset(args.dataset_root)

    items = discover_items(args.dataset_root)
    if args.limit is not None:
        items = items[: args.limit]
    if not items:
        raise RuntimeError(
            f"No OCRTurk items found under {args.dataset_root}. Use --download-data or --make-demo."
        )

    references = {item.item_id: load_reference_text(item) for item in items}
    lexicon = build_lexicon(references)
    engines = [build_engine(name, gpu=args.gpu) for name in args.engines]
    availability = {engine.name: dict(zip(("available", "reason"), engine.available())) for engine in engines}

    summary_rows: list[dict[str, object]] = []
    confusion_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    for item in items:
        reference = references[item.item_id]
        rendered_pages: list[Path] = []
        needs_images = any(engine.needs_images for engine in engines)
        if needs_images and not args.skip_render:
            try:
                rendered_pages = render_pdf_pages(item, args.rendered_dir, args.dpi, force=args.force_render)
            except Exception as exc:
                failures.append({"item_id": item.item_id, "stage": "render", "error": str(exc)})
        elif needs_images and args.skip_render:
            failures.append({"item_id": item.item_id, "stage": "render", "error": "--skip-render used with image OCR engines"})

        for engine in engines:
            is_available, reason = engine.available()
            raw_output_path = args.results_dir / "outputs" / engine.name / f"{item.item_id}.txt"
            if not is_available:
                row, _ = metric_row(item, engine.name, "raw", "skipped", reference, "", reason)
                summary_rows.append(row)
                continue
            if engine.needs_images and not rendered_pages:
                row, _ = metric_row(item, engine.name, "raw", "error", reference, "", "no rendered page image available")
                summary_rows.append(row)
                continue
            try:
                if raw_output_path.exists() and not args.force_ocr:
                    raw_text = raw_output_path.read_text(encoding="utf-8")
                    status = "ok"
                    error = ""
                else:
                    result = engine.recognize(rendered_pages, reference_text=reference)
                    raw_text = result.text
                    status = result.status
                    error = result.error
                    if status == "ok":
                        save_text(raw_output_path, raw_text)
            except Exception as exc:
                raw_text = ""
                status = "error"
                error = str(exc)
                failures.append({"item_id": item.item_id, "stage": engine.name, "error": error})

            normalized_raw = normalize_ocr_output(raw_text)
            row, rows = metric_row(item, engine.name, "raw", status, reference, normalized_raw, error)
            summary_rows.append(row)
            confusion_rows.extend(rows)
            if status == "ok":
                corrected = correct_text(normalized_raw, lexicon=lexicon, exclude_document=item.item_id)
                corrected_path = args.results_dir / "outputs" / f"{engine.name}_corrected" / f"{item.item_id}.txt"
                save_text(corrected_path, corrected)
                row, rows = metric_row(item, engine.name, "corrected", "ok", reference, corrected)
                summary_rows.append(row)
                confusion_rows.extend(rows)

    summary_path = args.results_dir / "experiment_summary.csv"
    confusion_path = args.results_dir / "diacritic_confusions.csv"
    aggregate_path = args.results_dir / "aggregate_by_engine.csv"
    save_csv(summary_path, summary_rows)
    save_csv(confusion_path, confusion_rows)
    aggregate = aggregate_summary(summary_path)
    if not aggregate.empty:
        aggregate.to_csv(aggregate_path, index=False)
    plot_paths = plot_metric_bars(summary_path, args.results_dir / "plots")
    confusion_plot = plot_confusion_heatmap(confusion_path, args.results_dir / "plots")
    if confusion_plot:
        plot_paths.append(confusion_plot)

    save_json(
        args.results_dir / "run_status.json",
        {
            "status": "completed",
            "dataset_root": str(args.dataset_root),
            "item_count": len(items),
            "engines": args.engines,
            "run_config": {
                "dpi": args.dpi,
                "gpu": args.gpu,
                "force_render": args.force_render,
                "force_ocr": args.force_ocr,
                "skip_render": args.skip_render,
            },
            "availability": availability,
            "failures": failures,
            "summary_path": str(summary_path),
            "diacritic_confusions_path": str(confusion_path),
            "plots": [str(path) for path in plot_paths],
            "runtime_seconds": round(time.time() - start, 2),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    )
    print(f"Wrote {summary_path}")
    print(f"Wrote {confusion_path}")
    print(f"Wrote {args.results_dir / 'run_status.json'}")


if __name__ == "__main__":
    main()
