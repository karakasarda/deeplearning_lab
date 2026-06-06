from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


def ensure_dirs(*paths: Path) -> None:
    for path in paths:
        path.mkdir(parents=True, exist_ok=True)


def save_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def save_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def markdown_table(frame: pd.DataFrame, max_rows: int = 20, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return "_No rows available._"
    clipped = frame.head(max_rows)
    lines = [
        "| " + " | ".join(map(str, clipped.columns)) + " |",
        "| " + " | ".join(["---"] * len(clipped.columns)) + " |",
    ]
    for _, row in clipped.iterrows():
        cells = []
        for value in row.tolist():
            if isinstance(value, float):
                cells.append(format(value, floatfmt))
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def plot_metric_bars(summary_path: Path, plots_dir: Path) -> list[Path]:
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return []
    frame = pd.read_csv(summary_path)
    frame = frame[(frame["status"] == "ok") & frame["cer"].notna()]
    if frame.empty:
        return []
    plots_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []

    grouped = frame.groupby(["engine", "stage"], as_index=False)[["cer", "wer", "diacritic_error_count"]].mean()
    for metric, title, ylabel in [
        ("cer", "Average Character Error Rate", "CER"),
        ("wer", "Average Word Error Rate", "WER"),
        ("diacritic_error_count", "Average Diacritic Errors per Page", "Errors/page"),
    ]:
        pivot = grouped.pivot(index="engine", columns="stage", values=metric).fillna(0.0)
        ax = pivot.plot(kind="bar", figsize=(8.5, 4.8), color=["#8B1E3F", "#0E7C7B"])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("OCR engine")
        ax.grid(axis="y", alpha=0.25)
        plt.xticks(rotation=0)
        plt.tight_layout()
        out = plots_dir / f"{metric}_by_engine_stage.png"
        plt.savefig(out, dpi=180)
        plt.close()
        outputs.append(out)

    delta = frame.pivot_table(index="engine", columns="stage", values="cer", aggfunc="mean")
    if {"raw", "corrected"}.issubset(delta.columns):
        delta["cer_delta_raw_minus_corrected"] = delta["raw"] - delta["corrected"]
        ax = delta["cer_delta_raw_minus_corrected"].sort_values().plot(
            kind="barh", figsize=(7.5, 3.8), color="#2F6F9F"
        )
        ax.set_title("Post-correction CER Reduction")
        ax.set_xlabel("Raw CER - corrected CER")
        ax.grid(axis="x", alpha=0.25)
        plt.tight_layout()
        out = plots_dir / "postcorrection_cer_delta.png"
        plt.savefig(out, dpi=180)
        plt.close()
        outputs.append(out)
    return outputs


def plot_confusion_heatmap(confusion_path: Path, plots_dir: Path) -> Path | None:
    if not confusion_path.exists() or confusion_path.stat().st_size == 0:
        return None
    try:
        frame = pd.read_csv(confusion_path)
    except pd.errors.EmptyDataError:
        return None
    if frame.empty:
        return None
    frame = frame[frame["stage"] == "raw"]
    if frame.empty:
        return None
    top = (
        frame.groupby(["reference_char", "hypothesis_char"], as_index=False)["count"]
        .sum()
        .sort_values("count", ascending=False)
        .head(20)
    )
    if top.empty:
        return None
    reference = top["reference_char"].astype("string").fillna("<ins>")
    hypothesis = top["hypothesis_char"].astype("string").fillna("<del>")
    top["pair"] = reference + "->" + hypothesis
    plt.figure(figsize=(9, 5))
    plt.bar(top["pair"], top["count"], color="#8B1E3F")
    plt.title("Top Turkish Diacritic Confusions")
    plt.ylabel("Count")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plots_dir.mkdir(parents=True, exist_ok=True)
    out = plots_dir / "top_diacritic_confusions.png"
    plt.savefig(out, dpi=180)
    plt.close()
    return out


def aggregate_summary(summary_path: Path) -> pd.DataFrame:
    if not summary_path.exists() or summary_path.stat().st_size == 0:
        return pd.DataFrame()
    frame = pd.read_csv(summary_path)
    frame = frame[(frame["status"] == "ok") & frame["cer"].notna()]
    if frame.empty:
        return pd.DataFrame()
    cols = [
        "cer",
        "wer",
        "ned",
        "char_accuracy",
        "word_accuracy",
        "diacritic_accuracy",
        "diacritic_error_count",
        "base_loss_count",
    ]
    grouped = frame.groupby(["engine", "stage"], as_index=False)[cols].mean()
    grouped["pages"] = frame.groupby(["engine", "stage"])["item_id"].count().to_numpy()
    return grouped.sort_values(["engine", "stage"])


def write_lines(path: Path, lines: Iterable[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
