from pathlib import Path
from typing import List

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def _prepare_output(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def plot_class_distribution(class_counts: pd.Series, output_path: Path) -> None:
    _prepare_output(output_path)
    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(x=class_counts.index, y=class_counts.values, color="#4477AA")
    ax.set_title("Class distribution after cleaning")
    ax.set_xlabel("Target")
    ax.set_ylabel("Rows")
    ax.bar_label(ax.containers[0], fmt="%d", padding=3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_source_distribution(source_counts: pd.Series, output_path: Path) -> None:
    _prepare_output(output_path)
    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(x=source_counts.index, y=source_counts.values, color="#66AA77")
    ax.set_title("Source distribution after cleaning")
    ax.set_xlabel("Source")
    ax.set_ylabel("Rows")
    ax.bar_label(ax.containers[0], fmt="%d", padding=3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_confusion_matrix(cm: pd.DataFrame, output_path: Path) -> None:
    _prepare_output(output_path)
    plt.figure(figsize=(7.2, 6.0))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Test confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_per_class_f1(report: pd.DataFrame, output_path: Path) -> None:
    _prepare_output(output_path)
    plt.figure(figsize=(8, 4.8))
    ax = sns.barplot(data=report, x="label", y="f1", color="#CC6677")
    ax.set_ylim(0, 1)
    ax.set_title("Per-class F1 on test set")
    ax.set_xlabel("Target")
    ax.set_ylabel("F1")
    ax.bar_label(ax.containers[0], fmt="%.3f", padding=3)
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_threshold_curve(sweep: pd.DataFrame, output_path: Path) -> None:
    _prepare_output(output_path)
    plt.figure(figsize=(8, 4.8))
    for column in ["precision", "recall", "f1", "flagged_rate"]:
        plt.plot(sweep["threshold"], sweep[column], label=column)
    plt.title("Validation threshold sweep for toxic / other")
    plt.xlabel("Toxic threshold")
    plt.ylabel("Score")
    plt.ylim(0, 1.02)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


def plot_reliability(
    probs: np.ndarray,
    y_true: np.ndarray,
    output_path: Path,
    n_bins: int = 10,
) -> None:
    _prepare_output(output_path)
    confidences = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y_true).astype(float)
    bins = np.linspace(0, 1, n_bins + 1)
    rows = []
    for low, high in zip(bins[:-1], bins[1:]):
        if high == 1.0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        if np.any(mask):
            rows.append(
                {
                    "confidence": float(confidences[mask].mean()),
                    "accuracy": float(correct[mask].mean()),
                    "count": int(mask.sum()),
                }
            )
    frame = pd.DataFrame(rows)
    plt.figure(figsize=(5.8, 5.2))
    plt.plot([0, 1], [0, 1], linestyle="--", color="#555555", label="ideal")
    if not frame.empty:
        plt.scatter(
            frame["confidence"],
            frame["accuracy"],
            s=np.maximum(frame["count"] / frame["count"].max(), 0.08) * 500,
            color="#4477AA",
            alpha=0.85,
            label="bins",
        )
    plt.title("Top-label reliability on test set")
    plt.xlabel("Mean confidence")
    plt.ylabel("Empirical accuracy")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()
