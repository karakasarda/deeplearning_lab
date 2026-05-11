import json
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import LABEL_ORDER, RESULTS_DIR


KEY_METRICS = [
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "macro_pr_auc",
    "log_loss",
    "top_label_ece",
    "classwise_ece",
]


def format_table_value(column: str, value: object, floatfmt: str) -> str:
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return ""
        if column == "learning_rate":
            return f"{value:.0e}"
        if abs(value) < 1e-4 and value != 0:
            return f"{value:.2e}"
        return format(value, floatfmt)
    return str(value)


def markdown_table(frame: pd.DataFrame, floatfmt: str = ".4f") -> str:
    if frame.empty:
        return "_No rows available._"
    headers = [str(col) for col in frame.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column, value in row.items():
            values.append(format_table_value(str(column), value, floatfmt))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def load_json(path: Path) -> Dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def baseline_rows() -> List[Dict[str, object]]:
    path = RESULTS_DIR / "experiment_summary.csv"
    if not path.exists():
        return []
    frame = pd.read_csv(path)
    rows: List[Dict[str, object]] = []
    for _, row in frame.iterrows():
        out = {
            "run_name": f"baseline_sgd_{row['variant']}",
            "family": "linear_baseline",
            "model": "TF-IDF + SGD log-loss",
            "model_key": "baseline_sgd",
            "preprocess": row["variant"],
            "epochs": 0,
            "learning_rate": np.nan,
            "weight_decay": np.nan,
            "warmup_ratio": np.nan,
            "class_weight": True,
            "temperature": row.get("temperature", np.nan),
            "output_dir": str(RESULTS_DIR / f"variant_{row['variant']}"),
        }
        for metric in KEY_METRICS:
            col = f"test_{metric}"
            out[f"test_{metric}"] = row[col] if col in row else np.nan
            val_col = f"val_{metric}"
            out[f"val_{metric}"] = row[val_col] if val_col in row else np.nan
        rows.append(out)
    return rows


def transformer_rows() -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    root = RESULTS_DIR / "transformers"
    if not root.exists():
        return rows
    for metrics_path in sorted(root.glob("*/metrics.json")):
        metrics = load_json(metrics_path)
        test_metrics = metrics["test_metrics"]
        val_metrics = metrics["val_metrics"]
        row = {
            "run_name": metrics["run_name"],
            "family": "transformer",
            "model": metrics["display_name"],
            "model_key": metrics["model_key"],
            "preprocess": metrics["preprocess"],
            "epochs": metrics["epochs"],
            "learning_rate": metrics["learning_rate"],
            "weight_decay": metrics["weight_decay"],
            "warmup_ratio": metrics["warmup_ratio"],
            "class_weight": metrics["class_weight"],
            "temperature": metrics["temperature"],
            "runtime_seconds": metrics["runtime_seconds"],
            "output_dir": str(metrics_path.parent),
        }
        for metric in KEY_METRICS:
            row[f"test_{metric}"] = test_metrics.get(metric, np.nan)
            row[f"val_{metric}"] = val_metrics.get(metric, np.nan)
        rows.append(row)
    return rows


def plot_normalized_confusion(cm_path: Path, output_path: Path) -> Optional[Path]:
    if not cm_path.exists():
        return None
    cm = pd.read_csv(cm_path, index_col=0)
    normalized = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    normalized.to_csv(output_path.with_suffix(".csv"))
    plt.figure(figsize=(7.2, 6.0))
    sns.heatmap(normalized, annot=True, fmt=".2f", cmap="Blues", cbar=False, vmin=0, vmax=1)
    plt.title("Normalized confusion matrix")
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()
    return output_path


def plot_model_comparison(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        return
    top = frame.sort_values("test_macro_f1", ascending=False).copy()
    top["label"] = top["model"] + "\n" + top["preprocess"].astype(str)
    plt.figure(figsize=(11, 5.8))
    ax = sns.barplot(data=top, x="label", y="test_macro_f1", hue="family", dodge=False)
    ax.set_ylim(0, max(1.0, top["test_macro_f1"].max() + 0.08))
    ax.set_title("Model comparison by test macro-F1")
    ax.set_xlabel("")
    ax.set_ylabel("Test macro-F1")
    ax.bar_label(ax.containers[0], fmt="%.3f", padding=3)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def image_link(path: Path) -> str:
    rel = path.relative_to(RESULTS_DIR.parent).as_posix()
    return f"![{path.stem}]({rel})"


def find_run_file(row: pd.Series, filename: str) -> Optional[Path]:
    output_dir = Path(str(row["output_dir"]))
    path = output_dir / filename
    return path if path.exists() else None


def confusion_reading(cm_path: Optional[Path]) -> List[str]:
    if cm_path is None or not cm_path.exists():
        return []
    cm = pd.read_csv(cm_path, index_col=0)
    normalized = cm.div(cm.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    notes = []
    off_diag = []
    for true_label in normalized.index:
        for pred_label in normalized.columns:
            if true_label == pred_label:
                continue
            rate = float(normalized.loc[true_label, pred_label])
            count = int(cm.loc[true_label, pred_label])
            if rate > 0:
                off_diag.append((rate, count, true_label, pred_label))
    for rate, count, true_label, pred_label in sorted(off_diag, reverse=True)[:5]:
        notes.append(f"- `{true_label}` -> `{pred_label}`: `{count}` örnek, normalized oran `{rate:.2f}`.")
    if "SEXIST" in normalized.index:
        sexist_diag = float(normalized.loc["SEXIST", "SEXIST"])
        notes.append(f"- `SEXIST` sınıfında doğru sınıf oranı `{sexist_diag:.2f}`; düşük destek nedeniyle bu sınıf ayrı izlenmeli.")
    return notes


def build_markdown(frame: pd.DataFrame, best: pd.Series) -> str:
    audit_path = RESULTS_DIR / "audit_summary.json"
    audit = load_json(audit_path) if audit_path.exists() else {}
    hpo_summary = RESULTS_DIR / "hpo" / "hpo_summary.csv"
    hpo_best = RESULTS_DIR / "hpo" / "hpo_best.json"

    metric_cols = [
        "run_name",
        "family",
        "model",
        "preprocess",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_f1",
        "test_weighted_f1",
        "test_macro_pr_auc",
        "test_log_loss",
        "test_top_label_ece",
        "test_classwise_ece",
    ]
    table = markdown_table(frame.sort_values("test_macro_f1", ascending=False)[metric_cols])
    best_baseline = frame[frame["family"] == "linear_baseline"].sort_values(
        "test_macro_f1", ascending=False
    ).iloc[0]
    best_transformer = frame[frame["family"] == "transformer"].sort_values(
        "test_macro_f1", ascending=False
    ).iloc[0]
    lines = [
        "# Project 4 Detailed Report",
        "",
        "## 1. Dataset audit and protocol",
        f"- Raw rows: `{audit.get('raw_rows', 'unknown')}`",
        f"- Clean rows: `{audit.get('clean_rows', 'unknown')}`",
        f"- Class distribution: `{audit.get('class_distribution', {})}`",
        "- Split: stratified `70/15/15`, seed `42`.",
        "- Primary task: five-class single-label classification.",
        "- Operational task: `OTHER` vs toxic threshold selection.",
        "",
        "## 2. Hyperparameter optimization",
    ]
    if hpo_summary.exists():
        hpo_table = markdown_table(pd.read_csv(hpo_summary))
        lines.extend([hpo_table, ""])
    else:
        lines.append("HPO has not completed yet. The default transformer setting is documented in the README.")
    if hpo_best.exists():
        best_hpo = load_json(hpo_best)
        lines.append(f"Selected config: `{best_hpo}`")
    lines.extend(
        [
            "",
            "## 3. Model comparison",
            table,
            "",
            "Main reading: macro-F1 is the primary metric because the `SEXIST` class is much smaller than the other labels. Accuracy is reported only as supporting context.",
            f"- Best transformer: `{best_transformer['run_name']}` with macro-F1 `{best_transformer['test_macro_f1']:.4f}`.",
            f"- Best linear baseline: `{best_baseline['run_name']}` with macro-F1 `{best_baseline['test_macro_f1']:.4f}`.",
            f"- Absolute macro-F1 lift over the best baseline: `{best_transformer['test_macro_f1'] - best_baseline['test_macro_f1']:.4f}`.",
            "",
            "## 4. Best model class-level behavior",
            f"Best run by test macro-F1: `{best['run_name']}` with macro-F1 `{best['test_macro_f1']:.4f}`.",
        ]
    )
    per_class = find_run_file(best, "per_class_report.csv")
    if per_class:
        lines.append(markdown_table(pd.read_csv(per_class)))
    raw_cm = find_run_file(best, "confusion_matrix.png")
    norm_cm = find_run_file(best, "confusion_matrix_normalized.png")
    if raw_cm:
        lines.extend(["", "Raw confusion matrix:", image_link(raw_cm)])
    if norm_cm:
        lines.extend(["", "Normalized confusion matrix:", image_link(norm_cm)])
    notes = confusion_reading(find_run_file(best, "confusion_matrix.csv"))
    if notes:
        lines.extend(["", "Confusion matrix reading:", *notes])
    lines.extend(
        [
            "",
            "## 5. Calibration and threshold policies",
            "All neural runs use validation-based temperature scaling before test metrics are reported. Threshold policies are selected on validation and evaluated on the held-out test set.",
        ]
    )
    policy_file = find_run_file(best, "threshold_policies_test.csv")
    if policy_file:
        policy_frame = pd.read_csv(policy_file)
        lines.append(markdown_table(policy_frame))
        if {"policy", "threshold", "precision", "recall", "f1", "flagged_rate"}.issubset(policy_frame.columns):
            for _, row in policy_frame.iterrows():
                lines.append(
                    f"- `{row['policy']}`: threshold `{row['threshold']:.2f}`, precision `{row['precision']:.4f}`, recall `{row['recall']:.4f}`, F1 `{row['f1']:.4f}`, flagged rate `{row['flagged_rate']:.4f}`."
                )
    reliability = find_run_file(best, "reliability_test.png")
    threshold = find_run_file(best, "threshold_sweep_val.png")
    if reliability:
        lines.extend(["", "Reliability plot:", image_link(reliability)])
    if threshold:
        lines.extend(["", "Threshold sweep:", image_link(threshold)])
    lines.extend(
        [
            "",
            "## 6. Confusion matrix appendix",
        ]
    )
    for _, row in frame.sort_values(["family", "run_name"]).iterrows():
        cm = find_run_file(row, "confusion_matrix.png")
        if cm:
            lines.extend([f"### {row['run_name']}", image_link(cm)])
    lines.extend(
        [
            "",
            "## 7. Limitations",
            "- Transformer runs use reduced epochs to keep the full model set feasible on a single laptop GPU.",
            "- HPO is intentionally limited to BERTurk and transferred to the other transformer models.",
            "- The dataset is a combined/pseudo-labeled resource, not a clean single-source benchmark.",
            "- `SEXIST` remains the most fragile class because its support is far lower than the other labels.",
            "",
            "## 8. Sources",
            "- Dataset: https://huggingface.co/datasets/Overfit-GM/turkish-toxic-language",
            "- BERTurk: https://huggingface.co/dbmdz/bert-base-turkish-cased",
            "- TurkishBERTweet: https://huggingface.co/VRLLab/TurkishBERTweet",
            "- ConvBERTurk: https://huggingface.co/dbmdz/convbert-base-turkish-cased",
            "- XLM-R: https://huggingface.co/FacebookAI/xlm-roberta-base",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    rows = baseline_rows() + transformer_rows()
    if not rows:
        raise RuntimeError("No results found")
    frame = pd.DataFrame(rows)
    frame.to_csv(RESULTS_DIR / "final_model_comparison.csv", index=False)
    plot_model_comparison(frame, RESULTS_DIR / "plots" / "final_model_comparison.png")
    best = frame.sort_values("test_macro_f1", ascending=False).iloc[0]
    cm_csv = find_run_file(best, "confusion_matrix.csv")
    if cm_csv:
        plot_normalized_confusion(cm_csv, cm_csv.parent / "confusion_matrix_normalized.png")
    report = build_markdown(frame, best)
    (RESULTS_DIR / "detailed_report.md").write_text(report, encoding="utf-8")
    print(f"Wrote {RESULTS_DIR / 'final_model_comparison.csv'}")
    print(f"Wrote {RESULTS_DIR / 'detailed_report.md'}")


if __name__ == "__main__":
    main()
