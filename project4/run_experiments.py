import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import (
    LABEL_ORDER,
    RESULTS_DIR,
    SEED,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TOXIC_COLUMN,
)
from src.data_utils import (
    apply_preprocessing,
    clean_dataset,
    load_raw_dataframe,
    make_stratified_splits,
    stratified_sample,
)
from src.metrics_utils import (
    choose_threshold_policies,
    confusion_frame,
    multiclass_metrics,
    per_class_report,
    softmax,
    temperature_grid_search,
    threshold_sweep,
)
from src.plotting import (
    plot_class_distribution,
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_reliability,
    plot_source_distribution,
    plot_threshold_curve,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Project 4 Turkish toxic language detection experiments"
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument(
        "--preprocess",
        nargs="+",
        default=["raw", "light", "light_masked"],
        choices=["raw", "light", "light_masked"],
    )
    parser.add_argument("--model", choices=["sgd", "logreg"], default="sgd")
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--skip-plots", action="store_true")
    return parser.parse_args()


def build_pipeline(model_name: str, seed: int = SEED) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=2,
                    max_features=80000,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
            (
                "char",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=2,
                    max_features=60000,
                    sublinear_tf=True,
                    lowercase=False,
                ),
            ),
        ]
    )
    if model_name == "logreg":
        classifier = LogisticRegression(
            C=2.0,
            max_iter=500,
            class_weight="balanced",
            solver="saga",
            n_jobs=-1,
            random_state=seed,
        )
    else:
        classifier = SGDClassifier(
            loss="log_loss",
            alpha=1e-5,
            penalty="l2",
            max_iter=35,
            early_stopping=True,
            validation_fraction=0.10,
            n_iter_no_change=4,
            class_weight="balanced",
            random_state=seed,
        )
    return Pipeline([("features", features), ("classifier", classifier)])


def labels_to_ids(labels: pd.Series) -> np.ndarray:
    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    return labels.map(label_to_id).to_numpy(dtype=int)


def binary_labels(labels: pd.Series) -> np.ndarray:
    return (labels != "OTHER").astype(int).to_numpy()


def get_logits(pipeline: Pipeline, texts: List[str]) -> np.ndarray:
    classifier = pipeline.named_steps["classifier"]
    logits = pipeline.decision_function(texts)
    if logits.ndim == 1:
        logits = np.column_stack([-logits, logits])
    if hasattr(classifier, "classes_"):
        order = [int(np.where(classifier.classes_ == idx)[0][0]) for idx in range(len(LABEL_ORDER))]
        logits = logits[:, order]
    return logits


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def dataframe_to_markdown(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    headers = [str(column) for column in df.columns]
    rows = []
    for _, row in df.iterrows():
        rendered = []
        for value in row:
            if isinstance(value, float):
                rendered.append(format(value, floatfmt))
            else:
                rendered.append(str(value))
        rows.append(rendered)
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def make_markdown_report(
    output_dir: Path,
    audit: Dict[str, object],
    summary: pd.DataFrame,
    best_variant: str,
    best_metrics: Dict[str, float],
    policies: Dict[str, float],
) -> None:
    report_path = output_dir / "report.md"
    top_rows = summary.sort_values("test_macro_f1", ascending=False).copy()
    metric_table = top_rows[
        [
            "variant",
            "model",
            "temperature",
            "test_accuracy",
            "test_macro_f1",
            "test_macro_pr_auc",
            "test_top_label_ece",
            "test_classwise_ece",
        ]
    ]
    threshold_table = pd.DataFrame(
        [{"policy": key, "threshold": value} for key, value in policies.items()]
    )
    metric_table_md = dataframe_to_markdown(metric_table, floatfmt=".4f")
    threshold_table_md = dataframe_to_markdown(threshold_table, floatfmt=".2f")
    text = f"""# Project 4 Experiment Report

## Dataset audit
- Raw rows: `{audit['raw_rows']}`
- Clean rows after conflict and exact duplicate removal: `{audit['clean_rows']}`
- Duplicate text rows: `{audit['duplicate_text_rows']}`
- Conflicting text count: `{audit['conflicting_text_count']}`
- Class distribution: `{audit['class_distribution']}`

## Experiment summary
{metric_table_md}

## Best linear run
- Best preprocessing variant: `{best_variant}`
- Test macro-F1: `{best_metrics['macro_f1']:.4f}`
- Test accuracy: `{best_metrics['accuracy']:.4f}`
- Test macro PR-AUC: `{best_metrics['macro_pr_auc']:.4f}`
- Top-label ECE after temperature scaling: `{best_metrics['top_label_ece']:.4f}`

## Toxic / other policies selected on validation
{threshold_table_md}

## Generated artifacts
- `audit_summary.json`
- `experiment_summary.csv`
- `best_per_class_report.csv`
- `best_confusion_matrix.csv`
- `threshold_sweep_val.csv`
- `threshold_policies_test.csv`
- `plots/`
"""
    report_path.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    start = time.time()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_dataframe(force_download=args.force_download)
    clean_df, audit = clean_dataset(raw_df)
    clean_df = stratified_sample(clean_df, args.sample_size, seed=SEED)
    if args.sample_size is not None:
        audit["sample_size_used"] = int(len(clean_df))
        audit["sampled_class_distribution"] = {
            label: int((clean_df[TARGET_COLUMN] == label).sum()) for label in LABEL_ORDER
        }

    save_json(output_dir / "audit_summary.json", audit)
    clean_df[TARGET_COLUMN].value_counts().reindex(LABEL_ORDER).to_csv(
        output_dir / "class_distribution.csv", header=["count"]
    )
    clean_df["source"].value_counts().sort_index().to_csv(
        output_dir / "source_distribution.csv", header=["count"]
    )

    if not args.skip_plots:
        plot_class_distribution(
            clean_df[TARGET_COLUMN].value_counts().reindex(LABEL_ORDER),
            output_dir / "plots" / "class_distribution.png",
        )
        plot_source_distribution(
            clean_df["source"].value_counts().sort_index(),
            output_dir / "plots" / "source_distribution.png",
        )

    splits = make_stratified_splits(clean_df, seed=SEED)
    for split_name, split_df in splits.items():
        split_df.to_csv(output_dir / f"{split_name}_split.csv", index=False)

    y_train = labels_to_ids(splits["train"][TARGET_COLUMN])
    y_val = labels_to_ids(splits["val"][TARGET_COLUMN])
    y_test = labels_to_ids(splits["test"][TARGET_COLUMN])
    y_val_binary = binary_labels(splits["val"][TARGET_COLUMN])
    y_test_binary = binary_labels(splits["test"][TARGET_COLUMN])

    summary_rows = []
    variant_payloads = {}

    for variant in args.preprocess:
        variant_start = time.time()
        print(f"Running variant={variant} model={args.model}", flush=True)
        x_train = apply_preprocessing(splits["train"][TEXT_COLUMN], variant).tolist()
        x_val = apply_preprocessing(splits["val"][TEXT_COLUMN], variant).tolist()
        x_test = apply_preprocessing(splits["test"][TEXT_COLUMN], variant).tolist()

        pipeline = build_pipeline(args.model, seed=SEED)
        pipeline.fit(x_train, y_train)

        logits_val = get_logits(pipeline, x_val)
        logits_test = get_logits(pipeline, x_test)
        temperature, val_nll = temperature_grid_search(logits_val, y_val)
        probs_val = softmax(logits_val, temperature=temperature)
        probs_test = softmax(logits_test, temperature=temperature)

        val_metrics = multiclass_metrics(y_val, probs_val, LABEL_ORDER)
        test_metrics = multiclass_metrics(y_test, probs_test, LABEL_ORDER)
        val_uncalibrated = multiclass_metrics(y_val, softmax(logits_val), LABEL_ORDER)
        test_uncalibrated = multiclass_metrics(y_test, softmax(logits_test), LABEL_ORDER)

        per_class = per_class_report(y_test, probs_test, LABEL_ORDER)
        cm = confusion_frame(y_test, probs_test, LABEL_ORDER)
        other_idx = LABEL_ORDER.index("OTHER")
        toxic_probs_val = 1.0 - probs_val[:, other_idx]
        toxic_probs_test = 1.0 - probs_test[:, other_idx]
        sweep_val = threshold_sweep(y_val_binary, toxic_probs_val)
        policies = choose_threshold_policies(sweep_val)
        policy_rows = []
        for policy_name, threshold in policies.items():
            test_sweep_at_threshold = threshold_sweep(
                y_test_binary, toxic_probs_test, thresholds=[threshold]
            )
            row = test_sweep_at_threshold.iloc[0].to_dict()
            row["policy"] = policy_name
            policy_rows.append(row)
        policy_test = pd.DataFrame(policy_rows)

        variant_dir = output_dir / f"variant_{variant}"
        variant_dir.mkdir(parents=True, exist_ok=True)
        per_class.to_csv(variant_dir / "per_class_report.csv", index=False)
        cm.to_csv(variant_dir / "confusion_matrix.csv")
        sweep_val.to_csv(variant_dir / "threshold_sweep_val.csv", index=False)
        policy_test.to_csv(variant_dir / "threshold_policies_test.csv", index=False)

        y_pred = probs_test.argmax(axis=1)
        confidences = probs_test.max(axis=1)
        errors = splits["test"].copy()
        errors["predicted"] = [LABEL_ORDER[idx] for idx in y_pred]
        errors["confidence"] = confidences
        errors = errors[errors[TARGET_COLUMN] != errors["predicted"]].sort_values(
            "confidence", ascending=False
        )
        errors.head(75).to_csv(variant_dir / "high_confidence_errors.csv", index=False)

        if not args.skip_plots:
            plot_confusion_matrix(
                cm, variant_dir / "confusion_matrix.png"
            )
            plot_per_class_f1(
                per_class, variant_dir / "per_class_f1.png"
            )
            plot_threshold_curve(
                sweep_val, variant_dir / "threshold_sweep_val.png"
            )
            plot_reliability(
                probs_test, y_test, variant_dir / "reliability_test.png"
            )

        row = {
            "variant": variant,
            "model": args.model,
            "temperature": temperature,
            "val_nll_after_temperature": val_nll,
            "runtime_seconds": time.time() - variant_start,
        }
        for prefix, metrics in [
            ("val", val_metrics),
            ("test", test_metrics),
            ("val_uncalibrated", val_uncalibrated),
            ("test_uncalibrated", test_uncalibrated),
        ]:
            for key, value in metrics.items():
                row[f"{prefix}_{key}"] = value
        summary_rows.append(row)
        variant_payloads[variant] = {
            "test_metrics": test_metrics,
            "val_metrics": val_metrics,
            "policies": policies,
        }

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(output_dir / "experiment_summary.csv", index=False)
    best_idx = summary["test_macro_f1"].idxmax()
    best_variant = str(summary.loc[best_idx, "variant"])
    best_dir = output_dir / f"variant_{best_variant}"

    for file_name, target_name in [
        ("per_class_report.csv", "best_per_class_report.csv"),
        ("confusion_matrix.csv", "best_confusion_matrix.csv"),
        ("threshold_sweep_val.csv", "threshold_sweep_val.csv"),
        ("threshold_policies_test.csv", "threshold_policies_test.csv"),
    ]:
        source = best_dir / file_name
        if source.exists():
            (output_dir / target_name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    make_markdown_report(
        output_dir=output_dir,
        audit=audit,
        summary=summary,
        best_variant=best_variant,
        best_metrics=variant_payloads[best_variant]["test_metrics"],
        policies=variant_payloads[best_variant]["policies"],
    )

    save_json(
        output_dir / "runtime_summary.json",
        {
            "total_runtime_seconds": time.time() - start,
            "model": args.model,
            "preprocess_variants": args.preprocess,
            "sample_size": args.sample_size,
            "best_variant": best_variant,
        },
    )
    print(f"Done. Best variant: {best_variant}. Results: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
