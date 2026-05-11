import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.config import RESULTS_DIR


TRIALS: List[Dict[str, object]] = [
    {"learning_rate": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.06, "class_weight": True},
    {"learning_rate": 3e-5, "weight_decay": 0.01, "warmup_ratio": 0.06, "class_weight": True},
    {"learning_rate": 1e-5, "weight_decay": 0.01, "warmup_ratio": 0.06, "class_weight": True},
    {"learning_rate": 2e-5, "weight_decay": 0.00, "warmup_ratio": 0.00, "class_weight": True},
    {"learning_rate": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.10, "class_weight": True},
    {"learning_rate": 2e-5, "weight_decay": 0.01, "warmup_ratio": 0.06, "class_weight": False},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Limited BERTurk HPO for Project 4")
    parser.add_argument("--sample-size", type=int, default=12000)
    parser.add_argument("--epochs", type=float, default=0.5)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR / "hpo")
    parser.add_argument("--logs-dir", type=Path, default=RESULTS_DIR / "logs")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def run_trial(args: argparse.Namespace, trial_idx: int, trial: Dict[str, object]) -> Dict[str, object]:
    run_name = f"hpo_berturk_trial_{trial_idx:02d}"
    trial_dir = args.output_dir / run_name
    log_path = args.logs_dir / f"{run_name}.log"
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    if trial_dir.exists() and (trial_dir / "metrics.json").exists() and not args.force:
        metrics = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))
        return summarize_trial(trial_idx, run_name, trial, metrics, skipped=True)

    command = [
        sys.executable,
        "train_transformer.py",
        "--model-key",
        "berturk",
        "--run-name",
        run_name,
        "--preprocess",
        "light",
        "--sample-size",
        str(args.sample_size),
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(trial["learning_rate"]),
        "--weight-decay",
        str(trial["weight_decay"]),
        "--warmup-ratio",
        str(trial["warmup_ratio"]),
        "--output-dir",
        str(trial_dir),
    ]
    if not trial["class_weight"]:
        command.append("--no-class-weight")

    started = time.time()
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True)
    if process.returncode != 0:
        return {
            "trial": trial_idx,
            "run_name": run_name,
            "status": "failed",
            "returncode": process.returncode,
            "runtime_seconds": time.time() - started,
            "log_path": str(log_path),
            **trial,
        }
    metrics = json.loads((trial_dir / "metrics.json").read_text(encoding="utf-8"))
    return summarize_trial(trial_idx, run_name, trial, metrics, skipped=False)


def summarize_trial(
    trial_idx: int,
    run_name: str,
    trial: Dict[str, object],
    metrics: Dict[str, object],
    skipped: bool,
) -> Dict[str, object]:
    val_metrics = metrics["val_metrics"]
    return {
        "trial": trial_idx,
        "run_name": run_name,
        "status": "skipped_existing" if skipped else "completed",
        "learning_rate": trial["learning_rate"],
        "weight_decay": trial["weight_decay"],
        "warmup_ratio": trial["warmup_ratio"],
        "class_weight": trial["class_weight"],
        "val_macro_f1": val_metrics["macro_f1"],
        "val_classwise_ece": val_metrics["classwise_ece"],
        "val_top_label_ece": val_metrics["top_label_ece"],
        "runtime_seconds": metrics["runtime_seconds"],
    }


def choose_best(summary: pd.DataFrame) -> Dict[str, object]:
    completed = summary[summary["status"].isin(["completed", "skipped_existing"])].copy()
    if completed.empty:
        raise RuntimeError("No completed HPO trials")
    max_f1 = completed["val_macro_f1"].max()
    close = completed[completed["val_macro_f1"] >= max_f1 - 0.003].copy()
    close = close.sort_values(["val_classwise_ece", "runtime_seconds"], ascending=[True, True])
    best = close.iloc[0].to_dict()
    return {
        "trial": int(best["trial"]),
        "run_name": best["run_name"],
        "learning_rate": float(best["learning_rate"]),
        "weight_decay": float(best["weight_decay"]),
        "warmup_ratio": float(best["warmup_ratio"]),
        "class_weight": bool(best["class_weight"]),
        "selection_rule": "max val_macro_f1; within 0.003 prefer lower classwise_ece, then runtime",
        "val_macro_f1": float(best["val_macro_f1"]),
        "val_classwise_ece": float(best["val_classwise_ece"]),
    }


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for idx, trial in enumerate(TRIALS, start=1):
        print(f"Running HPO trial {idx}/{len(TRIALS)}: {trial}", flush=True)
        row = run_trial(args, idx, trial)
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.output_dir / "hpo_summary.csv", index=False)
    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "hpo_summary.csv", index=False)
    best = choose_best(summary)
    (args.output_dir / "hpo_best.json").write_text(
        json.dumps(best, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Best HPO config: {best}", flush=True)


if __name__ == "__main__":
    main()
