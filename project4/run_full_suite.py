import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List

from src.config import RESULTS_DIR


FULL_MODEL_KEYS = ["berturk", "electra", "xlm_r", "turkishbertweet", "convberturk"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Project 4 transformer suite")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--skip-hpo", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--logs-dir", type=Path, default=RESULTS_DIR / "logs")
    return parser.parse_args()


def write_status(payload: Dict[str, object]) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "run_status.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def read_hpo_config(skip_hpo: bool) -> Dict[str, object]:
    hpo_dir = RESULTS_DIR / "hpo"
    best_path = hpo_dir / "hpo_best.json"
    if not skip_hpo and not best_path.exists():
        run_command(
            [
                sys.executable,
                "hpo_transformer.py",
            ],
            RESULTS_DIR / "logs" / "hpo_suite.log",
            stage="hpo",
            run_name="hpo",
        )
    if best_path.exists():
        return json.loads(best_path.read_text(encoding="utf-8"))
    return {
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.06,
        "class_weight": True,
        "selection_rule": "default config because HPO was skipped or unavailable",
    }


def run_command(command: List[str], log_path: Path, stage: str, run_name: str) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    write_status(
        {
            "status": "running",
            "stage": stage,
            "run_name": run_name,
            "command": command,
            "log_path": str(log_path),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    env = os.environ.copy()
    env.setdefault(
        "NODE_PATH",
        r"C:\Users\Arda Karakaş\.cache\codex-runtimes\codex-primary-runtime"
        r"\dependencies\node\node_modules",
    )
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=env)
    return process.returncode


def run_transformer(
    model_key: str,
    preprocess: str,
    hpo: Dict[str, object],
    args: argparse.Namespace,
    retry_oom: bool = True,
) -> None:
    run_name = f"{model_key}_{preprocess}"
    output_dir = RESULTS_DIR / "transformers" / run_name
    if args.skip_existing and (output_dir / "metrics.json").exists():
        print(f"Skipping existing {run_name}", flush=True)
        return
    command = [
        sys.executable,
        "train_transformer.py",
        "--model-key",
        model_key,
        "--run-name",
        run_name,
        "--preprocess",
        preprocess,
        "--epochs",
        str(args.epochs),
        "--learning-rate",
        str(hpo["learning_rate"]),
        "--weight-decay",
        str(hpo["weight_decay"]),
        "--warmup-ratio",
        str(hpo["warmup_ratio"]),
        "--output-dir",
        str(output_dir),
    ]
    if args.sample_size is not None:
        command.extend(["--sample-size", str(args.sample_size)])
    if not hpo.get("class_weight", True):
        command.append("--no-class-weight")
    log_path = args.logs_dir / f"{run_name}.log"
    rc = run_command(command, log_path, stage="train", run_name=run_name)
    if rc == 0:
        return
    log_text = log_path.read_text(encoding="utf-8", errors="ignore").lower()
    if retry_oom and ("out of memory" in log_text or "cuda" in log_text and "memory" in log_text):
        retry_command = command + ["--batch-size", "4", "--gradient-accumulation", "4"]
        retry_log = args.logs_dir / f"{run_name}_retry_oom.log"
        rc = run_command(retry_command, retry_log, stage="train_retry_oom", run_name=run_name)
    elif any(
        marker in log_text
        for marker in [
            "readerror",
            "connection",
            "forcibly closed",
            "can't load the model",
            "hf_hub_download",
            "varolan bir",
        ]
    ):
        retry_log = args.logs_dir / f"{run_name}_retry_download.log"
        rc = run_command(command, retry_log, stage="train_retry_download", run_name=run_name)
    if rc != 0:
        raise RuntimeError(f"Training failed for {run_name}. See {log_path}")


def main() -> None:
    args = parse_args()
    args.logs_dir.mkdir(parents=True, exist_ok=True)
    hpo = read_hpo_config(args.skip_hpo)
    (RESULTS_DIR / "selected_hyperparameters.json").write_text(
        json.dumps(hpo, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for model_key in FULL_MODEL_KEYS:
        run_transformer(model_key, "light", hpo, args)
    run_transformer("berturk", "raw", hpo, args)

    for command, stage in [
        ([sys.executable, "build_final_report.py"], "final_report"),
        ([sys.executable, "build_presentation.py"], "presentation_assets"),
    ]:
        rc = run_command(command, args.logs_dir / f"{stage}.log", stage=stage, run_name=stage)
        if rc != 0:
            raise RuntimeError(f"{stage} failed")

    node_path = (
        r"C:\Users\Arda Karakaş\.cache\codex-runtimes\codex-primary-runtime"
        r"\dependencies\node\bin\node.exe"
    )
    deck_script = Path("presentation") / "build_deck.mjs"
    if deck_script.exists():
        rc = run_command([node_path, str(deck_script)], args.logs_dir / "deck_export.log", "deck_export", "deck")
        if rc != 0:
            raise RuntimeError("Deck export failed")

    write_status(
        {
            "status": "completed",
            "stage": "suite",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )


if __name__ == "__main__":
    main()
