import argparse
import inspect
import json
import math
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support

from src.config import LABEL_ORDER, RESULTS_DIR, SEED, TARGET_COLUMN, TEXT_COLUMN
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
    plot_confusion_matrix,
    plot_per_class_f1,
    plot_reliability,
    plot_threshold_curve,
)


MODEL_REGISTRY: Dict[str, Dict[str, object]] = {
    "berturk": {
        "model_name": "dbmdz/bert-base-turkish-cased",
        "display_name": "BERTurk",
        "max_length": 128,
        "batch_size": 8,
        "gradient_accumulation": 2,
    },
    "electra": {
        "model_name": "dbmdz/electra-base-turkish-cased-discriminator",
        "display_name": "Turkish ELECTRA",
        "max_length": 128,
        "batch_size": 8,
        "gradient_accumulation": 2,
    },
    "xlm_r": {
        "model_name": "FacebookAI/xlm-roberta-base",
        "display_name": "XLM-RoBERTa base",
        "max_length": 128,
        "batch_size": 6,
        "gradient_accumulation": 3,
    },
    "turkishbertweet": {
        "model_name": "VRLLab/TurkishBERTweet",
        "display_name": "TurkishBERTweet",
        "max_length": 128,
        "batch_size": 8,
        "gradient_accumulation": 2,
    },
    "convberturk": {
        "model_name": "dbmdz/convbert-base-turkish-cased",
        "display_name": "ConvBERTurk",
        "max_length": 128,
        "batch_size": 8,
        "gradient_accumulation": 2,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Turkish transformer classifier")
    parser.add_argument("--model-key", choices=sorted(MODEL_REGISTRY), default="berturk")
    parser.add_argument("--model-name", default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--preprocess", choices=["raw", "light", "light_masked"], default="light")
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--gradient-accumulation", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--max-length", type=int, default=None)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--no-class-weight", action="store_true")
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--save-model", action="store_true")
    return parser.parse_args()


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=1)
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "weighted_f1": float(f1_score(labels, predictions, average="weighted", zero_division=0)),
    }


def safe_json(value):
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def labels_to_ids(labels: pd.Series) -> np.ndarray:
    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    return labels.map(label_to_id).to_numpy(dtype=int)


def get_training_args_class_kwargs(training_args_cls, kwargs: Dict[str, object]) -> Dict[str, object]:
    signature = inspect.signature(training_args_cls.__init__)
    params = signature.parameters
    translated = dict(kwargs)
    if "eval_strategy" in params:
        translated["eval_strategy"] = translated.pop("evaluation_strategy")
    elif "evaluation_strategy" not in params:
        translated.pop("evaluation_strategy", None)
    return {key: value for key, value in translated.items() if key in params}


def get_trainer_kwargs(trainer_cls, kwargs: Dict[str, object]) -> Dict[str, object]:
    params = inspect.signature(trainer_cls.__init__).parameters
    translated = dict(kwargs)
    if "processing_class" in params and "tokenizer" in translated:
        translated["processing_class"] = translated.pop("tokenizer")
    return {key: value for key, value in translated.items() if key in params}


def write_status(output_dir: Path, status: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "status.json").write_text(
        json.dumps(status, indent=2, ensure_ascii=False, default=safe_json),
        encoding="utf-8",
    )


def main() -> None:
    try:
        from datasets import Dataset
        from transformers import (
            AutoModelForSequenceClassification,
            AutoTokenizer,
            DataCollatorWithPadding,
            Trainer,
            TrainingArguments,
            set_seed,
        )
    except ImportError as exc:
        raise SystemExit(
            "Missing transformer dependencies. Install them with: "
            "python -m pip install transformers datasets accelerate sentencepiece safetensors"
        ) from exc

    args = parse_args()
    start_time = time.time()
    registry = MODEL_REGISTRY[args.model_key]
    model_name = args.model_name or str(registry["model_name"])
    display_name = str(registry["display_name"])
    run_name = args.run_name or f"{args.model_key}_{args.preprocess}"
    output_dir = args.output_dir or (RESULTS_DIR / "transformers" / run_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "plots").mkdir(parents=True, exist_ok=True)

    batch_size = args.batch_size or int(registry["batch_size"])
    grad_accum = args.gradient_accumulation or int(registry["gradient_accumulation"])
    max_length = args.max_length or int(registry["max_length"])
    use_class_weight = not args.no_class_weight

    write_status(
        output_dir,
        {
            "status": "starting",
            "run_name": run_name,
            "model_key": args.model_key,
            "model_name": model_name,
            "preprocess": args.preprocess,
            "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )

    set_seed(SEED)
    raw_df = load_raw_dataframe(force_download=args.force_download)
    clean_df, audit = clean_dataset(raw_df)
    clean_df = stratified_sample(clean_df, args.sample_size, seed=SEED)
    if args.sample_size is not None:
        audit["sample_size_used"] = int(len(clean_df))
    splits = make_stratified_splits(clean_df, seed=SEED)
    label_to_id = {label: idx for idx, label in enumerate(LABEL_ORDER)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}

    for split_name, split_df in splits.items():
        split_df = split_df[[TEXT_COLUMN, TARGET_COLUMN]].copy()
        split_df[TEXT_COLUMN] = apply_preprocessing(split_df[TEXT_COLUMN], args.preprocess)
        split_df["labels"] = split_df[TARGET_COLUMN].map(label_to_id).astype(int)
        splits[split_name] = split_df

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token or tokenizer.unk_token

    def tokenize(batch):
        return tokenizer(
            batch[TEXT_COLUMN],
            truncation=True,
            max_length=max_length,
        )

    dataset = {
        split_name: Dataset.from_pandas(split_df, preserve_index=False).map(
            tokenize,
            batched=True,
            remove_columns=[TEXT_COLUMN, TARGET_COLUMN],
        )
        for split_name, split_df in splits.items()
    }

    class_weights: Optional[torch.Tensor] = None
    if use_class_weight:
        train_labels = splits["train"]["labels"].to_numpy()
        class_counts = np.bincount(train_labels, minlength=len(LABEL_ORDER))
        weights = class_counts.sum() / np.maximum(class_counts, 1)
        weights = weights / weights.mean()
        class_weights = torch.tensor(weights, dtype=torch.float)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(LABEL_ORDER),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    if getattr(model.config, "pad_token_id", None) is None and tokenizer.pad_token_id is not None:
        model.config.pad_token_id = tokenizer.pad_token_id

    class WeightedTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
            labels = inputs.pop("labels")
            outputs = model(**inputs)
            if class_weights is None:
                loss_fct = torch.nn.CrossEntropyLoss()
            else:
                loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights.to(outputs.logits.device))
            loss = loss_fct(outputs.logits, labels)
            return (loss, outputs) if return_outputs else loss

    training_kwargs = {
        "output_dir": str(output_dir / "checkpoints"),
        "evaluation_strategy": "epoch",
        "save_strategy": "no" if not args.save_model else "epoch",
        "learning_rate": args.learning_rate,
        "per_device_train_batch_size": batch_size,
        "per_device_eval_batch_size": max(batch_size * 2, 1),
        "gradient_accumulation_steps": grad_accum,
        "num_train_epochs": args.epochs,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "fp16": torch.cuda.is_available(),
        "report_to": [],
        "logging_dir": str(output_dir / "trainer_logs"),
        "logging_steps": 50,
        "seed": SEED,
        "data_seed": SEED,
        "save_total_limit": 1,
    }
    training_args = TrainingArguments(
        **get_training_args_class_kwargs(TrainingArguments, training_kwargs)
    )

    trainer = WeightedTrainer(
        **get_trainer_kwargs(
            WeightedTrainer,
            {
                "model": model,
                "args": training_args,
                "train_dataset": dataset["train"],
                "eval_dataset": dataset["val"],
                "tokenizer": tokenizer,
                "data_collator": DataCollatorWithPadding(tokenizer=tokenizer),
                "compute_metrics": compute_metrics,
            },
        )
    )

    write_status(output_dir, {"status": "training", "run_name": run_name, "model_name": model_name})
    trainer.train()
    write_status(output_dir, {"status": "predicting", "run_name": run_name, "model_name": model_name})

    val_predictions = trainer.predict(dataset["val"])
    test_predictions = trainer.predict(dataset["test"])
    val_logits = val_predictions.predictions
    test_logits = test_predictions.predictions
    y_val = val_predictions.label_ids.astype(int)
    y_test = test_predictions.label_ids.astype(int)

    temperature, val_nll = temperature_grid_search(val_logits, y_val)
    probs_val = softmax(val_logits, temperature=temperature)
    probs_test = softmax(test_logits, temperature=temperature)
    uncalibrated_val = softmax(val_logits)
    uncalibrated_test = softmax(test_logits)

    val_metrics = multiclass_metrics(y_val, probs_val, LABEL_ORDER)
    test_metrics = multiclass_metrics(y_test, probs_test, LABEL_ORDER)
    val_uncalibrated_metrics = multiclass_metrics(y_val, uncalibrated_val, LABEL_ORDER)
    test_uncalibrated_metrics = multiclass_metrics(y_test, uncalibrated_test, LABEL_ORDER)

    per_class = per_class_report(y_test, probs_test, LABEL_ORDER)
    cm = confusion_frame(y_test, probs_test, LABEL_ORDER)
    other_idx = LABEL_ORDER.index("OTHER")
    y_val_binary = (y_val != other_idx).astype(int)
    y_test_binary = (y_test != other_idx).astype(int)
    toxic_probs_val = 1.0 - probs_val[:, other_idx]
    toxic_probs_test = 1.0 - probs_test[:, other_idx]
    sweep_val = threshold_sweep(y_val_binary, toxic_probs_val)
    policies = choose_threshold_policies(sweep_val)
    policy_rows = []
    for policy_name, threshold in policies.items():
        row = threshold_sweep(y_test_binary, toxic_probs_test, thresholds=[threshold]).iloc[0].to_dict()
        row["policy"] = policy_name
        policy_rows.append(row)
    policy_test = pd.DataFrame(policy_rows)

    np.save(output_dir / "val_logits.npy", val_logits)
    np.save(output_dir / "test_logits.npy", test_logits)
    np.save(output_dir / "val_labels.npy", y_val)
    np.save(output_dir / "test_labels.npy", y_test)
    per_class.to_csv(output_dir / "per_class_report.csv", index=False)
    cm.to_csv(output_dir / "confusion_matrix.csv")
    sweep_val.to_csv(output_dir / "threshold_sweep_val.csv", index=False)
    policy_test.to_csv(output_dir / "threshold_policies_test.csv", index=False)

    plot_confusion_matrix(cm, output_dir / "confusion_matrix.png")
    plot_per_class_f1(per_class, output_dir / "per_class_f1.png")
    plot_threshold_curve(sweep_val, output_dir / "threshold_sweep_val.png")
    plot_reliability(probs_test, y_test, output_dir / "reliability_test.png")

    if args.save_model:
        trainer.save_model(str(output_dir / "best_model"))
        tokenizer.save_pretrained(str(output_dir / "best_model"))

    metrics_payload = {
        "run_name": run_name,
        "family": "transformer",
        "model_key": args.model_key,
        "model_name": model_name,
        "display_name": display_name,
        "preprocess": args.preprocess,
        "label_order": LABEL_ORDER,
        "seed": SEED,
        "sample_size": args.sample_size,
        "epochs": args.epochs,
        "batch_size": batch_size,
        "gradient_accumulation": grad_accum,
        "effective_batch_size": batch_size * grad_accum,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "warmup_ratio": args.warmup_ratio,
        "max_length": max_length,
        "class_weight": use_class_weight,
        "class_weights": class_weights.cpu().numpy().tolist() if class_weights is not None else None,
        "temperature": temperature,
        "val_nll_after_temperature": val_nll,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "val_uncalibrated_metrics": val_uncalibrated_metrics,
        "test_uncalibrated_metrics": test_uncalibrated_metrics,
        "dataset_audit": audit,
        "runtime_seconds": time.time() - start_time,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, indent=2, ensure_ascii=False, default=safe_json),
        encoding="utf-8",
    )
    write_status(
        output_dir,
        {
            "status": "completed",
            "run_name": run_name,
            "model_name": model_name,
            "test_macro_f1": test_metrics["macro_f1"],
            "runtime_seconds": time.time() - start_time,
            "completed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    print(
        f"Done {run_name}: test_macro_f1={test_metrics['macro_f1']:.4f}, "
        f"test_acc={test_metrics['accuracy']:.4f}, temperature={temperature:.2f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
