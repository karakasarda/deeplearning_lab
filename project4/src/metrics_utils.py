from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import label_binarize


def softmax(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    scaled = logits / max(float(temperature), 1e-8)
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def temperature_grid_search(
    logits: np.ndarray,
    y_true: np.ndarray,
    temperatures: Iterable[float] = np.linspace(0.5, 5.0, 91),
) -> Tuple[float, float]:
    best_t = 1.0
    best_loss = float("inf")
    labels = np.arange(logits.shape[1])
    for temp in temperatures:
        probs = softmax(logits, temperature=float(temp))
        loss = log_loss(y_true, probs, labels=labels)
        if loss < best_loss:
            best_t = float(temp)
            best_loss = float(loss)
    return best_t, best_loss


def top_label_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = (predictions == y_true).astype(float)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for low, high in zip(bin_edges[:-1], bin_edges[1:]):
        if high == 1.0:
            mask = (confidences >= low) & (confidences <= high)
        else:
            mask = (confidences >= low) & (confidences < high)
        if not np.any(mask):
            continue
        ece += (mask.mean()) * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def classwise_ece(probs: np.ndarray, y_true: np.ndarray, n_bins: int = 15) -> float:
    n_classes = probs.shape[1]
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    eces: List[float] = []
    for class_idx in range(n_classes):
        confidences = probs[:, class_idx]
        labels = (y_true == class_idx).astype(float)
        ece = 0.0
        for low, high in zip(bin_edges[:-1], bin_edges[1:]):
            if high == 1.0:
                mask = (confidences >= low) & (confidences <= high)
            else:
                mask = (confidences >= low) & (confidences < high)
            if not np.any(mask):
                continue
            ece += mask.mean() * abs(labels[mask].mean() - confidences[mask].mean())
        eces.append(float(ece))
    return float(np.mean(eces))


def multiclass_metrics(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: List[str],
) -> Dict[str, float]:
    labels = np.arange(len(label_names))
    y_pred = probs.argmax(axis=1)
    y_binary = label_binarize(y_true, classes=labels)
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "macro_precision": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )[0]
        ),
        "macro_recall": float(
            precision_recall_fscore_support(
                y_true, y_pred, average="macro", zero_division=0
            )[1]
        ),
        "log_loss": float(log_loss(y_true, probs, labels=labels)),
        "top_label_ece": top_label_ece(probs, y_true),
        "classwise_ece": classwise_ece(probs, y_true),
    }
    try:
        metrics["macro_pr_auc"] = float(
            average_precision_score(y_binary, probs, average="macro")
        )
    except ValueError:
        metrics["macro_pr_auc"] = float("nan")
    return metrics


def per_class_report(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: List[str],
) -> pd.DataFrame:
    y_pred = probs.argmax(axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=np.arange(len(label_names)),
        zero_division=0,
    )
    return pd.DataFrame(
        {
            "label": label_names,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support,
        }
    )


def confusion_frame(
    y_true: np.ndarray,
    probs: np.ndarray,
    label_names: List[str],
) -> pd.DataFrame:
    y_pred = probs.argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(label_names)))
    return pd.DataFrame(matrix, index=label_names, columns=label_names)


def threshold_sweep(
    y_true_binary: np.ndarray,
    toxic_probs: np.ndarray,
    thresholds: Iterable[float] = np.linspace(0.05, 0.95, 91),
) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        y_pred = (toxic_probs >= threshold).astype(int)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true_binary,
            y_pred,
            average="binary",
            zero_division=0,
        )
        rows.append(
            {
                "threshold": float(threshold),
                "accuracy": float(accuracy_score(y_true_binary, y_pred)),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "flagged_rate": float(y_pred.mean()),
            }
        )
    return pd.DataFrame(rows)


def choose_threshold_policies(sweep: pd.DataFrame, min_recall: float = 0.90) -> Dict[str, float]:
    best_f1_idx = sweep["f1"].idxmax()
    policies = {"best_f1": float(sweep.loc[best_f1_idx, "threshold"])}
    recall_ok = sweep[sweep["recall"] >= min_recall]
    if not recall_ok.empty:
        recall_idx = recall_ok["precision"].idxmax()
    else:
        recall_idx = sweep["recall"].idxmax()
    policies["recall_floor_90"] = float(sweep.loc[recall_idx, "threshold"])
    return policies
