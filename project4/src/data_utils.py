import re
import urllib.request
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    HF_DATASET_URL,
    LABEL_ORDER,
    RAW_DATA_DIR,
    RAW_DATA_PATH,
    SEED,
    SOURCE_COLUMN,
    TARGET_COLUMN,
    TEXT_COLUMN,
    TOXIC_COLUMN,
)


URL_RE = re.compile(r"https?://\S+|www\.\S+", flags=re.IGNORECASE)
USER_RE = re.compile(r"(?<!\w)@\w+")
HASHTAG_RE = re.compile(r"#(\w+)")
SPACE_RE = re.compile(r"\s+")
REPEAT_CHAR_RE = re.compile(r"([a-zA-ZabcçdefgğhıijklmnoöprsştuüvyzABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZ])\1{2,}")
PROFANITY_RE = re.compile(
    r"\b("
    r"a+\.?m+\.?k+|a+q+|s[iı]kt\w*|s[iı]ker\w*|"
    r"orospu\w*|pi[cç]\w*|g[oö]t\w*|yarr\w*|"
    r"ibne\w*|kaltak\w*|salak\w*|aptal\w*"
    r")\b",
    flags=re.IGNORECASE,
)


def ensure_dirs() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)


def download_dataset(force: bool = False, target_path: Path = RAW_DATA_PATH) -> Path:
    ensure_dirs()
    if target_path.exists() and not force:
        return target_path
    urllib.request.urlretrieve(HF_DATASET_URL, target_path)
    return target_path


def load_raw_dataframe(path: Optional[Path] = None, force_download: bool = False) -> pd.DataFrame:
    csv_path = path if path is not None else download_dataset(force=force_download)
    return pd.read_csv(csv_path)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    required = {TEXT_COLUMN, TARGET_COLUMN, SOURCE_COLUMN, TOXIC_COLUMN}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    return df


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, object]]:
    df = _standardize_columns(df)
    audit: Dict[str, object] = {
        "raw_rows": int(len(df)),
        "raw_columns": list(df.columns),
        "missing_by_column": {c: int(df[c].isna().sum()) for c in df.columns},
    }

    clean = df[[TEXT_COLUMN, TARGET_COLUMN, SOURCE_COLUMN, TOXIC_COLUMN]].copy()
    clean[TEXT_COLUMN] = clean[TEXT_COLUMN].astype("string").fillna("").str.strip()
    clean[TARGET_COLUMN] = clean[TARGET_COLUMN].astype("string").fillna("").str.strip().str.upper()
    clean[SOURCE_COLUMN] = clean[SOURCE_COLUMN].astype("string").fillna("unknown").str.strip()
    clean[TOXIC_COLUMN] = pd.to_numeric(clean[TOXIC_COLUMN], errors="coerce").fillna(0).astype(int)

    before_drop = len(clean)
    clean = clean[(clean[TEXT_COLUMN] != "") & clean[TARGET_COLUMN].isin(LABEL_ORDER)]
    audit["dropped_empty_or_unknown_label"] = int(before_drop - len(clean))

    text_target_counts = clean.groupby(TEXT_COLUMN)[TARGET_COLUMN].nunique()
    conflicting_texts = set(text_target_counts[text_target_counts > 1].index)
    audit["duplicate_text_rows"] = int(clean.duplicated(subset=[TEXT_COLUMN]).sum())
    audit["conflicting_text_count"] = int(len(conflicting_texts))

    if conflicting_texts:
        clean = clean[~clean[TEXT_COLUMN].isin(conflicting_texts)].copy()
    before_dedup = len(clean)
    clean = clean.drop_duplicates(subset=[TEXT_COLUMN], keep="first").reset_index(drop=True)
    audit["deduplicated_rows_removed"] = int(before_dedup - len(clean))
    audit["clean_rows"] = int(len(clean))
    audit["class_distribution"] = {
        label: int((clean[TARGET_COLUMN] == label).sum()) for label in LABEL_ORDER
    }
    audit["source_distribution"] = {
        str(k): int(v) for k, v in clean[SOURCE_COLUMN].value_counts().sort_index().items()
    }
    audit["binary_distribution"] = {
        str(k): int(v) for k, v in clean[TOXIC_COLUMN].value_counts().sort_index().items()
    }
    lengths = clean[TEXT_COLUMN].str.len()
    audit["text_length"] = {
        "min": int(lengths.min()),
        "median": float(lengths.median()),
        "mean": float(lengths.mean()),
        "p95": float(lengths.quantile(0.95)),
        "max": int(lengths.max()),
    }
    return clean, audit


def make_stratified_splits(
    df: pd.DataFrame,
    seed: int = SEED,
    train_size: float = 0.70,
    val_size: float = 0.15,
    test_size: float = 0.15,
) -> Dict[str, pd.DataFrame]:
    if not np.isclose(train_size + val_size + test_size, 1.0):
        raise ValueError("train_size + val_size + test_size must equal 1.0")

    train_df, temp_df = train_test_split(
        df,
        train_size=train_size,
        random_state=seed,
        stratify=df[TARGET_COLUMN],
    )
    relative_test = test_size / (val_size + test_size)
    val_df, test_df = train_test_split(
        temp_df,
        test_size=relative_test,
        random_state=seed,
        stratify=temp_df[TARGET_COLUMN],
    )
    return {
        "train": train_df.reset_index(drop=True),
        "val": val_df.reset_index(drop=True),
        "test": test_df.reset_index(drop=True),
    }


def stratified_sample(df: pd.DataFrame, sample_size: Optional[int], seed: int = SEED) -> pd.DataFrame:
    if sample_size is None or sample_size >= len(df):
        return df
    sampled, _ = train_test_split(
        df,
        train_size=sample_size,
        random_state=seed,
        stratify=df[TARGET_COLUMN],
    )
    return sampled.reset_index(drop=True)


def turkish_lower(text: str) -> str:
    return text.translate(str.maketrans({"I": "ı", "İ": "i"})).lower()


def normalize_light(text: str) -> str:
    text = str(text)
    text = URL_RE.sub(" <URL> ", text)
    text = USER_RE.sub(" <USER> ", text)
    text = HASHTAG_RE.sub(r" \1 ", text)
    text = REPEAT_CHAR_RE.sub(r"\1\1", text)
    text = turkish_lower(text)
    return SPACE_RE.sub(" ", text).strip()


def preprocess_text(text: str, variant: str) -> str:
    text = SPACE_RE.sub(" ", str(text)).strip()
    if variant == "raw":
        return text
    if variant == "light":
        return normalize_light(text)
    if variant == "light_masked":
        return PROFANITY_RE.sub("<PROFANITY>", normalize_light(text))
    raise ValueError(f"Unknown preprocessing variant: {variant}")


def apply_preprocessing(series: pd.Series, variant: str) -> pd.Series:
    return series.map(lambda value: preprocess_text(value, variant))
