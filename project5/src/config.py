from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
DEFAULT_DATASET_ROOT = RAW_DATA_DIR / "ocrturk"
RENDERED_DIR = DATA_DIR / "rendered"
RESULTS_DIR = PROJECT_ROOT / "results"
OUTPUT_DIR = PROJECT_ROOT / "output"

OCRTURK_REPO_URL = "https://github.com/metunlp/ocrturk.git"
OCRTURK_PAPER_URL = "https://aclanthology.org/2026.sigturk-1.16/"
PROJECT_REPO_URL = "https://github.com/karakasarda/deeplearning_lab"

DEFAULT_DPI = 300
DEFAULT_ENGINES = ("tesseract", "easyocr")
DIACRITIC_CHARS = set("çğıİöşüÇĞIÖŞÜ")
