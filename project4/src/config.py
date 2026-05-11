from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
RESULTS_DIR = PROJECT_ROOT / "results"

HF_DATASET_URL = (
    "https://huggingface.co/datasets/Overfit-GM/"
    "turkish-toxic-language/resolve/main/turkish_toxic_language.csv"
)
RAW_DATA_PATH = RAW_DATA_DIR / "turkish_toxic_language.csv"

SEED = 42

TEXT_COLUMN = "text"
TARGET_COLUMN = "target"
SOURCE_COLUMN = "source"
TOXIC_COLUMN = "is_toxic"

LABEL_ORDER = ["OTHER", "PROFANITY", "INSULT", "RACIST", "SEXIST"]
TOXIC_LABELS = ["PROFANITY", "INSULT", "RACIST", "SEXIST"]

