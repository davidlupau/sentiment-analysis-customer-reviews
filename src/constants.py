"""
constants.py
Project-wide constants shared across modules.
"""

# Split then balance training set
SAMPLES_PER_CLASS = 5_000

# TF-IDF baseline
RANDOM_STATE: int = 42
TFIDF_MAX_FEATURES: int = 10_000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)

# DistilBERT champion
DISTILBERT_MODEL: str = "distilbert-base-uncased"
NUM_EPOCHS: int = 3
MAX_TOKEN_LENGTH: int = 256
BATCH_SIZE: int = 16
LABEL_TO_ID: dict[str, int] = {"negative": 0, "neutral": 1, "positive": 2}
ID_TO_LABEL: dict[int, str] = {0: "negative", 1: "neutral", 2: "positive"}
