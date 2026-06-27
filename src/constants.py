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
