"""
constants.py
Project-wide constants shared across modules.
"""

RANDOM_STATE: int = 42

# TF-IDF baseline
TFIDF_MAX_FEATURES: int = 10_000
TFIDF_NGRAM_RANGE: tuple[int, int] = (1, 2)
