"""
models.py
Model training and prediction functions for the sentiment classifier.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from constants import RANDOM_STATE, TFIDF_MAX_FEATURES, TFIDF_NGRAM_RANGE


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — TF-IDF + LOGISTIC REGRESSION BASELINE
# ─────────────────────────────────────────────────────────────────────────────

def train_baseline(
    X_train: list[str],
    y_train: list[str],
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """Fit a TF-IDF vectorizer and logistic regression on the training set.

    The vectorizer is fitted on training text only to prevent data leakage.
    No evaluation is performed — return the fitted objects for later use.

    Parameters:
        X_train (list[str]): Training review texts.
        y_train (list[str]): Training sentiment labels.

    Returns:
        tuple[TfidfVectorizer, LogisticRegression]: Fitted vectorizer and classifier.
    """
    print("Training baseline model — TF-IDF + logistic regression")

    vectorizer = TfidfVectorizer(
        max_features=TFIDF_MAX_FEATURES,
        ngram_range=TFIDF_NGRAM_RANGE,
        stop_words="english",
    )
    X_train_tfidf = vectorizer.fit_transform(X_train)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)

    print(f"  Vocabulary size : {len(vectorizer.vocabulary_):,}")
    print(f"  Training samples: {len(X_train):,}\n")

    return vectorizer, model


def predict_baseline(
    vectorizer: TfidfVectorizer,
    model: LogisticRegression,
    texts: list[str],
) -> list[str]:
    """Transform texts with the fitted vectorizer and return model predictions.

    Parameters:
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        model (LogisticRegression): Fitted logistic regression classifier.
        texts (list[str]): Review texts to classify.

    Returns:
        list[str]: Predicted sentiment labels ('positive', 'neutral', 'negative').
    """
    return model.predict(vectorizer.transform(texts))
