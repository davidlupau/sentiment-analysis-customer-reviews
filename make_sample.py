"""
make_sample.py
--------------
Create a small, class-balanced subset of the Amazon Fashion 800k reviews
dataset for the Task 2 sentiment analysis project.

The full CSV (~253 MB) is NOT committed to the repository. This script reads
the full file once (downloaded separately, see README) and writes a compact,
balanced sample that IS committed, so the project is reproducible on a laptop.

Rating -> label mapping (3 classes / valence levels):
    1.0, 2.0  -> negative
    3.0       -> neutral
    4.0, 5.0  -> positive

Because the source is heavily imbalanced (neutral is roughly half the size of
each other class), the sample is balanced by undersampling each class down to
the same number of rows (PER_CLASS).

Usage:
    python make_sample.py
"""

import pandas as pd

# ---- configuration -------------------------------------------------------
FULL_CSV = "data/amazon-fashion-800k+-user-reviews-dataset.csv"  # not in git
OUT_CSV = "data/sample_balanced.csv"                              # committed
PER_CLASS = 3000        # rows kept per class -> 9000 total
RANDOM_STATE = 42       # fixed seed for reproducibility
# -------------------------------------------------------------------------


def to_label(rating: float) -> str:
    """Map a 1-5 star rating to one of three valence labels."""
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"


def main() -> None:
    # Read only the columns we need to keep memory low.
    df = pd.read_csv(FULL_CSV, usecols=["rating", "title", "text"])

    # Drop rows with no review text, then build the label column.
    df = df.dropna(subset=["text"])
    df["label"] = df["rating"].astype(float).apply(to_label)

    print("Class counts in full dataset:")
    print(df["label"].value_counts(), "\n")

    # Undersample each class to PER_CLASS rows for a balanced subset.
    # If a class has fewer than PER_CLASS rows, take all of it.
    parts = []
    for label, group in df.groupby("label"):
        n = min(PER_CLASS, len(group))
        parts.append(group.sample(n=n, random_state=RANDOM_STATE))

    sample = (
        pd.concat(parts)
        .sample(frac=1, random_state=RANDOM_STATE)   # shuffle the rows
        .reset_index(drop=True)
    )

    sample.to_csv(OUT_CSV, index=False)

    print(f"Wrote {len(sample)} rows to {OUT_CSV}")
    print("Class counts in sample:")
    print(sample["label"].value_counts())


if __name__ == "__main__":
    main()
