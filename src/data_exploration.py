"""
data_exploration.py
Functions for exploring and cleaning the dataset before modelling.
"""

import pandas as pd


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable rows from the dataset.

    Drops rows where 'text' or 'rating' contain null values, then removes
    exact duplicate rows. Reports counts at each step.

    Args:
        df: Raw DataFrame loaded from the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with nulls and duplicates removed.
    """
    initial_rows = len(df)
    print(f"Initial row count: {initial_rows:,}")

    null_mask = df[["text", "rating"]].isnull().any(axis=1)
    null_count = int(null_mask.sum())
    if null_count:
        print(f"Dropping {null_count:,} rows with null values in 'text' or 'rating'.")
        df = df[~null_mask]
    else:
        print("No null values found in 'text' or 'rating'.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        print(f"Dropping {duplicate_count:,} duplicate rows.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")

    cleaned_rows = len(df)
    print(f"Final row count: {cleaned_rows:,} ({initial_rows - cleaned_rows:,} rows removed).\n")

    return df.reset_index(drop=True)
