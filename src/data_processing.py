"""
data_processing.py
Functions for splitting and balancing the dataset for model training.
"""

import pandas as pd
from sklearn.model_selection import train_test_split


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — TRAIN / TEST SPLIT
# ─────────────────────────────────────────────────────────────────────────────

def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train/test split on the 'sentiment' column.

    Parameters:
        df (pd.DataFrame): Full cleaned and labelled DataFrame.
        test_size (float): Proportion of the dataset reserved for testing.
        random_state (int): Random seed for reproducibility.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (df_train, df_test)
    """
    print(f"Splitting dataset — test size: {test_size:.0%}, random_state: {random_state}")
    df_train, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df["sentiment"]
    )
    df_train = df_train.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    print(f"  Train : {len(df_train):,} rows")
    print(f"  Test  : {len(df_test):,} rows")
    print("Test set distribution (realistic, kept imbalanced):")
    test_counts = df_test["sentiment"].value_counts()
    total_test = len(df_test)
    for cls in ["positive", "neutral", "negative"]:
        n = test_counts.get(cls, 0)
        print(f"  {cls}: {n:,} ({n / total_test * 100:.1f}%)")
    print()
    return df_train, df_test


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — TRAINING SET BALANCING
# ─────────────────────────────────────────────────────────────────────────────

def balance_training_set(
    df_train: pd.DataFrame,
    n_per_class: int,
    random_state: int = 42,
) -> pd.DataFrame:
    """Undersample each sentiment class in the training set to n_per_class rows.

    Samples n_per_class rows per class. If a class has fewer rows than
    n_per_class, a warning is printed and all available rows are used.
    The result is shuffled so classes are not in ordered blocks.

    Parameters:
        df_train (pd.DataFrame): Training split with a 'sentiment' column.
        n_per_class (int): Target number of rows per sentiment class.
        random_state (int): Random seed for reproducibility.

    Returns:
        pd.DataFrame: Balanced, shuffled training DataFrame.
    """
    classes = ["positive", "neutral", "negative"]

    print(f"Balancing training set — target: {n_per_class:,} rows per class.")
    print("Before:")
    counts_before = df_train["sentiment"].value_counts()
    for cls in classes:
        print(f"  {cls}: {counts_before.get(cls, 0):,}")

    samples = []
    for cls in classes:
        cls_df = df_train[df_train["sentiment"] == cls]
        available = len(cls_df)
        n = min(n_per_class, available)
        if available < n_per_class:
            print(f"Warning: '{cls}' has only {available:,} rows — using all available.")
        samples.append(cls_df.sample(n=n, random_state=random_state))

    df_balanced = (
        pd.concat(samples)
        .sample(frac=1, random_state=random_state)
        .reset_index(drop=True)
    )

    print("\nAfter:")
    counts_after = df_balanced["sentiment"].value_counts()
    for cls in classes:
        print(f"  {cls}: {counts_after.get(cls, 0):,}")
    print(f"Total training rows: {len(df_balanced):,}\n")

    return df_balanced
