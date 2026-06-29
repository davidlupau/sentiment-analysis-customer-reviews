"""
data_cleaning.py
Functions for exploring and cleaning the dataset before modelling.
"""

import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA CLEANING
# ─────────────────────────────────────────────────────────────────────────────

def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Remove unusable rows from the dataset.

    Drops rows where 'text' or 'rating' contain null values, rows where
    'text' is blank or whitespace-only, then removes exact duplicate rows.
    Reports counts at each step.

    Parameters:
        df (pd.DataFrame): Raw DataFrame loaded from the CSV file.

    Returns:
        pd.DataFrame: Cleaned DataFrame with nulls, blanks, and duplicates removed.
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

    blank_mask = df["text"].str.strip() == ""
    blank_count = int(blank_mask.sum())
    if blank_count:
        print(f"Dropping {blank_count:,} rows with blank or whitespace-only 'text'.")
        df = df[~blank_mask]
    else:
        print("No blank or whitespace-only 'text' values found.")

    duplicate_count = int(df.duplicated().sum())
    if duplicate_count:
        print(f"Dropping {duplicate_count:,} duplicate rows.")
        df = df.drop_duplicates()
    else:
        print("No duplicate rows found.")

    cleaned_rows = len(df)
    print(f"Final row count: {cleaned_rows:,} ({initial_rows - cleaned_rows:,} rows removed).\n")

    return df.reset_index(drop=True)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────────────────────────────────────

def map_rating_to_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'sentiment' column derived from the 'rating' column.

    Mapping:
        1, 2  → negative
        3     → neutral
        4, 5  → positive

    Parameters:
        df (pd.DataFrame): Cleaned DataFrame containing a numeric 'rating' column.

    Returns:
        pd.DataFrame: DataFrame with a new 'sentiment' column appended.
    """
    mapping = {1: "negative", 2: "negative", 3: "neutral", 4: "positive", 5: "positive"}
    df = df.copy()
    df["sentiment"] = df["rating"].map(mapping)
    counts = df["sentiment"].value_counts()
    print("Sentiment distribution:")
    for label in ["positive", "neutral", "negative"]:
        print(f"  {label}: {counts.get(label, 0):,}")
    print()
    return df


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — VISUALISATION
# ─────────────────────────────────────────────────────────────────────────────

def plot_class_imbalance(df: pd.DataFrame) -> None:
    """Save a bar chart of sentiment class counts to analysis_output/.

    Parameters:
        df (pd.DataFrame): DataFrame containing a 'sentiment' column with
            values 'positive', 'neutral', and 'negative'.
    """
    counts = df["sentiment"].value_counts().reindex(["positive", "neutral", "negative"])
    total = counts.sum()

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#4CAF50", "#9E9E9E", "#F44336"]
    bars = ax.bar(counts.index, counts.values, color=colors, edgecolor="white", linewidth=0.8)

    for bar, count in zip(bars, counts.values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + total * 0.004,
            f"{count:,}\n({count / total * 100:.1f}%)",
            ha="center", va="bottom", fontsize=10,
        )

    ax.set_title("Sentiment Class Distribution", fontsize=14, fontweight="bold", pad=15)
    ax.set_xlabel("Sentiment", fontsize=11)
    ax.set_ylabel("Number of reviews", fontsize=11)
    ax.set_ylim(0, counts.max() * 1.18)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()

    output_dir = Path(__file__).parent.parent / "analysis_output"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "class_imbalance.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Plot saved to {output_path}")
