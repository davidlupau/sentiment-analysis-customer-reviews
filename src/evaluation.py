"""
evaluation.py — Shared evaluation functions for the baseline and champion models.
Both models are scored with identical code on the same test set.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — METRICS
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_model(
    y_true: list[str],
    y_pred: list[str],
    model_name: str,
) -> dict | None:
    """Compute accuracy, macro-F1, and a full per-class classification report.

    Parameters:
        y_true (list[str]): True sentiment labels from the test set.
        y_pred (list[str]): Predicted labels returned by predict_baseline or
            predict_distilbert.
        model_name (str): Display name printed in the report header
            (e.g. "TF-IDF Baseline", "DistilBERT Champion").

    Returns:
        dict | None: Keys 'accuracy', 'macro_f1', and 'report', or None on error.
    """
    print(f"Evaluating {model_name}")
    try:
        accuracy = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro")  # primary metric
        report = classification_report(
            y_true,
            y_pred,
            labels=["negative", "neutral", "positive"],
            digits=4,
        )

        print(f"\n{'='*50}")
        print(f"  {model_name}")
        print(f"{'='*50}")
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  Macro-F1 : {macro_f1:.4f}  ← primary metric")
        print(f"\n{report}")

        return {"accuracy": accuracy, "macro_f1": macro_f1, "report": report}
    except Exception as e:
        print(f"  Error evaluating {model_name}: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — CONFUSION MATRIX
# ─────────────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(
    y_true: list[str],
    y_pred: list[str],
    model_name: str,
) -> None:
    """Plot and save a labelled confusion matrix as a seaborn heatmap.

    Parameters:
        y_true (list[str]): True sentiment labels.
        y_pred (list[str]): Predicted sentiment labels.
        model_name (str): Used in the plot title and output filename.

    Returns:
        None
    """
    print(f"Plotting confusion matrix — {model_name}")
    try:
        labels = ["negative", "neutral", "positive"]
        cm = confusion_matrix(y_true, y_pred, labels=labels)

        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
        )
        ax.set_title(
            f"Confusion Matrix — {model_name}",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("Actual", fontsize=11)
        plt.tight_layout()

        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        filename = f"{model_name.replace(' ', '_')}_confusion_matrix.png"
        output_path = output_dir / filename
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.show()
        print(f"  Confusion matrix saved to: {output_path}")
    except Exception as e:
        print(f"  Error plotting confusion matrix for {model_name}: {e}")
