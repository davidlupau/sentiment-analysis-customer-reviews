"""
evaluation.py — Shared evaluation functions for the baseline and champion models.
Both models are scored with identical code on the same test set.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from models import predict_baseline, predict_distilbert


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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — PROBE SET
# ─────────────────────────────────────────────────────────────────────────────

def evaluate_probe_set(
    vectorizer,
    baseline_model,
    distilbert_model,
    tokenizer,
    device: torch.device,
) -> pd.DataFrame | None:
    """Run both models on the probe set and print a per-phenomenon breakdown.

    Parameters:
        vectorizer (TfidfVectorizer): Fitted TF-IDF vectorizer.
        baseline_model (LogisticRegression): Fitted baseline classifier.
        distilbert_model: Fine-tuned DistilBERT model.
        tokenizer: DistilBERT tokenizer.
        device (torch.device): Compute device.

    Returns:
        pd.DataFrame | None: Results table with predictions and correctness
            flags, or None on error.
    """
    print("Running probe set evaluation")
    try:
        probe_path = Path(__file__).parent.parent / "data" / "probe_set.csv"
        df = pd.read_csv(probe_path)
        texts = df["text"].tolist()

        df["baseline_pred"] = predict_baseline(vectorizer, baseline_model, texts)
        df["distilbert_pred"] = predict_distilbert(
            distilbert_model, tokenizer, texts, device
        )
        df["baseline_correct"] = df["baseline_pred"] == df["true_label"]
        df["distilbert_correct"] = df["distilbert_pred"] == df["true_label"]

        results = df[[
            "id", "phenomenon", "text", "true_label",
            "baseline_pred", "distilbert_pred",
            "baseline_correct", "distilbert_correct",
        ]]

        # Per-phenomenon summary
        sep = "─" * 41
        print(f"\n{'Phenomenon':<14}  {'Baseline':<14}  DistilBERT")
        print(sep)
        for phenomenon, group in results.groupby("phenomenon"):
            n = len(group)
            b = int(group["baseline_correct"].sum())
            d = int(group["distilbert_correct"].sum())
            b_str = f"{b}/{n}  ({b / n * 100:.0f}%)"
            d_str = f"{d}/{n}  ({d / n * 100:.0f}%)"
            print(f"{phenomenon:<14}  {b_str:<14}  {d_str}")
        print(sep)
        n_total = len(results)
        b_total = int(results["baseline_correct"].sum())
        d_total = int(results["distilbert_correct"].sum())
        b_str = f"{b_total}/{n_total} ({b_total / n_total * 100:.0f}%)"
        d_str = f"{d_total}/{n_total} ({d_total / n_total * 100:.0f}%)"
        print(f"{'TOTAL':<14}  {b_str:<14}  {d_str}")

        # Per-row details
        print()
        for _, row in results.iterrows():
            b_mark = "✓" if row["baseline_correct"] else "✗"
            d_mark = "✓" if row["distilbert_correct"] else "✗"
            text_preview = str(row["text"])[:80]
            print(
                f"[{int(row['id']):>3}] {str(row['phenomenon']):<14} "
                f"true={row['true_label']:<10} "
                f"baseline={row['baseline_pred']:<10}{b_mark}  "
                f"distilbert={row['distilbert_pred']:<10}{d_mark}  "
                f'"{text_preview}"'
            )

        output_dir = Path("analysis_output")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "probe_set_results.csv"
        results.to_csv(output_path, index=False)
        print(f"\n  Probe set results saved to: {output_path}")

        return results
    except Exception as e:
        print(f"  Error running probe set evaluation: {e}")
        return None
