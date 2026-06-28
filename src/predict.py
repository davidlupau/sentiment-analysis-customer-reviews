"""
predict.py — Shared prediction interface for the Gradio UI.
"""

import torch
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

from models import predict_distilbert


def predict(
    text: str,
    distilbert_model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    device: torch.device,
) -> str | None:
    """Run DistilBERT inference on a single review string.

    Parameters:
        text (str): A single customer review to classify.
        distilbert_model (DistilBertForSequenceClassification): Fine-tuned model.
        tokenizer (DistilBertTokenizerFast): Tokenizer matched to the model.
        device (torch.device): Compute device to run inference on.

    Returns:
        str | None: Predicted label ('positive', 'neutral', or 'negative'),
            or None if an error occurs.
    """
    try:
        return predict_distilbert(distilbert_model, tokenizer, [text], device)[0]
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None
