"""
gradio_app.py — Gradio web interface for the sentiment analysis system.
Designed for non-technical marketing users — clean single prediction, no model internals exposed.
"""

import gradio as gr
import torch

from constants import MODEL_SAVE_DIR
from models import load_distilbert
from predict import predict
from utils import detect_device


# ─── SECTION 1 — MODEL LOADING ───────────────────────────────────────────────

def load_models() -> tuple | None:
    """Load the fine-tuned DistilBERT model and tokenizer from disk.

    Returns:
        tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast, torch.device]:
            Loaded model, tokenizer, and active compute device,
            or None if loading fails.
    """
    print("Loading models for Gradio interface...")
    try:
        device = detect_device()
        distilbert_model, tokenizer = load_distilbert(MODEL_SAVE_DIR, device)
        return distilbert_model, tokenizer, device
    except Exception as e:
        print(f"Error loading models: {e}")
        return None


# ─── SECTION 2 — GRADIO INTERFACE ────────────────────────────────────────────

def build_interface(
    distilbert_model,
    tokenizer,
    device: torch.device,
) -> gr.Interface:
    """Build and return the Gradio sentiment analysis interface.

    Parameters:
        distilbert_model (DistilBertForSequenceClassification): Fine-tuned model.
        tokenizer (DistilBertTokenizerFast): Tokenizer matched to the model.
        device (torch.device): Compute device to run inference on.

    Returns:
        gr.Interface: Configured Gradio interface ready to launch.
    """
    _LABEL_MAP = {
        "positive": "😊 Positive",
        "neutral":  "😐 Neutral",
        "negative": "😞 Negative",
    }

    def classify(text: str) -> str:
        label = predict(text, distilbert_model, tokenizer, device)
        return _LABEL_MAP.get(label, "Unable to classify — please try again.")

    return gr.Interface(
        fn=classify,
        inputs=gr.Textbox(
            label="Customer review",
            placeholder="Paste a customer review here...",
            lines=4,
        ),
        outputs=gr.Textbox(label="Sentiment"),
        title="Customer Review Sentiment Analyser",
        description=(
            "Paste any Amazon Fashion customer review to instantly detect "
            "whether the sentiment is positive, neutral, or negative."
        ),
        examples=[
            [
                "I absolutely love these jeans! The fit is perfect and the denim quality "
                "is outstanding. They arrived quickly and looked even better than the photos. "
                "I've already ordered two more pairs in different colours."
            ],
            [
                "These shoes fell apart after just three weeks of normal wear. The sole "
                "started peeling away and the stitching came undone on both sides. I followed "
                "the care instructions exactly and still had this problem — very disappointing "
                "for the price."
            ],
            [
                "The jacket is okay. The colour matches the listing and the sizing is accurate, "
                "but the material feels a bit thin for the price point. It does what it says — "
                "nothing more, nothing less."
            ],
        ],
        flagging_mode="never",
    )


if __name__ == "__main__":
    distilbert_model, tokenizer, device = load_models()
    iface = build_interface(distilbert_model, tokenizer, device)
    iface.launch(share=True)
