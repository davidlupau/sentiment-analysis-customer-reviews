"""
models.py
Model training and prediction functions for the sentiment classifier.
"""

import time
from pathlib import Path

import torch
from torch.utils.data import Dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments,
)

from constants import (
    RANDOM_STATE,
    TFIDF_MAX_FEATURES,
    TFIDF_NGRAM_RANGE,
    DISTILBERT_MODEL,
    NUM_EPOCHS,
    MAX_TOKEN_LENGTH,
    BATCH_SIZE,
    LABEL_TO_ID,
    ID_TO_LABEL,
    MODEL_SAVE_DIR,
)


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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — DISTILBERT FINE-TUNING CHAMPION
# ─────────────────────────────────────────────────────────────────────────────

class _SentimentDataset(Dataset):
    def __init__(self, encodings: dict, labels: list[int]) -> None:
        self.encodings = encodings
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def train_distilbert(
    X_train: list[str],
    y_train: list[str],
    X_val: list[str],
    y_val: list[str],
    device: torch.device,
    save_dir: str = MODEL_SAVE_DIR,
) -> tuple:
    """Fine-tune DistilBERT for 3-class sentiment classification.

    Parameters:
        X_train (list[str]): Training review texts.
        y_train (list[str]): Training sentiment labels (string).
        X_val (list[str]): Validation review texts.
        y_val (list[str]): Validation sentiment labels (string).
        device (torch.device): Compute device (mps, cuda, or cpu).
        save_dir (str): Directory to save the fine-tuned model and tokenizer.

    Returns:
        tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
            Fine-tuned model and tokenizer.
    """
    print("Training champion model — DistilBERT fine-tuning")

    tokenizer = DistilBertTokenizerFast.from_pretrained(DISTILBERT_MODEL)
    model = DistilBertForSequenceClassification.from_pretrained(
        DISTILBERT_MODEL, num_labels=3
    )
    model.to(device)
    print(f"  Model is on device: {next(model.parameters()).device}")

    y_train_ids = [LABEL_TO_ID[label] for label in y_train]
    y_val_ids = [LABEL_TO_ID[label] for label in y_val]

    train_encodings = tokenizer(
        X_train, truncation=True, padding=True, max_length=MAX_TOKEN_LENGTH
    )
    val_encodings = tokenizer(
        X_val, truncation=True, padding=True, max_length=MAX_TOKEN_LENGTH
    )

    train_dataset = _SentimentDataset(train_encodings, y_train_ids)
    val_dataset = _SentimentDataset(val_encodings, y_val_ids)

    training_args = TrainingArguments(
        output_dir="./distilbert_output",
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="epoch",
        seed=RANDOM_STATE,
        report_to="none",
        logging_strategy="steps",
        logging_steps=50,
        disable_tqdm=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    start = time.time()
    trainer.train()
    elapsed_minutes = (time.time() - start) / 60
    print(f"\n{'='*50}\nTraining completed in {elapsed_minutes:.1f} minutes\n{'='*50}")

    Path(save_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    print(f"  Model and tokenizer saved to: {save_dir}\n")

    return model, tokenizer


def predict_distilbert(
    model: DistilBertForSequenceClassification,
    tokenizer: DistilBertTokenizerFast,
    texts: list[str],
    device: torch.device,
) -> list[str]:
    """Run batched inference with the fine-tuned DistilBERT model.

    Parameters:
        model (DistilBertForSequenceClassification): Fine-tuned classification model.
        tokenizer (DistilBertTokenizerFast): Tokenizer matched to the model.
        texts (list[str]): Review texts to classify.
        device (torch.device): Compute device (mps, cuda, or cpu).

    Returns:
        list[str]: Predicted sentiment labels ('positive', 'neutral', 'negative').
    """
    model.eval()
    all_preds: list[int] = []

    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        encodings = tokenizer(
            batch,
            truncation=True,
            padding=True,
            max_length=MAX_TOKEN_LENGTH,
            return_tensors="pt",
        )
        encodings = {k: v.to(device) for k, v in encodings.items()}
        with torch.no_grad():
            logits = model(**encodings).logits
        all_preds.extend(logits.argmax(dim=-1).tolist())

    return [ID_TO_LABEL[i] for i in all_preds]


def load_distilbert(
    save_dir: str,
    device: torch.device,
) -> tuple:
    """Load a fine-tuned DistilBERT model and tokenizer from disk.

    Parameters:
        save_dir (str): Directory containing the saved model and tokenizer.
        device (torch.device): Compute device to move the model to.

    Returns:
        tuple[DistilBertForSequenceClassification, DistilBertTokenizerFast]:
            Loaded model (in eval mode) and tokenizer.
    """
    print(f"Loading DistilBERT model from: {save_dir}")
    abs_save_dir = str(Path(save_dir).resolve())
    tokenizer = DistilBertTokenizerFast.from_pretrained(abs_save_dir)
    model = DistilBertForSequenceClassification.from_pretrained(abs_save_dir)
    model.to(device)
    model.eval()
    print(f"  Model is on device: {next(model.parameters()).device}\n")
    return model, tokenizer
