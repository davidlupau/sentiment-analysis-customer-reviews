import argparse

from sklearn.model_selection import train_test_split

from utils import detect_device, load_dataset, save_to_excel
from data_exploration import clean_dataset, map_rating_to_sentiment, plot_class_imbalance
from data_processing import split_dataset, balance_training_set
from models import train_baseline, predict_baseline, train_distilbert, predict_distilbert, load_distilbert
from evaluation import evaluate_model, plot_confusion_matrix, evaluate_probe_set
from constants import SAMPLES_PER_CLASS, RANDOM_STATE, MODEL_SAVE_DIR


def main():
    parser = argparse.ArgumentParser(description="Sentiment analysis pipeline")
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Load saved DistilBERT model instead of retraining. Requires a saved model in MODEL_SAVE_DIR.",
    )
    parser.add_argument(
        "--save-dir",
        default=MODEL_SAVE_DIR,
        help="Directory to save the trained DistilBERT model (default: MODEL_SAVE_DIR).",
    )
    parser.add_argument(
        "--load-dir",
        default=MODEL_SAVE_DIR,
        help="Directory to load the saved DistilBERT model from when --skip-training is set (default: MODEL_SAVE_DIR).",
    )
    args = parser.parse_args()

    mode = "EVALUATION ONLY (loading saved model)" if args.skip_training else "FULL PIPELINE (training + evaluation)"
    print(f"\n{'='*50}\nMode: {mode}\n{'='*50}\n")

    DEVICE = detect_device()

    # Loading dataset
    df = load_dataset("amazon-fashion-reviews-dataset.csv")

    # Cleaning dataset
    df = clean_dataset(df)

    # Map ratings to sentiment labels
    df = map_rating_to_sentiment(df)
    #save_to_excel(df, "sentiment_mapped.xlsx")
    plot_class_imbalance(df)

    # Split then balance training set
    df_train, df_test = split_dataset(df)
    df_train = balance_training_set(df_train, SAMPLES_PER_CLASS)

    # Baseline model — TF-IDF + logistic regression
    vectorizer_baseline, model_baseline = train_baseline(
        df_train["text"].tolist(), df_train["sentiment"].tolist()
    )
    baseline_preds = predict_baseline(
        vectorizer_baseline, model_baseline, df_test["text"].tolist()
    )
    print(f"Predictions stored — {len(baseline_preds):,} test samples ready for evaluation.")

    # Champion model — DistilBERT fine-tuning
    if args.skip_training:
        model_distilbert, tokenizer = load_distilbert(args.load_dir, DEVICE)
    else:
        df_bert_train, df_bert_val = train_test_split(
            df_train,
            test_size=0.1,
            random_state=RANDOM_STATE,
            stratify=df_train["sentiment"],
        )
        model_distilbert, tokenizer = train_distilbert(
            df_bert_train["text"].tolist(),
            df_bert_train["sentiment"].tolist(),
            df_bert_val["text"].tolist(),
            df_bert_val["sentiment"].tolist(),
            DEVICE,
            save_dir=args.save_dir,
        )
    distilbert_preds = predict_distilbert(
        model_distilbert, tokenizer, df_test["text"].tolist(), DEVICE
    )
    print(f"Predictions stored — {len(distilbert_preds):,} test samples ready for evaluation.")

    # Evaluation — both models, same code
    y_true = df_test["sentiment"].tolist()

    metrics_baseline = evaluate_model(y_true, baseline_preds, "TF-IDF Baseline")
    plot_confusion_matrix(y_true, baseline_preds, "TF-IDF Baseline")

    metrics_distilbert = evaluate_model(y_true, distilbert_preds, "DistilBERT Champion")
    plot_confusion_matrix(y_true, distilbert_preds, "DistilBERT Champion")

    sep = "═" * 40
    print(f"\n{sep}\nMODEL COMPARISON SUMMARY\n{sep}")
    print(f"{'':16}  Accuracy  Macro-F1")
    if metrics_baseline:
        print(f"{'TF-IDF Baseline':<16}:  {metrics_baseline['accuracy']:.4f}    {metrics_baseline['macro_f1']:.4f}")
    if metrics_distilbert:
        print(f"{'DistilBERT':<16}:  {metrics_distilbert['accuracy']:.4f}    {metrics_distilbert['macro_f1']:.4f}")
    print(sep)

    # Probe set — qualitative evaluation
    evaluate_probe_set(
        vectorizer_baseline,
        model_baseline,
        model_distilbert,
        tokenizer,
        DEVICE,
    )


if __name__ == "__main__":
    main()
