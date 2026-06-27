from sklearn.model_selection import train_test_split

from utils import detect_device, load_dataset, save_to_excel
from data_exploration import clean_dataset, map_rating_to_sentiment, plot_class_imbalance
from data_processing import split_dataset, balance_training_set
from models import train_baseline, predict_baseline, train_distilbert, predict_distilbert
from constants import SAMPLES_PER_CLASS, RANDOM_STATE


def main():
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
    )
    distilbert_preds = predict_distilbert(
        model_distilbert, tokenizer, df_test["text"].tolist(), DEVICE
    )
    print(f"Predictions stored — {len(distilbert_preds):,} test samples ready for evaluation.")


if __name__ == "__main__":
    main()
