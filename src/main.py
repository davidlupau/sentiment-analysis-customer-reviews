from utils import detect_device, load_dataset, save_to_excel
from data_exploration import clean_dataset, map_rating_to_sentiment, plot_class_imbalance
from data_processing import split_dataset, balance_training_set
from models import train_baseline, predict_baseline
from constants import SAMPLES_PER_CLASS


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


if __name__ == "__main__":
    main()