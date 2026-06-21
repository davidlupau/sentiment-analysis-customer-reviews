from utils import detect_device, load_dataset, save_to_excel
from data_exploration import clean_dataset, map_rating_to_sentiment, plot_class_imbalance


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


if __name__ == "__main__":
    main()