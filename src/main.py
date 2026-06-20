from utils import detect_device, load_dataset
from data_exploration import clean_dataset


def main():
    DEVICE = detect_device()

    # Loading dataset
    df = load_dataset("amazon-fashion-reviews-dataset.csv")

    # Cleaning dataset
    df = clean_dataset(df)


if __name__ == "__main__":
    main()