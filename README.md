# Sentiment Analysis of Amazon Fashion Customer Reviews

Classifying customer reviews as **positive**, **neutral**, or **negative**, with particular attention to the cases that trip models up: sarcasm, negation, and mixed-sentiment ("multipolar") reviews.

Two models are built and compared on the same test set — a cheap TF-IDF baseline and a fine-tuned DistilBERT transformer — to test whether the more expensive model earns its cost.

---

## Business problem and impact

E-commerce teams receive far more written feedback than anyone can read. Star ratings compress a review into a single number and miss *why* a customer was happy or unhappy.

This project turns free-text Amazon Fashion reviews into a sentiment signal a marketing or product team can act on — surfacing dissatisfaction that a 3-star rating alone would hide, and doing so at a scale no manual review process could match. The deliverable is a simple interface that returns a sentiment label for any pasted review, so non-technical users get the insight without touching the model.

---

## Quick start

```bash
# 1. Clone and enter the repo
git clone https://github.com/davidlupau/sentiment-analysis-customer-reviews.git
cd sentiment-analysis-customer-reviews

# 2. Install dependencies
pip install -r requirements.txt

# 3. Get the dataset
# Download from Kaggle (link below) and place the CSV in the data/ folder.

# 4. Run the full pipeline (train both models + evaluate)
python src/main.py

# Or skip training and evaluate / launch the app from a saved model
python src/main.py --skip-training --load-dir saved_model/
```

Dataset: [Amazon Fashion 800k+ User Reviews](https://www.kaggle.com/datasets/fawadhossaini1415/amazon-fashion-800k-user-reviews-dataset) (McAuley Lab, CC BY 4.0).

> Training DistilBERT is much faster on a GPU. The pipeline auto-detects CUDA, Apple Silicon (MPS), or CPU and runs on whatever is available.

---

## Technical skills demonstrated

| Area | What was done |
|---|---|
| Data cleaning | Removed nulls, blank/whitespace-only text, and duplicates from 867k raw rows down to 861k usable reviews |
| Class balancing | Undersampled the **training** set to 5k per class; left the **test** set at its realistic distribution so evaluation reflects real-world performance |
| Classical ML baseline | TF-IDF features (unigrams + bigrams) with logistic regression via scikit-learn |
| Transformer fine-tuning | Fine-tuned DistilBERT using the Hugging Face Trainer API |
| Overfitting control | Cut training from 3 epochs to 2 after validation loss rose at epoch 3 |
| Model evaluation | Accuracy, macro-F1, per-class precision/recall, and confusion matrices — identical code for both models for a fair comparison |
| Targeted error analysis | Hand-built 20-review probe set to test sarcasm, negation, and mixed sentiment specifically |
| Deployment | Gradio interface wrapping a shared prediction function |
| Engineering practice | Modular `src/` package, self-contained notebook mirror, reproducible runs with a fixed random seed |

---

## Key findings

- **DistilBERT wins on aggregate metrics.** Macro-F1 of **0.70** vs the baseline's **0.62** on the realistic test set.
- **Macro-F1 was the right headline metric.** Overall accuracy would have hidden weak performance on the minority neutral class; macro-F1 weights all three classes equally.
- **DistilBERT handles negation and mixed sentiment better**, as expected from a model that reads word order rather than counting words.
- **DistilBERT fails completely on sarcasm** in the probe set (0 of 6), while the baseline catches most of them. The transformer trusts positive surface language ("absolutely stunning craftsmanship") that sarcastic reviews exploit.
- **The baseline's sarcasm "success" is largely accidental** — sarcastic reviews often contain genuinely negative words ("broke", "fell apart") that a bag-of-words model picks up. It isn't understanding sarcasm; it's getting lucky on keywords.

The headline-metric winner and the probe-set winner are not the same model — which is the most interesting result of the project.

---

## Project structure

```
sentiment-analysis-customer-reviews/
├── src/
│   ├── main.py            # Pipeline entry point (training, evaluation, app)
│   ├── constants.py       # Configuration: model names, hyperparameters, label maps
│   ├── utils.py           # Device detection and shared helpers
│   ├── data_processing.py # Loading, cleaning, label mapping, balancing
│   ├── models.py          # TF-IDF baseline and DistilBERT training/prediction
│   ├── evaluation.py      # Shared metrics and confusion matrices
│   └── predict.py         # Shared prediction function used by the Gradio app
├── notebook/              # Self-contained annotated walkthrough of the full pipeline
├── data/                  # Dataset goes here (full CSV is gitignored)
├── analysis_output/       # Generated plots, confusion matrices, probe-set others
├── requirements.txt
└── README.md
```

---

## Links

- **Notebook walkthrough** — [`notebook/`](notebook/): the full pipeline end to end with explanatory commentary at each step.

---

## License

Released under the [MIT License](LICENSE).
