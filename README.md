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

- **DistilBERT wins on aggregate metrics.** Macro-F1 of **0.69** vs the baseline's **0.62** on the realistic test set (accuracy 0.71 vs 0.65).
- **Macro-F1 was the right headline metric.** Both models score worst on the minority neutral class (F1 ≈ 0.40 baseline, 0.47 DistilBERT); overall accuracy would have hidden this. Macro-F1 weights all three classes equally so the weak class can't be masked.
- **The aggregate winner and the probe-set winner are different models.** On a hand-built 20-review probe set the *baseline* scores higher overall (14/20 vs 10/20), even though DistilBERT wins the large-scale metrics. This contrast is the most interesting result of the project.
- **DistilBERT struggles badly with sarcasm** (1 of 6 on the probe set, vs 5 of 6 for the baseline). It tends to trust positive surface language ("absolutely stunning craftsmanship") that sarcastic reviews exploit.
- **The baseline's sarcasm "success" is largely accidental** — sarcastic reviews often contain genuinely negative words ("broke", "fell apart") that a bag-of-words model picks up. It isn't understanding sarcasm; it's catching negative keywords.
- **The probe set is a small qualitative diagnostic, not a benchmark.** With only 20 items, a one- or two-review swing flips a category, so the probe is used to surface *failure modes* rather than to rank the models precisely. The large test set remains the basis for the headline metrics.

---

## Project structure

```
sentiment-analysis-customer-reviews/
├── src/
│   ├── main.py            # Pipeline entry point (training, evaluation, app)
│   ├── constants.py       # Configuration: model names, hyperparameters, label maps
│   ├── utils.py           # Device detection and shared helpers
│   ├── data_cleaning.py   # Loading, cleaning, label mapping, imbalance plot
│   ├── data_processing.py # Train/test split and class balancing
│   ├── models.py          # TF-IDF baseline and DistilBERT training/prediction
│   ├── evaluation.py      # Shared metrics and confusion matrices
│   ├── predict.py         # Shared prediction function used by the Gradio app
│   └── gradio_app.py      # Gradio interface for end users
├── notebook/              # Self-contained annotated walkthrough of the full pipeline
├── data/                  # Dataset goes here (full CSV is gitignored)
├── analysis_output/       # Generated plots, confusion matrices, probe-set results
├── make_sample.py         # Helper to create a small data sample
├── requirements.txt
└── README.md
```

---

## Links

- **Notebook walkthrough** — [`notebook/`](notebook/): the full pipeline end to end with explanatory commentary at each step.

---

## License

Released under the [MIT License](LICENSE).