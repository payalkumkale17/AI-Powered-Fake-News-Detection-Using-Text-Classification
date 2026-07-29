# AI-Powered Fake News Detection Using Text Classification

A from-scratch machine learning pipeline that classifies news articles as **REAL** or **FAKE** using TF-IDF text features and four classical ML models.

## Overview

- **Dataset:** 4,594 real, balanced political news articles (2,297 real / 2,297 fake), auto-downloaded from a public GitHub-hosted corpus.
- **Preprocessing:** URL/HTML removal, lowercasing, punctuation removal, stopword removal, and lemmatization (NLTK) — all implemented manually.
- **Features:** TF-IDF (5,000 features, unigrams + bigrams).
- **Models:** Logistic Regression, Random Forest, Multinomial Naive Bayes, and a Neural Network (MLP).
- **Best result:** Neural Network (MLP) — **93.47% accuracy**, F1 = 0.9355.

## Results

| Model                | Accuracy | Precision | Recall | F1     |
|-----------------------|----------|-----------|--------|--------|
| Logistic Regression   | 90.86%   | 88.03%    | 94.55% | 0.9118 |
| Random Forest         | 90.86%   | 89.14%    | 93.03% | 0.9104 |
| Naive Bayes           | 88.68%   | 86.75%    | 91.29% | 0.8896 |
| **Neural Network (MLP)** | **93.47%** | **92.36%** | **94.77%** | **0.9355** |

## Getting Started

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/fake-news-detection.git
cd fake-news-detection
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
python fake_news_detection_code.py
```

The script automatically downloads the dataset on first run, trains all four models, prints a full comparison table, and demos the trained model on new, unseen headlines.

## Project Structure
```
fake-news-detection/
├── fake_news_detection_code.py   # Full pipeline: data, preprocessing, training, evaluation, demo
├── requirements.txt
├── .gitignore
└── README.md
```

## How it works

1. **Data loading** — downloads `fake_and_real_news_dataset.csv` if not already present.
2. **Preprocessing** — cleans and lemmatizes each article's title + body text.
3. **Feature engineering** — converts cleaned text into a 5,000-dimensional TF-IDF matrix.
4. **Model training** — trains 4 classifiers on an 80:20 stratified train/test split.
5. **Evaluation** — reports accuracy, precision, recall, and F1 for each model.
6. **Prediction demo** — classifies a few brand-new sample headlines with confidence scores.

## License

MIT — feel free to use, modify, and share.
