"""
AI-Powered Fake News Detection Using Text Classification
==========================================================
Standalone, self-contained pipeline script.

What this script does:
  1. Downloads the real "fake_and_real_news_dataset" (4,594 balanced political
     news articles) from GitHub if not already present locally.
  2. Cleans and preprocesses the text from scratch (regex + NLTK).
  3. Builds TF-IDF features (unigrams + bigrams, 5,000 features).
  4. Trains and evaluates four classifiers: Logistic Regression, Random
     Forest, Multinomial Naive Bayes, and a Neural Network (MLP).
  5. Prints accuracy / precision / recall / F1 for every model and confirms
     the best model exceeds 93% accuracy.
  6. Demonstrates the trained prediction module on new, unseen headlines.

Requirements (install once):
    pip install pandas numpy scikit-learn nltk

Run:
    python fake_news_detection_code.py
"""

import os
import re
import sys
import urllib.request

import numpy as np
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import nltk

# ------------------------------------------------------------------
# 0. Setup: make sure NLTK resources are available
# ------------------------------------------------------------------
for resource in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{resource}")
    except LookupError:
        nltk.download(resource, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ------------------------------------------------------------------
# 1. Load data (download once, then reuse local copy)
# ------------------------------------------------------------------
DATA_PATH = "fake_and_real_news_dataset.csv"
DATA_URL = ("https://raw.githubusercontent.com/GeorgeMcIntire/"
            "fake_real_news_dataset/main/fake_and_real_news_dataset.csv")

if not os.path.exists(DATA_PATH):
    print(f"Downloading dataset to {DATA_PATH} ...")
    try:
        urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    except Exception as e:
        print(f"ERROR: could not download dataset automatically ({e}).")
        print("Please download it manually from:")
        print(f"  {DATA_URL}")
        print(f"and save it as '{DATA_PATH}' in this directory, then re-run.")
        sys.exit(1)

df = pd.read_csv(DATA_PATH)
df = df.dropna(subset=["text"]).reset_index(drop=True)
df["title"] = df["title"].fillna("")
print(f"Loaded {len(df)} articles. Class balance:")
print(df["label"].value_counts().to_string())

# ------------------------------------------------------------------
# 2. Text preprocessing (built from scratch)
# ------------------------------------------------------------------
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
url_re = re.compile(r"https?://\S+|www\.\S+")
html_re = re.compile(r"<.*?>")
nonalpha_re = re.compile(r"[^a-zA-Z\s]")


def clean_text(text):
    text = str(text)
    text = url_re.sub(" ", text)
    text = html_re.sub(" ", text)
    text = text.lower()
    text = nonalpha_re.sub(" ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)


print("\nCleaning text (this takes ~30-60 seconds)...")
df["clean_text"] = df["text"].apply(clean_text)
df["clean_title"] = df["title"].apply(clean_text)
df["combined"] = (df["clean_title"] + " " + df["clean_text"]).str.strip()
df = df[df["combined"].str.len() > 0].reset_index(drop=True)
print(f"After cleaning: {len(df)} articles retained.")

# ------------------------------------------------------------------
# 3. Feature engineering: TF-IDF
# ------------------------------------------------------------------
y = df["label"].map({"REAL": 0, "FAKE": 1})
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=3)
X = vectorizer.fit_transform(df["combined"])
print(f"\nTF-IDF matrix shape: {X.shape}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Train: {X_train.shape[0]} articles | Test: {X_test.shape[0]} articles")

# ------------------------------------------------------------------
# 4. Train and evaluate four models
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, C=1.0),
    "Random Forest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
    "Naive Bayes": MultinomialNB(alpha=0.3),
    "Neural Network (MLP)": MLPClassifier(
        hidden_layer_sizes=(100,), max_iter=300,
        early_stopping=True, random_state=42
    ),
}

results = {}
print("\n" + "=" * 60)
print(f"{'Model':<24}{'Accuracy':>10}{'Precision':>12}{'Recall':>10}{'F1':>8}")
print("=" * 60)
for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    results[name] = {"model": model, "accuracy": acc, "precision": prec,
                      "recall": rec, "f1": f1}
    print(f"{name:<24}{acc*100:>9.2f}%{prec*100:>11.2f}%{rec*100:>9.2f}%{f1:>8.4f}")
print("=" * 60)

best_name = max(results, key=lambda k: results[k]["accuracy"])
best_acc = results[best_name]["accuracy"]
best_model = results[best_name]["model"]

print(f"\nBest model: {best_name}  (accuracy = {best_acc*100:.2f}%)")
assert best_acc > 0.93, (
    f"Expected best-model accuracy above 93%, got {best_acc*100:.2f}%. "
    "Re-run (a different random seed / library version may shift results slightly)."
)
print("Check passed: best model accuracy exceeds 93%.")

# ------------------------------------------------------------------
# 5. Prediction module - try it on brand-new headlines
# ------------------------------------------------------------------
def predict_article(raw_text, model=best_model, vectorizer=vectorizer):
    cleaned = clean_text(raw_text)
    vec = vectorizer.transform([cleaned])
    pred = model.predict(vec)[0]
    conf = model.predict_proba(vec)[0].max()
    label = "FAKE" if pred == 1 else "REAL"
    return label, round(float(conf) * 100, 1)


print(f"\nDemo predictions using best model ({best_name}):")
demo_articles = [
    "The Federal Reserve announced today it will hold interest rates steady, "
    "citing stable inflation data from the Bureau of Labor Statistics.",
    "BREAKING: Scientists SHOCKED after discovering this one weird trick "
    "doctors don't want you to know about!!!",
    "Neil Armstrong: Their Ships Were Far Superior To Our - Boy, Were They Big, "
    "claims viral post citing anonymous NASA sources.",
]
for article in demo_articles:
    label, conf = predict_article(article)
    print(f'  "{article[:65]}..."')
    print(f"    -> Prediction: {label}  (confidence: {conf}%)\n")

print("Pipeline complete. No errors.")
