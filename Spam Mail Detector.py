import argparse
import io
import re
import urllib.request
import zipfile

import pandas as pd
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, precision_score,
                             recall_score)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"

# Stopwords: use NLTK if available, otherwise fall back to scikit-learn's list
try:
    import nltk
    nltk.download("stopwords", quiet=True)
    from nltk.corpus import stopwords
    STOP_WORDS = set(stopwords.words("english"))
except Exception:
    STOP_WORDS = set(ENGLISH_STOP_WORDS)


# ---------------------------------------------------------------- 1. Load data
def load_data(path=None):
    """Load messages and labels into a DataFrame with columns: label, message."""
    if path:
        return pd.read_csv(path, sep="\t", header=None, names=["label", "message"])

    print("Downloading dataset from UCI...")
    raw = urllib.request.urlopen(DATA_URL, timeout=30).read()
    with zipfile.ZipFile(io.BytesIO(raw)) as z:
        with z.open("SMSSpamCollection") as f:
            return pd.read_csv(f, sep="\t", header=None, names=["label", "message"])


# ----------------------------------------------------------- 2. Preprocessing
def preprocess(text):
    """Lowercase -> tokenize -> remove stopwords -> rejoin into a string."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " url ", text)
    text = re.sub(r"\d+", " num ", text)
    tokens = re.findall(r"[a-z]+", text)
    tokens = [t for t in tokens if t not in STOP_WORDS and len(t) > 1]
    return " ".join(tokens)


# ------------------------------------------------------- 3-6. Features + model
def build_models():
    """TF-IDF features feeding each classifier, wrapped in sklearn Pipelines."""
    def tfidf():
        return TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)

    return {
        "Naive Bayes": Pipeline([("tfidf", tfidf()), ("clf", MultinomialNB(alpha=0.1))]),
        "Logistic Regression": Pipeline([
            ("tfidf", tfidf()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
        ]),
    }


# -------------------------------------------------------------- 7. Evaluation
def evaluate(name, model, X_test, y_test):
    pred = model.predict(X_test)
    print(f"\n=== {name} ===")
    print(f"Accuracy : {accuracy_score(y_test, pred):.4f}")
    print(f"Precision: {precision_score(y_test, pred):.4f}   (of predicted spam, how much is spam)")
    print(f"Recall   : {recall_score(y_test, pred):.4f}   (of real spam, how much we caught)")
    print(f"F1 score : {f1_score(y_test, pred):.4f}")
    print("Confusion matrix [rows=true, cols=pred] (ham, spam):")
    print(confusion_matrix(y_test, pred))
    print(classification_report(y_test, pred, target_names=["ham", "spam"], digits=3))
    return f1_score(y_test, pred)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", help="path to local SMSSpamCollection file")
    args = parser.parse_args()

    df = load_data(args.data)
    print(f"Loaded {len(df)} messages")
    print(df["label"].value_counts().to_string(), "\n")

    df["clean"] = df["message"].apply(preprocess)
    y = (df["label"] == "spam").astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean"], y, test_size=0.2, random_state=42, stratify=y)

    best_name, best_model, best_f1 = None, None, -1
    for name, model in build_models().items():
        model.fit(X_train, y_train)
        f1 = evaluate(name, model, X_test, y_test)
        if f1 > best_f1:
            best_name, best_model, best_f1 = name, model, f1

    print(f"\nBest model: {best_name} (F1 = {best_f1:.4f})")

    samples = [
        "WINNER!! You have won a $1000 prize. Call 0900123456 now to claim!",
        "Hey, are we still meeting for lunch tomorrow?",
        "Free entry in a weekly competition. Text WIN to 80086 now",
        "Can you send me the notes from today's class?",
    ]
    print("\nSample predictions:")
    for s in samples:
        label = "SPAM" if best_model.predict([preprocess(s)])[0] == 1 else "HAM"
        print(f"  [{label}] {s}")


if __name__ == "__main__":
    main()