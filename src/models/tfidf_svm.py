# -*- coding: utf-8 -*-
import argparse, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report
from pathlib import Path
from ..data import build_text
from ..utils import macro_f1_acc

def train_and_eval(train_path: str, out_dir: str):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(train_path)
    X = build_text(df)
    y = df["label"].astype(str)

    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1,2), min_df=2, max_df=0.9, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char", ngram_range=(3,5), min_df=2, sublinear_tf=True)
    feat = FeatureUnion([("w", word_vec), ("c", char_vec)])
    clf = LinearSVC(class_weight="balanced", random_state=42)
    pipe = Pipeline([("tfidf", feat), ("clf", clf)])
    pipe.fit(X_tr, y_tr)

    pred_val = pipe.predict(X_val)
    macro, acc = macro_f1_acc(y_val, pred_val)
    (out/"val_report.txt").write_text(classification_report(y_val, pred_val, digits=3), encoding="utf-8")
    print(f"[TFIDF] val Macro-F1={macro:.4f} | Acc={acc:.4f}")
    pipe.fit(X, y)  # full train
    return pipe

def predict(pipe, test_path: str, out_csv: str):
    df = pd.read_csv(test_path)
    X = build_text(df)
    pred = pipe.predict(X)
    sub = pd.DataFrame({"id": df["id"], "label": pred})
    sub.to_csv(out_csv, index=False)
    print(f"Saved submission to {out_csv}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_path", type=str, default="data/vihallu-train.csv")
    ap.add_argument("--test_path", type=str, default="data/vihallu-public-test.csv")
    ap.add_argument("--out_dir", type=str, default="outputs")
    args = ap.parse_args()
    pipe = train_and_eval(args.train_path, args.out_dir)
    predict(pipe, args.test_path, str(Path(args.out_dir)/"submission_tfidf.csv"))
