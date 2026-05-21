"""
Traditional ML Baseline: TF-IDF + Logistic Regression with class weighting.
Used to establish a benchmark performance level on the ViHSD dataset.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score

try:
    from src.utils.config import load_yaml_config, resolve_path
except ImportError:
    from utils.config import load_yaml_config, resolve_path


def train_baseline(config_path: str = "configs/paths.yaml") -> dict[str, Any]:
    """Train TF-IDF + Logistic Regression baseline model and evaluate it."""
    # 1. Load configuration
    cfg = load_yaml_config(config_path)
    
    # 2. Get processed paths
    train_path = resolve_path(cfg["data"]["train_processed"])
    dev_path = resolve_path(cfg["data"]["val_processed"])
    
    model_path = resolve_path(cfg["models"]["baseline"])
    vectorizer_path = model_path.parent / "tfidf_vectorizer.pkl"
    report_path = resolve_path(cfg["results"]["baseline_report"])
    
    print(f"[BASELINE] Loading processed data...")
    print(f"  Train: {train_path}")
    print(f"  Dev:   {dev_path}")
    
    if not train_path.exists() or not dev_path.exists():
        raise FileNotFoundError(
            f"Processed data files not found. Please run preprocessing first.\n"
            f"Expected train: {train_path}\n"
            f"Expected dev: {dev_path}"
        )
        
    train_df = pd.read_parquet(train_path)
    dev_df = pd.read_parquet(dev_path)
    
    # 3. Vectorize text (Char n-grams 2-5 for robust handling of misspellings/slang)
    print(f"[BASELINE] Fitting TF-IDF Vectorizer...")
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=(2, 5),
        max_features=50000,
        sublinear_tf=True,
    )
    
    X_train = tfidf.fit_transform(train_df["text"])
    X_dev = tfidf.transform(dev_df["text"])
    
    y_train = train_df["label"]
    y_dev = dev_df["label"]
    
    # 4. Train Logistic Regression model
    print(f"[BASELINE] Training Logistic Regression classifier...")
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    
    # 5. Evaluate model performance
    y_pred = model.predict(X_dev)
    macro_f1 = float(f1_score(y_dev, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_dev, y_pred, average="weighted"))
    
    print(f"\n[BASELINE] Results on validation set:")
    print(f"  Macro F1 Score: {macro_f1:.4f}")
    
    report_dict = classification_report(
        y_dev,
        y_pred,
        target_names=["CLEAN", "OFFENSIVE", "HATE"],
        output_dict=True,
    )
    print("\nClassification Report:")
    print(classification_report(y_dev, y_pred, target_names=["CLEAN", "OFFENSIVE", "HATE"]))
    
    # 6. Save model artifacts
    model_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    
    joblib.dump(model, model_path)
    joblib.dump(tfidf, vectorizer_path)
    print(f"[BASELINE] Model saved to: {model_path}")
    print(f"[BASELINE] Vectorizer saved to: {vectorizer_path}")
    
    # Save metrics report
    report_payload = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_payload, f, indent=4, ensure_ascii=False)
    print(f"[BASELINE] Metrics report saved to: {report_path}")
    
    return report_payload


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate TF-IDF + Logistic Regression baseline model.")
    parser.add_argument("--config", default="configs/paths.yaml", help="Path to paths.yaml config file")
    args = parser.parse_args()
    
    try:
        train_baseline(args.config)
    except Exception as e:
        print(f"[ERROR] Baseline training failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
