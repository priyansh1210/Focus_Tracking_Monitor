"""
Random Forest Classifier for Student Cognitive State Classification
"""

import yaml
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from feature_engineering import get_feature_columns


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train_random_forest(df, config_path="config/config.yaml"):
    """Train Random Forest model and return model, encoder, and split data."""
    config = load_config(config_path)
    rf_cfg = config["models"]["random_forest"]
    train_cfg = config["training"]

    feature_cols = get_feature_columns(df)
    X = df[feature_cols].values
    y = df["state"].values

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
        stratify=y_encoded
    )

    print(f"Training Random Forest on {X_train.shape[0]} samples, {X_train.shape[1]} features...")

    model = RandomForestClassifier(
        n_estimators=rf_cfg["n_estimators"],
        max_depth=rf_cfg["max_depth"],
        min_samples_split=rf_cfg["min_samples_split"],
        min_samples_leaf=rf_cfg["min_samples_leaf"],
        class_weight=rf_cfg["class_weight"],
        random_state=rf_cfg["random_state"],
        n_jobs=rf_cfg["n_jobs"],
    )

    model.fit(X_train, y_train)

    # Quick evaluation
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"Test Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Cross-validation
    print(f"Running {train_cfg['cross_validation_folds']}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=train_cfg["cross_validation_folds"],
                         shuffle=True, random_state=train_cfg["random_state"])
    cv_scores = cross_val_score(model, X, y_encoded, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

    # Save model
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "random_forest.joblib")
    joblib.dump(model, model_path)
    le_path = os.path.join(models_dir, "label_encoder.joblib")
    joblib.dump(le, le_path)
    print(f"\nModel saved to: {model_path}")

    return {
        "model": model,
        "label_encoder": le,
        "feature_cols": feature_cols,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
        "cv_scores": cv_scores,
        "accuracy": acc,
    }


if __name__ == "__main__":
    config = load_config()
    df = pd.read_csv("data/student_focus_dataset_engineered.csv")
    results = train_random_forest(df)
