"""
Per-Student Model Retraining
=============================
Progressive personalization pipeline:
- After 300+ snapshots with at least 3 classes, trains a personal Random Forest
- Uses raw features + derived features from the student's own data
- Falls back to rule-based prediction if not enough data or class diversity
- Retrains every 50 new snapshots after the initial 300

Model is saved per-student as: models_saved/students/{student_id}.joblib
"""

import os
import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from datetime import datetime


STUDENT_MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "models_saved", "students"
)

# Minimum requirements for per-student training
MIN_SNAPSHOTS = 300
MIN_CLASSES = 3
MIN_PER_CLASS = 15
RETRAIN_INTERVAL = 50  # retrain every N new snapshots


def ensure_model_dir():
    os.makedirs(STUDENT_MODELS_DIR, exist_ok=True)


def build_student_features(df):
    """
    Build feature matrix from raw snapshot data for a single student.
    Creates features that capture the student's behavioral patterns:
    - Raw features (current snapshot)
    - Rolling stats (trend over recent snapshots)
    - Deviation from personal mean (z-scores)
    - Interaction features
    """
    df = df.sort_values("snapshot_index" if "snapshot_index" in df.columns else "id")
    df = df.reset_index(drop=True)

    raw_features = ["tab_switch", "idle_time", "clicks", "mouse_movement",
                    "replay_count", "skip_count", "playback_speed", "focus_score"]

    # Ensure columns exist
    for col in raw_features:
        if col not in df.columns:
            df[col] = 0

    feature_df = df[raw_features].copy()

    # Rolling statistics (window=3) — captures recent trends
    for feat in ["tab_switch", "idle_time", "clicks", "mouse_movement",
                 "replay_count", "skip_count"]:
        feature_df[f"{feat}_roll_mean"] = df[feat].rolling(3, min_periods=1).mean()
        feature_df[f"{feat}_roll_std"] = df[feat].rolling(3, min_periods=1).std().fillna(0)
        feature_df[f"{feat}_roll_max"] = df[feat].rolling(3, min_periods=1).max()

    # Z-score features (deviation from student's own mean)
    for feat in ["tab_switch", "idle_time", "clicks", "mouse_movement",
                 "replay_count", "skip_count", "focus_score"]:
        mean = df[feat].mean()
        std = df[feat].std()
        if std > 0.01:
            feature_df[f"{feat}_zscore"] = (df[feat] - mean) / std
        else:
            feature_df[f"{feat}_zscore"] = 0

    # Lagged features (previous snapshot values)
    for feat in ["tab_switch", "idle_time", "clicks", "focus_score"]:
        feature_df[f"{feat}_lag1"] = df[feat].shift(1).fillna(0)
        feature_df[f"{feat}_lag2"] = df[feat].shift(2).fillna(0)

    # Interaction features
    feature_df["tab_idle_ratio"] = df["tab_switch"] / (df["idle_time"] + 1)
    feature_df["click_mouse_ratio"] = df["clicks"] / (df["mouse_movement"] + 1)
    feature_df["replay_skip_diff"] = df["replay_count"] - df["skip_count"]
    feature_df["speed_deviation"] = abs(df["playback_speed"] - 1.0)
    feature_df["focus_change"] = df["focus_score"].diff().fillna(0)

    # Fill any remaining NaN
    feature_df = feature_df.fillna(0)

    return feature_df


def check_training_readiness(df):
    """
    Check if the student has enough data for model training.
    Returns (ready: bool, reason: str).
    """
    if len(df) < MIN_SNAPSHOTS:
        return False, f"Need {MIN_SNAPSHOTS} snapshots, have {len(df)}"

    if "predicted_state" not in df.columns:
        return False, "No predicted_state column"

    class_counts = df["predicted_state"].value_counts()
    n_classes = len(class_counts)

    if n_classes < MIN_CLASSES:
        return False, f"Need {MIN_CLASSES} classes, have {n_classes}: {dict(class_counts)}"

    small_classes = [cls for cls, count in class_counts.items() if count < MIN_PER_CLASS]
    if small_classes:
        return False, f"Classes {small_classes} have < {MIN_PER_CLASS} samples"

    return True, "Ready"


def train_student_model(student_id, df):
    """
    Train a personalized Random Forest model for a student.

    Args:
        student_id: Student identifier
        df: DataFrame with all snapshots for this student (must include predicted_state)

    Returns:
        dict with training results, or None if training failed
    """
    ensure_model_dir()

    ready, reason = check_training_readiness(df)
    if not ready:
        return {"status": "not_ready", "reason": reason}

    # Build features
    features_df = build_student_features(df)
    labels = df["predicted_state"].values

    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(labels)

    X = features_df.values

    # Train Random Forest tuned for per-student data
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        min_samples_split=10,
        min_samples_leaf=5,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    # Cross-validation to check model quality
    cv_scores = cross_val_score(rf, X, y, cv=5, scoring="f1_weighted")
    cv_mean = round(float(cv_scores.mean()), 4)
    cv_std = round(float(cv_scores.std()), 4)

    # Only save if model is reasonably good (>60% F1)
    if cv_mean < 0.60:
        return {
            "status": "too_weak",
            "reason": f"CV F1={cv_mean} (need >0.60)",
            "cv_mean": cv_mean,
            "cv_std": cv_std,
        }

    # Train on full data
    rf.fit(X, y)

    # Save model + encoder + metadata
    model_path = os.path.join(STUDENT_MODELS_DIR, f"{student_id}.joblib")
    meta = {
        "student_id": student_id,
        "trained_at": datetime.now().isoformat(),
        "n_snapshots": len(df),
        "n_features": X.shape[1],
        "feature_names": list(features_df.columns),
        "classes": list(le.classes_),
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "class_distribution": dict(df["predicted_state"].value_counts()),
    }

    joblib.dump({
        "model": rf,
        "label_encoder": le,
        "meta": meta,
    }, model_path)

    print(f"[StudentModel] Trained for {student_id}: "
          f"CV F1={cv_mean}±{cv_std}, {len(df)} samples, "
          f"classes={list(le.classes_)}")

    return {
        "status": "trained",
        "cv_mean": cv_mean,
        "cv_std": cv_std,
        "n_snapshots": len(df),
        "classes": list(le.classes_),
        "model_path": model_path,
    }


def load_student_model(student_id):
    """Load a student's personal model if it exists."""
    model_path = os.path.join(STUDENT_MODELS_DIR, f"{student_id}.joblib")
    if not os.path.exists(model_path):
        return None
    try:
        return joblib.load(model_path)
    except Exception:
        return None


def predict_with_student_model(student_id, snapshot_df):
    """
    Predict cognitive state using the student's personal model.

    Args:
        student_id: Student identifier
        snapshot_df: DataFrame with the student's recent snapshots
                     (need at least 3 rows for rolling features, last row is current)

    Returns:
        (predicted_state, confidence) or (None, None) if no model
    """
    bundle = load_student_model(student_id)
    if bundle is None:
        return None, None

    model = bundle["model"]
    le = bundle["label_encoder"]
    meta = bundle["meta"]

    # Build features from the snapshot context
    features_df = build_student_features(snapshot_df)

    # Get the last row (current snapshot)
    X = features_df.iloc[[-1]].values

    # Verify feature count matches
    if X.shape[1] != meta["n_features"]:
        return None, None

    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
    confidence = round(float(proba.max()), 3)
    state = le.inverse_transform([pred])[0]

    return state, confidence


def should_retrain(student_id, current_snapshot_count):
    """Check if a student's model needs retraining."""
    bundle = load_student_model(student_id)

    if bundle is None:
        # No model yet — train if we have enough data
        return current_snapshot_count >= MIN_SNAPSHOTS

    # Retrain every RETRAIN_INTERVAL new snapshots
    last_trained_count = bundle["meta"]["n_snapshots"]
    new_snapshots = current_snapshot_count - last_trained_count
    return new_snapshots >= RETRAIN_INTERVAL
