"""
LSTM Model for Student Cognitive State Classification
=====================================================
Processes sequences of behavioral snapshots to capture temporal patterns.
Unlike RF/XGBoost which use engineered lag features, LSTM learns
temporal dependencies directly from raw sequences.
"""

import yaml
import joblib
import numpy as np
import pandas as pd
import os
import json

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import classification_report, accuracy_score


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_sequences(df, feature_cols, seq_length):
    """
    Convert flat DataFrame into sequences for LSTM.
    Each sequence = last `seq_length` snapshots for a student within a session.
    Target = state of the last snapshot in the sequence.
    """
    sequences = []
    labels = []

    df = df.sort_values(["student_id", "session_id", "snapshot_index"]).reset_index(drop=True)

    for (sid, sess), group in df.groupby(["student_id", "session_id"]):
        features = group[feature_cols].values
        states = group["state"].values

        for i in range(len(group)):
            if i < seq_length - 1:
                # Pad shorter sequences with zeros
                pad_length = seq_length - 1 - i
                seq = np.vstack([
                    np.zeros((pad_length, len(feature_cols))),
                    features[:i + 1]
                ])
            else:
                seq = features[i - seq_length + 1:i + 1]

            sequences.append(seq)
            labels.append(states[i])

    return np.array(sequences), np.array(labels)


def train_lstm(df, config_path="config/config.yaml"):
    """Train LSTM model on sequential behavioral data."""
    config = load_config(config_path)
    lstm_cfg = config["models"]["lstm"]
    train_cfg = config["training"]

    # Use raw behavioral features (not engineered) for LSTM
    raw_features = ["tab_switch", "idle_time", "clicks", "mouse_movement",
                    "replay_count", "skip_count", "playback_speed", "focus_score"]

    # Scale features
    scaler = StandardScaler()
    df_scaled = df.copy()
    df_scaled[raw_features] = scaler.fit_transform(df[raw_features])

    # Create sequences
    seq_length = lstm_cfg["sequence_length"]
    print(f"Creating sequences (length={seq_length})...")
    X_seq, y_labels = create_sequences(df_scaled, raw_features, seq_length)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_labels)

    print(f"Sequence shape: {X_seq.shape}")
    print(f"Classes: {le.classes_}")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_seq, y_encoded,
        test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"],
        stratify=y_encoded
    )

    # Build LSTM model with GPU support
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
    from tensorflow.keras.utils import to_categorical

    # Detect GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"  [GPU detected: {gpus[0].name}]")
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    else:
        print("  [No GPU detected, using CPU]")

    num_classes = len(le.classes_)
    y_train_cat = to_categorical(y_train, num_classes)
    y_test_cat = to_categorical(y_test, num_classes)

    print(f"\nTraining LSTM on {X_train.shape[0]} sequences...")

    model = Sequential([
        LSTM(lstm_cfg["hidden_units"],
             input_shape=(seq_length, len(raw_features)),
             dropout=lstm_cfg["dropout"],
             recurrent_dropout=lstm_cfg["recurrent_dropout"],
             return_sequences=True),
        BatchNormalization(),
        LSTM(lstm_cfg["hidden_units"] // 2,
             dropout=lstm_cfg["dropout"]),
        BatchNormalization(),
        Dense(lstm_cfg["dense_units"], activation="relu"),
        Dropout(lstm_cfg["dropout"]),
        Dense(num_classes, activation="softmax"),
    ])

    optimizer = tf.keras.optimizers.Adam(learning_rate=lstm_cfg["learning_rate"])
    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=["accuracy"])

    model.summary()

    callbacks = [
        EarlyStopping(patience=lstm_cfg["early_stopping_patience"],
                      restore_best_weights=True, monitor="val_loss"),
        ReduceLROnPlateau(factor=0.5, patience=3, min_lr=1e-6),
    ]

    history = model.fit(
        X_train, y_train_cat,
        validation_split=lstm_cfg["validation_split"],
        epochs=lstm_cfg["epochs"],
        batch_size=lstm_cfg["batch_size"],
        callbacks=callbacks,
        verbose=1,
    )

    # Evaluate
    y_pred_prob = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_prob, axis=1)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Save model
    models_dir = config["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)
    model_path = os.path.join(models_dir, "lstm_model.keras")
    model.save(model_path)
    joblib.dump(scaler, os.path.join(models_dir, "lstm_scaler.joblib"))
    joblib.dump(le, os.path.join(models_dir, "lstm_label_encoder.joblib"))
    print(f"\nModel saved to: {model_path}")

    # Save training history
    hist_path = os.path.join(models_dir, "lstm_history.json")
    hist_data = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    with open(hist_path, "w") as f:
        json.dump(hist_data, f, indent=2)

    return {
        "model": model,
        "label_encoder": le,
        "scaler": scaler,
        "history": history.history,
        "X_train": X_train, "X_test": X_test,
        "y_train": y_train, "y_test": y_test,
        "y_pred": y_pred,
        "y_pred_prob": y_pred_prob,
        "accuracy": acc,
        "feature_cols": raw_features,
    }


if __name__ == "__main__":
    config = load_config()
    df = pd.read_csv("data/student_focus_dataset_engineered.csv")
    results = train_lstm(df)
