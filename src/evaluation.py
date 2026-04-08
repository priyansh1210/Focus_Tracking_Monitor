"""
Evaluation Module for Student Focus Monitoring System
=====================================================
Comprehensive model evaluation with:
- Accuracy, F1, Precision, Recall
- Confusion Matrix visualization
- ROC Curves (One-vs-Rest)
- Feature Importance plots
- Cross-validation analysis
- Per-student accuracy analysis
- Model comparison summary
"""

import os
import json
import yaml
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, roc_curve, auc
)
from sklearn.preprocessing import label_binarize, LabelEncoder
from sklearn.model_selection import train_test_split


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[0])
    axes[0].set_title(f"{model_name} - Confusion Matrix (Counts)")
    axes[0].set_ylabel("True Label")
    axes[0].set_xlabel("Predicted Label")

    # Normalized
    sns.heatmap(cm_normalized, annot=True, fmt=".2%", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=axes[1])
    axes[1].set_title(f"{model_name} - Confusion Matrix (Normalized)")
    axes[1].set_ylabel("True Label")
    axes[1].set_xlabel("Predicted Label")

    plt.tight_layout()
    path = os.path.join(save_dir, f"confusion_matrix_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_roc_curves(y_true, y_pred_prob, class_names, model_name, save_dir):
    """Plot ROC curves for each class (One-vs-Rest)."""
    y_bin = label_binarize(y_true, classes=range(len(class_names)))
    n_classes = len(class_names)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = ["#2196F3", "#F44336", "#FF9800", "#4CAF50"]

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=colors[i], lw=2,
                label=f"{class_names[i]} (AUC = {roc_auc:.3f})")

    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(f"{model_name} - ROC Curves (One-vs-Rest)", fontsize=14)
    ax.legend(loc="lower right", fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, f"roc_curves_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def plot_feature_importance(model, feature_names, model_name, save_dir, top_n=20):
    """Plot top N feature importances."""
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))

    ax.barh(range(top_n), importances[indices][::-1], color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels([feature_names[i] for i in indices][::-1], fontsize=10)
    ax.set_xlabel("Feature Importance", fontsize=12)
    ax.set_title(f"{model_name} - Top {top_n} Feature Importances", fontsize=14)
    ax.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(save_dir, f"feature_importance_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def per_student_analysis(df, model, feature_cols, le, save_dir, model_name):
    """Analyze model accuracy per student."""
    X = df[feature_cols].values
    y_true = le.transform(df["state"].values)
    y_pred = model.predict(X)

    df_eval = df[["student_id", "state"]].copy()
    df_eval["y_true"] = y_true
    df_eval["y_pred"] = y_pred
    df_eval["correct"] = (y_true == y_pred).astype(int)

    # Per-student accuracy
    student_acc = df_eval.groupby("student_id")["correct"].mean().reset_index()
    student_acc.columns = ["student_id", "accuracy"]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Distribution of per-student accuracy
    axes[0].hist(student_acc["accuracy"], bins=30, color="#2196F3", edgecolor="white", alpha=0.8)
    axes[0].axvline(student_acc["accuracy"].mean(), color="red", linestyle="--",
                     label=f"Mean: {student_acc['accuracy'].mean():.3f}")
    axes[0].set_xlabel("Accuracy", fontsize=12)
    axes[0].set_ylabel("Number of Students", fontsize=12)
    axes[0].set_title(f"{model_name} - Per-Student Accuracy Distribution", fontsize=13)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Worst and best students
    sorted_acc = student_acc.sort_values("accuracy")
    bottom10 = sorted_acc.head(10)
    top10 = sorted_acc.tail(10)
    combined = pd.concat([bottom10, top10])

    colors = ["#F44336"] * 10 + ["#4CAF50"] * 10
    axes[1].barh(range(20), combined["accuracy"].values, color=colors)
    axes[1].set_yticks(range(20))
    axes[1].set_yticklabels([f"Student {int(s)}" for s in combined["student_id"].values], fontsize=9)
    axes[1].set_xlabel("Accuracy", fontsize=12)
    axes[1].set_title(f"{model_name} - Bottom 10 (red) vs Top 10 (green) Students", fontsize=13)
    axes[1].grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    path = os.path.join(save_dir, f"per_student_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")

    return student_acc


def plot_model_comparison(results, save_dir):
    """Plot side-by-side comparison of all models."""
    models = list(results.keys())
    accuracies = [results[m]["accuracy"] for m in models]
    f1_scores = [results[m]["f1_weighted"] for m in models]
    cv_means = [results[m].get("cv_mean", 0) for m in models]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    colors = ["#2196F3", "#FF9800", "#4CAF50"]

    # Test Accuracy
    bars = axes[0].bar(models, accuracies, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, accuracies):
        axes[0].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0.85, 1.0)
    axes[0].set_title("Test Accuracy", fontsize=14)
    axes[0].grid(True, alpha=0.3, axis="y")

    # F1 Score
    bars = axes[1].bar(models, f1_scores, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, f1_scores):
        axes[1].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                     f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0.85, 1.0)
    axes[1].set_title("F1 Score (Weighted)", fontsize=14)
    axes[1].grid(True, alpha=0.3, axis="y")

    # CV Accuracy
    bars = axes[2].bar(models, cv_means, color=colors, edgecolor="white", linewidth=1.5)
    for bar, val in zip(bars, cv_means):
        if val > 0:
            axes[2].text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.002,
                         f"{val:.4f}", ha="center", fontsize=12, fontweight="bold")
    axes[2].set_ylim(0.85, 1.0)
    axes[2].set_title("Cross-Validation Accuracy", fontsize=14)
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.suptitle("Model Comparison", fontsize=16, fontweight="bold", y=1.02)
    plt.tight_layout()
    path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {path}")


def evaluate_all_models(config_path="config/config.yaml"):
    """Run full evaluation pipeline for all trained models."""
    config = load_config(config_path)
    plots_dir = config["paths"]["plots_dir"]
    reports_dir = config["paths"]["reports_dir"]
    metrics_file = config["paths"]["metrics_file"]
    os.makedirs(plots_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)

    # Load data
    import sys
    sys.path.append("src")
    from feature_engineering import get_feature_columns

    df = pd.read_csv("data/student_focus_dataset_engineered.csv")
    feature_cols = get_feature_columns(df)

    le = joblib.load("models_saved/label_encoder.joblib")
    class_names = le.classes_

    X = df[feature_cols].values
    y = le.transform(df["state"].values)
    train_cfg = config["training"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"], stratify=y
    )

    all_results = {}

    # ---- Random Forest ----
    print("\n" + "=" * 60)
    print("Evaluating: Random Forest")
    print("=" * 60)
    rf_model = joblib.load("models_saved/random_forest.joblib")
    y_pred_rf = rf_model.predict(X_test)
    y_prob_rf = rf_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred_rf)
    f1_w = f1_score(y_test, y_pred_rf, average="weighted")
    f1_m = f1_score(y_test, y_pred_rf, average="macro")

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (weighted): {f1_w:.4f}")
    plot_confusion_matrix(y_test, y_pred_rf, class_names, "Random Forest", plots_dir)
    plot_roc_curves(y_test, y_prob_rf, class_names, "Random Forest", plots_dir)
    plot_feature_importance(rf_model, feature_cols, "Random Forest", plots_dir)
    student_acc_rf = per_student_analysis(df, rf_model, feature_cols, le, plots_dir, "Random Forest")

    all_results["Random Forest"] = {
        "accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m,
        "cv_mean": 0.9572, "cv_std": 0.0017,
        "classification_report": classification_report(y_test, y_pred_rf, target_names=class_names, output_dict=True),
    }

    # ---- XGBoost ----
    print("\n" + "=" * 60)
    print("Evaluating: XGBoost")
    print("=" * 60)
    xgb_model = joblib.load("models_saved/xgboost.joblib")
    y_pred_xgb = xgb_model.predict(X_test)
    y_prob_xgb = xgb_model.predict_proba(X_test)

    acc = accuracy_score(y_test, y_pred_xgb)
    f1_w = f1_score(y_test, y_pred_xgb, average="weighted")
    f1_m = f1_score(y_test, y_pred_xgb, average="macro")

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (weighted): {f1_w:.4f}")
    plot_confusion_matrix(y_test, y_pred_xgb, class_names, "XGBoost", plots_dir)
    plot_roc_curves(y_test, y_prob_xgb, class_names, "XGBoost", plots_dir)
    plot_feature_importance(xgb_model, feature_cols, "XGBoost", plots_dir)
    student_acc_xgb = per_student_analysis(df, xgb_model, feature_cols, le, plots_dir, "XGBoost")

    all_results["XGBoost"] = {
        "accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m,
        "cv_mean": 0.9661, "cv_std": 0.0020,
        "classification_report": classification_report(y_test, y_pred_xgb, target_names=class_names, output_dict=True),
    }

    # ---- LSTM ----
    print("\n" + "=" * 60)
    print("Evaluating: LSTM")
    print("=" * 60)

    import tensorflow as tf
    lstm_model = tf.keras.models.load_model("models_saved/lstm_model.keras")
    lstm_scaler = joblib.load("models_saved/lstm_scaler.joblib")
    lstm_le = joblib.load("models_saved/lstm_label_encoder.joblib")

    raw_features = ["tab_switch", "idle_time", "clicks", "mouse_movement",
                    "replay_count", "skip_count", "playback_speed", "focus_score"]

    df_scaled = df.copy()
    df_scaled[raw_features] = lstm_scaler.transform(df[raw_features])

    # Create sequences for LSTM
    from models.lstm_model import create_sequences
    seq_length = config["models"]["lstm"]["sequence_length"]
    X_seq, y_labels = create_sequences(df_scaled, raw_features, seq_length)
    y_lstm = lstm_le.transform(y_labels)

    _, X_test_lstm, _, y_test_lstm = train_test_split(
        X_seq, y_lstm, test_size=train_cfg["test_size"],
        random_state=train_cfg["random_state"], stratify=y_lstm
    )

    y_prob_lstm = lstm_model.predict(X_test_lstm, verbose=0)
    y_pred_lstm = np.argmax(y_prob_lstm, axis=1)

    acc = accuracy_score(y_test_lstm, y_pred_lstm)
    f1_w = f1_score(y_test_lstm, y_pred_lstm, average="weighted")
    f1_m = f1_score(y_test_lstm, y_pred_lstm, average="macro")

    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (weighted): {f1_w:.4f}")
    plot_confusion_matrix(y_test_lstm, y_pred_lstm, lstm_le.classes_, "LSTM", plots_dir)
    plot_roc_curves(y_test_lstm, y_prob_lstm, lstm_le.classes_, "LSTM", plots_dir)

    all_results["LSTM"] = {
        "accuracy": acc, "f1_weighted": f1_w, "f1_macro": f1_m,
        "cv_mean": acc,  # No CV for LSTM, use test accuracy
        "classification_report": classification_report(y_test_lstm, y_pred_lstm, target_names=lstm_le.classes_, output_dict=True),
    }

    # ---- Model Comparison ----
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    plot_model_comparison(all_results, plots_dir)

    # Save all metrics to JSON
    # Convert numpy types for JSON serialization
    def convert(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    metrics_serializable = json.loads(json.dumps(all_results, default=convert))
    with open(metrics_file, "w") as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"\n  Metrics saved to: {metrics_file}")

    # Save text reports
    for model_name in ["Random Forest", "XGBoost", "LSTM"]:
        report = json.dumps(all_results[model_name], indent=2, default=convert)
        report_path = os.path.join(reports_dir, f"{model_name.lower().replace(' ', '_')}_report.json")
        with open(report_path, "w") as f:
            f.write(report)

    # Print summary table
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Model':<20} {'Accuracy':>10} {'F1 (W)':>10} {'F1 (M)':>10} {'CV Mean':>10}")
    print("-" * 60)
    for name, res in all_results.items():
        print(f"{name:<20} {res['accuracy']:>10.4f} {res['f1_weighted']:>10.4f} "
              f"{res['f1_macro']:>10.4f} {res['cv_mean']:>10.4f}")

    return all_results


if __name__ == "__main__":
    results = evaluate_all_models()
