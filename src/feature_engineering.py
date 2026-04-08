"""
Feature Engineering Module
===========================
Creates advanced features for cognitive state classification:
- Lagged features (previous N snapshots)
- Rolling statistics (mean, std, max over window)
- Sequential pattern features (state transition indicators)
- Interaction features (cross-feature combinations)
- Per-student normalized features (deviation from personal baseline)
"""

import yaml
import pandas as pd
import numpy as np


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def create_lagged_features(df, config):
    """
    Create lagged features: previous N snapshot values for key behavioral signals.
    This captures temporal patterns like 'idle time was increasing over last 3 snapshots'.
    """
    fe_cfg = config["feature_engineering"]
    window = fe_cfg["sliding_window_size"]
    lag_features = fe_cfg["lagged_features"]

    df = df.sort_values(["student_id", "session_id", "snapshot_index"]).reset_index(drop=True)

    for feat in lag_features:
        for lag in range(1, window + 1):
            col_name = f"{feat}_lag{lag}"
            df[col_name] = df.groupby(["student_id", "session_id"])[feat].shift(lag)

    return df


def create_rolling_features(df, config):
    """
    Create rolling statistics over a window of recent snapshots.
    Captures trends: e.g., 'average idle time is rising' = potential confusion/boredom.
    """
    fe_cfg = config["feature_engineering"]
    window = fe_cfg["rolling_stats_window"]
    stats = fe_cfg["rolling_stats"]
    lag_features = fe_cfg["lagged_features"]

    for feat in lag_features:
        grouped = df.groupby(["student_id", "session_id"])[feat]
        if "mean" in stats:
            df[f"{feat}_roll_mean"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).mean()
            )
        if "std" in stats:
            df[f"{feat}_roll_std"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).std().fillna(0)
            )
        if "max" in stats:
            df[f"{feat}_roll_max"] = grouped.transform(
                lambda x: x.rolling(window, min_periods=1).max()
            )

    return df


def create_delta_features(df):
    """
    Create change-over-time (delta) features.
    Captures acceleration: 'idle time is increasing faster' vs 'stable idle'.
    """
    behavioral = ["tab_switch", "idle_time", "clicks", "mouse_movement",
                   "replay_count", "skip_count"]

    for feat in behavioral:
        df[f"{feat}_delta"] = df.groupby(["student_id", "session_id"])[feat].diff().fillna(0)

    return df


def create_interaction_features(df):
    """
    Create cross-feature interactions that capture combined behavioral signals.
    E.g., high idle_time + high replay_count = strong confusion signal.
    """
    # Confusion indicator: high idle + high replay
    df["confusion_signal"] = df["idle_time"] * df["replay_count"]

    # Boredom indicator: high idle + high skip + high speed
    df["boredom_signal"] = df["idle_time"] * df["skip_count"] * (df["playback_speed"] - 1.0).clip(lower=0)

    # Distraction indicator: high tab switch + low clicks
    df["distraction_signal"] = df["tab_switch"] * (1.0 / (df["clicks"] + 1))

    # Engagement indicator: high clicks + high mouse movement
    df["engagement_signal"] = df["clicks"] * df["mouse_movement"] / 1000.0

    # Activity ratio: active signals vs passive signals
    df["activity_ratio"] = (df["clicks"] + df["mouse_movement"] / 100) / (df["idle_time"] + df["tab_switch"] + 1)

    return df


def create_session_position_features(df):
    """
    Create features based on position within session.
    Students may behave differently at start vs middle vs end of a session.
    """
    # Normalized position within session (0 to 1)
    session_lengths = df.groupby(["student_id", "session_id"])["snapshot_index"].transform("max")
    df["session_progress"] = df["snapshot_index"] / (session_lengths + 1)

    # Is this early, middle, or late in the session
    df["session_phase"] = pd.cut(df["session_progress"],
                                  bins=[0, 0.33, 0.66, 1.0],
                                  labels=[0, 1, 2],
                                  include_lowest=True).astype(float)

    return df


def create_student_deviation_features(df):
    """
    Create per-student normalized features: how much this snapshot deviates
    from the student's personal baseline. This enables personalization.
    """
    behavioral = ["tab_switch", "idle_time", "clicks", "mouse_movement",
                   "replay_count", "skip_count", "focus_score"]

    student_stats = df.groupby("student_id")[behavioral].agg(["mean", "std"])

    for feat in behavioral:
        mean_col = student_stats[(feat, "mean")]
        std_col = student_stats[(feat, "std")].replace(0, 1)  # avoid div by zero

        df[f"{feat}_student_zscore"] = df.apply(
            lambda row: (row[feat] - mean_col[row["student_id"]]) / std_col[row["student_id"]],
            axis=1
        )

    return df


def engineer_features(df, config_path="config/config.yaml"):
    """
    Main feature engineering pipeline.
    Returns DataFrame with all engineered features.
    """
    config = load_config(config_path)
    original_cols = len(df.columns)

    print("Feature Engineering Pipeline")
    print("=" * 50)

    print("1. Creating lagged features...")
    df = create_lagged_features(df, config)

    print("2. Creating rolling statistics...")
    df = create_rolling_features(df, config)

    print("3. Creating delta (change) features...")
    df = create_delta_features(df)

    print("4. Creating interaction features...")
    df = create_interaction_features(df)

    print("5. Creating session position features...")
    df = create_session_position_features(df)

    print("6. Creating per-student deviation features...")
    df = create_student_deviation_features(df)

    # Fill NaN from lagged/rolling with 0 (first snapshots in a session)
    new_cols = [c for c in df.columns if c not in ["student_id", "session_id", "timestamp", "state"]]
    df[new_cols] = df[new_cols].fillna(0)

    new_feature_count = len(df.columns) - original_cols
    print(f"\nTotal new features created: {new_feature_count}")
    print(f"Total columns: {len(df.columns)}")

    return df


def get_feature_columns(df):
    """Return list of feature columns (excluding identifiers and target)."""
    exclude = ["student_id", "session_id", "timestamp", "state", "snapshot_index"]
    return [c for c in df.columns if c not in exclude]


if __name__ == "__main__":
    config = load_config()
    data_path = config["dataset"]["output_path"]
    df = pd.read_csv(data_path)

    df = engineer_features(df)

    engineered_path = data_path.replace(".csv", "_engineered.csv")
    df.to_csv(engineered_path, index=False)
    print(f"\nEngineered dataset saved to: {engineered_path}")
    print(f"Shape: {df.shape}")
