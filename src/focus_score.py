"""
Dynamic Focus Score Computation
================================
Computes a continuous focus score (0-100) for each snapshot using:
- Weighted behavioral signals
- Temporal decay (recent activity matters more)
- Per-student baseline normalization
- Rolling smoothing for stability
"""

import yaml
import pandas as pd
import numpy as np


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def compute_raw_focus_score(row, weights, student_baseline=None):
    """
    Compute raw focus score for a single snapshot.
    Score starts at 70 (neutral) and adjusts based on weighted signals.
    Per-student baseline normalization makes the score personalized.
    """
    base_score = 70.0

    # Normalize features against student baseline if available
    if student_baseline is not None:
        tab_dev = row["tab_switch"] - student_baseline.get("tab_switch_mean", 0)
        idle_dev = row["idle_time"] - student_baseline.get("idle_time_mean", 0)
        click_dev = row["clicks"] - student_baseline.get("clicks_mean", 0)
        mouse_dev = row["mouse_movement"] - student_baseline.get("mouse_movement_mean", 0)
        replay_dev = row["replay_count"] - student_baseline.get("replay_count_mean", 0)
        skip_dev = row["skip_count"] - student_baseline.get("skip_count_mean", 0)
    else:
        tab_dev = row["tab_switch"]
        idle_dev = row["idle_time"]
        click_dev = row["clicks"]
        mouse_dev = row["mouse_movement"]
        replay_dev = row["replay_count"]
        skip_dev = row["skip_count"]

    speed_dev = abs(row["playback_speed"] - 1.0)

    score = base_score
    score += weights["tab_switch"] * tab_dev
    score += weights["idle_time"] * idle_dev
    score += weights["clicks"] * click_dev
    score += weights["mouse_movement"] * mouse_dev
    score += weights["replay_count"] * replay_dev
    score += weights["skip_count"] * skip_dev
    score += weights["playback_speed_deviation"] * speed_dev

    return np.clip(score, 0, 100)


def compute_student_baselines(df):
    """Compute per-student behavioral baselines from their history."""
    baselines = {}
    features = ["tab_switch", "idle_time", "clicks", "mouse_movement",
                 "replay_count", "skip_count"]

    for student_id, group in df.groupby("student_id"):
        baselines[student_id] = {
            f"{feat}_mean": group[feat].mean()
            for feat in features
        }
        baselines[student_id].update({
            f"{feat}_std": group[feat].std()
            for feat in features
        })
    return baselines


def apply_temporal_smoothing(scores, window, decay):
    """
    Apply exponentially weighted moving average for temporal smoothing.
    Recent snapshots have more influence on the focus score.
    """
    smoothed = []
    for i in range(len(scores)):
        start = max(0, i - window + 1)
        window_scores = scores[start:i + 1]
        # Apply exponential decay weights (most recent = highest weight)
        weights = [decay ** (len(window_scores) - 1 - j) for j in range(len(window_scores))]
        weighted_avg = np.average(window_scores, weights=weights)
        smoothed.append(round(weighted_avg, 2))
    return smoothed


def compute_focus_scores(df, config_path="config/config.yaml"):
    """
    Compute dynamic focus scores for the entire dataset.
    Returns DataFrame with focus_score column added.
    """
    config = load_config(config_path)
    fs_cfg = config["focus_score"]
    weights = fs_cfg["weights"]
    decay = fs_cfg["temporal_decay"]
    window = fs_cfg["smoothing_window"]

    print("Computing per-student baselines...")
    baselines = compute_student_baselines(df)

    print("Computing raw focus scores...")
    df = df.copy()
    df = df.sort_values(["student_id", "session_id", "snapshot_index"]).reset_index(drop=True)

    # Compute raw scores
    raw_scores = []
    for _, row in df.iterrows():
        baseline = baselines.get(row["student_id"])
        score = compute_raw_focus_score(row, weights, baseline)
        raw_scores.append(score)

    df["focus_score_raw"] = raw_scores

    # Apply temporal smoothing per student-session
    print("Applying temporal smoothing...")
    smoothed_scores = []
    for (sid, sess), group in df.groupby(["student_id", "session_id"], sort=False):
        group_scores = group["focus_score_raw"].values.tolist()
        smoothed = apply_temporal_smoothing(group_scores, window, decay)
        smoothed_scores.extend(smoothed)

    df["focus_score"] = smoothed_scores
    df = df.drop(columns=["focus_score_raw"])

    print(f"Focus score stats: min={df['focus_score'].min():.1f}, "
          f"max={df['focus_score'].max():.1f}, mean={df['focus_score'].mean():.1f}")

    # Show per-state averages
    print("\nAverage focus score by state:")
    for state in ["focused", "distracted", "confused", "bored"]:
        mean_score = df[df["state"] == state]["focus_score"].mean()
        print(f"  {state}: {mean_score:.1f}")

    return df


if __name__ == "__main__":
    config = load_config()
    data_path = config["dataset"]["output_path"]
    df = pd.read_csv(data_path)
    df = compute_focus_scores(df)
    df.to_csv(data_path, index=False)
    print(f"\nUpdated dataset saved with focus_score column.")
