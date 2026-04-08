"""
Synthetic Dataset Generator for Student Focus Monitoring System
===============================================================
Generates 40,000 rows of realistic student behavioral data with:
- Per-student personalization (unique behavioral baselines)
- Temporal coherence within sessions (state transitions, not random jumps)
- Balanced classes: focused, distracted, confused, bored (10k each)
- Controlled randomness for real-world variability
"""

import os
import yaml
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def load_config(config_path="config/config.yaml"):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def generate_student_profiles(num_students, rng):
    """
    Generate unique behavioral baselines for each student.
    Each student has personal tendencies that affect how their behavior
    maps to cognitive states.

    Returns dict of student_id -> profile dict.
    """
    profiles = {}
    for sid in range(1, num_students + 1):
        profiles[sid] = {
            # Base activity level (some students naturally click more)
            "base_click_rate": rng.uniform(10, 60),
            "base_mouse_activity": rng.uniform(500, 3000),
            # Tab switching tendency (some students multitask more)
            "tab_switch_tendency": rng.uniform(0.5, 2.0),
            # Replay behavior (some students replay more as learning style)
            "replay_tendency": rng.uniform(0.5, 2.5),
            # Skip tolerance (high = skipping is normal for this student)
            "skip_tolerance": rng.uniform(0.3, 2.0),
            # Idle threshold (some students pause to think more)
            "idle_baseline": rng.uniform(5, 40),
            # Preferred playback speed
            "preferred_speed": rng.choice([1.0, 1.0, 1.0, 1.25, 1.5, 1.75, 2.0]),
            # Focus stability (how likely to stay in same state)
            "state_persistence": rng.uniform(0.6, 0.9),
            # Time-of-day preference (morning=0, afternoon=1, evening=2)
            "active_period": rng.choice([0, 1, 2]),
        }
    return profiles


def generate_behavioral_snapshot(state, profile, prev_snapshot, rng):
    """
    Generate a single behavioral snapshot based on cognitive state and student profile.
    Uses the previous snapshot for temporal coherence.
    """
    p = profile

    # State-specific behavioral distributions
    state_params = {
        "focused": {
            "tab_switch": (0.3, 0.8),       # (mean_factor, std_factor) relative to tendency
            "idle_time": (0.5, 0.3),
            "clicks": (1.2, 0.3),
            "mouse_movement": (1.3, 0.3),
            "replay_count": (0.3, 0.4),
            "skip_count": (0.1, 0.2),
            "speed_deviation": (0.0, 0.1),
        },
        "distracted": {
            "tab_switch": (3.0, 1.0),
            "idle_time": (1.5, 0.8),
            "clicks": (0.4, 0.3),
            "mouse_movement": (0.5, 0.4),
            "replay_count": (0.2, 0.3),
            "skip_count": (0.8, 0.5),
            "speed_deviation": (0.3, 0.3),
        },
        "confused": {
            "tab_switch": (1.0, 0.5),
            "idle_time": (2.0, 0.8),
            "clicks": (0.6, 0.3),
            "mouse_movement": (0.7, 0.4),
            "replay_count": (2.5, 0.8),
            "skip_count": (0.2, 0.3),
            "speed_deviation": (-0.3, 0.15),  # slows down
        },
        "bored": {
            "tab_switch": (1.8, 0.7),
            "idle_time": (2.5, 1.0),
            "clicks": (0.3, 0.2),
            "mouse_movement": (0.3, 0.3),
            "replay_count": (0.1, 0.2),
            "skip_count": (2.5, 0.8),
            "speed_deviation": (0.7, 0.3),    # speeds up to skip through
        },
    }

    sp = state_params[state]
    temporal_weight = 0.3  # how much previous snapshot influences current

    # Generate each feature
    tab_switch_raw = max(0, rng.normal(
        sp["tab_switch"][0] * p["tab_switch_tendency"],
        sp["tab_switch"][1]
    ))
    if prev_snapshot is not None:
        tab_switch_raw = (1 - temporal_weight) * tab_switch_raw + temporal_weight * prev_snapshot["tab_switch"] / max(p["tab_switch_tendency"], 0.5)
    tab_switch = int(np.clip(round(tab_switch_raw * p["tab_switch_tendency"]), 0, 20))

    idle_raw = max(0, rng.normal(
        sp["idle_time"][0] * p["idle_baseline"],
        sp["idle_time"][1] * p["idle_baseline"]
    ))
    if prev_snapshot is not None:
        idle_raw = (1 - temporal_weight) * idle_raw + temporal_weight * prev_snapshot["idle_time"]
    idle_time = round(np.clip(idle_raw, 0, 300), 1)

    clicks_raw = max(0, rng.normal(
        sp["clicks"][0] * p["base_click_rate"],
        sp["clicks"][1] * p["base_click_rate"]
    ))
    if prev_snapshot is not None:
        clicks_raw = (1 - temporal_weight) * clicks_raw + temporal_weight * prev_snapshot["clicks"]
    clicks = int(np.clip(round(clicks_raw), 0, 100))

    mouse_raw = max(0, rng.normal(
        sp["mouse_movement"][0] * p["base_mouse_activity"],
        sp["mouse_movement"][1] * p["base_mouse_activity"]
    ))
    if prev_snapshot is not None:
        mouse_raw = (1 - temporal_weight) * mouse_raw + temporal_weight * prev_snapshot["mouse_movement"]
    mouse_movement = round(np.clip(mouse_raw, 0, 5000), 1)

    replay_raw = max(0, rng.normal(
        sp["replay_count"][0] * p["replay_tendency"],
        sp["replay_count"][1] * p["replay_tendency"]
    ))
    if prev_snapshot is not None:
        replay_raw = (1 - temporal_weight) * replay_raw + temporal_weight * prev_snapshot["replay_count"] / max(p["replay_tendency"], 0.5)
    replay_count = int(np.clip(round(replay_raw * p["replay_tendency"]), 0, 15))

    skip_raw = max(0, rng.normal(
        sp["skip_count"][0] * p["skip_tolerance"],
        sp["skip_count"][1] * p["skip_tolerance"]
    ))
    if prev_snapshot is not None:
        skip_raw = (1 - temporal_weight) * skip_raw + temporal_weight * prev_snapshot["skip_count"] / max(p["skip_tolerance"], 0.3)
    skip_count = int(np.clip(round(skip_raw * p["skip_tolerance"]), 0, 10))

    speed_dev = rng.normal(sp["speed_deviation"][0], sp["speed_deviation"][1])
    playback_speed = round(np.clip(p["preferred_speed"] + speed_dev, 0.5, 3.0), 2)

    return {
        "tab_switch": tab_switch,
        "idle_time": idle_time,
        "clicks": clicks,
        "mouse_movement": mouse_movement,
        "replay_count": replay_count,
        "skip_count": skip_count,
        "playback_speed": playback_speed,
    }


def get_state_transition(current_state, profile, rng):
    """
    Determine next cognitive state based on current state and student's persistence.
    Models realistic transitions (e.g., focused -> distracted is more likely than focused -> confused).
    """
    persistence = profile["state_persistence"]

    # Transition probabilities: current_state -> {next_state: probability}
    transition_matrix = {
        "focused": {
            "focused": persistence,
            "distracted": (1 - persistence) * 0.50,
            "confused": (1 - persistence) * 0.30,
            "bored": (1 - persistence) * 0.20,
        },
        "distracted": {
            "focused": (1 - persistence) * 0.35,
            "distracted": persistence,
            "confused": (1 - persistence) * 0.25,
            "bored": (1 - persistence) * 0.40,
        },
        "confused": {
            "focused": (1 - persistence) * 0.25,
            "distracted": (1 - persistence) * 0.30,
            "confused": persistence,
            "bored": (1 - persistence) * 0.45,
        },
        "bored": {
            "focused": (1 - persistence) * 0.20,
            "distracted": (1 - persistence) * 0.45,
            "confused": (1 - persistence) * 0.35,
            "bored": persistence,
        },
    }

    probs = transition_matrix[current_state]
    states = list(probs.keys())
    probabilities = np.array(list(probs.values()))
    probabilities /= probabilities.sum()  # normalize

    return rng.choice(states, p=probabilities)


def generate_dataset(config_path="config/config.yaml"):
    """
    Main dataset generation function.
    Produces balanced, temporally coherent synthetic data.
    """
    config = load_config(config_path)
    ds_cfg = config["dataset"]

    rng = np.random.default_rng(ds_cfg["random_seed"])
    num_students = ds_cfg["num_students"]
    total_rows = ds_cfg["total_rows"]
    rows_per_class = ds_cfg["rows_per_class"]
    classes = ds_cfg["classes"]

    print(f"Generating {total_rows} rows for {num_students} students...")
    print(f"Target: {rows_per_class} rows per class (balanced)")

    # Step 1: Generate student profiles
    profiles = generate_student_profiles(num_students, rng)

    # Step 2: Generate raw data with temporal coherence
    # We oversample and then balance
    all_rows = []
    class_counts = {c: 0 for c in classes}
    target_total = total_rows + 5000  # oversample buffer

    student_ids = list(profiles.keys())
    session_counter = 0

    base_time = datetime(2025, 9, 1, 8, 0, 0)

    while sum(class_counts.values()) < target_total:
        student_id = rng.choice(student_ids)
        profile = profiles[student_id]
        session_counter += 1
        session_id = f"S{session_counter:06d}"

        # Session length: 5-15 snapshots
        num_snapshots = rng.integers(5, 16)

        # Session start time (realistic: morning/afternoon/evening)
        day_offset = rng.integers(0, 120)
        hour_offsets = {0: (8, 12), 1: (13, 17), 2: (18, 22)}
        h_range = hour_offsets[profile["active_period"]]
        hour = rng.integers(h_range[0], h_range[1])
        session_start = base_time + timedelta(days=int(day_offset), hours=int(hour),
                                               minutes=int(rng.integers(0, 60)))

        # Initial state
        current_state = rng.choice(classes)
        prev_snapshot = None

        for snap_idx in range(num_snapshots):
            if sum(class_counts.values()) >= target_total:
                break

            # Timestamp: 30s - 3min between snapshots
            timestamp = session_start + timedelta(seconds=int(snap_idx * rng.integers(30, 180)))

            # Generate behavioral data
            snapshot = generate_behavioral_snapshot(current_state, profile, prev_snapshot, rng)

            row = {
                "student_id": student_id,
                "session_id": session_id,
                "timestamp": timestamp.strftime("%Y-%m-%d %H:%M:%S"),
                "snapshot_index": snap_idx,
                **snapshot,
                "state": current_state,
            }

            all_rows.append(row)
            class_counts[current_state] += 1
            prev_snapshot = snapshot

            # Transition to next state
            current_state = get_state_transition(current_state, profile, rng)

    # Step 3: Balance the dataset
    print(f"\nRaw generation complete. Class distribution before balancing:")
    for c, count in class_counts.items():
        print(f"  {c}: {count}")

    df = pd.DataFrame(all_rows)
    balanced_dfs = []
    for cls in classes:
        cls_df = df[df["state"] == cls]
        if len(cls_df) >= rows_per_class:
            balanced_dfs.append(cls_df.sample(n=rows_per_class, random_state=42))
        else:
            # Oversample if needed
            balanced_dfs.append(cls_df.sample(n=rows_per_class, replace=True, random_state=42))

    df_balanced = pd.concat(balanced_dfs, ignore_index=True)

    # Shuffle while keeping session order within groups
    df_balanced = df_balanced.sort_values(["student_id", "session_id", "snapshot_index"]).reset_index(drop=True)

    # Step 4: Verify and save
    print(f"\nFinal dataset shape: {df_balanced.shape}")
    print(f"Class distribution:")
    print(df_balanced["state"].value_counts())
    print(f"\nUnique students: {df_balanced['student_id'].nunique()}")
    print(f"Unique sessions: {df_balanced['session_id'].nunique()}")

    # Save
    output_path = ds_cfg["output_path"]
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_balanced.to_csv(output_path, index=False)
    print(f"\nDataset saved to: {output_path}")

    return df_balanced


if __name__ == "__main__":
    df = generate_dataset()
    print("\n--- Sample rows ---")
    print(df.head(10).to_string())
    print("\n--- Feature statistics ---")
    print(df.describe().round(2).to_string())
