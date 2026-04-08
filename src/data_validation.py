"""
Data Validation Module for Student Focus Monitoring System
==========================================================
Validates the generated dataset for:
- Missing values
- Unrealistic value ranges
- Class balance
- Temporal coherence
- Per-student statistics
"""

import os
import yaml
import pandas as pd
import numpy as np


def load_config(config_path="config/config.yaml"):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_missing_values(df):
    """Check for missing values in all columns."""
    print("=" * 60)
    print("1. MISSING VALUES CHECK")
    print("=" * 60)
    missing = df.isnull().sum()
    total_missing = missing.sum()

    if total_missing == 0:
        print("   [PASS] No missing values found.")
    else:
        print("   [FAIL] Missing values detected:")
        for col, count in missing[missing > 0].items():
            print(f"     - {col}: {count} ({count/len(df)*100:.2f}%)")

    return total_missing == 0


def validate_value_ranges(df, config):
    """Check that all features are within expected ranges."""
    print("\n" + "=" * 60)
    print("2. VALUE RANGE CHECK")
    print("=" * 60)

    features = config["dataset"]["features"]
    all_pass = True

    range_checks = {
        "tab_switch": (features["tab_switch"]["min"], features["tab_switch"]["max"]),
        "idle_time": (features["idle_time"]["min"], features["idle_time"]["max"]),
        "clicks": (features["clicks"]["min"], features["clicks"]["max"]),
        "mouse_movement": (features["mouse_movement"]["min"], features["mouse_movement"]["max"]),
        "replay_count": (features["replay_count"]["min"], features["replay_count"]["max"]),
        "skip_count": (features["skip_count"]["min"], features["skip_count"]["max"]),
        "playback_speed": (features["playback_speed"]["min"], features["playback_speed"]["max"]),
    }

    for col, (vmin, vmax) in range_checks.items():
        actual_min = df[col].min()
        actual_max = df[col].max()
        in_range = actual_min >= vmin and actual_max <= vmax

        status = "[PASS]" if in_range else "[FAIL]"
        print(f"   {status} {col}: [{actual_min}, {actual_max}] (expected [{vmin}, {vmax}])")
        if not in_range:
            all_pass = False
            out_of_range = ((df[col] < vmin) | (df[col] > vmax)).sum()
            print(f"          {out_of_range} rows out of range")

    return all_pass


def validate_class_balance(df, config):
    """Check class distribution balance."""
    print("\n" + "=" * 60)
    print("3. CLASS BALANCE CHECK")
    print("=" * 60)

    expected_per_class = config["dataset"]["rows_per_class"]
    class_counts = df["state"].value_counts()
    tolerance = 0.02  # 2% tolerance

    all_pass = True
    for cls in config["dataset"]["classes"]:
        count = class_counts.get(cls, 0)
        ratio = count / expected_per_class
        deviation = abs(ratio - 1.0)
        status = "[PASS]" if deviation <= tolerance else "[FAIL]"
        print(f"   {status} {cls}: {count} (expected {expected_per_class}, deviation: {deviation*100:.1f}%)")
        if deviation > tolerance:
            all_pass = False

    # Chi-squared test for uniformity
    from scipy import stats
    observed = [class_counts.get(c, 0) for c in config["dataset"]["classes"]]
    expected = [expected_per_class] * len(config["dataset"]["classes"])
    chi2, p_value = stats.chisquare(observed, expected)
    print(f"\n   Chi-squared test: chi2={chi2:.4f}, p-value={p_value:.4f}")
    print(f"   {'[PASS]' if p_value > 0.05 else '[WARN]'} Distribution uniformity (p > 0.05)")

    return all_pass


def validate_student_coverage(df, config):
    """Check student distribution and session coverage."""
    print("\n" + "=" * 60)
    print("4. STUDENT COVERAGE CHECK")
    print("=" * 60)

    num_students = df["student_id"].nunique()
    expected_students = config["dataset"]["num_students"]
    print(f"   Unique students: {num_students} (expected ~{expected_students})")

    rows_per_student = df.groupby("student_id").size()
    print(f"   Rows per student: min={rows_per_student.min()}, max={rows_per_student.max()}, "
          f"mean={rows_per_student.mean():.1f}, std={rows_per_student.std():.1f}")

    sessions_per_student = df.groupby("student_id")["session_id"].nunique()
    print(f"   Sessions per student: min={sessions_per_student.min()}, max={sessions_per_student.max()}, "
          f"mean={sessions_per_student.mean():.1f}")

    # Check no student dominates the dataset
    max_share = rows_per_student.max() / len(df) * 100
    status = "[PASS]" if max_share < 5 else "[WARN]"
    print(f"   {status} Max student share: {max_share:.2f}% (should be < 5%)")

    return True


def validate_temporal_coherence(df):
    """Check that timestamps are ordered within sessions."""
    print("\n" + "=" * 60)
    print("5. TEMPORAL COHERENCE CHECK")
    print("=" * 60)

    df_sorted = df.sort_values(["student_id", "session_id", "snapshot_index"])
    issues = 0
    total_sessions = df["session_id"].nunique()

    for session_id, group in df_sorted.groupby("session_id"):
        if len(group) < 2:
            continue
        indices = group["snapshot_index"].values
        if not np.all(np.diff(indices) >= 0):
            issues += 1

    status = "[PASS]" if issues == 0 else "[WARN]"
    print(f"   {status} Snapshot ordering: {issues}/{total_sessions} sessions with ordering issues")

    return issues == 0


def validate_feature_correlations(df):
    """Check that feature distributions differ meaningfully across states."""
    print("\n" + "=" * 60)
    print("6. FEATURE DISCRIMINABILITY CHECK")
    print("=" * 60)

    features = ["tab_switch", "idle_time", "clicks", "mouse_movement",
                 "replay_count", "skip_count", "playback_speed"]

    from scipy import stats

    for feat in features:
        groups = [group[feat].values for _, group in df.groupby("state")]
        f_stat, p_value = stats.f_oneway(*groups)
        status = "[PASS]" if p_value < 0.05 else "[WARN]"
        print(f"   {status} {feat}: F={f_stat:.2f}, p={p_value:.2e}")

    return True


def validate_dataset(data_path=None, config_path="config/config.yaml"):
    """Run all validation checks."""
    config = load_config(config_path)

    if data_path is None:
        data_path = config["dataset"]["output_path"]

    print(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Dataset shape: {df.shape}\n")

    results = {}
    results["missing_values"] = validate_missing_values(df)
    results["value_ranges"] = validate_value_ranges(df, config)
    results["class_balance"] = validate_class_balance(df, config)
    results["student_coverage"] = validate_student_coverage(df, config)
    results["temporal_coherence"] = validate_temporal_coherence(df)
    results["feature_discriminability"] = validate_feature_correlations(df)

    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    all_pass = True
    for check, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"   {status} {check}")
        if not passed:
            all_pass = False

    overall = "ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED"
    print(f"\n   >>> {overall} <<<")

    return all_pass, results


if __name__ == "__main__":
    validate_dataset()
