"""
Test Per-Student Model Retraining Pipeline
==========================================
Tests: feature building, training readiness, model training,
       prediction, retraining logic, and full backend integration.
"""

import os
import sys
import io
import json
import tempfile
import shutil

# Fix Windows encoding for arrow characters
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from src.student_model import (
    build_student_features,
    check_training_readiness,
    train_student_model,
    predict_with_student_model,
    load_student_model,
    should_retrain,
    STUDENT_MODELS_DIR,
    MIN_SNAPSHOTS,
    MIN_CLASSES,
    MIN_PER_CLASS,
    RETRAIN_INTERVAL,
)

passed = 0
failed = 0


def test(name, condition, detail=""):
    global passed, failed
    if condition:
        print(f"  PASS  {name}")
        passed += 1
    else:
        print(f"  FAIL  {name} -- {detail}")
        failed += 1


def make_snapshot_df(n, classes=None, correlated=True):
    """Generate a realistic snapshot DataFrame with n rows.
    If correlated=True, features are correlated with states so a model can learn.
    """
    np.random.seed(42)
    if classes is None:
        classes = ["focused", "distracted", "confused", "bored"]

    if not correlated or len(classes) < 3:
        # Random uncorrelated data (for readiness tests)
        df = pd.DataFrame({
            "id": range(1, n + 1),
            "snapshot_index": range(n),
            "student_id": "test_student",
            "tab_switch": np.random.randint(0, 10, n),
            "idle_time": np.random.uniform(0, 60, n).round(1),
            "clicks": np.random.randint(0, 30, n),
            "mouse_movement": np.random.randint(0, 500, n),
            "replay_count": np.random.randint(0, 5, n),
            "skip_count": np.random.randint(0, 5, n),
            "playback_speed": np.random.choice([0.75, 1.0, 1.25, 1.5, 1.75, 2.0], n),
            "focus_score": np.random.uniform(10, 95, n).round(2),
            "predicted_state": np.random.choice(classes, n),
        })
        return df

    # Generate data where features correlate with cognitive state
    per_class = n // len(classes)
    rows = []
    state_profiles = {
        "focused":    {"tab": (1, 1), "idle": (3, 2), "clicks": (20, 5), "mouse": (300, 80),
                       "replay": (0, 0.5), "skip": (0, 0.5), "speed": 1.0, "focus": (80, 8)},
        "distracted": {"tab": (7, 2), "idle": (25, 10), "clicks": (5, 3), "mouse": (50, 30),
                       "replay": (0, 0.5), "skip": (1, 1), "speed": 1.0, "focus": (35, 10)},
        "confused":   {"tab": (2, 1), "idle": (20, 8), "clicks": (10, 4), "mouse": (100, 50),
                       "replay": (4, 1.5), "skip": (0, 0.5), "speed": 0.8, "focus": (45, 10)},
        "bored":      {"tab": (4, 2), "idle": (10, 5), "clicks": (8, 4), "mouse": (80, 40),
                       "replay": (0, 0.5), "skip": (4, 1.5), "speed": 1.8, "focus": (30, 10)},
    }
    for state in classes:
        p = state_profiles.get(state, state_profiles["focused"])
        count = per_class if state != classes[-1] else n - len(rows)
        for _ in range(count):
            rows.append({
                "tab_switch": max(0, int(np.random.normal(*p["tab"]))),
                "idle_time": round(max(0, np.random.normal(*p["idle"])), 1),
                "clicks": max(0, int(np.random.normal(*p["clicks"]))),
                "mouse_movement": max(0, int(np.random.normal(*p["mouse"]))),
                "replay_count": max(0, int(np.random.normal(*p["replay"]))),
                "skip_count": max(0, int(np.random.normal(*p["skip"]))),
                "playback_speed": round(max(0.5, np.random.normal(p["speed"], 0.15)), 2),
                "focus_score": round(max(0, min(100, np.random.normal(*p["focus"]))), 2),
                "predicted_state": state,
            })

    np.random.shuffle(rows)
    df = pd.DataFrame(rows)
    df.insert(0, "id", range(1, n + 1))
    df.insert(1, "snapshot_index", range(n))
    df.insert(2, "student_id", "test_student")
    return df


# ============================================================
print("\n=== 1. Feature Building ===")
# ============================================================

df_small = make_snapshot_df(10)
features = build_student_features(df_small)

test("Returns DataFrame", isinstance(features, pd.DataFrame))
test("Same row count", len(features) == len(df_small), f"{len(features)} != {len(df_small)}")
test("Has raw features", all(c in features.columns for c in ["tab_switch", "idle_time", "clicks"]))
test("Has rolling features", any("roll_mean" in c for c in features.columns))
test("Has zscore features", any("zscore" in c for c in features.columns))
test("Has lag features", any("lag1" in c for c in features.columns))
test("Has interaction features", "tab_idle_ratio" in features.columns)
test("No NaN values", features.isna().sum().sum() == 0, f"NaN count: {features.isna().sum().sum()}")

n_features = len(features.columns)
print(f"  (Generated {n_features} features from {len(df_small)} rows)")


# ============================================================
print("\n=== 2. Training Readiness ===")
# ============================================================

# Too few snapshots
df_few = make_snapshot_df(50)
ready, reason = check_training_readiness(df_few)
test("Rejects <300 snapshots", not ready, reason)
test("Reason mentions count", "300" in reason or "snapshots" in reason.lower(), reason)

# Enough snapshots but only 2 classes
df_2class = make_snapshot_df(400, classes=["focused", "distracted"], correlated=False)
ready, reason = check_training_readiness(df_2class)
test("Rejects <3 classes", not ready, reason)

# Enough snapshots, 4 classes, but one class tiny
df_imbal = make_snapshot_df(400)
df_imbal.loc[:, "predicted_state"] = "focused"
df_imbal.loc[0:4, "predicted_state"] = "distracted"  # only 5
df_imbal.loc[5:9, "predicted_state"] = "confused"     # only 5
df_imbal.loc[10:14, "predicted_state"] = "bored"      # only 5
ready, reason = check_training_readiness(df_imbal)
test("Rejects tiny classes", not ready, reason)

# Good data
df_good = make_snapshot_df(400, correlated=False)
ready, reason = check_training_readiness(df_good)
test("Accepts good data (400, 4 classes)", ready, reason)

# No predicted_state column
df_nolabel = df_good.drop(columns=["predicted_state"])
ready, reason = check_training_readiness(df_nolabel)
test("Rejects missing label column", not ready, reason)


# ============================================================
print("\n=== 3. Model Training ===")
# ============================================================

# Use temp dir for models
import src.student_model as sm
original_dir = sm.STUDENT_MODELS_DIR
temp_dir = tempfile.mkdtemp(prefix="student_models_test_")
sm.STUDENT_MODELS_DIR = temp_dir

try:
    # Train with insufficient data
    result = train_student_model("stu_small", make_snapshot_df(50))
    test("Small data → not_ready", result["status"] == "not_ready", result.get("status"))

    # Train with good data
    df_train = make_snapshot_df(400)
    result = train_student_model("stu_good", df_train)
    test("Good data → trained or too_weak",
         result["status"] in ("trained", "too_weak"),
         result.get("status"))

    if result["status"] == "trained":
        test("CV score reported", "cv_mean" in result and result["cv_mean"] > 0)
        test("Model file saved",
             os.path.exists(os.path.join(temp_dir, "stu_good.joblib")))

        # Load and check
        bundle = load_student_model("stu_good")
        test("Model loads back", bundle is not None)
        test("Has model key", "model" in bundle)
        test("Has label_encoder key", "label_encoder" in bundle)
        test("Has meta key", "meta" in bundle)
        test("Meta has classes", len(bundle["meta"]["classes"]) >= 3,
             f"classes: {bundle['meta'].get('classes')}")
        test("Meta has feature count", bundle["meta"]["n_features"] == n_features,
             f"{bundle['meta']['n_features']} != {n_features}")

        print(f"  (Model CV F1: {result['cv_mean']}±{result['cv_std']}, "
              f"classes: {result['classes']})")
    else:
        print(f"  (Model was too weak: CV F1={result.get('cv_mean')} — skipping load tests)")


    # ============================================================
    print("\n=== 4. Prediction ===")
    # ============================================================

    if result["status"] == "trained":
        # Predict with recent snapshots
        recent = df_train.tail(10).reset_index(drop=True)
        state, conf = predict_with_student_model("stu_good", recent)
        test("Returns a state", state is not None, f"state={state}")
        test("State is valid class",
             state in ["focused", "distracted", "confused", "bored"],
             f"state={state}")
        test("Returns confidence", conf is not None and 0 <= conf <= 1,
             f"conf={conf}")
        print(f"  (Predicted: {state}, confidence: {conf})")

        # Predict with too few rows (should still work, just less accurate)
        tiny = df_train.tail(2).reset_index(drop=True)
        state2, conf2 = predict_with_student_model("stu_good", tiny)
        test("Works with 2 rows", state2 is not None)

        # Predict for non-existent student
        state3, conf3 = predict_with_student_model("nonexistent", recent)
        test("No model → None", state3 is None and conf3 is None)
    else:
        print("  (Skipped — no trained model)")


    # ============================================================
    print("\n=== 5. Retrain Logic ===")
    # ============================================================

    # No model exists
    test("No model + enough data → retrain",
         should_retrain("new_student", 350))
    test("No model + too little data → no retrain",
         not should_retrain("new_student", 100))

    if result["status"] == "trained":
        trained_count = result["n_snapshots"]
        test("Just trained → no retrain",
             not should_retrain("stu_good", trained_count + 10))
        test("50 new snapshots → retrain",
             should_retrain("stu_good", trained_count + RETRAIN_INTERVAL))
        test("49 new snapshots → no retrain",
             not should_retrain("stu_good", trained_count + RETRAIN_INTERVAL - 1))

finally:
    sm.STUDENT_MODELS_DIR = original_dir
    shutil.rmtree(temp_dir, ignore_errors=True)


# ============================================================
print("\n=== 6. Backend Integration ===")
# ============================================================

from backend.app import app, load_model

# Load config/model so endpoints work in test mode
try:
    load_model()
except Exception as e:
    print(f"  (load_model warning: {e})")

client = app.test_client()

# Create a test student first
signup_resp = client.post("/api/signup", json={
    "username": f"test_pmodel_{np.random.randint(10000, 99999)}",
    "password": "testpass123",
})
if signup_resp.status_code == 200:
    test_student_id = signup_resp.get_json()["student_id"]
else:
    test_student_id = f"test_stu_{np.random.randint(10000, 99999)}"

# Send a snapshot and check response has prediction_source
snap_resp = client.post("/api/snapshot", json={
    "student_id": test_student_id,
    "session_id": "test_session_001",
    "timestamp": "2026-04-08T10:00:00",
    "snapshot_index": 1,
    "tab_switch": 2,
    "idle_time": 5.0,
    "clicks": 15,
    "mouse_movement": 200,
    "replay_count": 0,
    "skip_count": 0,
    "playback_speed": 1.0,
    "website": "youtube.com",
    "elapsed_seconds": 30,
})

test("Snapshot endpoint returns 200", snap_resp.status_code == 200,
     f"status={snap_resp.status_code}")

snap_data = snap_resp.get_json()
test("Response has focus_score", "focus_score" in snap_data)
test("Response has state", "state" in snap_data)
test("Response has prediction_source", "prediction_source" in snap_data,
     f"keys={list(snap_data.keys())}")
test("New student uses rule_based",
     snap_data.get("prediction_source") == "rule_based",
     f"source={snap_data.get('prediction_source')}")

print(f"  (Snapshot response: state={snap_data.get('state')}, "
      f"source={snap_data.get('prediction_source')}, "
      f"focus={snap_data.get('focus_score')})")

# Send a few more snapshots to verify no crashes
for i in range(2, 6):
    r = client.post("/api/snapshot", json={
        "student_id": test_student_id,
        "session_id": "test_session_001",
        "timestamp": f"2026-04-08T10:0{i}:00",
        "snapshot_index": i,
        "tab_switch": np.random.randint(0, 5),
        "idle_time": float(np.random.randint(0, 30)),
        "clicks": np.random.randint(5, 25),
        "mouse_movement": np.random.randint(50, 400),
        "replay_count": 0,
        "skip_count": 0,
        "playback_speed": 1.0,
        "website": "youtube.com",
        "elapsed_seconds": 30 * i,
    })
test("Multiple snapshots succeed", r.status_code == 200)


# ============================================================
print("\n" + "=" * 50)
print(f"RESULTS: {passed} passed, {failed} failed out of {passed + failed}")
print("=" * 50)

if failed > 0:
    sys.exit(1)
else:
    print("All tests passed!")
