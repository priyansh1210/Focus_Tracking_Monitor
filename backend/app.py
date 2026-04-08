"""
Flask Backend — Student Focus Monitor
======================================
- Receives behavioral snapshots from Chrome extension
- Stores data in SQLite or PostgreSQL database
- Runs ML model inference (focus score + state prediction)
- Provides API for dashboard and analytics
- Supports per-student model updates
"""

import os
import sys
import json
import sqlite3
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS

import hashlib
import uuid

import numpy as np
import pandas as pd
import joblib
import yaml

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from student_model import (
    predict_with_student_model,
    should_retrain,
    train_student_model,
)

app = Flask(__name__)
CORS(app)  # Allow Chrome extension to call API

DB_PATH = os.path.join(PROJECT_ROOT, "backend", "focus_monitor.db")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config", "config.yaml")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models_saved")

# Database mode: "postgres" if DATABASE_URL is set, else "sqlite"
DATABASE_URL = os.environ.get("DATABASE_URL", "")
USE_POSTGRES = bool(DATABASE_URL)


# ---- Database ----

class PgConnectionWrapper:
    """Wraps a psycopg2 connection to behave like sqlite3 connection.
    Adds .execute() and .executescript() that return cursor with .fetchone()/.fetchall().
    """
    def __init__(self, conn):
        self._conn = conn

    def execute(self, sql, params=None):
        cur = self._conn.cursor()
        cur.execute(sql, params or ())
        return cur

    def executescript(self, sql):
        cur = self._conn.cursor()
        for stmt in sql.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt)
        return cur

    def commit(self):
        self._conn.commit()

    def rollback(self):
        self._conn.rollback()

    def close(self):
        self._conn.close()

    def cursor(self):
        return self._conn.cursor()

    @property
    def raw(self):
        """Access underlying connection for pandas read_sql_query."""
        return self._conn


def get_db():
    if USE_POSTGRES:
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL, sslmode="require",
                                cursor_factory=psycopg2.extras.RealDictCursor)
        return PgConnectionWrapper(conn)
    else:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def q(sql):
    """Convert SQLite-style ? placeholders to PostgreSQL %s if needed."""
    if USE_POSTGRES:
        return sql.replace("?", "%s")
    return sql


def init_db():
    conn = get_db()
    cur = conn.cursor()

    if USE_POSTGRES:
        auto_id = "SERIAL PRIMARY KEY"
    else:
        auto_id = "INTEGER PRIMARY KEY AUTOINCREMENT"

    tables = [
        f"""CREATE TABLE IF NOT EXISTS snapshots (
            id {auto_id},
            student_id TEXT NOT NULL,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            snapshot_index INTEGER,
            tab_switch INTEGER DEFAULT 0,
            idle_time REAL DEFAULT 0,
            clicks INTEGER DEFAULT 0,
            mouse_movement REAL DEFAULT 0,
            replay_count INTEGER DEFAULT 0,
            skip_count INTEGER DEFAULT 0,
            playback_speed REAL DEFAULT 1.0,
            website TEXT,
            elapsed_seconds INTEGER,
            focus_score REAL,
            predicted_state TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        f"""CREATE TABLE IF NOT EXISTS sessions (
            id {auto_id},
            student_id TEXT NOT NULL,
            session_id TEXT UNIQUE NOT NULL,
            website TEXT,
            start_time TEXT,
            end_time TEXT,
            total_snapshots INTEGER DEFAULT 0,
            avg_focus_score REAL,
            dominant_state TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        f"""CREATE TABLE IF NOT EXISTS student_baselines (
            student_id TEXT PRIMARY KEY,
            tab_switch_mean REAL DEFAULT 0,
            tab_switch_std REAL DEFAULT 1,
            idle_time_mean REAL DEFAULT 0,
            idle_time_std REAL DEFAULT 1,
            clicks_mean REAL DEFAULT 0,
            clicks_std REAL DEFAULT 1,
            mouse_movement_mean REAL DEFAULT 0,
            mouse_movement_std REAL DEFAULT 1,
            replay_count_mean REAL DEFAULT 0,
            replay_count_std REAL DEFAULT 1,
            skip_count_mean REAL DEFAULT 0,
            skip_count_std REAL DEFAULT 1,
            focus_score_mean REAL DEFAULT 50,
            focus_score_std REAL DEFAULT 15,
            total_snapshots INTEGER DEFAULT 0,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        f"""CREATE TABLE IF NOT EXISTS users (
            id {auto_id},
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
        f"""CREATE TABLE IF NOT EXISTS student_profiles (
            student_id TEXT PRIMARY KEY,
            learning_style TEXT DEFAULT 'unknown',
            tab_switch_tolerance REAL DEFAULT 0,
            skip_tolerance REAL DEFAULT 0,
            replay_tolerance REAL DEFAULT 0,
            speed_preference REAL DEFAULT 1.0,
            focus_weight_tab_switch REAL DEFAULT -3.0,
            focus_weight_idle_time REAL DEFAULT -0.15,
            focus_weight_clicks REAL DEFAULT 0.3,
            focus_weight_mouse_movement REAL DEFAULT 0.005,
            focus_weight_replay_count REAL DEFAULT -1.5,
            focus_weight_skip_count REAL DEFAULT -2.5,
            focus_weight_speed_deviation REAL DEFAULT -10.0,
            state_threshold_focused REAL DEFAULT 70,
            state_threshold_moderate REAL DEFAULT 50,
            state_threshold_low REAL DEFAULT 35,
            total_sessions INTEGER DEFAULT 0,
            last_updated TEXT DEFAULT CURRENT_TIMESTAMP
        )""",
    ]

    for stmt in tables:
        cur.execute(stmt)

    # Indexes
    cur.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_student ON snapshots(student_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_snapshots_session ON snapshots(session_id)")
    cur.execute("CREATE INDEX IF NOT EXISTS idx_sessions_student ON sessions(student_id)")

    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Database initialized ({'PostgreSQL' if USE_POSTGRES else 'SQLite'})")


# ---- Model Loading ----

model = None
label_encoder = None
config = None


def load_model():
    global model, label_encoder, config
    config_path = CONFIG_PATH
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_path = os.path.join(MODELS_DIR, "xgboost.joblib")
    if not os.path.exists(model_path):
        model_path = os.path.join(MODELS_DIR, "random_forest.joblib")

    if os.path.exists(model_path):
        model = joblib.load(model_path)
        print(f"[Model] Loaded: {model_path}")
    else:
        print("[Model] WARNING: No trained model found!")

    le_path = os.path.join(MODELS_DIR, "label_encoder.joblib")
    if os.path.exists(le_path):
        label_encoder = joblib.load(le_path)


# ---- Student Profile & Personalization ----

def get_student_profile(student_id):
    """Get or create a student's learning profile."""
    conn = get_db()
    row = conn.execute(
        q("SELECT * FROM student_profiles WHERE student_id = ?"), (student_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def classify_learning_style(df):
    """
    Classify student's learning style from their behavioral data.

    Types:
    - 'explorer'    : high tab switches + high clicks (references external material)
    - 'rewatcher'   : high replay count (needs repetition to absorb)
    - 'skimmer'     : high skip count + high speed (fast learner, skips known content)
    - 'steady'      : low variance across all metrics (consistent focused learner)
    - 'unknown'     : not enough data to classify
    """
    if len(df) < 10:
        return "unknown"

    avg_tabs = df["tab_switch"].mean()
    avg_clicks = df["clicks"].mean()
    avg_replays = df["replay_count"].mean()
    avg_skips = df["skip_count"].mean()
    avg_speed = df["playback_speed"].mean() if "playback_speed" in df.columns else 1.0

    # Score each style
    explorer_score = (avg_tabs / 3.0) + (avg_clicks / 15.0)
    rewatcher_score = (avg_replays / 2.0)
    skimmer_score = (avg_skips / 2.0) + max(0, (avg_speed - 1.2) * 3)
    # Steady = low deviation from means
    feature_stds = df[["tab_switch", "idle_time", "clicks", "replay_count", "skip_count"]].std()
    steadiness = 1.0 / (1.0 + feature_stds.mean())
    steady_score = steadiness * 3

    scores = {
        "explorer": explorer_score,
        "rewatcher": rewatcher_score,
        "skimmer": skimmer_score,
        "steady": steady_score,
    }

    return max(scores, key=scores.get)


def compute_personalized_weights(learning_style, baseline):
    """
    Adjust focus score weights based on the student's learning style.

    E.g., an 'explorer' who normally switches tabs shouldn't be penalized
    as hard for tab switches. A 'skimmer' who skips known content shouldn't
    be penalized for skips.
    """
    default_weights = config["focus_score"]["weights"].copy()

    if learning_style == "explorer":
        # Explorers naturally switch tabs — reduce penalty
        default_weights["tab_switch"] = -1.5      # was -3.0
        default_weights["clicks"] = 0.5            # reward engagement
    elif learning_style == "rewatcher":
        # Rewatchers replay a lot — that's their way of learning, not confusion
        default_weights["replay_count"] = -0.5     # was -1.5
        default_weights["idle_time"] = -0.1        # they pause to think
    elif learning_style == "skimmer":
        # Skimmers skip known content — not boredom
        default_weights["skip_count"] = -1.0       # was -2.5
        default_weights["playback_speed_deviation"] = -5.0  # was -10.0
    elif learning_style == "steady":
        # Steady learners — any deviation is more significant
        default_weights["tab_switch"] = -4.0
        default_weights["idle_time"] = -0.2

    return default_weights


def compute_personalized_thresholds(baseline):
    """
    Adjust state classification thresholds based on student's own focus history.

    Students with naturally higher focus get tighter thresholds (small drops matter).
    Students with naturally lower focus get looser thresholds (we meet them where they are).
    """
    if not baseline or baseline["total_snapshots"] < 10:
        return {"focused": 70, "moderate": 50, "low": 35}

    avg_focus = baseline["focus_score_mean"]
    std_focus = baseline["focus_score_std"]

    # Thresholds relative to student's own distribution
    focused_threshold = max(55, avg_focus - 0.3 * std_focus)
    moderate_threshold = max(35, avg_focus - 1.0 * std_focus)
    low_threshold = max(20, avg_focus - 1.8 * std_focus)

    return {
        "focused": round(focused_threshold, 1),
        "moderate": round(moderate_threshold, 1),
        "low": round(low_threshold, 1),
    }


def update_student_profile(student_id):
    """Recalculate a student's learning profile from all their data."""
    conn = get_db()
    rows = conn.execute(
        q("SELECT tab_switch, idle_time, clicks, mouse_movement, "
          "replay_count, skip_count, playback_speed, focus_score "
          "FROM snapshots WHERE student_id = ?"),
        (student_id,)
    ).fetchall()

    session_count_row = conn.execute(
        q("SELECT COUNT(*) as cnt FROM sessions WHERE student_id = ?"), (student_id,)
    ).fetchone()
    session_count = session_count_row["cnt"] if USE_POSTGRES else session_count_row[0]
    conn.close()

    if len(rows) < 10:
        return

    df = pd.DataFrame([dict(r) for r in rows] if not USE_POSTGRES else rows,
                       columns=["tab_switch", "idle_time", "clicks",
                                "mouse_movement", "replay_count",
                                "skip_count", "playback_speed", "focus_score"])

    # Classify learning style
    style = classify_learning_style(df)

    # Get baseline for threshold calculation
    baseline = get_student_baseline(student_id)

    # Compute personalized weights
    weights = compute_personalized_weights(style, baseline)

    # Compute personalized thresholds
    thresholds = compute_personalized_thresholds(baseline)

    # Save profile
    conn = get_db()
    conn.execute(q("""
        INSERT INTO student_profiles
            (student_id, learning_style, tab_switch_tolerance, skip_tolerance,
             replay_tolerance, speed_preference,
             focus_weight_tab_switch, focus_weight_idle_time, focus_weight_clicks,
             focus_weight_mouse_movement, focus_weight_replay_count,
             focus_weight_skip_count, focus_weight_speed_deviation,
             state_threshold_focused, state_threshold_moderate, state_threshold_low,
             total_sessions, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(student_id) DO UPDATE SET
            learning_style=excluded.learning_style,
            tab_switch_tolerance=excluded.tab_switch_tolerance,
            skip_tolerance=excluded.skip_tolerance,
            replay_tolerance=excluded.replay_tolerance,
            speed_preference=excluded.speed_preference,
            focus_weight_tab_switch=excluded.focus_weight_tab_switch,
            focus_weight_idle_time=excluded.focus_weight_idle_time,
            focus_weight_clicks=excluded.focus_weight_clicks,
            focus_weight_mouse_movement=excluded.focus_weight_mouse_movement,
            focus_weight_replay_count=excluded.focus_weight_replay_count,
            focus_weight_skip_count=excluded.focus_weight_skip_count,
            focus_weight_speed_deviation=excluded.focus_weight_speed_deviation,
            state_threshold_focused=excluded.state_threshold_focused,
            state_threshold_moderate=excluded.state_threshold_moderate,
            state_threshold_low=excluded.state_threshold_low,
            total_sessions=excluded.total_sessions,
            last_updated=excluded.last_updated
    """), (
        student_id, style,
        round(df["tab_switch"].mean(), 2),
        round(df["skip_count"].mean(), 2),
        round(df["replay_count"].mean(), 2),
        round(df["playback_speed"].mean(), 2),
        weights["tab_switch"], weights["idle_time"], weights["clicks"],
        weights["mouse_movement"], weights["replay_count"],
        weights["skip_count"], weights["playback_speed_deviation"],
        thresholds["focused"], thresholds["moderate"], thresholds["low"],
        session_count, datetime.now().isoformat(),
    ))
    conn.commit()
    conn.close()

    print(f"[Profile] {student_id}: style={style}, thresholds={thresholds}")


# ---- Focus Score & Prediction ----

def compute_focus_score(snapshot, baseline=None, profile=None):
    """
    Compute dynamic focus score using personalized weights.

    For new students: uses global config weights with raw values.
    For returning students: uses z-score normalization (deviation from their own
    baseline, divided by their own std) with personalized weights.
    """
    # Use personalized weights if profile exists, else global defaults
    if profile:
        weights = {
            "tab_switch": profile["focus_weight_tab_switch"],
            "idle_time": profile["focus_weight_idle_time"],
            "clicks": profile["focus_weight_clicks"],
            "mouse_movement": profile["focus_weight_mouse_movement"],
            "replay_count": profile["focus_weight_replay_count"],
            "skip_count": profile["focus_weight_skip_count"],
            "playback_speed_deviation": profile["focus_weight_speed_deviation"],
        }
    else:
        weights = config["focus_score"]["weights"]

    base_score = 70.0
    features = ["tab_switch", "idle_time", "clicks", "mouse_movement",
                "replay_count", "skip_count"]
    snapshot_keys = ["tab_switch", "idle_time", "clicks", "mouse_movement",
                     "replay_count", "skip_count"]

    if baseline and baseline["total_snapshots"] > 10:
        # Personalized: z-score normalization against student's own baseline
        # deviation = (value - mean) / std — tells us how unusual this is FOR THIS student
        deviations = {}
        for feat, snap_key in zip(features, snapshot_keys):
            mean = baseline[f"{feat}_mean"]
            std = baseline[f"{feat}_std"]
            raw = snapshot.get(snap_key, 0)
            deviations[feat] = (raw - mean) / std  # z-score
    elif baseline and baseline["total_snapshots"] > 5:
        # Partial personalization: deviation from mean but no std normalization yet
        deviations = {}
        for feat, snap_key in zip(features, snapshot_keys):
            mean = baseline[f"{feat}_mean"]
            raw = snapshot.get(snap_key, 0)
            deviations[feat] = raw - mean
    else:
        # New student: raw values
        deviations = {feat: snapshot.get(snap_key, 0)
                      for feat, snap_key in zip(features, snapshot_keys)}

    speed_dev = abs(snapshot.get("playback_speed", 1.0) - (
        profile["speed_preference"] if profile else 1.0))

    score = base_score
    for feat in features:
        score += weights[feat] * deviations[feat]
    score += weights["playback_speed_deviation"] * speed_dev

    # Heavy penalty for time spent away from study tab
    away_ratio = snapshot.get("away_ratio", 0)  # 0.0 to 1.0
    away_time = snapshot.get("away_time", 0)     # seconds
    if away_ratio > 0:
        # If student spent 80% of the window on other tabs, massive penalty
        score -= away_ratio * 60  # e.g., 80% away -> -48 points
    elif away_time > 5:
        # Even small away times get a proportional penalty
        score -= min(away_time * 1.5, 40)

    return max(0, min(100, round(score, 2)))


def predict_state(snapshot, focus_score, baseline=None, profile=None):
    """
    Predict cognitive state using personalized thresholds.

    For new students: uses fixed global thresholds.
    For returning students: thresholds are relative to their own focus distribution.
    The behavioral indicators (tab switches, replays, skips) are also compared
    against the student's own tolerance levels.
    """
    # Get thresholds
    if profile:
        t_focused = profile["state_threshold_focused"]
        t_moderate = profile["state_threshold_moderate"]
        t_low = profile["state_threshold_low"]
        tab_tol = profile["tab_switch_tolerance"]
        skip_tol = profile["skip_tolerance"]
        replay_tol = profile["replay_tolerance"]
    else:
        t_focused = 70
        t_moderate = 50
        t_low = 35
        tab_tol = 0
        skip_tol = 0
        replay_tol = 0

    # Behavioral deviations from student's own norms
    tab_excess = snapshot.get("tab_switch", 0) - tab_tol
    skip_excess = snapshot.get("skip_count", 0) - skip_tol
    replay_excess = snapshot.get("replay_count", 0) - replay_tol
    away_ratio = snapshot.get("away_ratio", 0)

    # If student spent most of the window on other tabs, they're distracted regardless
    if away_ratio > 0.6:
        return "distracted"

    # Personalized classification
    if focus_score >= t_focused:
        state = "focused"
    elif tab_excess >= 5 or (snapshot.get("idle_time", 0) > 30 and snapshot.get("clicks", 0) < 3):
        state = "distracted"
    elif replay_excess >= 3 or (snapshot.get("idle_time", 0) > 20 and snapshot.get("playback_speed", 1.0) < 0.9):
        state = "confused"
    elif skip_excess >= 3 or snapshot.get("playback_speed", 1.0) >= 1.8:
        state = "bored"
    elif focus_score >= t_moderate:
        state = "focused"
    elif focus_score >= t_low:
        state = "distracted"
    else:
        state = "bored"

    return state


def get_adaptive_message(state, focus_score, profile=None):
    """Return adaptive response message personalized to learning style."""
    style = profile["learning_style"] if profile else "unknown"

    if focus_score < 20:
        return "Your focus is very low. Consider taking a 5-minute break and coming back fresh."

    style_messages = {
        "explorer": {
            "focused": "Great focus! Your research approach is working well.",
            "distracted": "Lots of tab switching — try bookmarking references and coming back to them later.",
            "confused": "Struggling? Try searching for a different explanation of this topic.",
            "bored": "Need a challenge? Try applying what you've learned to a practice problem.",
        },
        "rewatcher": {
            "focused": "Solid focus! Your careful review is paying off.",
            "distracted": "Getting sidetracked — try replaying the last section instead of switching tabs.",
            "confused": "This is a tough part. Try watching at 0.75x speed with notes.",
            "bored": "You've got this down! Try skipping ahead to new material.",
        },
        "skimmer": {
            "focused": "Nice pace! You're efficiently covering the material.",
            "distracted": "Losing focus — try jumping to the next key concept.",
            "confused": "Slow down a bit here — this section needs more attention.",
            "bored": "Already know this? Skip ahead or try the advanced exercises.",
        },
        "steady": {
            "focused": "Great job! You're in the zone. Keep it up!",
            "distracted": "Looks like you're getting distracted. Try to refocus on the content.",
            "confused": "Struggling with this part? Try re-watching the last section slowly.",
            "bored": "Feeling bored? Maybe take a short break or try a practice problem.",
        },
    }

    messages = style_messages.get(style, style_messages["steady"])
    return messages.get(state, "Keep studying!")


# ---- Update Student Baseline ----

def update_student_baseline(student_id):
    """Recalculate student's behavioral baseline from all their data."""
    conn = get_db()
    rows = conn.execute(
        q("SELECT tab_switch, idle_time, clicks, mouse_movement, "
          "replay_count, skip_count, focus_score FROM snapshots WHERE student_id = ?"),
        (student_id,)
    ).fetchall()
    conn.close()

    if len(rows) < 3:
        return

    df = pd.DataFrame(rows, columns=["tab_switch", "idle_time", "clicks",
                                       "mouse_movement", "replay_count",
                                       "skip_count", "focus_score"])

    baseline = {
        "student_id": student_id,
        "total_snapshots": len(df),
        "last_updated": datetime.now().isoformat(),
    }

    for feat in ["tab_switch", "idle_time", "clicks", "mouse_movement",
                 "replay_count", "skip_count", "focus_score"]:
        baseline[f"{feat}_mean"] = round(df[feat].mean(), 2)
        baseline[f"{feat}_std"] = max(round(df[feat].std(), 2), 0.01)

    conn = get_db()
    conn.execute(q("""
        INSERT INTO student_baselines
            (student_id, tab_switch_mean, tab_switch_std, idle_time_mean, idle_time_std,
             clicks_mean, clicks_std, mouse_movement_mean, mouse_movement_std,
             replay_count_mean, replay_count_std, skip_count_mean, skip_count_std,
             focus_score_mean, focus_score_std, total_snapshots, last_updated)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(student_id) DO UPDATE SET
            tab_switch_mean=excluded.tab_switch_mean, tab_switch_std=excluded.tab_switch_std,
            idle_time_mean=excluded.idle_time_mean, idle_time_std=excluded.idle_time_std,
            clicks_mean=excluded.clicks_mean, clicks_std=excluded.clicks_std,
            mouse_movement_mean=excluded.mouse_movement_mean, mouse_movement_std=excluded.mouse_movement_std,
            replay_count_mean=excluded.replay_count_mean, replay_count_std=excluded.replay_count_std,
            skip_count_mean=excluded.skip_count_mean, skip_count_std=excluded.skip_count_std,
            focus_score_mean=excluded.focus_score_mean, focus_score_std=excluded.focus_score_std,
            total_snapshots=excluded.total_snapshots, last_updated=excluded.last_updated
    """), (
        student_id,
        baseline["tab_switch_mean"], baseline["tab_switch_std"],
        baseline["idle_time_mean"], baseline["idle_time_std"],
        baseline["clicks_mean"], baseline["clicks_std"],
        baseline["mouse_movement_mean"], baseline["mouse_movement_std"],
        baseline["replay_count_mean"], baseline["replay_count_std"],
        baseline["skip_count_mean"], baseline["skip_count_std"],
        baseline["focus_score_mean"], baseline["focus_score_std"],
        baseline["total_snapshots"], baseline["last_updated"],
    ))
    conn.commit()
    conn.close()


def get_student_baseline(student_id):
    """Get cached student baseline."""
    conn = get_db()
    row = conn.execute(
        q("SELECT * FROM student_baselines WHERE student_id = ?"), (student_id,)
    ).fetchone()
    conn.close()
    return dict(row) if row else None


# ---- Auth Routes ----

def _hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()


def _generate_student_id():
    short = uuid.uuid4().hex[:8].upper()
    return f"STU_{short}"


@app.route("/api/register", methods=["POST"])
def register():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400
    if len(username) < 3:
        return jsonify({"error": "Username must be at least 3 characters"}), 400
    if len(password) < 4:
        return jsonify({"error": "Password must be at least 4 characters"}), 400

    student_id = _generate_student_id()
    pw_hash = _hash_password(password)

    conn = get_db()
    try:
        conn.execute(
            q("INSERT INTO users (username, password_hash, student_id) VALUES (?, ?, ?)"),
            (username, pw_hash, student_id),
        )
        conn.commit()
    except Exception as e:
        conn.close()
        return jsonify({"error": "Username already taken"}), 409
    conn.close()

    return jsonify({"status": "ok", "student_id": student_id, "username": username})


@app.route("/api/login", methods=["POST"])
def login():
    data = request.get_json()
    username = data.get("username", "").strip()
    password = data.get("password", "").strip()

    if not username or not password:
        return jsonify({"error": "Username and password are required"}), 400

    pw_hash = _hash_password(password)
    conn = get_db()
    row = conn.execute(
        q("SELECT student_id, username FROM users WHERE username = ? AND password_hash = ?"),
        (username, pw_hash),
    ).fetchone()
    conn.close()

    if not row:
        return jsonify({"error": "Invalid username or password"}), 401

    return jsonify({"status": "ok", "student_id": row["student_id"], "username": row["username"]})


# ---- API Routes ----

@app.route("/api/snapshot", methods=["POST"])
def receive_snapshot():
    """Receive a behavioral snapshot from the extension."""
    data = request.get_json()

    student_id = data.get("student_id", "unknown")
    session_id = data.get("session_id", "unknown")

    # Get student baseline and profile for personalized scoring
    baseline = get_student_baseline(student_id)
    profile = get_student_profile(student_id)

    # Compute personalized focus score
    focus_score = compute_focus_score(data, baseline, profile)

    # Predict cognitive state — try per-student ML model first, fall back to rules
    student_state, student_conf = None, None
    try:
        # Need recent snapshots for rolling features
        conn_tmp = get_db()
        recent = pd.read_sql_query(
            q("SELECT * FROM snapshots WHERE student_id = ? ORDER BY id DESC LIMIT 10"),
            conn_tmp.raw if USE_POSTGRES else conn_tmp, params=(student_id,),
        )
        conn_tmp.close()
        if len(recent) >= 3:
            recent = recent.iloc[::-1].reset_index(drop=True)  # chronological order
            student_state, student_conf = predict_with_student_model(student_id, recent)
    except Exception:
        pass

    if student_state and student_conf and student_conf >= 0.55:
        state = student_state
        prediction_source = "personal_model"
    else:
        state = predict_state(data, focus_score, baseline, profile)
        prediction_source = "rule_based"

    # Get adaptive message tailored to learning style
    message = get_adaptive_message(state, focus_score, profile)

    # Store in database
    conn = get_db()
    conn.execute(q("""
        INSERT INTO snapshots
            (student_id, session_id, timestamp, snapshot_index, tab_switch,
             idle_time, clicks, mouse_movement, replay_count, skip_count,
             playback_speed, website, elapsed_seconds, focus_score, predicted_state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """), (
        student_id, session_id, data.get("timestamp"),
        data.get("snapshot_index", 0),
        data.get("tab_switch", 0), data.get("idle_time", 0),
        data.get("clicks", 0), data.get("mouse_movement", 0),
        data.get("replay_count", 0), data.get("skip_count", 0),
        data.get("playback_speed", 1.0),
        data.get("website"), data.get("elapsed_seconds"),
        focus_score, state,
    ))
    conn.commit()
    conn.close()

    # Update baseline and profile periodically (every 10 snapshots)
    snapshot_count = data.get("snapshot_index", 0)
    if snapshot_count > 0 and snapshot_count % 10 == 0:
        update_student_baseline(student_id)
        update_student_profile(student_id)

    # Check if per-student model needs (re)training
    if snapshot_count > 0 and snapshot_count % 50 == 0:
        try:
            conn_rt = get_db()
            all_snaps = pd.read_sql_query(
                q("SELECT * FROM snapshots WHERE student_id = ? ORDER BY id"),
                conn_rt.raw if USE_POSTGRES else conn_rt, params=(student_id,),
            )
            conn_rt.close()
            if should_retrain(student_id, len(all_snaps)):
                result = train_student_model(student_id, all_snaps)
                print(f"[Retrain] {student_id}: {result.get('status')}")
        except Exception as e:
            print(f"[Retrain] Error for {student_id}: {e}")

    learning_style = profile["learning_style"] if profile else "unknown"

    return jsonify({
        "status": "ok",
        "focus_score": focus_score,
        "state": state,
        "message": message,
        "learning_style": learning_style,
        "prediction_source": prediction_source,
    })


@app.route("/api/session/end", methods=["POST"])
def end_session():
    """Record session end and update student baseline."""
    data = request.get_json()
    student_id = data.get("student_id")
    session_id = data.get("session_id")

    # Calculate session summary
    conn = get_db()
    rows = conn.execute(
        q("SELECT focus_score, predicted_state FROM snapshots WHERE session_id = ?"),
        (session_id,)
    ).fetchall()

    avg_score = 0
    dominant_state = "unknown"
    if rows:
        scores = [r["focus_score"] for r in rows if r["focus_score"] is not None]
        states = [r["predicted_state"] for r in rows if r["predicted_state"]]
        if scores:
            avg_score = round(sum(scores) / len(scores), 2)
        if states:
            from collections import Counter
            dominant_state = Counter(states).most_common(1)[0][0]

    conn.execute(q("""
        INSERT INTO sessions (student_id, session_id, website, start_time, end_time,
                              total_snapshots, avg_focus_score, dominant_state)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(session_id) DO UPDATE SET
            end_time=excluded.end_time, total_snapshots=excluded.total_snapshots,
            avg_focus_score=excluded.avg_focus_score, dominant_state=excluded.dominant_state
    """), (
        student_id, session_id, data.get("website"),
        data.get("start_time"), data.get("end_time"),
        data.get("total_snapshots", 0), avg_score, dominant_state,
    ))
    conn.commit()
    conn.close()

    # Update student baseline and learning profile with all new data
    update_student_baseline(student_id)
    update_student_profile(student_id)

    profile = get_student_profile(student_id)

    return jsonify({
        "status": "ok",
        "avg_focus_score": avg_score,
        "dominant_state": dominant_state,
        "learning_style": profile["learning_style"] if profile else "unknown",
    })


@app.route("/api/student/<student_id>/history", methods=["GET"])
def student_history(student_id):
    """Get session history for a student."""
    conn = get_db()
    sessions = conn.execute(
        q("SELECT * FROM sessions WHERE student_id = ? ORDER BY start_time DESC LIMIT 50"),
        (student_id,)
    ).fetchall()
    conn.close()
    return jsonify([dict(s) for s in sessions])


@app.route("/api/student/<student_id>/snapshots", methods=["GET"])
def student_snapshots(student_id):
    """Get recent snapshots for a student."""
    limit = request.args.get("limit", 100, type=int)
    conn = get_db()
    rows = conn.execute(
        q("SELECT * FROM snapshots WHERE student_id = ? ORDER BY timestamp DESC LIMIT ?"),
        (student_id, limit)
    ).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/student/<student_id>/baseline", methods=["GET"])
def student_baseline_api(student_id):
    """Get a student's behavioral baseline."""
    baseline = get_student_baseline(student_id)
    if baseline:
        return jsonify(dict(baseline))
    return jsonify({"error": "No baseline found"}), 404


@app.route("/api/student/<student_id>/profile", methods=["GET"])
def student_profile_api(student_id):
    """Get a student's learning profile with personalized weights and thresholds."""
    profile = get_student_profile(student_id)
    baseline = get_student_baseline(student_id)

    if not profile and not baseline:
        return jsonify({"error": "No profile found — need more data"}), 404

    result = {}
    if profile:
        result.update(dict(profile))
    if baseline:
        result["baseline"] = dict(baseline)

    return jsonify(result)


@app.route("/api/students", methods=["GET"])
def list_students():
    """List all students with their stats."""
    conn = get_db()
    rows = conn.execute("""
        SELECT student_id, COUNT(*) as total_sessions,
               AVG(avg_focus_score) as avg_focus,
               MAX(end_time) as last_active
        FROM sessions GROUP BY student_id ORDER BY last_active DESC
    """).fetchall()
    conn.close()
    return jsonify([dict(r) for r in rows])


@app.route("/api/export", methods=["GET"])
def export_data():
    """Export all snapshots as CSV for model retraining."""
    conn = get_db()
    df = pd.read_sql_query("SELECT * FROM snapshots ORDER BY student_id, session_id, snapshot_index", conn)
    conn.close()

    export_path = os.path.join(PROJECT_ROOT, "data", "real_data_export.csv")
    df.to_csv(export_path, index=False)

    return jsonify({
        "status": "ok",
        "rows": len(df),
        "path": export_path,
    })


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": model is not None,
        "database": os.path.exists(DB_PATH),
    })


# ---- Main ----

if __name__ == "__main__":
    print("=" * 50)
    print("Student Focus Monitor — Backend")
    print("=" * 50)
    init_db()
    load_model()
    print(f"\nDatabase: {DB_PATH}")
    print(f"Starting server on http://localhost:5000")
    print("=" * 50)
    app.run(host="0.0.0.0", port=5000, debug=True)
