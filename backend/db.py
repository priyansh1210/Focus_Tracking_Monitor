"""
Database Adapter — SQLite / PostgreSQL
=======================================
Uses DATABASE_URL env var to decide:
- If set → PostgreSQL (cloud)
- If not set → SQLite (local)

Both Flask backend and Streamlit dashboard import from here.
"""

import os
import sqlite3

DB_MODE = None  # "postgres" or "sqlite"
DATABASE_URL = os.environ.get("DATABASE_URL", "")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SQLITE_PATH = os.path.join(PROJECT_ROOT, "backend", "focus_monitor.db")


def _detect_mode():
    global DB_MODE
    if DATABASE_URL:
        DB_MODE = "postgres"
    else:
        DB_MODE = "sqlite"
    return DB_MODE


def get_connection():
    """Get a database connection (PostgreSQL or SQLite)."""
    mode = _detect_mode()

    if mode == "postgres":
        import psycopg2
        import psycopg2.extras
        conn = psycopg2.connect(DATABASE_URL, sslmode="require")
        conn.autocommit = False
        return conn
    else:
        conn = sqlite3.connect(SQLITE_PATH)
        conn.row_factory = sqlite3.Row
        return conn


def get_db():
    """Alias for get_connection() — used by Flask backend."""
    return get_connection()


def execute_query(query, params=None):
    """Execute a query and return results as list of dicts."""
    conn = get_connection()
    mode = DB_MODE

    try:
        if mode == "postgres":
            import psycopg2.extras
            cur = conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor)
        else:
            cur = conn.cursor()

        cur.execute(query, params or ())
        rows = cur.fetchall()

        if mode == "sqlite":
            # Convert sqlite3.Row to dict
            rows = [dict(r) for r in rows]

        cur.close()
        conn.close()
        return rows
    except Exception as e:
        conn.close()
        raise e


def execute_write(query, params=None):
    """Execute an INSERT/UPDATE/DELETE query."""
    conn = get_connection()
    try:
        cur = conn.cursor()
        cur.execute(query, params or ())
        conn.commit()
        cur.close()
        conn.close()
    except Exception as e:
        conn.rollback()
        conn.close()
        raise e


def read_sql(query, params=None):
    """Read query results into a pandas DataFrame."""
    import pandas as pd
    conn = get_connection()
    try:
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        return df
    except Exception as e:
        conn.close()
        raise e


def init_tables():
    """Create all tables (works for both SQLite and PostgreSQL)."""
    mode = _detect_mode()

    if mode == "postgres":
        # PostgreSQL syntax
        auto_id = "SERIAL PRIMARY KEY"
        timestamp_default = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
    else:
        # SQLite syntax
        auto_id = "INTEGER PRIMARY KEY AUTOINCREMENT"
        timestamp_default = "TIMESTAMP DEFAULT CURRENT_TIMESTAMP"

    statements = [
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
            created_at {timestamp_default}
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
            created_at {timestamp_default}
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
            last_updated TEXT
        )""",

        f"""CREATE TABLE IF NOT EXISTS users (
            id {auto_id},
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            student_id TEXT UNIQUE NOT NULL,
            created_at {timestamp_default}
        )""",

        f"""CREATE TABLE IF NOT EXISTS student_profiles (
            student_id TEXT PRIMARY KEY,
            learning_style TEXT DEFAULT 'unknown',
            tab_switch_tolerance REAL DEFAULT 0,
            skip_tolerance REAL DEFAULT 0,
            replay_tolerance REAL DEFAULT 0,
            speed_preference REAL DEFAULT 1.0,
            focus_weight_tab_switch REAL DEFAULT -0.15,
            focus_weight_idle_time REAL DEFAULT -0.15,
            focus_weight_clicks REAL DEFAULT 0.05,
            focus_weight_mouse_movement REAL DEFAULT 0.03,
            focus_weight_replay_count REAL DEFAULT -0.05,
            focus_weight_skip_count REAL DEFAULT -0.1,
            focus_weight_speed_deviation REAL DEFAULT -0.08,
            state_threshold_focused REAL DEFAULT 70,
            state_threshold_moderate REAL DEFAULT 50,
            state_threshold_low REAL DEFAULT 35,
            total_sessions INTEGER DEFAULT 0,
            last_updated TEXT
        )""",
    ]

    conn = get_connection()
    cur = conn.cursor()
    for stmt in statements:
        cur.execute(stmt)
    conn.commit()
    cur.close()
    conn.close()
    print(f"[DB] Tables initialized ({mode})")


def migrate_sqlite_to_postgres():
    """
    One-time migration: copy all data from local SQLite to PostgreSQL.
    Run this manually when switching to cloud DB.
    """
    import pandas as pd

    if not DATABASE_URL:
        print("Set DATABASE_URL env var first!")
        return

    if not os.path.exists(SQLITE_PATH):
        print("No local SQLite database found.")
        return

    # Read from SQLite
    sqlite_conn = sqlite3.connect(SQLITE_PATH)
    tables = ["snapshots", "sessions", "student_baselines", "users", "student_profiles"]

    import psycopg2
    pg_conn = psycopg2.connect(DATABASE_URL, sslmode="require")
    pg_cur = pg_conn.cursor()

    for table in tables:
        try:
            df = pd.read_sql_query(f"SELECT * FROM {table}", sqlite_conn)
            if len(df) == 0:
                print(f"  {table}: empty, skipping")
                continue

            # Drop 'id' column (PostgreSQL SERIAL handles it)
            if "id" in df.columns:
                df = df.drop(columns=["id"])

            cols = list(df.columns)
            placeholders = ", ".join(["%s"] * len(cols))
            col_names = ", ".join(cols)
            insert_sql = f"INSERT INTO {table} ({col_names}) VALUES ({placeholders})"

            for _, row in df.iterrows():
                values = [None if pd.isna(v) else v for v in row.values]
                try:
                    pg_cur.execute(insert_sql, values)
                except Exception as e:
                    # Skip duplicates
                    pg_conn.rollback()
                    continue

            pg_conn.commit()
            print(f"  {table}: migrated {len(df)} rows")

        except Exception as e:
            print(f"  {table}: error — {e}")
            pg_conn.rollback()

    sqlite_conn.close()
    pg_cur.close()
    pg_conn.close()
    print("Migration complete!")


# Quick self-test
if __name__ == "__main__":
    mode = _detect_mode()
    print(f"Database mode: {mode}")
    if mode == "postgres":
        print(f"URL: {DATABASE_URL[:30]}...")
    else:
        print(f"Path: {SQLITE_PATH}")

    init_tables()
    print("Tables OK!")
