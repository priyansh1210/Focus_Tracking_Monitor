"""
Streamlit Dashboard — Student Focus Monitor (Owner View)
========================================================
Owner/admin dashboard with:
- Authentication gate
- Real-time all-student overview with alerts
- Per-student deep dive
- Personal model training status
- Live monitor with auto-refresh
- Model performance comparison
- Dataset explorer
"""

import os
import sys
import json
import sqlite3
import time
from datetime import datetime, timedelta

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

DB_PATH = os.path.join(PROJECT_ROOT, "backend", "focus_monitor.db")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "student_focus_dataset_engineered.csv")
METRICS_PATH = os.path.join(PROJECT_ROOT, "outputs", "metrics.json")
STUDENT_MODELS_DIR = os.path.join(PROJECT_ROOT, "models_saved", "students")

# Database: use DATABASE_URL for PostgreSQL (cloud), else SQLite (local)
DATABASE_URL = os.environ.get("DATABASE_URL", "")
# Streamlit Cloud: also check st.secrets
if not DATABASE_URL:
    try:
        DATABASE_URL = st.secrets.get("DATABASE_URL", "")
    except Exception:
        pass
USE_POSTGRES = bool(DATABASE_URL)

# Owner credentials (in production, use env vars or secrets)
OWNER_PASSWORD = os.environ.get("OWNER_PASSWORD", "admin123")
try:
    OWNER_PASSWORD = st.secrets.get("OWNER_PASSWORD", OWNER_PASSWORD)
except Exception:
    pass

# ---- Page Config ----
st.set_page_config(
    page_title="Focus Monitor - Owner Dashboard",
    page_icon="FM",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---- Custom CSS (Material Dashboard Light Theme) ----
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');

    /* Main background */
    .stApp {
        background-color: #f0f2f5;
        font-family: 'Roboto', sans-serif;
    }

    /* Metric cards — white with shadow, green icon style */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        transition: box-shadow 0.3s ease;
    }
    div[data-testid="stMetric"]:hover {
        box-shadow: 0 4px 20px rgba(0,0,0,0.12);
    }
    div[data-testid="stMetric"] label {
        color: #7b809a !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-family: 'Roboto', sans-serif !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #344767 !important;
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        white-space: nowrap !important;
        overflow: visible !important;
        text-overflow: unset !important;
        font-family: 'Roboto', sans-serif !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricDelta"] {
        color: #4caf50 !important;
        font-size: 0.8rem !important;
    }

    /* Sidebar — dark gradient like Material Dashboard */
    section[data-testid="stSidebar"] {
        background: linear-gradient(195deg, #42424a 0%, #191919 100%);
        border-right: none;
    }
    section[data-testid="stSidebar"] * {
        color: #ffffff !important;
    }
    /* Sidebar metric cards — dark background so text is visible */
    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: rgba(255,255,255,0.1) !important;
        box-shadow: none !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetric"] label {
        color: rgba(255,255,255,0.7) !important;
    }
    section[data-testid="stSidebar"] div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: rgba(255,255,255,0.8) !important;
        font-weight: 400 !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        color: #ffffff !important;
    }
    section[data-testid="stSidebar"] hr {
        border-color: rgba(255,255,255,0.15) !important;
    }

    /* Headers */
    h1 {
        color: #344767 !important;
        font-weight: 700 !important;
        font-family: 'Roboto', sans-serif !important;
        font-size: 1.8rem !important;
    }
    h2, h3 {
        color: #344767 !important;
        font-weight: 600 !important;
        font-family: 'Roboto', sans-serif !important;
    }

    /* Body text */
    p, span, div {
        font-family: 'Roboto', sans-serif;
    }
    .stMarkdown p {
        color: #344767;
    }

    /* Dividers */
    hr {
        border-color: #e9ecef !important;
    }

    /* Alert cards */
    .alert-card {
        background: #ffffff;
        border: none;
        border-left: 4px solid #f44335;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        color: #344767;
    }
    .alert-card-warning {
        background: #ffffff;
        border: none;
        border-left: 4px solid #fb8c00;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 8px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
        color: #344767;
    }

    /* Student card */
    .student-card {
        background: #ffffff;
        border: none;
        border-radius: 12px;
        padding: 18px;
        margin: 10px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
    }

    /* Status badges */
    .badge-focused { color: #4caf50; font-weight: 700; }
    .badge-distracted { color: #fb8c00; font-weight: 700; }
    .badge-confused { color: #1a73e8; font-weight: 700; }
    .badge-bored { color: #7b1fa2; font-weight: 700; }

    /* Progress bars — green like Material */
    .stProgress > div > div > div {
        background: linear-gradient(195deg, #66bb6a, #43a047) !important;
        border-radius: 4px;
    }
    .stProgress > div > div {
        background-color: #e9ecef !important;
        border-radius: 4px;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(195deg, #42424a, #191919);
        color: white;
        border: none;
        border-radius: 8px;
        font-weight: 500;
        padding: 8px 24px;
        font-family: 'Roboto', sans-serif;
        transition: box-shadow 0.3s ease;
    }
    .stButton > button:hover {
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(195deg, #66bb6a, #43a047);
    }

    /* Selectbox, text input */
    .stSelectbox, .stTextInput {
        font-family: 'Roboto', sans-serif;
    }
    .stTextInput input, .stSelectbox > div > div {
        border-radius: 8px !important;
        border-color: #d2d6da !important;
        color: #344767 !important;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab"] {
        color: #7b809a;
        font-weight: 500;
        font-family: 'Roboto', sans-serif;
    }
    .stTabs [aria-selected="true"] {
        color: #344767 !important;
        border-bottom-color: #4caf50 !important;
    }

    /* Dataframe */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }

    /* Expander */
    .streamlit-expanderHeader {
        background: #ffffff !important;
        border-radius: 8px !important;
        color: #344767 !important;
        font-weight: 500 !important;
    }

    /* Container/card effect for content blocks */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-radius: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.08);
        padding: 8px;
    }

    /* Hide default streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

STATE_COLORS = {
    "focused": "#4caf50",
    "distracted": "#fb8c00",
    "confused": "#1a73e8",
    "bored": "#7b1fa2",
}
STATE_ICONS = {
    "focused": "●",
    "distracted": "●",
    "confused": "●",
    "bored": "●",
}

# Chart layout defaults for light theme
CHART_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font_color="#344767",
    font_family="Roboto, sans-serif",
    xaxis=dict(gridcolor="#e9ecef", linecolor="#e9ecef"),
    yaxis=dict(gridcolor="#e9ecef", linecolor="#e9ecef"),
    legend=dict(font=dict(color="#344767")),
)


# ---- Authentication ----

def check_auth():
    """Simple owner authentication."""
    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False

    if not st.session_state.authenticated:
        st.markdown("## Student Focus Monitor")
        st.markdown("### Owner Login")
        st.markdown("---")

        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            password = st.text_input("Enter owner password", type="password", key="login_pw")
            if st.button("Login", use_container_width=True, type="primary"):
                if password == OWNER_PASSWORD:
                    st.session_state.authenticated = True
                    st.rerun()
                else:
                    st.error("Invalid password")
        return False
    return True


# ---- Data Loading ----

def _get_conn():
    """Get database connection — PostgreSQL or SQLite."""
    if USE_POSTGRES:
        import psycopg2
        return psycopg2.connect(DATABASE_URL, sslmode="require")
    else:
        if not os.path.exists(DB_PATH):
            return None
        return sqlite3.connect(DB_PATH)


def _read_sql(query):
    """Read SQL query into DataFrame, works with both backends."""
    conn = _get_conn()
    if conn is None:
        return None
    try:
        df = pd.read_sql_query(query, conn)
        conn.close()
        return df
    except Exception:
        conn.close()
        return None


@st.cache_data(ttl=10)
def load_real_snapshots():
    return _read_sql("SELECT * FROM snapshots ORDER BY id")


@st.cache_data(ttl=10)
def load_sessions():
    return _read_sql("SELECT * FROM sessions ORDER BY start_time DESC")


@st.cache_data(ttl=10)
def load_users():
    return _read_sql("SELECT student_id, username, created_at FROM users")


@st.cache_data(ttl=10)
def load_profiles():
    return _read_sql("SELECT * FROM student_profiles")


@st.cache_data(ttl=10)
def load_baselines():
    return _read_sql("SELECT * FROM student_baselines")


@st.cache_data
def load_synthetic_data():
    if os.path.exists(DATA_PATH):
        return pd.read_csv(DATA_PATH)
    return None


@st.cache_data
def load_metrics():
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH) as f:
            return json.load(f)
    return None


def load_student_models_info():
    """Load metadata from all personal models."""
    import joblib
    models = {}
    if not os.path.exists(STUDENT_MODELS_DIR):
        return models
    for fname in os.listdir(STUDENT_MODELS_DIR):
        if fname.endswith(".joblib"):
            try:
                bundle = joblib.load(os.path.join(STUDENT_MODELS_DIR, fname))
                meta = bundle.get("meta", {})
                models[meta.get("student_id", fname)] = meta
            except Exception:
                pass
    return models


# ---- Sidebar ----

def render_sidebar():
    st.sidebar.markdown("## Focus Monitor")
    st.sidebar.markdown(f"**Owner Dashboard**")
    st.sidebar.markdown("---")

    page = st.sidebar.radio("Navigation", [
        "Overview",
        "All Students",
        "Student Deep Dive",
        "Live Monitor",
        "Personal Models",
        "Model Performance",
        "Dataset Explorer",
    ])

    st.sidebar.markdown("---")

    # Quick stats in sidebar
    snaps = load_real_snapshots()
    if snaps is not None and len(snaps) > 0:
        st.sidebar.metric("Total Snapshots", len(snaps))
        st.sidebar.metric("Active Students", snaps["student_id"].nunique())
    else:
        st.sidebar.info("No live data yet")

    st.sidebar.markdown("---")
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.authenticated = False
        st.rerun()

    return page


# ---- Pages ----

def overview_page():
    st.markdown("# Dashboard Overview")

    snaps = load_real_snapshots()
    sessions = load_sessions()
    users = load_users()
    df_synth = load_synthetic_data()

    # Top metrics
    c1, c2, c3, c4, c5 = st.columns(5)

    if snaps is not None and len(snaps) > 0:
        c1.metric("Total Snapshots", f"{len(snaps):,}")
        c2.metric("Students", snaps["student_id"].nunique())
        n_sessions = sessions["session_id"].nunique() if sessions is not None and len(sessions) > 0 else 0
        c3.metric("Sessions", n_sessions)
        avg_focus = snaps["focus_score"].mean()
        c4.metric("Avg Focus Score", f"{avg_focus:.1f}")

        # Dominant state
        if "predicted_state" in snaps.columns:
            dominant = snaps["predicted_state"].mode()
            if len(dominant) > 0:
                c5.metric("Most Common State", dominant[0].capitalize())
    else:
        c1.metric("Total Snapshots", "0")
        c2.metric("Students", "0")
        c3.metric("Sessions", "0")
        c4.metric("Avg Focus Score", "--")
        c5.metric("Most Common State", "--")
        st.info("No live data yet. Start the backend and Chrome extension to collect data.")

    # Behavioral totals (paused / away / idle) — new fields from extension rewrite
    if snaps is not None and len(snaps) > 0:
        d1, d2, d3 = st.columns(3)
        total_paused = snaps["paused_time"].sum() if "paused_time" in snaps.columns else 0
        total_away = snaps["away_time"].sum() if "away_time" in snaps.columns else 0
        total_idle = snaps["idle_time"].sum() if "idle_time" in snaps.columns else 0
        d1.metric("Total Paused", f"{total_paused / 60:.1f} min")
        d2.metric("Total Away", f"{total_away / 60:.1f} min")
        d3.metric("Total Idle", f"{total_idle / 60:.1f} min")

    st.markdown("---")

    if snaps is not None and len(snaps) > 0:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### Cognitive State Distribution")
            if "predicted_state" in snaps.columns:
                state_counts = snaps["predicted_state"].value_counts().reset_index()
                state_counts.columns = ["State", "Count"]
                fig = px.pie(state_counts, values="Count", names="State",
                             color="State", color_discrete_map=STATE_COLORS, hole=0.45)
                fig.update_layout(
                    height=350, margin=dict(t=20, b=20, l=20, r=20),
                    **CHART_LAYOUT,
                )
                st.plotly_chart(fig, use_container_width=True)

        with col_right:
            st.markdown("### Focus Score Distribution")
            fig = px.histogram(snaps, x="focus_score", nbins=30,
                               color_discrete_sequence=["#4caf50"])
            fig.add_vline(x=50, line_dash="dash", line_color="#f44335",
                          annotation_text="Low Focus Threshold")
            fig.update_layout(
                height=350, margin=dict(t=20, b=20, l=20, r=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font_color="#344767",
                xaxis=dict(gridcolor="#e9ecef"),
                yaxis=dict(gridcolor="#e9ecef"),
            )
            st.plotly_chart(fig, use_container_width=True)

        # Focus score over time (all students)
        st.markdown("### Focus Score Timeline (All Students)")
        snaps_sorted = snaps.sort_values("id")
        fig = px.scatter(snaps_sorted, x="id", y="focus_score",
                         color="predicted_state" if "predicted_state" in snaps.columns else None,
                         color_discrete_map=STATE_COLORS,
                         opacity=0.7, size_max=6)
        fig.add_hline(y=50, line_dash="dash", line_color="#f44335", opacity=0.5)
        fig.update_layout(
            height=350, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font_color="#344767",
            xaxis=dict(gridcolor="#e9ecef", title="Snapshot #"),
            yaxis=dict(gridcolor="#e9ecef", title="Focus Score"),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Alerts
        st.markdown("### Alerts")
        alerts = []
        for sid in snaps["student_id"].unique():
            student_snaps = snaps[snaps["student_id"] == sid]
            recent = student_snaps.tail(5)
            avg_recent = recent["focus_score"].mean()

            if avg_recent < 30:
                alerts.append(("critical", sid, f"Very low focus ({avg_recent:.0f}) in last 5 snapshots"))
            elif avg_recent < 50:
                alerts.append(("warning", sid, f"Below-average focus ({avg_recent:.0f}) in last 5 snapshots"))

            if "predicted_state" in recent.columns:
                distracted_pct = (recent["predicted_state"] == "distracted").mean()
                if distracted_pct >= 0.6:
                    alerts.append(("warning", sid, f"Frequently distracted ({distracted_pct:.0%} of recent snapshots)"))

        if alerts:
            for level, sid, msg in alerts:
                css_class = "alert-card" if level == "critical" else "alert-card-warning"
                icon = "CRITICAL:" if level == "critical" else "WARNING:"
                st.markdown(f'<div class="{css_class}">{icon} <strong>{sid}</strong>: {msg}</div>',
                            unsafe_allow_html=True)
        else:
            st.success("No alerts — all students are doing well!")


def all_students_page():
    st.markdown("# All Students")

    snaps = load_real_snapshots()
    users = load_users()
    profiles = load_profiles()

    if snaps is None or len(snaps) == 0:
        st.info("No student data yet.")
        return

    # Build per-student summary
    student_data = []
    for sid in snaps["student_id"].unique():
        s = snaps[snaps["student_id"] == sid]
        recent = s.tail(10)

        username = "--"
        if users is not None:
            user_row = users[users["student_id"] == sid]
            if len(user_row) > 0:
                username = user_row.iloc[0]["username"]

        learning_style = "--"
        if profiles is not None:
            prof_row = profiles[profiles["student_id"] == sid]
            if len(prof_row) > 0:
                learning_style = prof_row.iloc[0]["learning_style"]

        dominant = "unknown"
        if "predicted_state" in recent.columns:
            mode = recent["predicted_state"].mode()
            if len(mode) > 0:
                dominant = mode[0]

        avg_paused = round(s["paused_time"].mean(), 1) if "paused_time" in s.columns else 0
        avg_away = round(s["away_time"].mean(), 1) if "away_time" in s.columns else 0
        avg_idle = round(s["idle_time"].mean(), 1) if "idle_time" in s.columns else 0

        student_data.append({
            "Student ID": sid,
            "Username": username,
            "Total Snapshots": len(s),
            "Sessions": s["session_id"].nunique(),
            "Avg Focus": round(s["focus_score"].mean(), 1),
            "Recent Focus (last 10)": round(recent["focus_score"].mean(), 1),
            "Current State": dominant,
            "Learning Style": learning_style.capitalize(),
            "Avg Idle (s)": avg_idle,
            "Avg Paused (s)": avg_paused,
            "Avg Away (s)": avg_away,
            "Last Active": s["timestamp"].iloc[-1] if "timestamp" in s.columns else "--",
        })

    df_students = pd.DataFrame(student_data)

    # Summary metrics
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Students", len(df_students))
    c2.metric("Avg Class Focus", f"{df_students['Avg Focus'].mean():.1f}")

    struggling = len(df_students[df_students["Avg Focus"] < 50])
    c3.metric("Struggling Students", struggling, delta=None)

    top_focus = df_students["Avg Focus"].max()
    c4.metric("Highest Focus", f"{top_focus:.1f}")

    st.markdown("---")

    # Student cards
    st.markdown("### Student Roster")

    for _, row in df_students.iterrows():
        state = row["Current State"]
        emoji = STATE_ICONS.get(state, "●")
        color = STATE_COLORS.get(state, "#888")
        focus = row["Avg Focus"]

        # Focus bar color
        if focus >= 70:
            bar_color = "#22c55e"
        elif focus >= 50:
            bar_color = "#f59e0b"
        else:
            bar_color = "#ef4444"

        with st.container():
            cols = st.columns([2, 1, 1, 1, 1, 1, 1])
            cols[0].markdown(f"**{row['Username']}** (`{row['Student ID']}`)")
            cols[1].markdown(f"Snapshots: **{row['Total Snapshots']}**")
            cols[2].markdown(f"Sessions: **{row['Sessions']}**")
            cols[3].markdown(f"Avg Focus: **{focus}**")
            cols[4].markdown(f"Recent: **{row['Recent Focus (last 10)']}**")
            cols[5].markdown(f"{emoji} **{state.capitalize()}**")
            cols[6].markdown(f"Style: **{row['Learning Style']}**")

            # Behavioral averages row
            bcols = st.columns([2, 1, 1, 1])
            bcols[0].markdown("<span style='color:#6c757d;font-size:12px;'>Avg behavioral (per snapshot)</span>", unsafe_allow_html=True)
            bcols[1].markdown(f"Idle: **{row['Avg Idle (s)']}s**")
            bcols[2].markdown(f"Paused: **{row['Avg Paused (s)']}s**")
            bcols[3].markdown(f"Away: **{row['Avg Away (s)']}s**")

            # Mini focus bar
            st.progress(min(focus / 100, 1.0))
            st.markdown("")

    # Comparison chart
    st.markdown("### Focus Score Comparison")
    fig = px.bar(df_students.sort_values("Avg Focus", ascending=True),
                 x="Avg Focus", y="Username", orientation="h",
                 color="Avg Focus",
                 color_continuous_scale=["#f44335", "#fb8c00", "#4caf50"],
                 range_color=[0, 100])
    fig.add_vline(x=50, line_dash="dash", line_color="#f44335", opacity=0.5)
    fig.update_layout(
        height=max(250, len(df_students) * 60),
        margin=dict(t=20, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#344767",
        xaxis=dict(gridcolor="#e9ecef", range=[0, 100]),
        yaxis=dict(gridcolor="#e9ecef"),
    )
    st.plotly_chart(fig, use_container_width=True)


def student_deep_dive_page():
    st.markdown("# Student Deep Dive")

    snaps = load_real_snapshots()
    df_synth = load_synthetic_data()

    # Choose data source
    source = st.radio("Data Source", ["Live Data", "Synthetic Dataset"], horizontal=True)

    if source == "Live Data":
        if snaps is None or len(snaps) == 0:
            st.info("No live data. Use synthetic dataset instead.")
            return
        df = snaps
        id_col = "student_id"
        state_col = "predicted_state"
    else:
        if df_synth is None:
            st.warning("Synthetic dataset not found.")
            return
        df = df_synth
        id_col = "student_id"
        state_col = "state"

    student_ids = sorted(df[id_col].unique())
    selected = st.selectbox("Select Student", student_ids)
    student_df = df[df[id_col] == selected].copy()

    if "timestamp" in student_df.columns:
        student_df = student_df.sort_values("timestamp")
    elif "id" in student_df.columns:
        student_df = student_df.sort_values("id")

    # Metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Snapshots", len(student_df))
    c2.metric("Sessions", student_df["session_id"].nunique() if "session_id" in student_df.columns else "--")
    c3.metric("Avg Focus", f"{student_df['focus_score'].mean():.1f}")

    if state_col in student_df.columns:
        dominant = student_df[state_col].mode()
        c4.metric("Dominant State", dominant[0].capitalize() if len(dominant) > 0 else "--")
        focused_pct = (student_df[state_col] == "focused").mean()
        c5.metric("Focused %", f"{focused_pct:.0%}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("### State Distribution")
        if state_col in student_df.columns:
            state_dist = student_df[state_col].value_counts().reset_index()
            state_dist.columns = ["State", "Count"]
            fig = px.pie(state_dist, values="Count", names="State",
                         color="State", color_discrete_map=STATE_COLORS, hole=0.45)
            fig.update_layout(
                height=300, margin=dict(t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#344767",
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.markdown("### Focus Score Over Time")
        fig = go.Figure()
        y_vals = student_df["focus_score"].values
        fig.add_trace(go.Scatter(
            y=y_vals, mode="lines+markers",
            marker=dict(size=4, color="#4caf50"),
            line=dict(width=2, color="#4caf50"),
        ))
        # Rolling average
        if len(y_vals) > 5:
            rolling = pd.Series(y_vals).rolling(5, min_periods=1).mean()
            fig.add_trace(go.Scatter(
                y=rolling, mode="lines",
                line=dict(width=3, color="#f59e0b", dash="dot"),
                name="5-pt Average",
            ))
        fig.add_hline(y=50, line_dash="dash", line_color="#f44335", opacity=0.5)
        fig.update_layout(
            height=300, margin=dict(t=10, b=10),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#344767",
            xaxis=dict(gridcolor="#e9ecef", title="Snapshot"),
            yaxis=dict(gridcolor="#e9ecef", title="Focus Score"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Behavioral radar
    st.markdown("### Behavioral Profile")
    features = ["tab_switch", "idle_time", "paused_time", "away_time", "clicks",
                "mouse_movement", "replay_count", "skip_count"]
    available_feats = [f for f in features if f in student_df.columns and f in df.columns]

    if available_feats:
        student_means = student_df[available_feats].mean()
        global_means = df[available_feats].mean()
        ratios = (student_means / global_means.replace(0, 1)).values

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=ratios, theta=available_feats, fill="toself",
            name=f"Student {selected}", line_color="#4caf50",
        ))
        fig.add_trace(go.Scatterpolar(
            r=[1] * len(available_feats), theta=available_feats, fill="toself",
            name="Class Average", line_color="#bdbdbd", opacity=0.3,
        ))
        fig.update_layout(
            height=400, margin=dict(t=30, b=30),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#344767",
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 3], gridcolor="#e9ecef"),
                angularaxis=dict(gridcolor="#e9ecef"),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Feature timeline
    st.markdown("### Feature Timeline")
    feature_choice = st.selectbox("Select Feature",
                                  ["focus_score"] + [f for f in features if f in student_df.columns])
    if feature_choice in student_df.columns:
        fig = px.line(student_df.reset_index(), y=feature_choice,
                      color_discrete_sequence=["#4caf50"])
        fig.update_layout(
            height=300, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font_color="#344767",
            xaxis=dict(gridcolor="#e9ecef"), yaxis=dict(gridcolor="#e9ecef"),
        )
        st.plotly_chart(fig, use_container_width=True)


def live_monitor_page():
    st.markdown("# Live Monitor")

    # Auto-refresh toggle
    col1, col2 = st.columns([3, 1])
    with col2:
        auto_refresh = st.checkbox("Auto-refresh (10s)", value=False)
    if auto_refresh:
        time.sleep(0.1)
        st.cache_data.clear()

    snaps = load_real_snapshots()

    if snaps is None or len(snaps) == 0:
        st.info("No live data yet. Start the backend and Chrome extension.")
        st.code("py backend/app.py", language="bash")
        return

    # Active students panel
    st.markdown("### Active Students")

    for sid in snaps["student_id"].unique():
        student_snaps = snaps[snaps["student_id"] == sid].copy()
        latest = student_snaps.iloc[-1]
        recent = student_snaps.tail(10)

        state = latest.get("predicted_state", "unknown")
        emoji = STATE_ICONS.get(state, "●")
        focus = latest.get("focus_score", 0)

        with st.container():
            cols = st.columns([3, 1, 1, 1, 3])

            website = latest.get("website", "")
            website_display = f"[{website}](https://{website})" if website else "--"
            cols[0].markdown(f"**{emoji} {sid}**\n\n{website_display}")
            cols[1].metric("Focus", f"{focus:.0f}")
            cols[2].metric("State", state.capitalize() if state else "--")
            cols[3].metric("Snapshots", len(student_snaps))

            # Latest behavioral readings
            idle_s = latest.get("idle_time", 0) or 0
            paused_s = latest.get("paused_time", 0) or 0
            away_s = latest.get("away_time", 0) or 0
            bcols = st.columns(3)
            bcols[0].metric("Idle (last snap)", f"{idle_s:.0f}s")
            bcols[1].metric("Paused (last snap)", f"{paused_s:.0f}s")
            bcols[2].metric("Away (last snap)", f"{away_s:.0f}s")

            # Mini sparkline
            with cols[4]:
                spark_data = recent["focus_score"].values
                fig = go.Figure()
                colors = [STATE_COLORS.get(s, "#888") for s in recent["predicted_state"]] if "predicted_state" in recent.columns else ["#3b82f6"] * len(spark_data)
                fig.add_trace(go.Scatter(
                    y=spark_data, mode="lines+markers",
                    marker=dict(size=5, color=colors),
                    line=dict(width=2, color="#4caf50"),
                    showlegend=False,
                ))
                fig.add_hline(y=50, line_dash="dot", line_color="#f44335", opacity=0.3)
                fig.update_layout(
                    height=80, margin=dict(t=5, b=5, l=5, r=5),
                    paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                    xaxis=dict(visible=False), yaxis=dict(visible=False, range=[0, 100]),
                )
                st.plotly_chart(fig, use_container_width=True, key=f"spark_{sid}")

            st.markdown("---")

    # Recent snapshots table
    st.markdown("### Recent Snapshots")
    recent_all = snaps.tail(20).iloc[::-1]
    display_cols = ["student_id", "timestamp", "website", "focus_score", "predicted_state",
                    "tab_switch", "idle_time", "paused_time", "away_time", "clicks",
                    "mouse_movement", "replay_count", "skip_count", "playback_speed"]
    available = [c for c in display_cols if c in recent_all.columns]
    st.dataframe(recent_all[available], use_container_width=True, hide_index=True)

    if auto_refresh:
        time.sleep(10)
        st.rerun()


def personal_models_page():
    st.markdown("# Personal Models")

    snaps = load_real_snapshots()
    models_info = load_student_models_info()

    if snaps is None or len(snaps) == 0:
        st.info("No data yet. Personal models train after 300 snapshots per student.")
        return

    st.markdown("""
    Personal models train automatically:
    - **First training**: After 300 snapshots with 3+ cognitive states
    - **Retraining**: Every 50 new snapshots
    - **Minimum quality**: CV F1 > 0.60 required to save
    - **Confidence threshold**: Model predictions used only when confidence >= 55%
    """)

    st.markdown("---")

    # Per-student status
    st.markdown("### Per-Student Model Status")

    for sid in sorted(snaps["student_id"].unique()):
        student_snaps = snaps[snaps["student_id"] == sid]
        n_snaps = len(student_snaps)
        progress = min(n_snaps / 300, 1.0)

        has_model = sid in models_info
        meta = models_info.get(sid, {})

        with st.container():
            cols = st.columns([2, 1, 1, 1, 2])

            cols[0].markdown(f"**{sid}**")
            cols[1].markdown(f"Snapshots: **{n_snaps}** / 300")

            if has_model:
                cols[2].markdown(f"**Model trained**")
                cols[3].markdown(f"CV F1: **{meta.get('cv_mean', '--')}**")
                cols[4].markdown(f"Trained: {meta.get('trained_at', '--')[:19]}")
            elif n_snaps >= 300:
                # Check class diversity
                if "predicted_state" in student_snaps.columns:
                    n_classes = student_snaps["predicted_state"].nunique()
                    if n_classes >= 3:
                        cols[2].markdown("**Ready to train**")
                    else:
                        cols[2].markdown(f"Need 3+ classes (have {n_classes})")
                else:
                    cols[2].markdown("No state labels")
                cols[3].markdown("--")
                cols[4].markdown("--")
            else:
                remaining = 300 - n_snaps
                cols[2].markdown(f"Need **{remaining}** more snapshots")
                cols[3].markdown("--")
                mins = remaining * 0.5  # 30s per snapshot
                cols[4].markdown(f"~{mins:.0f} min of study time")

            st.progress(progress)
            st.markdown("")

    # Model details
    if models_info:
        st.markdown("### Trained Model Details")
        for sid, meta in models_info.items():
            with st.expander(f"Model: {sid}"):
                c1, c2, c3 = st.columns(3)
                c1.metric("CV F1 Score", f"{meta.get('cv_mean', 0):.4f}")
                c2.metric("Training Samples", meta.get("n_snapshots", 0))
                c3.metric("Features", meta.get("n_features", 0))

                st.markdown(f"**Classes**: {', '.join(meta.get('classes', []))}")
                st.markdown(f"**Trained at**: {meta.get('trained_at', '--')}")

                if "class_distribution" in meta:
                    dist = meta["class_distribution"]
                    fig = px.bar(x=list(dist.keys()), y=list(dist.values()),
                                 color=list(dist.keys()), color_discrete_map=STATE_COLORS)
                    fig.update_layout(
                        height=250, margin=dict(t=20, b=20),
                        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                        font_color="#344767", showlegend=False,
                        xaxis=dict(gridcolor="#e9ecef", title="State"),
                        yaxis=dict(gridcolor="#e9ecef", title="Count"),
                    )
                    st.plotly_chart(fig, use_container_width=True)


def model_performance_page():
    st.markdown("# Model Performance")

    metrics = load_metrics()
    if metrics is None:
        st.warning("Run evaluation first: `py src/evaluation.py`")
        return

    # Model comparison
    st.markdown("### Model Comparison")
    comparison = []
    for model_name, data in metrics.items():
        comparison.append({
            "Model": model_name,
            "Accuracy": f"{data['accuracy']:.4f}",
            "F1 (Weighted)": f"{data['f1_weighted']:.4f}",
            "F1 (Macro)": f"{data['f1_macro']:.4f}",
            "CV Mean": f"{data.get('cv_mean', 'N/A')}",
        })
    st.dataframe(pd.DataFrame(comparison), use_container_width=True, hide_index=True)

    # Bar chart
    model_names = list(metrics.keys())
    accuracies = [metrics[m]["accuracy"] for m in model_names]
    f1_scores = [metrics[m]["f1_weighted"] for m in model_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Accuracy", x=model_names, y=accuracies,
                         marker_color="#4caf50",
                         text=[f"{a:.2%}" for a in accuracies], textposition="outside"))
    fig.add_trace(go.Bar(name="F1 Weighted", x=model_names, y=f1_scores,
                         marker_color="#1a73e8",
                         text=[f"{f:.2%}" for f in f1_scores], textposition="outside"))
    fig.update_layout(
        height=400, barmode="group",
        yaxis_range=[min(min(accuracies), min(f1_scores)) - 0.05, 1.02],
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font_color="#344767",
        xaxis=dict(gridcolor="#e9ecef"), yaxis=dict(gridcolor="#e9ecef"),
        legend=dict(font=dict(color="#e2e8f0")),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Per-class performance
    st.markdown("### Per-Class Performance")
    for model_name, data in metrics.items():
        with st.expander(f"{model_name}", expanded=True):
            report = data.get("classification_report", {})
            class_data = []
            for cls in ["focused", "confused", "distracted", "bored"]:
                if cls in report:
                    class_data.append({
                        "Class": cls.capitalize(),
                        "Precision": f"{report[cls]['precision']:.4f}",
                        "Recall": f"{report[cls]['recall']:.4f}",
                        "F1-Score": f"{report[cls]['f1-score']:.4f}",
                        "Support": report[cls].get("support", "--"),
                    })
            if class_data:
                st.dataframe(pd.DataFrame(class_data), use_container_width=True, hide_index=True)

    # Plots
    plots_dir = os.path.join(PROJECT_ROOT, "outputs", "plots")
    if os.path.exists(plots_dir):
        st.markdown("### Evaluation Plots")
        plot_files = sorted([f for f in os.listdir(plots_dir) if f.endswith(".png")])
        cols = st.columns(2)
        for i, pf in enumerate(plot_files):
            with cols[i % 2]:
                st.image(os.path.join(plots_dir, pf),
                         caption=pf.replace("_", " ").replace(".png", "").title())


def dataset_explorer_page():
    st.markdown("# Dataset Explorer")

    tab1, tab2 = st.tabs(["Synthetic Dataset", "Live Data"])

    with tab1:
        df = load_synthetic_data()
        if df is None:
            st.warning("Synthetic dataset not found.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Rows", f"{len(df):,}")
            c2.metric("Students", df["student_id"].nunique())
            c3.metric("Features", len(df.columns))

            st.markdown("### Sample Data")
            st.dataframe(df.head(100), use_container_width=True, hide_index=True)

            st.markdown("### Statistics")
            st.dataframe(df.describe().round(2), use_container_width=True)

            st.markdown("### Correlation Heatmap")
            numeric_cols = ["tab_switch", "idle_time", "paused_time", "away_time",
                            "clicks", "mouse_movement", "replay_count", "skip_count",
                            "playback_speed", "focus_score"]
            available = [c for c in numeric_cols if c in df.columns]
            corr = df[available].corr()
            fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                            zmin=-1, zmax=1, aspect="auto")
            fig.update_layout(
                height=500,
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                font_color="#344767",
            )
            st.plotly_chart(fig, use_container_width=True)

    with tab2:
        snaps = load_real_snapshots()
        if snaps is None or len(snaps) == 0:
            st.info("No live data yet.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Snapshots", len(snaps))
            c2.metric("Students", snaps["student_id"].nunique())
            c3.metric("Columns", len(snaps.columns))

            st.markdown("### Live Snapshots")
            st.dataframe(snaps.tail(100).iloc[::-1], use_container_width=True, hide_index=True)

            st.markdown("### Statistics")
            numeric = snaps.select_dtypes(include=[np.number])
            st.dataframe(numeric.describe().round(2), use_container_width=True)


# ---- Main ----

if check_auth():
    page = render_sidebar()

    page_map = {
        "Overview": overview_page,
        "All Students": all_students_page,
        "Student Deep Dive": student_deep_dive_page,
        "Live Monitor": live_monitor_page,
        "Personal Models": personal_models_page,
        "Model Performance": model_performance_page,
        "Dataset Explorer": dataset_explorer_page,
    }

    page_map[page]()
