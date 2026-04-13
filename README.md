# Student Focus Monitor: Real-Time Cognitive State Classification for Online Learning Environments

A full-stack intelligent system that captures behavioral telemetry from students during online learning sessions, classifies their cognitive state using ensemble and sequential machine learning models, and delivers adaptive interventions to improve engagement outcomes. The platform spans a Chrome browser extension for data acquisition, a Flask inference backend with dual-database support, three trained classifiers with per-student personalization, and a multi-page Streamlit analytics dashboard deployed on the cloud.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Proposed Approach](#proposed-approach)
3. [System Architecture](#system-architecture)
4. [Data Acquisition Layer](#data-acquisition-layer)
5. [Dataset Construction](#dataset-construction)
6. [Feature Engineering Pipeline](#feature-engineering-pipeline)
7. [Focus Score Computation](#focus-score-computation)
8. [Classification Models](#classification-models)
9. [Per-Student Model Personalization](#per-student-model-personalization)
10. [Adaptive Response Engine](#adaptive-response-engine)
11. [Backend Infrastructure](#backend-infrastructure)
12. [Analytics Dashboard](#analytics-dashboard)
13. [Deployment](#deployment)
14. [Project Structure](#project-structure)
15. [Setup and Usage](#setup-and-usage)
16. [Ethical Considerations](#ethical-considerations)
17. [Technology Stack](#technology-stack)
18. [Future Scope](#future-scope)

---

## Problem Statement

Existing online learning platforms predominantly rely on surface-level completion metrics (video watch percentage, quiz submission counts) to gauge student performance. These metrics fail to capture whether a student is genuinely engaged, mentally absent, struggling with comprehension, or simply disinterested. Without real-time insight into the cognitive dimension of the learning experience, educators cannot intervene before learning outcomes deteriorate. The absence of granular behavioral analytics means that a student who is confused and repeatedly rewinding a lecture segment receives the same treatment as one who is focused and progressing smoothly.

This project addresses the gap by building an end-to-end pipeline that moves beyond binary active/inactive classification. It captures fine-grained behavioral signals during learning sessions, maps them to four distinct cognitive states — focused, distracted, confused, and bored — and triggers state-appropriate interventions in real time.

---

## Proposed Approach

The system introduces several design decisions that distinguish it from conventional engagement trackers:

**Behavioral-to-Cognitive Mapping.** Rather than treating all disengagement as a single category, the system differentiates between distraction (frequent tab switching, low mouse activity), confusion (high replay counts, extended pauses on video), and boredom (skipping forward, increased playback speed). Each state demands a different pedagogical response, and the model is trained to distinguish between them.

**Per-Student Behavioral Baselines.** A student who habitually switches tabs ten times per session has a different baseline than one who rarely leaves the learning page. The focus score normalizes each behavioral signal against the individual student's historical mean, ensuring that deviations from personal norms — not population averages — drive classification decisions.

**Temporal Sequence Analysis.** A single snapshot of behavior provides limited information. The system analyzes ordered sequences of snapshots within a session: a pattern of increasing idle time followed by a video replay followed by a pause often signals confusion buildup, while a pattern of skips followed by speed increases signals boredom. Both lagged features and an LSTM network capture these temporal dependencies.

**Continuous Focus Scoring.** Instead of discrete labels alone, the system computes a real-valued focus score between 0 and 100 for every snapshot. This score uses weighted behavioral contributions with exponential temporal decay, producing a responsive metric that is more informative than a static threshold classification.

**Progressive Personalization.** After accumulating 300 or more snapshots across at least three cognitive states, the system trains a personal Random Forest classifier for that specific student using their own behavioral distribution. This per-student model supplements the global classifier and retrains automatically every 50 new snapshots.

---

## System Architecture

```
+------------------------+       +------------------------+       +------------------------+
|   Chrome Extension     |       |   Flask Backend         |       |   ML Inference Layer   |
|   (Manifest V3)        | ----> |   (REST API + DB)       | ----> |   (RF / XGB / LSTM)    |
|                        |       |                        |       |                        |
|  - Content Script      |       |  - Snapshot ingestion  |       |  - Global models       |
|  - Background Worker   |       |  - Session management  |       |  - Per-student models  |
|  - Popup UI            |       |  - User auth           |       |  - Focus score engine  |
+------------------------+       +------------------------+       +------------------------+
         |                               |                                |
         |                               v                                |
         |                       +----------------+                       |
         |                       |   Database     |                       |
         |                       |  SQLite (local)|                       |
         |                       |  PostgreSQL    |                       |
         |                       |  (cloud/Neon)  |                       |
         |                       +----------------+                       |
         |                               |                                |
         v                               v                                v
+------------------------+       +------------------------+       +------------------------+
|   Extension Popup      |       | Adaptive Response      |       |  Streamlit Dashboard   |
|   (Live Stats, Score)  | <---- | Engine (Interventions) | <---- |  (7 Analytics Pages)   |
+------------------------+       +------------------------+       +------------------------+
```

Data flows from left to right: the browser extension captures raw behavioral events, the backend persists and processes them, the ML layer classifies each snapshot, and the results feed back into the extension popup for student self-monitoring and into the dashboard for educator oversight.

---

## Data Acquisition Layer

The Chrome extension (Manifest V3) captures eight behavioral signals per snapshot interval through two coordinated scripts:

### Background Service Worker (`background.js`)

The service worker orchestrates session lifecycle, snapshot timing, and cross-tab state management. It maintains a session object that tracks cumulative behavioral counters between snapshot intervals.

| Signal | Detection Mechanism | Behavioral Indication |
|---|---|---|
| Tab switches | `chrome.tabs.onActivated` and `visibilitychange` events | Context switching, multitasking, potential distraction |
| Idle time | Combination of Chrome Idle API (60-second threshold) and interaction timestamp delta | Cognitive pause, confusion, or disengagement |
| Paused time | HTML5 `pause`/`play` events on `<video>` elements | Intentional stop for note-taking or processing |
| Away time | `chrome.windows.onFocusChanged` detecting browser defocus | Complete departure from the learning environment |

When the browser window loses focus, the service worker records the departure timestamp and increments the tab switch counter. On focus return, it closes the away-time interval and resets the idle clock so that the time spent in another application does not inflate the idle metric.

### Content Script (`content.js`)

Injected into every page the student visits, the content script captures granular interaction events and relays them to the background worker:

| Signal | Detection Mechanism | Behavioral Indication |
|---|---|---|
| Clicks | `mousedown` event listener | Active interaction with learning material |
| Mouse movement | `mousemove` with distance accumulation | Sustained visual attention and page navigation |
| Video replays | `seeked` event where `currentTime < previousTime` | Difficulty understanding content, need for repetition |
| Video skips | `seeked` event where `currentTime > previousTime` | Disinterest, already familiar with content |
| Playback speed | `ratechange` event on video element | Rushing through (boredom) or slowing down (confusion) |

The content script uses a resilient message-passing implementation that handles transient Manifest V3 service worker restarts without permanently killing the communication channel, only invalidating the context when the extension is genuinely uninstalled or reloaded.

### Extension Popup

The popup interface displays real-time session statistics to the student: current focus score, predicted cognitive state, session duration, snapshot count, and behavioral breakdowns (idle time, paused time, away time). It provides self-monitoring capability without requiring educator access.

---

## Dataset Construction

The system is pre-trained on a synthetic dataset generated with configurable parameters defined in `config/config.yaml`:

| Parameter | Value |
|---|---|
| Total rows | 40,000 |
| Students | 200 (unique behavioral baselines) |
| Sessions | ~4,500 (25 per student average) |
| Snapshots per session | 8 average |
| Cognitive state classes | 4 (focused, distracted, confused, bored) |
| Rows per class | 10,000 (perfectly balanced) |
| Random seed | 42 (reproducible) |

Each synthetic student is assigned a behavioral baseline profile with characteristic means and variances for all behavioral features. The data generation module (`src/data_generation.py`) applies class-specific behavioral rules:

- **Focused**: Low tab switches, low idle time, moderate-to-high clicks and mouse movement, low replays and skips, playback speed near 1.0
- **Distracted**: High tab switches, moderate idle time, low clicks, variable mouse movement, low replays, moderate skips
- **Confused**: Moderate tab switches, high idle time, low-to-moderate clicks, low mouse movement, high replays, low skips, reduced playback speed
- **Bored**: Moderate tab switches, moderate-to-high idle time, low clicks, low mouse movement, low replays, high skips, elevated playback speed

The generated dataset undergoes six-point validation (`src/data_validation.py`): missing value checks, realistic range enforcement, class balance verification, temporal coherence within sessions, per-student statistical consistency, and ANOVA-based feature discriminability testing (p < 0.05 for all behavioral features across classes).

---

## Feature Engineering Pipeline

The raw dataset contains 7 behavioral features and 6 metadata columns (13 total columns). The feature engineering module (`src/feature_engineering.py`) expands this to 81 columns through four transformation categories:

**Lagged Features (30 features).** For each of the 6 core behavioral signals, the previous 5 snapshot values are captured as separate features. These allow tree-based models to detect temporal trends such as rising idle time or declining click rates within a session.

**Rolling Statistics (18 features).** Three-snapshot rolling windows compute mean, standard deviation, and maximum for each of the 6 core signals. These capture short-term behavioral volatility and peaks.

**Interaction Features.** Cross-feature products and ratios that capture behavioral combinations: for example, the product of tab switches and idle time amplifies signals where both distraction indicators co-occur.

**Per-Student Normalized Features.** Z-score normalization of each behavioral signal against the individual student's historical baseline, producing deviation features that measure how unusual the current snapshot is relative to that specific student's behavioral patterns.

After engineering, the dataset contains approximately 76 usable numeric features for the tree-based models. The LSTM operates on the 8 raw temporal features (7 behavioral + focus score) organized into fixed-length sequences of 5 consecutive snapshots.

---

## Focus Score Computation

The dynamic focus score (`src/focus_score.py`) produces a continuous 0-100 metric for each snapshot. The computation proceeds as follows:

1. **Base score initialization** at 70 (neutral engagement assumed)
2. **Weighted signal contribution**: Each behavioral signal adds or subtracts from the base score proportional to configured weights:
   - Tab switches: -3.0 per switch
   - Idle time: -0.15 per second
   - Clicks: +0.3 per click
   - Mouse movement: +0.005 per pixel
   - Replay count: -1.5 per replay
   - Skip count: -2.5 per skip
   - Playback speed deviation from 1.0: -10.0 per unit deviation
3. **Per-student baseline normalization**: When a student baseline exists, signals are expressed as deviations from the student's personal mean before weight application
4. **Temporal decay**: Within a session, older snapshot contributions decay by a factor of 0.85 per snapshot interval
5. **Rolling smoothing**: A 3-snapshot rolling average stabilizes the score against momentary fluctuations
6. **Clamping**: Final score is bounded to [0, 100]

---

## Classification Models

Three classifiers are trained and evaluated on the engineered dataset with a 80/20 stratified train-test split and 5-fold stratified cross-validation:

### Random Forest

| Metric | Value |
|---|---|
| Test accuracy | 95.84% |
| Weighted F1 score | 0.9584 |
| Cross-validation accuracy | 95.72% (std: 0.0017) |

Configuration: 200 estimators, max depth 20, balanced class weights. Trained on 76 engineered features.

Per-class F1 scores: focused (0.983), confused (0.983), bored (0.940), distracted (0.927).

### XGBoost

| Metric | Value |
|---|---|
| Test accuracy | 96.60% |
| Weighted F1 score | 0.9660 |
| Cross-validation accuracy | 96.61% (std: 0.002) |

Configuration: 200 estimators, max depth 8, learning rate 0.1, subsample 0.8. Best-performing model, used as the primary classifier in the inference pipeline.

Per-class F1 scores: focused (0.988), confused (0.988), bored (0.949), distracted (0.938).

### LSTM (Long Short-Term Memory)

| Metric | Value |
|---|---|
| Test accuracy | 91.69% |
| Weighted F1 score | 0.9166 |

Configuration: Sequence length 5, 128 hidden units, dropout 0.3, 64-unit dense layer, trained for up to 50 epochs with early stopping (patience 5). Operates on raw 8-feature sequences without engineered features, capturing temporal dependencies through its recurrent architecture.

Per-class F1 scores: focused (0.981), confused (0.959), bored (0.879), distracted (0.847).

All models produce confusion matrices, ROC curves (one-vs-rest), and feature importance plots saved to `outputs/plots/`.

---

## Per-Student Model Personalization

The per-student retraining module (`src/student_model.py`) implements progressive model personalization:

1. **Trigger conditions**: A student must accumulate at least 300 snapshots spanning at least 3 distinct cognitive states with a minimum of 15 samples per state
2. **Feature construction**: Raw snapshot features are augmented with rolling statistics and z-score deviations computed from that individual student's data distribution
3. **Training**: A dedicated Random Forest classifier is trained on the student's own behavioral data
4. **Persistence**: The trained model is serialized to `models_saved/students/{student_id}.joblib`
5. **Incremental retraining**: After the initial training, the model retrains every 50 new snapshots to incorporate evolving behavioral patterns
6. **Fallback**: If the personal model does not exist or lacks sufficient class diversity, the system falls back to the global XGBoost classifier

This approach ensures that students whose behavioral patterns deviate significantly from the population average still receive accurate cognitive state classifications calibrated to their individual norms.

---

## Adaptive Response Engine

The adaptive response module (`src/adaptive_response.py`) implements a rule-based intervention system that maps detected cognitive states and focus scores to pedagogical actions:

| Detected Condition | Trigger Criteria | Intervention |
|---|---|---|
| Confused | Focus score < 40, replay count > 3 | Display contextual hint for the current section |
| Bored | Focus score < 35, skip count > 3 | Increase content difficulty or suggest advanced material |
| Distracted | Focus score < 30, tab switches > 5 | Send refocusing notification |
| Critical | Focus score < 20 (any state) | Alert the instructor for direct assistance |

The engine maintains a rolling history of the last 10 states per student to detect sustained patterns (such as three consecutive distracted snapshots) that warrant escalated intervention even if individual snapshots fall slightly above threshold values. Interventions are prioritized by severity so that critical alerts take precedence over routine nudges.

---

## Backend Infrastructure

The Flask backend (`backend/app.py`) serves as the central hub connecting the extension, database, ML models, and dashboard. It exposes 11 REST API endpoints:

| Endpoint | Method | Function |
|---|---|---|
| `/api/register` | POST | Create student account with hashed credentials |
| `/api/login` | POST | Authenticate student and return student ID |
| `/api/snapshot` | POST | Ingest behavioral snapshot, run ML inference, store results |
| `/api/session/end` | POST | Close active session, compute session-level aggregates |
| `/api/student/<id>/history` | GET | Retrieve session history for a student |
| `/api/student/<id>/snapshots` | GET | Retrieve all snapshots for a student |
| `/api/student/<id>/baseline` | GET | Retrieve computed behavioral baseline |
| `/api/student/<id>/profile` | GET | Retrieve learning style profile and personalization weights |
| `/api/students` | GET | List all registered students with summary statistics |
| `/api/export` | GET | Export all snapshot data as CSV |
| `/api/health` | GET | System health check |

The snapshot ingestion endpoint performs real-time inference: it computes the focus score, runs the XGBoost classifier (or per-student model if available), checks per-student retraining eligibility, updates the student's behavioral baseline, and returns the classification result to the extension within the same request cycle.

### Database Layer

The database adapter (`backend/db.py`) supports dual-mode operation:

- **Local development**: SQLite database at `backend/focus_monitor.db`
- **Cloud deployment**: PostgreSQL on Neon (Singapore region) via the `DATABASE_URL` environment variable

The adapter auto-detects the mode from environment variables and provides a unified interface. Five tables persist the application state: `snapshots` (behavioral telemetry), `sessions` (session-level aggregates), `student_baselines` (personal behavioral means and standard deviations), `users` (authentication credentials), and `student_profiles` (learning style, personalized focus weights, and tolerance thresholds).

A one-time migration utility (`scripts/migrate_to_postgres.py`, `backend/db.py:migrate_sqlite_to_postgres`) transfers all local SQLite data to the cloud PostgreSQL instance.

---

## Analytics Dashboard

The Streamlit dashboard (`dashboard/app.py`) provides a password-protected, multi-page analytics interface for educators and system administrators. It connects to the same database (PostgreSQL in cloud mode) and renders seven distinct pages:

1. **Overview**: Aggregate statistics across all students — total students, total snapshots, average focus score, most common cognitive state, and summed behavioral time metrics (total paused, away, and idle time). Includes a focus score distribution histogram and state breakdown pie chart.

2. **All Students**: Sortable and filterable student roster displaying per-student averages for focus score, snapshot counts, dominant cognitive state, detected learning style, and behavioral averages (idle, paused, away time per snapshot). Each student card includes a mini focus progress bar.

3. **Student Deep Dive**: Detailed analytics for an individual student, including focus score time series, state transition timeline, behavioral radar chart (comparing the student's behavioral profile to the population average across tab switches, idle time, paused time, away time, clicks, mouse movement, replays, and skips), session-by-session breakdown, and learning style analysis.

4. **Live Monitor**: Auto-refreshing view (configurable interval) showing currently active students with their latest focus scores, predicted states, and real-time behavioral readings (idle, paused, away time from the most recent snapshot). Includes focus score sparklines for recent activity.

5. **Personal Models**: Status overview of per-student model training — which students have personalized models, training sample counts, class distributions, and cross-validation accuracy of personal models versus the global classifier.

6. **Model Performance**: Side-by-side comparison of all three global classifiers with accuracy, F1 scores, confusion matrices, ROC curves, and feature importance rankings loaded from `outputs/metrics.json` and `outputs/plots/`.

7. **Dataset Explorer**: Interactive exploration of the training dataset with summary statistics, correlation heatmaps (including paused time, away time, and all behavioral features), and filterable data tables.

The dashboard uses Material Design-inspired styling with a light theme, white cards, green accent colors, and the Roboto font family.

---

## Deployment

The system is deployed across three platforms:

| Component | Platform | Details |
|---|---|---|
| Dashboard | Streamlit Community Cloud | Auto-deploys from GitHub repository, reads `DATABASE_URL` from Streamlit secrets |
| Database | Neon PostgreSQL | Serverless PostgreSQL in Singapore region (ap-southeast-1), connection pooling enabled |
| Extension | Local Chrome | Loaded as unpacked extension in developer mode, communicates with localhost backend |
| Backend | Local Flask server | Runs on `localhost:5000`, handles ML inference and snapshot ingestion |

The Streamlit Cloud deployment reads database credentials from `.streamlit/secrets.toml` (excluded from version control via `.gitignore`). The backend can operate against either local SQLite or cloud PostgreSQL depending on the `DATABASE_URL` environment variable.

---

## Project Structure

```
PR_Project/
├── config/
│   └── config.yaml                    # Central configuration for all parameters
├── data/
│   ├── student_focus_dataset.csv      # Raw generated dataset (40,000 rows, 13 columns)
│   └── student_focus_dataset_engineered.csv  # Engineered dataset (40,000 rows, 81 columns)
├── extension/                         # Chrome Extension (Manifest V3)
│   ├── manifest.json                  # Extension configuration and permissions
│   ├── popup.html                     # Popup UI layout
│   ├── css/popup.css                  # Popup styling
│   ├── js/background.js              # Session management, idle/away tracking, snapshots
│   ├── js/content.js                 # Click, mouse, video event capture
│   ├── js/popup.js                   # Popup logic and live stats display
│   └── icons/                        # Extension icons (16, 48, 128px)
├── backend/
│   ├── app.py                        # Flask API server with ML inference
│   └── db.py                         # Database adapter (SQLite / PostgreSQL)
├── src/
│   ├── data_generation.py            # Synthetic dataset generator with behavioral rules
│   ├── data_validation.py            # Six-point statistical validation suite
│   ├── focus_score.py                # Dynamic focus score computation (0-100)
│   ├── feature_engineering.py        # 76 engineered features from 7 raw signals
│   ├── adaptive_response.py          # Rule-based intervention engine
│   ├── evaluation.py                 # Model evaluation and comparison pipeline
│   ├── student_model.py              # Per-student model training and retraining
│   ├── utils/                        # Shared utility functions
│   └── models/
│       ├── random_forest.py          # Random Forest training script
│       ├── xgboost_model.py          # XGBoost training script (primary model)
│       └── lstm_model.py             # LSTM sequential model training script
├── models_saved/
│   ├── random_forest.joblib          # Serialized Random Forest
│   ├── xgboost.joblib                # Serialized XGBoost
│   ├── lstm_model.keras              # Serialized LSTM
│   ├── label_encoder.joblib          # Label encoder for state classes
│   ├── lstm_scaler.joblib            # Feature scaler for LSTM input
│   ├── lstm_label_encoder.joblib     # LSTM-specific label encoder
│   ├── lstm_history.json             # LSTM training history
│   └── students/                     # Per-student personalized models
├── outputs/
│   ├── metrics.json                  # Consolidated model performance metrics
│   ├── plots/                        # Confusion matrices, ROC curves, feature importance
│   └── reports/                      # Per-model detailed evaluation reports
├── dashboard/
│   └── app.py                        # Streamlit multi-page analytics dashboard
├── notebooks/
│   └── 01_complete_pipeline.ipynb    # End-to-end pipeline walkthrough notebook
├── scripts/
│   └── migrate_to_postgres.py        # SQLite to PostgreSQL migration utility
├── tests/
│   └── test_student_model.py         # Unit tests for per-student model module
├── .streamlit/
│   ├── config.toml                   # Streamlit theme configuration
│   └── secrets.toml                  # Database credentials (not in version control)
├── requirements.txt                  # Python dependencies
├── .gitignore
└── README.md
```

---

## Setup and Usage

### Prerequisites

- Python 3.11 or higher
- Google Chrome browser
- NVIDIA GPU with CUDA support (optional, for accelerated XGBoost and LSTM training)

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Generate Dataset and Train Models

```bash
py src/data_generation.py
py src/data_validation.py
py src/focus_score.py
py src/feature_engineering.py
py src/models/random_forest.py
py src/models/xgboost_model.py
py src/models/lstm_model.py
py src/evaluation.py
```

### 3. Start the Backend Server

```bash
py backend/app.py
```

The backend runs on `http://localhost:5000` and initializes the database tables on first launch.

### 4. Load the Chrome Extension

1. Navigate to `chrome://extensions/` in Google Chrome
2. Enable Developer Mode (toggle in the top-right corner)
3. Click "Load unpacked" and select the `extension/` directory
4. Click the extension icon in the toolbar, enter a Student ID, and begin a study session

### 5. Launch the Analytics Dashboard

```bash
streamlit run dashboard/app.py
```

The dashboard opens at `http://localhost:8501`. Enter the owner password to access the analytics pages.

### 6. Cloud Database Setup (Optional)

To use PostgreSQL instead of local SQLite:

1. Set the `DATABASE_URL` environment variable to your PostgreSQL connection string
2. Run the migration script: `py scripts/migrate_to_postgres.py`
3. For Streamlit Cloud deployment, add `DATABASE_URL` to `.streamlit/secrets.toml`

---

## Ethical Considerations

- **No browsing history is recorded.** The extension captures aggregate behavioral counts (e.g., "5 tab switches") but never records which websites or pages the student visited.
- **Consent-based activation.** Students manually start and stop each monitoring session. No background tracking occurs without explicit initiation.
- **Student-facing transparency.** Students see their own focus scores and behavioral statistics in real time through the extension popup, ensuring the monitoring is not opaque or hidden.
- **Data minimization.** Only behavioral signal counts and computed metrics are stored. No keystroke content, screen recordings, webcam feeds, or personally identifiable information beyond the student-chosen identifier is captured.
- **Local-first architecture.** The default configuration stores all data locally in SQLite. Cloud database usage is optional and requires explicit configuration.

---

## Technology Stack

| Layer | Technologies |
|---|---|
| Data capture | Chrome Extension (Manifest V3), JavaScript, Chrome APIs (tabs, idle, storage, alarms) |
| Backend API | Python 3.11, Flask, Flask-CORS |
| Database | SQLite (local), PostgreSQL via Neon (cloud), psycopg2 |
| Machine learning | scikit-learn (Random Forest), XGBoost, TensorFlow/Keras (LSTM) |
| Feature processing | NumPy, pandas, SciPy |
| Dashboard | Streamlit, Plotly, Matplotlib, Seaborn |
| Configuration | YAML (centralized parameter management) |
| Serialization | joblib (tree models), Keras native format (LSTM) |
| Deployment | Streamlit Community Cloud, Neon serverless PostgreSQL |

---

## Future Scope

- Integration of webcam-based gaze tracking to complement behavioral telemetry with physiological attention signals
- Federation with institutional Learning Management Systems (Moodle, Canvas) for automated grade correlation analysis
- Migration from rule-based adaptive responses to a reinforcement learning agent that optimizes intervention timing and type based on observed student response patterns
- Multi-modal data fusion combining behavioral, physiological, and contextual (time-of-day, session duration) features for improved classification robustness
- Mobile platform support through a companion application that extends monitoring to tablet-based learning environments
- Online model retraining pipeline that continuously updates the global classifier as the population of real student data grows, eventually replacing the synthetic pre-training dataset entirely
