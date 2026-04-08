# Student Focus Monitor — Intelligent Online Learning Analytics

An end-to-end intelligent system that monitors and analyzes student focus during online learning in real-time. It captures behavioral signals through a Chrome extension, maps them to cognitive states using machine learning, and provides adaptive responses to improve learning outcomes.

## Problem Statement

Online learning platforms lack real-time insight into student cognitive states. Students may be **distracted, confused, or bored** without the platform knowing — leading to poor learning outcomes. Traditional systems only track completion rates, not engagement quality.

## Our Solution

This system captures **low-level behavioral signals** (tab switches, idle time, clicks, mouse movement, video replays/skips) and maps them to **high-level cognitive states** (focused, distracted, confused, bored) using trained ML models.

## Key Novelties

1. **Behavioral-to-Cognitive Mapping**: Unlike simple "active/inactive" trackers, we classify *why* a student lost focus — confusion vs boredom vs distraction require different interventions.

2. **Per-Student Personalization**: Each student has a unique behavioral baseline. Switching tabs 10 times might be normal for a multitasker but alarming for a focused learner. The system adapts to individual patterns.

3. **Temporal Pattern Recognition**: We analyze **sequences** of behavior (e.g., idle → replay → pause = confusion buildup), not just single snapshots. Sequential patterns reveal deeper cognitive conditions.

4. **Dynamic Focus Score**: A continuously computed 0-100 score using weighted, time-sensitive factors with exponential decay — more responsive and realistic than static metrics.

5. **Real Data Pipeline**: Chrome extension captures real behavioral data → Flask backend stores and processes it → ML model classifies in real-time → Adaptive responses are triggered.

6. **Cold-Start with Synthetic, Warm-Up with Real**: The model is pre-trained on 40,000 synthetic rows with realistic behavioral rules, then improves as real student data flows in.

## Architecture

```
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Chrome Extension   │────→│   Flask Backend   │────→│   ML Models     │
│  (Behavioral Signals)│     │  (SQLite + API)   │     │  (RF/XGB/LSTM)  │
└─────────────────────┘     └──────────────────┘     └─────────────────┘
         │                          │                        │
         │                          ▼                        │
         │                  ┌──────────────────┐            │
         │                  │   Student DB      │            │
         │                  │  (Baselines +     │            │
         │                  │   Snapshots)      │            │
         │                  └──────────────────┘            │
         │                          │                        │
         ▼                          ▼                        ▼
┌─────────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Extension Popup    │←───│ Adaptive Response │←───│  Focus Score +   │
│  (Focus Score, State │     │    Engine         │     │  State Predict   │
│   Live Stats)        │     └──────────────────┘     └─────────────────┘
└─────────────────────┘
```

## Model Performance

| Model | Accuracy | F1 (Weighted) | CV Accuracy |
|-------|----------|---------------|-------------|
| Random Forest | 95.84% | 0.9584 | 95.72% |
| **XGBoost** | **96.60%** | **0.9660** | **96.61%** |
| LSTM | 91.69% | 0.9166 | — |

- XGBoost achieves the best performance with GPU-accelerated training
- LSTM uses raw sequences (8 features) vs RF/XGBoost (76 engineered features)
- All models validated with 5-fold stratified cross-validation

## Dataset

- **40,000 rows** — synthetic, balanced (10,000 per class)
- **200 students**, each with unique behavioral baselines
- **4,504 sessions** with temporal coherence
- **7 behavioral features** + **68 engineered features**
- Validated: no missing values, realistic ranges, statistical discriminability (ANOVA p < 0.05 for all features)

## Project Structure

```
PR_Project/
├── config/
│   └── config.yaml              # All parameters (dataset, models, thresholds)
├── data/
│   └── student_focus_dataset.csv # Generated dataset
├── extension/                    # Chrome Extension (Manifest V3)
│   ├── manifest.json
│   ├── popup.html
│   ├── css/popup.css
│   ├── js/background.js          # Tab switch + idle detection
│   ├── js/content.js              # Click + mouse + video tracking
│   └── js/popup.js                # UI logic
├── backend/
│   └── app.py                    # Flask API + SQLite + model inference
├── src/
│   ├── data_generation.py        # Synthetic dataset generator
│   ├── data_validation.py        # Statistical validation (6 checks)
│   ├── focus_score.py            # Dynamic focus score (0-100)
│   ├── feature_engineering.py    # 68 engineered features
│   ├── adaptive_response.py      # Rule-based intervention engine
│   ├── evaluation.py             # Full evaluation pipeline
│   └── models/
│       ├── random_forest.py
│       ├── xgboost_model.py
│       └── lstm_model.py
├── models_saved/                 # Trained model files
├── outputs/
│   ├── plots/                    # Confusion matrices, ROC curves, etc.
│   ├── reports/                  # Per-model JSON reports
│   └── metrics.json              # All metrics in one file
├── dashboard/
│   └── app.py                    # Streamlit interactive dashboard
├── notebooks/                    # Jupyter notebooks with explanations
├── requirements.txt
└── README.md
```

## Setup & Usage

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate Dataset & Train Models
```bash
py src/data_generation.py
py src/focus_score.py
py src/feature_engineering.py
py src/models/random_forest.py
py src/models/xgboost_model.py
py src/models/lstm_model.py
py src/evaluation.py
```

### 3. Start Backend
```bash
py backend/app.py
```

### 4. Load Chrome Extension
1. Open Chrome → `chrome://extensions/`
2. Enable "Developer mode"
3. Click "Load unpacked" → Select `extension/` folder
4. Click the extension icon, enter Student ID, start studying

### 5. Launch Dashboard
```bash
streamlit run dashboard/app.py
```

## Behavioral Signals Captured

| Signal | Method | What It Indicates |
|--------|--------|------------------|
| Tab Switches | `visibilitychange` API | Distraction, multitasking |
| Idle Time | No mouse/keyboard events | Thinking, confusion, or disengagement |
| Clicks | Click event count | Active engagement |
| Mouse Movement | Pixel distance traveled | Attention and interaction |
| Replay Count | Video seeked backward | Confusion, difficulty understanding |
| Skip Count | Video seeked forward | Boredom, already understood |
| Playback Speed | Video rate change | Rushing (bored) or slowing (confused) |

## Ethical Considerations

- **No browsing history** is tracked or stored
- Only **aggregate behavioral counts** are captured (e.g., "5 tab switches", not "visited facebook.com")
- Data is **student-facing** — students see their own focus stats
- All data stored **locally** (SQLite on localhost)
- **Consent-based**: Student manually starts each session

## Tech Stack

- **Python 3.11** — Core ML pipeline
- **scikit-learn, XGBoost, TensorFlow/Keras** — Model training
- **Flask** — Backend API
- **SQLite** — Lightweight database
- **Chrome Extension (Manifest V3)** — Behavioral data capture
- **Streamlit + Plotly** — Interactive dashboard
- **NVIDIA GTX 1650 (CUDA)** — GPU-accelerated training

## Future Improvements

- Retrain models on accumulated real data (online learning)
- Add webcam-based attention detection (eye tracking)
- Integrate with LMS platforms (Moodle, Canvas)
- Multi-language support for intervention messages
- Push notification API for mobile devices
