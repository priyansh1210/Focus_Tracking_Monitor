/**
 * Popup Script — Student Focus Monitor
 * =====================================
 * Handles login/register, start/stop sessions, and live stats display.
 */

const BACKEND_URL = "http://localhost:5000/api";

const loginScreen = document.getElementById("login-screen");
const setupScreen = document.getElementById("setup-screen");
const sessionScreen = document.getElementById("session-screen");
const loginBtn = document.getElementById("login-btn");
const registerBtn = document.getElementById("register-btn");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const logoutBtn = document.getElementById("logout-btn");
const usernameInput = document.getElementById("username");
const passwordInput = document.getElementById("password");
const authError = document.getElementById("auth-error");
const currentSiteEl = document.getElementById("current-site");
const loginMode = document.getElementById("login-mode");
const registerMode = document.getElementById("register-mode");

let updateInterval = null;

// ---- Initialize ----

document.addEventListener("DOMContentLoaded", async () => {
  // Check if already logged in
  chrome.storage.local.get(["authUser"], async (result) => {
    if (result.authUser && result.authUser.student_id) {
      await showLoggedInState(result.authUser);
    }
  });
});

// ---- Mode Switching ----

document.getElementById("switch-to-register").addEventListener("click", () => {
  loginMode.style.display = "none";
  registerMode.style.display = "block";
  authError.style.display = "none";
});

document.getElementById("switch-to-login").addEventListener("click", () => {
  registerMode.style.display = "none";
  loginMode.style.display = "block";
  authError.style.display = "none";
});

// ---- Auth ----

async function doAuth(endpoint) {
  const username = usernameInput.value.trim();
  const password = passwordInput.value.trim();
  if (!username || !password) {
    showAuthError("Please enter username and password");
    return;
  }

  try {
    const res = await fetch(`${BACKEND_URL}/${endpoint}`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });
    const data = await res.json();

    if (!res.ok) {
      showAuthError(data.error || `${endpoint} failed`);
      return;
    }

    const authUser = { student_id: data.student_id, username: data.username };
    chrome.storage.local.set({ authUser });
    await showLoggedInState(authUser);
  } catch (err) {
    showAuthError("Cannot reach server. Is the backend running?");
  }
}

loginBtn.addEventListener("click", () => doAuth("login"));
registerBtn.addEventListener("click", () => doAuth("register"));

logoutBtn.addEventListener("click", () => {
  chrome.storage.local.remove(["authUser"]);
  showScreen("login");
  // Reset to login mode
  registerMode.style.display = "none";
  loginMode.style.display = "block";
});

function showAuthError(msg) {
  authError.textContent = msg;
  authError.style.display = "block";
  setTimeout(() => { authError.style.display = "none"; }, 4000);
}

// ---- Logged-in State ----

async function showLoggedInState(authUser) {
  document.getElementById("logged-in-user").textContent = authUser.username;

  // Show current website (full URL path, not just hostname)
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  if (tab?.url) {
    try {
      const url = new URL(tab.url);
      // Show full hostname + pathname
      currentSiteEl.textContent = url.hostname + url.pathname + url.search;
    } catch {
      currentSiteEl.textContent = "this page";
    }
  }

  // Check if session is already active
  chrome.runtime.sendMessage({ type: "GET_STATUS" }, (response) => {
    if (response && response.active) {
      showScreen("session");
      showSessionData(response);
      startLiveUpdates();
    } else {
      showScreen("setup");
    }
  });
}

// ---- Start Session ----

startBtn.addEventListener("click", async () => {
  // Get student_id from stored auth
  const result = await chrome.storage.local.get(["authUser"]);
  const studentId = result.authUser?.student_id;
  if (!studentId) {
    showScreen("login");
    return;
  }

  // Get current tab info — full URL
  const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  let website = "unknown";
  try {
    const url = new URL(tab.url);
    website = url.hostname + url.pathname + url.search;
  } catch {}

  chrome.runtime.sendMessage({
    type: "START_SESSION",
    studentId: studentId,
    website: website,
    tabId: tab?.id,
  }, (response) => {
    if (response && response.status === "started") {
      showScreen("session");
      showSessionData({ website, elapsed: 0 });
      startLiveUpdates();
    }
  });
});

// ---- Stop Session ----

stopBtn.addEventListener("click", () => {
  chrome.runtime.sendMessage({ type: "STOP_SESSION" }, () => {
    stopLiveUpdates();
    showScreen("setup");
  });
});

// ---- Snapshot History ----

const historyScreen = document.getElementById("history-screen");
const historyBtn = document.getElementById("history-btn");
const backBtn = document.getElementById("back-btn");

historyBtn.addEventListener("click", () => {
  showScreen("history");
  renderSnapshotHistory();
});

backBtn.addEventListener("click", () => {
  showScreen("session");
});

function renderSnapshotHistory() {
  chrome.runtime.sendMessage({ type: "GET_SNAPSHOT_HISTORY" }, (response) => {
    const list = document.getElementById("snapshot-list");
    const snapshots = response?.snapshots || [];

    document.getElementById("history-subtitle").textContent =
      `${snapshots.length} snapshot${snapshots.length !== 1 ? "s" : ""} recorded`;

    if (snapshots.length === 0) {
      list.innerHTML = '<div class="no-snapshots">No snapshots yet. First snapshot will be taken in 30 seconds.</div>';
      return;
    }

    list.innerHTML = snapshots.map((snap, i) => {
      const time = new Date(snap.timestamp).toLocaleTimeString();
      const state = snap.predicted_state || snap.state || "---";
      const score = snap.focus_score_result != null ? snap.focus_score_result : "---";

      const stateColors = {
        focused: "background:#166534;color:#86efac",
        distracted: "background:#9a3412;color:#fdba74",
        confused: "background:#1e40af;color:#93c5fd",
        bored: "background:#581c87;color:#d8b4fe",
      };
      const badgeStyle = stateColors[state] || "background:#334155;color:#94a3b8";

      const events = snap.events || [];
      const eventsHtml = events.length > 0
        ? events.map(e =>
            `<div class="event-item ${e.type}">
              <span class="event-time">${e.timeStr}</span>
              <span class="event-detail">${e.detail}</span>
            </div>`
          ).join("")
        : '<div style="font-size:11px;color:#64748b;padding:4px 0;">No events in this window</div>';

      return `
        <div class="snapshot-card">
          <div class="snapshot-header" data-toggle="${i}">
            <div>
              <span class="snapshot-num">Snapshot #${snap.snapshot_index + 1}</span>
              <span class="snapshot-time"> &mdash; ${time}</span>
            </div>
            <span class="snapshot-state" style="${badgeStyle}">${score} / ${state}</span>
          </div>
          <div class="snapshot-metrics">
            <div class="metric"><span class="metric-label">Tab Switches</span><span class="metric-value">${snap.tab_switch}</span></div>
            <div class="metric"><span class="metric-label">Idle Time</span><span class="metric-value">${snap.idle_time}s</span></div>
            <div class="metric"><span class="metric-label">Paused</span><span class="metric-value">${snap.paused_time || 0}s</span></div>
            <div class="metric"><span class="metric-label">Clicks</span><span class="metric-value">${snap.clicks}</span></div>
            <div class="metric"><span class="metric-label">Mouse</span><span class="metric-value">${snap.mouse_movement}px</span></div>
            <div class="metric"><span class="metric-label">Replays</span><span class="metric-value">${snap.replay_count}</span></div>
            <div class="metric"><span class="metric-label">Skips</span><span class="metric-value">${snap.skip_count}</span></div>
            <div class="metric"><span class="metric-label">Speed</span><span class="metric-value">${snap.playback_speed}x</span></div>
            <div class="metric"><span class="metric-label">Elapsed</span><span class="metric-value">${formatTime(snap.elapsed_seconds)}</span></div>
          </div>
          <div class="toggle-events" data-toggle="${i}">▼ Event Timeline (${events.length})</div>
          <div class="snapshot-events" id="events-${i}">
            <div class="snapshot-events-title">Activity Log</div>
            ${eventsHtml}
          </div>
        </div>`;
    }).reverse().join(""); // newest first

    // Attach click listeners for toggling event timelines
    list.querySelectorAll("[data-toggle]").forEach(el => {
      el.addEventListener("click", () => {
        const idx = el.getAttribute("data-toggle");
        const eventsEl = document.getElementById(`events-${idx}`);
        if (eventsEl) eventsEl.classList.toggle("open");
      });
    });
  });
}

// ---- Screen Management ----

function showScreen(name) {
  loginScreen.style.display = name === "login" ? "block" : "none";
  setupScreen.style.display = name === "setup" ? "block" : "none";
  sessionScreen.style.display = name === "session" ? "block" : "none";
  historyScreen.style.display = name === "history" ? "block" : "none";
}

function showSessionData(data) {
  if (data.website) {
    const el = document.getElementById("session-website");
    el.textContent = data.website;
    el.title = data.website; // tooltip for full URL
  }
}

// ---- Live Updates ----

function startLiveUpdates() {
  updateStats();
  updateInterval = setInterval(updateStats, 1000);
}

function stopLiveUpdates() {
  if (updateInterval) {
    clearInterval(updateInterval);
    updateInterval = null;
  }
}

function updateStats() {
  chrome.runtime.sendMessage({ type: "GET_STATUS" }, (response) => {
    if (!response || !response.active) {
      stopLiveUpdates();
      showScreen("setup");
      return;
    }

    // Timer
    document.getElementById("session-timer").textContent = formatTime(response.elapsed);

    // Stats
    document.getElementById("stat-tabs").textContent = response.tabSwitches;
    document.getElementById("stat-idle").textContent = response.idleTime + "s";
    document.getElementById("stat-paused").textContent = (response.pausedTime || 0) + "s";
    document.getElementById("stat-away").textContent = response.awayTime + "s";
    document.getElementById("stat-clicks").textContent = response.clicks;
    document.getElementById("stat-replays").textContent = response.replayCount;
    document.getElementById("stat-skips").textContent = response.skipCount;
    document.getElementById("stat-speed").textContent = response.playbackSpeed + "x";
    document.getElementById("snapshot-count").textContent = response.snapshotCount;
  });

  // Get latest prediction from backend
  chrome.storage.local.get(["lastPrediction"], (result) => {
    if (result.lastPrediction) {
      updateFocusDisplay(result.lastPrediction);
    }
  });

  // Update learning style badge
  chrome.storage.local.get(["lastPrediction"], (result) => {
    const styleEl = document.getElementById("learning-style");
    if (styleEl && result.lastPrediction?.learning_style &&
        result.lastPrediction.learning_style !== "unknown") {
      styleEl.textContent = result.lastPrediction.learning_style;
      styleEl.style.display = "inline-block";
    }
  });
}

function updateFocusDisplay(prediction) {
  const scoreEl = document.getElementById("focus-score");
  const ringEl = document.getElementById("score-ring-fill");
  const badgeEl = document.getElementById("state-badge");
  const msgEl = document.getElementById("state-message");

  const score = Math.round(prediction.focus_score);
  scoreEl.textContent = score;

  // Update ring (circumference = 2 * pi * 52 = 326.7)
  const offset = 326.7 * (1 - score / 100);
  ringEl.style.strokeDashoffset = offset;

  // Color based on score
  let color;
  if (score >= 70) color = "#22c55e";
  else if (score >= 50) color = "#f59e0b";
  else if (score >= 30) color = "#f97316";
  else color = "#ef4444";
  ringEl.style.stroke = color;

  // State badge
  const state = prediction.state || "unknown";
  badgeEl.textContent = state.charAt(0).toUpperCase() + state.slice(1);
  badgeEl.className = `state-badge state-${state}`;

  // Message
  if (prediction.message) {
    msgEl.textContent = prediction.message;
  }
}

function formatTime(seconds) {
  const h = Math.floor(seconds / 3600);
  const m = Math.floor((seconds % 3600) / 60);
  const s = seconds % 60;
  return [h, m, s].map(v => String(v).padStart(2, "0")).join(":");
}
