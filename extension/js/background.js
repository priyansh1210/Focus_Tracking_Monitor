/**
 * Background Service Worker — Student Focus Monitor
 * ==================================================
 * Tracks: tab switches, idle time, active time per session.
 * Sends snapshots to Flask backend every 30 seconds.
 * Runs only when a study session is active.
 */

// ---- State ----
let session = {
  active: false,
  studentId: null,
  sessionId: null,
  startTime: null,
  website: null,
  tabSwitchCount: 0,
  lastActiveTab: null,
  tabAwayStart: null,
  totalAwayTime: 0,       // seconds away from study tab
  idleStart: null,
  totalIdleTime: 0,       // seconds idle (no input)
  lastInteraction: null,   // timestamp of last click/mouse on study tab
  clicks: 0,
  mouseDistance: 0,
  replayCount: 0,
  skipCount: 0,
  playbackSpeed: 1.0,
  snapshots: [],
  studyTabId: null,
  eventLog: [],           // detailed event log for current snapshot window
  snapshotWindowStart: null, // when current 30s window started
  videoPaused: false,     // true when student paused the video (may be taking notes)
  videoPausedStart: null, // timestamp when video was paused
  totalPausedTime: 0,     // seconds video has been paused
};

const BACKEND_URL = "http://localhost:5000/api";
const SNAPSHOT_INTERVAL = 30; // seconds

// ---- Helper ----

function timeStr(ts) {
  return new Date(ts).toLocaleTimeString();
}

function logEvent(type, detail) {
  session.eventLog.push({
    time: Date.now(),
    timeStr: timeStr(Date.now()),
    type: type,
    detail: detail || "",
  });
}

// ---- Session Management ----

chrome.runtime.onMessage.addListener((msg, sender, sendResponse) => {
  if (msg.type === "START_SESSION") {
    startSession(msg.studentId, msg.website, msg.tabId || sender.tab?.id);
    sendResponse({ status: "started", sessionId: session.sessionId });
  }
  else if (msg.type === "STOP_SESSION") {
    stopSession();
    sendResponse({ status: "stopped" });
  }
  else if (msg.type === "GET_STATUS") {
    sendResponse(getSessionStatus());
  }
  else if (msg.type === "GET_SNAPSHOT_HISTORY") {
    sendResponse({ snapshots: session.snapshots });
  }
  else if (msg.type === "CONTENT_EVENT") {
    handleContentEvent(msg.event);
    sendResponse({ status: "ok" });
  }
  return true; // async response
});

function startSession(studentId, website, tabId) {
  const now = Date.now();
  session = {
    active: true,
    studentId: studentId || "anonymous",
    sessionId: `S_${now}`,
    startTime: now,
    website: website || "unknown",
    tabSwitchCount: 0,
    lastActiveTab: tabId,
    tabAwayStart: null,
    totalAwayTime: 0,
    idleStart: null,
    totalIdleTime: 0,
    lastInteraction: now,
    clicks: 0,
    mouseDistance: 0,
    replayCount: 0,
    skipCount: 0,
    playbackSpeed: 1.0,
    snapshots: [],
    studyTabId: tabId,
    eventLog: [],
    snapshotWindowStart: now,
    videoPaused: false,
    videoPausedStart: null,
    totalPausedTime: 0,
  };

  // Save session state
  chrome.storage.local.set({ session });

  // Set up snapshot alarm
  chrome.alarms.create("snapshot", { periodInMinutes: SNAPSHOT_INTERVAL / 60 });

  // Set up idle detection (15 seconds threshold)
  chrome.idle.setDetectionInterval(15);

  console.log(`[Focus Monitor] Session started: ${session.sessionId} on ${website}`);
}

function stopSession() {
  if (!session.active) return;

  // Take final snapshot
  takeSnapshot();

  // Send all data to backend
  sendSessionToBackend();

  session.active = false;
  chrome.storage.local.set({ session });
  chrome.alarms.clear("snapshot");

  console.log(`[Focus Monitor] Session stopped. Total snapshots: ${session.snapshots.length}`);
}

// ---- Tab Switch Detection ----

chrome.tabs.onActivated.addListener((activeInfo) => {
  if (!session.active) return;

  const now = Date.now();

  if (activeInfo.tabId !== session.studyTabId) {
    // Every tab switch away from study tab counts
    session.tabSwitchCount++;
    logEvent("tab_away", `Switched away from study tab`);
    if (!session.tabAwayStart) {
      session.tabAwayStart = now;
    }
  } else {
    // Returned TO study tab
    if (session.tabAwayStart) {
      const awayDuration = (now - session.tabAwayStart) / 1000;
      session.totalAwayTime += awayDuration;
      logEvent("tab_return", `Returned after ${Math.round(awayDuration)}s away`);
      session.tabAwayStart = null;
    }
  }

  chrome.storage.local.set({ session });
});

// Also detect window focus changes
chrome.windows.onFocusChanged.addListener((windowId) => {
  if (!session.active) return;

  const now = Date.now();
  if (windowId === chrome.windows.WINDOW_ID_NONE) {
    // Browser lost focus entirely — always counts as away
    if (!session.tabAwayStart) {
      session.tabAwayStart = now;
      session.tabSwitchCount++;
      logEvent("window_away", "Browser lost focus");
    }
  } else {
    // Browser regained focus — close the away interval
    if (session.tabAwayStart) {
      const awayDuration = (now - session.tabAwayStart) / 1000;
      session.totalAwayTime += awayDuration;
      logEvent("window_return", `Browser regained focus after ${Math.round(awayDuration)}s`);
      session.tabAwayStart = null;
    }
    // Restart the idle clock from the moment of return so idle doesn't
    // keep counting the time the user was in another app.
    session.lastInteraction = now;
  }
  chrome.storage.local.set({ session });
});

// ---- Idle Detection ----

chrome.idle.onStateChanged.addListener((state) => {
  if (!session.active) return;

  const now = Date.now();
  if (state === "idle" || state === "locked") {
    if (!session.idleStart) {
      session.idleStart = now;
      logEvent("idle_start", `Student went ${state}`);
    }
  } else if (state === "active") {
    if (session.idleStart) {
      const idleDuration = (now - session.idleStart) / 1000;
      session.totalIdleTime += idleDuration;
      logEvent("idle_end", `Active again after ${Math.round(idleDuration)}s idle`);
      session.idleStart = null;
    }
    // Also handle tab away if window regains focus
    if (session.tabAwayStart) {
      const awayDuration = (now - session.tabAwayStart) / 1000;
      session.totalAwayTime += awayDuration;
      session.tabAwayStart = null;
    }
  }

  chrome.storage.local.set({ session });
});

// ---- Content Script Events ----

function handleContentEvent(event) {
  if (!session.active) return;

  // Track last interaction on study tab for real idle time
  if (event.type === "click" || event.type === "mousemove" || event.type === "scroll") {
    session.lastInteraction = Date.now();
  }

  switch (event.type) {
    case "click":
      session.clicks++;
      break;
    case "mousemove":
      session.mouseDistance += event.distance || 0;
      break;
    case "scroll":
      break;
    case "video_replay":
      session.replayCount++;
      logEvent("video_replay", "Student replayed video section");
      break;
    case "video_skip":
      session.skipCount++;
      logEvent("video_skip", "Student skipped forward");
      break;
    case "video_pause":
      session.videoPaused = true;
      session.videoPausedStart = Date.now();
      logEvent("video_pause", "Student paused video");
      break;
    case "video_play":
      if (session.videoPaused && session.videoPausedStart) {
        const pausedDuration = (Date.now() - session.videoPausedStart) / 1000;
        session.totalPausedTime += pausedDuration;
        logEvent("video_play", `Student resumed video (paused ${Math.round(pausedDuration)}s)`);
      } else {
        logEvent("video_play", "Student resumed video");
      }
      session.videoPaused = false;
      session.videoPausedStart = null;
      break;
    case "video_speed":
      const oldSpeed = session.playbackSpeed;
      session.playbackSpeed = event.speed || 1.0;
      logEvent("video_speed", `Speed changed: ${oldSpeed}x -> ${session.playbackSpeed}x`);
      break;
  }

  chrome.storage.local.set({ session });
}

// ---- Snapshot & Backend ----

chrome.alarms.onAlarm.addListener((alarm) => {
  if (alarm.name === "snapshot" && session.active) {
    takeSnapshot();
  }
});

function takeSnapshot() {
  if (!session.active) return;

  const now = Date.now();
  const elapsedSec = (now - session.startTime) / 1000;
  const windowDuration = (now - (session.snapshotWindowStart || session.startTime)) / 1000;

  // Calculate away time in this window (time on other tabs)
  let currentAwayTime = session.totalAwayTime;
  if (session.tabAwayStart) {
    currentAwayTime += (now - session.tabAwayStart) / 1000;
  }

  // Calculate real idle time: time with no clicks/mouse/scroll on study tab
  // This is more accurate than chrome.idle which doesn't trigger during video playback
  let interactionIdleTime = 0;
  if (session.lastInteraction) {
    // Time since last interaction on the study tab
    interactionIdleTime = (now - session.lastInteraction) / 1000;
  }
  // Also include chrome.idle time
  let chromeIdleTime = session.totalIdleTime;
  if (session.idleStart) {
    chromeIdleTime += (now - session.idleStart) / 1000;
  }
  // Use the higher of the two (interaction-based is more reliable)
  let realIdleTime = Math.max(interactionIdleTime, chromeIdleTime);

  // Calculate paused time in this window
  let currentPausedTime = session.totalPausedTime;
  if (session.videoPaused && session.videoPausedStart) {
    currentPausedTime += (now - session.videoPausedStart) / 1000;
  }

  // Subtract paused time and away time from idle
  // (paused is its own category, away is its own category)
  let idleOnStudyTab = Math.max(0, realIdleTime - currentAwayTime - currentPausedTime);

  // Away time ratio: what fraction of this window was spent away
  let awayRatio = windowDuration > 0 ? currentAwayTime / windowDuration : 0;

  const snapshot = {
    student_id: session.studentId,
    session_id: session.sessionId,
    timestamp: new Date().toISOString(),
    snapshot_index: session.snapshots.length,
    tab_switch: session.tabSwitchCount,
    idle_time: Math.round(idleOnStudyTab * 10) / 10,
    paused_time: Math.round(currentPausedTime * 10) / 10,
    away_time: Math.round(currentAwayTime * 10) / 10,
    away_ratio: Math.round(awayRatio * 100) / 100,
    clicks: session.clicks,
    mouse_movement: Math.round(session.mouseDistance * 10) / 10,
    replay_count: session.replayCount,
    skip_count: session.skipCount,
    playback_speed: session.playbackSpeed,
    website: session.website,
    elapsed_seconds: Math.round(elapsedSec),
    window_duration: Math.round(windowDuration),
    events: [...session.eventLog],  // attach detailed events to snapshot
  };

  session.snapshots.push(snapshot);
  chrome.storage.local.set({ session });

  // Send to backend
  sendSnapshotToBackend(snapshot);

  // Reset counters for next interval
  session.tabSwitchCount = 0;
  session.totalAwayTime = 0;
  session.totalIdleTime = 0;
  session.clicks = 0;
  session.mouseDistance = 0;
  session.replayCount = 0;
  session.skipCount = 0;
  session.idleStart = null;
  session.tabAwayStart = null;
  session.totalPausedTime = 0;
  if (session.videoPaused) {
    session.videoPausedStart = Date.now(); // reset window start for ongoing pause
  }
  session.eventLog = [];
  session.lastInteraction = Date.now();
  session.snapshotWindowStart = Date.now();

  console.log(`[Focus Monitor] Snapshot #${snapshot.snapshot_index} taken`);
}

async function sendSnapshotToBackend(snapshot) {
  try {
    const response = await fetch(`${BACKEND_URL}/snapshot`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(snapshot),
    });
    const data = await response.json();

    // Store the prediction result and attach to snapshot
    if (data.focus_score !== undefined) {
      // Update the last snapshot with prediction
      const lastIdx = session.snapshots.length - 1;
      if (lastIdx >= 0) {
        session.snapshots[lastIdx].focus_score_result = data.focus_score;
        session.snapshots[lastIdx].predicted_state = data.state;
        session.snapshots[lastIdx].message = data.message;
      }

      chrome.storage.local.set({
        session,
        lastPrediction: {
          focus_score: data.focus_score,
          state: data.state,
          message: data.message,
          learning_style: data.learning_style || "unknown",
          timestamp: Date.now(),
        }
      });
    }
  } catch (err) {
    console.log("[Focus Monitor] Backend not reachable, storing locally.", err.message);
  }
}

async function sendSessionToBackend() {
  try {
    await fetch(`${BACKEND_URL}/session/end`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        student_id: session.studentId,
        session_id: session.sessionId,
        website: session.website,
        start_time: new Date(session.startTime).toISOString(),
        end_time: new Date().toISOString(),
        total_snapshots: session.snapshots.length,
        snapshots: session.snapshots,
      }),
    });
  } catch (err) {
    // Store locally if backend is down
    chrome.storage.local.get(["pendingSessions"], (result) => {
      const pending = result.pendingSessions || [];
      pending.push(session.snapshots);
      chrome.storage.local.set({ pendingSessions: pending });
    });
  }
}

// ---- Status ----

function getSessionStatus() {
  if (!session.active) {
    return { active: false };
  }

  const now = Date.now();
  const elapsed = Math.round((now - session.startTime) / 1000);

  // Calculate paused time for display
  let displayPausedTime = Math.round(session.totalPausedTime);
  if (session.videoPaused && session.videoPausedStart) {
    displayPausedTime += Math.round((now - session.videoPausedStart) / 1000);
  }

  // Calculate away time for display
  let displayAwayTime = Math.round(session.totalAwayTime);
  if (session.tabAwayStart) {
    displayAwayTime += Math.round((now - session.tabAwayStart) / 1000);
  }

  // Calculate real idle time for display — mirror takeSnapshot's logic so the
  // popup never disagrees with what the snapshot is about to record.
  let interactionIdleTime = 0;
  if (session.lastInteraction) {
    interactionIdleTime = (now - session.lastInteraction) / 1000;
  }
  let chromeIdleTime = session.totalIdleTime;
  if (session.idleStart) {
    chromeIdleTime += (now - session.idleStart) / 1000;
  }
  const realIdleTime = Math.max(interactionIdleTime, chromeIdleTime);
  const displayIdleTime = Math.max(
    0,
    Math.round(realIdleTime - displayPausedTime - displayAwayTime)
  );

  return {
    active: true,
    studentId: session.studentId,
    sessionId: session.sessionId,
    website: session.website,
    elapsed: elapsed,
    tabSwitches: session.tabSwitchCount,
    idleTime: displayIdleTime,
    awayTime: displayAwayTime,
    pausedTime: displayPausedTime,
    clicks: session.clicks,
    mouseDistance: Math.round(session.mouseDistance),
    replayCount: session.replayCount,
    skipCount: session.skipCount,
    playbackSpeed: session.playbackSpeed,
    snapshotCount: session.snapshots.length,
  };
}

// Restore session on service worker restart
chrome.storage.local.get(["session"], (result) => {
  if (result.session && result.session.active) {
    Object.assign(session, result.session);
    chrome.alarms.create("snapshot", { periodInMinutes: SNAPSHOT_INTERVAL / 60 });
    console.log("[Focus Monitor] Session restored from storage");
  }
});
