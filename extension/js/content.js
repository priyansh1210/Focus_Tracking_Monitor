/**
 * Content Script — Student Focus Monitor
 * =======================================
 * Injected into every page to capture:
 * - Click events
 * - Mouse movement (distance in pixels)
 * - Video interactions (replay, skip, speed change)
 *
 * Sends events to background service worker.
 */

// Track whether the extension context is still valid
let contextValid = true;

function safeSendMessage(msg) {
  if (!contextValid) return;
  try {
    if (!chrome.runtime?.id) { contextValid = false; return; }
    chrome.runtime.sendMessage(msg).catch((err) => {
      // Only kill the channel if the extension was actually reloaded/uninstalled.
      // Transient "Receiving end does not exist" while the MV3 service worker
      // is waking up is recoverable — drop this one event and keep listening.
      const m = err?.message || "";
      if (m.includes("Extension context invalidated")) contextValid = false;
    });
  } catch (e) {
    if ((e?.message || "").includes("Extension context invalidated")) {
      contextValid = false;
    }
  }
}

// ---- Click Tracking ----
document.addEventListener("click", () => {
  if (!contextValid) return;
  safeSendMessage({ type: "CONTENT_EVENT", event: { type: "click" } });
});

// ---- Scroll Tracking ----
let scrollTimeout = null;
document.addEventListener("scroll", () => {
  if (!contextValid) return;
  // Throttle: send at most once per second
  if (!scrollTimeout) {
    safeSendMessage({ type: "CONTENT_EVENT", event: { type: "scroll" } });
    scrollTimeout = setTimeout(() => { scrollTimeout = null; }, 1000);
  }
});

// ---- Mouse Movement Tracking ----
let lastMouseX = 0;
let lastMouseY = 0;
let accumulatedDistance = 0;
const MOUSE_SEND_THRESHOLD = 100; // send every 100px of movement

document.addEventListener("mousemove", (e) => {
  if (!contextValid) return;
  if (lastMouseX && lastMouseY) {
    const dx = e.clientX - lastMouseX;
    const dy = e.clientY - lastMouseY;
    accumulatedDistance += Math.sqrt(dx * dx + dy * dy);

    if (accumulatedDistance >= MOUSE_SEND_THRESHOLD) {
      safeSendMessage({
        type: "CONTENT_EVENT",
        event: { type: "mousemove", distance: Math.round(accumulatedDistance) }
      });
      accumulatedDistance = 0;
    }
  }
  lastMouseX = e.clientX;
  lastMouseY = e.clientY;
});

// ---- Video Interaction Tracking ----
// Works with HTML5 video elements (YouTube, Coursera, etc.)

function trackVideoElement(video) {
  if (video._focusMonitorTracked) return;
  video._focusMonitorTracked = true;

  let lastTime = 0;

  video.addEventListener("seeked", () => {
    if (!contextValid) return;
    const timeDiff = video.currentTime - lastTime;
    if (timeDiff < -2) {
      safeSendMessage({ type: "CONTENT_EVENT", event: { type: "video_replay" } });
    } else if (timeDiff > 10) {
      safeSendMessage({ type: "CONTENT_EVENT", event: { type: "video_skip" } });
    }
  });

  video.addEventListener("ratechange", () => {
    if (!contextValid) return;
    safeSendMessage({ type: "CONTENT_EVENT", event: { type: "video_speed", speed: video.playbackRate } });
  });

  video.addEventListener("pause", () => {
    if (!contextValid) return;
    safeSendMessage({ type: "CONTENT_EVENT", event: { type: "video_pause" } });
  });

  video.addEventListener("play", () => {
    if (!contextValid) return;
    safeSendMessage({ type: "CONTENT_EVENT", event: { type: "video_play" } });
  });

  video.addEventListener("timeupdate", () => {
    lastTime = video.currentTime;
  });
}

// Find existing videos
document.querySelectorAll("video").forEach(trackVideoElement);

// Watch for dynamically added videos (YouTube, etc.)
const observer = new MutationObserver((mutations) => {
  if (!contextValid) { observer.disconnect(); return; }
  for (const mutation of mutations) {
    for (const node of mutation.addedNodes) {
      if (node.nodeName === "VIDEO") {
        trackVideoElement(node);
      }
      if (node.querySelectorAll) {
        node.querySelectorAll("video").forEach(trackVideoElement);
      }
    }
  }
});

observer.observe(document.body, { childList: true, subtree: true });
