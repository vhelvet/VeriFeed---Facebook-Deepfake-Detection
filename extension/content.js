// VeriFeed Content Script - Enhanced Video Detection with Auto-Refresh
// NO BUTTONS, NO POPUPS on Facebook - Everything happens in popup.html

console.log("[VeriFeed] Content script loaded (Silent Mode)");

class VeriFeedVideoDetector {
  constructor() {
    this.currentVideo = null;
    this.lastVideoSrc = null; // Track which video we're watching
    this.TARGET_FPS = 5;
    this.EXTRACT_DURATION = 30;
    this.TARGET_FRAMES = 150;

    this.init();
  }

  init() {
    console.log("[VeriFeed] Initializing video detector...");
    this.detectVideos();

    // Watch for new videos being loaded
    const observer = new MutationObserver(() => {
      this.detectVideos();
    });

    observer.observe(document.body, {
      childList: true,
      subtree: true,
    });

    // Watch for scroll events to detect new videos
    let scrollTimeout;
    window.addEventListener(
      "scroll",
      () => {
        clearTimeout(scrollTimeout);
        scrollTimeout = setTimeout(() => {
          this.detectVideos();
          this.notifyPopupOfVideoChange();
        }, 300);
      },
      { passive: true }
    );
  }

  detectVideos() {
    const videos = document.querySelectorAll("video");
    let foundNewVideo = false;

    if (videos.length > 0) {
      // Get the first visible, playing or paused video
      for (const video of videos) {
        const rect = video.getBoundingClientRect();

        // Check if video is in viewport (visible area)
        const isInViewport =
          rect.top >= -rect.height / 2 &&
          rect.top <= window.innerHeight - rect.height / 2 &&
          rect.left >= -rect.width / 2 &&
          rect.left <= window.innerWidth - rect.width / 2;

        const isVisible = rect.width > 0 && rect.height > 0;

        if (isVisible && isInViewport && video.readyState >= 2) {
          const videoSrc = video.currentSrc || video.src;

          // Check if this is a different video than before
          if (videoSrc !== this.lastVideoSrc) {
            foundNewVideo = true;
            this.lastVideoSrc = videoSrc;
            console.log("[VeriFeed] NEW video detected:", {
              src: videoSrc,
              width: video.videoWidth,
              height: video.videoHeight,
              duration: video.duration,
            });
          }

          this.currentVideo = video;
          break;
        }
      }
    } else {
      if (this.currentVideo !== null) {
        foundNewVideo = true;
      }
      this.currentVideo = null;
      this.lastVideoSrc = null;
    }

    // Notify popup if video changed
    if (foundNewVideo) {
      this.notifyPopupOfVideoChange();
    }
  }

  notifyPopupOfVideoChange() {
    // Send message to popup if it's open
    chrome.runtime.sendMessage(
      {
        action: "videoChanged",
        hasVideo: this.currentVideo !== null,
        videoInfo: this.getVideoInfo(),
      },
      () => {
        // Ignore errors if popup is not open
        if (chrome.runtime.lastError) {
          console.log("[VeriFeed] Popup not open, message not delivered");
        }
      }
    );
  }

  // OPTIMIZED: Fast frame extraction using seek-based method
  // OPTIMIZED: Fast frame extraction using seek-based method
  async extractFrames() {
    if (!this.currentVideo) {
      throw new Error("No video available");
    }

    const videoElement = this.currentVideo;

    return new Promise(async (resolve, reject) => {
      let hasResolved = false;
      let timeoutId = null;

      try {
        if (videoElement.readyState < 2) {
          reject(new Error("Video not ready"));
          return;
        }

        const duration = videoElement.duration;
        if (!duration || duration < 3) {
          reject(new Error(`Video too short: ${duration}s`));
          return;
        }

        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        canvas.width = videoElement.videoWidth || 640;
        canvas.height = videoElement.videoHeight || 480;

        if (canvas.width < 224 || canvas.height < 224) {
          reject(new Error("Video dimensions too small"));
          return;
        }

        const EXTRACT_DURATION = Math.min(duration, this.EXTRACT_DURATION);
        const TARGET_FRAMES = this.TARGET_FRAMES;
        const TARGET_FPS = this.TARGET_FPS;

        const timeStep = 1 / TARGET_FPS;

        console.log(
          `[VeriFeed] Target: ${TARGET_FRAMES} frames at ${TARGET_FPS}fps from first ${EXTRACT_DURATION}s`
        );

        // Save original video state
        const originalState = {
          paused: videoElement.paused,
          currentTime: videoElement.currentTime,
          muted: videoElement.muted,
          volume: videoElement.volume,
        };

        // Prepare video for extraction - mute and pause
        videoElement.muted = true;
        videoElement.volume = 0;
        videoElement.pause();

        console.log("[VeriFeed] 🔇 Video muted for analysis");

        const frames = [];
        let startTime = Date.now();

        const targetTimes = [];
        for (let i = 0; i < TARGET_FRAMES; i++) {
          let time = i * timeStep;
          if (time >= EXTRACT_DURATION) {
            break;
          }
          targetTimes.push(time);
        }

        timeoutId = setTimeout(() => {
          if (!hasResolved) {
            hasResolved = true;
            videoElement.removeEventListener("error", handleError);
            console.warn(
              `[VeriFeed] ⏱️ Timeout - captured ${frames.length} frames`
            );
            this.restoreVideoState(videoElement, originalState);
            resolve(frames);
          }
        }, 5000);

        const handleError = (e) => {
          if (hasResolved) return;
          console.error("[VeriFeed] Video element error during seek:", e);
          clearTimeout(timeoutId);
          this.restoreVideoState(videoElement, originalState);
          reject(new Error("Video Error during extraction."));
        };

        videoElement.addEventListener("error", handleError);

        for (const time of targetTimes) {
          if (hasResolved || frames.length >= TARGET_FRAMES) break;

          await new Promise((innerResolve) => {
            const onSeeked = () => {
              videoElement.removeEventListener("seeked", onSeeked);
              innerResolve();
            };
            videoElement.addEventListener("seeked", onSeeked);
            videoElement.currentTime = time;
          });

          ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
          frames.push(canvas.toDataURL("image/jpeg", 0.7));

          if (frames.length % 50 === 0) {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            console.log(
              `[VeriFeed] Captured ${frames.length} frames in ${elapsed}s`
            );
          }
        }

        clearTimeout(timeoutId);
        videoElement.removeEventListener("error", handleError);
        hasResolved = true;

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log("[VeriFeed] === FAST EXTRACTION COMPLETE ===");
        console.log(
          `[VeriFeed] ✅ Captured ${frames.length} frames in ${elapsed}s (Target: <5s)`
        );

        // Restore video state
        this.restoreVideoState(videoElement, originalState);

        resolve(frames);
      } catch (err) {
        if (!hasResolved) {
          if (timeoutId) clearTimeout(timeoutId);
          console.error("[VeriFeed] === EXTRACTION ERROR (FATAL) ===");
          if (videoElement) {
            this.restoreVideoState(videoElement, {
              paused: videoElement.paused,
              currentTime: videoElement.currentTime,
              muted: videoElement.muted,
            });
          }
          reject(err);
        }
      }
    });
  }

  restoreVideoState(video, originalState) {
    try {
      video.currentTime = originalState.currentTime;
      video.muted = originalState.muted;
      if (!originalState.paused) {
        video.play().catch(() => {});
      }
    } catch (error) {
      console.error("[VeriFeed] Error restoring video state:", error);
    }
  }

  hasVideo() {
    this.detectVideos();
    return this.currentVideo !== null;
  }

  getVideoInfo() {
    if (!this.currentVideo) return null;

    return {
      width: this.currentVideo.videoWidth,
      height: this.currentVideo.videoHeight,
      duration: this.currentVideo.duration,
      currentTime: this.currentVideo.currentTime,
      src: this.currentVideo.currentSrc || this.currentVideo.src,
    };
  }
}

// Initialize detector
const videoDetector = new VeriFeedVideoDetector();

// Listen for messages from popup
// Improved refresh handler in content.js
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("[VeriFeed] Received message:", request.action);

  if (request.action === "checkVideo") {
    const hasVideo = videoDetector.hasVideo();
    const videoInfo = videoDetector.getVideoInfo();

    console.log("[VeriFeed] Video check:", { hasVideo, videoInfo });
    sendResponse({ hasVideo, videoInfo });
    return false;
  }

  if (request.action === "extractFrames") {
    console.log("[VeriFeed] Frame extraction requested");

    videoDetector
      .extractFrames()
      .then((frames) => {
        console.log(`[VeriFeed] Sending ${frames.length} frames to popup`);
        sendResponse({ success: true, frames });
      })
      .catch((error) => {
        console.error("[VeriFeed] Frame extraction error:", error);
        sendResponse({ success: false, error: error.message });
      });

    return true; // Keep message channel open for async response
  }

  if (request.action === "updateSettings") {
    console.log("[VeriFeed] Settings updated:", request.settings);
    sendResponse({ received: true });
    return false;
  }

  if (request.action === "refresh") {
    console.log("[VeriFeed] Refreshing video detection");
    
    // ENHANCED REFRESH LOGIC:
    // 1. Clear all cached data
    videoDetector.lastVideoSrc = null;
    videoDetector.currentVideo = null;
    
    // 2. Force immediate re-detection (synchronous)
    videoDetector.detectVideos();
    
    // 3. Wait a bit for DOM to stabilize, then detect again
    setTimeout(() => {
      videoDetector.detectVideos();
      
      // 4. Get fresh status
      const hasVideo = videoDetector.hasVideo();
      const videoInfo = videoDetector.getVideoInfo();
      
      console.log("[VeriFeed] After delayed refresh:", { hasVideo, videoInfo });
      
      // Notify popup of the refreshed state
      videoDetector.notifyPopupOfVideoChange();
    }, 300);
    
    // 5. Immediate response with current state
    const hasVideo = videoDetector.hasVideo();
    const videoInfo = videoDetector.getVideoInfo();
    
    console.log("[VeriFeed] Immediate refresh response:", { hasVideo, videoInfo });
    sendResponse({ received: true, hasVideo, videoInfo });
    return false;
  }

  console.warn("[VeriFeed] Unknown action:", request.action);
  sendResponse({ error: "Unknown action" });
  return false;
});

// Periodic video detection (every 3 seconds)
setInterval(() => {
  videoDetector.detectVideos();
}, 3000);

console.log("[VeriFeed] Content script ready (Silent Mode - No UI)");
