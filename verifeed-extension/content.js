// VeriFeed Content Script - Training Data Export Version (FIXED)
// Based on working prediction code patterns

class VeriFeedDetector {
  constructor() {
    this.analyzedVideos = new Map();
    this.cachedFrames = new WeakMap();
    this.serverUrl = "http://localhost:5000";
    this.isEnabled = true;
    this.observer = null;
    this.activePopup = null;
    this.activeStyle = null;
    this.maxRetries = 3;
    this.retryDelay = 1000;
    this.scrollListener = null;
    this.clickListener = null;

    // Training configuration
    this.TRAINING_MODE = true; // ENABLED BY DEFAULT
    this.TARGET_FPS = 10;
    this.EXTRACT_DURATION = 20;
    this.TARGET_FRAMES = 200;

    this.init();
  }

  init() {
    console.log("VeriFeed initialized - Training Data Export Version");
    console.log(`Target: ${this.TARGET_FRAMES} frames at ${this.TARGET_FPS}fps for ${this.EXTRACT_DURATION}s`);
    this.loadSettings();
    this.setupMutationObserver();
    this.scanForVideos();
    console.log("Initial scan for videos triggered");
  }

  loadSettings() {
    chrome.storage.local.get(["verifeedEnabled", "trainingMode"], (result) => {
      this.isEnabled = result.verifeedEnabled !== false;
      this.TRAINING_MODE = result.trainingMode !== false; // Default to true
      console.log(`Training mode: ${this.TRAINING_MODE ? "ENABLED" : "DISABLED"}`);
    });
  }

  setupMutationObserver() {
    this.observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.addedNodes.length) {
          console.log("DOM mutation detected, rescanning for videos");
          setTimeout(() => this.scanForVideos(), 100);
        }
      });
    });

    this.observer.observe(document.body, {
      childList: true,
      subtree: true,
    });
  }

  scanForVideos() {
    if (!this.isEnabled) {
      console.log("VeriFeed is disabled, skipping scan");
      return;
    }

    console.log("Scanning for videos...");

    const videos = document.querySelectorAll("video");
    console.log(`Found ${videos.length} video elements`);

    const videoPosts = this.findVideoPosts();
    console.log(`Found ${videoPosts.length} video posts`);

    videos.forEach((videoElement, index) => {
      if (this.analyzedVideos.has(videoElement)) {
        console.log(`Video #${index} already analyzed, skipping`);
        return;
      }

      let container = this.findVideoPostContainer(videoElement);
      if (!container) {
        console.log(`No video post container found for video #${index}, skipping`);
        return;
      }

      if (container.querySelector(".verifeed-verify-btn")) {
        console.log(`Verify button already exists in container for video #${index}, skipping`);
        return;
      }

      console.log(`Adding verify button to video post #${index}`);
      this.addVerifyButton(container, videoElement);
    });

    videoPosts.forEach((post, index) => {
      if (post.querySelector(".verifeed-verify-btn")) {
        return;
      }

      const videoElement = post.querySelector("video");
      if (videoElement && !this.analyzedVideos.has(videoElement)) {
        console.log(`Found video in post #${index}, adding button`);
        this.addVerifyButton(post, videoElement);
      }
    });
  }

  findVideoPosts() {
    const selectors = [
      '[data-pagelet*="video"]',
      '[data-pagelet*="FeedUnit"]',
      '[role="article"]',
      '[data-ft*="video"]',
      '[data-pagelet*="permalink"]',
      '[data-pagelet*="root"]',
      '[data-pagelet*="timeline"]',
      '[data-pagelet*="main_column"]',
      '[data-pagelet*="content"]',
    ];

    const posts = new Set();

    selectors.forEach((selector) => {
      document.querySelectorAll(selector).forEach((element) => {
        if (
          element.querySelector("video") ||
          element.textContent?.includes("video") ||
          element.getAttribute("data-ft")?.includes("video") ||
          element.getAttribute("data-pagelet")?.includes("video")
        ) {
          posts.add(element);
        }
      });
    });

    return Array.from(posts);
  }

  findVideoPostContainer(videoElement) {
    let element = videoElement.parentElement;
    let attempts = 0;
    const maxAttempts = 15;

    while (element && attempts < maxAttempts) {
      const hasVideoContent =
        element.querySelector("video") ||
        element.textContent?.includes("video") ||
        element.getAttribute("data-ft")?.includes("video");

      const hasPostStructure =
        element.querySelector('[data-ad-preview="message"]') ||
        element.querySelector('[data-pagelet="FeedUnit_0"]') ||
        element.querySelector("h3") ||
        element.querySelector("h4") ||
        element.querySelector('[aria-label*="video"]') ||
        element.querySelector('[role="button"]');

      const hasVideoPostStructure =
        element.children.length > 3 &&
        (element.getAttribute("data-pagelet")?.includes("video") ||
          element.getAttribute("data-ft")?.includes("video") ||
          element.getAttribute("role") === "article");

      if (hasVideoContent && (hasPostStructure || hasVideoPostStructure)) {
        console.log(`Found video post container after ${attempts} attempts`);
        return element;
      }

      element = element.parentElement;
      attempts++;
    }

    return null;
  }

  addVerifyButton(container, videoElement) {
    if (container.querySelector(".verifeed-verify-btn")) {
      console.log("Verify button already exists in container");
      return;
    }

    const verifyBtn = document.createElement("button");
    verifyBtn.className = "verifeed-verify-btn";
    verifyBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px;">
        <path d="M9 12l2 2 4-4"/>
        <circle cx="12" cy="12" r="10"/>
      </svg>
      <span>VeriFeed</span>
    `;

    const postHeader = container
      .querySelector('h3, h4, [data-ad-preview="message"]')
      ?.closest("div");
    const targetContainer = postHeader || container;

    const targetContainerStyle = window.getComputedStyle(targetContainer);
    if (targetContainerStyle.position === "static") {
      targetContainer.style.position = "relative";
    }

    const menuButton = targetContainer.querySelector(
      '[aria-label*="more"], [aria-label*="options"], [aria-label*="menu"]'
    );
    let buttonPosition = "60px";

    if (menuButton) {
      const menuRect = menuButton.getBoundingClientRect();
      const targetRect = targetContainer.getBoundingClientRect();
      const relativeRight = targetRect.right - menuRect.right + menuRect.width + 8;
      buttonPosition = `${relativeRight}px`;
      console.log(`Found menu button in post header, positioning VeriFeed button at ${buttonPosition} from right`);
    } else {
      console.log("Menu button not found in post header, using fallback positioning");
    }

    verifyBtn.style.cssText = `
      position: absolute !important;
      top: 12px !important;
      right: ${buttonPosition} !important;
      left: auto !important;
      z-index: 2147483647 !important;
      background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
      color: white !important;
      border: none !important;
      border-radius: 6px !important;
      padding: 6px 10px !important;
      font-size: 12px !important;
      font-weight: 500 !important;
      cursor: pointer !important;
      box-shadow: 0 2px 8px rgba(102, 126, 234, 0.3) !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
      display: inline-flex !important;
      align-items: center !important;
      transition: all 0.2s ease !important;
    `;

    verifyBtn.onmouseenter = () => {
      verifyBtn.style.background = "linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)";
      verifyBtn.style.transform = "translateY(-1px)";
    };
    verifyBtn.onmouseleave = () => {
      verifyBtn.style.background = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)";
      verifyBtn.style.transform = "translateY(0)";
    };

    verifyBtn.onclick = (e) => {
      e.stopPropagation();
      e.preventDefault();
      this.handleVerifyClick(container, videoElement, verifyBtn);
    };

    console.log("Adding VeriFeed button to post header beside menu button");
    targetContainer.appendChild(verifyBtn);

    verifyBtn.style.display = "inline-flex";

    setTimeout(() => {
      const updatedMenuButton = targetContainer.querySelector(
        '[aria-label*="more"], [aria-label*="options"], [aria-label*="menu"]'
      );
      if (updatedMenuButton) {
        const menuRect = updatedMenuButton.getBoundingClientRect();
        const targetRect = targetContainer.getBoundingClientRect();
        const relativeRight = targetRect.right - menuRect.right + menuRect.width + 8;
        verifyBtn.style.right = `${relativeRight}px`;
      }
      verifyBtn.style.left = "auto";
      verifyBtn.style.position = "absolute";
      console.log("Reinforced button positioning");
    }, 100);

    this.analyzedVideos.set(videoElement, {
      container,
      button: verifyBtn,
    });

    console.log("VeriFeed button added successfully");
  }

  async handleVerifyClick(container, videoElement, buttonElement) {
    console.log("=== STARTING VIDEO VERIFICATION ===");

    // Prevent multiple clicks - SAME AS WORKING VERSION
    if (buttonElement.dataset.analyzing === "true") {
      console.log("Already analyzing this video, ignoring click");
      return;
    }

    buttonElement.dataset.analyzing = "true";

    // Show loading state - SAME AS WORKING VERSION
    const originalContent = buttonElement.innerHTML;
    buttonElement.innerHTML = `
      <div style="width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 4px;"></div>
      <span>Extracting...</span>
      <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
    `;
    buttonElement.disabled = true;

    // Save video state - SAME AS WORKING VERSION
    const originalVideoState = {
      paused: videoElement.paused,
      currentTime: videoElement.currentTime,
      muted: videoElement.muted,
    };

    videoElement.pause();
    videoElement.muted = true;

    // Lock scroll - SAME AS WORKING VERSION
    const scrollY = window.scrollY;
    document.body.style.overflow = "hidden";
    document.body.style.position = "fixed";
    document.body.style.top = `-${scrollY}px`;
    document.body.style.width = "100%";

    try {
      // Extract frames for training
      console.log("=== EXTRACTING FRAMES FOR TRAINING ===");
      const frames = await this.extractFramesForTraining(videoElement);
      
      if (!frames || frames.length === 0) {
        throw new Error("Could not extract frames from video");
      }
      console.log(`Successfully extracted ${frames.length} frames`);

      // Restore page state - SAME AS WORKING VERSION
      this.restorePageState(scrollY, originalVideoState, videoElement);
      
      // Restore button - SAME AS WORKING VERSION
      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;
      delete buttonElement.dataset.analyzing;

      // Show training options or analyze - SAME PATTERN AS WORKING VERSION
      if (this.TRAINING_MODE) {
        console.log("=== SHOWING TRAINING OPTIONS ===");
        setTimeout(() => {
          this.showTrainingOptionsPopup(buttonElement, frames, videoElement);
        }, 100);
      } else {
        console.log("=== SENDING TO BACKEND ===");
        buttonElement.innerHTML = `
          <div style="width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 4px;"></div>
          <span>Analyzing...</span>
        `;
        buttonElement.disabled = true;
        
        await this.sendToBackendForAnalysis(frames, buttonElement, originalContent);
      }

    } catch (error) {
      console.error("=== VERIFICATION ERROR ===");
      console.error(error.message);

      let userMessage = error.message || "Check failed";

      // Restore page state - SAME AS WORKING VERSION
      this.restorePageState(scrollY, originalVideoState, videoElement);
      
      // Restore button - SAME AS WORKING VERSION
      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;
      delete buttonElement.dataset.analyzing;
      
      // Show error popup - SAME PATTERN AS WORKING VERSION
      setTimeout(() => {
        this.showErrorPopup(buttonElement, userMessage);
      }, 100);
    }
  }

  restorePageState(scrollY, originalVideoState, videoElement) {
    console.log("Restoring page state");
    document.body.style.overflow = "";
    document.body.style.position = "";
    document.body.style.top = "";
    document.body.style.width = "";
    window.scrollTo(0, scrollY);

    videoElement.currentTime = originalVideoState.currentTime;
    videoElement.muted = originalVideoState.muted;
    if (!originalVideoState.paused) {
      videoElement.play().catch(() => {});
    }
  }

  async extractFramesForTraining(videoElement) {
    return new Promise((resolve, reject) => {
      try {
        console.log("=== FRAME EXTRACTION FOR TRAINING ===");
        
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
        const ctx = canvas.getContext("2d", { willReadFrequently: true });
        canvas.width = videoElement.videoWidth || 640;
        canvas.height = videoElement.videoHeight || 480;

        if (canvas.width < 224 || canvas.height < 224) {
          reject(new Error("Video dimensions too small"));
          return;
        }

        console.log(`Canvas: ${canvas.width}x${canvas.height}`);
        console.log(`Duration: ${duration}s`);

        const EXTRACT_DURATION = Math.min(duration, this.EXTRACT_DURATION);
        const TARGET_FRAMES = Math.floor(EXTRACT_DURATION * this.TARGET_FPS);
        
        console.log(`Target: ${TARGET_FRAMES} frames at ${this.TARGET_FPS}fps`);

        const frames = [];
        let captureInterval;
        let startTime = Date.now();

        const captureFrame = () => {
          try {
            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
            const frame = canvas.toDataURL("image/jpeg", 0.85);
            frames.push(frame);
            
            if (frames.length % 50 === 0) {
              const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
              console.log(`Captured ${frames.length} frames in ${elapsed}s`);
            }
          } catch (err) {
            console.warn("Frame capture error:", err);
          }
        };

        captureFrame();

        videoElement.currentTime = 0;
        videoElement.playbackRate = 1.0;
        
        videoElement.play().then(() => {
          console.log("Video playing, capturing frames...");
          
          const intervalMs = 1000 / this.TARGET_FPS;
          captureInterval = setInterval(captureFrame, intervalMs);
          
        }).catch(err => {
          reject(new Error("Could not play video: " + err.message));
        });

        const checkCompletion = setInterval(() => {
          const currentTime = videoElement.currentTime;
          
          if (frames.length >= TARGET_FRAMES || currentTime >= EXTRACT_DURATION) {
            clearInterval(captureInterval);
            clearInterval(checkCompletion);
            videoElement.pause();
            
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            const actualFps = (frames.length / currentTime).toFixed(1);
            
            console.log("=== EXTRACTION COMPLETE ===");
            console.log(`✅ Captured ${frames.length} frames in ${elapsed}s`);
            console.log(`   Actual FPS: ${actualFps}`);
            console.log(`   Video time: ${currentTime.toFixed(1)}s`);
            
            resolve(frames);
          }
        }, 100);

        setTimeout(() => {
          if (frames.length > 0) {
            clearInterval(captureInterval);
            clearInterval(checkCompletion);
            videoElement.pause();
            console.warn(`⏱️ Timeout - captured ${frames.length} frames`);
            resolve(frames);
          } else {
            reject(new Error("Extraction timeout"));
          }
        }, 25000);

      } catch (err) {
        console.error("=== EXTRACTION ERROR ===");
        console.error(err);
        reject(err);
      }
    });
  }

  async sendToBackendForAnalysis(frames, buttonElement, originalContent) {
    try {
      console.log("=== CHECKING SERVER HEALTH ===");
      const healthResponse = await this.makeRequest(`${this.serverUrl}/health`, "GET");

      if (!healthResponse.ok) {
        throw new Error("Server offline");
      }
      
      const healthData = await healthResponse.json();
      console.log("Server health:", healthData.status);

      if (healthData.status !== "healthy") {
        throw new Error("Server not in healthy state");
      }

      const requestData = {
        frames: frames,
        platform: "facebook",
      };

      console.log(`Sending ${frames.length} frames for analysis`);

      const response = await this.makeRequest(
        `${this.serverUrl}/frame_analyze`,
        "POST",
        requestData
      );

      let analysisData;
      try {
        analysisData = await response.json();
        console.log("=== BACKEND RESPONSE ===");
        console.log("Response:", analysisData);
      } catch (jsonError) {
        console.error("Failed to parse response:", jsonError);
        throw new Error("Invalid server response");
      }

      if (!response.ok) {
        console.error("=== SERVER ERROR ===");
        let errorMsg = analysisData.error || "Analysis failed";
        throw new Error(errorMsg);
      }

      console.log("=== ANALYSIS SUCCESS ===");
      console.log("Prediction:", analysisData.prediction);
      console.log("Confidence:", analysisData.confidence);

      // Restore button
      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;

      // Show results - SAME PATTERN AS WORKING VERSION
      setTimeout(() => {
        this.showResultsPopup(buttonElement, analysisData);
      }, 100);

    } catch (error) {
      // Restore button
      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;
      throw error;
    }
  }

  async makeRequest(url, method = "GET", data = null, retries = 0) {
    try {
      const options = {
        method: method,
        headers: {
          "Content-Type": "application/json",
        },
      };

      if (data) {
        options.body = JSON.stringify(data);
      }

      const response = await fetch(url, options);
      return response;
    } catch (error) {
      if (retries < this.maxRetries) {
        console.log(`Retrying... (${retries + 1}/${this.maxRetries})`);
        await new Promise((resolve) =>
          setTimeout(resolve, this.retryDelay * (retries + 1))
        );
        return this.makeRequest(url, method, data, retries + 1);
      }
      throw error;
    }
  }

  showTrainingOptionsPopup(buttonElement, frames, videoElement) {
    console.log("=== SHOWING TRAINING OPTIONS ===");

    this.removeExistingPopup();

    const buttonRect = buttonElement.getBoundingClientRect();
    const popup = document.createElement("div");
    popup.className = "verifeed-training-popup";

    popup.innerHTML = `
      <div class="training-content">
        <div class="training-header">
          <span>📊 Training Data Export</span>
          <button class="close-btn">×</button>
        </div>
        <div class="training-body">
          <div class="info">
            <p><strong>✅ Extracted ${frames.length} frames</strong></p>
            <p style="font-size: 11px; color: #666; margin-top: 4px;">
              ${this.TARGET_FPS}fps × ${this.EXTRACT_DURATION}s = ${this.TARGET_FRAMES} frames target
            </p>
          </div>
          <div class="actions">
            <button class="export-btn" data-action="export">
              💾 Export JSON
            </button>
            <button class="analyze-btn" data-action="analyze">
              🔍 Analyze Now
            </button>
            <button class="copy-btn" data-action="copy">
              📋 Copy Frames
            </button>
          </div>
          <div class="tip">
            <strong>For training:</strong> Export JSON and use with train.py
          </div>
        </div>
      </div>
    `;

    // SAME POSITIONING AS WORKING VERSION
    const topPosition = buttonRect.bottom + 8;
    const rightPosition = window.innerWidth - buttonRect.right;

    popup.style.cssText = `
      all: initial !important;
      position: fixed !important;
      top: ${topPosition}px !important;
      right: ${rightPosition}px !important;
      z-index: 2147483647 !important;
      width: 300px !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
      background: white !important;
      border-radius: 8px !important;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.8) !important;
      animation: slideDown 0.2s ease-out !important;
      display: block !important;
      visibility: visible !important;
      opacity: 1 !important;
      pointer-events: auto !important;
      transform: none !important;
      isolation: isolate !important;
    `;

    const style = document.createElement("style");
    style.id = "verifeed-training-styles";
    style.textContent = `
      @keyframes slideDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .verifeed-training-popup {
        pointer-events: auto !important;
      }
      .verifeed-training-popup .training-header {
        display: flex !important;
        align-items: center !important;
        justify-content: space-between !important;
        padding: 12px 16px !important;
        background: #f0f9ff !important;
        border-radius: 8px 8px 0 0 !important;
        font-weight: 600 !important;
        color: #0369a1 !important;
        font-size: 14px !important;
      }
      .verifeed-training-popup .close-btn {
        background: none !important;
        border: none !important;
        color: #64748b !important;
        font-size: 18px !important;
        cursor: pointer !important;
        padding: 0 !important;
        width: 20px !important;
        height: 20px !important;
      }
      .verifeed-training-popup .close-btn:hover {
        color: #475569 !important;
      }
      .verifeed-training-popup .training-body {
        padding: 16px !important;
      }
      .verifeed-training-popup .info {
        margin-bottom: 12px !important;
        padding: 12px !important;
        background: #f8fafc !important;
        border-radius: 6px !important;
      }
      .verifeed-training-popup .info p {
        margin: 0 !important;
        font-size: 13px !important;
        color: #334155 !important;
      }
      .verifeed-training-popup .actions {
        display: flex !important;
        flex-direction: column !important;
        gap: 8px !important;
        margin-bottom: 12px !important;
      }
      .verifeed-training-popup .actions button {
        padding: 10px 16px !important;
        border: none !important;
        border-radius: 6px !important;
        font-size: 13px !important;
        font-weight: 500 !important;
        cursor: pointer !important;
        transition: all 0.2s !important;
      }
      .verifeed-training-popup .export-btn {
        background: #10b981 !important;
        color: white !important;
      }
      .verifeed-training-popup .export-btn:hover {
        background: #059669 !important;
      }
      .verifeed-training-popup .analyze-btn {
        background: #667eea !important;
        color: white !important;
      }
      .verifeed-training-popup .analyze-btn:hover {
        background: #5a6fd8 !important;
      }
      .verifeed-training-popup .copy-btn {
        background: #f1f5f9 !important;
        color: #475569 !important;
      }
      .verifeed-training-popup .copy-btn:hover {
        background: #e2e8f0 !important;
      }
      .verifeed-training-popup .tip {
        font-size: 11px !important;
        color: #64748b !important;
        padding: 8px !important;
        background: #fef3c7 !important;
        border-radius: 4px !important;
        line-height: 1.4 !important;
      }
    `;

    document.head.appendChild(style);
    this.activeStyle = style;

    document.body.appendChild(popup);
    this.activePopup = popup;

    console.log("=== POPUP ADDED TO DOM ===");

    const closePopup = () => {
      console.log("Closing popup");
      if (popup.parentNode) popup.remove();
      if (style.parentNode) style.remove();
      this.activePopup = null;
      this.activeStyle = null;
    };

    popup.querySelector(".close-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      closePopup();
    });

    popup.querySelector(".export-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      this.exportFramesAsJSON(frames, videoElement);
      closePopup();
    });

    popup.querySelector(".analyze-btn").addEventListener("click", async (e) => {
      e.stopPropagation();
      e.preventDefault();
      closePopup();
      
      const button = this.analyzedVideos.get(videoElement)?.button;
      if (button) {
        const originalContent = button.innerHTML;
        button.innerHTML = `
          <div style="width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 4px;"></div>
          <span>Analyzing...</span>
        `;
        button.disabled = true;
        
        try {
          await this.sendToBackendForAnalysis(frames, button, originalContent);
        } catch (error) {
          button.innerHTML = originalContent;
          button.disabled = false;
          setTimeout(() => {
            this.showErrorPopup(button, error.message || "Analysis failed");
          }, 100);
        }
      }
    });

    popup.querySelector(".copy-btn").addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      this.copyFramesToClipboard(frames);
      closePopup();
    });

    // Auto-close after 30 seconds
    setTimeout(() => {
      if (popup.parentNode) closePopup();
    }, 30000);
  }

  exportFramesAsJSON(frames, videoElement) {
    console.log("=== EXPORTING FRAMES AS JSON ===");

    const exportData = {
      metadata: {
        totalFrames: frames.length,
        targetFrames: this.TARGET_FRAMES,
        fps: this.TARGET_FPS,
        duration: this.EXTRACT_DURATION,
        extractedAt: new Date().toISOString(),
        videoSrc: videoElement.src || videoElement.currentSrc,
        videoWidth: videoElement.videoWidth,
        videoHeight: videoElement.videoHeight,
        videoDuration: videoElement.duration,
        platform: "facebook",
      },
      frames: frames,
    };

    const jsonStr = JSON.stringify(exportData, null, 2);
    const blob = new Blob([jsonStr], { type: "application/json" });
    const url = URL.createObjectURL(blob);

    const timestamp = new Date().toISOString().replace(/[:.]/g, "-").slice(0, -5);
    const filename = `verifeed_frames_${frames.length}_${timestamp}.json`;

    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    console.log(`✅ Exported ${frames.length} frames to ${filename}`);
    console.log(`📦 File size: ~${(jsonStr.length / 1024 / 1024).toFixed(1)}MB`);

    this.showSuccessMessage(`Exported ${frames.length} frames!`);
  }

  copyFramesToClipboard(frames) {
    const data = {
      frames: frames,
      count: frames.length,
      timestamp: new Date().toISOString(),
    };

    navigator.clipboard.writeText(JSON.stringify(data))
      .then(() => {
        console.log("✅ Copied frames to clipboard");
        this.showSuccessMessage(`Copied ${frames.length} frames!`);
      })
      .catch((err) => {
        console.error("Failed to copy:", err);
        alert("Failed to copy frames");
      });
  }

  showSuccessMessage(message) {
    const toast = document.createElement("div");
    toast.textContent = message;
    toast.style.cssText = `
      position: fixed;
      top: 20px;
      right: 20px;
      background: #10b981;
      color: white;
      padding: 12px 20px;
      border-radius: 8px;
      box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
      z-index: 2147483647;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      font-size: 14px;
      font-weight: 500;
    `;

    document.body.appendChild(toast);

    setTimeout(() => {
      toast.style.transition = "opacity 0.3s";
      toast.style.opacity = "0";
      setTimeout(() => toast.remove(), 300);
    }, 3000);
  }

  showResultsPopup(buttonElement, result) {
    this.removeExistingPopup();

    const prediction = result.prediction;
    const confidence = result.confidence || 0;
    const isAuthentic = prediction === "REAL";

    const buttonRect = buttonElement.getBoundingClientRect();
    const resultsPopup = document.createElement("div");
    resultsPopup.className = "verifeed-results-popup";

    const statusIcon = isAuthentic ? "✅" : "⚠️";
    const statusText = isAuthentic ? "Authentic" : "Deepfake Detected";
    const statusColor = isAuthentic ? "#10b981" : "#f59e0b";

    resultsPopup.innerHTML = `
      <div class="verifeed-popup-content">
        <div class="verifeed-popup-header">
          <span class="status-icon">${statusIcon}</span>
          <span class="status-text">${statusText}</span>
          <button class="close-btn">×</button>
        </div>
        <div class="verifeed-popup-body">
          <div class="confidence-section">
            <span class="confidence-label">Confidence: ${confidence.toFixed(1)}%</span>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${confidence}%; background: ${statusColor};"></div>
            </div>
          </div>
          <div class="info-text">
            ${isAuthentic ? "This video appears to be genuine." : "This video may have been manipulated. Verify before sharing."}
          </div>
        </div>
      </div>
    `;

    resultsPopup.style.cssText = `
      position: fixed !important;
      top: ${buttonRect.bottom + 8}px !important;
      right: ${window.innerWidth - buttonRect.right}px !important;
      z-index: 2147483647 !important;
      width: 280px !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
      background: white !important;
      border-radius: 8px !important;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
      animation: slideDown 0.2s ease-out !important;
    `;

    const style = document.createElement("style");
    style.textContent = `
      .verifeed-popup-header {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        border-bottom: 1px solid #f3f4f6;
        background: #fafafa;
        border-radius: 8px 8px 0 0;
      }
      .verifeed-popup-header .status-icon { font-size: 16px; margin-right: 8px; }
      .verifeed-popup-header .status-text { font-weight: 600; color: #374151; font-size: 14px; flex: 1; }
      .verifeed-popup-header .close-btn { background: none; border: none; color: #9ca3af; font-size: 18px; cursor: pointer; }
      .verifeed-popup-body { padding: 16px; }
      .confidence-section { margin-bottom: 12px; }
      .confidence-label { font-size: 13px; font-weight: 600; color: #374151; display: block; margin-bottom: 6px; }
      .confidence-bar { width: 100%; height: 6px; background: #e5e7eb; border-radius: 3px; overflow: hidden; }
      .confidence-fill { height: 100%; border-radius: 3px; transition: width 0.8s ease-out; }
      .info-text { font-size: 13px; color: #4b5563; line-height: 1.4; }
    `;

    document.head.appendChild(style);
    document.body.appendChild(resultsPopup);
    this.activePopup = resultsPopup;
    this.activeStyle = style;

    const closeBtn = resultsPopup.querySelector(".close-btn");
    const closePopup = () => {
      if (resultsPopup.parentNode) resultsPopup.remove();
      if (style.parentNode) style.remove();
      this.activePopup = null;
      this.activeStyle = null;
    };

    closeBtn.addEventListener("click", closePopup);
    setTimeout(() => { if (resultsPopup.parentNode) closePopup(); }, 15000);
  }

  showErrorPopup(buttonElement, message) {
    this.removeExistingPopup();

    const buttonRect = buttonElement.getBoundingClientRect();
    const errorPopup = document.createElement("div");
    errorPopup.className = "verifeed-error-popup";

    errorPopup.innerHTML = `
      <div class="error-content">
        <div class="error-header">
          <span>⚠️ Error</span>
          <button class="close-btn">×</button>
        </div>
        <div class="error-body">
          <p>${message}</p>
          <button class="retry-btn">OK</button>
        </div>
      </div>
    `;

    errorPopup.style.cssText = `
      position: fixed !important;
      top: ${buttonRect.bottom + 8}px !important;
      right: ${window.innerWidth - buttonRect.right}px !important;
      z-index: 2147483647 !important;
      width: 280px !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
      background: white !important;
      border-radius: 8px !important;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.15) !important;
      border: 1px solid #fecaca !important;
    `;

    const style = document.createElement("style");
    style.textContent = `
      .verifeed-error-popup .error-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        background: #fef2f2;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
        color: #b91c1c;
        font-size: 14px;
      }
      .verifeed-error-popup .error-header .close-btn {
        background: none;
        border: none;
        color: #9ca3af;
        font-size: 18px;
        cursor: pointer;
      }
      .verifeed-error-popup .error-body {
        padding: 16px;
      }
      .verifeed-error-popup .error-body p {
        margin: 0 0 12px 0;
        font-size: 13px;
        color: #6b7280;
        line-height: 1.4;
      }
      .verifeed-error-popup .retry-btn {
        background: #1877f2;
        color: white;
        border: none;
        padding: 6px 12px;
        border-radius: 4px;
        font-size: 12px;
        cursor: pointer;
        font-weight: 500;
      }
    `;

    document.head.appendChild(style);
    document.body.appendChild(errorPopup);
    this.activePopup = errorPopup;
    this.activeStyle = style;

    const closeBtn = errorPopup.querySelector(".close-btn");
    const retryBtn = errorPopup.querySelector(".retry-btn");

    const closeErrorPopup = () => {
      if (errorPopup.parentNode) errorPopup.remove();
      if (style.parentNode) style.remove();
      this.activePopup = null;
      this.activeStyle = null;
    };

    closeBtn.addEventListener("click", closeErrorPopup);
    retryBtn.addEventListener("click", closeErrorPopup);

    setTimeout(() => {
      if (errorPopup.parentNode) closeErrorPopup();
    }, 8000);
  }

  removeExistingPopup() {
    if (this.scrollListener) {
      window.removeEventListener("scroll", this.scrollListener);
      this.scrollListener = null;
    }
    if (this.clickListener) {
      document.removeEventListener("click", this.clickListener);
      this.clickListener = null;
    }

    const existingPopups = document.querySelectorAll(
      ".verifeed-results-popup, .verifeed-error-popup, .verifeed-training-popup"
    );
    existingPopups.forEach((popup) => {
      if (popup.parentNode) popup.remove();
    });

    const existingStyles = document.querySelectorAll(
      "#verifeed-popup-styles, #verifeed-error-styles"
    );
    existingStyles.forEach((style) => {
      if (style.parentNode) style.remove();
    });

    if (this.activePopup && this.activePopup.parentNode) {
      this.activePopup.remove();
    }
    if (this.activeStyle && this.activeStyle.parentNode) {
      this.activeStyle.remove();
    }

    this.activePopup = null;
    this.activeStyle = null;
  }

  destroy() {
    if (this.observer) {
      this.observer.disconnect();
    }
    this.removeExistingPopup();
    this.analyzedVideos.clear();
  }
}

// Initialize VeriFeed
let veriFeedInstance = null;

function initializeVeriFeed() {
  if (window.location.hostname.includes("facebook.com") && !veriFeedInstance) {
    console.log("🚀 Initializing VeriFeed for Facebook...");
    veriFeedInstance = new VeriFeedDetector();
  }
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "toggleVeriFeed") {
    if (veriFeedInstance) {
      veriFeedInstance.isEnabled = request.enabled;
      if (!request.enabled) {
        veriFeedInstance.destroy();
        veriFeedInstance = null;
      }
    } else if (request.enabled) {
      initializeVeriFeed();
    }
    sendResponse({ success: true, enabled: request.enabled });
  }

  if (request.action === "toggleTrainingMode") {
    if (veriFeedInstance) {
      veriFeedInstance.TRAINING_MODE = request.enabled;
      console.log(`Training mode: ${request.enabled ? "ENABLED" : "DISABLED"}`);
      chrome.storage.local.set({ trainingMode: request.enabled });
      sendResponse({ success: true, trainingMode: request.enabled });
    } else {
      sendResponse({ success: false, error: "VeriFeed not initialized" });
    }
  }

  if (request.action === "getStatus") {
    sendResponse({
      enabled: veriFeedInstance ? veriFeedInstance.isEnabled : false,
      initialized: !!veriFeedInstance,
      trainingMode: veriFeedInstance ? veriFeedInstance.TRAINING_MODE : false,
      videoCount: veriFeedInstance ? veriFeedInstance.analyzedVideos.size : 0,
      serverUrl: veriFeedInstance ? veriFeedInstance.serverUrl : "http://localhost:5000",
      targetFrames: veriFeedInstance ? veriFeedInstance.TARGET_FRAMES : 200,
      targetFps: veriFeedInstance ? veriFeedInstance.TARGET_FPS : 10,
    });
  }

  return true;
});

// Initialize when DOM is ready
if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeVeriFeed);
} else {
  initializeVeriFeed();
}

// Re-scan periodically for new videos
setInterval(() => {
  if (veriFeedInstance && veriFeedInstance.isEnabled) {
    veriFeedInstance.scanForVideos();
  }
}, 5000); // Check every 5 seconds

// Listen for window messages
window.addEventListener('message', function(event) {
  if (event.source !== window) return;
  
  if (event.data.type === 'VERIFEED_ENABLE_TRAINING') {
    if (veriFeedInstance) {
      veriFeedInstance.TRAINING_MODE = true;
      chrome.storage.local.set({ trainingMode: true });
      console.log("✅ Training mode ENABLED");
    }
  } else if (event.data.type === 'VERIFEED_DISABLE_TRAINING') {
    if (veriFeedInstance) {
      veriFeedInstance.TRAINING_MODE = false;
      chrome.storage.local.set({ trainingMode: false });
      console.log("✅ Training mode DISABLED");
    }
  } else if (event.data.type === 'VERIFEED_STATUS') {
    if (veriFeedInstance) {
      console.log("=== VERIFEED STATUS ===");
      console.log(`Enabled: ${veriFeedInstance.isEnabled}`);
      console.log(`Training Mode: ${veriFeedInstance.TRAINING_MODE}`);
      console.log(`Target Frames: ${veriFeedInstance.TARGET_FRAMES}`);
      console.log(`Target FPS: ${veriFeedInstance.TARGET_FPS}`);
      console.log(`Extract Duration: ${veriFeedInstance.EXTRACT_DURATION}s`);
      console.log(`Videos Analyzed: ${veriFeedInstance.analyzedVideos.size}`);
    }
  }
});

// Debug commands - Available in console
window.enableTrainingMode = function() {
  if (veriFeedInstance) {
    veriFeedInstance.TRAINING_MODE = true;
    chrome.storage.local.set({ trainingMode: true });
    console.log("✅ Training mode ENABLED");
    console.log("   Click VeriFeed button to see export options");
  } else {
    console.log("❌ VeriFeed not initialized");
  }
};

window.disableTrainingMode = function() {
  if (veriFeedInstance) {
    veriFeedInstance.TRAINING_MODE = false;
    chrome.storage.local.set({ trainingMode: false });
    console.log("✅ Training mode DISABLED");
    console.log("   Click VeriFeed button for normal analysis");
  } else {
    console.log("❌ VeriFeed not initialized");
  }
};

window.getVeriFeedStatus = function() {
  if (veriFeedInstance) {
    console.log("=== VERIFEED STATUS ===");
    console.log(`Enabled: ${veriFeedInstance.isEnabled}`);
    console.log(`Training Mode: ${veriFeedInstance.TRAINING_MODE}`);
    console.log(`Target Frames: ${veriFeedInstance.TARGET_FRAMES}`);
    console.log(`Target FPS: ${veriFeedInstance.TARGET_FPS}`);
    console.log(`Extract Duration: ${veriFeedInstance.EXTRACT_DURATION}s`);
    console.log(`Videos Analyzed: ${veriFeedInstance.analyzedVideos.size}`);
    console.log(`Videos on page: ${document.querySelectorAll('video').length}`);
  } else {
    console.log("❌ VeriFeed not initialized");
  }
};

window.forceRescan = function() {
  if (veriFeedInstance) {
    console.log("🔄 Forcing video scan...");
    veriFeedInstance.scanForVideos();
  } else {
    console.log("❌ VeriFeed not initialized");
  }
};

console.log("=== VERIFEED TRAINING VERSION LOADED ===");
console.log("📊 Configuration: 200 frames at 10fps from first 20 seconds");
console.log("🎯 Training mode: ENABLED by default");
console.log("");
console.log("🎮 Available Commands:");
console.log("  enableTrainingMode()  - Enable training data export");
console.log("  disableTrainingMode() - Disable training mode");
console.log("  getVeriFeedStatus()   - Show current configuration");
console.log("  forceRescan()         - Force scan for videos");
console.log("");
console.log("💡 Usage:");
console.log("  1. Scroll to find videos on Facebook");
console.log("  2. Look for purple 'VeriFeed' buttons on videos");
console.log("  3. Click button → Choose 'Export JSON'");
console.log("  4. Use JSON file with train.py script");
console.log("");
console.log("🔧 Troubleshooting:");
console.log("  - No buttons? Run: forceRescan()");
console.log("  - Check status: getVeriFeedStatus()");