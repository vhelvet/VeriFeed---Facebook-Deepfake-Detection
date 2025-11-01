// VeriFeed Content Script - PREDICTION ONLY
// Analyzes videos using the prediction backend


class VeriFeedPredictor {
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


    // Prediction configuration
    this.TARGET_FPS = 5;
    this.EXTRACT_DURATION = 30;
    this.TARGET_FRAMES = 150;


    this.init();
  }


  init() {
    console.log("VeriFeed Predictor initialized");
    console.log(`Target: ${this.TARGET_FRAMES} frames at ${this.TARGET_FPS}fps for ${this.EXTRACT_DURATION}s`);
    this.loadSettings();
    this.checkServerHealth();
    this.setupMutationObserver();
    this.scanForVideos();
  }


  loadSettings() {
    chrome.storage.local.get(["verifeedEnabled"], (result) => {
      this.isEnabled = result.verifeedEnabled !== false;
    });
  }


  async checkServerHealth() {
    try {
      const response = await fetch(`${this.serverUrl}/health`);
      const data = await response.json();
     
      if (data.status === "healthy" && data.model_loaded) {
        console.log("✅ Backend server ready");
        console.log(`   Device: ${data.device}`);
        console.log(`   Model: ${data.model_path || 'best_model.pt'}`);
      } else if (!data.model_loaded) {
        console.warn("⚠️ Backend online but model not loaded");
        console.warn(`   Error: ${data.model_error || 'Unknown'}`);
      }
    } catch (error) {
      console.error("❌ Backend server offline");
      console.error("   Make sure the prediction server is running on port 5000");
    }
  }


  setupMutationObserver() {
    this.observer = new MutationObserver((mutations) => {
      mutations.forEach((mutation) => {
        if (mutation.addedNodes.length) {
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
    if (!this.isEnabled) return;


    const videos = document.querySelectorAll("video");
    const videoPosts = this.findVideoPosts();


    videos.forEach((videoElement) => {
      if (this.analyzedVideos.has(videoElement)) return;


      let container = this.findVideoPostContainer(videoElement);
      if (!container) return;


      if (container.querySelector(".verifeed-predict-btn")) return;


      this.addPredictButton(container, videoElement);
    });


    videoPosts.forEach((post) => {
      if (post.querySelector(".verifeed-predict-btn")) return;


      const videoElement = post.querySelector("video");
      if (videoElement && !this.analyzedVideos.has(videoElement)) {
        this.addPredictButton(post, videoElement);
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
    ];


    const posts = new Set();
    selectors.forEach((selector) => {
      document.querySelectorAll(selector).forEach((element) => {
        if (element.querySelector("video")) {
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
      const hasVideoContent = element.querySelector("video");
      const hasPostStructure =
        element.querySelector('[data-ad-preview="message"]') ||
        element.querySelector("h3, h4") ||
        element.querySelector('[role="button"]');


      if (hasVideoContent && hasPostStructure) {
        return element;
      }


      element = element.parentElement;
      attempts++;
    }


    return null;
  }


  addPredictButton(container, videoElement) {
    if (container.querySelector(".verifeed-predict-btn")) return;


    const predictBtn = document.createElement("button");
    predictBtn.className = "verifeed-predict-btn";
    predictBtn.innerHTML = `
      <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px;">
        <path d="M9 12l2 2 4-4"/>
        <circle cx="12" cy="12" r="10"/>
      </svg>
      <span>Check Video</span>
    `;


    const postHeader = container.querySelector('h3, h4, [data-ad-preview="message"]')?.closest("div");
    const targetContainer = postHeader || container;


    const targetContainerStyle = window.getComputedStyle(targetContainer);
    if (targetContainerStyle.position === "static") {
      targetContainer.style.position = "relative";
    }


    const menuButton = targetContainer.querySelector('[aria-label*="more"], [aria-label*="options"]');
    let buttonPosition = "60px";


    if (menuButton) {
      const menuRect = menuButton.getBoundingClientRect();
      const targetRect = targetContainer.getBoundingClientRect();
      const relativeRight = targetRect.right - menuRect.right + menuRect.width + 8;
      buttonPosition = `${relativeRight}px`;
    }


    predictBtn.style.cssText = `
      position: absolute !important;
      top: 12px !important;
      right: ${buttonPosition} !important;
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


    predictBtn.onmouseenter = () => {
      predictBtn.style.background = "linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)";
      predictBtn.style.transform = "translateY(-1px)";
    };
    predictBtn.onmouseleave = () => {
      predictBtn.style.background = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)";
      predictBtn.style.transform = "translateY(0)";
    };


    predictBtn.onclick = (e) => {
      e.stopPropagation();
      e.preventDefault();
      this.handlePredictClick(container, videoElement, predictBtn);
    };


    targetContainer.appendChild(predictBtn);


    this.analyzedVideos.set(videoElement, {
      container,
      button: predictBtn,
    });
  }


  async handlePredictClick(container, videoElement, buttonElement) {
    console.log("=== STARTING VIDEO PREDICTION ===");


    if (buttonElement.dataset.analyzing === "true") {
      console.log("Already analyzing, ignoring click");
      return;
    }


    buttonElement.dataset.analyzing = "true";


    const originalContent = buttonElement.innerHTML;
    buttonElement.innerHTML = `
      <div style="width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 4px;"></div>
      <span>Extracting...</span>
      <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
    `;
    buttonElement.disabled = true;


    const originalVideoState = {
      paused: videoElement.paused,
      currentTime: videoElement.currentTime,
      muted: videoElement.muted,
    };


    videoElement.pause();
    videoElement.muted = true;


    const scrollY = window.scrollY;
    document.body.style.overflow = "hidden";
    document.body.style.position = "fixed";
    document.body.style.top = `-${scrollY}px`;
    document.body.style.width = "100%";


    try {
      console.log("=== EXTRACTING FRAMES ===");
      const frames = await this.extractFrames(videoElement);


      if (!frames || frames.length === 0) {
        throw new Error("Could not extract frames from video");
      }
      console.log(`Successfully extracted ${frames.length} frames`);


      this.restorePageState(scrollY, originalVideoState, videoElement);


      buttonElement.innerHTML = `
        <div style="width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 4px;"></div>
        <span>Analyzing...</span>
      `;


      console.log("=== SENDING TO BACKEND ===");
      await this.sendToBackend(frames, buttonElement, originalContent);


    } catch (error) {
      console.error("=== PREDICTION ERROR ===");
      console.error(error.message);


      this.restorePageState(scrollY, originalVideoState, videoElement);


      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;
      delete buttonElement.dataset.analyzing;


      setTimeout(() => {
        this.showErrorPopup(buttonElement, error.message);
      }, 100);
    }
  }


  restorePageState(scrollY, originalVideoState, videoElement) {
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


  async extractFrames(videoElement) {
    return new Promise((resolve, reject) => {
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


  async sendToBackend(frames, buttonElement, originalContent) {
    try {
      console.log("=== CHECKING SERVER HEALTH ===");
      const healthResponse = await this.makeRequest(`${this.serverUrl}/health`, "GET");


      if (!healthResponse.ok) {
        throw new Error("Server offline");
      }


      const healthData = await healthResponse.json();
      console.log("Server health:", healthData.status);


      if (healthData.status !== "healthy") {
        throw new Error("Server not healthy");
      }


      if (!healthData.model_loaded) {
        throw new Error("Model not loaded on server");
      }


      const requestData = {
        frames: frames,
      };


      console.log(`Sending ${frames.length} frames for prediction`);


      const response = await this.makeRequest(
        `${this.serverUrl}/predict`,
        "POST",
        requestData
      );


      let predictionData;
      try {
        predictionData = await response.json();
      } catch (jsonError) {
        console.error("Failed to parse response:", jsonError);
        throw new Error("Invalid server response");
      }


      if (!response.ok) {
        console.error("=== SERVER ERROR ===");
        let errorMsg = predictionData.error || "Prediction failed";
        throw new Error(errorMsg);
      }


      console.log("=== PREDICTION SUCCESS ===");
      console.log("Prediction:", predictionData.prediction);
      console.log("Confidence:", predictionData.confidence);
      console.log("Real probability:", predictionData.real_probability);
      console.log("Fake probability:", predictionData.fake_probability);


      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;
      delete buttonElement.dataset.analyzing;


      setTimeout(() => {
        this.showResultsPopup(buttonElement, predictionData);
      }, 100);


    } catch (error) {
      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;
      delete buttonElement.dataset.analyzing;
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


  showResultsPopup(buttonElement, result) {
    this.removeExistingPopup();


    const prediction = result.prediction;
    const confidence = result.confidence || 0;
    const realProb = result.real_probability || 0;
    const fakeProb = result.fake_probability || 0;
    const isAuthentic = prediction === "REAL";


    const buttonRect = buttonElement.getBoundingClientRect();
    const resultsPopup = document.createElement("div");
    resultsPopup.className = "verifeed-results-popup";


    const statusIcon = isAuthentic ? "✅" : "⚠️";
    const statusText = isAuthentic ? "Authentic Video" : "Deepfake Detected";
    const statusColor = isAuthentic ? "#10b981" : "#ef4444";
    const bgColor = isAuthentic ? "#f0fdf4" : "#fef2f2";


    resultsPopup.innerHTML = `
      <div class="verifeed-popup-content">
        <div class="verifeed-popup-header" style="background: ${bgColor};">
          <span class="status-icon">${statusIcon}</span>
          <span class="status-text" style="color: ${statusColor};">${statusText}</span>
          <button class="close-btn">×</button>
        </div>
        <div class="verifeed-popup-body">
          <div class="confidence-section">
            <span class="confidence-label">Confidence: ${confidence.toFixed(1)}%</span>
            <div class="confidence-bar">
              <div class="confidence-fill" style="width: ${confidence}%; background: ${statusColor};"></div>
            </div>
          </div>
          <div class="probability-section">
            <div class="prob-item">
              <span class="prob-label">Real: ${realProb.toFixed(1)}%</span>
              <div class="prob-bar">
                <div class="prob-fill" style="width: ${realProb}%; background: #10b981;"></div>
              </div>
            </div>
            <div class="prob-item">
              <span class="prob-label">Fake: ${fakeProb.toFixed(1)}%</span>
              <div class="prob-bar">
                <div class="prob-fill" style="width: ${fakeProb}%; background: #ef4444;"></div>
              </div>
            </div>
          </div>
          <div class="info-text">
            ${isAuthentic
              ? "This video appears to be genuine."
              : "⚠️ This video may have been manipulated. Verify before sharing."}
          </div>
          <div class="metadata">
            <small>Frames analyzed: ${result.frames_processed || 'N/A'}</small>
          </div>
        </div>
      </div>
    `;


    resultsPopup.style.cssText = `
      position: fixed !important;
      top: ${buttonRect.bottom + 8}px !important;
      right: ${window.innerWidth - buttonRect.right}px !important;
      z-index: 2147483647 !important;
      width: 300px !important;
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
      background: white !important;
      border-radius: 8px !important;
      box-shadow: 0 8px 32px rgba(0, 0, 0, 0.2) !important;
      animation: slideDown 0.2s ease-out !important;
    `;


    const style = document.createElement("style");
    style.id = "verifeed-popup-styles";
    style.textContent = `
      @keyframes slideDown {
        from { opacity: 0; transform: translateY(-10px); }
        to { opacity: 1; transform: translateY(0); }
      }
      .verifeed-popup-header {
        display: flex;
        align-items: center;
        padding: 12px 16px;
        border-radius: 8px 8px 0 0;
      }
      .verifeed-popup-header .status-icon { font-size: 18px; margin-right: 8px; }
      .verifeed-popup-header .status-text { font-weight: 600; font-size: 14px; flex: 1; }
      .verifeed-popup-header .close-btn {
        background: none; border: none; color: #9ca3af;
        font-size: 20px; cursor: pointer; width: 24px; height: 24px;
        display: flex; align-items: center; justify-content: center;
      }
      .verifeed-popup-header .close-btn:hover { color: #6b7280; }
      .verifeed-popup-body { padding: 16px; }
      .confidence-section { margin-bottom: 16px; }
      .confidence-label {
        font-size: 13px; font-weight: 600; color: #374151;
        display: block; margin-bottom: 6px;
      }
      .confidence-bar {
        width: 100%; height: 8px; background: #e5e7eb;
        border-radius: 4px; overflow: hidden;
      }
      .confidence-fill {
        height: 100%; border-radius: 4px;
        transition: width 0.8s ease-out;
      }
      .probability-section { margin-bottom: 12px; }
      .prob-item { margin-bottom: 8px; }
      .prob-label {
        font-size: 12px; color: #6b7280;
        display: block; margin-bottom: 4px;
      }
      .prob-bar {
        width: 100%; height: 6px; background: #f3f4f6;
        border-radius: 3px; overflow: hidden;
      }
      .prob-fill {
        height: 100%; border-radius: 3px;
        transition: width 0.8s ease-out;
      }
      .info-text {
        font-size: 13px; color: #4b5563; line-height: 1.5;
        padding: 12px; background: #f9fafb; border-radius: 6px;
        margin-bottom: 8px;
      }
      .metadata {
        font-size: 11px; color: #9ca3af;
        text-align: center; padding-top: 8px;
        border-top: 1px solid #f3f4f6;
      }
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
    setTimeout(() => {
      if (resultsPopup.parentNode) closePopup();
    }, 20000);
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
      border: 2px solid #fecaca !important;
    `;


    const style = document.createElement("style");
    style.id = "verifeed-error-styles";
    style.textContent = `
      .verifeed-error-popup .error-header {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 12px 16px;
        background: #fef2f2;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        color: #dc2626;
        font-size: 14px;
      }
      .verifeed-error-popup .error-header .close-btn {
        background: none;
        border: none;
        color: #9ca3af;
        font-size: 20px;
        cursor: pointer;
        width: 24px;
        height: 24px;
      }
      .verifeed-error-popup .error-body {
        padding: 16px;
      }
      .verifeed-error-popup .error-body p {
        margin: 0 0 12px 0;
        font-size: 13px;
        color: #6b7280;
        line-height: 1.5;
      }
      .verifeed-error-popup .retry-btn {
        background: #667eea;
        color: white;
        border: none;
        padding: 8px 16px;
        border-radius: 6px;
        font-size: 13px;
        cursor: pointer;
        font-weight: 500;
      }
      .verifeed-error-popup .retry-btn:hover {
        background: #5a6fd8;
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
    }, 10000);
  }


  removeExistingPopup() {
    const existingPopups = document.querySelectorAll(
      ".verifeed-results-popup, .verifeed-error-popup"
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


// Initialize VeriFeed Predictor
let veriFeedInstance = null;


function initializeVeriFeed() {
  if (window.location.hostname.includes("facebook.com") && !veriFeedInstance) {
    console.log("🔮 Initializing VeriFeed Predictor for Facebook...");
    veriFeedInstance = new VeriFeedPredictor();
  }
}


// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getStatus") {
    sendResponse({
      enabled: veriFeedInstance ? veriFeedInstance.isEnabled : false,
      initialized: !!veriFeedInstance,
      videoCount: veriFeedInstance ? veriFeedInstance.analyzedVideos.size : 0,
      serverUrl: veriFeedInstance ? veriFeedInstance.serverUrl : "http://localhost:5000",
      targetFrames: veriFeedInstance ? veriFeedInstance.TARGET_FRAMES : 150,
      targetFps: veriFeedInstance ? veriFeedInstance.TARGET_FPS : 5,
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
}, 5000);


// Debug commands
window.checkServer = async function() {
  if (veriFeedInstance) {
    await veriFeedInstance.checkServerHealth();
  } else {
    console.log("❌ VeriFeed not initialized");
  }
};


window.getVeriFeedStatus = function() {
  if (veriFeedInstance) {
    console.log("=== VERIFEED PREDICTOR STATUS ===");
    console.log(`Enabled: ${veriFeedInstance.isEnabled}`);
    console.log(`Target Frames: ${veriFeedInstance.TARGET_FRAMES}`);
    console.log(`Target FPS: ${veriFeedInstance.TARGET_FPS}`);
    console.log(`Extract Duration: ${veriFeedInstance.EXTRACT_DURATION}s`);
    console.log(`Videos Found: ${veriFeedInstance.analyzedVideos.size}`);
    console.log(`Videos on page: ${document.querySelectorAll('video').length}`);
    console.log(`Server URL: ${veriFeedInstance.serverUrl}`);
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


window.testPrediction = async function() {
  console.log("🧪 Testing prediction on first video...");
  const video = document.querySelector('video');
  if (!video) {
    console.log("❌ No video found on page");
    return;
  }
 
  const data = veriFeedInstance.analyzedVideos.get(video);
  if (data && data.button) {
    data.button.click();
  } else {
    console.log("❌ Video not tracked by VeriFeed");
  }
};


console.log("=== VERIFEED PREDICTION MODE ===");
console.log("🔮 Real-time deepfake detection");
console.log("📊 Configuration: 150 frames at 5fps from first 30 seconds");
console.log("🔗 Connects to backend at http://localhost:5000");
console.log("");
console.log("🎮 Available Commands:");
console.log("  checkServer()         - Check backend server health");
console.log("  getVeriFeedStatus()   - Show current status");
console.log("  forceRescan()         - Force scan for videos");
console.log("  testPrediction()      - Test prediction on first video");
console.log("");
console.log("💡 Usage:");
console.log("  1. Ensure prediction backend is running (python app.py)");
console.log("  2. Scroll to find videos on Facebook");
console.log("  3. Click purple 'Check Video' button");
console.log("  4. View real-time deepfake analysis results");
console.log("");

