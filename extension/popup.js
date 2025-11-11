// VeriFeed Popup Script - Enhanced with Better Error Handling and UX
class VeriFeedPopup {
  constructor() {
    this.serverUrl = "http://localhost:5000";
    this.settings = {
      verifeedEnabled: true,
    };
    this.currentVideo = null;
    this.lastVerifiedVideoSrc = null;
    this.isAnalyzing = false;
    this.isCompactMode = false;
    this.healthCheckInterval = null;
    this.videoDetectionInterval = null;
    this.abortController = null;

    this.init();
  }

  async init() {
    console.log("[VeriFeed Popup] Initializing...");

    try {
      await this.loadSettings();
      this.setupEventListeners();
      this.setupMessageListener();
      this.updateUI();
      await this.checkServerStatus();
      await this.detectVideo();

      // Auto-refresh with cleanup capability
      this.healthCheckInterval = setInterval(() => this.checkServerStatus(), 10000);
      this.videoDetectionInterval = setInterval(() => this.detectVideo(), 2000);

      // Cleanup on window unload
      window.addEventListener('beforeunload', () => this.cleanup());
    } catch (error) {
      console.error("[VeriFeed Popup] Initialization error:", error);
      this.showInitializationError(error);
    }
  }

  cleanup() {
    if (this.healthCheckInterval) clearInterval(this.healthCheckInterval);
    if (this.videoDetectionInterval) clearInterval(this.videoDetectionInterval);
    if (this.abortController) this.abortController.abort();
  }

  async loadSettings() {
    try {
      const result = await chrome.storage.local.get([
        "verifeedEnabled",
        "serverUrl",
      ]);
      this.settings = {
        verifeedEnabled: result.verifeedEnabled !== false,
      };
      this.serverUrl = result.serverUrl || "http://localhost:5000";
      console.log("[VeriFeed Popup] Settings loaded:", this.settings);
    } catch (error) {
      console.error("[VeriFeed Popup] Error loading settings:", error);
      throw new Error("Failed to load extension settings");
    }
  }

  async saveSettings() {
    try {
      await chrome.storage.local.set({
        ...this.settings,
        serverUrl: this.serverUrl,
      });
      console.log("[VeriFeed Popup] Settings saved");
      return true;
    } catch (error) {
      console.error("[VeriFeed Popup] Error saving settings:", error);
      return false;
    }
  }

  setupMessageListener() {
    chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
      if (request.action === "videoChanged") {
        console.log("[VeriFeed Popup] Video changed:", request);
        this.handleVideoChange(request.hasVideo, request.videoInfo);
        sendResponse({ received: true });
        return true; // Keep channel open for async response
      }
    });
  }

  handleVideoChange(hasVideo, videoInfo) {
    const videoInfoEl = document.getElementById("videoInfo");
    const verifyBtn = document.getElementById("verifyBtn");
    const resultsSection = document.getElementById("resultsSection");

    const videoSrc = videoInfo?.src;
    const isDifferentVideo = videoSrc && videoSrc !== this.lastVerifiedVideoSrc;

    if (hasVideo) {
      if (videoInfoEl) {
        videoInfoEl.textContent = "✓ Video detected on page";
        videoInfoEl.className = "video-info video-found";
      }
      if (verifyBtn) {
        verifyBtn.disabled = !this.settings.verifeedEnabled || this.isAnalyzing;
      }
      this.currentVideo = videoInfo;

      // Clear previous results if different video
      if (isDifferentVideo && resultsSection) {
        resultsSection.classList.remove("show");
        this.resetResultsUI();
        console.log("[VeriFeed Popup] New video detected, cleared results");
      }
    } else {
      if (videoInfoEl) {
        videoInfoEl.textContent = "No video found on current page";
        videoInfoEl.className = "video-info no-video";
      }
      if (verifyBtn) {
        verifyBtn.disabled = true;
      }
      this.currentVideo = null;

      if (resultsSection) {
        resultsSection.classList.remove("show");
      }
    }
  }

  resetResultsUI() {
    // Reset all UI elements to default state
    const confidenceBar = document.getElementById("confidenceBar");
    const realBar = document.getElementById("realBar");
    const fakeBar = document.getElementById("fakeBar");

    if (confidenceBar) confidenceBar.style.width = "0%";
    if (realBar) realBar.style.width = "0%";
    if (fakeBar) fakeBar.style.width = "0%";
  }

  setupEventListeners() {
    // Enable/Disable toggle
    const toggleEnabled = document.getElementById("toggleEnabled");
    if (toggleEnabled) {
      toggleEnabled.checked = this.settings.verifeedEnabled;
      toggleEnabled.addEventListener("change", async (e) => {
        this.settings.verifeedEnabled = e.target.checked;
        const saved = await this.saveSettings();
        if (saved) {
          this.updateUI();
          this.showToast(e.target.checked ? "VeriFeed enabled" : "VeriFeed disabled");
        }
      });
    }

    // Verify button
    const verifyBtn = document.getElementById("verifyBtn");
    if (verifyBtn) {
      verifyBtn.addEventListener("click", () => this.verifyVideo());
    }

    // Minimize button
    const btnMinimize = document.getElementById("btnMinimize");
    if (btnMinimize) {
      btnMinimize.addEventListener("click", () => this.toggleCompactMode());
    }

    // Refresh button with debouncing
    const btnRefresh = document.getElementById("btnRefresh");
    if (btnRefresh) {
      let refreshTimeout;
      btnRefresh.addEventListener("click", async () => {
        clearTimeout(refreshTimeout);
        btnRefresh.disabled = true;
        
        refreshTimeout = setTimeout(async () => {
          await this.refreshAll();
          btnRefresh.disabled = false;
        }, 300);
      });
    }

    // Close button
    const btnClose = document.getElementById("btnClose");
    if (btnClose) {
      btnClose.addEventListener("click", () => {
        this.cleanup();
        window.close();
      });
    }
  }

  async refreshAll() {
    await this.checkServerStatus();
    await this.detectVideo();

    try {
      const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true,
      });
      
      if (tab?.id) {
        await chrome.tabs.sendMessage(tab.id, { action: "refresh" });
      }
    } catch (error) {
      console.log("[VeriFeed Popup] Could not refresh content script:", error);
    }
  }

  toggleCompactMode() {
    this.isCompactMode = !this.isCompactMode;
    document.body.classList.toggle("compact-mode", this.isCompactMode);
    console.log("[VeriFeed Popup] Compact mode:", this.isCompactMode);
  }

  updateUI() {
    const verifyBtn = document.getElementById("verifyBtn");
    const toggleEnabled = document.getElementById("toggleEnabled");
    
    if (verifyBtn) {
      verifyBtn.disabled = !this.settings.verifeedEnabled || 
                           !this.currentVideo || 
                           this.isAnalyzing;
    }
    
    if (toggleEnabled) {
      toggleEnabled.checked = this.settings.verifeedEnabled;
    }
  }

  async checkServerStatus() {
    const statusDot = document.getElementById("statusDot");
    const statusText = document.getElementById("statusText");
    const statusInfo = document.getElementById("statusInfo");

    if (!statusDot || !statusText || !statusInfo) return;

    try {
      statusText.textContent = "Checking...";
      statusDot.className = "status-dot";

      // Cancel previous request if exists
      if (this.abortController) {
        this.abortController.abort();
      }
      this.abortController = new AbortController();

      const startTime = Date.now();
      const response = await fetch(`${this.serverUrl}/health`, {
        method: "GET",
        signal: this.abortController.signal,
        headers: {
          'Content-Type': 'application/json'
        }
      });

      const latency = Date.now() - startTime;
      const data = await response.json();

      if (response.ok && data.status === "healthy") {
        statusDot.className = "status-dot online";
        statusText.textContent = "Online";

        const modelStatus = data.model_loaded ? "✓ Loaded" : "✗ Not loaded";
        const deviceInfo = data.device || "Unknown";

        statusInfo.textContent = `Server ready • Model: ${modelStatus} • Device: ${deviceInfo} • Latency: ${latency}ms`;
        console.log("[VeriFeed Popup] Server healthy:", data);
      } else {
        throw new Error(`Server returned status: ${response.status}`);
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log("[VeriFeed Popup] Server check aborted");
        return;
      }

      console.error("[VeriFeed Popup] Server check failed:", error);

      if (statusDot && statusText && statusInfo) {
        statusDot.className = "status-dot offline";
        statusText.textContent = "Offline";

        let errorMsg = "Cannot connect to analysis server.";
        if (error.message.includes("timeout") || error.name === "TimeoutError") {
          errorMsg = "Connection timeout. Server may be offline.";
        } else if (error.message.includes("Failed to fetch")) {
          errorMsg = "Server not reachable. Ensure backend is running on localhost:5000";
        } else {
          errorMsg = `Server error: ${error.message}`;
        }

        statusInfo.textContent = errorMsg;
      }
    }
  }

  async detectVideo() {
    const videoInfo = document.getElementById("videoInfo");
    const verifyBtn = document.getElementById("verifyBtn");

    if (!videoInfo || !verifyBtn) return;

    try {
      const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true,
      });

      if (!tab?.id || !tab?.url) {
        videoInfo.textContent = "No active tab found";
        videoInfo.className = "video-info no-video";
        verifyBtn.disabled = true;
        return;
      }

      const isFacebookPage = tab.url.includes("facebook.com") || tab.url.includes("fb.com");
      
      if (!isFacebookPage) {
        videoInfo.textContent = "Navigate to Facebook to detect videos";
        videoInfo.className = "video-info no-video";
        verifyBtn.disabled = true;
        return;
      }

      // Check with content script
      const response = await this.sendMessageToTab(tab.id, { action: "checkVideo" });
      this.handleVideoChange(response.hasVideo, response.videoInfo);
      
    } catch (error) {
      console.error("[VeriFeed Popup] Video detection error:", error);
      videoInfo.textContent = "Error detecting video";
      videoInfo.className = "video-info no-video";
      verifyBtn.disabled = true;
    }
  }

  async sendMessageToTab(tabId, message) {
    return new Promise((resolve) => {
      chrome.tabs.sendMessage(tabId, message, (response) => {
        if (chrome.runtime.lastError) {
          console.log("[VeriFeed Popup] Content script not ready:", chrome.runtime.lastError.message);
          resolve({ hasVideo: false });
        } else {
          resolve(response || { hasVideo: false });
        }
      });
    });
  }

  async verifyVideo() {
    if (this.isAnalyzing || !this.currentVideo) return;

    const verifyBtn = document.getElementById("verifyBtn");
    const verifyBtnText = document.getElementById("verifyBtnText");
    const resultsSection = document.getElementById("resultsSection");

    if (!verifyBtn || !verifyBtnText) return;

    this.isAnalyzing = true;
    verifyBtn.disabled = true;
    resultsSection?.classList.remove("show");

    verifyBtnText.textContent = "Extracting";
    verifyBtnText.className = "verifeed-status-text";

    try {
      const [tab] = await chrome.tabs.query({
        active: true,
        currentWindow: true,
      });

      if (!tab?.id) {
        throw new Error("No active tab found");
      }

      // Extract frames
      const framesResponse = await this.sendMessageToTab(tab.id, { action: "extractFrames" });

      if (!framesResponse.success) {
        throw new Error(framesResponse.error || "Frame extraction failed");
      }

      if (!framesResponse.frames?.length) {
        throw new Error("No frames extracted from video");
      }

      console.log(`[VeriFeed Popup] Extracted ${framesResponse.frames.length} frames`);

      // Analyze frames
      verifyBtnText.textContent = "Analyzing";
      const result = await this.analyzeFrames(framesResponse.frames);

      // Store verified video
      if (this.currentVideo?.src) {
        this.lastVerifiedVideoSrc = this.currentVideo.src;
      }

      // Display results
      this.displayResults(result);
      
    } catch (error) {
      console.error("[VeriFeed Popup] Verification error:", error);
      this.showError(error.message);
    } finally {
      this.isAnalyzing = false;
      verifyBtn.disabled = !this.settings.verifeedEnabled || !this.currentVideo;
      verifyBtnText.textContent = "Verify Video";
      verifyBtnText.className = "";
    }
  }

  async analyzeFrames(frames) {
    try {
      console.log(`[VeriFeed Popup] Analyzing ${frames.length} frames...`);

      // Check if verifeedAuth is available
      if (typeof verifeedAuth === 'undefined' || !verifeedAuth.predict) {
        throw new Error("Authentication module not loaded");
      }

      const result = await verifeedAuth.predict(frames);
      console.log("[VeriFeed Popup] Analysis complete:", result);
      
      return result;
    } catch (error) {
      console.error("[VeriFeed Popup] Analysis error:", error);
      throw new Error(`Analysis failed: ${error.message}`);
    }
  }

  displayResults(result) {
    const resultsSection = document.getElementById("resultsSection");
    if (!resultsSection) return;

    const isAuthentic = result.prediction === "REAL";
    const confidence = Math.min(100, Math.max(0, result.confidence || 0));
    const realProbability = Math.min(100, Math.max(0, result.real_probability || 0));
    const fakeProbability = Math.min(100, Math.max(0, result.fake_probability || 0));

    // Apply styling to results section
    this.applyResultStyling(resultsSection, isAuthentic);

    // Update icon and status
    this.updateResultStatus(isAuthentic);
    
    // Update confidence
    this.updateConfidence(confidence, isAuthentic);
    
    // Update probabilities
    this.updateProbabilities(realProbability, fakeProbability);
    
    // Update description
    this.updateDescription(result.prediction, confidence);
    
    // Update metadata
    this.updateMetadata(result);

    // Show results with animation
    setTimeout(() => resultsSection.classList.add("show"), 50);
  }

  applyResultStyling(resultsSection, isAuthentic) {
    // Remove any existing styling classes
    resultsSection.classList.remove('result-authentic', 'result-fake');
    
    // Apply new styling based on result
    if (isAuthentic) {
      resultsSection.classList.add('result-authentic');
    } else {
      resultsSection.classList.add('result-fake');
    }
  }

  updateResultStatus(isAuthentic) {
    const resultIcon = document.getElementById("resultIcon");
    const resultStatus = document.getElementById("resultStatus");

    if (resultIcon && resultStatus) {
      // Style the icon as a circle with background
      resultIcon.style.cssText = `
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 48px;
        height: 48px;
        border-radius: 50%;
        font-size: 24px;
        font-weight: bold;
        color: white;
      `;

      if (isAuthentic) {
        resultIcon.textContent = "✓";
        resultIcon.style.background = "#28a745";
        resultStatus.textContent = "Authentic Video";
      } else {
        resultIcon.textContent = "⚠";
        resultIcon.style.background = "#dc3545";
        resultStatus.textContent = "Deepfake Detected";
      }
      
      resultStatus.className = isAuthentic ? "result-status authentic" : "result-status fake";
    }
  }

  updateConfidence(confidence, isAuthentic) {
    const confidenceValue = document.getElementById("confidenceValue");
    const confidenceBar = document.getElementById("confidenceBar");

    if (confidenceValue && confidenceBar) {
      confidenceValue.textContent = `${confidence.toFixed(1)}%`;
      confidenceBar.className = isAuthentic ? "confidence-bar authentic" : "confidence-bar fake";
      
      requestAnimationFrame(() => {
        confidenceBar.style.width = `${confidence}%`;
      });
    }
  }

  updateProbabilities(realProbability, fakeProbability) {
    const realProb = document.getElementById("realProb");
    const fakeProb = document.getElementById("fakeProb");
    const realBar = document.getElementById("realBar");
    const fakeBar = document.getElementById("fakeBar");

    if (realProb && fakeProb && realBar && fakeBar) {
      realProb.textContent = `${realProbability.toFixed(1)}%`;
      fakeProb.textContent = `${fakeProbability.toFixed(1)}%`;
      
      requestAnimationFrame(() => {
        realBar.style.width = `${realProbability}%`;
        fakeBar.style.width = `${fakeProbability}%`;
      });
    }
  }

  updateDescription(prediction, confidence) {
    const resultDescription = document.getElementById("resultDescription");
    if (resultDescription) {
      resultDescription.textContent = this.generateDescription(prediction, confidence);
    }
  }

  updateMetadata(result) {
    const framesProcessed = document.getElementById("framesProcessed");
    const processingTime = document.getElementById("processingTime");

    if (framesProcessed) {
      framesProcessed.textContent = result.frames_processed || "N/A";
    }
    if (processingTime) {
      const time = result.processing_time ? `${result.processing_time}s` : "N/A";
      processingTime.textContent = time;
    }
  }

  generateDescription(prediction, confidence) {
    const isAuthentic = prediction === "REAL";

    if (isAuthentic) {
      if (confidence >= 80) {
        return "Our analysis strongly indicates this video is genuine and has not been digitally manipulated. However, always verify important content through trusted sources.";
      } else if (confidence >= 60) {
        return "This video appears to be authentic with no AI alterations detected. We recommend verification for complete certainty.";
      } else {
        return "We cannot confidently determine if this video is authentic. Please verify through multiple trusted sources.";
      }
    } else {
      if (confidence >= 80) {
        return "Our analysis strongly indicates this video has been manipulated using artificial intelligence. We recommend verifying this through other sources before sharing.";
      } else if (confidence >= 60) {
        return "This video may have been edited or created using AI tools. Please verify before sharing.";
      } else {
        return "While we cannot be certain, this video shows some signs of manipulation. Treat with skepticism and verify through multiple sources.";
      }
    }
  }

  showError(message) {
    const resultsSection = document.getElementById("resultsSection");
    const resultIcon = document.getElementById("resultIcon");
    const resultStatus = document.getElementById("resultStatus");
    const resultDescription = document.getElementById("resultDescription");
    const confidenceSection = document.querySelector(".confidence-section");
    const probabilityGrid = document.querySelector(".probability-grid");
    const metadata = document.querySelector(".metadata");

    if (!resultsSection) return;

    // Hide unnecessary sections
    [confidenceSection, probabilityGrid, metadata].forEach(el => {
      if (el) el.style.display = "none";
    });

    // Show error
    if (resultIcon) resultIcon.textContent = "❌";
    if (resultStatus) {
      resultStatus.textContent = "Analysis Failed";
      resultStatus.className = "result-status fake";
    }
    if (resultDescription) {
      resultDescription.textContent = `Error: ${message}. Please try again or check if the server is running.`;
    }

    resultsSection.classList.add("show");

    // Restore UI after delay
    setTimeout(() => {
      [confidenceSection, probabilityGrid, metadata].forEach(el => {
        if (el) el.style.display = "";
      });
    }, 5000);
  }

  showInitializationError(error) {
    console.error("[VeriFeed Popup] Initialization failed:", error);
    // Could show an error banner in the UI
  }

  showToast(message, duration = 2000) {
    // Simple toast notification (requires CSS styling)
    console.log(`[VeriFeed Toast] ${message}`);
    // Implementation depends on your HTML structure
  }
}

// Initialize popup when DOM is loaded
document.addEventListener("DOMContentLoaded", () => {
  console.log("[VeriFeed Popup] DOM loaded, initializing...");
  try {
    window.verifeedPopup = new VeriFeedPopup();
  } catch (error) {
    console.error("[VeriFeed Popup] Failed to initialize:", error);
  }
});