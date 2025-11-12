// VeriFeed Popup Script - Enhanced with NLP Integration
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
        return true;
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
    const confidenceBar = document.getElementById("confidenceBar");
    const realBar = document.getElementById("realBar");
    const fakeBar = document.getElementById("fakeBar");
    const confidenceValue = document.getElementById("confidenceValue");
    const realProb = document.getElementById("realProb");
    const fakeProb = document.getElementById("fakeProb");


    // Reset bars to 0 width
    if (confidenceBar) {
      confidenceBar.style.width = "0%";
      confidenceBar.classList.remove('animate');
    }
    if (realBar) {
      realBar.style.width = "0%";
      realBar.classList.remove('animate');
    }
    if (fakeBar) {
      fakeBar.style.width = "0%";
      fakeBar.classList.remove('animate');
    }
   
    // Reset text values
    if (confidenceValue) confidenceValue.textContent = "0%";
    if (realProb) realProb.textContent = "0%";
    if (fakeProb) fakeProb.textContent = "0%";
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


    // Refresh button
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


      // Display results with NLP
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


    // Apply styling
    this.applyResultStyling(resultsSection, isAuthentic);


    // Update icon and status
    this.updateResultStatus(isAuthentic);
   
    // Update confidence
    this.updateConfidence(confidence, isAuthentic);
   
    // Update probabilities
    this.updateProbabilities(realProbability, fakeProbability);
   
    // *** NLP INTEGRATION: Use DeepfakeNLG for description ***
    this.updateDescriptionWithNLP(result.prediction, confidence);
   
    // Update metadata
    this.updateMetadata(result);


    // Show results with animation
    setTimeout(() => resultsSection.classList.add("show"), 50);
  }


  applyResultStyling(resultsSection, isAuthentic) {
    resultsSection.classList.remove('result-authentic', 'result-fake');
   
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
      resultIcon.style.cssText = `
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 40px;
        height: 40px;
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
      // Animate the confidence value from 0 to target
      this.animateValue(confidenceValue, 0, confidence, 1500, '%');
     
      // Set bar color and reset width to 0
      confidenceBar.className = isAuthentic ? "confidence-bar authentic" : "confidence-bar fake";
      confidenceBar.style.width = "0%";
     
      // Trigger animation after a brief delay
      setTimeout(() => {
        confidenceBar.style.width = `${confidence}%`;
      }, 100);
    }
  }


  updateProbabilities(realProbability, fakeProbability) {
    const realProb = document.getElementById("realProb");
    const fakeProb = document.getElementById("fakeProb");
    const realBar = document.getElementById("realBar");
    const fakeBar = document.getElementById("fakeBar");


    if (realProb && fakeProb && realBar && fakeBar) {
      // Animate probability values from 0 to target
      this.animateValue(realProb, 0, realProbability, 1500, '%');
      this.animateValue(fakeProb, 0, fakeProbability, 1500, '%');
     
      // Reset bars to 0 width
      realBar.style.width = "0%";
      realBar.style.background = "linear-gradient(90deg, #10b981, #059669)";
      realBar.style.transition = "width 1.5s cubic-bezier(0.4, 0.0, 0.2, 1)";
     
      fakeBar.style.width = "0%";
      fakeBar.style.background = "linear-gradient(90deg, #ef4444, #dc2626)";
      fakeBar.style.transition = "width 1.5s cubic-bezier(0.4, 0.0, 0.2, 1)";
     
      // Trigger animations with slight delay for visual effect
      setTimeout(() => {
        realBar.classList.add('animate');
        realBar.style.width = `${realProbability}%`;
      }, 200);
     
      setTimeout(() => {
        fakeBar.classList.add('animate');
        fakeBar.style.width = `${fakeProbability}%`;
      }, 300);
    }
  }


  /**
   * *** NLP INTEGRATION ***
   * Uses DeepfakeNLG to generate natural language descriptions
   */
  updateDescriptionWithNLP(prediction, confidence) {
    const resultDescription = document.getElementById("resultDescription");
    if (!resultDescription) return;


    // Check if NLG module is loaded
    if (typeof deepfakeNLG === 'undefined') {
      console.warn("[VeriFeed Popup] NLG module not loaded, using fallback");
      resultDescription.textContent = this.generateDescriptionFallback(prediction, confidence);
      return;
    }


    try {
      // Generate NLP-powered description
      const nlgMessage = deepfakeNLG.generate(prediction, confidence);
      const nlgConfidenceText = deepfakeNLG.generateConfidenceText(confidence);


      // Apply grammar correction
      const finalMessage = deepfakeNLG.correctGrammar(`${nlgMessage} ${nlgConfidenceText}.`);


      resultDescription.textContent = finalMessage;
      console.log("[VeriFeed Popup] NLP description generated:", finalMessage);
    } catch (error) {
      console.error("[VeriFeed Popup] NLP generation error:", error);
      resultDescription.textContent = this.generateDescriptionFallback(prediction, confidence);
    }
  }


  /**
   * Fallback description generator (if NLP module fails to load)
   */
  generateDescriptionFallback(prediction, confidence) {
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


  /**
   * Animates a numeric value from start to end
   * @param {HTMLElement} element - The element to update
   * @param {number} start - Starting value
   * @param {number} end - Ending value
   * @param {number} duration - Animation duration in ms
   * @param {string} suffix - Suffix to append (e.g., '%')
   */
  animateValue(element, start, end, duration, suffix = '') {
    if (!element) return;
   
    const startTime = performance.now();
    const range = end - start;
   
    const updateValue = (currentTime) => {
      const elapsed = currentTime - startTime;
      const progress = Math.min(elapsed / duration, 1);
     
      // Easing function for smooth animation
      const easeOutQuart = 1 - Math.pow(1 - progress, 4);
      const current = start + (range * easeOutQuart);
     
      element.textContent = `${current.toFixed(1)}${suffix}`;
     
      if (progress < 1) {
        requestAnimationFrame(updateValue);
      } else {
        element.textContent = `${end.toFixed(1)}${suffix}`;
      }
    };
   
    requestAnimationFrame(updateValue);
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
  }


  showToast(message, duration = 2000) {
    console.log(`[VeriFeed Toast] ${message}`);
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

