// VeriFeed Content Script - Enhanced Debug Version
// Added extensive logging to identify popup display issues

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

    this.init();
  }

  init() {
    console.log("VeriFeed initialized - professional design");
    this.loadSettings();
    this.setupMutationObserver();
    this.scanForVideos();
    console.log("Initial scan for videos triggered");
  }

  loadSettings() {
    chrome.storage.local.get(["verifeedEnabled"], (result) => {
      this.isEnabled = result.verifeedEnabled !== false;
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
        console.log(
          `No video post container found for video #${index}, skipping`
        );
        return;
      }

      if (container.querySelector(".verifeed-verify-btn")) {
        console.log(
          `Verify button already exists in container for video #${index}, skipping`
        );
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
      const relativeRight =
        targetRect.right - menuRect.right + menuRect.width + 8;
      buttonPosition = `${relativeRight}px`;
      console.log(
        `Found menu button in post header, positioning VeriFeed button at ${buttonPosition} from right`
      );
    } else {
      console.log(
        "Menu button not found in post header, using fallback positioning"
      );
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
      verifyBtn.style.background =
        "linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)";
      verifyBtn.style.transform = "translateY(-1px)";
    };
    verifyBtn.onmouseleave = () => {
      verifyBtn.style.background =
        "linear-gradient(135deg, #667eea 0%, #764ba2 100%)";
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
        const relativeRight =
          targetRect.right - menuRect.right + menuRect.width + 8;
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
    console.log("Button element:", buttonElement);
    console.log("Container:", container);
    console.log("Video element:", videoElement);

    if (buttonElement.dataset.analyzing === "true") {
      console.log("Already analyzing this video, ignoring click");
      return;
    }
   
    buttonElement.dataset.analyzing = "true";

    const originalContent = buttonElement.innerHTML;
    buttonElement.innerHTML = `
          <div style="width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 4px;"></div>
          <span>Checking...</span>
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
      console.log("=== CHECKING SERVER HEALTH ===");
      const healthResponse = await this.makeRequest(
        `${this.serverUrl}/health`,
        "GET"
      );
      console.log("Health response status:", healthResponse.ok);
      
      if (!healthResponse.ok) {
        throw new Error("Server offline");
      }
      const healthData = await healthResponse.json();
      console.log("Server health data:", healthData);

      if (healthData.status !== "healthy") {
        throw new Error("Server not in healthy state");
      }
    } catch (error) {
      console.error("=== SERVER HEALTH CHECK FAILED ===");
      console.error("Error:", error);
      this.restorePageState(scrollY, originalVideoState, videoElement);
      this.showErrorPopup(
        buttonElement,
        "Cannot connect to video checker. Please try again later."
      );
      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;
      delete buttonElement.dataset.analyzing;
      return;
    }

    try {
      let frames = this.cachedFrames.get(videoElement);

      if (!frames) {
        console.log("=== EXTRACTING FRAMES ===");
        frames = await this.extractFrames(videoElement, 100);
        if (!frames || frames.length === 0) {
          throw new Error("Could not extract frames from video");
        }
        console.log(`Successfully extracted ${frames.length} frames`);
        this.cachedFrames.set(videoElement, frames);
      } else {
        console.log("Using cached frames for this video");
      }

      const requestData = {
        frames: frames,
        platform: "facebook",
      };

      console.log("=== SENDING ANALYSIS REQUEST ===");
      console.log("Request data:", {
        frameCount: frames.length,
        platform: requestData.platform,
        firstFramePreview: frames[0].substring(0, 50) + "..."
      });
      
      const response = await this.makeRequest(
        `${this.serverUrl}/frame_analyze`,
        "POST",
        requestData
      );
      console.log("Analysis response status:", response.ok);
      console.log("Analysis response status code:", response.status);

      // CRITICAL: Handle both success and error responses properly
      let analysisData;
      try {
        analysisData = await response.json();
        console.log("=== RESPONSE DATA RECEIVED ===");
        console.log("Full response data:", analysisData);
      } catch (jsonError) {
        console.error("Failed to parse JSON response:", jsonError);
        throw new Error("Server returned invalid response");
      }

      if (!response.ok) {
        console.error("=== SERVER RETURNED ERROR ===");
        console.error("Error data:", analysisData);
        
        // Extract meaningful error message
        let errorMsg = analysisData.error || analysisData.message || "Analysis failed";
        
        // Handle specific backend errors
        if (analysisData.error && analysisData.error.includes("No recognizable faces")) {
          errorMsg = "No faces detected in video. Please try a video with visible faces.";
        } else if (analysisData.error && analysisData.error.includes("Invalid frame count")) {
          errorMsg = "Video length not supported. Please try a different video.";
        }
        
        throw new Error(errorMsg);
      }

      console.log("=== ANALYSIS SUCCESSFUL ===");
      console.log("Prediction:", analysisData.prediction);
      console.log("Confidence:", analysisData.confidence);

      // Validate response has required fields
      if (!analysisData.prediction || analysisData.confidence === undefined) {
        console.error("=== INVALID RESPONSE STRUCTURE ===");
        console.error("Missing required fields in response");
        throw new Error("Invalid response from server");
      }

      this.restorePageState(scrollY, originalVideoState, videoElement);

      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;
      delete buttonElement.dataset.analyzing;

      console.log("=== CALLING showResultsPopup ===");
      console.log("Passing to popup - prediction:", analysisData.prediction, "confidence:", analysisData.confidence);
      
      // CRITICAL: Force a small delay to ensure DOM is ready
      setTimeout(() => {
        this.showResultsPopup(buttonElement, analysisData);
      }, 100);
      
    } catch (error) {
      console.error("=== VIDEO VERIFICATION ERROR ===");
      console.error("Error:", error);
      console.error("Error message:", error.message);
      console.error("Stack trace:", error.stack);
      this.restorePageState(scrollY, originalVideoState, videoElement);
      this.showErrorPopup(buttonElement, error.message || "Check failed");
      buttonElement.innerHTML = originalContent;
      buttonElement.disabled = false;
      delete buttonElement.dataset.analyzing;
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
      videoElement
        .play()
        .catch((err) => console.log("Could not resume video playback:", err));
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
        console.log(
          `Request failed, retrying... (${retries + 1}/${this.maxRetries})`
        );
        await new Promise((resolve) =>
          setTimeout(resolve, this.retryDelay * (retries + 1))
        );
        return this.makeRequest(url, method, data, retries + 1);
      }
      throw error;
    }
  }

  async extractFrames(videoElement, numFrames = 100) {
    return new Promise((resolve, reject) => {
      try {
        console.log(`Starting frame extraction - target: ${numFrames} frames`);

        const canvas = document.createElement("canvas");
        const ctx = canvas.getContext("2d");
        canvas.width = 112;
        canvas.height = 112;

        const frames = [];
        const duration = videoElement.duration;

        if (!duration || duration <= 0) {
          reject(new Error("Video duration not available"));
          return;
        }

        console.log(
          `Video duration: ${duration}s, extracting ${numFrames} frames`
        );

        let currentFrame = 0;
        const interval = duration / numFrames;
        let frameExtractionStart = Date.now();

        const extractNextFrame = () => {
          if (currentFrame >= numFrames) {
            const extractionTime = Date.now() - frameExtractionStart;
            console.log(
              `Frame extraction complete: ${frames.length} frames in ${extractionTime}ms`
            );
            resolve(frames);
            return;
          }

          const timeToSeek = currentFrame * interval;
          videoElement.currentTime = timeToSeek;

          const onSeeked = () => {
            videoElement.removeEventListener("seeked", onSeeked);

            try {
              ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
              const dataURL = canvas.toDataURL("image/png");
              const base64Data = dataURL.split(",")[1];
              frames.push(base64Data);

              if (currentFrame % 10 === 0) {
                console.log(`Extracted frame ${currentFrame}/${numFrames}`);
              }

              currentFrame++;
              setTimeout(extractNextFrame, 50);
            } catch (error) {
              reject(
                new Error(
                  `Failed to extract frame ${currentFrame}: ${error.message}`
                )
              );
            }
          };

          const onError = () => {
            videoElement.removeEventListener("error", onError);
            reject(new Error(`Video seek error at frame ${currentFrame}`));
          };

          videoElement.addEventListener("seeked", onSeeked);
          videoElement.addEventListener("error", onError);

          setTimeout(() => {
            videoElement.removeEventListener("seeked", onSeeked);
            videoElement.removeEventListener("error", onError);
            if (currentFrame < numFrames) {
              console.warn(
                `Seek timeout for frame ${currentFrame}, continuing...`
              );
              currentFrame++;
              setTimeout(extractNextFrame, 100);
            }
          }, 2000);
        };

        extractNextFrame();
      } catch (error) {
        reject(new Error(`Frame extraction setup failed: ${error.message}`));
      }
    });
  }

  showResultsPopup(buttonElement, result) {
    console.log("=== SHOW RESULTS POPUP CALLED ===");
    console.log("Function entry point reached");
    console.log("buttonElement:", buttonElement);
    console.log("result:", result);
    console.log("buttonElement exists:", !!buttonElement);
    console.log("result exists:", !!result);
    
    try {
      console.log("Attempting to remove existing popup");
      this.removeExistingPopup();
      console.log("Existing popup removed");

      const prediction = result.prediction;
      const confidence = result.confidence || 0;
      const isAuthentic = prediction === "REAL";
      console.log("Parsed result - prediction:", prediction, "confidence:", confidence, "isAuthentic:", isAuthentic);

      console.log("Getting button rect");
      const buttonRect = buttonElement.getBoundingClientRect();
      console.log("Button rect:", {
        top: buttonRect.top,
        right: buttonRect.right,
        bottom: buttonRect.bottom,
        left: buttonRect.left,
        width: buttonRect.width,
        height: buttonRect.height
      });

      console.log("Creating results popup element");
      const resultsPopup = document.createElement("div");
      resultsPopup.className = "verifeed-results-popup";
      console.log("Popup element created:", resultsPopup);

      const statusIcon = isAuthentic ? "✅" : "⚠️";
      const statusText = isAuthentic ? "Authentic" : "Deepfake Detected";
      const statusColor = isAuthentic ? "#10b981" : "#f59e0b";
      const confidenceText =
        confidence > 80
          ? "We are very confident"
          : confidence > 60
          ? "We are somewhat confident"
          : "We are not very confident";

      console.log("Setting popup innerHTML");
      resultsPopup.innerHTML = `
            <div class="verifeed-popup-content">
                <div class="verifeed-popup-header">
                    <span class="status-icon">${statusIcon}</span>
                    <span class="status-text">${statusText}</span>
                    <button class="close-btn">×</button>
                </div>
                <div class="verifeed-popup-body">
                    <div class="confidence-section">
                        <span class="confidence-label">How sure we are: ${confidence.toFixed(
                          1
                        )}%</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%; background: ${statusColor};"></div>
                        </div>
                        <span class="confidence-text">${confidenceText}</span>
                    </div>
                    <div class="info-text">
                        ${
                          isAuthentic
                            ? "This video is genuine and has not been edited by artificial intelligence."
                            : "This video may have been edit or created by artificial intelligence. Please verify it before sharing."
                        }
                    </div>
                    <div class="disclaimer">
                        Computer check • This is just a guess • Always check with other sources
                    </div>
                </div>
            </div>
        `;
      console.log("Popup innerHTML set with confidence:", confidence.toFixed(1));

      const topPosition = buttonRect.bottom + 8;
      const rightPosition = window.innerWidth - buttonRect.right;
      console.log("Calculated positions - top:", topPosition, "right:", rightPosition);

      console.log("Setting popup styles");
      resultsPopup.style.cssText = `
            all: initial !important;
            position: fixed !important;
            top: ${topPosition}px !important;
            right: ${rightPosition}px !important;
            z-index: 2147483647 !important;
            width: 280px !important;
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
            filter: none !important;
            clip: auto !important;
            clip-path: none !important;
            mask: none !important;
            isolation: isolate !important;
            min-height: 100px !important;
        `;
      console.log("Popup styles set");

      console.log("Creating style element");
      const style = document.createElement("style");
      style.id = "verifeed-popup-styles";
      style.textContent = `
            @keyframes slideDown {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
            }
            .verifeed-results-popup {
                pointer-events: auto !important;
            }
            .verifeed-popup-content {
                padding: 0 !important;
            }
            .verifeed-popup-header {
                display: flex !important;
                align-items: center !important;
                padding: 12px 16px !important;
                border-bottom: 1px solid #f3f4f6 !important;
                background: #fafafa !important;
                border-radius: 8px 8px 0 0 !important;
            }
            .verifeed-popup-header .status-icon {
                font-size: 16px !important;
                margin-right: 8px !important;
            }
            .verifeed-popup-header .status-text {
                font-weight: 600 !important;
                color: #374151 !important;
                font-size: 14px !important;
                flex: 1 !important;
            }
            .verifeed-popup-header .close-btn {
                background: none !important;
                border: none !important;
                color: #9ca3af !important;
                font-size: 18px !important;
                cursor: pointer !important;
                padding: 0 !important;
                width: 20px !important;
                height: 20px !important;
                display: flex !important;
                align-items: center !important;
                justify-content: center !important;
            }
            .verifeed-popup-header .close-btn:hover {
                color: #6b7280 !important;
            }
            .verifeed-popup-body {
                padding: 16px !important;
            }
            .verifeed-popup-body .confidence-section {
                margin-bottom: 12px !important;
            }
            .verifeed-popup-body .confidence-label {
                font-size: 13px !important;
                font-weight: 600 !important;
                color: #374151 !important;
                display: block !important;
                margin-bottom: 6px !important;
            }
            .verifeed-popup-body .confidence-bar {
                width: 100% !important;
                height: 6px !important;
                background: #e5e7eb !important;
                border-radius: 3px !important;
                overflow: hidden !important;
                margin-bottom: 4px !important;
            }
            .verifeed-popup-body .confidence-fill {
                height: 100% !important;
                border-radius: 3px !important;
                transition: width 0.8s ease-out !important;
            }
            .verifeed-popup-body .confidence-text {
                font-size: 12px !important;
                color: #6b7280 !important;
            }
            .verifeed-popup-body .info-text {
                font-size: 13px !important;
                color: #4b5563 !important;
                line-height: 1.4 !important;
                margin-bottom: 12px !important;
            }
            .verifeed-popup-body .disclaimer {
                font-size: 11px !important;
                color: #9ca3af !important;
                text-align: center !important;
                line-height: 1.3 !important;
                padding-top: 8px !important;
                border-top: 1px solid #f3f4f6 !important;
            }
        `;
      console.log("Style element created and content set");

      console.log("Appending style to head");
      document.head.appendChild(style);
      this.activeStyle = style;
      console.log("Style appended to head");

      console.log("Appending popup to body");
      document.body.appendChild(resultsPopup);
      this.activePopup = resultsPopup;
      console.log("Popup appended to body");

      console.log("=== POPUP DOM CHECK ===");
      console.log("Popup in DOM:", document.body.contains(resultsPopup));
      console.log("Popup parent:", resultsPopup.parentNode);
      const computedStyle = window.getComputedStyle(resultsPopup);
      console.log("Computed styles:");
      console.log("  display:", computedStyle.display);
      console.log("  visibility:", computedStyle.visibility);
      console.log("  opacity:", computedStyle.opacity);
      console.log("  z-index:", computedStyle.zIndex);
      console.log("  position:", computedStyle.position);
      console.log("  top:", computedStyle.top);
      console.log("  right:", computedStyle.right);
      console.log("  width:", computedStyle.width);
      console.log("  height:", computedStyle.height);
      console.log("  background:", computedStyle.background);

      console.log("Setting up close button");
      const closeBtn = resultsPopup.querySelector(".close-btn");
      console.log("Close button found:", !!closeBtn);
      
      const closePopup = () => {
        console.log("Closing popup");
        
        if (this.scrollListener) {
          window.removeEventListener("scroll", this.scrollListener);
          this.scrollListener = null;
        }
        if (this.clickListener) {
          document.removeEventListener("click", this.clickListener);
          this.clickListener = null;
        }
        
        if (resultsPopup.parentNode) {
          resultsPopup.remove();
        }
        if (style.parentNode) {
          style.remove();
        }
        
        this.activePopup = null;
        this.activeStyle = null;
      };

      this.scrollListener = () => {
        console.log("Scroll detected, closing popup");
        closePopup();
      };

      closeBtn.addEventListener("click", (e) => {
        console.log("Close button clicked");
        e.stopPropagation();
        e.preventDefault();
        closePopup();
      });

      console.log("Setting up auto-close timer (15s)");
      setTimeout(() => {
        if (resultsPopup.parentNode) {
          console.log("Auto-closing popup after 15s");
          closePopup();
        }
      }, 15000);

      console.log("Adding scroll listener");
      window.addEventListener("scroll", this.scrollListener, { passive: true });

      console.log("Setting up click-outside listener");
      setTimeout(() => {
        this.clickListener = (e) => {
          console.log("Click detected:", e.target);
          if (
            !resultsPopup.contains(e.target) &&
            !buttonElement.contains(e.target)
          ) {
            console.log("Click outside popup, closing");
            closePopup();
          } else {
            console.log("Click inside popup or button, keeping open");
          }
        };
        document.addEventListener("click", this.clickListener);
        console.log("Click-outside listener added");
      }, 100);

      console.log("=== POPUP SETUP COMPLETE ===");
      console.log("Popup should be visible now");
    } catch (error) {
      console.error("=== ERROR IN showResultsPopup ===");
      console.error("Error message:", error.message);
      console.error("Error stack:", error.stack);
      console.error("Full error:", error);
    }
  }

  showErrorPopup(buttonElement, message) {
    console.log("=== SHOWING ERROR POPUP ===");
    console.log("Message:", message);
    console.log("Button element:", buttonElement);
    
    this.removeExistingPopup();

    const buttonRect = buttonElement.getBoundingClientRect();
    console.log("Button rect:", buttonRect);

    const errorPopup = document.createElement("div");
    errorPopup.className = "verifeed-error-popup";
    errorPopup.innerHTML = `
            <div class="error-content">
                <div class="error-header">
                    <span>⚠️ Cannot check video</span>
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
            animation: slideDown 0.2s ease-out !important;
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        `;

    const errorStyle = document.createElement("style");
    errorStyle.id = "verifeed-error-styles";
    errorStyle.textContent = `
            .verifeed-error-popup {
                pointer-events: auto !important;
            }
            .verifeed-error-popup .error-content {
                padding: 0 !important;
            }
            .verifeed-error-popup .error-header {
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                padding: 12px 16px !important;
                background: #fef2f2 !important;
                border-radius: 8px 8px 0 0 !important;
                font-weight: 600 !important;
                color: #b91c1c !important;
                font-size: 14px !important;
            }
            .verifeed-error-popup .error-header .close-btn {
                background: none !important;
                border: none !important;
                color: #9ca3af !important;
                font-size: 18px !important;
                cursor: pointer !important;
                padding: 0 !important;
                width: 20px !important;
                height: 20px !important;
            }
            .verifeed-error-popup .error-header .close-btn:hover {
                color: #6b7280 !important;
            }
            .verifeed-error-popup .error-body {
                padding: 16px !important;
            }
            .verifeed-error-popup .error-body p {
                margin: 0 0 12px 0 !important;
                font-size: 13px !important;
                color: #6b7280 !important;
                line-height: 1.4 !important;
            }
            .verifeed-error-popup .retry-btn {
                background: #1877f2 !important;
                color: white !important;
                border: none !important;
                padding: 6px 12px !important;
                border-radius: 4px !important;
                font-size: 12px !important;
                cursor: pointer !important;
                font-weight: 500 !important;
            }
            .verifeed-error-popup .retry-btn:hover {
                background: #166fe5 !important;
            }
        `;
    document.head.appendChild(errorStyle);
    this.activeStyle = errorStyle;
    console.log("Error style added to head");

    document.body.appendChild(errorPopup);
    this.activePopup = errorPopup;
    console.log("Error popup appended to body");
    console.log("Error popup in DOM:", document.body.contains(errorPopup));

    const closeBtn = errorPopup.querySelector(".close-btn");
    const retryBtn = errorPopup.querySelector(".retry-btn");

    const closeErrorPopup = () => {
      console.log("Closing error popup");
      if (errorPopup.parentNode) {
        errorPopup.remove();
      }
      if (errorStyle.parentNode) {
        errorStyle.remove();
      }
      this.activePopup = null;
      this.activeStyle = null;
    };

    closeBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      closeErrorPopup();
    });

    retryBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      e.preventDefault();
      closeErrorPopup();
    });

    setTimeout(() => {
      if (errorPopup.parentNode) {
        closeErrorPopup();
      }
    }, 8000);
  }

  removeExistingPopup() {
    console.log("=== REMOVING EXISTING POPUP ===");
    
    if (this.scrollListener) {
      console.log("Removing scroll listener");
      window.removeEventListener("scroll", this.scrollListener);
      this.scrollListener = null;
    }
    if (this.clickListener) {
      console.log("Removing click listener");
      document.removeEventListener("click", this.clickListener);
      this.clickListener = null;
    }

    const existingPopups = document.querySelectorAll(
      ".verifeed-results-popup, .verifeed-error-popup"
    );
    console.log("Found existing popups:", existingPopups.length);
    existingPopups.forEach((popup, index) => {
      console.log(`Removing popup ${index + 1}`);
      if (popup.parentNode) {
        popup.remove();
      }
    });

    const existingStyles = document.querySelectorAll(
      "#verifeed-popup-styles, #verifeed-error-styles"
    );
    console.log("Found existing styles:", existingStyles.length);
    existingStyles.forEach((style, index) => {
      console.log(`Removing style ${index + 1}`);
      if (style.parentNode) {
        style.remove();
      }
    });

    this.activePopup = null;
    this.activeStyle = null;
    console.log("Existing popup removal complete");
  }

  destroy() {
    console.log("Destroying VeriFeed detector");
    if (this.observer) {
      this.observer.disconnect();
    }
    this.removeExistingPopup();
    this.analyzedVideos.clear();
    console.log("VeriFeed detector destroyed");
  }
}

// Initialize VeriFeed when page loads
let veriFeedInstance = null;

function initializeVeriFeed() {
  if (window.location.hostname.includes("facebook.com") && !veriFeedInstance) {
    console.log("Initializing VeriFeed for Facebook...");
    veriFeedInstance = new VeriFeedDetector();
  }
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log("Received message:", request);

  if (request.action === "toggleVeriFeed") {
    if (veriFeedInstance) {
      veriFeedInstance.isEnabled = request.enabled;
      console.log(`VeriFeed ${request.enabled ? "enabled" : "disabled"}`);
      if (!request.enabled) {
        veriFeedInstance.destroy();
        veriFeedInstance = null;
      }
    } else if (request.enabled) {
      initializeVeriFeed();
    }
    sendResponse({ success: true, enabled: request.enabled });
  }

  if (request.action === "analyzeSpecificVideo") {
    if (veriFeedInstance && veriFeedInstance.isEnabled) {
      const videoElement = document.querySelector(request.videoSelector);
      if (videoElement) {
        const container = videoElement.closest(
          '[role="article"], [data-pagelet*="video"]'
        );
        if (container) {
          console.log("Analyzing specific video...");
          veriFeedInstance.handleVerifyClick(container, videoElement);
          sendResponse({ success: true, message: "Analysis started" });
        } else {
          sendResponse({ success: false, error: "Container not found" });
        }
      } else {
        sendResponse({ success: false, error: "Video element not found" });
      }
    } else {
      sendResponse({
        success: false,
        error: "VeriFeed not enabled or not initialized",
      });
    }
  }

  if (request.action === "getStatus") {
    sendResponse({
      enabled: veriFeedInstance ? veriFeedInstance.isEnabled : false,
      initialized: !!veriFeedInstance,
      videoCount: veriFeedInstance ? veriFeedInstance.analyzedVideos.size : 0,
      serverUrl: veriFeedInstance
        ? veriFeedInstance.serverUrl
        : "http://localhost:5000",
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

// Re-scan periodically for new content
setInterval(() => {
  if (veriFeedInstance && veriFeedInstance.isEnabled) {
    veriFeedInstance.scanForVideos();
  }
}, 3000);

// Enhanced button positioning fix
function fixVeriFeedButtonPositioning() {
  const veriFeedButtons = document.querySelectorAll(".verifeed-verify-btn");
  console.log(`Fixing positioning for ${veriFeedButtons.length} buttons`);

  veriFeedButtons.forEach((button, index) => {
    const postContainer = button.closest(
      '[role="article"], [data-pagelet*="video"], [data-pagelet*="FeedUnit"]'
    );
    if (!postContainer) return;

    const postHeader = postContainer
      .querySelector('h3, h4, [data-ad-preview="message"]')
      ?.closest("div");
    const targetContainer = postHeader || postContainer;

    const menuButton = targetContainer.querySelector(
      '[aria-label*="more"], [aria-label*="options"], [aria-label*="menu"]'
    );

    if (menuButton && targetContainer.contains(button)) {
      const menuRect = menuButton.getBoundingClientRect();
      const targetRect = targetContainer.getBoundingClientRect();
      const relativeRight =
        targetRect.right - menuRect.right + menuRect.width + 8;

      button.style.position = "absolute";
      button.style.top = "12px";
      button.style.right = `${relativeRight}px`;
      button.style.left = "auto";
      button.style.zIndex = "2147483647";

      console.log(`Fixed button ${index + 1} position`);
    }
  });
}

fixVeriFeedButtonPositioning();
setTimeout(fixVeriFeedButtonPositioning, 1000);

const positioningObserver = new MutationObserver((mutations) => {
  let shouldFix = false;
  mutations.forEach((mutation) => {
    if (mutation.addedNodes.length || mutation.removedNodes.length) {
      shouldFix = true;
    }
  });
  if (shouldFix) {
    setTimeout(fixVeriFeedButtonPositioning, 500);
  }
});

positioningObserver.observe(document.body, {
  childList: true,
  subtree: true,
});

let currentUrl = window.location.href;
const urlObserver = new MutationObserver(() => {
  if (window.location.href !== currentUrl) {
    currentUrl = window.location.href;
    console.log("URL changed, reinitializing VeriFeed...");
    setTimeout(() => {
      if (veriFeedInstance && veriFeedInstance.isEnabled) {
        veriFeedInstance.scanForVideos();
      }
    }, 1000);
  }
});

urlObserver.observe(document.body, {
  childList: true,
  subtree: true,
});

window.addEventListener("error", (event) => {
  if (
    event.error &&
    event.error.message &&
    event.error.message.includes("verifeed")
  ) {
    console.error("VeriFeed error:", event.error);
  }
});

console.log(
  "VeriFeed content script fully loaded with enhanced debug logging"
);

// DEBUG: Enhanced test function
window.testVeriFeedPopup = function() {
  console.log("=== TESTING VERIFEED POPUP ===");
  console.log("veriFeedInstance exists:", !!veriFeedInstance);
  
  const button = document.querySelector(".verifeed-verify-btn");
  console.log("Button found:", !!button);
  
  if (!button) {
    console.log("ERROR: No VeriFeed button found on page");
    console.log("Available buttons:", document.querySelectorAll("button").length);
    return;
  }
  
  console.log("Button element:", button);
  console.log("Button rect:", button.getBoundingClientRect());
  
  const testResult = {
    prediction: "REAL",
    confidence: 85
  };
  
  console.log("Test result:", testResult);
  
  if (veriFeedInstance) {
    console.log("Calling showResultsPopup...");
    try {
      veriFeedInstance.showResultsPopup(button, testResult);
      console.log("showResultsPopup call completed");
      
      // Check if popup was created
      setTimeout(() => {
        const popup = document.querySelector(".verifeed-results-popup");
        console.log("Popup exists after call:", !!popup);
        if (popup) {
          console.log("Popup element:", popup);
          console.log("Popup computed style:", window.getComputedStyle(popup));
        } else {
          console.log("ERROR: Popup not found in DOM after showResultsPopup call");
          console.log("All elements with verifeed class:", document.querySelectorAll("[class*='verifeed']"));
        }
      }, 500);
    } catch (error) {
      console.error("ERROR calling showResultsPopup:", error);
      console.error("Error stack:", error.stack);
    }
  } else {
    console.log("ERROR: No veriFeedInstance found");
  }
};

console.log("=== DEBUG COMMANDS AVAILABLE ===");
console.log("Run 'testVeriFeedPopup()' to test popup display");
console.log("Run 'veriFeedInstance' to inspect the instance");