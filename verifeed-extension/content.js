// VeriFeed Content Script - Enhanced Debug Version
// Added extensive logging to identify popup display issues

class VeriFeedDetector {
  constructor() {
    this.analyzedVideos = new Map();
    this.cachedFrames = new WeakMap();
    this.serverUrl = "http://localhost:5000";
    this.isEnabled = true;
    this.observer = null;
    this.maxRetries = 3;
    this.retryDelay = 1000;
    // Popup methods are now handled directly in this class

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
      this.updateUI();
    });
  }

  updateSettings(newSettings) {
    this.isEnabled = newSettings.verifeedEnabled !== false;
    this.updateUI();
  }

  updateUI() {
    if (this.isEnabled) {
      this.scanForVideos();
    } else {
      this.removeAllButtons();
    }
  }

  removeAllButtons() {
    const buttons = document.querySelectorAll('.verifeed-verify-btn');
    buttons.forEach(button => button.remove());
    this.analyzedVideos.clear();
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

      // Ultra-comprehensive exclusion logic for Facebook stories and MyDay
      const isStory = container.closest('[data-pagelet*="story"]') ||
                     container.closest('[data-pagelet*="Stories"]') ||
                     container.closest('[data-pagelet*="stories"]') ||
                     container.closest('[aria-label*="story"]') ||
                     container.closest('[aria-label*="Stories"]') ||
                     container.closest('[aria-label*="stories"]') ||
                     container.closest('.story') ||
                     container.closest('[class*="story"]') ||
                     container.closest('[data-visualcompletion*="story"]') ||
                     container.closest('[data-visualcompletion*="Stories"]') ||
                     container.closest('[data-visualcompletion*="stories"]') ||
                     container.getAttribute('data-pagelet')?.includes('story') ||
                     container.getAttribute('data-pagelet')?.includes('Stories') ||
                     container.getAttribute('data-pagelet')?.includes('stories') ||
                     container.classList?.contains('story') ||
                     container.classList?.contains('Stories') ||
                     container.classList?.contains('stories') ||
                     container.closest('[role*="story"]') ||
                     container.closest('[data-testid*="story"]') ||
                     container.closest('[data-testid*="Stories"]') ||
                     container.closest('[data-testid*="stories"]') ||
                     // Check parent containers for story indicators
                     container.closest('[data-pagelet*="feed"]')?.querySelector('[data-pagelet*="story"]') ||
                     container.closest('[data-pagelet*="timeline"]')?.querySelector('[data-pagelet*="story"]') ||
                     // Additional story patterns
                     container.closest('[data-pagelet*="Story"]') ||
                     container.closest('[aria-label*="Story"]') ||
                     container.closest('[class*="Story"]') ||
                     container.closest('[data-visualcompletion*="Story"]') ||
                     container.getAttribute('data-pagelet')?.includes('Story') ||
                     container.classList?.contains('Story') ||
                     container.closest('[role*="Story"]') ||
                     container.closest('[data-testid*="Story"]');

      const isMyDay = container.closest('[data-pagelet*="myday"]') ||
                      container.closest('[data-pagelet*="MyDay"]') ||
                      container.closest('[data-pagelet*="My Day"]') ||
                      container.closest('[data-pagelet*="my day"]') ||
                      container.closest('[aria-label*="myday"]') ||
                      container.closest('[aria-label*="MyDay"]') ||
                      container.closest('[aria-label*="My Day"]') ||
                      container.closest('[aria-label*="my day"]') ||
                      container.closest('.myday') ||
                      container.closest('.MyDay') ||
                      container.closest('[class*="myday"]') ||
                      container.closest('[class*="MyDay"]') ||
                      container.closest('[class*="my-day"]') ||
                      container.closest('[data-visualcompletion*="myday"]') ||
                      container.closest('[data-visualcompletion*="MyDay"]') ||
                      container.closest('[data-visualcompletion*="My Day"]') ||
                      container.closest('[data-visualcompletion*="my day"]') ||
                      container.getAttribute('data-pagelet')?.includes('myday') ||
                      container.getAttribute('data-pagelet')?.includes('MyDay') ||
                      container.getAttribute('data-pagelet')?.includes('My Day') ||
                      container.getAttribute('data-pagelet')?.includes('my day') ||
                      container.classList?.contains('myday') ||
                      container.classList?.contains('MyDay') ||
                      container.classList?.contains('my-day') ||
                      container.closest('[role*="myday"]') ||
                      container.closest('[role*="MyDay"]') ||
                      container.closest('[data-testid*="myday"]') ||
                      container.closest('[data-testid*="MyDay"]') ||
                      container.closest('[data-testid*="my-day"]') ||
                      // Check parent containers for MyDay indicators
                      container.closest('[data-pagelet*="feed"]')?.querySelector('[data-pagelet*="myday"]') ||
                      container.closest('[data-pagelet*="timeline"]')?.querySelector('[data-pagelet*="myday"]') ||
                      // Additional MyDay patterns
                      container.closest('[data-pagelet*="Myday"]') ||
                      container.closest('[aria-label*="Myday"]') ||
                      container.closest('[class*="Myday"]') ||
                      container.closest('[data-visualcompletion*="Myday"]') ||
                      container.getAttribute('data-pagelet')?.includes('Myday') ||
                      container.classList?.contains('Myday') ||
                      container.closest('[role*="Myday"]') ||
                      container.closest('[data-testid*="Myday"]');

      // Additional broad exclusions for story-like content
      const isStoryLike = container.closest('[data-pagelet*="reel"]') ||
                         container.closest('[data-pagelet*="Reel"]') ||
                         container.closest('[aria-label*="reel"]') ||
                         container.closest('[aria-label*="Reel"]') ||
                         container.closest('.reel') ||
                         container.closest('[class*="reel"]') ||
                         container.closest('[data-pagelet*="highlight"]') ||
                         container.closest('[data-pagelet*="Highlight"]') ||
                         container.closest('[aria-label*="highlight"]') ||
                         container.closest('[aria-label*="Highlight"]') ||
                         container.closest('.highlight') ||
                         container.closest('[class*="highlight"]');

      if (isStory || isMyDay || isStoryLike) {
        console.log("Excluding video from button addition:", {
          isStory,
          isMyDay,
          isStoryLike,
          container: container,
          dataPagelet: container.getAttribute('data-pagelet'),
          ariaLabel: container.getAttribute('aria-label'),
          className: container.className
        });
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
        // Simple exclusion for Facebook stories and MyDay posts
        const dataPagelet = post.getAttribute('data-pagelet') || '';
        if (dataPagelet.toLowerCase().includes('story') || dataPagelet.toLowerCase().includes('myday')) {
          console.log(`Excluding post #${index} - appears to be story or MyDay: ${dataPagelet}`);
          return;
        }

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
      // Additional selectors for current Facebook structure
      '[data-visualcompletion="ignore-dynamic"]',
      '[data-instancekey]',
      'div[data-pagelet]',
      // More specific video post selectors
      'div[role="article"]',
      'article',
      // Facebook's current video post structure
      'div[data-ad-preview="message"]',
      'div[aria-label*="video"]',
    ];

    const posts = new Set();

    selectors.forEach((selector) => {
      document.querySelectorAll(selector).forEach((element) => {
        // Ultra-comprehensive exclusion logic for Facebook stories and MyDay
        const isStory = element.closest('[data-pagelet*="story"]') ||
                       element.closest('[data-pagelet*="Stories"]') ||
                       element.closest('[data-pagelet*="stories"]') ||
                       element.closest('[aria-label*="story"]') ||
                       element.closest('[aria-label*="Stories"]') ||
                       element.closest('[aria-label*="stories"]') ||
                       element.closest('.story') ||
                       element.closest('[class*="story"]') ||
                       element.closest('[data-visualcompletion*="story"]') ||
                       element.closest('[data-visualcompletion*="Stories"]') ||
                       element.closest('[data-visualcompletion*="stories"]') ||
                       element.getAttribute('data-pagelet')?.includes('story') ||
                       element.getAttribute('data-pagelet')?.includes('Stories') ||
                       element.getAttribute('data-pagelet')?.includes('stories') ||
                       element.classList?.contains('story') ||
                       element.classList?.contains('Stories') ||
                       element.classList?.contains('stories') ||
                       element.closest('[role*="story"]') ||
                       element.closest('[data-testid*="story"]') ||
                       element.closest('[data-testid*="Stories"]') ||
                       element.closest('[data-testid*="stories"]') ||
                       // Check parent containers for story indicators
                       element.closest('[data-pagelet*="feed"]')?.querySelector('[data-pagelet*="story"]') ||
                       element.closest('[data-pagelet*="timeline"]')?.querySelector('[data-pagelet*="story"]') ||
                       // Additional story patterns
                       element.closest('[data-pagelet*="Story"]') ||
                       element.closest('[aria-label*="Story"]') ||
                       element.closest('[class*="Story"]') ||
                       element.closest('[data-visualcompletion*="Story"]') ||
                       element.getAttribute('data-pagelet')?.includes('Story') ||
                       element.classList?.contains('Story') ||
                       element.closest('[role*="Story"]') ||
                       element.closest('[data-testid*="Story"]');

        const isMyDay = element.closest('[data-pagelet*="myday"]') ||
                        element.closest('[data-pagelet*="MyDay"]') ||
                        element.closest('[data-pagelet*="My Day"]') ||
                        element.closest('[data-pagelet*="my day"]') ||
                        element.closest('[aria-label*="myday"]') ||
                        element.closest('[aria-label*="MyDay"]') ||
                        element.closest('[aria-label*="My Day"]') ||
                        element.closest('[aria-label*="my day"]') ||
                        element.closest('.myday') ||
                        element.closest('.MyDay') ||
                        element.closest('[class*="myday"]') ||
                        element.closest('[class*="MyDay"]') ||
                        element.closest('[class*="my-day"]') ||
                        element.closest('[data-visualcompletion*="myday"]') ||
                        element.closest('[data-visualcompletion*="MyDay"]') ||
                        element.closest('[data-visualcompletion*="My Day"]') ||
                        element.closest('[data-visualcompletion*="my day"]') ||
                        element.getAttribute('data-pagelet')?.includes('myday') ||
                        element.getAttribute('data-pagelet')?.includes('MyDay') ||
                        element.getAttribute('data-pagelet')?.includes('My Day') ||
                        element.getAttribute('data-pagelet')?.includes('my day') ||
                        element.classList?.contains('myday') ||
                        element.classList?.contains('MyDay') ||
                        element.classList?.contains('my-day') ||
                        element.closest('[role*="myday"]') ||
                        element.closest('[role*="MyDay"]') ||
                        element.closest('[data-testid*="myday"]') ||
                        element.closest('[data-testid*="MyDay"]') ||
                        element.closest('[data-testid*="my-day"]') ||
                        // Check parent containers for MyDay indicators
                        element.closest('[data-pagelet*="feed"]')?.querySelector('[data-pagelet*="myday"]') ||
                        element.closest('[data-pagelet*="timeline"]')?.querySelector('[data-pagelet*="myday"]') ||
                        // Additional MyDay patterns
                        element.closest('[data-pagelet*="Myday"]') ||
                        element.closest('[aria-label*="Myday"]') ||
                        element.closest('[class*="Myday"]') ||
                        element.closest('[data-visualcompletion*="Myday"]') ||
                        element.getAttribute('data-pagelet')?.includes('Myday') ||
                        element.classList?.contains('Myday') ||
                        element.closest('[role*="Myday"]') ||
                        element.closest('[data-testid*="Myday"]');

        // Additional broad exclusions for story-like content
        const isStoryLike = element.closest('[data-pagelet*="reel"]') ||
                           element.closest('[data-pagelet*="Reel"]') ||
                           element.closest('[aria-label*="reel"]') ||
                           element.closest('[aria-label*="Reel"]') ||
                           element.closest('.reel') ||
                           element.closest('[class*="reel"]') ||
                           element.closest('[data-pagelet*="highlight"]') ||
                           element.closest('[data-pagelet*="Highlight"]') ||
                           element.closest('[aria-label*="highlight"]') ||
                           element.closest('[aria-label*="Highlight"]') ||
                           element.closest('.highlight') ||
                           element.closest('[class*="highlight"]');

        if (isStory || isMyDay || isStoryLike) {
          console.log("Excluding element from video posts scan:", {
            isStory,
            isMyDay,
            isStoryLike,
            element: element,
            dataPagelet: element.getAttribute('data-pagelet'),
            ariaLabel: element.getAttribute('aria-label'),
            className: element.className
          });
          return;
        }

        if (
          element.querySelector("video") ||
          element.textContent?.includes("video") ||
          element.getAttribute("data-ft")?.includes("video") ||
          element.getAttribute("data-pagelet")?.includes("video") ||
          element.querySelector('[aria-label*="video"]') ||
          element.querySelector('[data-visualcompletion*="media"]')
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
        canvas.width = videoElement.videoWidth || videoElement.clientWidth;
        canvas.height = videoElement.videoHeight || videoElement.clientHeight;
        
        console.log(`Canvas dimensions: ${canvas.width}x${canvas.height}`);

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

      // Create isolated container using Shadow DOM for better CSS isolation
      console.log("Creating isolated popup container with Shadow DOM");
      const shadowHost = document.createElement("div");
      shadowHost.id = "verifeed-popup-shadow-host";
      shadowHost.style.cssText = `
        position: fixed !important;
        top: 0 !important;
        left: 0 !important;
        width: 100vw !important;
        height: 100vh !important;
        pointer-events: none !important;
        z-index: 2147483647 !important;
        margin: 0 !important;
        padding: 0 !important;
        border: none !important;
        background: transparent !important;
      `;

      // Attach shadow root
      const shadowRoot = shadowHost.attachShadow({ mode: 'open' });
      console.log("Shadow root created:", shadowRoot);

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
            position: absolute !important;
            top: ${topPosition}px !important;
            right: ${rightPosition}px !important;
            z-index: 1000000 !important;
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
            margin: 0 !important;
            padding: 0 !important;
            border: 1px solid rgba(0, 0, 0, 0.1) !important;
            font-size: 14px !important;
            line-height: 1.4 !important;
            color: #333 !important;
            text-align: left !important;
            overflow: visible !important;
            white-space: normal !important;
            word-wrap: break-word !important;
        `;
      console.log("Popup styles set");

      console.log("Creating comprehensive style element");
      const style = document.createElement("style");
      style.textContent = `
            @keyframes slideDown {
                from {
                    opacity: 0 !important;
                    transform: translateY(-10px) !important;
                }
                to {
                    opacity: 1 !important;
                    transform: translateY(0) !important;
                }
            }

            .verifeed-results-popup {
                pointer-events: auto !important;
                position: absolute !important;
                z-index: 1000000 !important;
                display: block !important;
                visibility: visible !important;
                opacity: 1 !important;
            }

            .verifeed-popup-content {
                padding: 0 !important;
                margin: 0 !important;
                width: 100% !important;
                height: auto !important;
                display: block !important;
                position: relative !important;
            }

            .verifeed-popup-header {
                display: flex !important;
                align-items: center !important;
                justify-content: space-between !important;
                padding: 12px 16px !important;
                border-bottom: 1px solid #f3f4f6 !important;
                background: #fafafa !important;
                border-radius: 8px 8px 0 0 !important;
                margin: 0 !important;
                position: relative !important;
                z-index: 1 !important;
            }

            .verifeed-popup-header .status-icon {
                font-size: 16px !important;
                margin-right: 8px !important;
                display: inline-block !important;
                flex-shrink: 0 !important;
            }

            .verifeed-popup-header .status-text {
                font-weight: 600 !important;
                color: #374151 !important;
                font-size: 14px !important;
                flex: 1 !important;
                margin: 0 !important;
                padding: 0 !important;
                text-align: left !important;
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
                flex-shrink: 0 !important;
                margin-left: auto !important;
                transition: color 0.2s ease !important;
            }

            .verifeed-popup-header .close-btn:hover {
                color: #6b7280 !important;
            }

            .verifeed-popup-body {
                padding: 16px !important;
                margin: 0 !important;
                background: white !important;
                border-radius: 0 0 8px 8px !important;
            }

            .verifeed-popup-body .confidence-section {
                margin-bottom: 12px !important;
                display: block !important;
            }

            .verifeed-popup-body .confidence-label {
                font-size: 13px !important;
                font-weight: 600 !important;
                color: #374151 !important;
                display: block !important;
                margin-bottom: 6px !important;
                line-height: 1.3 !important;
            }

            .verifeed-popup-body .confidence-bar {
                width: 100% !important;
                height: 6px !important;
                background: #e5e7eb !important;
                border-radius: 3px !important;
                overflow: hidden !important;
                margin-bottom: 4px !important;
                position: relative !important;
            }

            .verifeed-popup-body .confidence-fill {
                height: 100% !important;
                border-radius: 3px !important;
                transition: width 0.8s ease-out !important;
                position: absolute !important;
                left: 0 !important;
                top: 0 !important;
            }

            .verifeed-popup-body .confidence-text {
                font-size: 12px !important;
                color: #6b7280 !important;
                display: block !important;
                margin-top: 4px !important;
                line-height: 1.3 !important;
            }

            .verifeed-popup-body .info-text {
                font-size: 13px !important;
                color: #4b5563 !important;
                line-height: 1.4 !important;
                margin-bottom: 12px !important;
                display: block !important;
                text-align: left !important;
            }

            .verifeed-popup-body .disclaimer {
                font-size: 11px !important;
                color: #9ca3af !important;
                text-align: center !important;
                line-height: 1.3 !important;
                padding-top: 8px !important;
                border-top: 1px solid #f3f4f6 !important;
                margin-top: 8px !important;
                display: block !important;
            }

            /* Additional isolation styles */
            * {
                box-sizing: border-box !important;
            }

            button, input, textarea, select {
                font-family: inherit !important;
                font-size: inherit !important;
            }
        `;
      console.log("Comprehensive style element created");

      console.log("Appending style to shadow root");
      shadowRoot.appendChild(style);
      console.log("Style appended to shadow root");

      console.log("Appending popup to shadow root");
      shadowRoot.appendChild(resultsPopup);
      console.log("Popup appended to shadow root");

      console.log("Appending shadow host to document body");
      document.body.appendChild(shadowHost);
      this.activePopup = shadowHost;
      this.activeStyle = style;
      console.log("Shadow host appended to body");

      console.log("=== POPUP DOM CHECK ===");
      console.log("Shadow host in DOM:", document.body.contains(shadowHost));
      console.log("Shadow root exists:", !!shadowRoot);
      console.log("Popup in shadow root:", shadowRoot.contains(resultsPopup));
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

        if (shadowHost.parentNode) {
          shadowHost.remove();
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
        if (shadowHost.parentNode) {
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
            !shadowHost.contains(e.target) &&
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
      console.log("Popup should be visible now with Shadow DOM isolation");
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

  if (request.action === "updateSettings") {
    if (veriFeedInstance) {
      veriFeedInstance.updateSettings(request.settings);
    }
    sendResponse({ success: true });
  }

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

  if (request.action === "analyzeAllVideos") {
    if (veriFeedInstance && veriFeedInstance.isEnabled) {
      console.log("Analyzing all videos...");
      const veriFeedButtons = document.querySelectorAll(".verifeed-verify-btn");
      console.log(`Found ${veriFeedButtons.length} VeriFeed buttons to analyze`);

      veriFeedButtons.forEach((button, index) => {
        setTimeout(() => {
          console.log(`Triggering analysis for video ${index + 1}/${veriFeedButtons.length}`);
          const container = button.closest('[role="article"], [data-pagelet*="video"], [data-pagelet*="FeedUnit"]');
          const videoElement = container ? container.querySelector("video") : null;

          if (container && videoElement) {
            veriFeedInstance.handleVerifyClick(container, videoElement, button);
          } else {
            console.log(`Skipping video ${index + 1}: container or video not found`);
          }
        }, index * 2000); // Stagger analysis by 2 seconds to avoid overwhelming the server
      });

      sendResponse({ success: true, message: `Started analysis for ${veriFeedButtons.length} videos` });
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