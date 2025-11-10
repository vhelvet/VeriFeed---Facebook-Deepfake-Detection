// VeriFeed Content Script - SECURED + OPTIMIZED
// Combines authentication with all performance optimizations from faster version

// ===== AUTH MODULE (EMBEDDED) =====
class VerifeedAuth {
  constructor() {
    this.apiUrl = "http://localhost:5000";
    this.apiKey = "5hTeoaOm5m-91clhe2iVqKy2jpkiN54JLQ4vNbiDodU";
    this.token = null;
    this.tokenExpiry = null;

    this.loadToken();
  }

  async loadToken() {
    try {
      const result = await chrome.storage.local.get([
        "auth_token",
        "token_expiry",
      ]);
      if (result.auth_token && result.token_expiry) {
        const expiry = new Date(result.token_expiry);
        if (expiry > new Date()) {
          this.token = result.auth_token;
          this.tokenExpiry = expiry;
          console.log("✓ Loaded valid token from storage");
          return true;
        } else {
          console.log("Token expired, clearing...");
          await this.clearToken();
        }
      }
    } catch (error) {
      console.error("Error loading token:", error);
    }
    return false;
  }

  async saveToken(token, expiresIn) {
    try {
      const expiry = new Date();
      expiry.setSeconds(expiry.getSeconds() + expiresIn);

      await chrome.storage.local.set({
        auth_token: token,
        token_expiry: expiry.toISOString(),
      });

      this.token = token;
      this.tokenExpiry = expiry;

      console.log("✓ Token saved to storage");
    } catch (error) {
      console.error("Error saving token:", error);
    }
  }

  async clearToken() {
    try {
      await chrome.storage.local.remove(["auth_token", "token_expiry"]);
      this.token = null;
      this.tokenExpiry = null;
      console.log("✓ Token cleared");
    } catch (error) {
      console.error("Error clearing token:", error);
    }
  }

  isTokenValid() {
    if (!this.token || !this.tokenExpiry) {
      return false;
    }

    const now = new Date();
    const fiveMinutesFromNow = new Date(now.getTime() + 5 * 60000);

    return this.tokenExpiry > fiveMinutesFromNow;
  }

  async generateToken() {
    try {
      console.log("Generating new JWT token...");

      const response = await fetch(`${this.apiUrl}/auth/token`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          api_key: this.apiKey,
        }),
      });

      if (!response.ok) {
        const error = await response.json();
        throw new Error(error.error || "Failed to generate token");
      }

      const data = await response.json();

      await this.saveToken(data.token, data.expires_in);

      console.log("✓ Token generated successfully");
      return data.token;
    } catch (error) {
      console.error("Error generating token:", error);
      throw error;
    }
  }

  async ensureToken() {
    if (this.isTokenValid()) {
      return this.token;
    }

    return await this.generateToken();
  }

  async getAuthHeaders() {
    const token = await this.ensureToken();

    return {
      "Content-Type": "application/json",
      Authorization: `Bearer ${token}`,
    };
  }

  async authenticatedFetch(url, options = {}) {
    try {
      const headers = await this.getAuthHeaders();

      const response = await fetch(url, {
        ...options,
        headers: {
          ...options.headers,
          ...headers,
        },
      });

      if (response.status === 401) {
        console.log("Token invalid, regenerating...");
        await this.clearToken();

        const newHeaders = await this.getAuthHeaders();
        const retryResponse = await fetch(url, {
          ...options,
          headers: {
            ...options.headers,
            ...newHeaders,
          },
        });

        return retryResponse;
      }

      return response;
    } catch (error) {
      console.error("Authenticated fetch error:", error);
      throw error;
    }
  }
}

const verifeedAuth = new VerifeedAuth();

// ===== NLG SYSTEM =====
class DeepfakeNLG {
  constructor() {
    this.templates = {
      deepfake: {
        high_confidence: [
          {
            structure: "[DETERMINATION] [EVIDENCE] [ACTION]",
            determination: [
              "Our analysis strongly indicates",
              "We've identified clear signs that",
              "Evidence suggests",
              "Multiple indicators show that",
            ],
            evidence: [
              "this video has been manipulated using artificial intelligence",
              "deepfake technology was used to create this content",
              "AI-generated alterations are present in this video",
              "this content was synthetically modified using AI tools",
            ],
            action: [
              "We recommend verifying this through other sources before sharing.",
              "Please fact-check before distributing this content.",
              "Cross-reference with original sources before sharing.",
              "Verify through trusted sources before considering it authentic.",
            ],
          },
        ],
        medium_confidence: [
          {
            structure: "[POSSIBILITY] [EVIDENCE] [CAUTION]",
            possibility: [
              "This video may have been",
              "There are indicators suggesting this was",
              "We've detected signs that this could be",
              "Analysis indicates this might have been",
            ],
            evidence: [
              "edited or created using artificial intelligence",
              "manipulated with deepfake technology",
              "generated using AI tools",
              "altered using synthetic media techniques",
            ],
            caution: [
              "Please verify it before sharing.",
              "We recommend additional verification.",
              "Exercise caution when sharing this content.",
              "Fact-check through multiple sources before sharing.",
            ],
          },
        ],
        low_confidence: [
          {
            structure: "[UNCERTAINTY] [OBSERVATION] [RECOMMENDATION]",
            uncertainty: [
              "While we cannot be certain,",
              "Our analysis is inconclusive, but",
              "We have limited confidence that",
              "Though uncertain,",
            ],
            observation: [
              "this video shows some signs of manipulation",
              "there may be AI-generated elements present",
              "artificial alterations might be present",
              "possible synthetic modifications were detected",
            ],
            recommendation: [
              "Treat this content with skepticism and verify through multiple sources.",
              "We strongly recommend fact-checking before sharing.",
              "Additional analysis is needed before drawing conclusions.",
              "Consult additional verification methods before trusting this content.",
            ],
          },
        ],
      },
      authentic: {
        high_confidence: [
          {
            structure: "[DETERMINATION] [EVIDENCE] [ASSESSMENT]",
            determination: [
              "Our analysis indicates",
              "We've found strong evidence that",
              "Multiple factors suggest",
              "Our assessment shows",
            ],
            evidence: [
              "this video is genuine and has not been digitally manipulated",
              "this content appears authentic with no signs of AI generation",
              "this video shows no indicators of deepfake technology",
              "this is authentic content without synthetic alterations",
            ],
            assessment: [
              "However, always verify important content through trusted sources.",
              "Still, cross-referencing with original sources is good practice.",
              "We still recommend verifying through official channels when possible.",
              "As always, verify critical content through additional sources.",
            ],
          },
        ],
        medium_confidence: [
          {
            structure: "[LIKELIHOOD] [EVIDENCE] [CAUTION]",
            likelihood: [
              "This video appears to be",
              "We believe this is likely",
              "Evidence suggests this is probably",
              "Analysis indicates this is most likely",
            ],
            evidence: [
              "authentic and unmanipulated",
              "genuine with no AI alterations",
              "real content without deepfake elements",
              "legitimate with no synthetic modifications",
            ],
            caution: [
              "though we recommend verification for complete certainty.",
              "but additional verification is always recommended.",
              "though exercising caution is still advisable.",
              "however, verify through trusted sources when important.",
            ],
          },
        ],
        low_confidence: [
          {
            structure: "[UNCERTAINTY] [OBSERVATION] [RECOMMENDATION]",
            uncertainty: [
              "We cannot confidently determine",
              "Our analysis is inconclusive about",
              "We have low confidence in assessing",
              "It's unclear from our analysis",
            ],
            observation: [
              "whether this video is authentic or manipulated",
              "if this content contains AI-generated elements",
              "the authenticity of this video",
              "whether synthetic alterations are present",
            ],
            recommendation: [
              "Please verify through multiple trusted sources before relying on this content.",
              "We recommend treating this with caution until verified.",
              "Additional expert analysis may be needed for confirmation.",
              "Seek verification from authoritative sources before trusting this content.",
            ],
          },
        ],
      },
    };
  }

  generate(prediction, confidence) {
    const isAuthentic = prediction === "REAL";
    const category = isAuthentic ? "authentic" : "deepfake";

    let confidenceLevel;
    if (confidence >= 80) {
      confidenceLevel = "high_confidence";
    } else if (confidence >= 60) {
      confidenceLevel = "medium_confidence";
    } else {
      confidenceLevel = "low_confidence";
    }

    const templateSet = this.templates[category][confidenceLevel];
    const template =
      templateSet[Math.floor(Math.random() * templateSet.length)];

    const parts = template.structure.match(/\[([^\]]+)\]/g).map((slot) => {
      const slotName = slot.replace(/[\[\]]/g, "").toLowerCase();
      const options = template[slotName];
      return options[Math.floor(Math.random() * options.length)];
    });

    return parts.join(" ");
  }

  generateConfidenceText(confidence) {
    if (confidence >= 90) {
      const options = [
        "We have very high confidence in this assessment",
        "Our analysis provides very strong certainty",
        "We are highly confident in this determination",
        "This assessment has very high reliability",
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else if (confidence >= 80) {
      const options = [
        "We have high confidence in this assessment",
        "Our analysis provides strong certainty",
        "We are confident in this determination",
        "This assessment has high reliability",
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else if (confidence >= 70) {
      const options = [
        "We have moderate confidence in this assessment",
        "Our analysis suggests reasonable certainty",
        "We are moderately confident in this determination",
        "This assessment has moderate reliability",
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else if (confidence >= 60) {
      const options = [
        "We have limited confidence in this assessment",
        "Our analysis suggests some uncertainty",
        "We are somewhat confident in this determination",
        "This assessment has limited reliability",
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else {
      const options = [
        "We have low confidence in this assessment",
        "Our analysis is highly uncertain",
        "We have minimal confidence in this determination",
        "This assessment has low reliability",
      ];
      return options[Math.floor(Math.random() * options.length)];
    }
  }
}

const deepfakeNLG = new DeepfakeNLG();

// ===== MAIN DETECTOR CLASS (OPTIMIZED) =====
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
    this.scrollCloseHandler = null;
    this.scrollCloseQueued = false;

    // OPTIMIZATION: Frame extraction config
    this.TARGET_FPS = 5;
    this.EXTRACT_DURATION = 30;
    this.TARGET_FRAMES = 150;

    this.init();
  }

  init() {
    console.log("VeriFeed Predictor initialized (SECURED + OPTIMIZED)");
    console.log(
      `Target: ${this.TARGET_FRAMES} frames at ${this.TARGET_FPS}fps for ${this.EXTRACT_DURATION}s`
    );
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
        console.log("✅ Backend server ready (SECURED + OPTIMIZED)");
        console.log(`   Device: ${data.device}`);
        if (data.optimizations) {
          console.log(
            `   Optimizations: ${JSON.stringify(data.optimizations)}`
          );
        }
      } else if (!data.model_loaded) {
        console.warn("⚠️ Backend online but model not loaded");
      }
    } catch (error) {
      console.error("❌ Backend server offline");
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

  getLoader(statusText) {
    return `
        <div class="verifeed-loader-container">
            <div class="verifeed-hologram">
                <div class="verifeed-core"></div>
                <div class="verifeed-ring verifeed-ring-1"></div>
                <div class="verifeed-ring verifeed-ring-2"></div>
                <div class="verifeed-ring verifeed-ring-3"></div>
                <div class="verifeed-particles">
                    <span></span><span></span><span></span><span></span>
                    <span></span><span></span><span></span><span></span>
                </div>
            </div>
            <span class="verifeed-status-text">${statusText}</span>           
        </div>
        <style>
            .verifeed-loader-container {
                display: flex;
                align-items: center;
                gap: 8px;
                position: relative;
            }
            
            .verifeed-hologram {
                position: relative;
                width: 24px;
                height: 24px;
                transform-style: preserve-3d;
                animation: hologramFloat 3s ease-in-out infinite;
            }
            
            @keyframes hologramFloat {
                0%, 100% { transform: translateY(0) rotateY(0deg); }
                50% { transform: translateY(-3px) rotateY(180deg); }
            }
            
            .verifeed-core {
                position: absolute;
                top: 50%;
                left: 50%;
                width: 8px;
                height: 8px;
                background: radial-gradient(circle, #fff, #667eea);
                border-radius: 50%;
                transform: translate(-50%, -50%);
                box-shadow: 0 0 20px #667eea, 0 0 40px #764ba2, inset 0 0 10px #fff;
                animation: corePulse 1.5s ease-in-out infinite;
            }
            
            @keyframes corePulse {
                0%, 100% { transform: translate(-50%, -50%) scale(1); opacity: 1; }
                50% { transform: translate(-50%, -50%) scale(1.3); opacity: 0.8; }
            }
            
            .verifeed-ring {
                position: absolute;
                top: 50%;
                left: 50%;
                border: 2px solid transparent;
                border-radius: 50%;
                transform: translate(-50%, -50%);
                animation: ringExpand 2s cubic-bezier(0.4, 0, 0.6, 1) infinite;
            }
            
            .verifeed-ring-1 {
                width: 16px;
                height: 16px;
                border-top-color: rgba(102, 126, 234, 0.8);
                border-right-color: rgba(102, 126, 234, 0.6);
                animation-delay: 0s;
            }
            
            .verifeed-ring-2 {
                width: 20px;
                height: 20px;
                border-top-color: rgba(118, 75, 162, 0.6);
                border-left-color: rgba(118, 75, 162, 0.4);
                animation-delay: 0.4s;
            }
            
            .verifeed-ring-3 {
                width: 24px;
                height: 24px;
                border-top-color: rgba(255, 255, 255, 0.4);
                border-bottom-color: rgba(255, 255, 255, 0.2);
                animation-delay: 0.8s;
            }
            
            @keyframes ringExpand {
                0% {
                    transform: translate(-50%, -50%) rotate(0deg) scale(0.5);
                    opacity: 0;
                }
                50% {
                    opacity: 1;
                }
                100% {
                    transform: translate(-50%, -50%) rotate(360deg) scale(1.2);
                    opacity: 0;
                }
            }
            
            .verifeed-particles {
                position: absolute;
                width: 100%;
                height: 100%;
                top: 0;
                left: 0;
            }
            
            .verifeed-particles span {
                position: absolute;
                width: 3px;
                height: 3px;
                background: linear-gradient(45deg, #667eea, #764ba2);
                border-radius: 50%;
                box-shadow: 0 0 10px currentColor;
                animation: particleOrbit 3s linear infinite;
            }
            
            .verifeed-particles span:nth-child(1) { animation-delay: 0s; }
            .verifeed-particles span:nth-child(2) { animation-delay: 0.375s; }
            .verifeed-particles span:nth-child(3) { animation-delay: 0.75s; }
            .verifeed-particles span:nth-child(4) { animation-delay: 1.125s; }
            .verifeed-particles span:nth-child(5) { animation-delay: 1.5s; }
            .verifeed-particles span:nth-child(6) { animation-delay: 1.875s; }
            .verifeed-particles span:nth-child(7) { animation-delay: 2.25s; }
            .verifeed-particles span:nth-child(8) { animation-delay: 2.625s; }
            
            @keyframes particleOrbit {
                0% {
                    transform: rotate(0deg) translateX(15px) scale(0);
                    opacity: 0;
                }
                10% {
                    opacity: 1;
                    transform: rotate(36deg) translateX(15px) scale(1);
                }
                90% {
                    opacity: 1;
                    transform: rotate(324deg) translateX(15px) scale(1);
                }
                100% {
                    transform: rotate(360deg) translateX(15px) scale(0);
                    opacity: 0;
                }
            }
            
            .verifeed-status-text {
                font-weight: 600;
                font-size: 12px;
                background: linear-gradient(90deg, #fff 0%, #667eea 50%, #fff 100%);
                background-size: 200% 100%;
                -webkit-background-clip: text;
                background-clip: text;
                -webkit-text-fill-color: transparent;
                animation: shimmerText 2s linear infinite;
                letter-spacing: 0.5px;
            }
            
            @keyframes shimmerText {
                0% { background-position: 200% 0; }
                100% { background-position: -200% 0; }
            }
            
            .verifeed-dots {
                display: flex;
                gap: 2px;
            }
            
            .verifeed-dots span {
                color: white;
                font-weight: bold;
                animation: dotBounce 1.4s ease-in-out infinite;
            }
            
            .verifeed-dots span:nth-child(1) { animation-delay: 0s; }
            .verifeed-dots span:nth-child(2) { animation-delay: 0.2s; }
            .verifeed-dots span:nth-child(3) { animation-delay: 0.4s; }
            
            @keyframes dotBounce {
                0%, 60%, 100% { transform: translateY(0); opacity: 0.4; }
                30% { transform: translateY(-8px); opacity: 1; }
            }
        </style>
    `;
  }

  setupScrollCloseListener() {
    this.scrollCloseHandler = () => {
      if (!this.scrollCloseQueued) {
        this.scrollCloseQueued = true;
        window.requestAnimationFrame(() => {
          if (this.activePopup) {
            this.removeExistingPopup();
            console.log("Popup closed due to user scrolling.");
          }
          this.scrollCloseQueued = false;
        });
      }
    };
    window.addEventListener("scroll", this.scrollCloseHandler, {
      passive: true,
    });

    this.clickOutsideHandler = (event) => {
      if (this.activePopup && !this.activePopup.contains(event.target)) {
        const isVerifeedButton = event.target.closest(".verifeed-predict-btn");
        if (!isVerifeedButton) {
          this.removeExistingPopup();
          console.log("Popup closed due to click outside.");
        }
      }
    };
    document.addEventListener("click", this.clickOutsideHandler, true);
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
            <span>Verifeed</span>
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
      '[aria-label*="more"], [aria-label*="options"]'
    );
    let buttonPosition = "60px";

    if (menuButton) {
      const menuRect = menuButton.getBoundingClientRect();
      const targetRect = targetContainer.getBoundingClientRect();
      const relativeRight =
        targetRect.right - menuRect.right + menuRect.width + 8;
      buttonPosition = `${relativeRight}px`;
    }

    predictBtn.style.cssText = `
            position: absolute !important;
            top: 12px !important;
            right: ${buttonPosition} !important;
            z-index: 2147483647 !important;
            background: linear-gradient(135deg, #764ba2 0%, #667eea 100%) !important;
            color: white !important;
            border: 1px solid rgba(255, 255, 255, 0.1) !important;
            border-radius: 4px !important;
            padding: 6px 12px !important;
            font-size: 12px !important;
            font-weight: 600 !important;
            cursor: pointer !important;
            box-shadow: 0 3px 10px rgba(0, 0, 0, 0.2) !important;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif !important;
            display: inline-flex !important;
            align-items: center !important;
            transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        `;

    predictBtn.onmouseenter = () => {
      predictBtn.style.background =
        "linear-gradient(135deg, #6a4190 0%, #5a6fd8 100%)";
      predictBtn.style.transform = "translateY(-1px) scale(1.02)";
      predictBtn.style.boxShadow = "0 6px 12px rgba(0, 0, 0, 0.3)";
    };
    predictBtn.onmouseleave = () => {
      predictBtn.style.background =
        "linear-gradient(135deg, #764ba2 0%, #667eea 100%)";
      predictBtn.style.transform = "translateY(0) scale(1)";
      predictBtn.style.boxShadow = "0 3px 10px rgba(0, 0, 0, 0.2)";
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
    buttonElement.innerHTML = this.getLoader("Extracting");

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
      console.log("=== EXTRACTING FRAMES (OPTIMIZED) ===");
      const frames = await this.extractFramesFast(videoElement);

      if (!frames || frames.length === 0) {
        throw new Error("Could not extract frames from video");
      }
      console.log(`Successfully extracted ${frames.length} frames`);

      this.restorePageState(scrollY, originalVideoState, videoElement);

      buttonElement.innerHTML = this.getLoader("Analyzing");

      console.log("=== SENDING TO BACKEND (AUTHENTICATED) ===");
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

  // OPTIMIZED: Fast frame extraction using seek-based method
  async extractFramesFast(videoElement) {
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
          `Target: ${TARGET_FRAMES} frames at ${TARGET_FPS}fps from first ${EXTRACT_DURATION}s`
        );

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
            console.warn(`⏱️ Timeout - captured ${frames.length} frames`);
            resolve(frames);
          }
        }, 5000);

        const handleError = (e) => {
          if (hasResolved) return;
          console.error("Video element error during seek:", e);
          clearTimeout(timeoutId);
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
          frames.push(canvas.toDataURL("image/jpeg", 0.7)); // OPTIMIZATION: Lower quality for speed

          if (frames.length % 50 === 0) {
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
            console.log(`Captured ${frames.length} frames in ${elapsed}s`);
          }
        }

        clearTimeout(timeoutId);
        videoElement.removeEventListener("error", handleError);
        hasResolved = true;

        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        console.log("=== FAST EXTRACTION COMPLETE ===");
        console.log(
          `✅ Captured ${frames.length} frames in ${elapsed}s (Target: <5s)`
        );

        resolve(frames);
      } catch (err) {
        if (!hasResolved) {
          if (timeoutId) clearTimeout(timeoutId);
          console.error("=== EXTRACTION ERROR (FATAL) ===");
          reject(err);
        }
      }
    });
  }

  async sendToBackend(frames, buttonElement, originalContent) {
    try {
      console.log("=== CHECKING SERVER HEALTH ===");
      const healthResponse = await fetch(`${this.serverUrl}/health`);

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

      console.log(
        `Sending ${frames.length} frames for prediction (AUTHENTICATED)`
      );

      // OPTIMIZATION: Use authenticated fetch
      const response = await verifeedAuth.authenticatedFetch(
        `${this.serverUrl}/predict`,
        {
          method: "POST",
          body: JSON.stringify(requestData),
        }
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

      if (predictionData.processing_time) {
        console.log("Processing time:", predictionData.processing_time);
      }

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
    const bgGradient = isAuthentic
      ? "linear-gradient(135deg, #d1fae5 0%, #a7f3d0 100%)"
      : "linear-gradient(135deg, #fee2e2 0%, #fecaca 100%)";

    const processingInfo = result.processing_time
      ? `<div class="processing-time">
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                </svg>
                <span>Analyzed in ${result.processing_time}s</span>
               </div>`
      : "";

    resultsPopup.innerHTML = `
            <div class="verifeed-popup-content">
                <div class="verifeed-popup-header" style="background: ${bgGradient};">
                    <div class="status-indicator">
                        <span class="status-icon-animated">${statusIcon}</span>
                        <div class="status-info">
                            <span class="status-text" style="color: ${statusColor};">${statusText}</span>
                            <span class="status-subtitle">AI Analysis Complete</span>
                        </div>
                    </div>
                    <button class="close-btn" title="Close">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"/><line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div class="verifeed-popup-body">
                    <div class="confidence-section">
                        <div class="confidence-header">
                            <span class="confidence-label">Overall Confidence</span>
                            <span class="confidence-value" style="color: ${statusColor};">${confidence.toFixed(
      1
    )}%</span>
                        </div>
                        <div class="confidence-bar-container">
                            <div class="confidence-bar">
                                <div class="confidence-fill" style="width: 0%; background: ${statusColor};" data-width="${confidence}"></div>
                            </div>
                            <div class="confidence-markers">
                                <span>0%</span>
                                <span>50%</span>
                                <span>100%</span>
                            </div>
                        </div>
                    </div>
                   
                    <div class="probability-section">
                        <div class="section-title">Detailed Analysis</div>
                        <div class="prob-grid">
                            <div class="prob-card prob-real">
                                <div class="prob-icon">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#10b981" stroke-width="2">
                                        <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/>
                                        <polyline points="22 4 12 14.01 9 11.01"/>
                                    </svg>
                                </div>
                                <div class="prob-content">
                                    <div class="prob-label">Authentic</div>
                                    <div class="prob-value">${realProb.toFixed(
                                      1
                                    )}%</div>
                                    <div class="prob-bar">
                                        <div class="prob-fill prob-fill-real" style="width: 0%;" data-width="${realProb}"></div>
                                    </div>
                                </div>
                            </div>
                            <div class="prob-card prob-fake">
                                <div class="prob-icon">
                                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2">
                                        <circle cx="12" cy="12" r="10"/>
                                        <line x1="15" y1="9" x2="9" y2="15"/>
                                        <line x1="9" y1="9" x2="15" y2="15"/>
                                    </svg>
                                </div>
                                <div class="prob-content">
                                    <div class="prob-label">Manipulated</div>
                                    <div class="prob-value">${fakeProb.toFixed(
                                      1
                                    )}%</div>
                                    <div class="prob-bar">
                                        <div class="prob-fill prob-fill-fake" style="width: 0%;" data-width="${fakeProb}"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                   
                    <div class="info-card" style="border-left-color: ${statusColor};">
                        <div class="info-icon">${
                          isAuthentic ? "📋" : "⚠️"
                        }</div>
                        <div class="info-content">
                            <div class="info-title">${
                              isAuthentic ? "Verification Result" : "Warning"
                            }</div>
                            <div class="info-description">
                                ${deepfakeNLG.generate(prediction, confidence)}
                            </div>
                        </div>
                    </div>
                   
                    <div class="metadata-section">
                        <div class="metadata-item">
                            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
                                <line x1="9" y1="9" x2="15" y2="15"/>
                                <line x1="15" y1="9" x2="9" y2="15"/>
                            </svg>
                            <span>Frames: ${
                              result.frames_processed || "N/A"
                            }</span>
                        </div>
                        ${processingInfo}
                    </div>
                </div>
            </div>
        `;

    resultsPopup.style.cssText = `
            position: fixed !important;
            top: ${buttonRect.bottom + 8}px !important;
            right: ${window.innerWidth - buttonRect.right}px !important;
            z-index: 2147483647 !important;
            width: 360px !important;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
            background: white !important;
            border-radius: 12px !important;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.2), 0 0 0 1px rgba(0, 0, 0, 0.05) !important;
            animation: popupEntrance 0.4s cubic-bezier(0.34, 1.56, 0.64, 1) !important;
            overflow: hidden !important;
        `;

    const style = document.createElement("style");
    style.id = "verifeed-popup-styles";
    style.textContent = `
            @keyframes popupEntrance {
                0% { opacity: 0; transform: translateY(-20px) scale(0.9); }
                100% { opacity: 1; transform: translateY(0) scale(1); }
            }
           
            @keyframes iconPulse {
                0%, 100% { transform: scale(1); }
                50% { transform: scale(1.1); }
            }
           
            @keyframes shimmer {
                0% { background-position: -200% 0; }
                100% { background-position: 200% 0; }
            }
           
            .verifeed-popup-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 20px;
                border-radius: 12px 12px 0 0;
                position: relative;
                overflow: hidden;
            }
           
            .verifeed-popup-header::before {
                content: '';
                position: absolute;
                top: 0;
                left: -200%;
                width: 200%;
                height: 100%;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.3), transparent);
                animation: shimmer 3s infinite;
            }
           
            .status-indicator {
                display: flex;
                align-items: center;
                gap: 12px;
                position: relative;
                z-index: 1;
            }
           
            .status-icon-animated {
                font-size: 28px;
                animation: iconPulse 2s ease-in-out infinite;
                display: flex;
                align-items: center;
                justify-content: center;
            }
           
            .status-info {
                display: flex;
                flex-direction: column;
                gap: 2px;
            }
           
            .status-text {
                font-weight: 700;
                font-size: 16px;
                letter-spacing: -0.01em;
            }
           
            .status-subtitle {
                font-size: 11px;
                color: #6b7280;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
           
            .close-btn {
                background: rgba(255, 255, 255, 0.9);
                border: none;
                color: #6b7280;
                cursor: pointer;
                width: 28px;
                height: 28px;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                position: relative;
                z-index: 1;
            }
           
            .close-btn:hover {
                background: white;
                color: #374151;
                transform: rotate(90deg);
            }
           
            .verifeed-popup-body {
                padding: 24px;
                background: white;
            }
           
            .confidence-section {
                margin-bottom: 24px;
            }
           
            .confidence-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 12px;
            }
           
            .confidence-label {
                font-size: 13px;
                font-weight: 600;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
           
            .confidence-value {
                font-size: 24px;
                font-weight: 700;
                letter-spacing: -0.02em;
            }
           
            .confidence-bar-container {
                position: relative;
            }
           
            .confidence-bar {
                width: 100%;
                height: 12px;
                background: #f3f4f6;
                border-radius: 6px;
                overflow: hidden;
                position: relative;
                box-shadow: inset 0 2px 4px rgba(0, 0, 0, 0.06);
            }
           
            .confidence-fill {
                height: 100%;
                border-radius: 6px;
                transition: width 1.2s cubic-bezier(0.65, 0, 0.35, 1);
                position: relative;
                overflow: hidden;
            }
           
            .confidence-fill::after {
                content: '';
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(90deg, transparent, rgba(255,255,255,0.4), transparent);
                animation: shimmer 2s infinite;
            }
           
            .confidence-markers {
                display: flex;
                justify-content: space-between;
                margin-top: 6px;
                font-size: 10px;
                color: #9ca3af;
                font-weight: 500;
            }
           
            .section-title {
                font-size: 13px;
                font-weight: 600;
                color: #6b7280;
                text-transform: uppercase;
                letter-spacing: 0.05em;
                margin-bottom: 12px;
            }
           
            .prob-grid {
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 12px;
                margin-bottom: 20px;
            }
           
            .prob-card {
                background: #f9fafb;
                border-radius: 8px;
                padding: 14px;
                border: 1px solid #e5e7eb;
                transition: all 0.3s ease;
            }
           
            .prob-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
            }
           
            .prob-icon {
                margin-bottom: 8px;
            }
           
            .prob-content {
                display: flex;
                flex-direction: column;
                gap: 6px;
            }
           
            .prob-label {
                font-size: 11px;
                color: #6b7280;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
           
            .prob-value {
                font-size: 20px;
                font-weight: 700;
                color: #1f2937;
                letter-spacing: -0.02em;
            }
           
            .prob-bar {
                width: 100%;
                height: 6px;
                background: #e5e7eb;
                border-radius: 3px;
                overflow: hidden;
            }
           
            .prob-fill {
                height: 100%;
                border-radius: 3px;
                transition: width 1s cubic-bezier(0.65, 0, 0.35, 1) 0.2s;
            }
           
            .prob-fill-real {
                background: linear-gradient(90deg, #10b981, #059669);
            }
           
            .prob-fill-fake {
                background: linear-gradient(90deg, #ef4444, #dc2626);
            }
           
            .info-card {
                background: #f9fafb;
                border-radius: 8px;
                padding: 16px;
                border-left: 4px solid;
                display: flex;
                gap: 12px;
                margin-bottom: 20px;
                animation: slideInLeft 0.5s ease-out 0.3s both;
            }
           
            @keyframes slideInLeft {
                from { opacity: 0; transform: translateX(-10px); }
                to { opacity: 1; transform: translateX(0); }
            }
           
            .info-icon {
                font-size: 24px;
                flex-shrink: 0;
            }
           
            .info-content {
                flex: 1;
            }
           
            .info-title {
                font-size: 13px;
                font-weight: 700;
                color: #374151;
                margin-bottom: 6px;
            }
           
            .info-description {
                font-size: 13px;
                color: #6b7280;
                line-height: 1.6;
            }
           
            .metadata-section {
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-top: 16px;
                border-top: 1px solid #e5e7eb;
            }
           
            .metadata-item, .processing-time {
                display: flex;
                align-items: center;
                gap: 6px;
                font-size: 11px;
                color: #6b7280;
                font-weight: 500;
            }
           
            .processing-time svg {
                color: #10b981;
            }
           
            .metadata-item svg {
                opacity: 0.6;
            }
        `;

    document.head.appendChild(style);
    document.body.appendChild(resultsPopup);
    this.activePopup = resultsPopup;
    this.activeStyle = style;

    setTimeout(() => {
      const confidenceFill = resultsPopup.querySelector(".confidence-fill");
      const probFills = resultsPopup.querySelectorAll(".prob-fill");

      if (confidenceFill) {
        confidenceFill.style.width = confidenceFill.dataset.width + "%";
      }

      probFills.forEach((fill) => {
        fill.style.width = fill.dataset.width + "%";
      });
    }, 100);

    const closeBtn = resultsPopup.querySelector(".close-btn");
    const closePopup = () => {
      resultsPopup.style.animation = "popupExit 0.3s ease-out forwards";

      const exitKeyframes = `
                @keyframes popupExit {
                    0% { opacity: 1; transform: translateY(0) scale(1); }
                    100% { opacity: 0; transform: translateY(-10px) scale(0.95); }
                }
            `;

      if (!document.getElementById("verifeed-exit-animation")) {
        const exitStyle = document.createElement("style");
        exitStyle.id = "verifeed-exit-animation";
        exitStyle.textContent = exitKeyframes;
        document.head.appendChild(exitStyle);
      }

      setTimeout(() => {
        if (resultsPopup.parentNode) resultsPopup.remove();
        if (style.parentNode) style.remove();
        this.activePopup = null;
        this.activeStyle = null;
        window.removeEventListener("scroll", this.scrollCloseHandler);
        this.scrollCloseHandler = null;
      }, 300);
    };

    closeBtn.addEventListener("click", closePopup);
    this.setupScrollCloseListener();

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
                    <div class="error-indicator">
                        <div class="error-icon-wrapper">
                            <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                                <circle cx="12" cy="12" r="10"/>
                                <line x1="12" y1="8" x2="12" y2="12"/>
                                <line x1="12" y1="16" x2="12.01" y2="16"/>
                            </svg>
                        </div>
                        <div class="error-info">
                            <span class="error-title">Analysis Failed</span>
                            <span class="error-subtitle">Unable to process video</span>
                        </div>
                    </div>
                    <button class="close-btn" title="Close">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <line x1="18" y1="6" x2="6" y2="18"/>
                            <line x1="6" y1="6" x2="18" y2="18"/>
                        </svg>
                    </button>
                </div>
                <div class="error-body">
                    <div class="error-message">
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ef4444" stroke-width="2">
                            <path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/>
                            <line x1="12" y1="9" x2="12" y2="13"/>
                            <line x1="12" y1="17" x2="12.01" y2="17"/>
                        </svg>
                        <p>${message}</p>
                    </div>
                    <button class="retry-btn">
                        <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                            <polyline points="23 4 23 10 17 10"/>
                            <path d="M20.49 15a9 9 0 1 1-2.12-9.36L23 10"/>
                        </svg>
                        <span>Got it</span>
                    </button>
                </div>
            </div>
        `;

    errorPopup.style.cssText = `
            position: fixed !important;
            top: ${buttonRect.bottom + 8}px !important;
            right: ${window.innerWidth - buttonRect.right}px !important;
            z-index: 2147483647 !important;
            width: 340px !important;
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif !important;
            background: white !important;
            border-radius: 12px !important;
            box-shadow: 0 20px 60px rgba(239, 68, 68, 0.15), 0 0 0 1px rgba(239, 68, 68, 0.1) !important;
            animation: errorShake 0.5s cubic-bezier(0.36, 0.07, 0.19, 0.97) !important;
            overflow: hidden !important;
        `;

    const style = document.createElement("style");
    style.id = "verifeed-error-styles";
    style.textContent = `
            @keyframes errorShake {
                0%, 100% { transform: translateX(0); }
                10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
                20%, 40%, 60%, 80% { transform: translateX(5px); }
            }
           
            @keyframes errorPulse {
                0%, 100% { transform: scale(1); opacity: 1; }
                50% { transform: scale(1.05); opacity: 0.8; }
            }
           
            .verifeed-error-popup .error-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                padding: 20px;
                background: linear-gradient(135deg, #fee2e2 0%, #fecaca 100%);
                border-radius: 12px 12px 0 0;
                position: relative;
                overflow: hidden;
            }
           
            .error-indicator {
                display: flex;
                align-items: center;
                gap: 12px;
                position: relative;
                z-index: 1;
            }
           
            .error-icon-wrapper {
                width: 36px;
                height: 36px;
                background: white;
                border-radius: 8px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: #ef4444;
                animation: errorPulse 2s ease-in-out infinite;
            }
           
            .error-info {
                display: flex;
                flex-direction: column;
                gap: 2px;
            }
           
            .error-title {
                font-weight: 700;
                font-size: 16px;
                color: #dc2626;
                letter-spacing: -0.01em;
            }
           
            .error-subtitle {
                font-size: 11px;
                color: #991b1b;
                font-weight: 500;
                text-transform: uppercase;
                letter-spacing: 0.05em;
            }
           
            .verifeed-error-popup .close-btn {
                background: rgba(255, 255, 255, 0.9);
                border: none;
                color: #6b7280;
                cursor: pointer;
                width: 28px;
                height: 28px;
                border-radius: 6px;
                display: flex;
                align-items: center;
                justify-content: center;
                transition: all 0.2s ease;
                position: relative;
                z-index: 1;
            }
           
            .verifeed-error-popup .close-btn:hover {
                background: white;
                color: #374151;
                transform: rotate(90deg);
            }
           
            .verifeed-error-popup .error-body {
                padding: 24px;
            }
           
            .error-message {
                display: flex;
                gap: 12px;
                align-items: flex-start;
                background: #fef2f2;
                padding: 16px;
                border-radius: 8px;
                border-left: 4px solid #ef4444;
                margin-bottom: 20px;
            }
           
            .error-message svg {
                flex-shrink: 0;
                margin-top: 2px;
            }
           
            .error-message p {
                margin: 0;
                font-size: 13px;
                color: #6b7280;
                line-height: 1.6;
                flex: 1;
            }
           
            .verifeed-error-popup .retry-btn {
                background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 8px;
                font-size: 13px;
                font-weight: 600;
                cursor: pointer;
                width: 100%;
                display: flex;
                align-items: center;
                justify-content: center;
                gap: 8px;
                transition: all 0.3s ease;
                box-shadow: 0 4px 12px rgba(102, 126, 234, 0.3);
            }
           
            .verifeed-error-popup .retry-btn:hover {
                background: linear-gradient(135deg, #6a4190 0%, #5a6fd8 100%);
                transform: translateY(-2px);
                box-shadow: 0 6px 16px rgba(102, 126, 234, 0.4);
            }
           
            .verifeed-error-popup .retry-btn:active {
                transform: translateY(0);
            }
        `;

    document.head.appendChild(style);
    document.body.appendChild(errorPopup);
    this.activePopup = errorPopup;
    this.activeStyle = style;

    const closeBtn = errorPopup.querySelector(".close-btn");
    const retryBtn = errorPopup.querySelector(".retry-btn");

    const closeErrorPopup = () => {
      errorPopup.style.animation = "errorExit 0.3s ease-out forwards";

      const exitKeyframes = `
                @keyframes errorExit {
                    0% { opacity: 1; transform: scale(1); }
                    100% { opacity: 0; transform: scale(0.9); }
                }
            `;

      if (!document.getElementById("verifeed-error-exit-animation")) {
        const exitStyle = document.createElement("style");
        exitStyle.id = "verifeed-error-exit-animation";
        exitStyle.textContent = exitKeyframes;
        document.head.appendChild(exitStyle);
      }

      setTimeout(() => {
        if (errorPopup.parentNode) errorPopup.remove();
        if (style.parentNode) style.remove();
        this.activePopup = null;
        this.activeStyle = null;
        window.removeEventListener("scroll", this.scrollCloseHandler);
        this.scrollCloseHandler = null;
      }, 300);
    };

    closeBtn.addEventListener("click", closeErrorPopup);
    retryBtn.addEventListener("click", closeErrorPopup);
    this.setupScrollCloseListener();

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

    if (this.scrollCloseHandler) {
      window.removeEventListener("scroll", this.scrollCloseHandler);
      this.scrollCloseHandler = null;
      this.scrollCloseQueued = false;
    }

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

// ===== INITIALIZATION =====
let veriFeedInstance = null;

function initializeVeriFeed() {
  if (window.location.hostname.includes("facebook.com") && !veriFeedInstance) {
    console.log("🔮 Initializing VeriFeed Predictor (SECURED + OPTIMIZED)...");
    veriFeedInstance = new VeriFeedPredictor();
  }
}

chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  if (request.action === "getStatus") {
    sendResponse({
      enabled: veriFeedInstance ? veriFeedInstance.isEnabled : false,
      initialized: !!veriFeedInstance,
      videoCount: veriFeedInstance ? veriFeedInstance.analyzedVideos.size : 0,
      serverUrl: veriFeedInstance
        ? veriFeedInstance.serverUrl
        : "http://localhost:5000",
      targetFrames: veriFeedInstance ? veriFeedInstance.TARGET_FRAMES : 150,
      targetFps: veriFeedInstance ? veriFeedInstance.TARGET_FPS : 5,
    });
  }

  return true;
});

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", initializeVeriFeed);
} else {
  initializeVeriFeed();
}

setInterval(() => {
  if (veriFeedInstance && veriFeedInstance.isEnabled) {
    veriFeedInstance.scanForVideos();
  }
}, 5000);

// ===== DEBUG UTILITIES =====
window.checkServer = async function () {
  if (veriFeedInstance) {
    await veriFeedInstance.checkServerHealth();
  } else {
    console.log("❌ VeriFeed not initialized");
  }
};

window.getVeriFeedStatus = function () {
  if (veriFeedInstance) {
    console.log("=== VERIFEED PREDICTOR STATUS ===");
    console.log(`Enabled: ${veriFeedInstance.isEnabled}`);
    console.log(`Target Frames: ${veriFeedInstance.TARGET_FRAMES}`);
    console.log(`Target FPS: ${veriFeedInstance.TARGET_FPS}`);
    console.log(`Extract Duration: ${veriFeedInstance.EXTRACT_DURATION}s`);
    console.log(`Videos Found: ${veriFeedInstance.analyzedVideos.size}`);
    console.log(`Videos on page: ${document.querySelectorAll("video").length}`);
    console.log(`Server URL: ${veriFeedInstance.serverUrl}`);
  } else {
    console.log("❌ VeriFeed not initialized");
  }
};

window.forceRescan = function () {
  if (veriFeedInstance) {
    console.log("🔄 Forcing video scan...");
    veriFeedInstance.scanForVideos();
  } else {
    console.log("❌ VeriFeed not initialized");
  }
};

window.testPrediction = async function () {
  console.log("🧪 Testing prediction on first video...");
  const video = document.querySelector("video");
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

console.log("=== VERIFEED PREDICTION MODE (SECURED + OPTIMIZED) ===");
console.log("🔒 Authentication: JWT Token with auto-refresh");
console.log("🚀 Optimizations: Fast frame extraction (seek-based)");
console.log("📊 Configuration: 150 frames at 5fps from first 30 seconds");
console.log("⚡ Performance Features:");
console.log("   • Seek-based extraction (3-5x faster)");
console.log("   • Lower JPEG quality (0.7 for network speed)");
console.log("   • Smart frame sampling");
console.log("   • Cached face detection");
console.log("   • Mixed precision inference (CUDA)");
console.log("🔗 Connects to secure backend at http://localhost:5000");
console.log("");
console.log("🎮 Available Commands:");
console.log("  checkServer()          - Check backend server health");
console.log("  getVeriFeedStatus()    - Show current status");
console.log("  forceRescan()          - Force scan for videos");
console.log("  testPrediction()       - Test prediction on first video");
console.log("");
console.log("💡 Usage:");
console.log(
  "  1. Ensure secured backend is running (python app8_optimized.py)"
);
console.log("  2. Scroll to find videos on Facebook");
console.log("  3. Click purple 'Verifeed' button");
console.log("  4. View real-time deepfake analysis results");
console.log("");
