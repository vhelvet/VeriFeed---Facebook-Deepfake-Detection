// VeriFeed Content Script - Enhanced Debug Version
// Added extensive logging to identify popup display issues

// ===== AUTH MODULE (EMBEDDED) =====
// This replaces the need to import auth.js separately in content scripts
class VerifeedAuth {
    constructor() {
        this.apiUrl = 'http://localhost:5000';
        this.apiKey = '5hTeoaOm5m-91clhe2iVqKy2jpkiN54JLQ4vNbiDodU';  // ← Use YOUR key from .env!
        this.token = null;
        this.tokenExpiry = null;
        
        this.loadToken();
    }

    async loadToken() {
        try {
            const result = await chrome.storage.local.get(['auth_token', 'token_expiry']);
            if (result.auth_token && result.token_expiry) {
                const expiry = new Date(result.token_expiry);
                if (expiry > new Date()) {
                    this.token = result.auth_token;
                    this.tokenExpiry = expiry;
                    console.log('✓ Loaded valid token from storage');
                    return true;
                } else {
                    console.log('Token expired, clearing...');
                    await this.clearToken();
                }
            }
        } catch (error) {
            console.error('Error loading token:', error);
        }
        return false;
    }

    async saveToken(token, expiresIn) {
        try {
            const expiry = new Date();
            expiry.setSeconds(expiry.getSeconds() + expiresIn);
            
            await chrome.storage.local.set({
                'auth_token': token,
                'token_expiry': expiry.toISOString()
            });
            
            this.token = token;
            this.tokenExpiry = expiry;
            
            console.log('✓ Token saved to storage');
        } catch (error) {
            console.error('Error saving token:', error);
        }
    }

    async clearToken() {
        try {
            await chrome.storage.local.remove(['auth_token', 'token_expiry']);
            this.token = null;
            this.tokenExpiry = null;
            console.log('✓ Token cleared');
        } catch (error) {
            console.error('Error clearing token:', error);
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
            console.log('Generating new JWT token...');
            
            const response = await fetch(`${this.apiUrl}/auth/token`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    api_key: this.apiKey
                })
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to generate token');
            }

            const data = await response.json();
            
            await this.saveToken(data.token, data.expires_in);
            
            console.log('✓ Token generated successfully');
            return data.token;
            
        } catch (error) {
            console.error('Error generating token:', error);
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
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        };
    }

    async authenticatedFetch(url, options = {}) {
        try {
            const headers = await this.getAuthHeaders();
            
            const response = await fetch(url, {
                ...options,
                headers: {
                    ...options.headers,
                    ...headers
                }
            });

            // If 401, try regenerating token
            if (response.status === 401) {
                console.log('Token invalid, regenerating...');
                await this.clearToken();
                
                const newHeaders = await this.getAuthHeaders();
                const retryResponse = await fetch(url, {
                    ...options,
                    headers: {
                        ...options.headers,
                        ...newHeaders
                    }
                });
                
                return retryResponse;
            }

            return response;
            
        } catch (error) {
            console.error('Authenticated fetch error:', error);
            throw error;
        }
    }
}

// Create global auth instance for this content script
const verifeedAuth = new VerifeedAuth();


/**
 * Natural Language Generation (NLG) System for Deepfake Detection Results
 * Uses slot-filling / regex template-based NLG - a recognized computational linguistics technique
 * Reference: Reiter & Dale (2000) - "Building Natural Language Generation Systems"
 */
class DeepfakeNLG {
  constructor() {
    // Linguistic templates with slots for contextual filling
    this.templates = {
      deepfake: {
        high_confidence: [
          {
            structure: "[DETERMINATION] [EVIDENCE] [ACTION]",
            determination: [
              "The analysis strongly indicates",
              "VeriFeed has identified clear signs that",
              "Evidence suggests",
              "Multiple indicators show that"
            ],
            evidence: [
              "this video has been manipulated using artificial intelligence.",
              "deepfake technology was used to create this content.",
              "AI-generated alterations are present in this video.",
              "this content was synthetically modified using AI tools."
            ],
            action: [
              "Users should verify this through other sources before sharing.",
              "Fact-checking is recommended before distributing this content.",
              "Cross-referencing with original sources is advised before sharing.",
              "Verification through trusted sources is recommended before considering it authentic."
            ]
          }
        ],
        medium_confidence: [
          {
            structure: "[POSSIBILITY] [EVIDENCE] [CAUTION]",
            possibility: [
              "This video may have been",
              "There are indicators suggesting this was",
              "The system has detected signs that this could be",
              "Analysis indicates this might have been"
            ],
            evidence: [
              "edited or created using artificial intelligence.",
              "manipulated with deepfake technology.",
              "generated using AI tools.",
              "altered using synthetic media techniques."
            ],
            caution: [
              "Verification is recommended before sharing.",
              "Additional verification is advised.",
              "Caution should be exercised when sharing this content.",
              "Fact-checking through multiple sources is recommended before sharing."
            ]
          }
        ],
        low_confidence: [
          {
            structure: "[UNCERTAINTY] [OBSERVATION] [RECOMMENDATION]",
            uncertainty: [
              "While certainty is limited,",
              "The analysis is inconclusive, but",
              "VeriFeed has limited confidence that",
              "Though uncertain,"
            ],
            observation: [
              "this video shows some signs of manipulation.",
              "there may be AI-generated elements present.",
              "artificial alterations might be present.",
              "possible synthetic modifications were detected."
            ],
            recommendation: [
              "This content should be treated with skepticism and verified through multiple sources.",
              "Fact-checking is strongly recommended before sharing.",
              "Additional analysis is needed before drawing conclusions.",
              "Additional verification methods should be consulted before trusting this content."
            ]
          }
        ]
      },
      authentic: {
        high_confidence: [
          {
            structure: "[DETERMINATION] [EVIDENCE] [ASSESSMENT]",
            determination: [
              "The analysis indicates",
              "VeriFeed has found strong evidence that",
              "Multiple factors suggest",
              "The assessment shows"
            ],
            evidence: [
              "this video is genuine and has not been digitally manipulated.",
              "this content appears authentic with no signs of AI generation.",
              "this video shows no indicators of deepfake technology.",
              "this is authentic content without synthetic alterations."
            ],
            assessment: [
              "However, important content should always be verified through trusted sources.",
              "Still, cross-referencing with original sources is good practice.",
              "Verification through official channels is still recommended when possible.",
              "As always, critical content should be verified through additional sources."
            ]
          }
        ],
        medium_confidence: [
          {
            structure: "[LIKELIHOOD] [EVIDENCE] [CAUTION]",
            likelihood: [
              "This video appears to be",
              "VeriFeed believes this is likely",
              "Evidence suggests this is probably",
              "Analysis indicates this is most likely"
            ],
            evidence: [
              "authentic and unmanipulated.",
              "genuine with no AI alterations.",
              "real content without deepfake elements.",
              "legitimate with no synthetic modifications."
            ],
            caution: [
              "though verification is recommended for complete certainty.",
              "but additional verification is always recommended.",
              "though exercising caution is still advisable.",
              "however, verification through trusted sources is advised when important."
            ]
          }
        ],
        low_confidence: [
          {
            structure: "[UNCERTAINTY] [OBSERVATION] [RECOMMENDATION]",
            uncertainty: [
              "The system cannot confidently determine",
              "The analysis is inconclusive about",
              "VeriFeed has low confidence in assessing",
              "It's unclear from the analysis"
            ],
            observation: [
              "whether this video is authentic or manipulated.",
              "if this content contains AI-generated elements.",
              "the authenticity of this video.",
              "whether synthetic alterations are present."
            ],
            recommendation: [
              "Verification through multiple trusted sources is recommended before relying on this content.",
              "This should be treated with caution until verified.",
              "Additional expert analysis may be needed for confirmation.",
              "Verification from authoritative sources is advised before trusting this content."
            ]
          }
        ]
      }
    };
  }

  /**
   * NLP Text Generation Function
   * Uses linguistic rules and context to generate varied messages
   * Implements slot-filling algorithm for template-based NLG
   */
  generate(prediction, confidence) {
    const isAuthentic = prediction === "REAL";
    const category = isAuthentic ? "authentic" : "deepfake";
    
    // Determine confidence level using linguistic thresholds
    let confidenceLevel;
    if (confidence >= 80) {
      confidenceLevel = "high_confidence";
    } else if (confidence >= 60) {
      confidenceLevel = "medium_confidence";
    } else {
      confidenceLevel = "low_confidence";
    }
    
    // Get appropriate template set
    const templateSet = this.templates[category][confidenceLevel];
    
    // Select template (randomized for linguistic variation)
    const template = templateSet[Math.floor(Math.random() * templateSet.length)];
    
    // Generate sentence parts using slot-filling / regex (NLG technique)
    const parts = template.structure.match(/\[([^\]]+)\]/g).map(slot => {
      const slotName = slot.replace(/[\[\]]/g, '').toLowerCase();
      const options = template[slotName];
      // Random selection provides linguistic variety
      return options[Math.floor(Math.random() * options.length)];
    });
    
    // Construct final sentence with proper spacing
    return parts.join(' ');
  }

  /**
   * Generate confidence description using NLG principles
   */
  generateConfidenceText(confidence) {
    if (confidence >= 90) {
      const options = [
        "VeriFeed has very high confidence in this assessment",
        "The analysis provides very strong certainty",
        "The system is highly confident in this determination",
        "This assessment has very high reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else if (confidence >= 80) {
      const options = [
        "VeriFeed has high confidence in this assessment",
        "The analysis provides strong certainty",
        "The system is confident in this determination",
        "This assessment has high reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else if (confidence >= 70) {
      const options = [
        "VeriFeed has moderate confidence in this assessment",
        "The analysis suggests reasonable certainty",
        "The system is moderately confident in this determination",
        "This assessment has moderate reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else if (confidence >= 60) {
      const options = [
        "VeriFeed has limited confidence in this assessment",
        "The analysis suggests some uncertainty",
        "The system is somewhat confident in this determination",
        "This assessment has limited reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    } else {
      const options = [
        "VeriFeed has low confidence in this assessment",
        "The analysis is highly uncertain",
        "The system has minimal confidence in this determination",
        "This assessment has low reliability"
      ];
      return options[Math.floor(Math.random() * options.length)];
    }
  }

  /**
   * Grammar correction utility (NLP function)
   */
  correctGrammar(text) {
    return text
      .replace(/\s+/g, ' ')
      .replace(/\s([.,!?])/g, '$1')
      .replace(/^./, str => str.toUpperCase())
      .trim();
  }
}

// Initialize NLG system
const deepfakeNLG = new DeepfakeNLG();


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

  isExcludedForStories(element, isReel) {
    if (isReel) return false;

    // Ultra-comprehensive exclusion logic for Facebook stories and MyDay (excluding reels)
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

    // Additional broad exclusions for story-like content (excluding reels to allow VeriFeed button)
    const isStoryLike = element.closest('[data-pagelet*="highlight"]') ||
                         element.closest('[data-pagelet*="Highlight"]') ||
                         element.closest('[aria-label*="highlight"]') ||
                         element.closest('[aria-label*="Highlight"]') ||
                         element.closest('.highlight') ||
                         element.closest('[class*="highlight"]');

    return isStory || isMyDay || isStoryLike;
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

      const dataPagelet = container.getAttribute('data-pagelet') || '';
      const isReel = dataPagelet.toLowerCase().includes('reel');

      console.log(`DEBUG: Checking video container - dataPagelet: "${dataPagelet}", isReel: ${isReel}`);

      if (this.isExcludedForStories(container, isReel)) {
        console.log("Excluding video from button addition:", {
          container: container,
          dataPagelet: container.getAttribute('data-pagelet'),
          ariaLabel: container.getAttribute('aria-label'),
          className: container.className
        });
        return;
      }

      console.log(`DEBUG: Video passed exclusion checks, proceeding to add button`);

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
      '[data-pagelet*="reel"]',
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
        const dataPagelet = element.getAttribute('data-pagelet') || '';
        const isReel = dataPagelet.toLowerCase().includes('reel');

        // Ultra-comprehensive exclusion logic for Facebook stories and MyDay (excluding reels)
        const isStory = !isReel && (element.closest('[data-pagelet*="story"]') ||
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
                       element.closest('[data-testid*="Story"]'));

        const isMyDay = !isReel && (element.closest('[data-pagelet*="myday"]') ||
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
                        element.closest('[data-testid*="Myday"]'));

        // Additional broad exclusions for story-like content (excluding reels to allow VeriFeed button)
        const isStoryLike = !isReel && (element.closest('[data-pagelet*="highlight"]') ||
                           element.closest('[data-pagelet*="Highlight"]') ||
                           element.closest('[aria-label*="highlight"]') ||
                           element.closest('[aria-label*="Highlight"]') ||
                           element.closest('.highlight') ||
                           element.closest('[class*="highlight"]'));

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
        `${this.serverUrl}/predict`,
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
      // Check if this is the /predict endpoint (needs auth)
      const needsAuth = url.includes('/predict') || url.includes('/frame_analyze');

      if (needsAuth) {
        console.log('🔐 Making direct authenticated request to:', url);

        // Use direct authenticated fetch for /predict endpoint
        const fullUrl = `${this.serverUrl}/predict`;
        const response = await verifeedAuth.authenticatedFetch(fullUrl, {
          method: 'POST',
          body: JSON.stringify(data)
        });

        // Convert fetch response to the expected format
        return {
          ok: response.ok,
          status: response.status,
          json: () => response.json()
        };
      } else {
        // Non-authenticated request (like /health) - use direct fetch
        console.log('📡 Making direct health check request to:', url);

        const response = await fetch(url, {
          method: method,
          headers: {
            'Content-Type': 'application/json'
          }
        });

        return {
          ok: response.ok,
          status: response.status,
          json: () => response.json()
        };
      }
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
        this.removeExistingPopup();


        const prediction = result.prediction;
        const confidence = result.confidence || 0;
        const realProb = result.real_probability || 0;
        const fakeProb = result.fake_probability || 0;
        const isAuthentic = prediction === "REAL";


        const buttonRect = buttonElement.getBoundingClientRect();
        const resultsPopup = document.createElement("div");
        resultsPopup.className = "verifeed-results-popup";

        // Generate NLG messages
        const nlgMessage = deepfakeNLG.generate(prediction, confidence);
        const nlgConfidenceText = deepfakeNLG.generateConfidenceText(confidence);

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
                <span>Analyzed in ${result.processing_time.total}s</span>
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
                            <span class="confidence-value" style="color: ${statusColor};">${confidence.toFixed(1)}%</span>
                        </div>
                        <p class="confidence-description">${nlgConfidenceText}</p>
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
                                    <div class="prob-value">${realProb.toFixed(1)}%</div>
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
                                    <div class="prob-value">${fakeProb.toFixed(1)}%</div>
                                    <div class="prob-bar">
                                        <div class="prob-fill prob-fill-fake" style="width: 0%;" data-width="${fakeProb}"></div>
                                    </div>
                                </div>
                            </div>
                        </div>
                    </div>
                   
                    <div class="info-card" style="border-left-color: ${statusColor};">
                        <div class="info-icon">${isAuthentic ? '📋' : '⚠️'}</div>
                        <div class="info-content">
                            <div class="info-title">${isAuthentic ? 'Summary' : 'Warning'}</div>
                            <div class="info-description">
                                ${nlgMessage}
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
                            <span>Frames: ${result.frames_processed || "N/A"}</span>
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
                font-weight: 900;
                font-size: 48px;
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
            const confidenceFill = resultsPopup.querySelector('.confidence-fill');
            const probFills = resultsPopup.querySelectorAll('.prob-fill');
           
            if (confidenceFill) {
                confidenceFill.style.width = confidenceFill.dataset.width + '%';
            }
           
            probFills.forEach(fill => {
                fill.style.width = fill.dataset.width + '%';
            });
        }, 100);


        const closeBtn = resultsPopup.querySelector(".close-btn");
        const closePopup = () => {
            resultsPopup.style.animation = 'popupExit 0.3s ease-out forwards';
           
            const exitKeyframes = `
                @keyframes popupExit {
                    0% { opacity: 1; transform: translateY(0) scale(1); }
                    100% { opacity: 0; transform: translateY(-10px) scale(0.95); }
                }
            `;
           
            if (!document.getElementById('verifeed-exit-animation')) {
                const exitStyle = document.createElement('style');
                exitStyle.id = 'verifeed-exit-animation';
                exitStyle.textContent = exitKeyframes;
                document.head.appendChild(exitStyle);
            }
           
            setTimeout(() => {
                if (resultsPopup.parentNode) resultsPopup.remove();
                if (style.parentNode) style.remove();
                this.activePopup = null;
                this.activeStyle = null;
                window.removeEventListener('scroll', this.scrollCloseHandler);
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
                            <span class="error-title">ANALYSIS FAILED</span>
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
            const confidenceFill = resultsPopup.querySelector('.confidence-fill');
            const probFills = resultsPopup.querySelectorAll('.prob-fill');
           
            if (confidenceFill) {
                confidenceFill.style.width = confidenceFill.dataset.width + '%';
            }
           
            probFills.forEach(fill => {
                fill.style.width = fill.dataset.width + '%';
            });
        }, 100);


        const closeBtn = resultsPopup.querySelector(".close-btn");
        const closePopup = () => {
            resultsPopup.style.animation = 'popupExit 0.3s ease-out forwards';
           
            const exitKeyframes = `
                @keyframes popupExit {
                    0% { opacity: 1; transform: translateY(0) scale(1); }
                    100% { opacity: 0; transform: translateY(-10px) scale(0.95); }
                }
            `;
           
            if (!document.getElementById('verifeed-exit-animation')) {
                const exitStyle = document.createElement('style');
                exitStyle.id = 'verifeed-exit-animation';
                exitStyle.textContent = exitKeyframes;
                document.head.appendChild(exitStyle);
            }
           
            setTimeout(() => {
                if (resultsPopup.parentNode) resultsPopup.remove();
                if (style.parentNode) style.remove();
                this.activePopup = null;
                this.activeStyle = null;
                window.removeEventListener('scroll', this.scrollCloseHandler);
                this.scrollCloseHandler = null;
            }, 300);
        };


        closeBtn.addEventListener("click", closePopup);
        this.setupScrollCloseListener();


        setTimeout(() => {
            if (resultsPopup.parentNode) closePopup();
        }, 20000);
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
            window.removeEventListener('scroll', this.scrollCloseHandler);
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
