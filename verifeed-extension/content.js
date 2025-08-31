// VeriFeed Content Script - Tunai-style deepfake verification for Facebook videos

class VeriFeedDetector {
    constructor() {
        this.analyzedVideos = new Map();
        this.serverUrl = 'http://localhost:5000';
        this.isEnabled = true;
        this.observer = null;
        this.activePopup = null;
        
        this.init();
    }

    init() {
        console.log('VeriFeed initialized on Facebook');
        this.loadSettings();
        this.setupMutationObserver();
        this.scanForVideos();
    }

    loadSettings() {
        chrome.storage.local.get(['verifeedEnabled'], (result) => {
            this.isEnabled = result.verifeedEnabled !== false;
        });
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
            subtree: true
        });
    }

    scanForVideos() {
        if (!this.isEnabled) return;

        // Find video containers on Facebook (more specific targeting)
        const videoContainers = document.querySelectorAll('[data-pagelet*="video"], [role="article"] video, .fbVideo, [data-video-id]');
        
        videoContainers.forEach((container) => {
            const videoElement = container.querySelector('video');
            if (videoElement && !this.analyzedVideos.has(videoElement)) {
                this.addVerifyButton(container, videoElement);
            }
        });
    }

    addVerifyButton(container, videoElement) {
        // Check if button already exists
        if (container.querySelector('.verifeed-verify-btn')) return;

        const verifyBtn = document.createElement('button');
        verifyBtn.className = 'verifeed-verify-btn';
        verifyBtn.innerHTML = `
            <span class="verifeed-btn-icon">🔍</span>
            <span class="verifeed-btn-text">Verify</span>
        `;
        
        verifyBtn.onclick = (e) => {
            e.stopPropagation();
            this.handleVerifyClick(container, videoElement);
        };

        // Find a good position for the button (top-right corner)
        const buttonContainer = container.querySelector('[role="button"], .videoControls, .uiScaledImageContainer') || container;
        buttonContainer.style.position = 'relative';
        buttonContainer.appendChild(verifyBtn);

        this.analyzedVideos.set(videoElement, { container, button: verifyBtn });
    }

    async handleVerifyClick(container, videoElement) {
        const videoSrc = videoElement.src || videoElement.currentSrc;
        if (!videoSrc) {
            this.showError(container, 'No video source found');
            return;
        }

        // Show loading state
        this.showLoading(container);

        try {
            const response = await fetch(`${this.serverUrl}/analyze`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    videoUrl: videoSrc,
                    platform: 'facebook',
                    sequence_length: 20
                })
            });

            if (!response.ok) {
                throw new Error(`Analysis failed: ${response.status}`);
            }

            const result = await response.json();
            this.showResultsPopup(container, result);

        } catch (error) {
            console.error('Video verification error:', error);
            this.showError(container, 'Verification failed. Please try again.');
        }
    }

    showLoading(container) {
        this.removeExistingPopup(container);
        
        const loadingPopup = document.createElement('div');
        loadingPopup.className = 'verifeed-popup verifeed-loading-popup';
        loadingPopup.innerHTML = `
            <div class="verifeed-popup-content">
                <div class="verifeed-popup-header">
                    <h3>🔍 Verifying Video</h3>
                </div>
                <div class="verifeed-popup-body">
                    <div class="verifeed-loading-spinner"></div>
                    <p>Analyzing for deepfake content...</p>
                    <p class="verifeed-popup-note">This may take a few moments</p>
                </div>
            </div>
        `;

        container.appendChild(loadingPopup);
        this.activePopup = loadingPopup;
    }

    showResultsPopup(container, result) {
        this.removeExistingPopup(container);

        const confidenceClass = result.confidence > 70 ? 'high-confidence' : 
                              result.confidence > 40 ? 'medium-confidence' : 'low-confidence';
        
        const isReal = result.prediction === 'REAL';
        const resultIcon = isReal ? '✅' : '⚠️';
        const resultTitle = isReal ? 'Likely Authentic' : 'Potential Deepfake';

        const resultsPopup = document.createElement('div');
        resultsPopup.className = `verifeed-popup verifeed-results-popup ${confidenceClass}`;
        resultsPopup.innerHTML = `
            <div class="verifeed-popup-content">
                <div class="verifeed-popup-header">
                    <h3>${resultIcon} ${resultTitle}</h3>
                    <button class="verifeed-close-btn" onclick="this.parentElement.parentElement.parentElement.remove()">×</button>
                </div>
                <div class="verifeed-popup-body">
                    <div class="verifeed-result-details">
                        <div class="verifeed-confidence-meter">
                            <div class="verifeed-confidence-bar" style="width: ${result.confidence}%"></div>
                        </div>
                        <p class="verifeed-confidence-text">Confidence: ${result.confidence}%</p>
                        <p class="verifeed-analysis-text">
                            ${isReal ? 
                                'This video appears to be authentic based on our AI analysis.' : 
                                'This video shows signs of potential manipulation or deepfake content.'}
                        </p>
                    </div>
                    <div class="verifeed-popup-footer">
                        <p class="verifeed-disclaimer">
                            <small>AI-powered analysis by VeriFeed. Results are estimates and should be used as guidance only.</small>
                        </p>
                    </div>
                </div>
            </div>
        `;

        container.appendChild(resultsPopup);
        this.activePopup = resultsPopup;

        // Auto-close after 10 seconds
        setTimeout(() => {
            if (resultsPopup.parentNode) {
                resultsPopup.remove();
            }
        }, 10000);
    }

    showError(container, message) {
        this.removeExistingPopup(container);

        const errorPopup = document.createElement('div');
        errorPopup.className = 'verifeed-popup verifeed-error-popup';
        errorPopup.innerHTML = `
            <div class="verifeed-popup-content">
                <div class="verifeed-popup-header">
                    <h3>❌ Verification Error</h3>
                </div>
                <div class="verifeed-popup-body">
                    <p>${message}</p>
                    <button class="verifeed-retry-btn" onclick="this.closest('.verifeed-popup').remove()">Try Again</button>
                </div>
            </div>
        `;

        container.appendChild(errorPopup);
        this.activePopup = errorPopup;
    }

    removeExistingPopup(container) {
        const existingPopup = container.querySelector('.verifeed-popup');
        if (existingPopup) {
            existingPopup.remove();
        }
    }

    destroy() {
        if (this.observer) {
            this.observer.disconnect();
        }
        this.analyzedVideos.clear();
    }
}

// Initialize VeriFeed when page loads
let veriFeedInstance = null;

function initializeVeriFeed() {
    if (window.location.hostname.includes('facebook.com') && !veriFeedInstance) {
        veriFeedInstance = new VeriFeedDetector();
    }
}

// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'toggleVeriFeed') {
        if (veriFeedInstance) {
            veriFeedInstance.isEnabled = request.enabled;
            if (request.enabled) {
                veriFeedInstance.scanForVideos();
            } else {
                veriFeedInstance.destroy();
                veriFeedInstance = null;
            }
        }
        sendResponse({success: true});
    }
    
    if (request.action === 'analyzeSpecificVideo') {
        if (veriFeedInstance) {
            const videoElement = document.querySelector(request.videoSelector);
            if (videoElement) {
                const container = videoElement.closest('[role="article"], [data-pagelet*="video"]');
                if (container) {
                    veriFeedInstance.handleVerifyClick(container, videoElement);
                }
            }
        }
        sendResponse({success: true});
    }
});

// Initialize when DOM is ready
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', initializeVeriFeed);
} else {
    initializeVeriFeed();
}

// Re-scan periodically for new content
setInterval(() => {
    if (veriFeedInstance && veriFeedInstance.isEnabled) {
        veriFeedInstance.scanForVideos();
    }
}, 3000);
