// VeriFeed Content Script - Professional Design for Facebook Users
// Optimized for user-friendly experience with clear, simple language


class VeriFeedDetector {
    constructor() {
        this.analyzedVideos = new Map();
        this.serverUrl = 'http://localhost:5000';
        this.isEnabled = true;
        this.observer = null;
        this.activePopup = null;
        this.maxRetries = 3;
        this.retryDelay = 1000;


        this.init();
    }


    init() {
        console.log('VeriFeed initialized - professional design');
        this.loadSettings();
        this.setupMutationObserver();
        this.scanForVideos();
        console.log('Initial scan for videos triggered');
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
                    console.log('DOM mutation detected, rescanning for videos');
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
        if (!this.isEnabled) {
            console.log('VeriFeed is disabled, skipping scan');
            return;
        }


        console.log('Scanning for videos...');


        // Find all video elements
        const videos = document.querySelectorAll('video');
        console.log(`Found ${videos.length} video elements`);


        // Find all video posts/articles that contain videos
        const videoPosts = this.findVideoPosts();
        console.log(`Found ${videoPosts.length} video posts`);


        // Process each video element
        videos.forEach((videoElement, index) => {
            if (this.analyzedVideos.has(videoElement)) {
                console.log(`Video #${index} already analyzed, skipping`);
                return;
            }


            // Find the video post container
            let container = this.findVideoPostContainer(videoElement);
            if (!container) {
                console.log(`No video post container found for video #${index}, skipping`);
                return;
            }


            if (container.querySelector('.verifeed-verify-btn')) {
                console.log(`Verify button already exists in container for video #${index}, skipping`);
                return;
            }


            console.log(`Adding verify button to video post #${index}`);
            this.addVerifyButton(container, videoElement);
        });


        // Process video posts that might not have video elements loaded yet
        videoPosts.forEach((post, index) => {
            if (post.querySelector('.verifeed-verify-btn')) {
                return; // Button already exists
            }


            const videoElement = post.querySelector('video');
            if (videoElement && !this.analyzedVideos.has(videoElement)) {
                console.log(`Found video in post #${index}, adding button`);
                this.addVerifyButton(post, videoElement);
            }
        });
    }


    findVideoPosts() {
        // Find all video posts using multiple selectors
        const selectors = [
            '[data-pagelet*="video"]',
            '[data-pagelet*="FeedUnit"]',
            '[role="article"]',
            '[data-ft*="video"]',
            '[data-pagelet*="permalink"]',
            '[data-pagelet*="root"]',
            '[data-pagelet*="timeline"]',
            '[data-pagelet*="main_column"]',
            '[data-pagelet*="content"]'
        ];


        const posts = new Set();


        selectors.forEach(selector => {
            document.querySelectorAll(selector).forEach(element => {
                // Check if this element contains or is likely to contain a video
                if (element.querySelector('video') ||
                    element.textContent?.includes('video') ||
                    element.getAttribute('data-ft')?.includes('video') ||
                    element.getAttribute('data-pagelet')?.includes('video')) {
                    posts.add(element);
                }
            });
        });


        return Array.from(posts);
    }


    findVideoPostContainer(videoElement) {
        // Start from the video element and go up to find the post container
        let element = videoElement.parentElement;
        let attempts = 0;
        const maxAttempts = 15;


        while (element && attempts < maxAttempts) {
            // Check if this element looks like a video post
            const hasVideoContent = element.querySelector('video') ||
                                  element.textContent?.includes('video') ||
                                  element.getAttribute('data-ft')?.includes('video');


            const hasPostStructure = element.querySelector('[data-ad-preview="message"]') ||
                                   element.querySelector('[data-pagelet="FeedUnit_0"]') ||
                                   element.querySelector('h3') ||
                                   element.querySelector('h4') ||
                                   element.querySelector('[aria-label*="video"]') ||
                                   element.querySelector('[role="button"]');


            // Check if this element has the typical video post structure
            const hasVideoPostStructure = element.children.length > 3 &&
                                        (element.getAttribute('data-pagelet')?.includes('video') ||
                                         element.getAttribute('data-ft')?.includes('video') ||
                                         element.getAttribute('role') === 'article');


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
        // Check if button already exists
        if (container.querySelector('.verifeed-verify-btn')) {
            console.log('Verify button already exists in container');
            return;
        }


        const verifyBtn = document.createElement('button');
        verifyBtn.className = 'verifeed-verify-btn';
        verifyBtn.innerHTML = `
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" style="margin-right: 4px;">
                <path d="M9 12l2 2 4-4"/>
                <circle cx="12" cy="12" r="10"/>
            </svg>
            <span>VeriFeed</span>
        `;


        // Find the post header area where the menu button is located
        const postHeader = container.querySelector('h3, h4, [data-ad-preview="message"]')?.closest('div');
        const targetContainer = postHeader || container;


        // Ensure target container has relative positioning
        const targetContainerStyle = window.getComputedStyle(targetContainer);
        if (targetContainerStyle.position === 'static') {
            targetContainer.style.position = 'relative';
        }


        // Find the three dots menu button in the post header
        const menuButton = targetContainer.querySelector('[aria-label*="more"], [aria-label*="options"], [aria-label*="menu"]');
        let buttonPosition = '60px'; // fallback position


        if (menuButton) {
            // Position button to the left of the menu button in the post header area
            const menuRect = menuButton.getBoundingClientRect();
            const targetRect = targetContainer.getBoundingClientRect();
            const relativeRight = targetRect.right - menuRect.right + menuRect.width + 8; // 8px gap
            buttonPosition = `${relativeRight}px`;
            console.log(`Found menu button in post header, positioning VeriFeed button at ${buttonPosition} from right`);
        } else {
            console.log('Menu button not found in post header, using fallback positioning');
        }


        // Professional styling with original gradient colors
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


        // Hover effect
        verifyBtn.onmouseenter = () => {
            verifyBtn.style.background = 'linear-gradient(135deg, #5a6fd8 0%, #6a4190 100%)';
            verifyBtn.style.transform = 'translateY(-1px)';
        };
        verifyBtn.onmouseleave = () => {
            verifyBtn.style.background = 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)';
            verifyBtn.style.transform = 'translateY(0)';
        };


        verifyBtn.onclick = (e) => {
            e.stopPropagation();
            this.handleVerifyClick(container, videoElement, verifyBtn);
        };


        console.log('Adding VeriFeed button to post header beside menu button');
        targetContainer.appendChild(verifyBtn);


        // Make button visible immediately
        verifyBtn.style.display = 'inline-flex';


        // Force reposition after a short delay
        setTimeout(() => {
            const updatedMenuButton = targetContainer.querySelector('[aria-label*="more"], [aria-label*="options"], [aria-label*="menu"]');
            if (updatedMenuButton) {
                const menuRect = updatedMenuButton.getBoundingClientRect();
                const targetRect = targetContainer.getBoundingClientRect();
                const relativeRight = targetRect.right - menuRect.right + menuRect.width + 8;
                verifyBtn.style.right = `${relativeRight}px`;
            }
            verifyBtn.style.left = 'auto';
            verifyBtn.style.position = 'absolute';
            console.log('Reinforced button positioning');
        }, 100);


        this.analyzedVideos.set(videoElement, {
            container,
            button: verifyBtn
        });


        console.log('VeriFeed button added successfully');
    }


    async handleVerifyClick(container, videoElement, buttonElement) {
        console.log('Starting video verification process...');


        // Update button to show loading
        const originalContent = buttonElement.innerHTML;
        buttonElement.innerHTML = `
            <div style="width: 12px; height: 12px; border: 2px solid rgba(255,255,255,0.3); border-top: 2px solid white; border-radius: 50%; animation: spin 1s linear infinite; margin-right: 4px;"></div>
            <span>Checking...</span>
            <style>@keyframes spin { 0% { transform: rotate(0deg); } 100% { transform: rotate(360deg); } }</style>
        `;
        buttonElement.disabled = true;


        // Check server status first
        try {
            console.log('Checking server health...');
            const healthResponse = await this.makeRequest(`${this.serverUrl}/health`, 'GET');
            if (!healthResponse.ok) {
                throw new Error('Server offline');
            }
            const healthData = await healthResponse.json();
            console.log('Server health check successful:', healthData);
           
            if (healthData.status !== 'healthy') {
                throw new Error('Server not in healthy state');
            }
        } catch (error) {
            console.error('Server health check failed:', error);
            this.showErrorPopup(buttonElement, 'Cannot connect to video checker. Please try again later.');
            buttonElement.innerHTML = originalContent;
            buttonElement.disabled = false;
            return;
        }


        try {
            console.log('Extracting frames from video...');
            const frames = await this.extractFrames(videoElement, 100);
            if (!frames || frames.length === 0) {
                throw new Error('Could not extract frames from video');
            }
            console.log(`Successfully extracted ${frames.length} frames`);


            // Prepare request data
            const requestData = {
                frames: frames,
                platform: 'facebook'
            };


            console.log('Sending analysis request to backend...');
            const response = await this.makeRequest(`${this.serverUrl}/frame_analyze`, 'POST', requestData);


            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ error: 'Unknown error' }));
                throw new Error(errorData.error || `Analysis failed: ${response.status}`);
            }


            const analysisData = await response.json();
            console.log('Analysis complete:', analysisData);


            // Reset button
            buttonElement.innerHTML = originalContent;
            buttonElement.disabled = false;


            // Show results popup
            this.showResultsPopup(buttonElement, analysisData);


        } catch (error) {
            console.error('Video verification error:', error);
            this.showErrorPopup(buttonElement, `Check failed: ${error.message}`);
            buttonElement.innerHTML = originalContent;
            buttonElement.disabled = false;
        }
    }


    async makeRequest(url, method = 'GET', data = null, retries = 0) {
        try {
            const options = {
                method: method,
                headers: {
                    'Content-Type': 'application/json',
                }
            };


            if (data) {
                options.body = JSON.stringify(data);
            }


            const response = await fetch(url, options);
            return response;


        } catch (error) {
            if (retries < this.maxRetries) {
                console.log(`Request failed, retrying... (${retries + 1}/${this.maxRetries})`);
                await new Promise(resolve => setTimeout(resolve, this.retryDelay * (retries + 1)));
                return this.makeRequest(url, method, data, retries + 1);
            }
            throw error;
        }
    }


    async extractFrames(videoElement, numFrames = 100) {
        return new Promise((resolve, reject) => {
            try {
                console.log(`Starting frame extraction - target: ${numFrames} frames`);
               
                const canvas = document.createElement('canvas');
                const ctx = canvas.getContext('2d');
                canvas.width = 112;
                canvas.height = 112;


                const frames = [];
                const duration = videoElement.duration;
               
                if (!duration || duration <= 0) {
                    reject(new Error('Video duration not available'));
                    return;
                }


                console.log(`Video duration: ${duration}s, extracting ${numFrames} frames`);


                let currentFrame = 0;
                const interval = duration / numFrames;
                let frameExtractionStart = Date.now();


                const extractNextFrame = () => {
                    if (currentFrame >= numFrames) {
                        const extractionTime = Date.now() - frameExtractionStart;
                        console.log(`Frame extraction complete: ${frames.length} frames in ${extractionTime}ms`);
                        resolve(frames);
                        return;
                    }


                    const timeToSeek = (currentFrame * interval);
                    videoElement.currentTime = timeToSeek;


                    const onSeeked = () => {
                        videoElement.removeEventListener('seeked', onSeeked);
                       
                        try {
                            ctx.drawImage(videoElement, 0, 0, canvas.width, canvas.height);
                            const dataURL = canvas.toDataURL('image/jpeg', 0.8);
                            const base64Data = dataURL.split(',')[1];
                            frames.push(base64Data);
                           
                            if (currentFrame % 10 === 0) {
                                console.log(`Extracted frame ${currentFrame}/${numFrames}`);
                            }
                           
                            currentFrame++;
                            setTimeout(extractNextFrame, 50);
                        } catch (error) {
                            reject(new Error(`Failed to extract frame ${currentFrame}: ${error.message}`));
                        }
                    };


                    const onError = () => {
                        videoElement.removeEventListener('error', onError);
                        reject(new Error(`Video seek error at frame ${currentFrame}`));
                    };


                    videoElement.addEventListener('seeked', onSeeked);
                    videoElement.addEventListener('error', onError);


                    setTimeout(() => {
                        videoElement.removeEventListener('seeked', onSeeked);
                        videoElement.removeEventListener('error', onError);
                        if (currentFrame < numFrames) {
                            console.warn(`Seek timeout for frame ${currentFrame}, continuing...`);
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
        const isAuthentic = prediction === 'REAL';


        // Get button position for popup placement
        const buttonRect = buttonElement.getBoundingClientRect();
       
        const resultsPopup = document.createElement('div');
        resultsPopup.className = 'verifeed-results-popup';
       
        // Determine result display
        const statusIcon = isAuthentic ? '✅' : '⚠️';
        const statusText = isAuthentic ? 'Authentic' : 'Deepfake Detected';
        const statusColor = isAuthentic ? '#10b981' : '#f59e0b';
        const confidenceText = confidence > 80 ? 'We are very confident' :
                              confidence > 60 ? 'We are somewhat confident' : 'We are not very confident';


        resultsPopup.innerHTML = `
            <div class="verifeed-popup-content">
                <div class="verifeed-popup-header">
                    <span class="status-icon">${statusIcon}</span>
                    <span class="status-text">${statusText}</span>
                    <button class="close-btn" onclick="this.closest('.verifeed-results-popup').remove()">×</button>
                </div>
                <div class="verifeed-popup-body">
                    <div class="confidence-section">
                        <span class="confidence-label">How sure we are: ${confidence.toFixed(0)}%</span>
                        <div class="confidence-bar">
                            <div class="confidence-fill" style="width: ${confidence}%; background: ${statusColor};"></div>
                        </div>
                        <span class="confidence-text">${confidenceText}</span>
                    </div>
                    <div class="info-text">
                        ${isAuthentic ?
                            'This video is genuine and has not been edited by artificial intelligence.' :
                            'This video may have been edited or created by artificial intelligence. Please verify it before sharing.'
                        }
                    </div>
                    <div class="disclaimer">
                        Computer check • This is just a guess • Always check with other sources
                    </div>
                </div>
            </div>
        `;


        // Position popup below button
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
            border: 1px solid #e5e7eb !important;
            animation: slideDown 0.2s ease-out !important;
        `;


        // Add styles
        const style = document.createElement('style');
        style.textContent = `
            @keyframes slideDown {
                from { opacity: 0; transform: translateY(-10px); }
                to { opacity: 1; transform: translateY(0); }
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
            .status-icon {
                font-size: 16px !important;
                margin-right: 8px !important;
            }
            .status-text {
                font-weight: 600 !important;
                color: #374151 !important;
                font-size: 14px !important;
                flex: 1 !important;
            }
            .close-btn {
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
            .close-btn:hover {
                color: #6b7280 !important;
            }
            .verifeed-popup-body {
                padding: 16px !important;
            }
            .confidence-section {
                margin-bottom: 12px !important;
            }
            .confidence-label {
                font-size: 13px !important;
                font-weight: 600 !important;
                color: #374151 !important;
                display: block !important;
                margin-bottom: 6px !important;
            }
            .confidence-bar {
                width: 100% !important;
                height: 6px !important;
                background: #e5e7eb !important;
                border-radius: 3px !important;
                overflow: hidden !important;
                margin-bottom: 4px !important;
            }
            .confidence-fill {
                height: 100% !important;
                border-radius: 3px !important;
                transition: width 0.8s ease-out !important;
            }
            .confidence-text {
                font-size: 12px !important;
                color: #6b7280 !important;
            }
            .info-text {
                font-size: 13px !important;
                color: #4b5563 !important;
                line-height: 1.4 !important;
                margin-bottom: 12px !important;
            }
            .disclaimer {
                font-size: 11px !important;
                color: #9ca3af !important;
                text-align: center !important;
                line-height: 1.3 !important;
                padding-top: 8px !important;
                border-top: 1px solid #f3f4f6 !important;
            }
        `;
        document.head.appendChild(style);


        document.body.appendChild(resultsPopup);
        this.activePopup = resultsPopup;


        // Auto-close after 15 seconds
        setTimeout(() => {
            if (resultsPopup.parentNode) {
                resultsPopup.remove();
                style.remove();
            }
        }, 15000);


        // Close on click outside
        setTimeout(() => {
            const closeOnOutside = (e) => {
                if (!resultsPopup.contains(e.target) && !buttonElement.contains(e.target)) {
                    resultsPopup.remove();
                    style.remove();
                    document.removeEventListener('click', closeOnOutside);
                }
            };
            document.addEventListener('click', closeOnOutside);
        }, 100);
    }


    showErrorPopup(buttonElement, message) {
        this.removeExistingPopup();


        const buttonRect = buttonElement.getBoundingClientRect();
       
        const errorPopup = document.createElement('div');
        errorPopup.className = 'verifeed-error-popup';
        errorPopup.innerHTML = `
            <div class="error-content">
                <div class="error-header">
                    <span>⚠️ Cannot check video</span>
                    <button class="close-btn" onclick="this.closest('.verifeed-error-popup').remove()">×</button>
                </div>
                <div class="error-body">
                    <p>${message}</p>
                    <button class="retry-btn" onclick="this.closest('.verifeed-error-popup').remove();">OK</button>
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
        `;


        const errorStyle = document.createElement('style');
        errorStyle.textContent = `
            .error-content {
                padding: 0 !important;
            }
            .error-header {
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
            .error-body {
                padding: 16px !important;
            }
            .error-body p {
                margin: 0 0 12px 0 !important;
                font-size: 13px !important;
                color: #6b7280 !important;
                line-height: 1.4 !important;
            }
            .retry-btn {
                background: #1877f2 !important;
                color: white !important;
                border: none !important;
                padding: 6px 12px !important;
                border-radius: 4px !important;
                font-size: 12px !important;
                cursor: pointer !important;
                font-weight: 500 !important;
            }
        `;
        document.head.appendChild(errorStyle);


        document.body.appendChild(errorPopup);
        this.activePopup = errorPopup;


        setTimeout(() => {
            if (errorPopup.parentNode) {
                errorPopup.remove();
                errorStyle.remove();
            }
        }, 8000);
    }


    removeExistingPopup() {
        const existingPopups = document.querySelectorAll('.verifeed-results-popup, .verifeed-error-popup');
        existingPopups.forEach(popup => popup.remove());
        this.activePopup = null;
    }


    destroy() {
        if (this.observer) {
            this.observer.disconnect();
        }
        this.removeExistingPopup();
        this.analyzedVideos.clear();
        console.log('VeriFeed detector destroyed');
    }
}


// Initialize VeriFeed when page loads
let veriFeedInstance = null;


function initializeVeriFeed() {
    if (window.location.hostname.includes('facebook.com') && !veriFeedInstance) {
        console.log('Initializing VeriFeed for Facebook...');
        veriFeedInstance = new VeriFeedDetector();
    }
}


// Listen for messages from background script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('Received message:', request);


    if (request.action === 'toggleVeriFeed') {
        if (veriFeedInstance) {
            veriFeedInstance.isEnabled = request.enabled;
            console.log(`VeriFeed ${request.enabled ? 'enabled' : 'disabled'}`);
            if (!request.enabled) {
                veriFeedInstance.destroy();
                veriFeedInstance = null;
            }
        } else if (request.enabled) {
            initializeVeriFeed();
        }
        sendResponse({success: true, enabled: request.enabled});
    }


    if (request.action === 'analyzeSpecificVideo') {
        if (veriFeedInstance && veriFeedInstance.isEnabled) {
            const videoElement = document.querySelector(request.videoSelector);
            if (videoElement) {
                const container = videoElement.closest('[role="article"], [data-pagelet*="video"]');
                if (container) {
                    console.log('Analyzing specific video...');
                    veriFeedInstance.handleVerifyClick(container, videoElement);
                    sendResponse({success: true, message: 'Analysis started'});
                } else {
                    sendResponse({success: false, error: 'Container not found'});
                }
            } else {
                sendResponse({success: false, error: 'Video element not found'});
            }
        } else {
            sendResponse({success: false, error: 'VeriFeed not enabled or not initialized'});
        }
    }


    if (request.action === 'getStatus') {
        sendResponse({
            enabled: veriFeedInstance ? veriFeedInstance.isEnabled : false,
            initialized: !!veriFeedInstance,
            videoCount: veriFeedInstance ? veriFeedInstance.analyzedVideos.size : 0,
            serverUrl: veriFeedInstance ? veriFeedInstance.serverUrl : 'http://localhost:5000'
        });
    }


    return true;
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


// Enhanced button positioning fix
function fixVeriFeedButtonPositioning() {
    const veriFeedButtons = document.querySelectorAll('.verifeed-verify-btn');
    console.log(`Fixing positioning for ${veriFeedButtons.length} buttons`);


    veriFeedButtons.forEach((button, index) => {
        const postContainer = button.closest('[role="article"], [data-pagelet*="video"], [data-pagelet*="FeedUnit"]');
        if (!postContainer) return;


        const postHeader = postContainer.querySelector('h3, h4, [data-ad-preview="message"]')?.closest('div');
        const targetContainer = postHeader || postContainer;


        const menuButton = targetContainer.querySelector('[aria-label*="more"], [aria-label*="options"], [aria-label*="menu"]');


        if (menuButton && targetContainer.contains(button)) {
            const menuRect = menuButton.getBoundingClientRect();
            const targetRect = targetContainer.getBoundingClientRect();
            const relativeRight = targetRect.right - menuRect.right + menuRect.width + 8;


            button.style.position = 'absolute';
            button.style.top = '12px';
            button.style.right = `${relativeRight}px`;
            button.style.left = 'auto';
            button.style.zIndex = '2147483647';


            console.log(`Fixed button ${index + 1} position`);
        }
    });
}


// Run positioning fixes
fixVeriFeedButtonPositioning();
setTimeout(fixVeriFeedButtonPositioning, 1000);


// Set up mutation observer for positioning fixes
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
    subtree: true
});


// Handle page navigation (for SPAs like Facebook)
let currentUrl = window.location.href;
const urlObserver = new MutationObserver(() => {
    if (window.location.href !== currentUrl) {
        currentUrl = window.location.href;
        console.log('URL changed, reinitializing VeriFeed...');
        setTimeout(() => {
            if (veriFeedInstance && veriFeedInstance.isEnabled) {
                veriFeedInstance.scanForVideos();
            }
        }, 1000);
    }
});


urlObserver.observe(document.body, {
    childList: true,
    subtree: true
});


// Global error handler for unhandled errors
window.addEventListener('error', (event) => {
    if (event.error && event.error.message && event.error.message.includes('verifeed')) {
        console.error('VeriFeed error:', event.error);
    }
});


console.log('VeriFeed content script fully loaded with professional design for Facebook users');

