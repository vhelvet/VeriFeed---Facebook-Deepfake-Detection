// VeriFeed Background Script - Handles communication and state management

class VeriFeedBackground {
    constructor() {
        this.serverStatus = 'unknown';
        this.analysisQueue = [];
        this.isProcessing = false;
        this.settings = {
            enabled: true,
            confidenceThreshold: 40,
            autoAnalyze: true,
            showIndicators: true
        };

        this.init();
    }

    async init() {
        console.log('VeriFeed background script initialized');
        await this.loadSettings();
        this.checkServerStatus();
        this.setupPeriodicChecks();
    }

    async loadSettings() {
        const storedSettings = await chrome.storage.local.get(['verifeedSettings']);
        if (storedSettings.verifeedSettings) {
            this.settings = { ...this.settings, ...storedSettings.verifeedSettings };
        }
        
        // Store initial settings
        await chrome.storage.local.set({ 
            verifeedSettings: this.settings,
            verifeedEnabled: this.settings.enabled
        });
    }

    async saveSettings() {
        await chrome.storage.local.set({ 
            verifeedSettings: this.settings,
            verifeedEnabled: this.settings.enabled
        });
    }

    async checkServerStatus() {
        try {
            const response = await fetch('http://localhost:5000/health', {
                method: 'GET',
                timeout: 5000
            });
            
            this.serverStatus = response.ok ? 'online' : 'offline';
        } catch (error) {
            this.serverStatus = 'offline';
            console.warn('VeriFeed server is offline:', error);
        }

        // Update badge based on server status
        this.updateBadge();
    }

    updateBadge() {
        const text = this.serverStatus === 'online' ? '✓' : '⚠️';
        const color = this.serverStatus === 'online' ? '#28a745' : '#dc3545';
        
        chrome.action.setBadgeText({ text });
        chrome.action.setBadgeBackgroundColor({ color });
    }

    setupPeriodicChecks() {
        // Check server status every 30 seconds
        setInterval(() => this.checkServerStatus(), 30000);
        
        // Process analysis queue every 5 seconds
        setInterval(() => this.processQueue(), 5000);
    }

    async processVideoAnalysis(videoUrl, tabId) {
        if (this.serverStatus !== 'online') {
            console.warn('Cannot process video: server offline');
            return null;
        }

        try {
            const response = await fetch('http://localhost:5000/analyze', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    videoUrl: videoUrl,
                    platform: 'facebook',
                    tabId: tabId
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }

            const result = await response.json();
            return result;

        } catch (error) {
            console.error('Video analysis failed:', error);
            return null;
        }
    }

    addToQueue(videoData) {
        this.analysisQueue.push(videoData);
        if (!this.isProcessing) {
            this.processQueue();
        }
    }

    async processQueue() {
        if (this.isProcessing || this.analysisQueue.length === 0) return;

        this.isProcessing = true;
        const videoData = this.analysisQueue.shift();

        try {
            const result = await this.processVideoAnalysis(videoData.videoUrl, videoData.tabId);
            
            if (result) {
                // Send result back to content script
                chrome.tabs.sendMessage(videoData.tabId, {
                    action: 'analysisResult',
                    videoUrl: videoData.videoUrl,
                    result: result
                });
            }

        } catch (error) {
            console.error('Queue processing error:', error);
        } finally {
            this.isProcessing = false;
            if (this.analysisQueue.length > 0) {
                setTimeout(() => this.processQueue(), 1000);
            }
        }
    }

    handleMessage(request, sender, sendResponse) {
        switch (request.action) {
            case 'analyzeVideo':
                this.addToQueue({
                    videoUrl: request.videoUrl,
                    tabId: sender.tab.id
                });
                sendResponse({ success: true, queued: true });
                break;

            case 'getServerStatus':
                sendResponse({ status: this.serverStatus });
                break;

            case 'getSettings':
                sendResponse({ settings: this.settings });
                break;

            case 'updateSettings':
                this.settings = { ...this.settings, ...request.settings };
                this.saveSettings();
                
                // Notify all tabs about settings change
                chrome.tabs.query({}, (tabs) => {
                    tabs.forEach(tab => {
                        chrome.tabs.sendMessage(tab.id, {
                            action: 'settingsUpdated',
                            settings: this.settings
                        }).catch(() => {});
                    });
                });
                
                sendResponse({ success: true });
                break;

            case 'checkHealth':
                this.checkServerStatus().then(() => {
                    sendResponse({ status: this.serverStatus });
                });
                return true; // Keep message channel open for async response

            default:
                sendResponse({ success: false, error: 'Unknown action' });
        }
    }

    setupMessageListener() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            this.handleMessage(request, sender, sendResponse);
            return true; // Keep message channel open for async responses
        });
    }

    setupTabListeners() {
        // Update badge when tab changes
        chrome.tabs.onActivated.addListener(() => {
            this.updateBadge();
        });

        // Check server status when tab updates
        chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
            if (changeInfo.status === 'complete' && tab.url?.includes('facebook.com')) {
                this.checkServerStatus();
            }
        });
    }
}

// Initialize the background service
const veriFeedBackground = new VeriFeedBackground();

// Setup listeners
veriFeedBackground.setupMessageListener();
veriFeedBackground.setupTabListeners();

// Export for testing (if needed)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VeriFeedBackground;
}
