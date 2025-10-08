// VeriFeed Popup Script
class VeriFeedPopup {
    constructor() {
        this.serverUrl = 'http://localhost:5000';
        this.settings = {
            verifeedEnabled: true,
            autoAnalyze: true,
            showIndicators: true,
            confidenceThreshold: 40
        };
        
        this.init();
    }

    async init() {
        console.log('[VeriFeed Popup] Initializing...');
        
        // Load settings first
        await this.loadSettings();
        
        // Setup UI
        this.setupEventListeners();
        this.updateUI();
        
        // Check server status
        this.checkServerStatus();
        
        // Auto-refresh status every 10 seconds
        setInterval(() => this.checkServerStatus(), 10000);
    }

    async loadSettings() {
        return new Promise((resolve) => {
            chrome.storage.local.get([
                'verifeedEnabled',
                'autoAnalyze', 
                'showIndicators',
                'confidenceThreshold',
                'serverUrl'
            ], (result) => {
                this.settings = {
                    verifeedEnabled: result.verifeedEnabled !== false,
                    autoAnalyze: result.autoAnalyze !== false,
                    showIndicators: result.showIndicators !== false,
                    confidenceThreshold: result.confidenceThreshold || 40
                };
                this.serverUrl = result.serverUrl || 'http://localhost:5000';
                resolve();
            });
        });
    }

    async saveSettings() {
        return new Promise((resolve) => {
            chrome.storage.local.set({
                ...this.settings,
                serverUrl: this.serverUrl
            }, resolve);
        });
    }

    setupEventListeners() {
        // Enable/Disable toggle
        const toggleEnabled = document.getElementById('toggleEnabled');
        toggleEnabled.checked = this.settings.verifeedEnabled;
        toggleEnabled.addEventListener('change', async (e) => {
            this.settings.verifeedEnabled = e.target.checked;
            await this.saveSettings();
            this.notifyContentScript();
            this.updateUI();
        });

        // Auto-analyze toggle
        const toggleAutoAnalyze = document.getElementById('toggleAutoAnalyze');
        toggleAutoAnalyze.checked = this.settings.autoAnalyze;
        toggleAutoAnalyze.addEventListener('change', async (e) => {
            this.settings.autoAnalyze = e.target.checked;
            await this.saveSettings();
            this.notifyContentScript();
        });

        // Show indicators toggle
        const toggleShowIndicators = document.getElementById('toggleShowIndicators');
        toggleShowIndicators.checked = this.settings.showIndicators;
        toggleShowIndicators.addEventListener('change', async (e) => {
            this.settings.showIndicators = e.target.checked;
            await this.saveSettings();
            this.notifyContentScript();
        });

        // Confidence threshold slider
        const confidenceThreshold = document.getElementById('confidenceThreshold');
        const thresholdValue = document.getElementById('thresholdValue');
        
        confidenceThreshold.value = this.settings.confidenceThreshold;
        thresholdValue.textContent = `${this.settings.confidenceThreshold}%`;
        
        confidenceThreshold.addEventListener('input', (e) => {
            const value = parseInt(e.target.value);
            thresholdValue.textContent = `${value}%`;
        });

        confidenceThreshold.addEventListener('change', async (e) => {
            this.settings.confidenceThreshold = parseInt(e.target.value);
            await this.saveSettings();
            this.notifyContentScript();
        });

        // Refresh button
        document.getElementById('btnRefresh').addEventListener('click', () => {
            this.checkServerStatus();
            this.refreshContentScript();
        });

        // Analyze Now button
        document.getElementById('btnAnalyze').addEventListener('click', () => {
            this.triggerAnalysis();
        });
    }

    updateUI() {
        const isEnabled = this.settings.verifeedEnabled;
        
        // Enable/disable other controls based on main toggle
        document.getElementById('toggleAutoAnalyze').disabled = !isEnabled;
        document.getElementById('toggleShowIndicators').disabled = !isEnabled;
        document.getElementById('confidenceThreshold').disabled = !isEnabled;
        document.getElementById('btnAnalyze').disabled = !isEnabled;

        // Update button text based on status
        const analyzeBtn = document.getElementById('btnAnalyze');
        if (!isEnabled) {
            analyzeBtn.textContent = 'Disabled';
            analyzeBtn.style.opacity = '0.5';
        } else {
            analyzeBtn.textContent = 'Analyze Now';
            analyzeBtn.style.opacity = '1';
        }
    }

    async checkServerStatus() {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const statusInfo = document.getElementById('statusInfo');

        try {
            statusText.textContent = 'Checking...';
            statusDot.className = 'status-dot';
            
            const response = await fetch(`${this.serverUrl}/health`, {
                method: 'GET',
                timeout: 5000
            });

            if (response.ok) {
                const data = await response.json();
                statusDot.className = 'status-dot online';
                statusText.textContent = 'Online';
                statusInfo.textContent = `Server ready • Model: ${data.model_loaded ? 'Loaded' : 'Not loaded'} • Device: ${data.device || 'Unknown'}`;
            } else {
                throw new Error(`Server returned ${response.status}`);
            }
        } catch (error) {
            console.error('[VeriFeed Popup] Server check failed:', error);
            statusDot.className = 'status-dot offline';
            statusText.textContent = 'Offline';
            statusInfo.textContent = 'Cannot connect to analysis server. Make sure the backend is running on localhost:5000';
        }
    }

    async notifyContentScript() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            if (tab && (tab.url.includes('facebook.com') || tab.url.includes('fb.com'))) {
                chrome.tabs.sendMessage(tab.id, {
                    action: 'updateSettings',
                    settings: this.settings
                });
            }
        } catch (error) {
            console.error('[VeriFeed Popup] Failed to notify content script:', error);
        }
    }

    async refreshContentScript() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            if (tab && (tab.url.includes('facebook.com') || tab.url.includes('fb.com'))) {
                chrome.tabs.sendMessage(tab.id, {
                    action: 'refresh'
                });
            }
        } catch (error) {
            console.error('[VeriFeed Popup] Failed to refresh content script:', error);
        }
    }

    async triggerAnalysis() {
        if (!this.settings.verifeedEnabled) return;

        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            if (tab && (tab.url.includes('facebook.com') || tab.url.includes('fb.com'))) {
                // Send message to content script to analyze visible videos
                chrome.tabs.sendMessage(tab.id, {
                    action: 'analyzeAllVideos'
                });

                // Update button to show it's working
                const analyzeBtn = document.getElementById('btnAnalyze');
                const originalText = analyzeBtn.textContent;
                analyzeBtn.textContent = 'Analyzing...';
                analyzeBtn.disabled = true;

                setTimeout(() => {
                    analyzeBtn.textContent = originalText;
                    analyzeBtn.disabled = false;
                }, 3000);

            } else {
                alert('Please navigate to Facebook to analyze videos.');
            }
        } catch (error) {
            console.error('[VeriFeed Popup] Failed to trigger analysis:', error);
        }
    }

    async getCurrentTabInfo() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            return tab;
        } catch (error) {
            console.error('[VeriFeed Popup] Failed to get tab info:', error);
            return null;
        }
    }
}

// Initialize popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new VeriFeedPopup();
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    if (request.action === 'analysisComplete') {
        // You can update UI here if needed when analysis completes
        console.log('[VeriFeed Popup] Analysis completed:', request.result);
    }
    
    if (request.action === 'analysisError') {
        console.error('[VeriFeed Popup] Analysis error:', request.error);
    }
});