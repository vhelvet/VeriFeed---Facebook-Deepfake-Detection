// VeriFeed Popup Script
class VeriFeedPopup {
    constructor() {
        this.serverUrl = 'http://localhost:5000';
        this.settings = {
            verifeedEnabled: true
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
                'serverUrl'
            ], (result) => {
                this.settings = {
                    verifeedEnabled: result.verifeedEnabled !== false
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

        // Refresh button
        document.getElementById('btnRefresh').addEventListener('click', () => {
            this.checkServerStatus();
            this.refreshContentScript();
        });

        // Close button
        document.getElementById('btnClose').addEventListener('click', () => {
            window.close();
        });
    }

    updateUI() {
        // No other controls to enable/disable
    }

    async checkServerStatus() {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const statusInfo = document.getElementById('statusInfo');

        try {
            statusText.textContent = 'Checking...';
            statusDot.className = 'status-dot';

            const startTime = Date.now();
            const response = await fetch(`${this.serverUrl}/health`, {
                method: 'GET',
                timeout: 5000
            });
            const latency = Date.now() - startTime;

            if (response.ok) {
                const data = await response.json();
                statusDot.className = 'status-dot online';
                statusText.textContent = 'Online';
                statusInfo.textContent = `Server ready • Model: ${data.model_loaded ? 'Loaded' : 'Not loaded'} • Device: ${data.device || 'Unknown'} • Latency: ${latency}ms`;
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
