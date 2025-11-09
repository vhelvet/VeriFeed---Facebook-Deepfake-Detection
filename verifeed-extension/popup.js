// VeriFeed Popup Script - Fixed Async Messaging
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

        try {
            // Load settings first
            await this.loadSettings();

            // Setup UI
            this.setupEventListeners();
            this.updateUI();

            // Check server status
            await this.checkServerStatus();

            // Auto-refresh status every 10 seconds
            setInterval(() => this.checkServerStatus(), 10000);
        } catch (error) {
            console.error('[VeriFeed Popup] Initialization error:', error);
        }
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.local.get([
                'verifeedEnabled',
                'serverUrl'
            ]);
            
            this.settings = {
                verifeedEnabled: result.verifeedEnabled !== false
            };
            this.serverUrl = result.serverUrl || 'http://localhost:5000';
            
            console.log('[VeriFeed Popup] Settings loaded:', this.settings);
        } catch (error) {
            console.error('[VeriFeed Popup] Error loading settings:', error);
            // Use defaults if loading fails
            this.settings = { verifeedEnabled: true };
            this.serverUrl = 'http://localhost:5000';
        }
    }

    async saveSettings() {
        try {
            await chrome.storage.local.set({
                ...this.settings,
                serverUrl: this.serverUrl
            });
            console.log('[VeriFeed Popup] Settings saved:', this.settings);
        } catch (error) {
            console.error('[VeriFeed Popup] Error saving settings:', error);
        }
    }

    setupEventListeners() {
        // Enable/Disable toggle
        const toggleEnabled = document.getElementById('toggleEnabled');
        if (toggleEnabled) {
            toggleEnabled.checked = this.settings.verifeedEnabled;
            toggleEnabled.addEventListener('change', async (e) => {
                this.settings.verifeedEnabled = e.target.checked;
                await this.saveSettings();
                await this.notifyContentScript();
                this.updateUI();
            });
        }

        // Refresh button
        const btnRefresh = document.getElementById('btnRefresh');
        if (btnRefresh) {
            btnRefresh.addEventListener('click', async () => {
                await this.checkServerStatus();
                await this.refreshContentScript();
            });
        }

        // Close button
        const btnClose = document.getElementById('btnClose');
        if (btnClose) {
            btnClose.addEventListener('click', () => {
                window.close();
            });
        }
    }

    updateUI() {
        // Update UI based on settings
        console.log('[VeriFeed Popup] UI updated');
    }

    async checkServerStatus() {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const statusInfo = document.getElementById('statusInfo');

        if (!statusDot || !statusText || !statusInfo) {
            console.error('[VeriFeed Popup] Status elements not found in DOM');
            return;
        }

        try {
            statusText.textContent = 'Checking...';
            statusDot.className = 'status-dot';

            const startTime = Date.now();

            // Use background script for server health check
            const health = await new Promise((resolve, reject) => {
                chrome.runtime.sendMessage({
                    action: 'checkServerHealth',
                    serverUrl: this.serverUrl
                }, (response) => {
                    if (chrome.runtime.lastError) {
                        reject(new Error(chrome.runtime.lastError.message));
                    } else {
                        resolve(response);
                    }
                });
            });

            const latency = Date.now() - startTime;

            if (health.healthy) {
                statusDot.className = 'status-dot online';
                statusText.textContent = 'Online';

                const modelStatus = health.modelLoaded ? '✓ Loaded' : '✗ Not loaded';
                const deviceInfo = health.device || 'Unknown';

                statusInfo.textContent = `Server ready • Model: ${modelStatus} • Device: ${deviceInfo} • Latency: ${latency}ms`;

                console.log('[VeriFeed Popup] Server healthy:', health);
            } else {
                throw new Error(health.error || 'Server health check failed');
            }
        } catch (error) {
            console.error('[VeriFeed Popup] Server check failed:', error);

            if (statusDot && statusText && statusInfo) {
                statusDot.className = 'status-dot offline';
                statusText.textContent = 'Offline';

                let errorMsg = 'Cannot connect to analysis server.';
                if (error.message.includes('timeout') || error.message.includes('AbortError')) {
                    errorMsg = 'Connection timeout. Server may be offline.';
                } else if (error.message.includes('Failed to fetch') || error.message.includes('network')) {
                    errorMsg = 'Server not reachable. Make sure backend is running on localhost:5000';
                }

                statusInfo.textContent = errorMsg;
            }
        }
    }

    async notifyContentScript() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            if (tab && tab.id && (tab.url?.includes('facebook.com') || tab.url?.includes('fb.com'))) {
                // Use Promise wrapper for sendMessage to handle response properly
                await new Promise((resolve, reject) => {
                    chrome.tabs.sendMessage(
                        tab.id,
                        {
                            action: 'updateSettings',
                            settings: this.settings
                        },
                        (response) => {
                            if (chrome.runtime.lastError) {
                                // This is normal if content script isn't loaded yet
                                console.log('[VeriFeed Popup] Content script not ready:', chrome.runtime.lastError.message);
                                resolve();
                            } else {
                                console.log('[VeriFeed Popup] Settings sent to content script:', response);
                                resolve(response);
                            }
                        }
                    );
                });
            } else {
                console.log('[VeriFeed Popup] Not on Facebook, skipping content script notification');
            }
        } catch (error) {
            console.error('[VeriFeed Popup] Failed to notify content script:', error);
        }
    }

    async refreshContentScript() {
        try {
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

            if (tab && tab.id && (tab.url?.includes('facebook.com') || tab.url?.includes('fb.com'))) {
                await new Promise((resolve, reject) => {
                    chrome.tabs.sendMessage(
                        tab.id,
                        { action: 'refresh' },
                        (response) => {
                            if (chrome.runtime.lastError) {
                                console.log('[VeriFeed Popup] Content script not ready for refresh');
                                resolve();
                            } else {
                                console.log('[VeriFeed Popup] Content script refreshed');
                                resolve(response);
                            }
                        }
                    );
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
    console.log('[VeriFeed Popup] DOM loaded, initializing...');
    try {
        new VeriFeedPopup();
    } catch (error) {
        console.error('[VeriFeed Popup] Failed to initialize:', error);
    }
});

// Handle messages from content script
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
    console.log('[VeriFeed Popup] Received message:', request.action);

    if (request.action === 'analysisComplete') {
        console.log('[VeriFeed Popup] Analysis completed:', request.result);
        sendResponse({ received: true });
        return false; // Synchronous response
    }

    if (request.action === 'analysisError') {
        console.error('[VeriFeed Popup] Analysis error:', request.error);
        sendResponse({ received: true });
        return false; // Synchronous response
    }

    // Unknown message
    console.warn('[VeriFeed Popup] Unknown message action:', request.action);
    sendResponse({ received: false, error: 'Unknown action' });
    return false;   
});