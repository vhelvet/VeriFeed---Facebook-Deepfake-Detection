// VeriFeed Popup Script - Handles user interface and settings

class VeriFeedPopup {
    constructor() {
        this.settings = {
            enabled: true,
            autoAnalyze: true,
            showIndicators: true,
            confidenceThreshold: 40
        };

        this.init();
    }

    async init() {
        console.log('VeriFeed popup initialized');
        await this.loadSettings();
        this.setupEventListeners();
        this.checkServerStatus();
        this.updateUI();
    }

    async loadSettings() {
        try {
            const result = await chrome.storage.local.get(['verifeedSettings']);
            if (result.verifeedSettings) {
                this.settings = { ...this.settings, ...result.verifeedSettings };
            }
        } catch (error) {
            console.error('Error loading settings:', error);
        }
    }

    async saveSettings() {
        try {
            await chrome.storage.local.set({ verifeedSettings: this.settings });
            
            // Notify background script about settings change
            await chrome.runtime.sendMessage({
                action: 'updateSettings',
                settings: this.settings
            });
        } catch (error) {
            console.error('Error saving settings:', error);
        }
    }

    setupEventListeners() {
        // Toggle switches
        document.getElementById('toggleEnabled').addEventListener('change', (e) => {
            this.settings.enabled = e.target.checked;
            this.saveSettings();
        });

        document.getElementById('toggleAutoAnalyze').addEventListener('change', (e) => {
            this.settings.autoAnalyze = e.target.checked;
            this.saveSettings();
        });

        document.getElementById('toggleShowIndicators').addEventListener('change', (e) => {
            this.settings.showIndicators = e.target.checked;
            this.saveSettings();
        });

        // Confidence threshold slider
        const thresholdSlider = document.getElementById('confidenceThreshold');
        const thresholdValue = document.getElementById('thresholdValue');

        thresholdSlider.addEventListener('input', (e) => {
            const value = e.target.value;
            thresholdValue.textContent = `${value}%`;
            this.settings.confidenceThreshold = parseInt(value);
            this.saveSettings();
        });

        // Action buttons
        document.getElementById('btnRefresh').addEventListener('click', () => {
            this.checkServerStatus();
        });

        document.getElementById('btnAnalyze').addEventListener('click', () => {
            this.analyzeCurrentPage();
        });
    }

    updateUI() {
        // Update toggle switches
        document.getElementById('toggleEnabled').checked = this.settings.enabled;
        document.getElementById('toggleAutoAnalyze').checked = this.settings.autoAnalyze;
        document.getElementById('toggleShowIndicators').checked = this.settings.showIndicators;
        
        // Update confidence threshold
        const thresholdSlider = document.getElementById('confidenceThreshold');
        const thresholdValue = document.getElementById('thresholdValue');
        thresholdSlider.value = this.settings.confidenceThreshold;
        thresholdValue.textContent = `${this.settings.confidenceThreshold}%`;
    }

    async checkServerStatus() {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const statusInfo = document.getElementById('statusInfo');

        // Show loading state
        statusDot.className = 'status-dot';
        statusText.textContent = 'Checking...';
        statusInfo.textContent = 'Connecting to analysis server...';

        try {
            const response = await chrome.runtime.sendMessage({
                action: 'checkHealth'
            });

            if (response && response.status) {
                this.updateServerStatus(response.status);
            } else {
                this.updateServerStatus('offline');
            }
        } catch (error) {
            console.error('Error checking server status:', error);
            this.updateServerStatus('offline');
        }
    }

    updateServerStatus(status) {
        const statusDot = document.getElementById('statusDot');
        const statusText = document.getElementById('statusText');
        const statusInfo = document.getElementById('statusInfo');

        statusDot.className = `status-dot ${status}`;
        
        switch (status) {
            case 'online':
                statusText.textContent = 'Online';
                statusInfo.textContent = 'Server is ready to analyze videos';
                break;
            case 'offline':
                statusText.textContent = 'Offline';
                statusInfo.textContent = 'Analysis server is not available. Please make sure the server is running.';
                break;
            default:
                statusText.textContent = 'Unknown';
                statusInfo.textContent = 'Unable to determine server status';
        }
    }

    async analyzeCurrentPage() {
        try {
            // Get current active tab
            const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
            
            if (tab && tab.url.includes('facebook.com')) {
                // Send message to content script to analyze videos
                const response = await chrome.tabs.sendMessage(tab.id, {
                    action: 'analyzeSpecificVideo',
                    videoSelector: 'video'
                });

                if (response && response.success) {
                    this.showNotification('Analysis started', 'success');
                } else {
                    this.showNotification('No videos found to analyze', 'warning');
                }
            } else {
                this.showNotification('Please open Facebook to analyze videos', 'info');
            }
        } catch (error) {
            console.error('Error analyzing current page:', error);
            this.showNotification('Error starting analysis', 'error');
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = document.createElement('div');
        notification.style.cssText = `
            position: fixed;
            top: 20px;
            right: 20px;
            padding: 12px 16px;
            border-radius: 8px;
            color: white;
            font-size: 14px;
            font-weight: 500;
            z-index: 10000;
            animation: slideIn 0.3s ease;
            max-width: 280px;
        `;

        // Set background color based on type
        const colors = {
            success: '#28a745',
            error: '#dc3545',
            warning: '#ffc107',
            info: '#17a2b8'
        };
        
        notification.style.background = colors[type] || colors.info;
        notification.textContent = message;

        // Add to document
        document.body.appendChild(notification);

        // Remove after 3 seconds
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 3000);
    }

    // Handle messages from background script
    setupMessageListener() {
        chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
            switch (request.action) {
                case 'serverStatusUpdate':
                    this.updateServerStatus(request.status);
                    break;
                case 'settingsUpdated':
                    this.settings = { ...this.settings, ...request.settings };
                    this.updateUI();
                    break;
            }
        });
    }
}

// Initialize the popup when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    const veriFeedPopup = new VeriFeedPopup();
    veriFeedPopup.setupMessageListener();
});

// Add CSS for notifications
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);
