/**
 * VERIFEED EXTENSION - AUTHENTICATION MODULE
 * Handles API authentication with the secured backend
 */

class VerifeedAuth {
    constructor() {
        this.apiUrl = 'http://localhost:5000';
        this.apiKey = '5hTeoaOm5m-91clhe2iVqKy2jpkiN54JLQ4vNbiDodU';  // ← PASTE YOUR API_KEY HERE!
        this.token = null;
        this.tokenExpiry = null;
        
        this.loadToken();
    }

    /**
     * Load saved token from chrome storage
     */
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

    /**
     * Save token to chrome storage
     */
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

    /**
     * Clear stored token
     */
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

    /**
     * Check if token is valid and not expired
     */
    isTokenValid() {
        if (!this.token || !this.tokenExpiry) {
            return false;
        }
        
        // Check if token expires in next 5 minutes
        const now = new Date();
        const fiveMinutesFromNow = new Date(now.getTime() + 5 * 60000);
        
        return this.tokenExpiry > fiveMinutesFromNow;
    }

    /**
     * Generate JWT token from API key
     */
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
            
            // Save token
            await this.saveToken(data.token, data.expires_in);
            
            console.log('✓ Token generated successfully');
            return data.token;
            
        } catch (error) {
            console.error('Error generating token:', error);
            throw error;
        }
    }

    /**
     * Ensure we have a valid token (generate if needed)
     */
    async ensureToken() {
        if (this.isTokenValid()) {
            return this.token;
        }
        
        // Token invalid or expired, generate new one
        return await this.generateToken();
    }

    /**
     * Get authentication headers for API requests
     */
    async getAuthHeaders() {
        const token = await this.ensureToken();
        
        return {
            'Content-Type': 'application/json',
            'Authorization': `Bearer ${token}`
        };
    }

    /**
     * Make authenticated request to predict endpoint
     */
    async predict(frames) {
        try {
            const headers = await this.getAuthHeaders();
            
            console.log(`Sending ${frames.length} frames for prediction...`);
            
            const response = await fetch(`${this.apiUrl}/predict`, {
                method: 'POST',
                headers: headers,
                body: JSON.stringify({ frames: frames })
            });

            if (response.status === 401) {
                // Token expired or invalid, try regenerating
                console.log('Token invalid, regenerating...');
                await this.clearToken();
                
                // Retry with new token
                const newHeaders = await this.getAuthHeaders();
                const retryResponse = await fetch(`${this.apiUrl}/predict`, {
                    method: 'POST',
                    headers: newHeaders,
                    body: JSON.stringify({ frames: frames })
                });
                
                if (!retryResponse.ok) {
                    const error = await retryResponse.json();
                    throw new Error(error.error || 'Prediction failed');
                }
                
                return await retryResponse.json();
            }

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Prediction failed');
            }

            const result = await response.json();
            console.log('✓ Prediction successful:', result);
            
            return result;
            
        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    /**
     * Health check (no auth required)
     */
    async healthCheck() {
        try {
            const response = await fetch(`${this.apiUrl}/health`);
            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Health check failed:', error);
            throw error;
        }
    }

    /**
     * Get model info (requires auth)
     */
    async getModelInfo() {
        try {
            const headers = await this.getAuthHeaders();
            
            const response = await fetch(`${this.apiUrl}/model/info`, {
                method: 'GET',
                headers: headers
            });

            if (!response.ok) {
                const error = await response.json();
                throw new Error(error.error || 'Failed to get model info');
            }

            return await response.json();
            
        } catch (error) {
            console.error('Error getting model info:', error);
            throw error;
        }
    }
}

// Create singleton instance
const verifeedAuth = new VerifeedAuth();

// Export for use in other scripts
if (typeof module !== 'undefined' && module.exports) {
    module.exports = VerifeedAuth;
}