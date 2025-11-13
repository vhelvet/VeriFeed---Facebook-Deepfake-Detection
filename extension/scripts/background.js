// VeriFeed Background Script - Fixed Async Messaging
console.log('VeriFeed background script loaded');

// Stats tracking
const stats = {
  totalAnalyses: 0,
  fakesDetected: 0,
  realsDetected: 0,
  averageProcessingTime: 0,
  errors: 0
};

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('VeriFeed extension installed');
    // Set default settings
    chrome.storage.local.set({
      verifeedEnabled: true,
      serverUrl: 'http://localhost:5000',
      autoAnalyze: true,
      showNotifications: true,
      minConfidence: 70
    });
    
    // Show welcome notification
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon48.png',
      title: 'VeriFeed Installed',
      message: 'Deepfake detection is now active on Facebook videos!'
    });
  } else if (details.reason === 'update') {
    console.log('VeriFeed extension updated');
  }
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  console.log('[Background] Received message:', request.action);

  switch (request.action) {
    case 'getSettings':
      // FIXED: Use promise-based approach for async operations
      chrome.storage.local.get([
        'verifeedEnabled',
        'serverUrl',
        'autoAnalyze',
        'showNotifications',
        'minConfidence'
      ]).then((result) => {
        const settings = {
          verifeedEnabled: result.verifeedEnabled ?? true,
          serverUrl: result.serverUrl || 'http://localhost:5000',
          autoAnalyze: result.autoAnalyze ?? true,
          showNotifications: result.showNotifications ?? true,
          minConfidence: result.minConfidence || 70
        };
        console.log('[Background] Sending settings:', settings);
        sendResponse(settings);
      }).catch((error) => {
        console.error('[Background] Error getting settings:', error);
        sendResponse({ error: error.message });
      });
      return true; // Keep message channel open for async response
      
    case 'updateSettings':
      chrome.storage.local.set(request.settings).then(() => {
        console.log('[Background] Settings updated:', request.settings);
        sendResponse({ success: true });
      }).catch((error) => {
        console.error('[Background] Error updating settings:', error);
        sendResponse({ success: false, error: error.message });
      });
      return true; // Keep message channel open
      
    case 'analysisStarted':
      console.log('[Background] Analysis started', {
        tabId: sender.tab?.id,
        url: sender.tab?.url
      });
      sendResponse({ received: true });
      return false; // Synchronous response
      
    case 'analysisComplete':
      handleAnalysisComplete(request, sender);
      sendResponse({ received: true });
      return false; // Synchronous response
      
    case 'analysisError':
      handleAnalysisError(request, sender);
      sendResponse({ received: true });
      return false; // Synchronous response
      
    case 'getStats':
      sendResponse(stats);
      return false; // Synchronous response
      
    case 'resetStats':
      Object.keys(stats).forEach(key => {
        stats[key] = 0;
      });
      sendResponse({ success: true });
      return false; // Synchronous response
      
    case 'checkServerHealth':
      checkServerHealth(request.serverUrl)
        .then(sendResponse)
        .catch((error) => {
          console.error('[Background] Health check error:', error);
          sendResponse({ healthy: false, error: error.message });
        });
      return true; // Keep message channel open for async response

    default:
      console.warn('[Background] Unknown action:', request.action);
      sendResponse({ error: 'Unknown action' });
      return false;
  }
});

function handleAnalysisComplete(request, sender) {
  const result = request.result;
  
  // Update stats
  stats.totalAnalyses++;
  if (result.prediction === 'FAKE') {
    stats.fakesDetected++;
  } else {
    stats.realsDetected++;
  }
  
  // Update average processing time
  if (result.processing_time) {
    stats.averageProcessingTime = 
      (stats.averageProcessingTime * (stats.totalAnalyses - 1) + result.processing_time) / stats.totalAnalyses;
  }
  
  console.log('[Background] Analysis completed:', {
    prediction: result.prediction,
    confidence: result.confidence,
    processingTime: result.processing_time,
    tabId: sender.tab?.id
  });
  
  // Show notification for high-confidence fake detections
  chrome.storage.local.get(['showNotifications', 'minConfidence']).then((settings) => {
    if (settings.showNotifications && 
        result.prediction === 'FAKE' && 
        result.confidence >= (settings.minConfidence || 70)) {
      
      chrome.notifications.create({
        type: 'basic',
        iconUrl: 'icons/icon48.png',
        title: '⚠️ Potential Deepfake Detected',
        message: `Confidence: ${result.confidence.toFixed(1)}%`,
        priority: 2
      });
    }
  });
  
  // Store result in local storage for popup
  chrome.storage.local.get(['recentAnalyses']).then((data) => {
    const recentAnalyses = data.recentAnalyses || [];
    recentAnalyses.unshift({
      ...result,
      timestamp: new Date().toISOString(),
      url: sender.tab?.url || 'unknown'
    });
    
    // Keep only last 20 analyses
    if (recentAnalyses.length > 20) {
      recentAnalyses.pop();
    }
    
    chrome.storage.local.set({ recentAnalyses });
  });
}

function handleAnalysisError(request, sender) {
  stats.errors++;
  
  console.error('[Background] Analysis error:', {
    error: request.error,
    tabId: sender.tab?.id,
    url: sender.tab?.url
  });
  
  // Show error notification if critical
  if (request.error.includes('server') || request.error.includes('connection')) {
    chrome.notifications.create({
      type: 'basic',
      iconUrl: 'icons/icon48.png',
      title: 'VeriFeed Error',
      message: 'Cannot connect to analysis server. Make sure it\'s running.',
      priority: 1
    });
  }
}

async function checkServerHealth(serverUrl) {
  try {
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 5000);
    
    const response = await fetch(`${serverUrl}/health`, {
      method: 'GET',
      signal: controller.signal
    });
    
    clearTimeout(timeoutId);
    
    if (response.ok) {
      const data = await response.json();
      return {
        healthy: true,
        status: data.status,
        modelLoaded: data.model_loaded,
        device: data.device
      };
    } else {
      return {
        healthy: false,
        error: `Server returned ${response.status}`
      };
    }
  } catch (error) {
    if (error.name === 'AbortError') {
      return {
        healthy: false,
        error: 'Connection timeout'
      };
    }
    return {
      healthy: false,
      error: error.message
    };
  }
}

// Optional: Handle browser action click to show stats
chrome.action.onClicked.addListener((tab) => {
  console.log('[Background] Extension icon clicked', {
    tabId: tab.id,
    url: tab.url,
    stats: stats
  });
  
  // Show quick stats notification
  chrome.notifications.create({
    type: 'basic',
    iconUrl: 'icons/icon48.png',
    title: 'VeriFeed Statistics',
    message: `Analyzed: ${stats.totalAnalyses} | Fakes: ${stats.fakesDetected} | Avg Time: ${stats.averageProcessingTime.toFixed(1)}s`
  });
});

// Periodically check server health (every 5 minutes)
setInterval(() => {
  chrome.storage.local.get(['serverUrl', 'verifeedEnabled']).then(async (settings) => {
    if (settings.verifeedEnabled) {
      const health = await checkServerHealth(settings.serverUrl || 'http://localhost:5000');
      if (!health.healthy) {
        console.warn('[Background] Server health check failed:', health.error);
      } else {
        console.log('[Background] Server healthy:', health);
      }
    }
  });
}, 5 * 60 * 1000);

// Listen for tab updates to inject content script on Facebook
chrome.tabs.onUpdated.addListener((tabId, changeInfo, tab) => {
  if (changeInfo.status === 'complete' && tab.url?.includes('facebook.com')) {
    chrome.storage.local.get(['verifeedEnabled']).then((settings) => {
      if (settings.verifeedEnabled) {
        console.log('[Background] Facebook tab detected:', tabId);
      }
    });
  }
});



console.log('VeriFeed background script ready ✓');
