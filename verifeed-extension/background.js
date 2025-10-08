// VeriFeed Background Script
console.log('VeriFeed background script loaded');

// Handle extension installation
chrome.runtime.onInstalled.addListener((details) => {
  if (details.reason === 'install') {
    console.log('VeriFeed extension installed');
    // Set default settings
    chrome.storage.local.set({
      verifeedEnabled: true,
      autoAnalyze: true,
      showIndicators: true,
      confidenceThreshold: 40,
      serverUrl: 'http://localhost:5000'
    });
  }
});

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((request, sender, sendResponse) => {
  switch (request.action) {
    case 'getSettings':
      chrome.storage.local.get([
        'verifeedEnabled',
        'autoAnalyze',
        'showIndicators', 
        'confidenceThreshold',
        'serverUrl'
      ], (result) => {
        sendResponse(result);
      });
      return true; // Keep message channel open for async response
      
    case 'analysisComplete':
      console.log('VeriFeed analysis completed:', request);
      // You can add additional logging or notifications here
      break;
      
    case 'analysisError':
      console.error('VeriFeed analysis error:', request);
      break;
  }
});

// Optional: Handle browser action click
chrome.action.onClicked.addListener((tab) => {
  console.log('VeriFeed icon clicked on tab:', tab.url);
});