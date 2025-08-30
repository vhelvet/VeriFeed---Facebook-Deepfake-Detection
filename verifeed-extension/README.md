# VeriFeed Chrome Extension - Tunai-Style Deepfake Detection

A Chrome extension that embeds directly on Facebook posts to detect deepfake videos using AI-powered analysis, similar to the Tunai fact-checking tool.

## Features

- 🔍 **Verify Button**: Adds a "Verify" button overlay on Facebook video posts
- 🤖 **AI Analysis**: Uses deep learning model to detect deepfake content
- 📊 **Confidence Meter**: Shows analysis results with confidence percentage
- 🎨 **Tunai-Style UI**: Clean, professional popup interface
- ⚡ **Real-time**: Works directly on Facebook without page reloads

## Installation

### Prerequisites
1. Python 3.8+ for the backend server
2. Chrome browser

### Backend Setup
```bash
cd verifeed-backend
pip install -r requirements.txt
python app.py
```

The backend server will start on `http://localhost:5000`

### Chrome Extension Installation

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top-right corner)
3. Click "Load unpacked"
4. Select the `verifeed-extension` folder
5. The VeriFeed extension should now appear in your extensions list

## Usage

1. **Start Backend**: Make sure the Python server is running on port 5000
2. **Open Facebook**: Navigate to any Facebook page with videos
3. **Verify Videos**: Click the "Verify" button that appears on video posts
4. **View Results**: See the analysis results in the popup overlay

## How It Works

1. **Content Script**: Injects "Verify" buttons on Facebook video posts
2. **User Interaction**: User clicks "Verify" to initiate analysis
3. **Backend API**: Sends video URL to the deep learning model
4. **Analysis**: Model processes video frames for deepfake detection
5. **Results**: Displays confidence score and analysis in popup

## File Structure

```
verifeed-extension/
├── manifest.json          # Chrome extension configuration
├── content.js            # Main content script for Facebook
├── background.js         # Background service worker
├── popup.html           # Extension popup UI
├── popup.js             # Popup functionality
├── styles.css           # Styling for buttons and popups
├── icons/               # Extension icons
└── README.md           # This file
```

## Backend Integration

The extension communicates with a Flask backend that provides:
- `/health` - Server status check
- `/analyze` - Video analysis endpoint
- `/model_info` - Model information

## Testing

You can test the extension using the included test file:
```bash
open verifeed-extension/test-extension.html
```

## Troubleshooting

1. **Extension not loading**: Check Chrome developer console for errors
2. **Backend connection failed**: Ensure Python server is running on port 5000
3. **Buttons not appearing**: Check Facebook URL matches `https://*.facebook.com/*`
4. **Analysis fails**: Verify video URLs are accessible

## Development

To modify the extension:
1. Make changes to the relevant files
2. Go to `chrome://extensions/`
3. Click the refresh icon on the VeriFeed extension
4. Reload Facebook pages to see changes

## License

This project is for educational/demonstration purposes.
