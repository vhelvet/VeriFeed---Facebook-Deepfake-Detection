# VeriFeed - Deepfake Detection for Social Media

VeriFeed is a Chrome extension that detects deepfake videos on social media platforms like Facebook. It uses a deep learning model (ResNext50 + LSTM) to analyze videos and provide real-time deepfake detection.

## Project Structure

```
verifeed/
├── verifeed-backend/          # Flask API server for deepfake detection
│   ├── app.py                # Main Flask application
│   ├── requirements.txt      # Python dependencies
│   └── test_api.py          # API testing script
├── verifeed-extension/       # Chrome extension
│   ├── manifest.json        # Extension manifest
│   ├── content.js           # Content script for Facebook
│   ├── background.js        # Background script
│   ├── popup.html          # Extension popup UI
│   ├── popup.js            # Popup JavaScript
│   └── styles.css          # Extension styles
└── deepfake_detection/      # Original deepfake detection code
```

## Setup Instructions

### 1. Backend Server Setup

1. Navigate to the backend directory:
   ```bash
   cd verifeed-backend
   ```

2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the Flask server:
   ```bash
   python app.py
   ```

The server will start on `http://localhost:5000`

### 2. Chrome Extension Setup

1. Open Chrome and go to `chrome://extensions/`
2. Enable "Developer mode" (toggle in top right)
3. Click "Load unpacked"
4. Select the `verifeed-extension` directory
5. The VeriFeed extension should now appear in your toolbar

### 3. Usage

1. Make sure the backend server is running
2. Navigate to Facebook and find a video post
3. The extension will automatically analyze videos and show detection results
4. Click the VeriFeed extension icon to see detailed information

## API Endpoints

- `GET /health` - Health check endpoint
- `GET /model_info` - Get model information
- `POST /analyze` - Analyze a video for deepfakes
- `POST /batch_analyze` - Analyze multiple videos

## Model Information

- **Architecture**: ResNext50 + LSTM
- **Input Size**: 112x112 pixels
- **Sequence Length**: 20 frames
- **Classes**: FAKE, REAL
- **Device**: CPU/GPU auto-detection

## Current Status

✅ Backend server running successfully on port 5000  
✅ API endpoints working correctly  
✅ Chrome extension files created  
✅ Deep learning model initialized  
❌ Pre-trained model weights (using random weights for demo)  
❌ Real video processing integration  

## Next Steps

1. Add pre-trained model weights for accurate detection
2. Implement real video download/processing from social media
3. Add more social media platform support
4. Improve UI/UX of the Chrome extension
5. Add batch processing capabilities

## Development Notes

- The backend uses Flask with CORS enabled for extension communication
- The model is initialized with random weights (demo mode)
- Video validation currently fails for non-existent URLs (expected behavior)
- Extension injects detection UI into Facebook video posts
