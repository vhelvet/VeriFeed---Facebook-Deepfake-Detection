<div align="center">

# VeriFeed

<img src="extension/assets/VeriFeed-Logo.png" alt="VeriFeed Logo" width="250px">

**AI-Powered Deepfake Video Detection for Facebook**

[![Version](https://img.shields.io/badge/version-1.0.0-blue.svg)](https://github.com/verifeed/verifeed)
[![Python](https://img.shields.io/badge/python-3.8+-green.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/flask-2.0+-red.svg)](https://flask.palletsprojects.com/)
[![PyTorch](https://img.shields.io/badge/pytorch-1.9+-orange.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-purple.svg)](LICENSE)

[📖 Documentation](#documentation) • [📧 Contact](mailto:verifeedofficial@gmail.com) • [🌐 Website](http://localhost:5173/)

</div>

---

## Overview

VeriFeed is an AI-powered browser extension and backend service designed to detect deepfake videos on Facebook. The system uses advanced machine learning models to analyze video content and provide users with authenticity assessments.

## Features

### Core Functionality
- **Deepfake Detection**: Analyzes video frames using a ResNeXt50 + LSTM neural network architecture
- **Browser Extension**: Chrome/Firefox extension for smooth Facebook integration
- **Real-time Analysis**: Processes videos with optimized performance for quick results
- **Confidence Scoring**: Provides probability scores for both authentic and manipulated classifications

### Security & Performance
- **Multi-layer Authentication**: Supports API key and JWT token authentication
- **Rate Limiting**: Configurable request limits to prevent abuse
- **Input Validation**: Comprehensive validation of incoming data
- **Optimized Processing**: Smart frame sampling and GPU acceleration for efficient analysis

### User Interface
- **Extension Popup**: Clean, responsive interface with status indicators
- **Visual Results**: Animated confidence bars and detailed probability displays
- **Compact Mode**: Minimizable interface for reduced screen space usage

## Architecture

### System Components

#### Backend API (Flask)
- **Endpoints**: `/health`, `/predict`, `/auth/token`, `/model/info`, `/model/reload`
- **Model**: PyTorch-based deepfake detection model with 91.43% accuracy
- **Security**: Production-ready with CORS, rate limiting, and input sanitization
- **Deployment**: Waitress WSGI server for production serving

#### Browser Extension
- **Manifest V3**: Modern Chrome extension architecture
- **Content Scripts**: Facebook page integration for video detection
- **Background Service**: Handles API communication and authentication
- **Popup Interface**: User-facing results display

### AI Pipeline
1. **Frame Extraction**: Intelligent sampling from video streams
2. **Face Detection**: OpenCV and face_recognition for face isolation
3. **Feature Extraction**: ResNeXt50 backbone for feature vectors
4. **Sequence Analysis**: Bidirectional LSTM for temporal processing
5. **Classification**: Binary classification with confidence scoring

## Installation

### Prerequisites
- Python 3.8 or higher
- Chrome or Firefox browser
- CUDA-compatible GPU (recommended for optimal performance)

### Backend Setup
```bash
# Clone repository
git clone https://github.com/verifeed/verifeed.git
cd verifeed

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start server
python backend/src/app.py
```

### Extension Installation
1. Open browser extensions page (`chrome://extensions/` or `about:addons`)
2. Enable developer mode
3. Load unpacked extension from the `extension/` directory
4. VeriFeed will be available in the browser toolbar

## Usage

1. Navigate to Facebook and locate a video
2. Click the VeriFeed extension icon
3. The extension will detect available videos and enable analysis
4. Click "Verify Video" to initiate analysis
5. View results with confidence scores and detailed probabilities

## API Reference

### Authentication
The API supports multiple authentication methods:
- **API Key**: Include `X-API-Key` header
- **JWT Token**: Include `Authorization: Bearer <token>` header

### Endpoints

#### GET /health
Public health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "device": "cuda",
  "model_loaded": true,
  "authenticated": false
}
```

#### POST /predict
Main prediction endpoint requiring authentication.

**Request:**
```json
{
  "frames": ["base64_encoded_frame_1", "base64_encoded_frame_2", ...]
}
```

**Response:**
```json
{
  "prediction": "REAL",
  "confidence": 87.5,
  "fake_probability": 12.5,
  "real_probability": 87.5,
  "faces_analyzed": 20,
  "frames_processed": 60,
  "processing_time": 2.3
}
```

## Performance

### Model Metrics
- **Accuracy**: 91.43% (primary model)
- **Precision**: 92.1%
- **Recall**: 90.8%
- **F1-Score**: 91.4%

### Benchmarks
- **Processing Time**: < 3 seconds for 60-frame videos
- **Memory Usage**: < 2GB peak during inference
- **Concurrent Users**: Supports 1000+ with rate limiting

## Security

### Authentication & Authorization
- API key validation with secure hashing
- JWT token support with expiration
- Admin-only endpoints for model management

### Input Security
- Base64 validation and size limits
- Path traversal prevention
- Request size limits (100MB default)

### Operational Security
- Rate limiting (configurable per minute/hour/day)
- CORS restrictions to allowed origins
- Comprehensive logging and audit trails

## Development

### Project Structure
```
verifeed/
├── backend/                 # Flask API server
│   ├── src/
│   │   ├── app.py          # Main application
│   │   └── keys.py         # Configuration
│   └── models/             # PyTorch models
├── extension/               # Browser extension
│   ├── scripts/            # JavaScript files
│   ├── styles/             # CSS stylesheets
│   └── assets/             # Images and icons
├── dfdc_repo/               # Training pipeline
└── requirements.txt         # Python dependencies
```

### Contributing
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Make changes and add tests
4. Commit changes (`git commit -am 'Add new feature'`)
5. Push to branch (`git push origin feature/new-feature`)
6. Create a Pull Request

### Testing
```bash
# Install test dependencies
pip install -r requirements-dev.txt

# Run tests
python -m pytest

# Run with coverage
python -m pytest --cov=backend
```

## Configuration

### Environment Variables
- `FLASK_DEBUG`: Enable/disable debug mode
- `API_KEY`: Primary API key for authentication
- `ADMIN_API_KEY`: Admin-level API key
- `MAX_CONTENT_MB`: Maximum request size
- `RATE_LIMIT_PER_MINUTE`: Rate limiting configuration
- `MODELS_DIR`: Path to model files

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

- **Email**: verifeedofficial@gmail.com
- **Website**: http://localhost:5173/
- **Issues**: [GitHub Issues](https://github.com/verifeed/verifeed/issues)

## Acknowledgments

- Facebook Deepfake Detection Challenge (DFDC) dataset
- PyTorch and related deep learning libraries
- Open-source computer vision community
