#!/usr/bin/env python3
"""
VeriFeed Backend Server - Handles deepfake detection model inference
Simplified version that doesn't rely on deepfake_detection package structure
"""

import os
import sys
import logging
import threading
from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import numpy as np
import cv2
from pathlib import Path
from torchvision import transforms
from torch import nn
from torchvision import models

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('verifeed-backend')

# Define the Model class here since we can't import it
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(pretrained=True)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(hidden_dim if not bidirectional else hidden_dim*2, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x_lstm, _ = self.lstm(x, None)
        x_lstm = x_lstm[:, -1, :]
        x_lstm = self.dp(x_lstm)
        out = self.linear1(x_lstm)
        return fmap, out

# Helper functions
def get_transforms():
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transforms, test_transforms

def validate_video(video_path, transform, sequence_length=20):
    """Validate if a video is corrupted and can be processed"""
    try:
        count = sequence_length
        frames = []
        
        vidObj = cv2.VideoCapture(video_path)
        success = 1
        frame_count = 0
        
        while success and frame_count < count:
            success, image = vidObj.read()
            if success:
                frames.append(transform(image))
                frame_count += 1
        
        vidObj.release()
        
        if len(frames) < count:
            return False
        
        frames = torch.stack(frames)
        frames = frames[:count]
        return True
        
    except Exception as e:
        print(f"Error validating video {video_path}: {e}")
        return False

def predict_video(video_path, model, sequence_length=20, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Predict whether a video is real or fake"""
    # Set up transforms
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Simple frame extraction for demo
    try:
        vidObj = cv2.VideoCapture(video_path)
        frames = []
        success = 1
        frame_count = 0
        
        while success and frame_count < sequence_length:
            success, image = vidObj.read()
            if success:
                frames.append(train_transforms(image))
                frame_count += 1
        
        vidObj.release()
        
        if len(frames) < sequence_length:
            return 0, 50.0  # Default to FAKE with low confidence if not enough frames
        
        frames = torch.stack(frames)
        frames = frames.unsqueeze(0)  # Add batch dimension
        
        # Move to device and make prediction
        model = model.to(device)
        model.eval()
        
        with torch.no_grad():
            _, outputs = model(frames.to(device))
            sm = nn.Softmax(dim=1)
            logits = sm(outputs)
            
            _, prediction = torch.max(logits, 1)
            confidence = logits[:, int(prediction.item())].item() * 100
            
            return int(prediction.item()), confidence
            
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 0, 50.0  # Default to FAKE with low confidence on error

class VeriFeedServer:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        self.model = None
        self.device = None
        self.transform = None
        self.initialize_model()
        self.setup_routes()
        
    def initialize_model(self):
        """Initialize the deepfake detection model"""
        try:
            # Determine device
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            logger.info(f"Using device: {self.device}")
            
            # Load model
            self.model = Model(num_classes=2)
            
            # Load pre-trained weights if available
            model_path = Path('models') / 'deepfake_model.pt'
            if model_path.exists():
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                logger.info("Loaded pre-trained model weights")
            else:
                logger.warning("No pre-trained model found. Using randomly initialized weights")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Get transforms
            _, self.transform = get_transforms()
            
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            raise
    
    def setup_routes(self):
        """Setup Flask routes"""
        
        @self.app.route('/health', methods=['GET'])
        def health_check():
            """Health check endpoint"""
            return jsonify({
                'status': 'healthy',
                'model_loaded': self.model is not None,
                'device': str(self.device)
            })
        
        @self.app.route('/analyze', methods=['POST'])
        def analyze_video():
            """Analyze a video for deepfake detection"""
            try:
                data = request.get_json()
                if not data or 'videoUrl' not in data:
                    return jsonify({'error': 'Missing videoUrl parameter'}), 400
                
                video_url = data['videoUrl']
                platform = data.get('platform', 'unknown')
                sequence_length = data.get('sequence_length', 20)
                
                logger.info(f"Analyzing video from {platform}: {video_url}")
                
                # Validate video
                if not validate_video(video_url, self.transform, sequence_length):
                    return jsonify({'error': 'Invalid or corrupted video'}), 400
                
                # Make prediction
                prediction, confidence = predict_video(
                    video_url, 
                    self.model, 
                    sequence_length=sequence_length,
                    device=self.device
                )
                
                result = {
                    'prediction': 'REAL' if prediction == 1 else 'FAKE',
                    'confidence': round(confidence, 2),
                    'video_url': video_url,
                    'platform': platform,
                    'model_version': '1.0.0'
                }
                
                logger.info(f"Analysis result: {result}")
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/batch_analyze', methods=['POST'])
        def batch_analyze():
            """Analyze multiple videos in batch"""
            try:
                data = request.get_json()
                if not data or 'videoUrls' not in data:
                    return jsonify({'error': 'Missing videoUrls parameter'}), 400
                
                video_urls = data['videoUrls']
                results = []
                
                for video_url in video_urls:
                    try:
                        if validate_video(video_url, self.transform):
                            prediction, confidence = predict_video(
                                video_url, 
                                self.model,
                                device=self.device
                            )
                            results.append({
                                'video_url': video_url,
                                'prediction': 'REAL' if prediction == 1 else 'FAKE',
                                'confidence': round(confidence, 2),
                                'status': 'success'
                            })
                        else:
                            results.append({
                                'video_url': video_url,
                                'status': 'error',
                                'error': 'Invalid video'
                            })
                    except Exception as e:
                        results.append({
                            'video_url': video_url,
                            'status': 'error',
                            'error': str(e)
                        })
                
                return jsonify({'results': results})
                
            except Exception as e:
                logger.error(f"Batch analysis error: {e}")
                return jsonify({'error': str(e)}), 500
        
        @self.app.route('/model_info', methods=['GET'])
        def model_info():
            """Get model information"""
            return jsonify({
                'model_architecture': 'ResNext50 + LSTM',
                'input_size': '112x112',
                'sequence_length': '20 frames',
                'classes': ['FAKE', 'REAL'],
                'device': str(self.device),
                'version': '1.0.0'
            })
        
        @self.app.errorhandler(404)
        def not_found(error):
            return jsonify({'error': 'Endpoint not found'}), 404
        
        @self.app.errorhandler(500)
        def internal_error(error):
            return jsonify({'error': 'Internal server error'}), 500
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the Flask server"""
        logger.info(f"Starting VeriFeed server on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug, threaded=True)

def create_app():
    """Factory function to create the Flask app"""
    server = VeriFeedServer()
    return server.app

if __name__ == '__main__':
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    server = VeriFeedServer()
    server.run(debug=True)
