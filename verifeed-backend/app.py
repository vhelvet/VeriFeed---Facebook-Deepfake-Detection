from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
import numpy as np
import cv2
import face_recognition
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter
import os
import glob
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import hashlib

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration optimized for Facebook videos
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = 'models'

# Facebook-specific thresholds
FACEBOOK_COMPRESSION_THRESHOLD = 0.85
MIN_FACE_QUALITY_SCORE = 0.3

# Global resources
cv_face_cascade = None

# CRITICAL FIX: Set deterministic behavior
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def init_cv_cascade():
    global cv_face_cascade
    if cv_face_cascade is None:
        cv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return cv_face_cascade

class FacebookAwareTransforms:
    """Transform pipeline optimized for Facebook video compression"""
    
    def __init__(self, im_size=IM_SIZE):
        self.im_size = im_size
        
        # Standard transforms for good quality frames
        self.standard_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    
    def __call__(self, image, quality_score=1.0):
        # Apply denoising for compressed Facebook videos
        if isinstance(image, np.ndarray) and quality_score < 0.7:
            # Convert to PIL for preprocessing
            pil_img = Image.fromarray(image)
            
            # Mild denoising and sharpening for compressed videos
            if quality_score < 0.5:
                pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))
                enhancer = ImageEnhance.Sharpness(pil_img)
                pil_img = enhancer.enhance(1.2)
            
            # Convert back to numpy for standard transforms
            image = np.array(pil_img)
        
        return self.standard_transforms(image)

facebook_transforms = FacebookAwareTransforms()

def calculate_frame_quality(frame):
    """Calculate quality score for Facebook video frames"""
    if frame is None or len(frame.shape) != 3:
        return 0.0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    
    # Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(1.0, laplacian_var / 100.0)
    
    # Contrast
    contrast = gray.std()
    contrast_score = min(1.0, contrast / 50.0)
    
    # Weighted average favoring sharpness (important for compressed videos)
    quality_score = sharpness_score * 0.6 + contrast_score * 0.4
    return max(0.0, min(1.0, quality_score))

class Model(nn.Module):
    def __init__(self, num_classes, lstm_layers=1, hidden_dim=2048):
        super(Model, self).__init__()
        
        # Fix: Use torchvision.models directly
        try:
            backbone = torchvision.models.resnext50_32x4d(weights='DEFAULT')
            print("Using ResNeXt-50")
        except:
            try:
                backbone = torchvision.models.resnet50(weights='DEFAULT')  
                print("Warning: ResNeXt-50 not available, using ResNet-50 as fallback")
            except:
                backbone = torchvision.models.resnet50(pretrained=True)
                print("Using ResNet-50 (older torchvision)")
        
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        self.lstm = nn.LSTM(2048, hidden_dim, lstm_layers, bidirectional=False)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        # CRITICAL FIX: Remove dropout during inference
        # classification = self.dp(self.linear1(x_lstm[:, -1, :]))
        classification = self.linear1(x_lstm[:, -1, :])
        return fmap, classification

class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.model_info = self._scan_models()
        self._lock = threading.Lock()
    
    def _scan_models(self):
        model_info = {}
        if not os.path.exists(MODELS_DIR):
            return model_info
        
        for model_path in glob.glob(os.path.join(MODELS_DIR, "*.pt")):
            filename = os.path.basename(model_path)
            try:
                parts = filename.replace('.pt', '').split('_')
                if len(parts) >= 4:
                    accuracy = float(parts[1])
                    sequence_length = int(parts[3])
                    
                    if sequence_length not in model_info:
                        model_info[sequence_length] = []
                    
                    model_info[sequence_length].append({
                        'path': model_path,
                        'filename': filename,
                        'accuracy': accuracy,
                        'sequence_length': sequence_length
                    })
            except (ValueError, IndexError):
                continue
        
        for key in model_info:
            model_info[key].sort(key=lambda x: x['accuracy'], reverse=True)
        
        logger.info(f"Found models for sequence lengths: {list(model_info.keys())}")
        return model_info
    
    def load_model(self, sequence_length):
        if sequence_length in self.loaded_models:
            return self.loaded_models[sequence_length]
        
        with self._lock:
            if sequence_length in self.loaded_models:
                return self.loaded_models[sequence_length]
            
            available_lengths = list(self.model_info.keys())
            if not available_lengths:
                raise ValueError("No models found")
            
            if sequence_length not in self.model_info:
                sequence_length = min(available_lengths, key=lambda x: abs(x - sequence_length))
            
            model_info = self.model_info[sequence_length][0]
            
            model = Model(num_classes=2)
            checkpoint = torch.load(model_info['path'], map_location=DEVICE, weights_only=True)
            
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                for key in ['model_state_dict', 'state_dict', 'model']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
            
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE)
            model.eval()  # CRITICAL: Set to eval mode
            
            # CRITICAL FIX: Disable dropout explicitly
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
            
            self.loaded_models[sequence_length] = {
                'model': model,
                'info': model_info
            }
            
            logger.info(f"Loaded Facebook-optimized model: {model_info['filename']}")
            return self.loaded_models[sequence_length]

model_manager = ModelManager()

def decode_base64_image(base64_string):
    try:
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string, validate=True)
        
        # Skip very small images (likely corrupted)
        if len(image_data) < 1000:
            return None
        
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Apply mild denoising for small/compressed images (Facebook characteristic)
        if len(image_data) < 50000:  # Likely compressed
            image = image.filter(ImageFilter.MedianFilter(size=3))
        
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except:
        return None

def process_facebook_frame(frame_data):
    """Process frame with deterministic face detection"""
    i, frame = frame_data
    
    frame_img = decode_base64_image(frame)
    if frame_img is None:
        return None, i, 0.0
    
    rgb_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    quality_score = calculate_frame_quality(rgb_frame)
    
    # Facebook-optimized face detection with FIXED parameters for consistency
    try:
        cascade = init_cv_cascade()
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        # CRITICAL FIX: Use consistent parameters regardless of quality
        # This ensures same frames get same preprocessing
        scale_factor = 1.1
        min_neighbors = 3
        
        # Apply histogram equalization for low quality (but consistently)
        if quality_score < 0.6:
            gray = cv2.equalizeHist(gray)
        
        faces = cascade.detectMultiScale(
            gray, 
            scaleFactor=scale_factor, 
            minNeighbors=min_neighbors,
            minSize=(20, 20)
        )
        
        if len(faces) > 0:
            # Select largest face (deterministic)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # FIXED padding for consistency
            padding = max(2, int(min(w, h) * 0.1))
            
            top = max(0, y - padding)
            bottom = min(rgb_frame.shape[0], y + h + padding)
            left = max(0, x - padding)
            right = min(rgb_frame.shape[1], x + w + padding)
            
            rgb_frame = rgb_frame[top:bottom, left:right]
        
        # Fallback to face_recognition (only if cascade fails completely)
        elif len(faces) == 0:
            try:
                faces = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=0)
                if faces:
                    top, right, bottom, left = faces[0]
                    rgb_frame = rgb_frame[top:bottom, left:right]
            except:
                pass
    except:
        pass
    
    return rgb_frame, i, quality_score

def preprocess_frames_facebook(frames):
    """Preprocess frames in DETERMINISTIC order"""
    frame_data = [(i, frame) for i, frame in enumerate(frames)]
    
    # CRITICAL FIX: Process frames sequentially to maintain order
    # ThreadPoolExecutor can cause race conditions
    results = []
    for data in frame_data:
        result = process_facebook_frame(data)
        results.append(result)
    
    # Sort by original index to ensure consistent ordering
    results.sort(key=lambda x: x[1])
    
    processed_frames = []
    quality_scores = []
    faces_detected = 0
    
    for frame_img, original_idx, quality_score in results:
        if frame_img is None:
            continue
        
        faces_detected += 1
        quality_scores.append(quality_score)
        
        try:
            frame_tensor = facebook_transforms(frame_img, quality_score)
            processed_frames.append(frame_tensor)
        except:
            continue
    
    if not processed_frames:
        raise ValueError("No frames could be processed successfully")
    
    avg_quality = np.mean(quality_scores) if quality_scores else 0.0
    
    return torch.stack(processed_frames).unsqueeze(0), faces_detected, avg_quality

def predict_deepfake_facebook(model, frames_tensor, avg_quality):
    """Deterministic prediction with consistent confidence calculation"""
    try:
        with torch.no_grad():
            frames_tensor = frames_tensor.to(DEVICE, non_blocking=True)
            
            # CRITICAL: Use inference mode for deterministic behavior
            with torch.inference_mode():
                fmap, logits = model(frames_tensor)
            
            probabilities = torch.softmax(logits, dim=1)
            _, prediction = torch.max(probabilities, 1)
            
            fake_prob = probabilities[0, 0].item()
            real_prob = probabilities[0, 1].item()
            predicted_class = prediction.item()
            
            # SIMPLIFIED: Use raw model confidence without random adjustments
            base_confidence = probabilities[0, predicted_class].item() * 100
            
            # CRITICAL FIX: Minimal, consistent adjustments only
            adjusted_confidence = base_confidence
            
            # Only apply quality penalty if very low quality (threshold-based, not gradual)
            if avg_quality < 0.4:
                adjusted_confidence = max(50, adjusted_confidence * 0.90)
            elif avg_quality < 0.6:
                adjusted_confidence = max(50, adjusted_confidence * 0.95)
            
            adjusted_confidence = max(50, min(99.9, adjusted_confidence))
            
            is_real = predicted_class == 1
            result = "REAL" if is_real else "FAKE"
            
            # Generate warnings
            warnings = []
            if avg_quality < 0.4:
                warnings.append("Very low video quality - results may be less reliable")
            elif avg_quality < 0.6:
                warnings.append("Low video quality detected")
            
            if adjusted_confidence < 65:
                warnings.append("Model confidence is moderate")
            
            return {
                'prediction': result,
                'confidence': adjusted_confidence,
                'base_confidence': base_confidence,
                'is_deepfake': not is_real,
                'probabilities': {'fake': fake_prob, 'real': real_prob},
                'warnings': warnings,
                'facebook_adjustments': {
                    'avg_quality': avg_quality,
                    'confidence_adjustment': adjusted_confidence - base_confidence
                }
            }
    except Exception as e:
        logger.error(f"Error in Facebook prediction: {e}")
        raise

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'available_models': len(model_manager.model_info),
        'facebook_optimization': 'enabled',
        'deterministic_mode': 'enabled'
    })

@app.route('/frame_analyze', methods=['POST'])
def analyze_frames():
    start_time = time.time()
    
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
        
        frames = data.get('frames', [])
        platform = data.get('platform', 'facebook')
        
        if not frames:
            return jsonify({'error': 'No frames provided'}), 400
        
        # CRITICAL FIX: Calculate frame hash for debugging consistency
        frame_hash = hashlib.md5(''.join(frames[:3]).encode()).hexdigest()[:8]
        logger.info(f"Analyzing {len(frames)} frames from {platform} (hash: {frame_hash})")
        
        # Validate frames
        validated_frames = []
        for frame in frames:
            if isinstance(frame, str) and len(frame) > 100:
                if not frame.startswith('data:image/'):
                    frame = f'data:image/jpeg;base64,{frame}'
                validated_frames.append(frame)
        
        if not validated_frames:
            return jsonify({'error': 'No valid frames in request'}), 400
        
        logger.info(f"Validated {len(validated_frames)} frames for processing")
        
        # Load model
        model_data = model_manager.load_model(len(validated_frames))
        model = model_data['model']
        model_info = model_data['info']
        
        # Facebook-optimized preprocessing (now deterministic)
        frames_tensor, faces_detected, avg_quality = preprocess_frames_facebook(validated_frames)
        
        # Facebook-aware prediction (now deterministic)
        prediction_result = predict_deepfake_facebook(model, frames_tensor, avg_quality)
        
        processing_time = time.time() - start_time
        
        # Response matching content script expectations
        response = {
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'is_deepfake': prediction_result['is_deepfake'],
            'probabilities': prediction_result['probabilities'],
            'processing_time': round(processing_time, 2),
            'frames_processed': frames_tensor.shape[1],
            'faces_detected': faces_detected,
            'model_accuracy': model_info['accuracy'],
            'warnings': prediction_result['warnings'],
            'base_confidence': prediction_result['base_confidence'],
            'video_analysis': {
                'avg_quality': round(avg_quality, 3),
                'quality_category': 'high' if avg_quality >= 0.8 else 'medium' if avg_quality >= 0.6 else 'low'
            },
            'facebook_adjustments': prediction_result['facebook_adjustments'],
            'platform': platform,
            'frame_hash': frame_hash  # For debugging
        }
        
        logger.info(f"FB Analysis [{frame_hash}]: {response['prediction']} ({response['confidence']:.1f}%) - "
                   f"Quality: {avg_quality:.2f}, Time: {processing_time:.2f}s")
        
        return jsonify(response)
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in Facebook frame_analyze: {error_message}")
        
        return jsonify({
            'error': error_message,
            'details': 'Facebook-optimized deepfake analysis failed',
            'timestamp': datetime.now().isoformat(),
            'platform': platform if 'platform' in locals() else 'unknown'
        }), 500

if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Facebook-Compatible VeriFeed Backend Starting")
        print("DETERMINISTIC MODE ENABLED")
        print("=" * 60)
        
        print("Step 1: Initializing OpenCV cascade...")
        init_cv_cascade()
        print("✓ OpenCV cascade initialized")
        
        print("Step 2: Checking model availability...")
        if len(model_manager.model_info) == 0:
            print("⚠️  WARNING: No models found in models/ directory")
        else:
            print(f"✓ Found models for {len(model_manager.model_info)} sequence lengths")
            for seq_len, models in model_manager.model_info.items():
                best_model = models[0]
                print(f"   {seq_len} frames: {best_model['filename']} (acc: {best_model['accuracy']}%)")
        
        print("Step 3: Deterministic configuration...")
        print(f"✓ Torch seed: 42")
        print(f"✓ NumPy seed: 42")
        print(f"✓ CUDNN deterministic: True")
        print(f"✓ Dropout disabled in inference")
        print(f"✓ Sequential frame processing: enabled")
        print(f"✓ Device: {DEVICE}")
        
        print("\n" + "=" * 60)
        print("🚀 Starting DETERMINISTIC server...")
        print("📍 Content script endpoint: http://localhost:5000/frame_analyze")
        print("🔍 Facebook video optimization: ACTIVE")
        print("🎯 Consistent predictions: GUARANTEED")
        print("=" * 60)
        
        app.run(host='localhost', port=5000, debug=False, threaded=True, use_reloader=False)
        
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()