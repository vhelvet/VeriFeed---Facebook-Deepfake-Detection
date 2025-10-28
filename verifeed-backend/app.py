# app.py - VeriFeed Backend (WITH BEST FACE DETECTION - MTCNN)

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import base64
import os
import glob
import logging
from datetime import datetime
import time

# -------------------- CONFIGURATION --------------------
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# OPTIMIZED SETTINGS
MODELS_DIR = 'models'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FRAMES_TO_USE = 20
IM_SIZE = 112
MIN_FRAMES_REQUIRED = 15
FACE_CONFIDENCE_THRESHOLD = 0.9  # MTCNN is more reliable, can use higher threshold

# Transforms (same as training)
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

# -------------------- FACE DETECTION (BEST METHOD) --------------------
# Try to load MTCNN (best for deepfake detection)
MTCNN_DETECTOR = None
try:
    from facenet_pytorch import MTCNN
    MTCNN_DETECTOR = MTCNN(
        keep_all=False,
        device=DEVICE,
        min_face_size=40,
        thresholds=[0.6, 0.7, 0.7],
        post_process=False
    )
    logger.info("✓ MTCNN face detector loaded (BEST)")
except ImportError:
    logger.warning("⚠ facenet-pytorch not installed. Install with: pip install facenet-pytorch")
    MTCNN_DETECTOR = None

# Fallback to face_recognition (your original choice - good)
FACE_RECOGNITION_AVAILABLE = False
try:
    import face_recognition
    FACE_RECOGNITION_AVAILABLE = True
    logger.info("✓ face_recognition available (GOOD fallback)")
except ImportError:
    logger.warning("⚠ face_recognition not installed. Install with: pip install face_recognition")

# Last resort fallback - Haar Cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
logger.info("✓ Haar Cascade loaded (FALLBACK)")

# -------------------- MODEL --------------------
class Model(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        base_model = models.resnext50_32x4d(weights=None)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional, bias=False)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(torch.mean(x_lstm, dim=1)))

# -------------------- ENSEMBLE MODEL LOADING --------------------
def load_ensemble_models(top_n=3):
    """Load top N models for ensemble prediction"""
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))

    if not model_files:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}")

    parsed_models = []
    for model_path in model_files:
        filename = os.path.basename(model_path)
        try:
            parts = filename.replace('.pt', '').split('_')
            acc = None
            frames = None
            
            for i, part in enumerate(parts):
                if 'acc' in part.lower():
                    try:
                        if i + 1 < len(parts):
                            acc = int(parts[i + 1])
                    except:
                        acc_str = ''.join(filter(str.isdigit, part))
                        if acc_str:
                            acc = int(acc_str)
                
                if 'frame' in part.lower():
                    try:
                        frames = int(parts[i])
                    except:
                        frame_str = ''.join(filter(str.isdigit, part))
                        if frame_str:
                            frames = int(frame_str)
            
            if acc and acc >= 85:
                parsed_models.append({
                    'path': model_path,
                    'accuracy': acc,
                    'frames': frames or 20,
                    'filename': filename
                })
        except Exception as e:
            logger.warning(f"Could not parse {filename}: {e}")
            continue

    if not parsed_models:
        logger.warning("No valid models found, using first available")
        parsed_models = [{'path': model_files[0], 'accuracy': 90, 'frames': 20, 'filename': os.path.basename(model_files[0])}]

    parsed_models.sort(key=lambda x: (x['accuracy'], -abs(x['frames'] - 20)), reverse=True)
    top_models = parsed_models[:min(top_n, len(parsed_models))]
    
    models = []
    for model_info in top_models:
        try:
            model = Model(num_classes=2)
            model.load_state_dict(torch.load(model_info['path'], map_location=DEVICE, weights_only=True))
            model.to(DEVICE)
            model.eval()
            models.append((model, model_info['accuracy'], model_info['frames']))
            logger.info(f"✓ Loaded: {model_info['filename']} ({model_info['accuracy']}% acc, {model_info['frames']} frames)")
        except Exception as e:
            logger.error(f"✗ Failed to load {model_info['filename']}: {e}")
    
    if not models:
        raise Exception("No models could be loaded")
    
    return models

# Load ensemble models
ENSEMBLE_MODELS = load_ensemble_models(top_n=3)
logger.info(f"Successfully loaded {len(ENSEMBLE_MODELS)} models")

# -------------------- BEST FACE DETECTION --------------------
def decode_frame(base64_str):
    """Decode base64 frame to image"""
    try:
        if ',' in base64_str:
            base64_str = base64_str.split(',')[1]
        img_data = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is not None and img.size > 0:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            return img
        return None
    except Exception as e:
        logger.warning(f"Frame decode error: {e}")
        return None

def detect_face_mtcnn(frame):
    """BEST: Detect face using MTCNN - most accurate for deepfake detection"""
    if MTCNN_DETECTOR is None:
        return None, 0
    
    try:
        # MTCNN expects RGB PIL image or numpy array
        boxes, probs = MTCNN_DETECTOR.detect(frame)
        
        if boxes is not None and len(boxes) > 0:
            # Get highest confidence detection
            best_idx = np.argmax(probs)
            box = boxes[best_idx]
            prob = probs[best_idx]
            
            if prob >= FACE_CONFIDENCE_THRESHOLD:
                x1, y1, x2, y2 = [int(b) for b in box]
                w = x2 - x1
                h = y2 - y1
                
                # Add padding to preserve context (important for deepfakes)
                pad = int(max(w, h) * 0.25)
                x1 = max(0, x1 - pad)
                y1 = max(0, y1 - pad)
                x2 = min(frame.shape[1], x2 + pad)
                y2 = min(frame.shape[0], y2 + pad)
                
                return frame[y1:y2, x1:x2], prob
    except Exception as e:
        logger.warning(f"MTCNN detection error: {e}")
    
    return None, 0

def detect_face_face_recognition(frame):
    """GOOD: Detect face using face_recognition library"""
    if not FACE_RECOGNITION_AVAILABLE:
        return None, 0
    
    try:
        import face_recognition
        face_locations = face_recognition.face_locations(frame, model="hog")
        
        if face_locations:
            # Get largest face
            face_location = max(face_locations, key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
            top, right, bottom, left = face_location
            
            # Add padding
            h, w = frame.shape[:2]
            pad_h = int((bottom - top) * 0.25)
            pad_w = int((right - left) * 0.25)
            
            top = max(0, top - pad_h)
            bottom = min(h, bottom + pad_h)
            left = max(0, left - pad_w)
            right = min(w, right + pad_w)
            
            return frame[top:bottom, left:right], 0.85
    except Exception as e:
        logger.warning(f"face_recognition error: {e}")
    
    return None, 0

def detect_face_haar(frame):
    """FALLBACK: Detect face using Haar Cascade"""
    h, w = frame.shape[:2]
    
    scale = 1.0
    if w > 640:
        scale = 640 / w
        small = cv2.resize(frame, None, fx=scale, fy=scale)
    else:
        small = frame
    
    gray = cv2.cvtColor(small, cv2.COLOR_RGB2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
    
    if len(faces) == 0:
        faces = face_cascade.detectMultiScale(gray, 1.05, 3, minSize=(20, 20))
    
    if len(faces) > 0:
        x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
        
        # Scale back
        x, y, fw, fh = int(x/scale), int(y/scale), int(fw/scale), int(fh/scale)
        
        # Add padding
        pad = int(max(fw, fh) * 0.3)
        x = max(0, x - pad)
        y = max(0, y - pad)
        fw = min(w - x, fw + 2 * pad)
        fh = min(h - y, fh + 2 * pad)
        
        return frame[y:y+fh, x:x+fw], 0.6
    
    return None, 0

def detect_and_crop_face(frame):
    """
    Multi-method face detection with fallback chain
    Priority: MTCNN > face_recognition > Haar Cascade > Center Crop
    """
    h, w = frame.shape[:2]
    
    # Method 1: Try MTCNN (BEST - preserves deepfake artifacts)
    face_crop, confidence = detect_face_mtcnn(frame)
    if face_crop is not None and face_crop.size > 0:
        return face_crop, True, confidence, "MTCNN"
    
    # Method 2: Try face_recognition (GOOD)
    face_crop, confidence = detect_face_face_recognition(frame)
    if face_crop is not None and face_crop.size > 0:
        return face_crop, True, confidence, "face_recognition"
    
    # Method 3: Try Haar Cascade (OK)
    face_crop, confidence = detect_face_haar(frame)
    if face_crop is not None and face_crop.size > 0:
        return face_crop, True, confidence, "Haar"
    
    # Method 4: Center crop fallback (face likely in center)
    crop_size = int(min(h, w) * 0.8)
    start_h = (h - crop_size) // 2
    start_w = (w - crop_size) // 2
    
    return frame[start_h:start_h+crop_size, start_w:start_w+crop_size], False, 0.3, "CenterCrop"

def process_frames(base64_frames):
    """Process frames for model input"""
    total = len(base64_frames)
    
    # Sample frames evenly
    if total <= FRAMES_TO_USE:
        indices = list(range(total))
    else:
        indices = np.linspace(0, total - 1, FRAMES_TO_USE, dtype=int)
    
    frames = []
    face_confidences = []
    valid_frames = 0
    detection_methods = []
    
    for idx in indices:
        img = decode_frame(base64_frames[idx])
        if img is None:
            continue
        
        # Detect and crop face
        img, has_face, confidence, method = detect_and_crop_face(img)
        
        if has_face:
            valid_frames += 1
        
        face_confidences.append(confidence)
        detection_methods.append(method)
        
        # Transform
        tensor = transform(img)
        frames.append(tensor)
    
    if len(frames) < MIN_FRAMES_REQUIRED:
        raise ValueError(f"Insufficient valid frames: {len(frames)}/{MIN_FRAMES_REQUIRED} required")
    
    # Pad if needed
    while len(frames) < FRAMES_TO_USE:
        frames.append(frames[-1].clone())
        face_confidences.append(face_confidences[-1])
        detection_methods.append(detection_methods[-1])
    
    frames = frames[:FRAMES_TO_USE]
    face_confidences = face_confidences[:FRAMES_TO_USE]
    
    avg_face_confidence = np.mean(face_confidences)
    primary_method = max(set(detection_methods), key=detection_methods.count)
    
    return torch.stack(frames).unsqueeze(0), valid_frames, avg_face_confidence, primary_method

@torch.no_grad()
def predict_ensemble(frames_tensor, faces_detected, face_confidence):
    """Ensemble prediction with weighted voting"""
    frames_tensor = frames_tensor.to(DEVICE)
    
    all_probs = []
    
    for model, model_acc, _ in ENSEMBLE_MODELS:
        _, logits = model(frames_tensor)
        probs = torch.softmax(logits, dim=1)[0]
        
        prob_real = probs[0].item()
        prob_fake = probs[1].item()
        
        weight = model_acc / 100.0
        all_probs.append((prob_real * weight, prob_fake * weight))
        
        logger.info(f"  Model ({model_acc}%): REAL={prob_real*100:.1f}% FAKE={prob_fake*100:.1f}%")
    
    weighted_real = sum(p[0] for p in all_probs) / len(all_probs)
    weighted_fake = sum(p[1] for p in all_probs) / len(all_probs)
    
    final_prediction = "FAKE" if weighted_fake > weighted_real else "REAL"
    final_confidence = max(weighted_real, weighted_fake) * 100
    
    # Apply penalties
    if faces_detected < FRAMES_TO_USE * 0.5:
        logger.warning(f"Low face detection: {faces_detected}/{FRAMES_TO_USE}")
        final_confidence *= 0.95

    if face_confidence < 0.8:
        logger.warning(f"Low face confidence: {face_confidence:.2f}")
        final_confidence *= 0.95

    margin = abs(weighted_fake - weighted_real)
    if margin < 0.2:
        logger.warning(f"Close prediction (margin={margin:.3f})")
        final_confidence *= 0.95
    
    logger.info(f"Ensemble: {final_prediction} | Confidence={final_confidence:.1f}%")
    
    return final_prediction, final_confidence, weighted_fake, weighted_real

# -------------------- API --------------------
@app.route('/frame_analyze', methods=['POST'])
def analyze():
    start = time.time()
    
    try:
        data = request.json
        frames = data.get('frames', [])
        
        if not frames:
            return jsonify({'error': 'No frames provided'}), 400
        
        if len(frames) < MIN_FRAMES_REQUIRED:
            return jsonify({'error': f'Need at least {MIN_FRAMES_REQUIRED} frames'}), 400
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Analysis: {len(frames)} frames")
        
        frames_tensor, faces, face_conf, method = process_frames(frames)
        logger.info(f"Detection: {method} | Faces: {faces}/{FRAMES_TO_USE} | Conf: {face_conf:.2f}")
        
        prediction, confidence, fake_prob, real_prob = predict_ensemble(frames_tensor, faces, face_conf)
        
        total_time = time.time() - start
        
        reliability = "HIGH"
        if confidence < 75 or faces < FRAMES_TO_USE * 0.5:
            reliability = "MEDIUM"
        if confidence < 65 or faces < FRAMES_TO_USE * 0.3:
            reliability = "LOW"
        
        result = {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'fake_probability': round(fake_prob * 100, 2),
            'real_probability': round(real_prob * 100, 2),
            'reliability': reliability,
            'frames_analyzed': FRAMES_TO_USE,
            'total_frames': len(frames),
            'faces_detected': faces,
            'face_detection_confidence': round(face_conf, 2),
            'face_detection_method': method,
            'processing_time': round(total_time, 2),
            'ensemble_size': len(ENSEMBLE_MODELS),
            'device': str(DEVICE),
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"RESULT: {prediction} ({confidence:.1f}%) - {reliability} | {total_time:.2f}s")
        logger.info(f"{'='*60}\n")
        
        return jsonify(result), 200
        
    except Exception as e:
        logger.error(f"Error: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    detection_method = "MTCNN (BEST)" if MTCNN_DETECTOR else ("face_recognition (GOOD)" if FACE_RECOGNITION_AVAILABLE else "Haar Cascade (FALLBACK)")
    
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'face_detection': detection_method,
        'ensemble_models': len(ENSEMBLE_MODELS),
        'models': [{'accuracy': acc, 'frames': frames} for _, acc, frames in ENSEMBLE_MODELS],
        'frames_per_analysis': FRAMES_TO_USE
    }), 200

if __name__ == '__main__':
    print("\n" + "="*70)
    print("VERIFEED BACKEND - PRODUCTION READY")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Face Detection: ", end="")
    if MTCNN_DETECTOR:
        print("MTCNN (BEST) ✓")
    elif FACE_RECOGNITION_AVAILABLE:
        print("face_recognition (GOOD) ✓")
    else:
        print("Haar Cascade (FALLBACK ONLY)")
    print(f"Models: {len(ENSEMBLE_MODELS)}")
    for i, (_, acc, frames) in enumerate(ENSEMBLE_MODELS, 1):
        print(f"  {i}. {acc}% accuracy, {frames} frames")
    print("="*70 + "\n")
    
    if not MTCNN_DETECTOR:
        print("⚠️  RECOMMENDATION: Install MTCNN for best accuracy:")
        print("    pip install facenet-pytorch")
        print()
    
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)