from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import face_recognition
import base64
import os
import glob
import logging
from datetime import datetime
import time
import warnings
import re

# -------------------- CONFIGURATION --------------------
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

DETECTED_FACES_DIR = 'detected_faces'
os.makedirs(DETECTED_FACES_DIR, exist_ok=True)

SAVE_FACES = True
SUPPORTED_PLATFORM = "facebook"
MIN_FRAMES = 10  # Minimum faces required
MAX_FRAMES = 100  # Maximum frames to accept from client (but will stop early if target reached)
INFER_MAX_SEQ = 40  # *** CHANGED TO 10 - Target faces to collect ***

# Try flipping if predictions are wrong
LABEL_FLIP = False  # Set to True if REAL/FAKE are reversed

logging.basicConfig(
    level=logging.WARNING,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

app = Flask(__name__)
CORS(app)

# -------------------- MODEL CONFIGURATION --------------------
im_size = 112
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]
MODELS_DIR = 'models'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

sm = nn.Softmax(dim=1)

# Class mapping with flip support
BASE_CLASS_MAP = {0: "FAKE", 1: "REAL"}
CLASS_MAP = {0: "REAL", 1: "FAKE"} if LABEL_FLIP else BASE_CLASS_MAP

# -------------------- MODEL ARCHITECTURE --------------------
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(Model, self).__init__()
        base_model = models.resnext50_32x4d(weights=None)
        self.model = nn.Sequential(*list(base_model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
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

# -------------------- HELPERS --------------------
def decode_base64_frame(base64_string):
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        logger.debug(f"Error decoding base64 frame: {e}")
        return None

def process_frames(base64_frames, max_faces=10):
    """
    Fast face detection - processes frames until we have EXACTLY 10 faces
    Stops immediately when 10 faces is reached
    """
    frames = []
    faces_detected = 0
    frames_processed = 0
    
    # Process up to MAX_FRAMES from client
    frames_to_check = min(len(base64_frames), MAX_FRAMES)
    
    logger.info(f"Starting face detection (will stop at exactly {max_faces} faces)")
    
    for i in range(frames_to_check):
        # STOP IMMEDIATELY when we have exactly 10 faces
        if len(frames) >= max_faces:
            logger.info(f"✓ Reached exactly {max_faces} faces at frame {i}, stopping detection")
            break
            
        frame = decode_base64_frame(base64_frames[i])
        if frame is None:
            continue
        
        frames_processed += 1
        
        # Downscale for faster detection
        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        
        try:
            # Fast HOG-based face detection
            face_locations = face_recognition.face_locations(small_frame, model="hog", number_of_times_to_upsample=1)
            
            if len(face_locations) > 0:
                # Use first detected face
                top, right, bottom, left = face_locations[0]
                
                # Scale back to original size
                top, right, bottom, left = top*2, right*2, bottom*2, left*2
                
                # Add small padding
                padding = 10
                top = max(0, top - padding)
                left = max(0, left - padding)
                bottom = min(frame.shape[0], bottom + padding)
                right = min(frame.shape[1], right + padding)
                
                # Extract face
                face_img = frame[top:bottom, left:right, :]
                
                if face_img.size > 0:
                    # Transform and add to list FIRST, then count
                    try:
                        frame_tensor = train_transforms(face_img)
                        frames.append(frame_tensor)
                        faces_detected += 1
                        
                        # Save face if enabled (optional, doesn't affect count)
                        if SAVE_FACES:
                            try:
                                face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                                filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_frame{i}.jpg"
                                cv2.imwrite(os.path.join(DETECTED_FACES_DIR, filename), face_bgr)
                            except:
                                pass
                                
                    except Exception as e:
                        logger.debug(f"Transform failed for frame {i}: {e}")
                        
        except Exception as e:
            logger.debug(f"Face detection failed for frame {i}: {e}")

    if len(frames) == 0:
        raise ValueError(f"No faces detected in {frames_processed} frames")
    
    logger.info(f"✓ Detection complete: {faces_detected} faces from {frames_processed} frames processed")
    
    # Stack frames (already limited by early stopping at 10)
    frames_tensor = torch.stack(frames)
    return frames_tensor.unsqueeze(0), faces_detected

def predict(model, img):
    """Make prediction with the model"""
    img = img.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        fmap, logits = model(img)
        probs = sm(logits)
        probs_np = probs.cpu().numpy()[0]
        
        pred_idx = int(np.argmax(probs_np))
        confidence = float(probs_np[pred_idx]) * 100.0
        
        # Get top predictions
        top_indices = np.argsort(probs_np)[::-1][:2]
        top_predictions = []
        for idx in top_indices:
            top_predictions.append({
                'class': CLASS_MAP.get(int(idx), str(idx)),
                'confidence': float(probs_np[idx]) * 100.0
            })
        
        logger.info(f"Raw probabilities: {probs_np.tolist()}")
        logger.info(f"Prediction: idx={pred_idx} -> {CLASS_MAP[pred_idx]} ({confidence:.2f}%)")
        
        return pred_idx, confidence, probs_np.tolist(), top_predictions

def parse_model_info(filename):
    """Extract both accuracy and sequence length from model filename"""
    name = os.path.basename(filename)
    
    # Extract accuracy (e.g., model_84_acc or model_97_acc)
    acc_match = re.search(r'model_(\d+)_acc', name, re.IGNORECASE)
    accuracy = int(acc_match.group(1)) if acc_match else 0
    
    # Extract sequence length
    seq_patterns = [
        r'_acc_(\d+)_frames',  # model_XX_acc_20_frames (your format!)
        r'acc_(\d+)_frames',   # acc_20_frames
        r'(\d+)_frames',       # 20_frames
        r'seq[\-_]?(\d+)',     # seq20, seq-20, seq_20
        r'_(\d+)frames',       # _20frames
        r's(\d+)_',            # s20_
    ]
    
    seq_len = None
    for pattern in seq_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            try:
                potential_seq = int(match.group(1))
                # Sanity check: sequence lengths are typically 10-150
                if 5 <= potential_seq <= 150:
                    seq_len = potential_seq
                    break
            except:
                continue
    
    return accuracy, seq_len

def get_accurate_model(face_count):
    """
    Select best model based on:
    1. Closest sequence length to face_count
    2. Highest accuracy when sequence lengths are similar
    """
    available_models = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    
    if not available_models:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}")
    
    # If only one model, use it
    if len(available_models) == 1:
        logger.info(f"Only one model available, using: {os.path.basename(available_models[0])}")
        return available_models[0]
    
    # Parse all models and find candidates
    candidates = []
    logger.info(f"Selecting model for {face_count} detected faces:")
    
    for model_path in available_models:
        accuracy, seq_len = parse_model_info(model_path)
        
        if seq_len is not None:
            # Calculate difference between model's seq length and our face count
            diff = abs(seq_len - face_count)
            # Store: (seq_diff, negative_accuracy, model_path, accuracy, seq_len)
            # Negative accuracy so higher accuracy comes first when sorting
            candidates.append((diff, -accuracy, model_path, accuracy, seq_len))
            logger.info(f"  - {os.path.basename(model_path)}: acc={accuracy}%, seq={seq_len}, diff={diff}")
        else:
            # If we can't parse, put it at the end with high penalty
            candidates.append((9999, 0, model_path, accuracy, None))
            logger.warning(f"  - {os.path.basename(model_path)}: acc={accuracy}%, seq=unknown")
    
    # Sort by:
    # 1. Sequence length difference (smallest first)
    # 2. Accuracy (highest first - that's why we use negative)
    candidates.sort(key=lambda x: (x[0], x[1]))
    
    chosen = candidates[0]
    chosen_model = chosen[2]
    chosen_acc = chosen[3]
    chosen_seq = chosen[4]
    
    # Show top 3 candidates for transparency
    logger.info(f"Top 3 candidates:")
    for i, (diff, neg_acc, path, acc, seq) in enumerate(candidates[:3]):
        logger.info(f"  {i+1}. {os.path.basename(path)} - acc={acc}%, seq={seq}, diff={diff}")
    
    logger.info(f"✓ SELECTED: {os.path.basename(chosen_model)} (acc={chosen_acc}%, seq={chosen_seq}, best match for {face_count} faces)")
    return chosen_model

MODEL_CACHE = {}
def load_model_cached(model_path):
    """Load model with caching"""
    if model_path in MODEL_CACHE:
        logger.info(f"Using cached model: {os.path.basename(model_path)}")
        return MODEL_CACHE[model_path]
    
    logger.info(f"Loading model: {os.path.basename(model_path)}")
    model = Model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=True)
    model.to(DEVICE)
    model.eval()
    MODEL_CACHE[model_path] = model
    return model

# -------------------- API --------------------
@app.route('/frame_analyze', methods=['POST'])
def analyze_frames():
    start_time = time.time()
    request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        platform = data.get('platform', 'unknown').lower()

        if platform != SUPPORTED_PLATFORM:
            return jsonify({'error': 'Unsupported platform'}), 400
        if not frames:
            return jsonify({'error': 'No frames provided'}), 400

        # Process frames and detect faces - ALWAYS COLLECT EXACTLY 10
        detection_start = time.time()
        max_faces_to_collect = 10  # *** HARDCODED TO 10 ***
        frames_tensor, faces_detected = process_frames(frames, max_faces=max_faces_to_collect)
        detection_time = time.time() - detection_start
        
        logger.info(f"Collected {faces_detected} faces in {detection_time:.2f}s")
        
        # Check minimum faces requirement (always 10)
        if faces_detected < MIN_FRAMES:
            return jsonify({
                'status': 'error',
                'message': f'Too few faces detected. Found {faces_detected}, need exactly {MIN_FRAMES}.',
                'faces_detected': faces_detected
            }), 400
        
        # Select appropriate model based on face count (will be 10)
        model_path = get_accurate_model(faces_detected)
        model = load_model_cached(model_path)

        # Run prediction
        inference_start = time.time()
        pred_idx, confidence, probs, top_predictions = predict(model, frames_tensor)
        inference_time = time.time() - inference_start
        
        output = CLASS_MAP.get(pred_idx, "UNKNOWN")
        processing_time = round(time.time() - start_time, 2)
        finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        response = {
            'model_used': os.path.basename(model_path),
            'model_path': model_path,
            'device': str(DEVICE),
            'prediction': output,
            'raw_prediction': int(pred_idx),
            'confidence': round(confidence, 2),
            'all_probabilities': probs,
            'top_predictions': top_predictions,
            'frames_analyzed': frames_tensor.shape[1],
            'faces_detected': faces_detected,
            'processing_time_seconds': processing_time,
            'detection_time_seconds': round(detection_time, 2),
            'inference_time_seconds': round(inference_time, 2),
            'started_at': request_time,
            'finished_at': finish_time,
            'timestamp': datetime.now().isoformat(),
            'label_mapping': CLASS_MAP,
            'label_flip_enabled': LABEL_FLIP
        }

        # Console output
        print("\n" + "="*60)
        print("VERIFEED PREDICTION SUMMARY")
        print("="*60)
        print(f"🕒 Started: {request_time}")
        print(f"📊 Model: {os.path.basename(model_path)}")
        print(f"👤 Faces Detected: {faces_detected} (EXACTLY 10)")
        print(f"🎯 Prediction: {output} ({confidence:.2f}%)")
        print(f"📈 All Predictions: {top_predictions}")
        print(f"🔧 Label Flip: {LABEL_FLIP} | Mapping: {CLASS_MAP}")
        print(f"⏱️  Detection: {detection_time:.2f}s | Inference: {inference_time:.2f}s | Total: {processing_time}s")
        print(f"🕒 Finished: {finish_time}")
        print("="*60 + "\n")

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error during analysis: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health():
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    return jsonify({
        'status': 'healthy',
        'models_available': len(model_files),
        'model_files': [os.path.basename(f) for f in model_files],
        'device': str(DEVICE),
        'label_flip': LABEL_FLIP,
        'class_mapping': CLASS_MAP,
        'max_faces': 10,
        'timestamp': datetime.now().isoformat()
    }), 200

# -------------------- MAIN --------------------
if __name__ == '__main__':
    print("=" * 70)
    print("VERIFEED SERVER - DEEPFAKE DETECTION (10 FACES ONLY)")
    print(f"🕒 Server started at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"🏷️  Label Mapping: {CLASS_MAP}")
    print(f"🔄 Label Flip: {LABEL_FLIP}")
    print(f"💻 Device: {DEVICE}")
    print(f"📁 Models Directory: {MODELS_DIR}")
    print(f"👤 Face Detection: EXACTLY 10 FACES")

    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    print(f"📊 Available Models: {len(model_files)}")

    if not model_files:
        print("⚠️  No models found in the models directory!")
    else:
        best_model = None
        best_acc = -1
        best_seq = None

        print("🔍 Scanning available models:")
        for mf in model_files:
            acc, seq = parse_model_info(mf)
            seq_display = seq if seq is not None else 'unknown'
            print(f"   - {os.path.basename(mf)} → Accuracy: {acc}% | Sequence: {seq_display}")

            # Keep the highest accuracy model
            if acc > best_acc:
                best_acc = acc
                best_model = mf
                best_seq = seq

        print("\n🏆 Best model selected automatically:")
        if best_model:
            print(f"   📦 {os.path.basename(best_model)} (Accuracy: {best_acc}%, Sequence: {best_seq})")
        else:
            print("   ⚠️  No valid model info found.")

    print("=" * 70)
    app.run(host='localhost', port=5000, debug=False, threaded=True)