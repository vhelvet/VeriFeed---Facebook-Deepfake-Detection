"""
SIMPLIFIED FLASK SERVER - First 100 Face Detection
Based on research paper preprocessing methodology
"""

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

SUPPORTED_PLATFORM = "facebook"
SEQUENCE_LENGTH = 20
MIN_FRAMES_REQUIRED = 10
MIN_FACE_SIZE = 40
CLASS_MAP = {0: "FAKE", 1: "REAL"}

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

# -------------------- MODEL ARCHITECTURE --------------------
class Model(nn.Module):
    def __init__(self, num_classes, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False, lstm_bias=False):
        super(Model, self).__init__()
        model = models.resnext50_32x4d(weights=None)
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional, bias=lstm_bias)
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
        return out

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

def detect_first_100_faces(base64_frames, sequence_length=20):
    """
    Extract first 100 faces from 300 frames
    Based on paper: "only first 100 frames for training the model"
    """
    face_frames = []
    frames_with_faces = 0
    total_frames = len(base64_frames)
    
    logger.info(f"Starting face detection on {total_frames} frames")
    logger.info(f"Target: First 100 face detections, collecting {sequence_length} for model")
    
    detection_model = "cnn" if DEVICE.type == "cuda" else "hog"
    faces_found = 0
    
    # Process all frames until we find 100 faces
    for i in range(total_frames):
        if faces_found >= 100:
            logger.info(f"✓ Found 100 faces, stopping at frame {i}")
            break
            
        frame = decode_base64_frame(base64_frames[i])
        if frame is None:
            continue
        
        try:
            # Resize for faster detection
            h, w = frame.shape[:2]
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                scale_back = max(h, w) / 800
            else:
                small_frame = frame
                scale_back = 1.0
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                small_frame,
                model=detection_model,
                number_of_times_to_upsample=0
            )
            
            if len(face_locations) > 0:
                # Take first face only
                top, right, bottom, left = face_locations[0]
                
                # Scale back to original size
                if scale_back != 1.0:
                    top = int(top * scale_back)
                    right = int(right * scale_back)
                    bottom = int(bottom * scale_back)
                    left = int(left * scale_back)
                
                # Crop face
                face_img = frame[top:bottom, left:right, :]
                
                if face_img.size > 0 and face_img.shape[0] >= MIN_FACE_SIZE and face_img.shape[1] >= MIN_FACE_SIZE:
                    faces_found += 1
                    frames_with_faces += 1
                    
                    # Only collect frames needed for model sequence
                    if len(face_frames) < sequence_length:
                        frame_tensor = train_transforms(face_img)
                        face_frames.append(frame_tensor)
                        
                        if len(face_frames) % 10 == 0:
                            logger.info(f"  Collected {len(face_frames)}/{sequence_length} frames ({faces_found} faces found)")
                    
        except Exception as e:
            logger.debug(f"Face detection error frame {i}: {e}")
            continue

    if len(face_frames) == 0:
        raise ValueError("No faces detected in video. Cannot analyze.")
    
    if len(face_frames) < MIN_FRAMES_REQUIRED:
        raise ValueError(
            f"Insufficient face frames: got {len(face_frames)}, need at least {MIN_FRAMES_REQUIRED}"
        )
    
    # Pad if needed
    if len(face_frames) < sequence_length:
        logger.warning(f"Only got {len(face_frames)} face frames, padding to {sequence_length}")
        last_frame = face_frames[-1]
        while len(face_frames) < sequence_length:
            face_frames.append(last_frame)
    
    logger.info(
        f"✓ Final: {len(face_frames)} frames for model, "
        f"{frames_with_faces} frames had faces, "
        f"{faces_found} total faces detected"
    )
    
    frames_tensor = torch.stack(face_frames[:sequence_length])
    return frames_tensor.unsqueeze(0), frames_with_faces

def predict(model, img_tensor):
    """Simple prediction without visualization"""
    img = img_tensor.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        logits = model(img)
        probs = sm(logits)
        _, prediction = torch.max(probs, 1)
        pred_idx = int(prediction.item())
        confidence = probs[:, pred_idx].item() * 100
        probs_np = probs.cpu().numpy()[0]
        
        logger.info(f"Probabilities: FAKE={probs_np[0]:.4f} ({probs_np[0]*100:.2f}%), REAL={probs_np[1]:.4f} ({probs_np[1]*100:.2f}%)")
        logger.info(f"Prediction: {CLASS_MAP[pred_idx]} with {confidence:.2f}% confidence")
        
        return pred_idx, confidence, probs_np.tolist()

def parse_model_info(filename):
    """Parse model accuracy and sequence length from filename"""
    name = os.path.basename(filename)
    
    acc_match = re.search(r'model_(\d+)_acc', name, re.IGNORECASE)
    accuracy = int(acc_match.group(1)) if acc_match else 0
    
    seq_patterns = [
        r'_acc_(\d+)_frames',
        r'acc_(\d+)_frames',
        r'(\d+)_frames',
        r'seq[\-_]?(\d+)',
    ]
    
    seq_len = None
    for pattern in seq_patterns:
        match = re.search(pattern, name, re.IGNORECASE)
        if match:
            potential_seq = int(match.group(1))
            if 5 <= potential_seq <= 150:
                seq_len = potential_seq
                break
    
    if seq_len is None:
        seq_len = SEQUENCE_LENGTH
        logger.warning(f"Could not parse sequence length from {name}, using default {SEQUENCE_LENGTH}")
    
    return accuracy, seq_len

def get_best_model(num_frames_with_faces):
    """
    Select best model based on number of frames with detected faces
    Strategy: More frames with faces = Use longer sequence models
    """
    available_models = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    
    if not available_models:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}")
    
    if len(available_models) == 1:
        model_path = available_models[0]
        _, seq_len = parse_model_info(model_path)
        logger.info(f"Using only available model: {os.path.basename(model_path)}")
        return model_path, seq_len
    
    model_info = []
    for model_path in available_models:
        accuracy, seq_len = parse_model_info(model_path)
        model_info.append({
            'path': model_path,
            'accuracy': accuracy,
            'seq_len': seq_len,
            'name': os.path.basename(model_path)
        })
    
    logger.info(f"Available models for {num_frames_with_faces} frames with faces:")
    for info in model_info:
        logger.info(f"  - {info['name']}: acc={info['accuracy']}%, seq={info['seq_len']}")
    
    # Selection: More frames with faces → longer sequence model
    if num_frames_with_faces <= 30:
        best_model = min(model_info, key=lambda x: (x['seq_len'], -x['accuracy']))
        logger.info(f"✓ Few face frames ({num_frames_with_faces}) → Selecting SHORTER sequence model")
    else:
        best_model = max(model_info, key=lambda x: (x['seq_len'], x['accuracy']))
        logger.info(f"✓ Many face frames ({num_frames_with_faces}) → Selecting LONGER sequence model")
    
    logger.info(f"✓ Selected: {best_model['name']} (acc={best_model['accuracy']}%, seq={best_model['seq_len']})")
    
    return best_model['path'], best_model['seq_len']

MODEL_CACHE = {}

def load_model_cached(model_path):
    """Load model with caching"""
    if model_path in MODEL_CACHE:
        logger.info(f"Using cached model: {os.path.basename(model_path)}")
        return MODEL_CACHE[model_path]
    
    logger.info(f"Loading model: {os.path.basename(model_path)}")
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except:
        state_dict = torch.load(model_path, map_location=DEVICE)
    
    has_lstm_bias = 'lstm.bias_ih_l0' in state_dict
    
    model = Model(num_classes=2, lstm_bias=has_lstm_bias)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    MODEL_CACHE[model_path] = model
    logger.info(f"✓ Model loaded successfully")
    return model

# -------------------- API ENDPOINTS --------------------
@app.route('/frame_analyze', methods=['POST'])
def analyze_frames():
    """
    Main endpoint: Detect first 100 faces, select model, predict
    Based on research paper preprocessing methodology
    """
    start_time = time.time()
    request_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        platform = data.get('platform', 'unknown').lower()

        if platform != SUPPORTED_PLATFORM:
            return jsonify({'error': f'Unsupported platform: {platform}'}), 400
        if not frames:
            return jsonify({'error': 'No frames provided'}), 400
        
        logger.info(f"Received {len(frames)} frames from {platform}")
        
        # STEP 1: Detect first 100 faces from all frames
        logger.info("STEP 1: Detecting first 100 faces from frames...")
        detection_start = time.time()
        
        # Get a quick preview of faces for model selection
        quick_face_count = 0
        sample_limit = min(50, len(frames))
        detection_model = "cnn" if DEVICE.type == "cuda" else "hog"
        
        for i in range(sample_limit):
            frame = decode_base64_frame(frames[i])
            if frame is None:
                continue
            try:
                h, w = frame.shape[:2]
                if max(h, w) > 800:
                    scale = 800 / max(h, w)
                    small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                else:
                    small_frame = frame
                
                face_locations = face_recognition.face_locations(
                    small_frame,
                    model=detection_model,
                    number_of_times_to_upsample=0
                )
                
                if len(face_locations) > 0:
                    quick_face_count += 1
                    
            except Exception as e:
                logger.debug(f"Quick detection error frame {i}: {e}")
                continue
        
        logger.info(f"Preview: {quick_face_count} frames with faces in first {sample_limit} frames")
        
        # STEP 2: Select model based on preview
        logger.info("STEP 2: Selecting model based on face frame count...")
        model_path, expected_seq_len = get_best_model(quick_face_count)
        logger.info(f"Model expects {expected_seq_len} frames")
        
        # STEP 3: Full face detection (first 100 faces)
        logger.info("STEP 3: Processing frames to extract first 100 faces...")
        frames_tensor, frames_with_faces = detect_first_100_faces(frames, sequence_length=expected_seq_len)
        detection_time = time.time() - detection_start
        
        logger.info(f"Face detection took {detection_time:.2f}s")
        
        # STEP 4: Load model
        logger.info("STEP 4: Loading model...")
        model = load_model_cached(model_path)

        # STEP 5: Predict
        logger.info("STEP 5: Running prediction...")
        inference_start = time.time()
        pred_idx, confidence, probs = predict(model, frames_tensor)
        inference_time = time.time() - inference_start
        
        prediction_label = CLASS_MAP[pred_idx]
        total_time = time.time() - start_time
        finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        response = {
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'raw_prediction': int(pred_idx),
            'probabilities': {
                'FAKE': round(probs[0] * 100, 2),
                'REAL': round(probs[1] * 100, 2)
            },
            'model_used': os.path.basename(model_path),
            'model_sequence_length': expected_seq_len,
            'frames_with_faces': frames_with_faces,
            'frames_received': len(frames),
            'processing_time_seconds': round(total_time, 2),
            'detection_time_seconds': round(detection_time, 2),
            'inference_time_seconds': round(inference_time, 2),
            'device': str(DEVICE),
            'started_at': request_time,
            'finished_at': finish_time,
            'timestamp': datetime.now().isoformat()
        }

        print("\n" + "="*70)
        print("🎯 VERIFEED DEEPFAKE DETECTION RESULTS")
        print("="*70)
        print(f"🕒 Started:           {request_time}")
        print(f"📊 Model:             {os.path.basename(model_path)}")
        print(f"📦 Sequence Length:   {expected_seq_len} frames")
        print(f"📥 Frames Received:   {len(frames)}")
        print(f"👤 Frames w/ Faces:   {frames_with_faces}")
        print(f"🎯 Prediction:        {prediction_label} ({confidence:.2f}%)")
        print(f"📈 Probabilities:     FAKE={probs[0]*100:.2f}% | REAL={probs[1]*100:.2f}%")
        print(f"💻 Device:            {DEVICE}")
        print(f"⏱️  Total Time:        {total_time:.2f}s")
        print(f"   └─ Detection:      {detection_time:.2f}s")
        print(f"   └─ Inference:      {inference_time:.2f}s")
        print("="*70 + "\n")

        return jsonify(response), 200

    except ValueError as ve:
        logger.error(f"Validation error: {ve}")
        return jsonify({'error': str(ve)}), 400
        
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return jsonify({'error': f'Analysis failed: {str(e)}'}), 500

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    
    models_info = []
    for mf in model_files:
        acc, seq = parse_model_info(mf)
        models_info.append({
            'filename': os.path.basename(mf),
            'accuracy': acc,
            'sequence_length': seq
        })
    
    return jsonify({
        'status': 'healthy',
        'models_available': len(model_files),
        'models': models_info,
        'device': str(DEVICE),
        'methodology': 'First 100 faces detection from 300 frames',
        'class_mapping': CLASS_MAP,
        'timestamp': datetime.now().isoformat()
    }), 200

# -------------------- STARTUP --------------------
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 VERIFEED DEEPFAKE DETECTION SERVER")
    print("="*70)
    print(f"🕒 Server Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Device:            {DEVICE}")
    print(f"📁 Models Directory:  {MODELS_DIR}")
    print(f"🏷️  Label Mapping:     {CLASS_MAP}")
    print(f"📖 Methodology:       First 100 faces from 300 frames")

    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    print(f"\n📦 Available Models: {len(model_files)}")

    if model_files:
        print("🔍 Model Details:")
        for mf in model_files:
            acc, seq = parse_model_info(mf)
            print(f"   - {os.path.basename(mf)}: {acc}% acc, {seq} frames")

    print("\n✨ PROCESSING FLOW:")
    print("   1. Receive up to 300 frames from frontend")
    print("   2. Detect first 100 faces from frames")
    print("   3. Select model based on frames with faces")
    print("   4. Feed face frames to selected model")
    print("   5. Return prediction results")
    
    print("="*70)
    print("✅ Server ready at http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='localhost', port=5000, debug=False, threaded=True)