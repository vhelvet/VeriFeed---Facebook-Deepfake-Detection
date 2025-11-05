"""
VERIFEED PREDICTION BACKEND (SECURED)
Handles only inference/prediction for deepfake detection
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
import sys
import logging
from functools import wraps
from typing import List, Optional, Dict, Any

# Try to load .env file for secrets (Best practice)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # This is fine if the environment is set via a different method (e.g., k8s/Docker)
    pass 

# --------------------------------------------------------------------------
# LOGGING AND CONFIGURATION
# --------------------------------------------------------------------------

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration loaded from environment or defaults
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MIN_FACE_SIZE = 40
MAX_FACES = 300
MAX_ACCEPTABLE_FRAMES = MAX_FACES * 2 # Set a reasonable upper limit for DoS prevention
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = 'models'
MODEL_FILENAME = os.environ.get('MODEL_FILENAME', 'model_acc_95.00_e8.pt')


app = Flask(__name__)

# --------------------------------------------------------------------------
# CRITICAL SECURITY CONFIGURATION
# --------------------------------------------------------------------------

# 1. Production Mode and Secrets
# MUST be False in production!
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', 'default-insecure-key')

# Enforce secure key usage outside of development mode
if app.config['SECRET_KEY'] == 'default-insecure-key' and not app.config['DEBUG']:
    logger.error("FATAL: FLASK_SECRET_KEY is insecure. Terminating.")
    sys.exit(1)

# 2. Content Length Limit (DoS Prevention)
# Sets a cap on the total request size (e.g., 100MB)
MAX_CONTENT_MB = int(os.environ.get('MAX_CONTENT_MB', 100))
app.config['MAX_CONTENT_LENGTH'] = MAX_CONTENT_MB * 1024 * 1024

# 3. CORS Configuration Hardening (Least Privilege)
# Only allow specific origins (your extension/frontend)
extension_origins = os.environ.get('EXTENSION_ORIGINS', 'http://localhost:3000').split(',')
CORS(app, resources={r"/predict": {"origins": extension_origins, "methods": ["POST"]},
                     r"/health*": {"origins": "*", "methods": ["GET"]},
                     r"/model/*": {"origins": extension_origins, "methods": ["POST", "GET"]}})

# 4. Conceptual Authentication Decorator
# NOTE: This uses a simple shared secret (VERIFEED_AUTH_TOKEN) for demonstration.
# In a real-world scenario, replace this with a robust JWT/OAuth validation function.
def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth_token = request.headers.get('Authorization')
        required_token = os.environ.get('VERIFEED_AUTH_TOKEN')
        
        # Check 1: Token existence
        if not auth_token:
            return jsonify({'error': 'Authentication required. Token missing.'}), 401
        
        # Check 2: Token validation
        if auth_token != required_token:
            logger.warning("Unauthorized access attempt rejected.")
            return jsonify({'error': 'Authentication required. Access denied.'}), 401
            
        return f(*args, **kwargs)
    return decorated

# Model Architecture (Structure remains the same)
class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DeepfakeDetectionModel, self).__init__()
        # Load pre-trained ResNeXt model weights
        model = models.resnext50_32x4d(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional=bidirectional)
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

# Transforms
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Load model logic
inference_model: Optional[DeepfakeDetectionModel] = None
model_info: Dict[str, Any] = {'loaded': False, 'path': None, 'error': None}

def load_model(model_path: Optional[str] = None) -> bool:
    """Load the trained model, handling errors securely."""
    global inference_model, model_info
    
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
    
    try:
        if not os.path.exists(model_path):
            model_info['error'] = f"Model file '{MODEL_FILENAME}' not found at {model_path}"
            logger.error(model_info['error'])
            return False
        
        # Load model using map_location to mitigate potential serialization issues
        inference_model = DeepfakeDetectionModel(num_classes=2).to(DEVICE)
        inference_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
        inference_model.eval()
        
        model_info['loaded'] = True
        model_info['path'] = model_path
        model_info['error'] = None
        
        logger.info(f"✓ Model loaded successfully from {model_path}")
        return True
        
    except Exception as e:
        model_info['error'] = str(e)
        logger.error(f"Failed to load model: {e}")
        return False

# Try to load model on startup
load_model()

def decode_base64_frame(b64_frame: str) -> Optional[np.ndarray]:
    """Decode base64 frame to cv2 image with robust error handling."""
    try:
        # Input sanitation: remove data URI prefix if present
        if ',' in b64_frame:
            b64_frame = b64_frame.split(',', 1)[1]
            
        image_data = base64.b64decode(b64_frame)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            # Ensure color conversion is done
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        # Avoid logging raw frame data, only the error type
        logger.error(f"Error decoding frame: {e}") 
        return None

def detect_faces_from_frames(frames: List[np.ndarray], max_faces: int = MAX_FACES) -> Optional[List[np.ndarray]]:
    """Extract faces from frames (retaining original logic but protecting against over-processing)."""
    face_frames: List[np.ndarray] = []
    faces_found = 0
    detection_model = "cnn" if DEVICE.type == "cuda" else "hog"
    
    logger.info(f"Starting face detection (max: {max_faces})")
    
    for frame in frames:
        if faces_found >= max_faces:
            logger.info(f"Reached maximum face limit ({max_faces}). Stopping detection.")
            break
            
        if frame is None or frame.size == 0:
            continue
            
        try:
            h, w = frame.shape[:2]
            # Frame scaling logic to speed up face detection on large images
            scale_back = 1.0
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                scale_back = max(h, w) / 800
            else:
                small_frame = frame
            
            face_locations = face_recognition.face_locations(
                small_frame, model=detection_model, number_of_times_to_upsample=0
            )
            
            if len(face_locations) > 0:
                # Take the largest face (or just the first one found)
                top, right, bottom, left = face_locations[0]
                
                if scale_back != 1.0:
                    top = int(top * scale_back)
                    right = int(right * scale_back)
                    bottom = int(bottom * scale_back)
                    left = int(left * scale_back)
                
                face_img = frame[top:bottom, left:right, :]
                
                # Minimum size check to prevent small, artifact-filled crops
                if face_img.size > 0 and face_img.shape[0] >= MIN_FACE_SIZE and face_img.shape[1] >= MIN_FACE_SIZE:
                    faces_found += 1
                    face_frames.append(face_img)
                    
        except Exception as e:
            # Silently handle errors on individual frames
            continue
    
    logger.info(f"Total faces extracted: {len(face_frames)}")
    
    # Selection/Padding logic (retaining original sequence requirement)
    if len(face_frames) >= SEQUENCE_LENGTH:
        indices = np.linspace(0, len(face_frames) - 1, SEQUENCE_LENGTH, dtype=int)
        selected_faces = [face_frames[i] for i in indices]
        return selected_faces
    elif len(face_frames) > 0:
        # Pad if we have some faces but less than SEQUENCE_LENGTH
        while len(face_frames) < SEQUENCE_LENGTH:
            face_frames.append(face_frames[-1])
        return face_frames[:SEQUENCE_LENGTH]
    
    return None

# --------------------------------------------------------------------------
# ENDPOINTS
# --------------------------------------------------------------------------

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint (publicly accessible)"""
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'model_loaded': model_info['loaded'],
        'model_path': model_info['path'],
        'model_error': model_info['error'],
        'max_faces': MAX_FACES
    })

@app.route('/predict', methods=['POST'])
@requires_auth # CRITICAL: Authentication is required for this heavy endpoint.
def predict():
    """Main prediction endpoint - analyze video frames for deepfakes"""
    try:
        if not model_info['loaded']:
            # Use 503 Service Unavailable when the service dependency (model) is missing
            return jsonify({
                'error': 'Model not loaded',
                'details': model_info['error']
            }), 503
        
        # --- Input Validation and DoS Prevention ---
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 415
            
        data = request.json
        frames_b64 = data.get('frames', [])
        
        if not isinstance(frames_b64, list):
             return jsonify({'error': 'Input frames must be a list'}), 400
        
        # Limit the number of frames to prevent excessive memory/CPU use
        if not frames_b64 or len(frames_b64) > MAX_ACCEPTABLE_FRAMES: 
            return jsonify({'error': f'Invalid number of frames provided ({len(frames_b64)}). Max acceptable: {MAX_ACCEPTABLE_FRAMES}'}), 400
        # --- End Validation ---
            
        logger.info(f"Received {len(frames_b64)} frames for prediction")
        
        # Decode frames
        frames: List[np.ndarray] = []
        for b64_frame in frames_b64:
            # Validate individual frame structure before heavy base64 decoding
            if not isinstance(b64_frame, str) or len(b64_frame) < 100:
                logger.warning("Skipping frame due to invalid type or insufficient length.")
                continue

            frame = decode_base64_frame(b64_frame)
            if frame is not None:
                frames.append(frame)
        
        if len(frames) < 10:
            return jsonify({'error': f'Not enough valid frames (minimum 10 required) after decoding/filtering. Only {len(frames)} valid frames remain.'}), 400
        
        logger.info(f"Successfully decoded {len(frames)} frames")
        
        # Detect faces (heavy computation)
        face_frames = detect_faces_from_frames(frames, max_faces=MAX_FACES)
        
        if face_frames is None or len(face_frames) == 0:
            return jsonify({'error': 'No faces detected in video or faces too small'}), 400
        
        # Prepare input tensor
        transformed_frames = []
        for frame in face_frames:
            frame_tensor = val_transforms(frame)
            transformed_frames.append(frame_tensor)
        
        sequence = torch.stack(transformed_frames).unsqueeze(0).to(DEVICE)
        
        # Run inference
        with torch.no_grad():
            outputs = inference_model(sequence)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0][predicted_class].item() * 100
            
            fake_confidence = probabilities[0][0].item() * 100
            real_confidence = probabilities[0][1].item() * 100
        
        prediction = "REAL" if predicted_class == 1 else "FAKE"
        
        logger.info(f"Prediction: {prediction}, Confidence: {confidence:.2f}%")
        
        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'fake_probability': round(fake_confidence, 2),
            'real_probability': round(real_confidence, 2),
            'faces_analyzed': len(face_frames),
            'frames_processed': len(frames)
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        # Return a generic, non-informative error message to the client in production
        return jsonify({'error': 'Internal server error during prediction'}), 500


@app.route('/model/reload', methods=['POST'])
@requires_auth # CRITICAL: This endpoint must be restricted to administrators only.
def reload_model():
    """Reload the model (useful after training)"""
    try:
        data = request.json or {}
        model_path = data.get('model_path', None)
        
        # Security enhancement: Sanitize input path against directory traversal (e.g., ../../)
        if model_path is not None and ('..' in model_path or not model_path.endswith('.pt')):
             return jsonify({'success': False, 'error': 'Invalid model path format or disallowed directory traversal'}), 400
        
        success = load_model(model_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully',
                'model_path': model_info['path']
            })
        else:
            return jsonify({
                'success': False,
                'error': model_info['error']
            }), 500
            
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        return jsonify({'error': 'Internal server error during model reload'}), 500


@app.route('/model/info', methods=['GET'])
@requires_auth # Restrict access to internal model details
def model_info_endpoint():
    """Get model information"""
    return jsonify({
        'loaded': model_info['loaded'],
        'path': model_info['path'],
        'error': model_info['error'],
        'device': str(DEVICE),
        'sequence_length': SEQUENCE_LENGTH,
        'image_size': IM_SIZE,
        'max_faces': MAX_FACES
    })


if __name__ == '__main__':
    # Startup logging (retain for diagnostics)
    print("\n" + "="*70)
    print("🔮 VERIFEED PREDICTION SERVER")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Model file: {MODEL_FILENAME}")
    print(f"Model loaded: {model_info['loaded']}")
    if model_info['loaded']:
        print(f"✓ Model path: {model_info['path']}")
    else:
        print(f"✗ Model error: {model_info['error']}")
        print(f"  Please ensure '{MODEL_FILENAME}' exists in '{MODELS_DIR}/' directory")
    print(f"Max request size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f} MB")
    print(f"Debug Mode: {app.config['DEBUG']}")
    print("="*70 + "\n")
    
    if not model_info['loaded']:
        print("⚠️  WARNING: Server starting without loaded model!")
        
    from waitress import serve
    print("🚀 Starting Waitress production server on http://0.0.0.0:5000")
    
    # Waitress does not support debug mode; ensure app.config['DEBUG'] is False
    serve(app, host='0.0.0.0', port=5000) 
