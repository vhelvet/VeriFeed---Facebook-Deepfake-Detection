"""
app.py
VERIFEED PREDICTION BACKEND - PRODUCTION SECURED + FIXES
All critical security and deployment issues resolved
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import face_recognition
import base64
import os
import logging
import io
import time
from PIL import Image
import traceback
from functools import wraps, lru_cache
import jwt
from datetime import datetime, timedelta
import hashlib
import secrets
import threading
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

load_dotenv()

# --- FLASK SETUP ---
app = Flask(__name__)

# --- PRODUCTION SECURITY CONFIGURATION ---
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_MB', '100')) * 1024 * 1024

# Security Keys - FIXED: Require in production, generate only in dev
SECRET_KEY = os.environ.get('JWT_SECRET_KEY')
API_KEY = os.environ.get('API_KEY')
ADMIN_API_KEY = os.environ.get('ADMIN_API_KEY')

# Generate defaults ONLY in development mode
if app.config['DEBUG']:
    if not SECRET_KEY:
        SECRET_KEY = secrets.token_hex(32)
        print("⚠️  DEV: Using generated JWT_SECRET_KEY")
    if not API_KEY:
        API_KEY = secrets.token_hex(32)
        print("⚠️  DEV: Using generated API_KEY")
    if not ADMIN_API_KEY:
        ADMIN_API_KEY = secrets.token_hex(32)
        print("⚠️  DEV: Using generated ADMIN_API_KEY")
else:
    # Production mode - keys are REQUIRED
    if not SECRET_KEY or not API_KEY or not ADMIN_API_KEY:
        raise ValueError(
            "CRITICAL: Missing required environment variables in production: "
            "JWT_SECRET_KEY, API_KEY, and ADMIN_API_KEY must be set"
        )

# DoS Prevention Settings
MAX_FRAMES_INPUT = int(os.environ.get('MAX_FRAMES_INPUT', '100'))
REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', '60'))
RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
RATE_LIMIT_PER_MINUTE = os.environ.get('RATE_LIMIT_PER_MINUTE', '20')
RATE_LIMIT_PER_HOUR = os.environ.get('RATE_LIMIT_PER_HOUR', '200')
RATE_LIMIT_PER_DAY = os.environ.get('RATE_LIMIT_PER_DAY', '1000')

# Model Security
ALLOW_MODEL_RELOAD = os.environ.get('ALLOW_MODEL_RELOAD', 'false').lower() == 'true'

# FIXED: Never log API keys, even partially
print(f"🛡️  Production Mode: {not app.config['DEBUG']}")
print(f"🚦 Rate Limiting: {RATE_LIMIT_ENABLED}")
print(f"📦 Max Content Size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
print(f"🔑 API Keys: {'✓ Configured' if API_KEY else '✗ Missing'}")

# --- CORS CONFIGURATION ---
ALLOWED_ORIGINS = os.environ.get('ALLOWED_ORIGINS', 
    'chrome-extension://iljbbfgejddphakhekbonjioflbodjoh,http://localhost,http://127.0.0.1'
).split(',')

CORS(app, resources={
    r"/*": {
        "origins": ALLOWED_ORIGINS,
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": [
            "Content-Type",
            "Authorization",
            "X-API-Key",
            "X-Requested-With"
        ],
        "expose_headers": ["Content-Type", "Authorization"],
        "supports_credentials": True,
        "max_age": 3600
    }
})

# --- RATE LIMITING SETUP ---
if RATE_LIMIT_ENABLED:
    limiter = Limiter(
        app=app,
        key_func=get_remote_address,
        default_limits=[f"{RATE_LIMIT_PER_DAY} per day", f"{RATE_LIMIT_PER_HOUR} per hour"],
        storage_uri="memory://",
        strategy="fixed-window"
    )
else:
    class DummyLimiter:
        def limit(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    limiter = DummyLimiter()

# --- LOGGING CONFIGURATION ---
logging.basicConfig(
    level=logging.INFO if not app.config['DEBUG'] else logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

if not app.config['DEBUG']:
    logging.getLogger('werkzeug').setLevel(logging.WARNING)

# --- AUTHENTICATION DECORATORS ---
@lru_cache(maxsize=256)
def hash_api_key(key):
    """Hash API key for secure comparison - cached"""
    return hashlib.sha256(key.encode()).hexdigest()

def requires_auth(f):
    """Decorator for endpoints requiring authentication"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if request.method == "OPTIONS":
            return jsonify({"status": "ok"}), 200

        api_key = request.headers.get('X-API-Key')
        auth_header = request.headers.get('Authorization')

        if api_key:
            if hash_api_key(api_key) == hash_api_key(API_KEY):
                return f(*args, **kwargs)
            else:
                logger.warning(f"Invalid API Key from {request.remote_addr}")
                return jsonify({'error': 'Invalid API key'}), 401

        elif auth_header and auth_header.startswith('Bearer '):
            token = auth_header.split(' ')[1]
            try:
                payload = jwt.decode(token, SECRET_KEY, algorithms=['HS256'])
                request.user_id = payload.get('user_id')
                return f(*args, **kwargs)
            except jwt.ExpiredSignatureError:
                return jsonify({'error': 'Token expired'}), 401
            except jwt.InvalidTokenError:
                return jsonify({'error': 'Invalid token'}), 401

        logger.warning(f"Unauthorized access attempt to {request.path} from {request.remote_addr}")
        return jsonify({'error': 'Authentication required'}), 401

    return decorated

def requires_admin(f):
    """Decorator for admin-only endpoints"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not ALLOW_MODEL_RELOAD:
            return jsonify({'error': 'Model reload is disabled in production'}), 403
            
        admin_key = request.headers.get('X-Admin-Key')
        
        if not admin_key:
            return jsonify({'error': 'Admin authentication required'}), 403
        
        if hash_api_key(admin_key) == hash_api_key(ADMIN_API_KEY):
            return f(*args, **kwargs)
        else:
            logger.warning(f"Invalid admin key attempt from {request.remote_addr}")
            return jsonify({'error': 'Invalid admin key'}), 403
    
    return decorated

# --- INPUT VALIDATION ---
def validate_base64_string(s):
    """Validate base64 string format and size"""
    if not isinstance(s, str):
        return False
    
    if ',' in s:
        s = s.split(',', 1)[1]
    
    MAX_BASE64_LENGTH = 10 * 1024 * 1024
    if len(s) > MAX_BASE64_LENGTH:
        return False
    
    try:
        base64.b64decode(s, validate=True)
        return True
    except Exception:
        return False

def validate_frames_input(frames_b64):
    """Comprehensive input validation for frames"""
    if not isinstance(frames_b64, list):
        return False, "frames must be a list"
    
    if len(frames_b64) == 0:
        return False, "frames list cannot be empty"
    
    if len(frames_b64) > MAX_FRAMES_INPUT:
        return False, f"Too many frames (max {MAX_FRAMES_INPUT} allowed)"
    
    sample_size = min(5, len(frames_b64))
    for i in range(sample_size):
        if not validate_base64_string(frames_b64[i]):
            return False, f"Invalid base64 data at frame {i}"
    
    return True, None

# --- CONFIGURATION ---
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MIN_FACE_SIZE = 40
MAX_FACES = 300
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

MAX_FRAMES_TO_PROCESS = 60
DETECTION_STRIDE = 3
MAX_WORKERS = min(4, multiprocessing.cpu_count())

# Model directory
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

possible_model_dirs = [
    os.path.join(SCRIPT_DIR, 'models'),
    'verifeed-backend/models',
    'models'
]

MODELS_DIR = os.environ.get('MODELS_DIR', None)
if MODELS_DIR is None:
    for path in possible_model_dirs:
        if os.path.exists(path):
            MODELS_DIR = path
            break

if MODELS_DIR is None:
    MODELS_DIR = os.path.join(SCRIPT_DIR, 'models')
    logger.warning(f"Models directory not found, will use: {MODELS_DIR}")

MODEL_FILENAME = 'model_acc_91.43_epoch12_20251110_204221.pt'

# --- MODEL ARCHITECTURE ---
class ImprovedDeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=2,
                 hidden_dim=2048, bidirectional=True, dropout=0.5):
        super(ImprovedDeepfakeDetectionModel, self).__init__()
        
        model = models.resnext50_32x4d(weights='IMAGENET1K_V2') 
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        self.lstm = nn.LSTM(
            latent_dim, 
            hidden_dim, 
            lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=False 
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        x = x.permute(1, 0, 2)
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[-1] 
        out = self.classifier(x_lstm)
        return out

# --- OPTIMIZED TRANSFORMS ---
val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE), interpolation=transforms.InterpolationMode.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# --- MODEL LOADING ---
inference_model = None
model_info = {'loaded': False, 'path': None, 'error': None}
model_lock = threading.Lock()

def load_model(model_path=None):
    """
    FIXED: Load model with PyTorch 2.6+ compatibility
    """
    global inference_model, model_info
    
    with model_lock:
        if model_path is None:
            model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
        
        # SECURITY: Prevent path traversal
        if '..' in str(model_path) or not str(model_path).startswith(MODELS_DIR):
            model_info['error'] = "Invalid model path (path traversal detected)"
            logger.error(model_info['error'])
            return False
        
        try:
            logger.info(f"Attempting to load model from: {model_path}")
            
            if not os.path.exists(model_path):
                model_info['error'] = f"Model file '{MODEL_FILENAME}' not found at {model_path}"
                logger.error(model_info['error'])
                return False
            
            # Initialize model architecture FIRST
            inference_model = ImprovedDeepfakeDetectionModel(
                num_classes=2,
                lstm_layers=2,
                bidirectional=True,
                dropout=0.5
            ).to(DEVICE)
            
            # ✅ CRITICAL FIX: PyTorch 2.6+ requires weights_only=False for custom models
            try:
                state_dict = torch.load(
                    model_path, 
                    map_location=DEVICE,
                    weights_only=False  # Required for custom nn.Module
                )
            except TypeError:
                # Fallback for older PyTorch versions
                state_dict = torch.load(model_path, map_location=DEVICE)
            
            inference_model.load_state_dict(state_dict)
            inference_model.eval()
            
            model_info['loaded'] = True
            model_info['path'] = model_path
            model_info['error'] = None
            
            logger.info(f"✓ Model loaded successfully")
            logger.info(f"✓ Device: {DEVICE}")
            logger.info(f"✓ Parameters: {sum(p.numel() for p in inference_model.parameters()):,}")
            
            return True
            
        except Exception as e:
            model_info['error'] = str(e)
            logger.error(f"Failed to load model: {e}")
            logger.error(traceback.format_exc())
            return False

# Load model on startup
load_model()

# --- OPTIMIZED HELPER FUNCTIONS ---

@lru_cache(maxsize=128)
def get_detection_model():
    """Cache detection model type"""
    return "cnn" if DEVICE.type == "cuda" else "hog"

def decode_base64_frame(b64_frame):
    """Optimized frame decoding with error handling"""
    try:
        if ',' in b64_frame:
            b64_frame = b64_frame.split(',', 1)[1]
        
        image_data = base64.b64decode(b64_frame)
        
        if len(image_data) > 20 * 1024 * 1024:
            logger.warning("Decoded frame exceeds size limit")
            return None
            
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        return np.array(image)
    except Exception as e:
        logger.debug(f"Frame decode error: {e}")
        return None

def smart_frame_sampling(total_frames, target_frames=MAX_FRAMES_TO_PROCESS):
    """Sample frames intelligently across video duration"""
    if total_frames <= target_frames:
        return list(range(total_frames))
    
    indices = np.linspace(0, total_frames - 1, target_frames, dtype=int)
    return indices.tolist()

def batch_decode_frames(frames_b64):
    """
    IMPROVED: Parallel frame decoding for better performance
    """
    if len(frames_b64) > MAX_FRAMES_TO_PROCESS:
        sample_indices = smart_frame_sampling(len(frames_b64))
        frames_b64 = [frames_b64[i] for i in sample_indices]
    
    frames = [None] * len(frames_b64)
    failed_count = 0
    
    # Use parallel processing for large batches
    if len(frames_b64) > 10:
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
            future_to_index = {
                executor.submit(decode_base64_frame, b64): idx 
                for idx, b64 in enumerate(frames_b64)
            }
            
            for future in as_completed(future_to_index):
                idx = future_to_index[future]
                try:
                    frame = future.result(timeout=5)
                    if frame is not None:
                        frames[idx] = frame
                    else:
                        failed_count += 1
                except Exception as e:
                    logger.debug(f"Frame {idx} decode failed: {e}")
                    failed_count += 1
        
        # Filter out None values
        frames = [f for f in frames if f is not None]
    else:
        # Sequential for small batches
        frames = []
        for b64_frame in frames_b64:
            frame = decode_base64_frame(b64_frame)
            if frame is not None:
                frames.append(frame)
            else:
                failed_count += 1
    
    return frames, failed_count

def detect_faces_optimized(frames, max_faces=MAX_FACES):
    """Optimized face detection with intelligent stride and caching"""
    face_frames = []
    detection_model = get_detection_model()
    
    detection_indices = list(range(0, len(frames), DETECTION_STRIDE))
    last_valid_face = None
    
    for idx in detection_indices:
        if len(face_frames) >= max_faces:
            break
        
        frame = frames[idx]
        if frame is None or frame.size == 0:
            continue
        
        try:
            h, w = frame.shape[:2]
            
            max_dim = max(h, w)
            if max_dim > 640:
                scale = 640 / max_dim
                small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)), 
                                        interpolation=cv2.INTER_LINEAR)
                scale_back = max_dim / 640
            else:
                small_frame = frame
                scale_back = 1.0
            
            face_locations = face_recognition.face_locations(
                small_frame, 
                model=detection_model, 
                number_of_times_to_upsample=0
            )
            
            if len(face_locations) > 0:
                best_face_loc = max(face_locations, 
                                   key=lambda loc: (loc[2] - loc[0]) * (loc[1] - loc[3]))
                
                top, right, bottom, left = best_face_loc
                
                if scale_back != 1.0:
                    top = int(top * scale_back)
                    right = int(right * scale_back)
                    bottom = int(bottom * scale_back)
                    left = int(left * scale_back)
                
                face_img = frame[top:bottom, left:right]
                
                if face_img.size > 0 and \
                   face_img.shape[0] >= MIN_FACE_SIZE and \
                   face_img.shape[1] >= MIN_FACE_SIZE:
                    face_frames.append(face_img)
                    last_valid_face = face_img
            elif last_valid_face is not None:
                face_frames.append(last_valid_face)
                
        except Exception as e:
            logger.debug(f"Face detection error: {e}")
            if last_valid_face is not None:
                face_frames.append(last_valid_face)
            continue
    
    if len(face_frames) >= SEQUENCE_LENGTH:
        indices = np.linspace(0, len(face_frames) - 1, SEQUENCE_LENGTH, dtype=int)
        return [face_frames[i] for i in indices]
    elif len(face_frames) > 0:
        while len(face_frames) < SEQUENCE_LENGTH:
            face_frames.append(face_frames[-1])
        return face_frames[:SEQUENCE_LENGTH]
    
    return None

def process_prediction(frames_b64):
    """Optimized prediction logic with comprehensive error handling"""
    try:
        if not model_info['loaded']:
            return ({
                'error': 'Model not loaded',
                'details': model_info.get('error', 'Model initialization failed')
            }, 503)
            
        is_valid, error_msg = validate_frames_input(frames_b64)
        if not is_valid:
            return {'error': error_msg}, 400

        logger.info(f"Received {len(frames_b64)} frames for prediction from {request.remote_addr}")
        
        frames, failed_count = batch_decode_frames(frames_b64)
        
        total_input_frames = len(frames_b64)
        if failed_count > total_input_frames * 0.5:
            logger.warning(f"Poor connection: {failed_count}/{total_input_frames} frames failed")
            return {
                'error': 'VeriFeed could not effectively detect the video due to a low or unstable internet connection. Please check your connection and try again.'
            }, 400
        
        if len(frames) < SEQUENCE_LENGTH:
            return {
                'error': f'Not enough valid frames (minimum {SEQUENCE_LENGTH} required, got {len(frames)})'
            }, 400

        face_frames = detect_faces_optimized(frames, max_faces=MAX_FACES)
        
        if face_frames is None:
            return {'error': 'No faces detected in video'}, 400

        transformed_frames = [val_transforms(frame) for frame in face_frames]
        sequence = torch.stack(transformed_frames).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            if DEVICE.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = inference_model(sequence)
            else:
                outputs = inference_model(sequence)
            
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            
            fake_confidence = probabilities[0][0].item() * 100
            real_confidence = probabilities[0][1].item() * 100
        
        prediction = "REAL" if predicted_class == 1 else "FAKE"
        confidence = max(fake_confidence, real_confidence)

        return {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'fake_probability': round(fake_confidence, 2),
            'real_probability': round(real_confidence, 2),
            'faces_analyzed': len(face_frames),
            'frames_processed': len(frames),
            'frames_sampled': len(frames_b64) > MAX_FRAMES_TO_PROCESS
        }, 200
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        logger.error(traceback.format_exc())
        
        error_msg = str(e) if app.config['DEBUG'] else "Internal server error during prediction"
        return {'error': error_msg}, 500

# --- API ENDPOINTS ---
@app.route('/health', methods=['GET'])
def health_check():
    """Public health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'model_loaded': model_info['loaded'],
        'model_error': model_info.get('error') if not model_info['loaded'] else None,
        'authenticated': False,
        'sequence_length': SEQUENCE_LENGTH,
        'production_mode': not app.config['DEBUG'],
        'rate_limiting': RATE_LIMIT_ENABLED
    })

@app.route('/auth/token', methods=['POST'])
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE} per minute")
def generate_token():
    """Generate JWT token for authentication"""
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Request body required'}), 400
            
        api_key = data.get('api_key')
        
        if not api_key or hash_api_key(api_key) != hash_api_key(API_KEY):
            logger.warning(f"Failed token generation from {request.remote_addr}")
            return jsonify({'error': 'Invalid API key'}), 401
        
        payload = {
            'user_id': 'extension_user',
            'exp': datetime.utcnow() + timedelta(hours=24),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(payload, SECRET_KEY, algorithm='HS256')
        
        logger.info(f"JWT token generated for {request.remote_addr}")
        
        return jsonify({
            'token': token,
            'expires_in': 86400,
            'type': 'Bearer'
        })
        
    except Exception as e:
        logger.error(f"Token generation error: {e}")
        return jsonify({'error': 'Token generation failed'}), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
@requires_auth
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE} per minute")
def predict():
    """Main prediction endpoint - requires authentication"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    start_time = time.time()
    
    try:
        data = request.json
        if not data:
            return jsonify({'error': 'Request body required'}), 400
            
        frames_b64 = data.get('frames', [])
        
        result, status_code = process_prediction(frames_b64)
        
        if status_code == 200:
            result['processing_time'] = round(time.time() - start_time, 2)
            logger.info(f"Request completed in {result['processing_time']}s")
        
        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/frame_analyze', methods=['POST', 'OPTIONS'])
@requires_auth
@limiter.limit(f"{RATE_LIMIT_PER_MINUTE} per minute")
def frame_analyze():
    """Alternative endpoint name - requires authentication"""
    if request.method == 'OPTIONS':
        return jsonify({'status': 'ok'}), 200
    
    return predict()

@app.route('/model/info', methods=['GET'])
@requires_auth
def model_info_endpoint():
    """Get model information - requires authentication"""
    info = {
        'loaded': model_info['loaded'],
        'device': str(DEVICE),
        'model_filename': MODEL_FILENAME
    }
    
    if app.config['DEBUG']:
        info['path'] = model_info['path']
        info['error'] = model_info['error']
    
    return jsonify(info)

@app.route('/model/reload', methods=['POST'])
@requires_admin
def reload_model():
    """Reload the model - requires ADMIN authentication"""
    try:
        data = request.json or {}
        model_path = data.get('model_path', None)
        
        logger.info(f"Admin initiating model reload from {request.remote_addr}")
        success = load_model(model_path)
        
        if success:
            return jsonify({
                'success': True,
                'message': 'Model reloaded successfully',
                'model_path': model_info['path'] if app.config['DEBUG'] else None
            })
        else:
            error_msg = model_info['error'] if app.config['DEBUG'] else 'Model reload failed'
            return jsonify({
                'success': False,
                'error': error_msg
            }), 500
            
    except Exception as e:
        logger.error(f"Model reload error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Model reload failed'}), 500

# --- ERROR HANDLERS ---
@app.errorhandler(413)
def request_entity_too_large(error):
    """Handle requests that exceed MAX_CONTENT_LENGTH"""
    return jsonify({
        'error': 'Request too large',
        'max_size_mb': app.config['MAX_CONTENT_LENGTH'] / (1024 * 1024)
    }), 413

@app.errorhandler(429)
def ratelimit_handler(e):
    """Handle rate limit exceeded"""
    return jsonify({
        'error': 'Rate limit exceeded',
        'message': str(e.description)
    }), 429

@app.errorhandler(500)
def internal_error(error):
    """Handle internal server errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({
        'error': 'Internal server error',
        'message': 'Please try again later'
    }), 500

@app.after_request
def after_request(response):
    """
    FIXED: Security headers without overriding CORS
    """
    # Don't override CORS - Flask-CORS handles it correctly
    
    # Add security headers
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('X-Frame-Options', 'DENY')
    response.headers.add('X-XSS-Protection', '1; mode=block')
    response.headers.add('Strict-Transport-Security', 'max-age=31536000; includeSubDomains')
    
    # Remove server header to avoid fingerprinting
    response.headers.pop('Server', None)
    
    return response

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 VERIFEED PREDICTION SERVER - FIXED & SECURED")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Model Loaded: {model_info['loaded']}")
    if model_info['loaded']:
        print(f"✓ Model ready for inference")
    else:
        print(f"✗ Model Error: {model_info['error']}")
    print("\n🔐 Security Features:")
    print("  ✓ API Key Authentication")
    print("  ✓ JWT Token Support")
    print("  ✓ CORS Properly Configured (No Wildcard)")
    print("  ✓ Admin-Only Model Reload" + (" [DISABLED]" if not ALLOW_MODEL_RELOAD else ""))
    print("  ✓ Secure Key Management (No Defaults in Prod)")
    print("  ✓ Path Traversal Prevention")
    print("  ✓ Input Validation & Sanitization")
    print(f"  ✓ Rate Limiting: {RATE_LIMIT_ENABLED}")
    print(f"  ✓ Max Content Size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print(f"  ✓ Max Frames Input: {MAX_FRAMES_INPUT}")
    print(f"  ✓ Production Mode: {not app.config['DEBUG']}")
    print("\n⚡ Performance Optimizations:")
    print(f"  ✓ Max Frames to Process: {MAX_FRAMES_TO_PROCESS}")
    print(f"  ✓ Detection Stride: {DETECTION_STRIDE}")
    print(f"  ✓ Parallel Workers: {MAX_WORKERS}")
    print(f"  ✓ Face Detection Model: {get_detection_model()}")
    print("  ✓ Smart Frame Sampling")
    print("  ✓ Parallel Frame Decoding")
    print("  ✓ Mixed Precision Inference (CUDA)")
    print("\n📡 Available Endpoints:")
    print("  - GET  /health (Public)")
    print("  - POST /auth/token (API Key → JWT)")
    print("  - POST /predict (Authenticated + Rate Limited)")
    print("  - POST /frame_analyze (Authenticated + Rate Limited)")
    print("  - GET  /model/info (Authenticated)")
    print("  - POST /model/reload (Admin Only" + (" - DISABLED)" if not ALLOW_MODEL_RELOAD else ")"))
    
    if RATE_LIMIT_ENABLED:
        print("\n🚦 Rate Limits:")
        print(f"  - Per Minute: {RATE_LIMIT_PER_MINUTE}")
        print(f"  - Per Hour: {RATE_LIMIT_PER_HOUR}")
        print(f"  - Per Day: {RATE_LIMIT_PER_DAY}")
    
    print("\n✅ FIXES APPLIED:")
    print("  ✓ PyTorch 2.6 model loading (weights_only=False)")
    print("  ✓ CORS wildcard removed")
    print("  ✓ API key logging removed")
    print("  ✓ Parallel frame decoding")
    print("  ✓ Better error messages")
    
    print("\n⚠️  Using Waitress for production deployment")
    print("="*70 + "\n")
    
    # Use Waitress for production serving
    try:
        from waitress import serve
        print("🚀 Starting Waitress WSGI Server...")
        port = int(os.environ.get("PORT", 5000))
        print(f"📍 Listening on http://0.0.0.0:{port}")
        print("✓ Press CTRL+C to stop\n")
        serve(
            app,
            host='0.0.0.0',
            port=port,
            threads=4,
            channel_timeout=REQUEST_TIMEOUT,
            cleanup_interval=30,
            connection_limit=1000,
            ident=None
        )
    except ImportError:
        print("⚠️  Waitress not installed. Install with: pip install waitress")
        print("⚠️  Falling back to Flask development server (NOT FOR PRODUCTION)")
        port = int(os.environ.get("PORT", 5000))
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)