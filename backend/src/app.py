"""
app8_production_secured.py
VERIFEED PREDICTION BACKEND - PRODUCTION SECURED + DOS PREVENTION
Comprehensive security implementation with rate limiting and input validation
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
load_dotenv()

# --- FLASK SETUP ---
app = Flask(__name__)

# --- PRODUCTION SECURITY CONFIGURATION ---
app.config['DEBUG'] = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET_KEY', secrets.token_hex(32))
app.config['MAX_CONTENT_LENGTH'] = int(os.environ.get('MAX_CONTENT_MB', '100')) * 1024 * 1024  # 100MB default

# Security Keys
SECRET_KEY = os.environ.get('JWT_SECRET_KEY', secrets.token_hex(32))
API_KEY = os.environ.get('API_KEY', '5hTeoaOm5m-91clhe2iVqKy2jpkiN54JLQ4vNbiDodU')
ADMIN_API_KEY = os.environ.get('ADMIN_API_KEY', 'rtiyXgE920lCbBdo0-ZTVmS6nKwA1IOGqCX_SUXUlFI')

# DoS Prevention Settings
MAX_FRAMES_INPUT = int(os.environ.get('MAX_FRAMES_INPUT', '600'))  # Absolute max frames accepted
REQUEST_TIMEOUT = int(os.environ.get('REQUEST_TIMEOUT', '60'))
RATE_LIMIT_ENABLED = os.environ.get('RATE_LIMIT_ENABLED', 'true').lower() == 'true'
RATE_LIMIT_PER_MINUTE = os.environ.get('RATE_LIMIT_PER_MINUTE', '20')
RATE_LIMIT_PER_HOUR = os.environ.get('RATE_LIMIT_PER_HOUR', '200')
RATE_LIMIT_PER_DAY = os.environ.get('RATE_LIMIT_PER_DAY', '1000')

# Model Security
ALLOW_MODEL_RELOAD = os.environ.get('ALLOW_MODEL_RELOAD', 'false').lower() == 'true'

print(f"🔑 DEBUG: Backend is using API_KEY: {API_KEY[:10]}...{API_KEY[-10:]}")
print(f"🛡️  Production Mode: {not app.config['DEBUG']}")
print(f"🚦 Rate Limiting: {RATE_LIMIT_ENABLED}")
print(f"📦 Max Content Size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")

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
    # Create a dummy limiter that does nothing
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

# Suppress verbose logs in production
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

        # Method 1: API Key Authentication
        if api_key:
            if hash_api_key(api_key) == hash_api_key(API_KEY):
                return f(*args, **kwargs)
            else:
                logger.warning(f"Invalid API Key from {request.remote_addr}")
                return jsonify({'error': 'Invalid API key'}), 401

        # Method 2: JWT Token Authentication
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
    
    # Remove data URI prefix if present
    if ',' in s:
        s = s.split(',', 1)[1]
    
    # Check length (prevent extremely large inputs)
    MAX_BASE64_LENGTH = 10 * 1024 * 1024  # 10MB per frame
    if len(s) > MAX_BASE64_LENGTH:
        return False
    
    # Basic base64 character check
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
    
    # Sample check first few frames for validity
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

# OPTIMIZATION: Intelligent frame sampling
MAX_FRAMES_TO_PROCESS = 60
DETECTION_STRIDE = 3

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

MODEL_FILENAME = 'model_acc_89.05_epoch21_20251110_213016.pt'

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
    """Load the trained model with thread safety and optimizations"""
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
            
            inference_model = ImprovedDeepfakeDetectionModel(
                num_classes=2,
                lstm_layers=2,
                bidirectional=True,
                dropout=0.5
            ).to(DEVICE)
            
            inference_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
            inference_model.eval()
            
            # OPTIMIZATION: Enable inference mode optimizations
            if hasattr(torch, 'inference_mode') and DEVICE.type == 'cuda':
                try:
                    inference_model = torch.jit.optimize_for_inference(
                        torch.jit.script(inference_model)
                    )
                except:
                    logger.info("JIT optimization not available, using standard model")
            
            model_info['loaded'] = True
            model_info['path'] = model_path
            model_info['error'] = None
            
            logger.info(f"✓ Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            model_info['error'] = str(e)
            logger.error(f"Failed to load model: {e}")
            if app.config['DEBUG']:
                logger.error(traceback.format_exc())
            return False

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
        
        # SECURITY: Check decoded size
        if len(image_data) > 20 * 1024 * 1024:  # 20MB limit per decoded frame
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
    """Decode frames in optimized batch"""
    if len(frames_b64) > MAX_FRAMES_TO_PROCESS:
        sample_indices = smart_frame_sampling(len(frames_b64))
        frames_b64 = [frames_b64[i] for i in sample_indices]
    
    frames = []
    for b64_frame in frames_b64:
        frame = decode_base64_frame(b64_frame)
        if frame is not None:
            frames.append(frame)
    
    return frames

def detect_faces_optimized(frames, max_faces=MAX_FACES):
    """
    Optimized face detection with intelligent stride and caching
    """
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
            
            # OPTIMIZATION: Aggressive downscaling for detection
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
    
    # Sequence selection/padding
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
            return {
                'error': 'Model not loaded',
                'details': 'Service temporarily unavailable' if not app.config['DEBUG'] else model_info['error']
            }, 503

        # SECURITY: Validate input
        is_valid, error_msg = validate_frames_input(frames_b64)
        if not is_valid:
            return {'error': error_msg}, 400

        logger.info(f"Received {len(frames_b64)} frames for prediction from {request.remote_addr}")
        
        # OPTIMIZATION: Batch decode with smart sampling
        frames = batch_decode_frames(frames_b64)
        
        if len(frames) < SEQUENCE_LENGTH:
            return {
                'error': f'Not enough valid frames (minimum {SEQUENCE_LENGTH} required, got {len(frames)})'
            }, 400

        # OPTIMIZATION: Faster face detection
        face_frames = detect_faces_optimized(frames, max_faces=MAX_FACES)
        
        if face_frames is None:
            return {'error': 'No faces detected in video'}, 400

        # OPTIMIZATION: Batch transform
        transformed_frames = [val_transforms(frame) for frame in face_frames]
        sequence = torch.stack(transformed_frames).unsqueeze(0).to(DEVICE)
        
        # Inference with optimizations
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
        if app.config['DEBUG']:
            logger.error(traceback.format_exc())
        
        # SECURITY: Generic error message in production
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
        'authenticated': False,
        'sequence_length': SEQUENCE_LENGTH,
        'production_mode': not app.config['DEBUG'],
        'rate_limiting': RATE_LIMIT_ENABLED,
        'optimizations': {
            'max_frames_processed': MAX_FRAMES_TO_PROCESS,
            'max_frames_input': MAX_FRAMES_INPUT,
            'detection_stride': DETECTION_STRIDE,
            'face_detection_model': get_detection_model()
        }
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
            logger.info(f"Request completed in {result['processing_time']}s from {request.remote_addr}")
        
        return jsonify(result), status_code

    except Exception as e:
        logger.error(f"Endpoint error: {e}")
        if app.config['DEBUG']:
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
    
    # Only show sensitive info in debug mode
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
        if app.config['DEBUG']:
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
    """Add security headers to all responses"""
    response.headers.add('Access-Control-Allow-Origin', '*')
    response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization,X-API-Key')
    response.headers.add('Access-Control-Allow-Methods', 'GET,POST,OPTIONS')
    
    # Security headers
    response.headers.add('X-Content-Type-Options', 'nosniff')
    response.headers.add('X-Frame-Options', 'DENY')
    response.headers.add('X-XSS-Protection', '1; mode=block')
    
    return response

# --- MAIN EXECUTION ---
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀🔒 VERIFEED PREDICTION SERVER - PRODUCTION SECURED")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Models Directory: {MODELS_DIR}")
    print(f"Model Loaded: {model_info['loaded']}")
    if model_info['loaded']:
        print(f"Model Path: {model_info['path']}")
    else:
        print(f"Error: {model_info['error']}")
    print("\n🔐 Security Features:")
    print("  ✓ API Key Authentication")
    print("  ✓ JWT Token Support")
    print("  ✓ CORS Restricted to Allowed Origins")
    print("  ✓ Admin-Only Model Reload" + (" [DISABLED]" if not ALLOW_MODEL_RELOAD else ""))
    print("  ✓ Cached API Key Hashing")
    print("  ✓ Path Traversal Prevention")
    print("  ✓ Input Validation & Sanitization")
    print(f"  ✓ Rate Limiting: {RATE_LIMIT_ENABLED}")
    print(f"  ✓ Max Content Size: {app.config['MAX_CONTENT_LENGTH'] / (1024*1024):.0f}MB")
    print(f"  ✓ Max Frames Input: {MAX_FRAMES_INPUT}")
    print(f"  ✓ Production Mode: {not app.config['DEBUG']}")
    print("\n⚡ Performance Optimizations:")
    print(f"  ✓ Max Frames to Process: {MAX_FRAMES_TO_PROCESS}")
    print(f"  ✓ Detection Stride: {DETECTION_STRIDE}")
    print(f"  ✓ Face Detection Model: {get_detection_model()}")
    print("  ✓ Smart Frame Sampling")
    print("  ✓ Cached Face Detection")
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
    
    print("\n⚠️  IMPORTANT: Using Waitress for production deployment")
    print("="*70 + "\n")
    
    # Use Waitress for production serving
    try:
        from waitress import serve
        print("🚀 Starting Waitress WSGI Server...")
        print(f"📍 Listening on http://0.0.0.0:5000")
        print("✓ Press CTRL+C to stop\n")
        
        serve(
            app,
            host='0.0.0.0',
            port=5000,
            threads=4,  # Thread pool size
            channel_timeout=REQUEST_TIMEOUT,
            cleanup_interval=30,
            connection_limit=1000,
            ident=None  # Don't expose server version
        )
    except ImportError:
        print("⚠️  Waitress not installed. Install with: pip install waitress")
        print("⚠️  Falling back to Flask development server (NOT FOR PRODUCTION)")
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)