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
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache

# -------------------- CONFIGURATION --------------------
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

DETECTED_FACES_DIR = 'detected_faces'
os.makedirs(DETECTED_FACES_DIR, exist_ok=True)

SAVE_FACES = False
SUPPORTED_PLATFORM = "facebook"
MIN_FRAMES = 15
MAX_FRAMES = 100

# Optimized frame processing
FRAME_SKIP = 2  # Process every Nth frame for speed
FACE_DETECTION_SCALE = 0.25  # Reduced from 0.5 for faster detection
MAX_WORKERS = 4  # Parallel processing threads

BENCHMARK_DATASETS = [
    "FaceForensics++",
    "Deepfake Detection Challenge (DFDC)",
    "Celeb-DF"
]

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

# Pre-compile transforms for faster execution
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

sm = nn.Softmax()

# Thread pool for parallel frame processing
executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)

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

# -------------------- OPTIMIZED HELPERS --------------------
@lru_cache(maxsize=128)
def get_cached_transform_params():
    """Cache transform parameters to avoid recreation"""
    return (im_size, mean, std)

def decode_base64_frame(base64_string):
    """Optimized base64 decoding"""
    try:
        if ',' in base64_string:
            base64_string = base64_string.split(',', 1)[1]  # Only split once
        image_data = base64.b64decode(base64_string)
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
    except Exception as e:
        return None

def process_single_frame(args):
    """Process a single frame - designed for parallel execution"""
    frame_data, idx, save_faces = args
    
    frame = decode_base64_frame(frame_data)
    if frame is None:
        return None, False
    
    # Faster face detection with smaller resolution
    small_frame = cv2.resize(frame, (0, 0), fx=FACE_DETECTION_SCALE, fy=FACE_DETECTION_SCALE)
    face_detected = False
    
    try:
        face_locations = face_recognition.face_locations(small_frame, model="hog", number_of_times_to_upsample=0)
        if len(face_locations) > 0:
            scale = 1 / FACE_DETECTION_SCALE
            top, right, bottom, left = face_locations[0]
            top, right, bottom, left = int(top*scale), int(right*scale), int(bottom*scale), int(left*scale)
            
            # Clamp to frame boundaries
            top, left = max(0, top), max(0, left)
            bottom = min(frame.shape[0], bottom)
            right = min(frame.shape[1], right)
            
            face_img = frame[top:bottom, left:right, :]
            if face_img.size > 0:
                frame = face_img
                face_detected = True
                
                if save_faces:
                    try:
                        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                        filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_frame{idx}.jpg"
                        cv2.imwrite(os.path.join(DETECTED_FACES_DIR, filename), face_bgr)
                    except:
                        pass
    except:
        pass
    
    try:
        frame_tensor = train_transforms(frame)
        return frame_tensor, face_detected
    except Exception as e:
        return None, False

def process_frames_parallel(base64_frames, sequence_length):
    """Parallel frame processing with frame skipping"""
    frames = []
    faces_detected = 0
    
    # Smart frame selection: skip frames for speed
    total_frames = len(base64_frames)
    if total_frames > sequence_length * FRAME_SKIP:
        # Select evenly distributed frames
        indices = np.linspace(0, total_frames - 1, sequence_length * 2, dtype=int)
    else:
        indices = range(min(total_frames, sequence_length * 2))
    
    # Prepare arguments for parallel processing
    frame_args = [(base64_frames[i], i, SAVE_FACES) for i in indices]
    
    # Process frames in parallel
    results = executor.map(process_single_frame, frame_args)
    
    for frame_tensor, face_detected in results:
        if frame_tensor is not None:
            frames.append(frame_tensor)
            if face_detected:
                faces_detected += 1
        if len(frames) >= sequence_length:
            break
    
    if len(frames) == 0:
        raise ValueError("No frames processed successfully")
    
    frames = torch.stack(frames)
    frames = frames[:sequence_length]
    return frames.unsqueeze(0), faces_detected

def process_frames(base64_frames, sequence_length):
    """Original sequential processing (kept for compatibility)"""
    return process_frames_parallel(base64_frames, sequence_length)

@torch.inference_mode()  # Faster than no_grad
def predict(model, img):
    """Optimized prediction with inference mode"""
    img = img.to(DEVICE, non_blocking=True)  # Async transfer
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    
    # Log only if needed
    if logger.level <= logging.INFO:
        logger.info(f"Raw logits: {logits.cpu().numpy().tolist()}")
        logger.info(f"Prediction: {prediction.item()} | Confidence: {confidence:.2f}%")
    
    return [int(prediction.item()), confidence, logits.cpu().numpy().tolist()]

@lru_cache(maxsize=10)
def get_accurate_model(sequence_length):
    """Cached model path resolution"""
    model_name = []
    sequence_model = []
    final_model = ""
    list_models = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    if not list_models:
        raise FileNotFoundError(f"No models found in {MODELS_DIR}")
    for model_path in list_models:
        model_name.append(os.path.basename(model_path))
    for model_filename in model_name:
        try:
            seq = model_filename.split("_")[3]
            if int(seq) == sequence_length:
                sequence_model.append(model_filename)
        except:
            pass
    if len(sequence_model) >= 1:
        final_model = os.path.join(MODELS_DIR, sequence_model[0])
    else:
        final_model = os.path.join(MODELS_DIR, model_name[0])
    return final_model

MODEL_CACHE = {}
def load_model_cached(model_path):
    """Load and cache models with optimization flags"""
    if model_path in MODEL_CACHE:
        logger.info(f"Using cached model: {os.path.basename(model_path)}")
        return MODEL_CACHE[model_path]
    
    logger.info(f"Loading model from file: {os.path.basename(model_path)}")
    model = Model(num_classes=2)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE), strict=True)
    model.to(DEVICE)
    model.eval()
    
    # Set to inference mode
    for param in model.parameters():
        param.requires_grad = False
    
    MODEL_CACHE[model_path] = model
    return model

# -------------------- API --------------------
@app.route('/frame_analyze', methods=['POST'])
def analyze_frames():
    start_time = time.time()
    timing = {}
    
    try:
        # Step 1: Parse request
        t1 = time.time()
        data = request.get_json()
        frames = data.get('frames', [])
        platform = data.get('platform', 'unknown').lower()
        timing['request_parsing'] = round((time.time() - t1) * 1000, 2)

        if platform != SUPPORTED_PLATFORM:
            return jsonify({'error': 'Unsupported platform'}), 400
        if not frames:
            return jsonify({'error': 'No frames provided'}), 400

        # Step 2: Process frames
        t2 = time.time()
        sequence_length = min(len(frames), MAX_FRAMES)
        frames_tensor, faces_detected = process_frames(frames, sequence_length)
        timing['frame_processing'] = round((time.time() - t2) * 1000, 2)
        
        # Step 3: Load model
        t3 = time.time()
        model_path = get_accurate_model(frames_tensor.shape[1])
        model = load_model_cached(model_path)
        timing['model_loading'] = round((time.time() - t3) * 1000, 2)

        logger.info(f"Model used for prediction: {model_path}")

        # Step 4: Run prediction
        t4 = time.time()
        prediction = predict(model, frames_tensor)
        timing['model_inference'] = round((time.time() - t4) * 1000, 2)

        confidence = round(prediction[1], 2)
        output = "REAL" if prediction[0] == 1 else "FAKE"
        processing_time = round(time.time() - start_time, 2)
        timing['total'] = round((time.time() - start_time) * 1000, 2)

        response = {
            'model_used': os.path.basename(model_path),
            'model_path': model_path,
            'device': str(DEVICE),
            'prediction': output,
            'raw_prediction': prediction[0],
            'confidence': confidence,
            'logits': prediction[2],
            'frames_analyzed': frames_tensor.shape[1],
            'faces_detected': faces_detected,
            'processing_time': processing_time,
            'timing_breakdown_ms': timing,
            'timestamp': datetime.now().isoformat()
        }

        print("\n==== VERIFEED PREDICTION ====")
        print(f"Model Used: {model_path}")
        print(f"Prediction: {output} ({confidence:.2f}%)")
        print(f"Raw logits: {prediction[2]}")
        print(f"\n--- TIMING BREAKDOWN (ms) ---")
        print(f"Request Parsing:    {timing['request_parsing']:>8.2f} ms")
        print(f"Frame Processing:   {timing['frame_processing']:>8.2f} ms")
        print(f"Model Loading:      {timing['model_loading']:>8.2f} ms")
        print(f"Model Inference:    {timing['model_inference']:>8.2f} ms")
        print(f"TOTAL:              {timing['total']:>8.2f} ms")
        print("=============================\n")

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
        'device': str(DEVICE),
        'optimizations': {
            'parallel_processing': True,
            'frame_skip': FRAME_SKIP,
            'max_workers': MAX_WORKERS,
            'torch_compile': hasattr(torch, 'compile')
        },
        'timestamp': datetime.now().isoformat()
    }), 200

# -------------------- MAIN --------------------
if __name__ == '__main__':
    print("=" * 70)
    print("VERIFEED SERVER - OPTIMIZED MODE")
    print(f"Device: {DEVICE}")
    print(f"Parallel Workers: {MAX_WORKERS}")
    print(f"Frame Skip Factor: {FRAME_SKIP}")
    print("=" * 70)
    app.run(host='localhost', port=5000, debug=False, threaded=True)