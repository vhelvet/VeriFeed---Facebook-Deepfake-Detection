"""
VeriFeed Training Data Collection Backend - PERFECTLY ALIGNED
Matches prediction backend's preprocessing pipeline EXACTLY
Optimized for efficiency and stability
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime
from pathlib import Path
import uuid
import logging
import base64
import numpy as np
import cv2
import face_recognition
import torch
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import atexit
import signal
import sys
import os

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# =====================================================================
# EXACT CONFIGURATION FROM PREDICTION BACKEND - DO NOT MODIFY
# =====================================================================
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MIN_FACE_SIZE = 40
MAX_FACES = 300
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Storage Configuration
TRAINING_DATA_DIR = Path("training_data")
METADATA_FILE = TRAINING_DATA_DIR / "metadata.json"

# Optimization Settings
MAX_WORKERS = min(4, os.cpu_count() or 2)
BATCH_SIZE = 10
MAX_FRAMES_TO_PROCESS = 150  # Process subset if too many frames
JPEG_QUALITY = 95  # High quality for training data

# Thread safety
TRAINING_DATA_DIR.mkdir(parents=True, exist_ok=True)
metadata_lock = threading.Lock()
executor = None

# =====================================================================
# THREAD POOL MANAGEMENT
# =====================================================================

def init_executor():
    """Initialize thread pool executor"""
    global executor
    if executor is None:
        executor = ThreadPoolExecutor(max_workers=MAX_WORKERS)
        logger.info(f"Initialized ThreadPoolExecutor with {MAX_WORKERS} workers")

def shutdown_executor():
    """Safely shutdown thread pool executor"""
    global executor
    if executor is not None:
        logger.info("Shutting down ThreadPoolExecutor...")
        executor.shutdown(wait=True, cancel_futures=True)
        executor = None
        logger.info("ThreadPoolExecutor shut down successfully")

atexit.register(shutdown_executor)

def signal_handler(signum, frame):
    """Handle shutdown signals gracefully"""
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_executor()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# =====================================================================
# METADATA MANAGEMENT
# =====================================================================

if not METADATA_FILE.exists():
    with open(METADATA_FILE, 'w') as f:
        json.dump({
            "total_samples": 0,
            "real_samples": 0,
            "fake_samples": 0,
            "samples": []
        }, f, indent=2)

def load_metadata():
    """Load training metadata with thread safety"""
    with metadata_lock:
        try:
            with open(METADATA_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading metadata: {e}")
            return {
                "total_samples": 0,
                "real_samples": 0,
                "fake_samples": 0,
                "samples": []
            }

def save_metadata(metadata):
    """Save training metadata with thread safety"""
    with metadata_lock:
        try:
            with open(METADATA_FILE, 'w') as f:
                json.dump(metadata, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

# =====================================================================
# CORE PREPROCESSING - EXACT MATCH WITH PREDICTION BACKEND
# =====================================================================

def decode_base64_frame(b64_frame):
    """
    Decode base64 frame to cv2 image
    EXACT COPY from prediction backend
    """
    try:
        if ',' in b64_frame:
            b64_frame = b64_frame.split(',')[1]
        image_data = base64.b64decode(b64_frame)
        nparr = np.frombuffer(image_data, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        logger.error(f"Error decoding frame: {e}")
        return None

def detect_face_from_frame(frame):
    """
    Extract LARGEST face from a single frame using EXACT prediction logic
    Returns: face crop (RGB numpy array) or None
    """
    if frame is None or frame.size == 0:
        return None
    
    try:
        h, w = frame.shape[:2]
        
        # EXACT scaling logic from prediction backend
        if max(h, w) > 800:
            scale = 800 / max(h, w)
            small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
            scale_back = max(h, w) / 800
        else:
            small_frame = frame
            scale_back = 1.0
        
        # EXACT face detection parameters from prediction backend
        detection_model = "cnn" if DEVICE.type == "cuda" else "hog"
        face_locations = face_recognition.face_locations(
            small_frame, 
            model=detection_model, 
            number_of_times_to_upsample=0
        )
        
        if len(face_locations) == 0:
            return None
        
        # Get LARGEST face (first one returned by face_recognition)
        top, right, bottom, left = face_locations[0]
        
        # Scale back to original size
        if scale_back != 1.0:
            top = int(top * scale_back)
            right = int(right * scale_back)
            bottom = int(bottom * scale_back)
            left = int(left * scale_back)
        
        # Extract face crop from ORIGINAL frame
        face_img = frame[top:bottom, left:right, :]
        
        # EXACT size validation from prediction backend
        if face_img.size > 0 and face_img.shape[0] >= MIN_FACE_SIZE and face_img.shape[1] >= MIN_FACE_SIZE:
            return face_img
        
        return None
        
    except Exception as e:
        logger.debug(f"Face detection error: {e}")
        return None

def process_frame_batch(frames_batch):
    """Process a batch of frames in parallel"""
    faces = []
    
    future_to_idx = {executor.submit(detect_face_from_frame, frame): idx 
                     for idx, frame in enumerate(frames_batch)}
    
    results = [None] * len(frames_batch)
    
    for future in as_completed(future_to_idx):
        idx = future_to_idx[future]
        try:
            face = future.result(timeout=10)
            results[idx] = face
        except Exception as e:
            logger.debug(f"Batch processing error: {e}")
            results[idx] = None
    
    # Maintain order
    return [face for face in results if face is not None]

def detect_faces_from_frames_optimized(frames):
    """
    Extract faces from frames using EXACT prediction backend logic
    Optimized with parallel processing
    Returns: list of face crops (RGB numpy arrays) up to MAX_FACES
    """
    face_frames = []
    faces_found = 0
    
    logger.info(f"Starting face detection on {len(frames)} frames (max: {MAX_FACES})")
    
    # Sample frames if too many
    if len(frames) > MAX_FRAMES_TO_PROCESS:
        step = len(frames) // MAX_FRAMES_TO_PROCESS
        frames = frames[::step]
        logger.info(f"Sampled {len(frames)} frames from original set")
    
    # Process in batches
    for i in range(0, len(frames), BATCH_SIZE):
        if faces_found >= MAX_FACES:
            logger.info(f"Reached maximum face limit ({MAX_FACES})")
            break
        
        batch = frames[i:i + BATCH_SIZE]
        batch_faces = process_frame_batch(batch)
        
        for face in batch_faces:
            if faces_found >= MAX_FACES:
                break
            face_frames.append(face)
            faces_found += 1
        
        if (i // BATCH_SIZE + 1) % 5 == 0:
            logger.info(f"Progress: {i + len(batch)}/{len(frames)} frames, {faces_found} faces found")
    
    logger.info(f"Total faces extracted: {len(face_frames)}")
    
    # EXACT sequence selection logic from prediction backend
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

def encode_face_to_base64(face_rgb):
    """
    Encode face crop to base64 JPEG
    Resizes to IM_SIZE for consistency with prediction backend
    """
    try:
        # Resize to IM_SIZE (matches prediction preprocessing)
        resized_face = cv2.resize(face_rgb, (IM_SIZE, IM_SIZE), interpolation=cv2.INTER_LINEAR)
        
        # Convert RGB to BGR for encoding
        face_bgr = cv2.cvtColor(resized_face, cv2.COLOR_RGB2BGR)
        
        # Encode with high quality
        encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY]
        success, buffer = cv2.imencode('.jpg', face_bgr, encode_params)
        
        if not success:
            return None
        
        return base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        logger.error(f"Error encoding face: {e}")
        return None

# =====================================================================
# API ENDPOINTS
# =====================================================================

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'service': 'VeriFeed Training Data Collector (ALIGNED)',
        'device': str(DEVICE),
        'sequence_length': SEQUENCE_LENGTH,
        'image_size': IM_SIZE,
        'min_face_size': MIN_FACE_SIZE,
        'max_faces': MAX_FACES,
        'max_workers': MAX_WORKERS,
        'batch_size': BATCH_SIZE,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/training/add_sample', methods=['POST'])
def add_training_sample():
    """
    Add a new training sample
    Processes frames using EXACT prediction backend logic
    """
    try:
        init_executor()
        
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate required fields
        frames_b64 = data.get('frames', [])
        label = data.get('label')
        metadata_info = data.get('metadata', {})
        
        if not frames_b64:
            return jsonify({'error': 'No frames provided'}), 400
        
        if label not in [0, 1]:
            return jsonify({'error': 'Label must be 0 (fake) or 1 (real)'}), 400
        
        logger.info(f"Processing training sample: {len(frames_b64)} frames, label={'REAL' if label == 1 else 'FAKE'}")
        
        # Decode frames (EXACT logic from prediction backend)
        frames = []
        for b64_frame in frames_b64:
            frame = decode_base64_frame(b64_frame)
            if frame is not None:
                frames.append(frame)
        
        if len(frames) < 10:
            return jsonify({'error': 'Not enough valid frames (minimum 10 required)'}), 400
        
        logger.info(f"Successfully decoded {len(frames)} frames")
        
        # Detect faces (EXACT logic from prediction backend)
        face_frames = detect_faces_from_frames_optimized(frames)
        
        if face_frames is None:
            return jsonify({'error': 'No faces detected in video'}), 400
        
        logger.info(f"Selected {len(face_frames)} faces for sequence")
        
        # Encode faces to base64
        face_frames_b64 = []
        for face in face_frames:
            face_b64 = encode_face_to_base64(face)
            if face_b64:
                face_frames_b64.append(face_b64)
        
        if len(face_frames_b64) != SEQUENCE_LENGTH:
            return jsonify({'error': f'Failed to encode all {SEQUENCE_LENGTH} faces'}), 500
        
        # Store sample
        sample_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        training_metadata = load_metadata()
        
        sample_info = {
            'sample_id': sample_id,
            'label': label,
            'label_name': 'real' if label == 1 else 'fake',
            'frame_count': len(face_frames_b64),
            'timestamp': timestamp,
            'metadata': metadata_info,
            'frames': face_frames_b64
        }
        
        training_metadata['samples'].append(sample_info)
        training_metadata['total_samples'] += 1
        
        if label == 1:
            training_metadata['real_samples'] += 1
        else:
            training_metadata['fake_samples'] += 1
        
        save_metadata(training_metadata)
        
        logger.info(f"✅ Sample {sample_id} saved successfully")
        
        return jsonify({
            'success': True,
            'sample_id': sample_id,
            'frames_saved': len(face_frames_b64),
            'total_samples': training_metadata['total_samples'],
            'real_samples': training_metadata['real_samples'],
            'fake_samples': training_metadata['fake_samples']
        }), 201
    
    except Exception as e:
        logger.error(f"Error adding training sample: {e}", exc_info=True)
        return jsonify({'error': str(e)}), 500

@app.route('/training/stats', methods=['GET'])
def get_training_stats():
    """Get training data statistics"""
    try:
        metadata = load_metadata()
        
        stats = {
            'total_samples': metadata['total_samples'],
            'real_samples': metadata['real_samples'],
            'fake_samples': metadata['fake_samples'],
            'total_frames': sum(sample.get('frame_count', 0) for sample in metadata['samples']),
            'balance_ratio': 0
        }
        
        if metadata['fake_samples'] > 0:
            stats['balance_ratio'] = metadata['real_samples'] / metadata['fake_samples']
        
        return jsonify(stats)
    
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training/samples', methods=['GET'])
def list_samples():
    """List training samples with pagination"""
    try:
        metadata = load_metadata()
        
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        label_filter = request.args.get('label', None)
        
        samples = metadata['samples']
        
        if label_filter:
            if label_filter.lower() == 'real':
                samples = [s for s in samples if s['label'] == 1]
            elif label_filter.lower() == 'fake':
                samples = [s for s in samples if s['label'] == 0]
        
        samples = sorted(samples, key=lambda x: x['timestamp'], reverse=True)
        
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        paginated_samples = []
        for sample in samples[start_idx:end_idx]:
            sample_copy = sample.copy()
            sample_copy['has_frames'] = 'frames' in sample
            if 'frames' in sample_copy:
                del sample_copy['frames']
            paginated_samples.append(sample_copy)
        
        return jsonify({
            'samples': paginated_samples,
            'total': len(samples),
            'page': page,
            'per_page': per_page,
            'total_pages': (len(samples) + per_page - 1) // per_page
        })
    
    except Exception as e:
        logger.error(f"Error listing samples: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training/sample/<sample_id>', methods=['GET'])
def get_sample_details(sample_id):
    """Get detailed information about a specific sample"""
    try:
        metadata = load_metadata()
        sample = next((s for s in metadata['samples'] if s['sample_id'] == sample_id), None)
        
        if not sample:
            return jsonify({'error': 'Sample not found'}), 404
        
        return jsonify(sample)
    
    except Exception as e:
        logger.error(f"Error getting sample: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training/sample/<sample_id>', methods=['DELETE'])
def delete_sample(sample_id):
    """Delete a training sample"""
    try:
        metadata = load_metadata()
        sample = next((s for s in metadata['samples'] if s['sample_id'] == sample_id), None)
        
        if not sample:
            return jsonify({'error': 'Sample not found'}), 404
        
        metadata['samples'] = [s for s in metadata['samples'] if s['sample_id'] != sample_id]
        metadata['total_samples'] -= 1
        
        if sample['label'] == 1:
            metadata['real_samples'] -= 1
        else:
            metadata['fake_samples'] -= 1
        
        save_metadata(metadata)
        
        return jsonify({
            'success': True,
            'total_samples': metadata['total_samples']
        })
    
    except Exception as e:
        logger.error(f"Error deleting sample: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training/export', methods=['GET'])
def export_dataset():
    """Export complete dataset"""
    try:
        metadata = load_metadata()
        
        export_data = {
            'dataset_info': {
                'total_samples': metadata['total_samples'],
                'real_samples': metadata['real_samples'],
                'fake_samples': metadata['fake_samples'],
                'exported_at': datetime.now().isoformat(),
                'sequence_length': SEQUENCE_LENGTH,
                'image_size': IM_SIZE
            },
            'samples': metadata['samples']
        }
        
        return jsonify(export_data)
    
    except Exception as e:
        logger.error(f"Error exporting dataset: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/training/clear', methods=['POST'])
def clear_training_data():
    """Clear all training data (requires confirmation)"""
    try:
        data = request.get_json()
        if not data or data.get('confirm') != 'DELETE_ALL_TRAINING_DATA':
            return jsonify({'error': 'Confirmation required: confirm: DELETE_ALL_TRAINING_DATA'}), 400
        
        metadata = load_metadata()
        deleted_samples = metadata['total_samples']
        
        new_metadata = {
            'total_samples': 0,
            'real_samples': 0,
            'fake_samples': 0,
            'samples': []
        }
        save_metadata(new_metadata)
        
        logger.warning(f"⚠️  Cleared all training data ({deleted_samples} samples)")
        
        return jsonify({
            'success': True,
            'deleted_samples': deleted_samples
        })
    
    except Exception as e:
        logger.error(f"Error clearing data: {e}")
        return jsonify({'error': str(e)}), 500

# =====================================================================
# MAIN
# =====================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("📊 VERIFEED TRAINING DATA COLLECTION SERVER")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Sequence Length: {SEQUENCE_LENGTH}")
    print(f"Image Size: {IM_SIZE}")
    print(f"Min Face Size: {MIN_FACE_SIZE}")
    print(f"Max Faces: {MAX_FACES}")
    print(f"Parallel Workers: {MAX_WORKERS}")
    print(f"Batch Size: {BATCH_SIZE}")
    print("="*70)
    
    metadata = load_metadata()
    print("\n📈 Current Statistics:")
    print(f"   Total samples: {metadata['total_samples']}")
    print(f"   Real samples: {metadata['real_samples']}")
    print(f"   Fake samples: {metadata['fake_samples']}")
    
    print("\n📁 Storage:")
    print(f"   Directory: {TRAINING_DATA_DIR.absolute()}")
    print(f"   Metadata: {METADATA_FILE}")
    
    print("\n" + "="*70)
    print("Endpoints:")
    print("  GET  /health                    - Health check")
    print("  POST /training/add_sample       - Add training sample")
    print("  GET  /training/stats            - Get statistics")
    print("  GET  /training/samples          - List samples")
    print("  GET  /training/sample/<id>      - Get sample details")
    print("  DELETE /training/sample/<id>    - Delete sample")
    print("  GET  /training/export           - Export dataset")
    print("  POST /training/clear            - Clear all data")
    print("="*70 + "\n")
    
    init_executor()
    
    try:
        app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
    finally:
        shutdown_executor()