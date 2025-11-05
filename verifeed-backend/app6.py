"""
FIXED FLASK SERVER - Model Selection Based on Face Count
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

DETECTED_FACES_DIR = 'detected_faces'
os.makedirs(DETECTED_FACES_DIR, exist_ok=True)

SAVE_FACES = False
SUPPORTED_PLATFORM = "facebook"

SEQUENCE_LENGTH = 20
MIN_FRAMES_REQUIRED = 10
MAX_FRAMES_TO_PROCESS = 150

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
        return fmap, out

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

def process_frames(base64_frames, sequence_length=20):
    frames = []
    faces_detected = 0
    total_frames = len(base64_frames)
    
    frames_to_check = min(total_frames, MAX_FRAMES_TO_PROCESS)
    
    logger.info(f"Processing {frames_to_check} frames to collect {sequence_length} frames")
    
    detection_model = "cnn" if DEVICE.type == "cuda" else "hog"
    
    for i in range(frames_to_check):
        if len(frames) >= sequence_length:
            logger.info(f"✓ Collected {sequence_length} frames, stopping at frame {i}")
            break
            
        frame = decode_base64_frame(base64_frames[i])
        if frame is None:
            continue
        
        frame_to_use = frame
        
        try:
            h, w = frame.shape[:2]
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                scale_back = max(h, w) / 800
            else:
                small_frame = frame
                scale_back = 1.0
            
            face_locations = face_recognition.face_locations(
                small_frame,
                model=detection_model,
                number_of_times_to_upsample=0
            )
            
            if len(face_locations) > 0:
                top, right, bottom, left = face_locations[0]
                
                if scale_back != 1.0:
                    top = int(top * scale_back)
                    right = int(right * scale_back)
                    bottom = int(bottom * scale_back)
                    left = int(left * scale_back)
                
                face_img = frame[top:bottom, left:right, :]
                
                if face_img.size > 0 and face_img.shape[0] >= MIN_FACE_SIZE and face_img.shape[1] >= MIN_FACE_SIZE:
                    frame_to_use = face_img
                    faces_detected += 1
                    
                    if SAVE_FACES:
                        try:
                            face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                            filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_frame{i:04d}.jpg"
                            cv2.imwrite(os.path.join(DETECTED_FACES_DIR, filename), face_bgr)
                        except Exception as save_err:
                            logger.debug(f"Could not save face: {save_err}")
                            
        except Exception as detect_err:
            logger.debug(f"Face detection failed for frame {i}: {detect_err}")
        
        try:
            frame_tensor = train_transforms(frame_to_use)
            frames.append(frame_tensor)
            
            if len(frames) % 10 == 0:
                logger.info(f"  Collected {len(frames)}/{sequence_length} frames")
                    
        except Exception as transform_err:
            logger.debug(f"Transform failed for frame {i}: {transform_err}")
            continue

    if len(frames) == 0:
        raise ValueError("Could not process any frames. Video may be corrupted.")
    
    if len(frames) < sequence_length:
        if len(frames) < MIN_FRAMES_REQUIRED:
            raise ValueError(
                f"Insufficient frames: got {len(frames)}, need at least {MIN_FRAMES_REQUIRED}"
            )
        
        logger.warning(f"Only got {len(frames)} frames, padding to {sequence_length}")
        last_frame = frames[-1]
        while len(frames) < sequence_length:
            frames.append(last_frame)
    
    logger.info(
        f"✓ Final: {len(frames)} frames ({faces_detected} with faces, "
        f"{len(frames) - faces_detected} full frames)"
    )
    
    frames_tensor = torch.stack(frames[:sequence_length])
    return frames_tensor.unsqueeze(0), faces_detected

def predict_with_visualization(model, img_tensor, out_dir=DETECTED_FACES_DIR):
    img = img_tensor.to(DEVICE)
    model.eval()
    
    with torch.no_grad():
        fmap, logits = model(img)
        probs = sm(logits)
        _, prediction = torch.max(probs, 1)
        pred_idx = int(prediction.item())
        confidence = probs[:, pred_idx].item() * 100
        probs_np = probs.cpu().numpy()[0]
        
        prob_sum = np.sum(probs_np)
        if not (0.99 <= prob_sum <= 1.01):
            logger.warning(f"⚠️ Invalid probability sum: {prob_sum}")
        
        top_predictions = []
        for idx in [0, 1]:
            top_predictions.append({
                'class': CLASS_MAP[idx],
                'confidence': float(probs_np[idx]) * 100.0
            })
        
        prob_diff = abs(probs_np[1] - probs_np[0])
        prediction_strength = "HIGH" if prob_diff > 0.3 else "MEDIUM" if prob_diff > 0.15 else "LOW"
        
        visualization_path = None
        visualization_b64 = None
        
        try:
            weight_softmax = model.linear1.weight.detach().cpu().numpy()
            bz, nc, h, w = fmap.shape
            idx_for_cam = np.argmax(probs.detach().cpu().numpy())
            
            out = np.dot(
                fmap[-1].detach().cpu().numpy().reshape((nc, h * w)).T,
                weight_softmax[idx_for_cam, :].T
            )
            predict_cam = out.reshape(h, w)
            predict_cam = predict_cam - np.min(predict_cam)
            predict_img = predict_cam / np.max(predict_cam)
            predict_img = np.uint8(255 * predict_img)
            out_resized = cv2.resize(predict_img, (im_size, im_size))
            heatmap = cv2.applyColorMap(out_resized, cv2.COLORMAP_JET)
            
            inv_normalize = transforms.Normalize(
                mean=-1 * np.divide(mean, std),
                std=np.divide([1, 1, 1], std)
            )
            image = img_tensor[:, -1, :, :, :].to("cpu").clone().detach()
            image = image.squeeze()
            image = inv_normalize(image)
            image = image.numpy().transpose(1, 2, 0).clip(0, 1)
            
            result = heatmap * 0.5 + image * 0.8 * 255
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            vis_name = f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            vis_path = os.path.join(out_dir, vis_name)
            cv2.imwrite(vis_path, result)
            
            _, buffer = cv2.imencode('.png', result)
            vis_b64 = f"data:image/png;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"
            
            visualization_path = vis_path
            visualization_b64 = vis_b64
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
        
        logger.info(f"Probabilities: FAKE={probs_np[0]:.4f} ({probs_np[0]*100:.2f}%), REAL={probs_np[1]:.4f} ({probs_np[1]*100:.2f}%)")
        logger.info(f"Prediction: {CLASS_MAP[pred_idx]} with {confidence:.2f}% confidence [{prediction_strength}]")
        
        return pred_idx, confidence, probs_np.tolist(), top_predictions, visualization_path, visualization_b64, prediction_strength

def parse_model_info(filename):
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

def get_best_model(num_faces_detected):
    """
    Select best model based on number of faces detected
    - Single face (0-1): Use highest accuracy model
    - Multiple faces (2+): Use model optimized for multiple faces
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
    
    logger.info(f"Available models for {num_faces_detected} face(s) detected:")
    for info in model_info:
        logger.info(f"  - {info['name']}: acc={info['accuracy']}%, seq={info['seq_len']}")
    
    if num_faces_detected <= 1:
        best_model = max(model_info, key=lambda x: x['accuracy'])
        logger.info(f"✓ Single/No face detected → Selecting highest accuracy model")
    else:
        best_model = max(model_info, key=lambda x: (x['seq_len'], x['accuracy']))
        logger.info(f"✓ Multiple faces detected → Selecting model optimized for context")
    
    logger.info(f"✓ Selected: {best_model['name']} (acc={best_model['accuracy']}%, seq={best_model['seq_len']})")
    
    return best_model['path'], best_model['seq_len']

MODEL_CACHE = {}

def load_model_cached(model_path):
    if model_path in MODEL_CACHE:
        logger.info(f"Using cached model: {os.path.basename(model_path)}")
        return MODEL_CACHE[model_path]
    
    logger.info(f"Loading model: {os.path.basename(model_path)}")
    
    try:
        state_dict = torch.load(model_path, map_location=DEVICE, weights_only=True)
    except:
        state_dict = torch.load(model_path, map_location=DEVICE)
    
    has_lstm_bias = 'lstm.bias_ih_l0' in state_dict
    logger.info(f"Model LSTM bias detected: {has_lstm_bias}")
    
    model = Model(num_classes=2, lstm_bias=has_lstm_bias)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()
    MODEL_CACHE[model_path] = model
    logger.info(f"✓ Model loaded successfully (LSTM bias={has_lstm_bias})")
    return model

# -------------------- API ENDPOINTS --------------------
@app.route('/frame_analyze', methods=['POST'])
def analyze_frames():
    """Main endpoint for deepfake detection with face-based model selection"""
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
        
        # STEP 1: Quick face detection for model selection
        logger.info("STEP 1: Detecting faces for model selection...")
        detection_start = time.time()
        
        sample_frames = frames[:min(30, len(frames))]
        quick_face_count = 0
        detection_model = "cnn" if DEVICE.type == "cuda" else "hog"
        
        for i, b64_frame in enumerate(sample_frames):
            frame = decode_base64_frame(b64_frame)
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
                    quick_face_count = len(face_locations)
                    break
                    
            except Exception as e:
                logger.debug(f"Quick face detection error frame {i}: {e}")
                continue
        
        logger.info(f"Face detection preview: {quick_face_count} face(s) detected")
        
        # STEP 2: Select model based on face count
        logger.info("STEP 2: Selecting optimal model...")
        model_path, expected_seq_len = get_best_model(quick_face_count)
        logger.info(f"Model expects {expected_seq_len} frames")
        
        # STEP 3: Full frame processing
        logger.info("STEP 3: Processing all frames...")
        frames_tensor, faces_detected = process_frames(frames, sequence_length=expected_seq_len)
        detection_time = time.time() - detection_start
        
        logger.info(f"Total processing took {detection_time:.2f}s")
        
        frames_collected = frames_tensor.shape[1]
        
        if frames_collected < MIN_FRAMES_REQUIRED:
            return jsonify({
                'error': f'Insufficient frames. Collected {frames_collected}, need at least {MIN_FRAMES_REQUIRED}.',
                'frames_collected': frames_collected,
                'faces_detected': faces_detected
            }), 400
        
        model = load_model_cached(model_path)

        inference_start = time.time()
        pred_idx, confidence, probs, top_predictions, vis_path, vis_b64, pred_strength = \
            predict_with_visualization(model, frames_tensor)
        inference_time = time.time() - inference_start
        
        prediction_label = CLASS_MAP[pred_idx]
        total_time = time.time() - start_time
        finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        response = {
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'prediction_strength': pred_strength,
            'raw_prediction': int(pred_idx),
            'all_probabilities': {
                'FAKE': round(probs[0] * 100, 2),
                'REAL': round(probs[1] * 100, 2)
            },
            'top_predictions': top_predictions,
            'model_used': os.path.basename(model_path),
            'model_selection_basis': f"{quick_face_count} face(s) detected",
            'model_expected_sequence_length': expected_seq_len,
            'faces_detected': faces_detected,
            'frames_collected': frames_collected,
            'frames_received': len(frames),
            'processing_time_seconds': round(total_time, 2),
            'detection_time_seconds': round(detection_time, 2),
            'inference_time_seconds': round(inference_time, 2),
            'device': str(DEVICE),
            'started_at': request_time,
            'finished_at': finish_time,
            'timestamp': datetime.now().isoformat(),
            'label_mapping': CLASS_MAP,
            'visualization_path': vis_path,
            'visualization_bytes_b64': vis_b64
        }

        print("\n" + "="*70)
        print("🎯 VERIFEED DEEPFAKE DETECTION RESULTS")
        print("="*70)
        print(f"🕒 Started:           {request_time}")
        print(f"📊 Model:             {os.path.basename(model_path)}")
        print(f"🎭 Selection Basis:   {quick_face_count} face(s) detected")
        print(f"📦 Expected Seq:      {expected_seq_len} frames")
        print(f"📥 Frames Received:   {len(frames)}")
        print(f"👤 Faces Detected:    {faces_detected}/{frames_collected}")
        print(f"🎬 Frames Used:       {frames_collected}")
        print(f"🎯 Prediction:        {prediction_label} ({confidence:.2f}%) [{pred_strength}]")
        print(f"📈 Probabilities:     FAKE={probs[0]*100:.2f}% | REAL={probs[1]*100:.2f}%")
        print(f"💻 Device:            {DEVICE}")
        print(f"⏱️  Total Time:        {total_time:.2f}s")
        if vis_path:
            print(f"🖼️  Visualization:     {vis_path}")
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
        'default_sequence_length': SEQUENCE_LENGTH,
        'min_frames_required': MIN_FRAMES_REQUIRED,
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
    print(f"🎬 Sequence Length:   {SEQUENCE_LENGTH} frames")
    print(f"📊 Min Frames:        {MIN_FRAMES_REQUIRED} frames")
    print(f"🏷️  Label Mapping:     {CLASS_MAP}")

    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    print(f"\n📦 Available Models: {len(model_files)}")

    if model_files:
        print("🔍 Model Details:")
        for mf in model_files:
            acc, seq = parse_model_info(mf)
            print(f"   - {os.path.basename(mf)}: {acc}% acc, {seq} frames")

    print("\n✨ MODEL SELECTION:")
    print("   • 0-1 faces → Highest accuracy model")
    print("   • 2+ faces → Longest sequence model (better context)")
    
    print("="*70)
    print("✅ Server ready at http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='localhost', port=5000, debug=False, threaded=True)