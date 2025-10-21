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

# -------------------- CONFIGURATION --------------------
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")

DETECTED_FACES_DIR = 'detected_faces'
os.makedirs(DETECTED_FACES_DIR, exist_ok=True)

SAVE_FACES = False
SUPPORTED_PLATFORM = "facebook"
MIN_FRAMES = 15
MAX_FRAMES = 100

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

train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((im_size, im_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])

sm = nn.Softmax()

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
        logger.error(f"Error decoding base64 frame: {e}")
        return None

def process_frames(base64_frames, sequence_length):
    frames = []
    faces_detected = 0
    frames_to_process = min(len(base64_frames), sequence_length + 10)

    for i in range(frames_to_process):
        if len(frames) >= sequence_length:
            break
        frame = decode_base64_frame(base64_frames[i])
        if frame is None:
            continue

        small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
        try:
            face_locations = face_recognition.face_locations(small_frame, model="hog")
            if len(face_locations) > 0:
                top, right, bottom, left = face_locations[0]
                top, right, bottom, left = top*2, right*2, bottom*2, left*2
                top, left = max(0, top), max(0, left)
                bottom = min(frame.shape[0], bottom)
                right = min(frame.shape[1], right)
                face_img = frame[top:bottom, left:right, :]
                if face_img.size == 0:
                    face_img = frame
                else:
                    faces_detected += 1
                if SAVE_FACES:
                    try:
                        face_bgr = cv2.cvtColor(face_img, cv2.COLOR_RGB2BGR)
                        filename = f"face_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}_frame{i}.jpg"
                        cv2.imwrite(os.path.join(DETECTED_FACES_DIR, filename), face_bgr)
                    except:
                        pass
                frame = face_img
        except:
            pass

        try:
            frame_tensor = train_transforms(frame)
            frames.append(frame_tensor)
        except Exception as e:
            logger.error(f"Frame {i} transform failed: {e}")
            continue

    if len(frames) == 0:
        raise ValueError("No frames processed successfully")

    frames = torch.stack(frames)
    frames = frames[:sequence_length]
    return frames.unsqueeze(0), faces_detected

def predict(model, img):
    img = img.to(DEVICE)
    fmap, logits = model(img)
    logits = sm(logits)
    _, prediction = torch.max(logits, 1)
    confidence = logits[:, int(prediction.item())].item() * 100
    # ---- Console proof ----
    logger.info(f"Raw logits: {logits.cpu().numpy().tolist()}")
    logger.info(f"Prediction: {prediction.item()} | Confidence: {confidence:.2f}%")
    return [int(prediction.item()), confidence, logits.cpu().numpy().tolist()]

def get_accurate_model(sequence_length):
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
    if model_path in MODEL_CACHE:
        logger.info(f"Using cached model: {os.path.basename(model_path)}")
        return MODEL_CACHE[model_path]
    logger.info(f"Loading model from file: {os.path.basename(model_path)}")
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
    try:
        data = request.get_json()
        frames = data.get('frames', [])
        platform = data.get('platform', 'unknown').lower()

        if platform != SUPPORTED_PLATFORM:
            return jsonify({'error': 'Unsupported platform'}), 400
        if not frames:
            return jsonify({'error': 'No frames provided'}), 400

        sequence_length = min(len(frames), MAX_FRAMES)
        frames_tensor, faces_detected = process_frames(frames, sequence_length)
        model_path = get_accurate_model(frames_tensor.shape[1])
        model = load_model_cached(model_path)

        # ----- LOG which model is used -----
        logger.info(f"Model used for prediction: {model_path}")

        with torch.no_grad():
            prediction = predict(model, frames_tensor)

        confidence = round(prediction[1], 2)
        output = "REAL" if prediction[0] == 1 else "FAKE"
        processing_time = round(time.time() - start_time, 2)

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
            'timestamp': datetime.now().isoformat()
        }

        print("\n==== VERIFEED PREDICTION ====")
        print(f"Model Used: {model_path}")
        print(f"Prediction: {output} ({confidence:.2f}%)")
        print(f"Raw logits: {prediction[2]}")
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
        'timestamp': datetime.now().isoformat()
    }), 200

# -------------------- MAIN --------------------
if __name__ == '__main__':
    print("=" * 70)
    print("VERIFEED SERVER - MODEL PROOF MODE ENABLED")
    print("=" * 70)
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
