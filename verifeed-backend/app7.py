"""
VERIFEED PREDICTION BACKEND
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
import logging




app = Flask(__name__)
CORS(app)




logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)




# Configuration
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MIN_FACE_SIZE = 40
MAX_FACES = 300
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')




# Model directory
MODELS_DIR = 'models'




# Model Architecture
class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1,
                 hidden_dim=2048, bidirectional=False):
        super(DeepfakeDetectionModel, self).__init__()
        model = models.resnext50_32x4d(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-2])
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers,
                           bidirectional=bidirectional)
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




# Load model
inference_model = None
model_info = {'loaded': False, 'path': None, 'error': None}
MODEL_FILENAME = 'model_acc_95.00_e8.pt'




def load_model(model_path=None):
    """Load the trained model - defaults to best_model.pt"""
    global inference_model, model_info
   
    if model_path is None:
        model_path = os.path.join(MODELS_DIR, MODEL_FILENAME)
   
    try:
        if not os.path.exists(model_path):
            model_info['error'] = f"Model file '{MODEL_FILENAME}' not found at {model_path}"
            logger.error(model_info['error'])
            return False
       
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




def decode_base64_frame(b64_frame):
    """Decode base64 frame to cv2 image"""
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




def detect_faces_from_frames(frames, max_faces=MAX_FACES):
    """Extract faces from frames (up to max_faces limit)"""
    face_frames = []
    faces_found = 0
    detection_model = "cnn" if DEVICE.type == "cuda" else "hog"
   
    logger.info(f"Starting face detection (max: {max_faces})")
   
    for idx, frame in enumerate(frames):
        if faces_found >= max_faces:
            logger.info(f"Reached maximum face limit ({max_faces})")
            break
           
        if frame is None or frame.size == 0:
            continue
           
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
                small_frame, model=detection_model, number_of_times_to_upsample=0
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
                    faces_found += 1
                    face_frames.append(face_img)
                   
                    if faces_found % 50 == 0:
                        logger.info(f"Extracted {faces_found} faces so far...")
                       
        except Exception as e:
            continue
   
    logger.info(f"Total faces extracted: {len(face_frames)}")
   
    # Select evenly distributed faces for sequence
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




@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'device': str(DEVICE),
        'model_loaded': model_info['loaded'],
        'model_path': model_info['path'],
        'model_error': model_info['error'],
        'max_faces': MAX_FACES
    })




@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint - analyze video frames for deepfakes"""
    try:
        if not model_info['loaded']:
            return jsonify({
                'error': 'Model not loaded',
                'details': model_info['error']
            }), 503
       
        data = request.json
        frames_b64 = data.get('frames', [])
       
        if not frames_b64:
            return jsonify({'error': 'No frames provided'}), 400
       
        logger.info(f"Received {len(frames_b64)} frames for prediction")
       
        # Decode frames
        frames = []
        for b64_frame in frames_b64:
            frame = decode_base64_frame(b64_frame)
            if frame is not None:
                frames.append(frame)
       
        if len(frames) < 10:
            return jsonify({'error': 'Not enough valid frames (minimum 10 required)'}), 400
       
        logger.info(f"Successfully decoded {len(frames)} frames")
       
        # Detect faces
        face_frames = detect_faces_from_frames(frames, max_faces=MAX_FACES)
       
        if face_frames is None:
            return jsonify({'error': 'No faces detected in video'}), 400
       
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
           
            # Get both class probabilities
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
        return jsonify({'error': str(e)}), 500




@app.route('/model/reload', methods=['POST'])
def reload_model():
    """Reload the model (useful after training)"""
    try:
        data = request.json or {}
        model_path = data.get('model_path', None)
       
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
        return jsonify({'error': str(e)}), 500




@app.route('/model/info', methods=['GET'])
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
        print(f"  Please ensure '{MODEL_FILENAME}' exists in '{MODELS_DIR}/' directory")
    print(f"Max faces per video: {MAX_FACES}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print("="*70)
    print("\nEndpoints:")
    print("  GET  /health          - Health check")
    print("  POST /predict         - Predict if video is real or fake")
    print("  POST /model/reload    - Reload model from disk")
    print("  GET  /model/info      - Get model information")
    print("="*70 + "\n")
   
    if not model_info['loaded']:
        print("⚠️  WARNING: Server starting without loaded model!")
        print(f"   Predictions will fail until '{MODEL_FILENAME}' is available.\n")
   
    app.run(host='0.0.0.0', port=5000, debug=True)





