from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
import torchvision.models
from torchvision import transforms
import numpy as np
import cv2
import face_recognition
import base64
import io
from PIL import Image, ImageEnhance, ImageFilter
import os
import glob
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import hashlib



logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


app = Flask(__name__)
CORS(app)


# Configuration optimized for Facebook videos
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODELS_DIR = 'models'


# Facebook-specific thresholds
FACEBOOK_COMPRESSION_THRESHOLD = 0.85
MIN_FACE_QUALITY_SCORE = 0.3


# Global resources
cv_face_cascade = None


# CRITICAL FIX: Set deterministic behavior
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def init_cv_cascade():
    global cv_face_cascade
    if cv_face_cascade is None:
        cv_face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    return cv_face_cascade


class FacebookAwareTransforms:
    """Transform pipeline optimized for Facebook video compression"""
   
    def __init__(self, im_size=IM_SIZE):
        self.im_size = im_size
       
        # Standard transforms for good quality frames
        self.standard_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((im_size, im_size)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
   
    def __call__(self, image, quality_score=1.0):
        # Apply denoising for compressed Facebook videos
        if isinstance(image, np.ndarray) and quality_score < 0.7:
            # Convert to PIL for preprocessing
            pil_img = Image.fromarray(image)
           
            # Mild denoising and sharpening for compressed videos
            if quality_score < 0.5:
                pil_img = pil_img.filter(ImageFilter.MedianFilter(size=3))
                enhancer = ImageEnhance.Sharpness(pil_img)
                pil_img = enhancer.enhance(1.2)
           
            # Convert back to numpy for standard transforms
            image = np.array(pil_img)
       
        return self.standard_transforms(image)


facebook_transforms = FacebookAwareTransforms()


def calculate_frame_quality(frame):
    """Calculate quality score for Facebook video frames"""
    if frame is None or len(frame.shape) != 3:
        return 0.0
   
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
   
    # Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(1.0, laplacian_var / 100.0)
   
    # Contrast
    contrast = gray.std()
    contrast_score = min(1.0, contrast / 50.0)
   
    # Weighted average favoring sharpness (important for compressed videos)
    quality_score = sharpness_score * 0.6 + contrast_score * 0.4
    return max(0.0, min(1.0, quality_score))


class Model(nn.Module):
    def __init__(self, num_classes, lstm_layers=1, hidden_dim=2048):
        super(Model, self).__init__()
       
        # Fix: Use torchvision.models directly
        try:
            backbone = torchvision.models.resnext50_32x4d(weights='DEFAULT')
            print("Using ResNeXt-50")
        except:
            try:
                backbone = torchvision.models.resnet50(weights='DEFAULT')  
                print("Warning: ResNeXt-50 not available, using ResNet-50 as fallback")
            except:
                backbone = torchvision.models.resnet50(pretrained=True)
                print("Using ResNet-50 (older torchvision)")
       
        self.model = nn.Sequential(*list(backbone.children())[:-2])
        self.lstm = nn.LSTM(2048, hidden_dim, lstm_layers, bidirectional=False)
        self.linear1 = nn.Linear(2048, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.LeakyReLU()
        self.dp = nn.Dropout(0.4)


    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        # CRITICAL FIX: Remove dropout during inference
        classification = self.linear1(x_lstm[:, -1, :])
        return fmap, classification


class ModelManager:
    def __init__(self):
        self.loaded_models = {}
        self.model_info = self._scan_models()
        self._lock = threading.Lock()
   
    def _scan_models(self):
        model_info = {}
        if not os.path.exists(MODELS_DIR):
            return model_info
       
        for model_path in glob.glob(os.path.join(MODELS_DIR, "*.pt")):
            filename = os.path.basename(model_path)
            try:
                parts = filename.replace('.pt', '').split('_')
                if len(parts) >= 4:
                    accuracy = float(parts[1])
                    sequence_length = int(parts[3])
                   
                    if sequence_length not in model_info:
                        model_info[sequence_length] = []
                   
                    model_info[sequence_length].append({
                        'path': model_path,
                        'filename': filename,
                        'accuracy': accuracy,
                        'sequence_length': sequence_length
                    })
            except (ValueError, IndexError):
                continue
       
        for key in model_info:
            model_info[key].sort(key=lambda x: x['accuracy'], reverse=True)
       
        logger.info(f"Found models for sequence lengths: {list(model_info.keys())}")
        return model_info
   
    def load_model(self, sequence_length):
        if sequence_length in self.loaded_models:
            return self.loaded_models[sequence_length]
       
        with self._lock:
            if sequence_length in self.loaded_models:
                return self.loaded_models[sequence_length]
           
            available_lengths = list(self.model_info.keys())
            if not available_lengths:
                raise ValueError("No models found")
           
            if sequence_length not in self.model_info:
                sequence_length = min(available_lengths, key=lambda x: abs(x - sequence_length))
           
            model_info = self.model_info[sequence_length][0]
           
            model = Model(num_classes=2)
            checkpoint = torch.load(model_info['path'], map_location=DEVICE, weights_only=True)
           
            state_dict = checkpoint
            if isinstance(checkpoint, dict):
                for key in ['model_state_dict', 'state_dict', 'model']:
                    if key in checkpoint:
                        state_dict = checkpoint[key]
                        break
           
            model.load_state_dict(state_dict, strict=False)
            model.to(DEVICE)
            model.eval()
           
            # CRITICAL FIX: Disable dropout explicitly
            for module in model.modules():
                if isinstance(module, nn.Dropout):
                    module.p = 0.0
           
            self.loaded_models[sequence_length] = {
                'model': model,
                'info': model_info
            }
           
            logger.info(f"Loaded Facebook-optimized model: {model_info['filename']}")
            return self.loaded_models[sequence_length]


model_manager = ModelManager()


def decode_base64_image(base64_string):
    try:
        if base64_string.startswith('data:image/'):
            base64_string = base64_string.split(',')[1]
       
        image_data = base64.b64decode(base64_string, validate=True)
       
        # Skip very small images (likely corrupted)
        if len(image_data) < 1000:
            return None
       
        image = Image.open(io.BytesIO(image_data))
       
        if image.mode != 'RGB':
            image = image.convert('RGB')
       
        # Apply mild denoising for small/compressed images (Facebook characteristic)
        if len(image_data) < 50000:  # Likely compressed
            image = image.filter(ImageFilter.MedianFilter(size=3))
       
        return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    except:
        return None


def extract_face_only(rgb_frame):
    """
    PRODUCTION: Robust face detection with multiple fallback strategies.
    Optimized for low-quality Facebook videos with comprehensive error handling.
    """
    try:
        cascade = init_cv_cascade()
        gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
        
        # Strategy 1: OpenCV Haar Cascade with multiple passes
        gray_eq = cv2.equalizeHist(gray)
        
        # Multiple detection passes with increasingly relaxed parameters
        detection_configs = [
            # Config 1: Standard detection
            {'scaleFactor': 1.1, 'minNeighbors': 3, 'minSize': (20, 20)},
            # Config 2: More relaxed for low quality
            {'scaleFactor': 1.05, 'minNeighbors': 2, 'minSize': (15, 15)},
            # Config 3: Very relaxed for compressed videos
            {'scaleFactor': 1.1, 'minNeighbors': 1, 'minSize': (10, 10)},
            # Config 4: Ultra relaxed (last resort)
            {'scaleFactor': 1.05, 'minNeighbors': 1, 'minSize': (8, 8)},
        ]
        
        faces = []
        for config in detection_configs:
            # Try on equalized image first
            faces = cascade.detectMultiScale(gray_eq, **config)
            if len(faces) > 0:
                break
            
            # Try on original grayscale
            faces = cascade.detectMultiScale(gray, **config)
            if len(faces) > 0:
                break
        
        if len(faces) > 0:
            # Select largest face (deterministic)
            largest_face = max(faces, key=lambda f: f[2] * f[3])
            x, y, w, h = largest_face
            
            # More generous padding for better context
            padding = max(5, int(min(w, h) * 0.15))
            
            top = max(0, y - padding)
            bottom = min(rgb_frame.shape[0], y + h + padding)
            left = max(0, x - padding)
            right = min(rgb_frame.shape[1], x + w + padding)
            
            face_region = rgb_frame[top:bottom, left:right]
            
            # Lower size threshold for compressed videos
            if face_region.shape[0] >= 10 and face_region.shape[1] >= 10:
                return face_region
        
        # Strategy 2: face_recognition HOG model (more robust for difficult cases)
        try:
            # Try with 1 upsample for better detection
            faces = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=1)
            
            # Try without upsample if first attempt fails
            if not faces:
                faces = face_recognition.face_locations(rgb_frame, model="hog", number_of_times_to_upsample=0)
            
            if faces:
                top, right, bottom, left = faces[0]
                
                # Add padding
                h, w = rgb_frame.shape[:2]
                padding = max(5, int(min(right-left, bottom-top) * 0.15))
                
                top = max(0, top - padding)
                bottom = min(h, bottom + padding)
                left = max(0, left - padding)
                right = min(w, right + padding)
                
                face_region = rgb_frame[top:bottom, left:right]
                
                if face_region.shape[0] >= 10 and face_region.shape[1] >= 10:
                    return face_region
        except Exception as e:
            logger.debug(f"face_recognition HOG failed: {e}")
        
        # Strategy 3: Try CNN model ONLY if available (requires GPU dlib)
        try:
            # Check if CNN model is available (will be fast on GPU, slow on CPU)
            faces = face_recognition.face_locations(rgb_frame, model="cnn", number_of_times_to_upsample=0)
            
            if faces:
                top, right, bottom, left = faces[0]
                
                h, w = rgb_frame.shape[:2]
                padding = max(5, int(min(right-left, bottom-top) * 0.15))
                
                top = max(0, top - padding)
                bottom = min(h, bottom + padding)
                left = max(0, left - padding)
                right = min(w, right + padding)
                
                face_region = rgb_frame[top:bottom, left:right]
                
                if face_region.shape[0] >= 10 and face_region.shape[1] >= 10:
                    return face_region
        except Exception as e:
            logger.debug(f"face_recognition CNN failed (this is normal if dlib not compiled with CUDA): {e}")
        
        # Strategy 4: Skin-tone detection (last resort for very compressed frames)
        hsv = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2HSV)
        
        # Expanded skin color ranges for different skin tones
        # Range 1: Lighter skin tones
        lower_skin1 = np.array([0, 20, 70], dtype=np.uint8)
        upper_skin1 = np.array([20, 255, 255], dtype=np.uint8)
        
        # Range 2: Darker skin tones
        lower_skin2 = np.array([0, 10, 40], dtype=np.uint8)
        upper_skin2 = np.array([25, 255, 200], dtype=np.uint8)
        
        skin_mask1 = cv2.inRange(hsv, lower_skin1, upper_skin1)
        skin_mask2 = cv2.inRange(hsv, lower_skin2, upper_skin2)
        skin_mask = cv2.bitwise_or(skin_mask1, skin_mask2)
        
        skin_ratio = np.sum(skin_mask > 0) / (rgb_frame.shape[0] * rgb_frame.shape[1])
        
        # If significant skin-tone area detected, use center crop as fallback
        if skin_ratio > 0.12:  # Lowered threshold from 15% to 12%
            logger.debug(f"Using skin-tone center crop fallback (skin ratio: {skin_ratio:.2f})")
            
            h, w = rgb_frame.shape[:2]
            
            # Find the centroid of skin pixels for better centering
            moments = cv2.moments(skin_mask)
            if moments["m00"] != 0:
                center_x = int(moments["m10"] / moments["m00"])
                center_y = int(moments["m01"] / moments["m00"])
            else:
                center_x, center_y = w // 2, h // 2
            
            # Extract region around skin centroid
            crop_size = min(h, w, max(h, w) // 2)  # Smaller crop for better face focus
            half_size = crop_size // 2
            
            top = max(0, center_y - half_size)
            bottom = min(h, center_y + half_size)
            left = max(0, center_x - half_size)
            right = min(w, center_x + half_size)
            
            face_region = rgb_frame[top:bottom, left:right]
            
            if face_region.shape[0] >= 10 and face_region.shape[1] >= 10:
                return face_region
        
        # Strategy 5: Edge detection fallback (detect high-edge areas = potential face features)
        try:
            edges = cv2.Canny(gray_eq, 50, 150)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Find largest contour
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)
                
                # Validate aspect ratio (faces are roughly square)
                aspect_ratio = w / float(h) if h > 0 else 0
                if 0.5 < aspect_ratio < 2.0 and w > 20 and h > 20:
                    padding = max(5, int(min(w, h) * 0.15))
                    
                    top = max(0, y - padding)
                    bottom = min(rgb_frame.shape[0], y + h + padding)
                    left = max(0, x - padding)
                    right = min(rgb_frame.shape[1], x + w + padding)
                    
                    face_region = rgb_frame[top:bottom, left:right]
                    
                    if face_region.shape[0] >= 10 and face_region.shape[1] >= 10:
                        logger.debug("Using edge detection fallback")
                        return face_region
        except Exception as e:
            logger.debug(f"Edge detection fallback failed: {e}")
        
        # NO FACE FOUND - Return None
        return None
        
    except Exception as e:
        logger.error(f"Error in face extraction: {e}")
        return None


def process_facebook_frame(frame_data):
    """Process frame with enhanced preprocessing for low-quality videos"""
    i, frame = frame_data
   
    frame_img = decode_base64_image(frame)
    if frame_img is None:
        return None, i, 0.0
   
    rgb_frame = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)
    
    # OPTIONAL: Enhance image before face detection for very low quality frames
    # Calculate if enhancement is needed
    gray_check = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)
    brightness = np.mean(gray_check)
    contrast = gray_check.std()
    
    # Enhance if too dark or low contrast
    enhancement_applied = False
    if brightness < 80 or contrast < 30:
        # Convert to PIL for enhancement
        pil_img = Image.fromarray(rgb_frame)
        
        # Increase brightness
        if brightness < 80:
            enhancer = ImageEnhance.Brightness(pil_img)
            pil_img = enhancer.enhance(min(1.5, 100 / brightness))  # Adaptive enhancement
            enhancement_applied = True
        
        # Increase contrast
        if contrast < 30:
            enhancer = ImageEnhance.Contrast(pil_img)
            pil_img = enhancer.enhance(1.5)
            enhancement_applied = True
        
        # Apply slight sharpening after enhancement
        if enhancement_applied:
            enhancer = ImageEnhance.Sharpness(pil_img)
            pil_img = enhancer.enhance(1.2)
        
        rgb_frame = np.array(pil_img)
    
    # Extract face with improved detection
    face_region = extract_face_only(rgb_frame)
    
    # If no face found, skip this frame
    if face_region is None:
        logger.debug(f"Frame {i}: No face detected - skipping")
        return None, i, 0.0
    
    # Calculate quality score for the face region
    quality_score = calculate_frame_quality(face_region)
   
    return face_region, i, quality_score


# Rest of the code remains the same...
# (preprocess_frames_facebook, predict_deepfake_facebook, analyze_frames, etc.)


def preprocess_frames_facebook(frames):
    """Preprocess frames - only include frames with detected faces"""
    frame_data = [(i, frame) for i, frame in enumerate(frames)]
   
    # Process frames sequentially to maintain order
    results = []
    for data in frame_data:
        result = process_facebook_frame(data)
        results.append(result)
   
    # Sort by original index
    results.sort(key=lambda x: x[1])
   
    processed_frames = []
    quality_scores = []
    faces_detected = 0
    frames_skipped = 0
   
    for face_region, original_idx, quality_score in results:
        # CRITICAL: Skip frames where no face was detected
        if face_region is None:
            frames_skipped += 1
            continue
       
        faces_detected += 1
        quality_scores.append(quality_score)
       
        try:
            frame_tensor = facebook_transforms(face_region, quality_score)
            processed_frames.append(frame_tensor)
        except Exception as e:
            logger.error(f"Error transforming face region: {e}")
            continue
   
    if not processed_frames:
        raise ValueError(f"No faces detected in any of the {len(frames)} frames. "
                        f"Please ensure the video contains visible faces.")
    
    # Log face detection statistics
    logger.info(f"Face extraction: {faces_detected} faces detected, {frames_skipped} frames skipped (no face)")
   
    avg_quality = np.mean(quality_scores) if quality_scores else 0.0
   
    return torch.stack(processed_frames).unsqueeze(0), faces_detected, avg_quality


def predict_deepfake_facebook(model, frames_tensor, avg_quality):
    """Deterministic prediction with consistent confidence calculation"""
    try:
        with torch.no_grad():
            frames_tensor = frames_tensor.to(DEVICE, non_blocking=True)
           
            # CRITICAL: Use inference mode for deterministic behavior
            with torch.inference_mode():
                fmap, logits = model(frames_tensor)
           
            probabilities = torch.softmax(logits, dim=1)
            _, prediction = torch.max(probabilities, 1)
           
            fake_prob = probabilities[0, 0].item()
            real_prob = probabilities[0, 1].item()
            predicted_class = prediction.item()
           
            # SIMPLIFIED: Use raw model confidence without random adjustments
            base_confidence = probabilities[0, predicted_class].item() * 100
           
            # CRITICAL FIX: Minimal, consistent adjustments only
            adjusted_confidence = base_confidence
           
            # Only apply quality penalty if very low quality (threshold-based, not gradual)
            if avg_quality < 0.4:
                adjusted_confidence = max(50, adjusted_confidence * 0.90)
            elif avg_quality < 0.6:
                adjusted_confidence = max(50, adjusted_confidence * 0.95)
           
            adjusted_confidence = max(50, min(99.9, adjusted_confidence))
           
            is_real = predicted_class == 1
            result = "REAL" if is_real else "FAKE"
           
            # Generate warnings
            warnings = []
            if avg_quality < 0.4:
                warnings.append("Very low video quality - results may be less reliable")
            elif avg_quality < 0.6:
                warnings.append("Low video quality detected")
           
            if adjusted_confidence < 65:
                warnings.append("Model confidence is moderate")
           
            return {
                'prediction': result,
                'confidence': adjusted_confidence,
                'base_confidence': base_confidence,
                'is_deepfake': not is_real,
                'probabilities': {'fake': fake_prob, 'real': real_prob},
                'warnings': warnings,
                'facebook_adjustments': {
                    'avg_quality': avg_quality,
                    'confidence_adjustment': adjusted_confidence - base_confidence
                }
            }
    except Exception as e:
        logger.error(f"Error in Facebook prediction: {e}")
        raise


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'device': str(DEVICE),
        'available_models': len(model_manager.model_info),
        'facebook_optimization': 'enabled',
        'deterministic_mode': 'enabled',
        'face_only_mode': 'enabled'
    })


@app.route('/frame_analyze', methods=['POST'])
def analyze_frames():
    start_time = time.time()
   
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data provided'}), 400
       
        frames = data.get('frames', [])
        platform = data.get('platform', 'facebook')
       
        if not frames:
            return jsonify({'error': 'No frames provided'}), 400
       
        # Calculate frame hash for debugging
        frame_hash = hashlib.md5(''.join(frames[:3]).encode()).hexdigest()[:8]
        logger.info(f"Analyzing {len(frames)} frames from {platform} (hash: {frame_hash})")
       
        # Validate frames
        validated_frames = []
        for frame in frames:
            if isinstance(frame, str) and len(frame) > 100:
                if not frame.startswith('data:image/'):
                    frame = f'data:image/jpeg;base64,{frame}'
                validated_frames.append(frame)
       
        if not validated_frames:
            return jsonify({'error': 'No valid frames in request'}), 400
       
        logger.info(f"Validated {len(validated_frames)} frames for face extraction")
       
        # CRITICAL FIX: Extract faces FIRST, then load model based on actual face count
        frames_tensor, faces_detected, avg_quality = preprocess_frames_facebook(validated_frames)
        
        # Check if we have enough faces
        if faces_detected == 0:
            raise ValueError(f"No faces detected in any of the {len(validated_frames)} frames. "
                           f"Please ensure the video contains visible faces.")
        
        # Calculate face detection rate
        detection_rate = (faces_detected / len(validated_frames)) * 100
        
        # Warning if detection rate is too low
        if detection_rate < 50:
            logger.warning(f"Low face detection rate: {detection_rate:.1f}% "
                         f"({faces_detected}/{len(validated_frames)} frames)")
        
        # NOW load model based on ACTUAL number of extracted faces
        logger.info(f"Loading model for {faces_detected} extracted faces (from {len(validated_frames)} submitted frames)")
        model_data = model_manager.load_model(faces_detected)
        model = model_data['model']
        model_info = model_data['info']
        
        # Verify tensor shape matches model expectation
        actual_sequence_length = frames_tensor.shape[1]
        expected_sequence_length = model_info['sequence_length']
        
        if actual_sequence_length != expected_sequence_length:
            logger.warning(f"Sequence mismatch: tensor has {actual_sequence_length} frames, "
                         f"model expects {expected_sequence_length}. Adjusting...")
            
            # Handle mismatch by padding or truncating
            if actual_sequence_length < expected_sequence_length:
                # Pad with repeated last frame
                padding_needed = expected_sequence_length - actual_sequence_length
                last_frame = frames_tensor[:, -1:, :, :, :].repeat(1, padding_needed, 1, 1, 1)
                frames_tensor = torch.cat([frames_tensor, last_frame], dim=1)
                logger.info(f"Padded {padding_needed} frames")
            else:
                # Truncate to model's expected length
                frames_tensor = frames_tensor[:, :expected_sequence_length, :, :, :]
                logger.info(f"Truncated to {expected_sequence_length} frames")
       
        # Predict using ONLY face regions
        prediction_result = predict_deepfake_facebook(model, frames_tensor, avg_quality)
        
        # Add warning if detection rate is low
        if detection_rate < 50:
            prediction_result['warnings'].append(
                f"Low face detection rate ({detection_rate:.0f}%) - only {faces_detected}/{len(validated_frames)} frames had detectable faces"
            )
       
        processing_time = time.time() - start_time
       
        # Response
        response = {
            'prediction': prediction_result['prediction'],
            'confidence': prediction_result['confidence'],
            'is_deepfake': prediction_result['is_deepfake'],
            'probabilities': prediction_result['probabilities'],
            'processing_time': round(processing_time, 2),
            'frames_processed': frames_tensor.shape[1],
            'faces_detected': faces_detected,
            'frames_submitted': len(validated_frames),
            'model_sequence_length': model_info['sequence_length'],
            'model_accuracy': model_info['accuracy'],
            'warnings': prediction_result['warnings'],
            'base_confidence': prediction_result['base_confidence'],
            'video_analysis': {
                'avg_quality': round(avg_quality, 3),
                'quality_category': 'high' if avg_quality >= 0.8 else 'medium' if avg_quality >= 0.6 else 'low',
                'face_detection_rate': round(detection_rate, 1)
            },
            'facebook_adjustments': prediction_result['facebook_adjustments'],
            'platform': platform,
            'frame_hash': frame_hash
        }
       
        logger.info(f"FB Analysis [{frame_hash}]: {response['prediction']} ({response['confidence']:.1f}%) - "
                   f"Quality: {avg_quality:.2f}, Faces: {faces_detected}/{len(validated_frames)} ({detection_rate:.1f}%), "
                   f"Model: {expected_sequence_length}-frame, Time: {processing_time:.2f}s")
       
        return jsonify(response)
       
    except ValueError as ve:
        # Handle no faces detected error
        error_message = str(ve)
        logger.warning(f"Face detection failed: {error_message}")
        
        return jsonify({
            'error': error_message,
            'details': 'No faces could be detected in the video frames',
            'timestamp': datetime.now().isoformat(),
            'platform': platform if 'platform' in locals() else 'unknown',
            'suggestion': 'Please ensure the video contains clear, visible faces. Try videos with: better lighting, frontal faces, less motion blur, or higher resolution.'
        }), 400
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Error in Facebook frame_analyze: {error_message}")
       
        return jsonify({
            'error': error_message,
            'details': 'Facebook-optimized deepfake analysis failed',
            'timestamp': datetime.now().isoformat(),
            'platform': platform if 'platform' in locals() else 'unknown'
        }), 500


if __name__ == '__main__':
    try:
        print("=" * 60)
        print("Facebook-Compatible VeriFeed Backend Starting")
        print("FACE-ONLY MODE ENABLED")
        print("=" * 60)
       
        print("Step 1: Initializing OpenCV cascade...")
        init_cv_cascade()
        print("✓ OpenCV cascade initialized")
       
        print("Step 2: Checking model availability...")
        if len(model_manager.model_info) == 0:
            print("⚠️  WARNING: No models found in models/ directory")
        else:
            print(f"✓ Found models for {len(model_manager.model_info)} sequence lengths")
            for seq_len, models in model_manager.model_info.items():
                best_model = models[0]
                print(f"   {seq_len} frames: {best_model['filename']} (acc: {best_model['accuracy']}%)")
       
        print("Step 3: Configuration...")
        print(f"✓ FACE-ONLY mode: Enabled (frames without faces are skipped)")
        print(f"✓ Torch seed: 42")
        print(f"✓ NumPy seed: 42")
        print(f"✓ CUDNN deterministic: True")
        print(f"✓ Dropout disabled in inference")
        print(f"✓ Sequential frame processing: enabled")
        print(f"✓ Device: {DEVICE}")
       
        print("\n" + "=" * 60)
        print("🚀 Starting FACE-ONLY DETECTION server...")
        print("📍 Content script endpoint: http://localhost:5000/frame_analyze")
        print("🔍 Only extracted faces will be analyzed")
        print("⚠️  Frames without detectable faces will be skipped")
        print("=" * 60)
       
        app.run(host='localhost', port=5000, debug=False, threaded=True, use_reloader=False)
       
    except Exception as e:
        print(f"❌ Server startup failed: {e}")
        import traceback
        traceback.print_exc()