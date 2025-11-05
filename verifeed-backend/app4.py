"""
FIXED FLASK SERVER - Model Selection Based on Face Count
All bugs fixed:
1. Function call mismatch (process_frames -> enhanced_process_frames)
2. UnboundLocalError for model variable
3. Expression analysis integration
4. dlib import handling
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
from scipy import stats

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

# Check for dlib availability
try:
    import dlib
    DLIB_AVAILABLE = True
except ImportError:
    DLIB_AVAILABLE = False
    logging.warning("dlib not available - landmark stability analysis will be disabled")

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


def analyze_expression_intensity(frames_rgb):
    """
    Analyze expression intensity across frames
    Returns metrics to help distinguish real over-reactions from fake artifacts
    """
    if len(frames_rgb) == 0:
        return {
            'mean_intensity': 0,
            'intensity_variance': 0,
            'sudden_changes': 0,
            'smoothness_score': 0
        }
    
    try:
        intensities = []
        
        # Get target size from first frame
        target_h, target_w = frames_rgb[0].shape[:2]
        
        for frame in frames_rgb:
            # Resize frame if needed for consistency
            if frame.shape[:2] != (target_h, target_w):
                frame = cv2.resize(frame, (target_w, target_h))
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            
            # Calculate frame variance (movement/expression intensity)
            intensity = np.var(gray)
            intensities.append(intensity)
        
        if len(intensities) < 2:
            return {
                'mean_intensity': float(intensities[0]) if intensities else 0,
                'intensity_variance': 0,
                'sudden_changes': 0,
                'smoothness_score': 0
            }
        
        intensities = np.array(intensities)
        
        # Calculate metrics
        mean_intensity = np.mean(intensities)
        intensity_variance = np.var(intensities)
        
        # Count sudden changes (potential over-reactions OR artifacts)
        diffs = np.abs(np.diff(intensities))
        sudden_changes = np.sum(diffs > np.percentile(diffs, 90))
        
        # Smoothness: real expressions tend to be smoother even when dramatic
        # Fakes often have jittery artifacts
        smoothness_score = 1.0 / (1.0 + np.std(np.diff(intensities)))
        
        return {
            'mean_intensity': float(mean_intensity),
            'intensity_variance': float(intensity_variance),
            'sudden_changes': int(sudden_changes),
            'smoothness_score': float(smoothness_score)
        }
        
    except Exception as e:
        logger.warning(f"Expression intensity analysis failed: {e}")
        return {
            'mean_intensity': 0,
            'intensity_variance': 0,
            'sudden_changes': 0,
            'smoothness_score': 0
        }

def detect_facial_landmarks_stability(frames_rgb):
    """
    Check if facial landmarks remain stable during expressions
    Real faces: landmarks move smoothly even with over-reactions
    Fake faces: landmarks may jump/warp during movement
    """
    if not DLIB_AVAILABLE:
        logger.debug("dlib not available, skipping landmark analysis")
        return None
        
    try:
        # Initialize dlib face detector and landmark predictor
        detector = dlib.get_frontal_face_detector()
        
        # You'll need to download this file:
        # http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        if not os.path.exists(predictor_path):
            logger.warning("Landmark predictor not found, skipping landmark analysis")
            return None
            
        predictor = dlib.shape_predictor(predictor_path)
        
        landmark_positions = []
        
        for frame in frames_rgb[:30]:  # Sample first 30 frames
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            faces = detector(gray)
            
            if len(faces) == 0:
                continue
                
            # Get landmarks for first face
            landmarks = predictor(gray, faces[0])
            points = np.array([[p.x, p.y] for p in landmarks.parts()])
            landmark_positions.append(points)
        
        if len(landmark_positions) < 2:
            return None
        
        # Calculate landmark stability
        landmark_positions = np.array(landmark_positions)
        
        # Measure jitter: difference between consecutive frames
        jitter = []
        for i in range(len(landmark_positions) - 1):
            diff = np.linalg.norm(landmark_positions[i+1] - landmark_positions[i], axis=1)
            jitter.append(np.mean(diff))
        
        jitter = np.array(jitter)
        
        # Real faces: smooth movement even during expressions
        # Fake faces: erratic/jumpy landmarks
        stability_score = 1.0 / (1.0 + np.std(jitter))
        max_jitter = np.max(jitter)
        
        return {
            'stability_score': float(stability_score),
            'max_jitter': float(max_jitter),
            'mean_jitter': float(np.mean(jitter))
        }
        
    except Exception as e:
        logger.warning(f"Landmark stability check failed: {e}")
        return None

def analyze_optical_flow_consistency(frames_rgb):
    """
    Analyze optical flow patterns during expressions
    Real over-reactions: consistent flow patterns
    Fake artifacts: inconsistent/broken flow
    """
    if len(frames_rgb) < 2:
        return None
    
    try:
        flows = []
        
        # Get target size from first frame
        target_h, target_w = frames_rgb[0].shape[:2]
        
        for i in range(min(20, len(frames_rgb) - 1)):
            # Convert to grayscale
            frame1 = cv2.cvtColor(frames_rgb[i], cv2.COLOR_RGB2GRAY)
            frame2 = cv2.cvtColor(frames_rgb[i+1], cv2.COLOR_RGB2GRAY)
            
            # CRITICAL FIX: Ensure both frames are the same size
            if frame1.shape != (target_h, target_w):
                frame1 = cv2.resize(frame1, (target_w, target_h))
            if frame2.shape != (target_h, target_w):
                frame2 = cv2.resize(frame2, (target_w, target_h))
            
            # Verify shapes match
            if frame1.shape != frame2.shape:
                logger.debug(f"Skipping flow calc: shape mismatch {frame1.shape} vs {frame2.shape}")
                continue
            
            # Calculate dense optical flow
            flow = cv2.calcOpticalFlowFarneback(
                frame1, frame2, None, 
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0
            )
            
            # Calculate flow magnitude
            magnitude = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            flows.append(magnitude)
        
        if len(flows) == 0:
            logger.warning("No valid optical flow could be calculated")
            return None
        
        flows = np.array(flows)
        
        # Metrics
        mean_flow = np.mean(flows)
        flow_variance = np.var(flows)
        
        # Check for flow consistency
        # Real: smooth flow even with dramatic movements
        # Fake: erratic flow patterns
        consistency_score = 1.0 / (1.0 + np.std([np.mean(f) for f in flows]))
        
        return {
            'mean_flow': float(mean_flow),
            'flow_variance': float(flow_variance),
            'consistency_score': float(consistency_score)
        }
        
    except Exception as e:
        logger.warning(f"Optical flow analysis failed: {e}")
        return None

def enhanced_process_frames(base64_frames, sequence_length=20):
    """
    Enhanced frame processing with expression analysis
    """
    frames = []
    frames_rgb = []  # Keep RGB versions for analysis
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
                    
        except Exception as detect_err:
            logger.debug(f"Face detection failed for frame {i}: {detect_err}")
        
        try:
            # Store RGB version for analysis
            frames_rgb.append(frame_to_use)
            
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
        last_rgb = frames_rgb[-1]
        while len(frames) < sequence_length:
            frames.append(last_frame)
            frames_rgb.append(last_rgb)
    
    # Perform expression analysis
    logger.info("🎭 Analyzing expression patterns...")
    expression_metrics = analyze_expression_intensity(frames_rgb[:sequence_length])
    flow_metrics = analyze_optical_flow_consistency(frames_rgb[:sequence_length])
    landmark_metrics = detect_facial_landmarks_stability(frames_rgb[:sequence_length])
    
    logger.info(
        f"✓ Final: {len(frames)} frames ({faces_detected} with faces, "
        f"{len(frames) - faces_detected} full frames)"
    )
    
    frames_tensor = torch.stack(frames[:sequence_length])
    
    return (frames_tensor.unsqueeze(0), faces_detected, 
            expression_metrics, flow_metrics, landmark_metrics)

def adjust_prediction_for_expressions(pred_idx, confidence, probs_np, expression_metrics, flow_metrics, landmark_metrics):
    """
    BALANCED adjustment for expression analysis
    
    Key principle: Only adjust when there's STRONG evidence of deepfake artifacts
    Avoid over-penalizing real videos with dramatic expressions
    
    Strategy:
    1. Trust the model's initial prediction more
    2. Only apply adjustments when MULTIPLE red flags align
    3. Use smaller penalties to avoid flipping predictions too easily
    """
    original_pred = CLASS_MAP[pred_idx]
    original_confidence = confidence
    adjustments = []
    
    # === Calculate risk scores (0-1, higher = more suspicious) ===
    intensity_risk = 0.0
    smoothness_risk = 0.0
    flow_risk = 0.0
    landmark_risk = 0.0
    
    if expression_metrics:
        mean_intensity = expression_metrics.get('mean_intensity', 0)
        intensity_variance = expression_metrics.get('intensity_variance', 0)
        sudden_changes = expression_metrics.get('sudden_changes', 0)
        smoothness_score = expression_metrics.get('smoothness_score', 1.0)
        
        # Intensity risk: Only flag EXTREMELY high values
        if mean_intensity > 1200:
            intensity_risk = 1.0
        elif mean_intensity > 1000:
            intensity_risk = 0.7
        elif mean_intensity > 850:
            intensity_risk = 0.4
        
        # Smoothness risk: Only flag VERY jerky movement
        if smoothness_score < 0.2:
            smoothness_risk = 1.0
        elif smoothness_score < 0.3:
            smoothness_risk = 0.6
        elif smoothness_score < 0.4:
            smoothness_risk = 0.3
        
        # Variance risk
        variance_risk = 0.0
        if intensity_variance > 20000:
            variance_risk = 0.8
        elif intensity_variance > 15000:
            variance_risk = 0.4
    
    if flow_metrics:
        consistency_score = flow_metrics.get('consistency_score', 1.0)
        mean_flow = flow_metrics.get('mean_flow', 0)
        
        # Flow risk: Only flag truly inconsistent patterns
        if consistency_score < 0.3:
            flow_risk = 1.0
        elif consistency_score < 0.4:
            flow_risk = 0.6
        elif consistency_score < 0.5:
            flow_risk = 0.3
    
    if landmark_metrics:
        max_jitter = landmark_metrics.get('max_jitter', 0)
        stability_score = landmark_metrics.get('stability_score', 1.0)
        
        # Landmark risk
        if max_jitter > 15 or stability_score < 0.4:
            landmark_risk = 0.8
        elif max_jitter > 12 or stability_score < 0.5:
            landmark_risk = 0.5
    
    # === COMBINED RISK ASSESSMENT (most important) ===
    # Calculate overall suspicion score
    risk_scores = []
    if expression_metrics:
        risk_scores.extend([intensity_risk, smoothness_risk, variance_risk])
    if flow_metrics:
        risk_scores.append(flow_risk)
    if landmark_metrics:
        risk_scores.append(landmark_risk)
    
    if len(risk_scores) == 0:
        overall_risk = 0.0
    else:
        # Need MULTIPLE high risks to be truly suspicious
        high_risks = sum(1 for r in risk_scores if r >= 0.6)
        medium_risks = sum(1 for r in risk_scores if 0.3 <= r < 0.6)
        
        # Calculate weighted risk
        overall_risk = (
            max(risk_scores) * 0.4 +  # Worst single indicator
            np.mean(risk_scores) * 0.6  # Average risk
        )
    
    # === APPLY ADJUSTMENTS (only when justified) ===
    
    # Rule 1: CRITICAL - Multiple strong red flags
    if high_risks >= 2 and overall_risk > 0.7:
        if pred_idx == 1:  # Model said REAL
            penalty = min(25, 15 + (high_risks * 5))
            adjustments.append({
                'reason': f'🚨 Multiple deepfake signatures detected ({high_risks} strong indicators)',
                'confidence_adjustment': -penalty
            })
            logger.warning(f"🚨 CRITICAL: {high_risks} strong deepfake indicators detected")
    
    # Rule 2: Exaggerated expression with unnatural motion
    elif intensity_risk > 0.6 and smoothness_risk > 0.5:
        if pred_idx == 1:  # Model said REAL
            adjustments.append({
                'reason': 'Exaggerated expressions with jerky motion (possible deepfake)',
                'confidence_adjustment': -15
            })
    
    # Rule 3: High intensity with inconsistent flow
    elif intensity_risk > 0.6 and flow_risk > 0.5:
        if pred_idx == 1:  # Model said REAL
            adjustments.append({
                'reason': 'Dramatic expression with inconsistent motion pattern',
                'confidence_adjustment': -12
            })
    
    # Rule 4: Very poor smoothness alone (but be conservative)
    elif smoothness_risk >= 1.0:  # Only the most extreme cases
        if pred_idx == 1:  # Model said REAL
            adjustments.append({
                'reason': 'Severely jerky movement detected',
                'confidence_adjustment': -10
            })
    
    # Rule 5: Inconsistent flow with other indicators
    elif flow_risk > 0.8 and (intensity_risk > 0.3 or smoothness_risk > 0.3):
        if pred_idx == 1:  # Model said REAL
            adjustments.append({
                'reason': 'Inconsistent optical flow with expression anomalies',
                'confidence_adjustment': -10
            })
    
    # Rule 6: Unstable landmarks (only when very unstable)
    elif landmark_risk > 0.7:
        if pred_idx == 1:  # Model said REAL
            adjustments.append({
                'reason': 'Unstable facial landmarks detected',
                'confidence_adjustment': -8
            })
    
    # === BONUS: Correct false FAKE predictions ===
    # If model said FAKE but movement is smooth and consistent, reduce confidence
    if pred_idx == 0:  # Model said FAKE
        if smoothness_risk < 0.2 and flow_risk < 0.3 and intensity_risk < 0.5:
            adjustments.append({
                'reason': 'Smooth and consistent expressions (likely real)',
                'confidence_adjustment': -8
            })
    
    # Apply adjustments
    adjusted_confidence = confidence
    total_adjustment = 0
    
    for adj in adjustments:
        adjusted_confidence += adj['confidence_adjustment']
        total_adjustment += adj['confidence_adjustment']
        logger.info(f"  📊 {adj['reason']}: {adj['confidence_adjustment']:+.1f}%")
    
    # IMPORTANT: Don't let adjustments go too far
    # Keep confidence in reasonable range
    adjusted_confidence = max(50, min(100, adjusted_confidence))
    
    # Recalculate probabilities
    adjusted_probs = probs_np.copy()
    if pred_idx == 0:  # FAKE
        adjustment_ratio = adjusted_confidence / confidence if confidence > 0 else 1.0
        adjusted_probs[0] = probs_np[0] * adjustment_ratio
        adjusted_probs[1] = 1 - adjusted_probs[0]
    else:  # REAL
        adjustment_ratio = adjusted_confidence / confidence if confidence > 0 else 1.0
        adjusted_probs[1] = probs_np[1] * adjustment_ratio
        adjusted_probs[0] = 1 - adjusted_probs[1]
    
    # Re-determine prediction
    new_pred_idx = np.argmax(adjusted_probs)
    new_confidence = adjusted_probs[new_pred_idx] * 100
    
    # Log results
    if new_pred_idx != pred_idx:
        logger.warning(
            f"⚠️ PREDICTION CHANGED: {original_pred} ({original_confidence:.1f}%) → "
            f"{CLASS_MAP[new_pred_idx]} ({new_confidence:.1f}%)"
        )
        logger.warning(f"   Reason: {total_adjustment:+.1f}% adjustment from expression analysis")
        logger.warning(f"   Risk profile: intensity={intensity_risk:.2f}, smoothness={smoothness_risk:.2f}, "
                      f"flow={flow_risk:.2f}, overall={overall_risk:.2f}")
    elif len(adjustments) > 0:
        logger.info(
            f"📊 Confidence adjusted: {original_confidence:.1f}% → {new_confidence:.1f}% "
            f"({new_confidence - original_confidence:+.1f}%)"
        )
        logger.info(f"   Risk profile: overall={overall_risk:.2f}, high_risks={high_risks}, "
                   f"intensity={intensity_risk:.2f}, smoothness={smoothness_risk:.2f}")
    
    return new_pred_idx, new_confidence, adjusted_probs.tolist(), adjustments


def get_expression_report(expression_metrics, flow_metrics, landmark_metrics):
    """
    Generate a balanced report about expression patterns
    """
    report = {
        'exaggeration_level': 'UNKNOWN',
        'naturalness_score': 0.0,
        'confidence_level': 'UNCERTAIN',
        'risk_assessment': {
            'intensity_risk': 0.0,
            'smoothness_risk': 0.0,
            'flow_risk': 0.0,
            'landmark_risk': 0.0,
            'overall_risk': 0.0
        },
        'flags': []
    }
    
    if not expression_metrics:
        return report
    
    mean_intensity = expression_metrics.get('mean_intensity', 0)
    smoothness = expression_metrics.get('smoothness_score', 1.0)
    sudden_changes = expression_metrics.get('sudden_changes', 0)
    
    # Determine exaggeration level
    if mean_intensity > 1200:
        report['exaggeration_level'] = 'EXTREME'
    elif mean_intensity > 1000:
        report['exaggeration_level'] = 'VERY HIGH'
    elif mean_intensity > 850:
        report['exaggeration_level'] = 'HIGH'
    elif mean_intensity > 700:
        report['exaggeration_level'] = 'MODERATE'
    else:
        report['exaggeration_level'] = 'LOW'
    
    # Calculate naturalness score (0-1, higher is more natural)
    naturalness = smoothness * 0.5
    if flow_metrics:
        naturalness += flow_metrics.get('consistency_score', 0.5) * 0.3
    else:
        naturalness += 0.15  # Neutral value if no flow data
    
    if landmark_metrics:
        naturalness += landmark_metrics.get('stability_score', 0.5) * 0.2
    else:
        naturalness += 0.1  # Neutral value if no landmark data
    
    report['naturalness_score'] = min(1.0, naturalness)
    
    # Calculate risk scores
    intensity_risk = 0.0
    if mean_intensity > 1200:
        intensity_risk = 1.0
    elif mean_intensity > 1000:
        intensity_risk = 0.7
    elif mean_intensity > 850:
        intensity_risk = 0.4
    
    smoothness_risk = 0.0
    if smoothness < 0.2:
        smoothness_risk = 1.0
    elif smoothness < 0.3:
        smoothness_risk = 0.6
    elif smoothness < 0.4:
        smoothness_risk = 0.3
    
    flow_risk = 0.0
    if flow_metrics:
        consistency = flow_metrics.get('consistency_score', 1.0)
        if consistency < 0.3:
            flow_risk = 1.0
        elif consistency < 0.4:
            flow_risk = 0.6
        elif consistency < 0.5:
            flow_risk = 0.3
    
    landmark_risk = 0.0
    if landmark_metrics:
        max_jitter = landmark_metrics.get('max_jitter', 0)
        stability = landmark_metrics.get('stability_score', 1.0)
        if max_jitter > 15 or stability < 0.4:
            landmark_risk = 0.8
        elif max_jitter > 12 or stability < 0.5:
            landmark_risk = 0.5
    
    report['risk_assessment'] = {
        'intensity_risk': round(intensity_risk, 2),
        'smoothness_risk': round(smoothness_risk, 2),
        'flow_risk': round(flow_risk, 2),
        'landmark_risk': round(landmark_risk, 2),
        'overall_risk': round((intensity_risk + smoothness_risk + flow_risk + landmark_risk) / 4, 2)
    }
    
    # Determine confidence level
    overall_risk = report['risk_assessment']['overall_risk']
    if overall_risk > 0.7:
        report['confidence_level'] = 'HIGH_SUSPICION'
    elif overall_risk > 0.5:
        report['confidence_level'] = 'MODERATE_SUSPICION'
    elif overall_risk > 0.3:
        report['confidence_level'] = 'LOW_SUSPICION'
    else:
        report['confidence_level'] = 'APPEARS_NATURAL'
    
    # Add flags based on evidence
    if intensity_risk > 0.6 and smoothness_risk > 0.5:
        report['flags'].append('Exaggerated expressions with unnatural motion')
    
    if flow_risk > 0.6:
        report['flags'].append('Inconsistent optical flow pattern')
    
    if smoothness_risk >= 1.0:
        report['flags'].append('Severely jerky movement')
    
    if sudden_changes > 8:
        report['flags'].append('Excessive sudden expression changes')
    
    if landmark_risk > 0.7:
        report['flags'].append('Unstable facial landmarks')
    
    # If no flags, add positive note
    if len(report['flags']) == 0:
        report['flags'].append('No significant anomalies detected')
    
    return report

def get_expression_report(expression_metrics, flow_metrics, landmark_metrics):
    """
    Generate a detailed report about expression patterns
    Helps users understand why a video might be classified as fake despite exaggerated expressions
    """
    report = {
        'exaggeration_level': 'UNKNOWN',
        'naturalness_score': 0.0,
        'deepfake_signature_detected': False,
        'flags': []
    }
    
    if not expression_metrics:
        return report
    
    mean_intensity = expression_metrics.get('mean_intensity', 0)
    smoothness = expression_metrics.get('smoothness_score', 0)
    sudden_changes = expression_metrics.get('sudden_changes', 0)
    
    # Determine exaggeration level
    if mean_intensity > 1000:
        report['exaggeration_level'] = 'EXTREME'
    elif mean_intensity > 800:
        report['exaggeration_level'] = 'HIGH'
    elif mean_intensity > 600:
        report['exaggeration_level'] = 'MODERATE'
    else:
        report['exaggeration_level'] = 'LOW'
    
    # Calculate naturalness score (0-1, higher is more natural)
    naturalness = smoothness * 0.5  # Base from smoothness
    if flow_metrics:
        naturalness += flow_metrics.get('consistency_score', 0) * 0.3
    if landmark_metrics:
        naturalness += landmark_metrics.get('stability_score', 0) * 0.2
    
    report['naturalness_score'] = min(1.0, naturalness)
    
    # Detect deepfake signatures
    if mean_intensity > 700 and smoothness < 0.35:
        report['deepfake_signature_detected'] = True
        report['flags'].append('Exaggerated expressions with unnatural transitions')
    
    if flow_metrics and flow_metrics.get('consistency_score', 1.0) < 0.4:
        report['deepfake_signature_detected'] = True
        report['flags'].append('Inconsistent optical flow pattern')
    
    if smoothness < 0.25:
        report['flags'].append('Jerky, non-smooth movements')
    
    if sudden_changes > 7:
        report['flags'].append('Excessive sudden changes in expression')
    
    return report
   
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
        
        if prob_diff > 0.5:
            prediction_strength = "VERY HIGH"
        elif prob_diff > 0.3:
            prediction_strength = "HIGH"
        elif prob_diff > 0.15:
            prediction_strength = "MEDIUM"
        elif prob_diff > 0.05:
            prediction_strength = "LOW"
        else:
            prediction_strength = "UNCERTAIN"
        
        bias_warning = None
        if pred_idx == 1 and prob_diff < 0.20:
            bias_warning = "Low confidence REAL prediction - review manually"
        
        visualization_path = None
        visualization_b64 = None
        
        try:
            weight_softmax = model.linear1.weight.detach().cpu().numpy()
            bz, nc, h, w = fmap.shape
            idx_for_cam = np.argmax(probs.detach().cpu().numpy())
            
            # Generate CAM for ALL feature maps, not just predicted class
            cam_maps = []
            for class_idx in [0, 1]:  # Generate for both FAKE and REAL
                out = np.dot(
                    fmap[-1].detach().cpu().numpy().reshape((nc, h * w)).T,
                    weight_softmax[class_idx, :].T
                )
                predict_cam = out.reshape(h, w)
                
                # Better normalization to spread attention
                predict_cam = predict_cam - np.min(predict_cam)
                cam_max = np.max(predict_cam)
                if cam_max > 0:
                    predict_img = predict_cam / cam_max
                else:
                    predict_img = predict_cam
                
                # Apply gamma correction to enhance mid-range values
                predict_img = np.power(predict_img, 0.5)  # Gamma = 0.5 brightens mid-tones
                
                cam_maps.append(predict_img)
            
            # Use the predicted class CAM
            predict_img = cam_maps[idx_for_cam]
            
            # Convert to uint8
            predict_img = np.uint8(255 * predict_img)
            
            out_resized = cv2.resize(predict_img, (im_size, im_size))
            heatmap = cv2.applyColorMap(out_resized, cv2.COLORMAP_JET)
            
            # Get original image
            inv_normalize = transforms.Normalize(
                mean=-1 * np.divide(mean, std),
                std=np.divide([1, 1, 1], std)
            )
            image = img_tensor[:, -1, :, :, :].to("cpu").clone().detach()
            image = image.squeeze()
            image = inv_normalize(image)
            image = image.numpy().transpose(1, 2, 0).clip(0, 1)
            
            # Adjust blending for better visibility
            result = heatmap * 0.4 + image * 0.8 * 255
            result = np.clip(result, 0, 255).astype(np.uint8)
            
            # FIX: Create original_display as a proper numpy array copy in uint8 format
            original_display = (image * 255).astype(np.uint8).copy()
            
            # FIX: Ensure it's contiguous and in the right format
            if not original_display.flags['C_CONTIGUOUS']:
                original_display = np.ascontiguousarray(original_display)
            
            # Fake class CAM
            fake_cam = np.uint8(255 * cam_maps[0])
            fake_cam_resized = cv2.resize(fake_cam, (im_size, im_size))
            fake_heatmap = cv2.applyColorMap(fake_cam_resized, cv2.COLORMAP_JET).copy()
            
            # Real class CAM
            real_cam = np.uint8(255 * cam_maps[1])
            real_cam_resized = cv2.resize(real_cam, (im_size, im_size))
            real_heatmap = cv2.applyColorMap(real_cam_resized, cv2.COLORMAP_JET).copy()
            
            # FIX: Ensure result is also contiguous
            result = np.ascontiguousarray(result)
            
            # Add labels - now should work without errors
            cv2.putText(original_display, "Original", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(fake_heatmap, f"FAKE Focus", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(real_heatmap, f"REAL Focus", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(result, f"{CLASS_MAP[pred_idx]} ({confidence:.1f}%)", (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Stack horizontally
            multi_view = np.hstack([
                cv2.cvtColor(original_display, cv2.COLOR_RGB2BGR),
                fake_heatmap,
                real_heatmap,
                result
            ])
            
            # Save both versions
            vis_name = f"gradcam_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            vis_path = os.path.join(out_dir, vis_name)
            
            multi_name = f"gradcam_multi_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.png"
            multi_path = os.path.join(out_dir, multi_name)
            
            cv2.imwrite(vis_path, result)
            cv2.imwrite(multi_path, multi_view)
            
            # Encode the multi-view for response
            _, buffer = cv2.imencode('.png', multi_view)
            vis_b64 = f"data:image/png;base64,{base64.b64encode(buffer.tobytes()).decode('utf-8')}"
            
            visualization_path = multi_path
            visualization_b64 = vis_b64
            
            # Calculate attention coverage score
            attention_threshold = 0.3
            attention_coverage = np.sum(predict_img > (attention_threshold * 255)) / predict_img.size
            
            logger.info(f"Model attention coverage: {attention_coverage*100:.1f}% of face region")
            if attention_coverage < 0.4:
                logger.warning(f"⚠️ Low attention coverage! Model focusing on <40% of face")
            
        except Exception as e:
            logger.warning(f"Visualization generation failed: {e}")
            import traceback
            traceback.print_exc()
        
        logger.info(f"Probabilities: FAKE={probs_np[0]:.4f} ({probs_np[0]*100:.2f}%), REAL={probs_np[1]:.4f} ({probs_np[1]*100:.2f}%)")
        logger.info(f"Prediction: {CLASS_MAP[pred_idx]} with {confidence:.2f}% confidence [{prediction_strength}]")
        if bias_warning:
            logger.warning(f"⚠️ {bias_warning}")
        
        return pred_idx, confidence, probs_np.tolist(), top_predictions, visualization_path, visualization_b64, prediction_strength, bias_warning
    
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
    Strategy:
    - Few/No faces (0-30): Use shorter sequence models (faster, less overfitting)
    - Many faces (31+): Use longer sequence models (better context, patterns)
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
    
    # Selection logic based on face count
    if num_faces_detected <= 30:
        # For few/no faces: prioritize SHORTER sequences and higher accuracy
        best_model = min(model_info, key=lambda x: (x['seq_len'], -x['accuracy']))
        logger.info(f"✓ Few faces ({num_faces_detected}) → Selecting SHORTER sequence model (faster, less overfitting)")
    else:
        # For many faces: prioritize LONGER sequences for better temporal context
        best_model = max(model_info, key=lambda x: (x['seq_len'], x['accuracy']))
        logger.info(f"✓ Many faces ({num_faces_detected}) → Selecting LONGER sequence model (better context)")
    
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
    
    # Initialize variables at the top to avoid UnboundLocalError
    model = None
    frames_tensor = None
    faces_detected = 0
    expression_metrics = None
    flow_metrics = None
    landmark_metrics = None
    pred_idx = None
    confidence = 0
    probs = [0.5, 0.5]
    
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
        
        # STEP 2.5: Load model BEFORE processing frames (prevents UnboundLocalError)
        logger.info("STEP 2.5: Loading model...")
        model = load_model_cached(model_path)
        
        # STEP 3: Full frame processing (FIXED: using enhanced_process_frames)
        logger.info("STEP 3: Processing all frames...")
        frames_tensor, faces_detected, expression_metrics, flow_metrics, landmark_metrics = \
            enhanced_process_frames(frames, sequence_length=expected_seq_len)
        detection_time = time.time() - detection_start
        
        logger.info(f"Total processing took {detection_time:.2f}s")
        
        frames_collected = frames_tensor.shape[1]
        
        if frames_collected < MIN_FRAMES_REQUIRED:
            return jsonify({
                'error': f'Insufficient frames. Collected {frames_collected}, need at least {MIN_FRAMES_REQUIRED}.',
                'frames_collected': frames_collected,
                'faces_detected': faces_detected
            }), 400
        
        # STEP 4: Run inference
        logger.info("STEP 4: Running model inference...")
        inference_start = time.time()
        pred_idx, confidence, probs, top_predictions, vis_path, vis_b64, pred_strength, bias_warning = \
            predict_with_visualization(model, frames_tensor)
        inference_time = time.time() - inference_start
        
        # STEP 5: Apply expression analysis adjustments
        adjustments = []
        if expression_metrics:
            logger.info("STEP 5: Applying expression analysis adjustments...")
            pred_idx, confidence, probs, adjustments = adjust_prediction_for_expressions(
                pred_idx, confidence, np.array(probs), 
                expression_metrics, flow_metrics, landmark_metrics
            )
        
        prediction_label = CLASS_MAP[pred_idx]
        total_time = time.time() - start_time
        finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        response = {
            'prediction': prediction_label,
            'confidence': round(confidence, 2),
            'prediction_strength': pred_strength,
            'bias_warning': bias_warning,
            'raw_prediction': int(pred_idx),
            'all_probabilities': {
                'FAKE': round(probs[0] * 100, 2),
                'REAL': round(probs[1] * 100, 2)
            },
            'probability_difference': round(abs(probs[1] - probs[0]) * 100, 2),
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
        
        # Add expression analysis data if available
        if expression_metrics:
            response['expression_analysis'] = {
                'metrics': expression_metrics,
                'flow_metrics': flow_metrics,
                'landmark_metrics': landmark_metrics,
                'adjustments_applied': adjustments
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
        print(f"📊 Margin:            {abs(probs[1] - probs[0])*100:.2f}%")
        if bias_warning:
            print(f"⚠️  Warning:           {bias_warning}")
        if adjustments:
            print(f"🎭 Adjustments:       {len(adjustments)} expression-based adjustments applied")
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
        'dlib_available': DLIB_AVAILABLE,
        'timestamp': datetime.now().isoformat()
    }), 200

# -------------------- STARTUP --------------------
if __name__ == '__main__':
    print("\n" + "="*70)
    print("🚀 VERIFEED DEEPFAKE DETECTION SERVER (FIXED)")
    print("="*70)
    print(f"🕒 Server Started:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"💻 Device:            {DEVICE}")
    print(f"📁 Models Directory:  {MODELS_DIR}")
    print(f"🎬 Sequence Length:   {SEQUENCE_LENGTH} frames")
    print(f"📊 Min Frames:        {MIN_FRAMES_REQUIRED} frames")
    print(f"🏷️  Label Mapping:     {CLASS_MAP}")
    print(f"🔧 dlib Available:    {DLIB_AVAILABLE}")

    model_files = glob.glob(os.path.join(MODELS_DIR, "*.pt"))
    print(f"\n📦 Available Models: {len(model_files)}")

    if model_files:
        print("🔍 Model Details:")
        for mf in model_files:
            acc, seq = parse_model_info(mf)
            print(f"   - {os.path.basename(mf)}: {acc}% acc, {seq} frames")

    print("\n✨ MODEL SELECTION STRATEGY:")
    print("   • Few faces (0-30): Shorter sequence models (faster, less overfitting)")
    print("   • Many faces (31+): Longer sequence models (better temporal context)")
    print("   • Expression analysis applied when available")
    print("   • Note: Monitor confidence margins for potential model bias")
    
    print("\n🔧 FIXES APPLIED:")
    print("   ✓ Fixed function call mismatch (process_frames → enhanced_process_frames)")
    print("   ✓ Fixed UnboundLocalError (model loaded before frame processing)")
    print("   ✓ Integrated expression analysis adjustments")
    print("   ✓ Added dlib availability check")
    
    print("="*70)
    print("✅ Server ready at http://localhost:5000")
    print("="*70 + "\n")
    
    app.run(host='localhost', port=5000, debug=False, threaded=True)