import json
import base64
import cv2
import numpy as np
import pickle
import os
from tqdm import tqdm
from pathlib import Path

# --- ALIGNED CONFIGURATION ---
TRAINING_DATA_DIR = Path("training_data")
METADATA_FILE = TRAINING_DATA_DIR / "metadata.json"
PKL_OUTPUT_FILE = TRAINING_DATA_DIR / "training_samples.pkl"
SEQUENCE_LENGTH = 20
IM_SIZE = 112
# -----------------------------

def decode_base64_to_frame(b64_frame: str) -> np.ndarray | None:
    """Decodes and resizes a single base64 image frame."""
    try:
        if ',' in b64_frame:
            b64_frame = b64_frame.split(',', 1)[1]
        nparr = np.frombuffer(base64.b64decode(b64_frame), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            # Convert BGR to RGB and resize for consistency
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, (IM_SIZE, IM_SIZE))
        return frame
    except Exception:
        return None

def preprocess_data():
    if not METADATA_FILE.exists():
        print(f"ERROR: Metadata file not found at {METADATA_FILE}")
        return

    print("--- Starting Data Preprocessing (Base64 -> Pickle) ---")
    
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)

    processed_data = []
    
    for sample in tqdm(metadata.get('samples', []), desc="Processing Samples"):
        frames_b64 = sample.get('frames', [])
        label = sample.get('label')
        
        if len(frames_b64) != SEQUENCE_LENGTH or label is None:
            # Skip invalid samples
            continue

        face_frames_np = []
        is_valid = True
        for b64 in frames_b64:
            frame_np = decode_base64_to_frame(b64)
            if frame_np is None:
                is_valid = False
                break
            face_frames_np.append(frame_np)

        if is_valid:
            processed_data.append({
                'face_frames': face_frames_np,
                'label': label
            })

    # Save to pickle file
    with open(PKL_OUTPUT_FILE, 'wb') as f:
        pickle.dump(processed_data, f)

    real_count = sum(1 for d in processed_data if d['label'] == 1)
    fake_count = len(processed_data) - real_count
    
    print("\n🎉 Preprocessing Complete!")
    print(f"Total valid samples saved: {len(processed_data)}")
    print(f"  - REAL: {real_count}, FAKE: {fake_count}")
    print(f"Data saved to: {PKL_OUTPUT_FILE}")

if __name__ == '__main__':
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
    preprocess_data()