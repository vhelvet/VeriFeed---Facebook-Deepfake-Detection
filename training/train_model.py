"""
VERIFEED MODEL TRAINING SCRIPT
Trains deepfake detection model using frames extracted from content.js
Extracts first 100 faces from 200 frames (10fps x 20s)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import cv2
import face_recognition
import json
import os
from datetime import datetime
import logging
from tqdm import tqdm
import pickle

# -------------------- CONFIGURATION --------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Model configuration
SEQUENCE_LENGTH = 20  # Number of frames to use for training
MIN_FACE_SIZE = 40
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# Training configuration
BATCH_SIZE = 4
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Directories
DATA_DIR = 'training_data'
MODELS_DIR = 'models'
CHECKPOINT_DIR = 'checkpoints'

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# -------------------- TRANSFORMS --------------------
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(5),
    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# -------------------- MODEL ARCHITECTURE --------------------
class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, 
                 hidden_dim=2048, bidirectional=False, lstm_bias=True):
        super(DeepfakeDetectionModel, self).__init__()
        
        # ResNeXt-50 backbone
        model = models.resnext50_32x4d(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-2])
        
        # LSTM for temporal modeling
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, 
                           bidirectional=bidirectional, bias=lstm_bias)
        
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
        return out

# -------------------- FACE DETECTION --------------------
def detect_first_100_faces(frames, sequence_length=20):
    """
    Extract first 100 faces from frames
    Returns sequence_length frames for model training
    """
    face_frames = []
    faces_found = 0
    total_frames = len(frames)
    
    logger.info(f"Detecting faces in {total_frames} frames")
    logger.info(f"Target: First 100 faces, collecting {sequence_length} for model")
    
    detection_model = "cnn" if DEVICE.type == "cuda" else "hog"
    
    for i in tqdm(range(total_frames), desc="Face Detection"):
        if faces_found >= 100:
            break
            
        frame = frames[i]
        if frame is None or frame.size == 0:
            continue
        
        try:
            # Resize for faster detection
            h, w = frame.shape[:2]
            if max(h, w) > 800:
                scale = 800 / max(h, w)
                small_frame = cv2.resize(frame, (int(w * scale), int(h * scale)))
                scale_back = max(h, w) / 800
            else:
                small_frame = frame
                scale_back = 1.0
            
            # Detect faces
            face_locations = face_recognition.face_locations(
                small_frame,
                model=detection_model,
                number_of_times_to_upsample=0
            )
            
            if len(face_locations) > 0:
                # Take first face only
                top, right, bottom, left = face_locations[0]
                
                # Scale back to original size
                if scale_back != 1.0:
                    top = int(top * scale_back)
                    right = int(right * scale_back)
                    bottom = int(bottom * scale_back)
                    left = int(left * scale_back)
                
                # Crop face
                face_img = frame[top:bottom, left:right, :]
                
                if face_img.size > 0 and face_img.shape[0] >= MIN_FACE_SIZE and face_img.shape[1] >= MIN_FACE_SIZE:
                    faces_found += 1
                    
                    # Only collect frames needed for model sequence
                    if len(face_frames) < sequence_length:
                        face_frames.append(face_img)
                    
        except Exception as e:
            logger.debug(f"Face detection error frame {i}: {e}")
            continue
    
    if len(face_frames) < sequence_length:
        logger.warning(f"Only got {len(face_frames)} face frames, padding to {sequence_length}")
        if len(face_frames) > 0:
            last_frame = face_frames[-1]
            while len(face_frames) < sequence_length:
                face_frames.append(last_frame)
        else:
            raise ValueError("No faces detected in video")
    
    logger.info(f"Collected {len(face_frames)} face frames ({faces_found} total faces detected)")
    
    return face_frames[:sequence_length]

# -------------------- DATASET --------------------
class DeepfakeDataset(Dataset):
    def __init__(self, data_file, transform=None, sequence_length=20):
        self.data = []
        self.transform = transform
        self.sequence_length = sequence_length
        
        # Load data from file
        if os.path.exists(data_file):
            with open(data_file, 'rb') as f:
                self.data = pickle.load(f)
            logger.info(f"Loaded {len(self.data)} samples from {data_file}")
        else:
            logger.warning(f"Data file {data_file} not found")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        face_frames = sample['face_frames']
        label = sample['label']
        
        # Apply transforms
        transformed_frames = []
        for frame in face_frames:
            if self.transform:
                frame_tensor = self.transform(frame)
            else:
                frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).float() / 255.0
            transformed_frames.append(frame_tensor)
        
        # Stack frames into sequence
        sequence = torch.stack(transformed_frames)
        
        return sequence, label

# -------------------- DATA COLLECTION --------------------
def collect_training_data():
    """
    Interactive data collection from content.js extracted frames
    """
    print("\n" + "="*70)
    print("🎥 VERIFEED TRAINING DATA COLLECTION")
    print("="*70)
    print("This script will help you collect training data from videos.")
    print("For each video:")
    print("  1. Load frames from a JSON file (exported from content.js)")
    print("  2. Extract first 100 faces from frames")
    print("  3. Label the video as REAL (1) or FAKE (0)")
    print("  4. Save to training dataset")
    print("="*70 + "\n")
    
    training_data = []
    data_file = os.path.join(DATA_DIR, 'training_samples.pkl')
    
    # Load existing data if available
    if os.path.exists(data_file):
        with open(data_file, 'rb') as f:
            training_data = pickle.load(f)
        print(f"📦 Loaded {len(training_data)} existing samples")
    
    while True:
        print("\n" + "-"*70)
        print("Options:")
        print("  1. Add new video sample")
        print("  2. Load frames from JSON file")
        print("  3. View dataset statistics")
        print("  4. Finish and save dataset")
        print("-"*70)
        
        choice = input("Enter choice (1-4): ").strip()
        
        if choice == '1':
            # Manual frame input (for testing)
            print("\n⚠️  Manual input not recommended. Use option 2 to load from JSON.")
            
        elif choice == '2':
            # Load from JSON file (frames exported from content.js)
            json_path = input("\nEnter path to JSON file with frames: ").strip()
            
            if not os.path.exists(json_path):
                print(f"❌ File not found: {json_path}")
                continue
            
            try:
                print(f"📂 Loading frames from {json_path}...")
                with open(json_path, 'r') as f:
                    data = json.load(f)
                
                frames_b64 = data.get('frames', [])
                
                if not frames_b64:
                    print("❌ No frames found in JSON file")
                    continue
                
                print(f"✓ Loaded {len(frames_b64)} frames")
                
                # Decode base64 frames
                print("🔄 Decoding frames...")
                frames = []
                for i, b64_frame in enumerate(tqdm(frames_b64, desc="Decoding")):
                    try:
                        if ',' in b64_frame:
                            b64_frame = b64_frame.split(',')[1]
                        image_data = np.frombuffer(np.frombuffer(b64_frame.encode(), dtype=np.uint8), dtype=np.uint8)
                        import base64
                        image_data = base64.b64decode(b64_frame)
                        nparr = np.frombuffer(image_data, np.uint8)
                        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        if frame is not None:
                            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            frames.append(frame)
                    except Exception as e:
                        logger.debug(f"Error decoding frame {i}: {e}")
                
                print(f"✓ Decoded {len(frames)} frames successfully")
                
                # Detect faces
                print("\n👤 Detecting faces...")
                face_frames = detect_first_100_faces(frames, sequence_length=SEQUENCE_LENGTH)
                
                print(f"✓ Extracted {len(face_frames)} face frames")
                
                # Get label
                print("\n🏷️  Label this video:")
                print("  Enter 1 for REAL video")
                print("  Enter 0 for FAKE/DEEPFAKE video")
                
                while True:
                    label_input = input("Label (0 or 1): ").strip()
                    if label_input in ['0', '1']:
                        label = int(label_input)
                        break
                    print("❌ Invalid input. Enter 0 or 1.")
                
                # Save sample
                sample = {
                    'face_frames': face_frames,
                    'label': label,
                    'timestamp': datetime.now().isoformat(),
                    'source_file': json_path,
                    'num_frames': len(frames)
                }
                
                training_data.append(sample)
                
                label_text = "REAL" if label == 1 else "FAKE"
                print(f"\n✓ Added sample #{len(training_data)}: {label_text}")
                
                # Auto-save
                with open(data_file, 'wb') as f:
                    pickle.dump(training_data, f)
                print(f"💾 Dataset saved ({len(training_data)} total samples)")
                
            except Exception as e:
                print(f"❌ Error processing video: {e}")
                logger.error(f"Error: {e}", exc_info=True)
        
        elif choice == '3':
            # Show statistics
            if not training_data:
                print("\n📊 Dataset is empty")
            else:
                real_count = sum(1 for s in training_data if s['label'] == 1)
                fake_count = len(training_data) - real_count
                
                print("\n" + "="*70)
                print("📊 DATASET STATISTICS")
                print("="*70)
                print(f"Total samples:     {len(training_data)}")
                print(f"REAL videos:       {real_count} ({real_count/len(training_data)*100:.1f}%)")
                print(f"FAKE videos:       {fake_count} ({fake_count/len(training_data)*100:.1f}%)")
                print(f"Sequence length:   {SEQUENCE_LENGTH} frames")
                print("="*70)
        
        elif choice == '4':
            # Finish
            if len(training_data) < 10:
                print(f"\n⚠️  Warning: Only {len(training_data)} samples. Recommended: at least 50-100 samples.")
                confirm = input("Continue anyway? (y/n): ").strip().lower()
                if confirm != 'y':
                    continue
            
            # Save final dataset
            with open(data_file, 'wb') as f:
                pickle.dump(training_data, f)
            
            print(f"\n✓ Dataset saved: {data_file}")
            print(f"✓ Total samples: {len(training_data)}")
            return data_file
        
        else:
            print("❌ Invalid choice")

# -------------------- TRAINING --------------------
def train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS):
    """
    Train the deepfake detection model
    """
    print("\n" + "="*70)
    print("🚀 STARTING MODEL TRAINING")
    print("="*70)
    print(f"Device:           {DEVICE}")
    print(f"Batch size:       {BATCH_SIZE}")
    print(f"Learning rate:    {LEARNING_RATE}")
    print(f"Epochs:           {num_epochs}")
    print(f"Sequence length:  {SEQUENCE_LENGTH}")
    print("="*70 + "\n")
    
    # Initialize model
    model = DeepfakeDetectionModel(num_classes=2).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5)
    
    best_val_acc = 0.0
    best_model_path = None
    
    for epoch in range(num_epochs):
        print(f"\n{'='*70}")
        print(f"EPOCH {epoch+1}/{num_epochs}")
        print('='*70)
        
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        for sequences, labels in train_pbar:
            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            train_total += labels.size(0)
            train_correct += (predicted == labels).sum().item()
            
            train_pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100*train_correct/train_total:.2f}%'
            })
        
        train_acc = 100 * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            for sequences, labels in val_pbar:
                sequences = sequences.to(DEVICE)
                labels = labels.to(DEVICE)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                
                val_pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100*val_correct/val_total:.2f}%'
                })
        
        val_acc = 100 * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        scheduler.step(avg_val_loss)
        
        # Print epoch summary
        print(f"\n📊 Epoch {epoch+1} Summary:")
        print(f"   Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"   Val Loss:   {avg_val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"model_{int(val_acc)}_acc_{SEQUENCE_LENGTH}_frames_{timestamp}.pt"
            model_path = os.path.join(MODELS_DIR, model_filename)
            
            torch.save(model.state_dict(), model_path)
            best_model_path = model_path
            print(f"   ✓ Best model saved: {model_filename}")
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f"checkpoint_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': avg_train_loss,
            'val_loss': avg_val_loss,
            'train_acc': train_acc,
            'val_acc': val_acc,
        }, checkpoint_path)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved at: {best_model_path}")
    print("="*70 + "\n")
    
    return best_model_path, best_val_acc

# -------------------- MAIN --------------------
def main():
    print("\n" + "="*70)
    print("🎯 VERIFEED MODEL TRAINING PIPELINE")
    print("="*70)
    print("This script trains a deepfake detection model using:")
    print("  - 200 frames extracted at 10fps for 20 seconds")
    print("  - First 100 faces detected from frames")
    print("  - Sequences of 20 frames for temporal analysis")
    print("="*70 + "\n")
    
    # Step 1: Collect training data
    print("STEP 1: Data Collection")
    data_file = collect_training_data()
    
    if not os.path.exists(data_file):
        print("❌ No training data collected. Exiting.")
        return
    
    # Step 2: Prepare datasets
    print("\nSTEP 2: Preparing Datasets")
    
    # Load all data
    with open(data_file, 'rb') as f:
        all_data = pickle.load(f)
    
    total_samples = len(all_data)
    train_size = int(0.8 * total_samples)
    val_size = total_samples - train_size
    
    print(f"Total samples: {total_samples}")
    print(f"Training: {train_size} | Validation: {val_size}")
    
    # Split data
    import random
    random.shuffle(all_data)
    
    train_data_file = os.path.join(DATA_DIR, 'train_data.pkl')
    val_data_file = os.path.join(DATA_DIR, 'val_data.pkl')
    
    with open(train_data_file, 'wb') as f:
        pickle.dump(all_data[:train_size], f)
    
    with open(val_data_file, 'wb') as f:
        pickle.dump(all_data[train_size:], f)
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_data_file, transform=train_transforms, 
                                   sequence_length=SEQUENCE_LENGTH)
    val_dataset = DeepfakeDataset(val_data_file, transform=val_transforms,
                                 sequence_length=SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                             shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=2)
    
    print(f"✓ Train batches: {len(train_loader)}")
    print(f"✓ Val batches: {len(val_loader)}")
    
    # Step 3: Train model
    print("\nSTEP 3: Model Training")
    confirm = input("Start training? (y/n): ").strip().lower()
    
    if confirm == 'y':
        best_model_path, best_acc = train_model(train_loader, val_loader, num_epochs=NUM_EPOCHS)
        
        print(f"\n🎉 Training completed!")
        print(f"📁 Best model: {best_model_path}")
        print(f"📊 Best accuracy: {best_acc:.2f}%")
    else:
        print("Training cancelled.")

if __name__ == '__main__':
    main()