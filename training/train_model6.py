"""
VERIFEED MODEL TRAINING
High-accuracy deepfake detection model training with 70/30 split
Target: 94-100% accuracy
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import cv2 # Kept for decode_base64_to_frame helper function, using cv2.imdecode
import json
import os
import base64  # NEW: For decoding frames
import io      # NEW: For handling image bytes
from PIL import Image # Used for safely decoding Base64 image bytes
from pathlib import Path
import logging
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# CONFIGURATION (Matches prediction backend exactly)
# ============================================================================
SEQUENCE_LENGTH = 20
IM_SIZE = 112
# MIN_FACE_SIZE = 40  <- NO LONGER NEEDED, FACES ARE ALREADY EXTRACTED
# MAX_FACES = 300     <- NO LONGER NEEDED
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Configuration
BATCH_SIZE = 4
NUM_EPOCHS = 15
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.3
RANDOM_SEED = 42

# Paths (Adjusted for local use if necessary, but using existing names)
# NOTE: Assuming you have corrected METADATA_PATH/VIDEO_BASE_PATH if needed
METADATA_PATH = 'training_data/metadata.json' # Adjusted path for common scenario
VIDEO_BASE_PATH = 'training_data' # Still used for logging/context, but not file access
MODELS_DIR = 'models'
CHECKPOINT_DIR = 'checkpoints'
LOGS_DIR = 'logs'

# Create directories
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Set random seeds for reproducibility
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(RANDOM_SEED)

# ============================================================================
# MODEL ARCHITECTURE (No changes needed)
# ============================================================================
class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1,
                 hidden_dim=2048, bidirectional=False):
        super(DeepfakeDetectionModel, self).__init__()
        model = models.resnext50_32x4d(weights='DEFAULT')
        self.model = nn.Sequential(*list(model.children())[:-2])
        # Using batch_first=True is generally good practice
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers,
                            bidirectional=bidirectional, batch_first=True) 
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

# ============================================================================
# DATA PREPROCESSING (No changes needed)
# ============================================================================
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# ============================================================================
# FRAME DECODING HELPER FUNCTION (NEW)
# ============================================================================
def decode_base64_to_frame(b64_frame: str) -> np.ndarray | None:
    """Decode a Base64 string to a NumPy array (RGB) for torchvision."""
    try:
        # Check for and remove common header (e.g., 'data:image/jpeg;base64,')
        if ',' in b64_frame:
            b64_frame = b64_frame.split(',', 1)[1]
            
        img_bytes = base64.b64decode(b64_frame)
        
        # Use PIL and numpy for robust decoding
        image = Image.open(io.BytesIO(img_bytes))
        
        # Convert to NumPy array in RGB format (H, W, C)
        frame_rgb = np.array(image.convert("RGB"))
        return frame_rgb
    except Exception as e:
        logger.warning(f"Error decoding base64 frame: {e}")
        return None

# ============================================================================
# MOCK FACE EXTRACTION FUNCTION (DELETED/REPLACED by direct frame loading)
# ============================================================================
# The original extract_faces_from_video function is REMOVED
# as we load pre-extracted frames directly from JSON.


# ============================================================================
# DATASET CLASS (MODIFIED for Base64 input)
# ============================================================================
class DeepfakeDataset(Dataset):
    # This class now takes the raw data (list of frame lists) and labels
    def __init__(self, data_samples, labels, transform=None):
        self.data_samples = data_samples
        self.labels = labels
        self.transform = transform
        
    def __len__(self):
        return len(self.data_samples)
    
    def __getitem__(self, idx):
        base64_frames = self.data_samples[idx]
        label = self.labels[idx]
        
        transformed_frames = []
        
        # Decode and transform each Base64 frame
        for b64_str in base64_frames:
            frame_rgb = decode_base64_to_frame(b64_str)
            
            if frame_rgb is None:
                # Use a black image placeholder if decoding fails
                frame_rgb = np.zeros((IM_SIZE, IM_SIZE, 3), dtype=np.uint8) 
            
            # Apply transformations (requires input as H, W, C NumPy array)
            if self.transform:
                frame_tensor = self.transform(frame_rgb)
            else:
                frame_tensor = val_transforms(frame_rgb) # Fallback to val transforms
            
            transformed_frames.append(frame_tensor)
        
        # Sanity check and stacking
        if len(transformed_frames) != SEQUENCE_LENGTH:
            logger.warning(f"Sample {idx} has {len(transformed_frames)} frames, expected {SEQUENCE_LENGTH}. Returning zeros.")
            sequence = torch.zeros(SEQUENCE_LENGTH, 3, IM_SIZE, IM_SIZE)
        else:
            sequence = torch.stack(transformed_frames)
            
        return sequence, torch.tensor(label, dtype=torch.long) # Return label as torch.long

# ============================================================================
# LOAD METADATA (MODIFIED to extract Base64 frames and labels)
# ============================================================================
def load_metadata():
    """Load and parse metadata.json to extract frame sequences and labels."""
    logger.info(f"Loading metadata from {METADATA_PATH}")
    
    with open(METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    # data_samples holds a list of lists of Base64 strings (the frames)
    data_samples = []
    labels = []
    
    # Keys that are NOT actual video samples (like summary keys)
    NON_VIDEO_KEYS = ['total_samples', 'real_samples', 'fake_samples', 'samples'] 
    
    # Check if the structure uses a top-level 'samples' key (common in data pipelines)
    if 'samples' in metadata and isinstance(metadata['samples'], list):
        items_to_process = metadata['samples']
        logger.info("Processing data from top-level 'samples' list.")
    else:
        items_to_process = metadata.items()
        
    
    real_count = 0
    fake_count = 0
        
    for item in items_to_process:
        if isinstance(item, dict):
            # If the item is already a dictionary (from a top-level 'samples' list)
            info = item
        else:
            # If the item is a (key, value) tuple (from metadata.items())
            video_name_id, info = item 
            if video_name_id in NON_VIDEO_KEYS:
                continue

        base64_frames = info.get('frames', [])
        label = int(info.get('label', -1))
        
        # Skip if label is invalid or frame count is incorrect
        if label not in [0, 1] or len(base64_frames) != SEQUENCE_LENGTH:
            logger.warning(f"Skipping sample (Label: {label}, Frames: {len(base64_frames)}, Expected: {SEQUENCE_LENGTH})")
            continue
            
        data_samples.append(base64_frames)
        labels.append(label)
        
        if label == 1: real_count += 1
        else: fake_count += 1
    
    logger.info(f"Loaded {len(data_samples)} sequences successfully.")
    logger.info(f"REAL sequences: {real_count}")
    logger.info(f"FAKE sequences: {fake_count}")
    
    # data_samples is now a list of lists of Base64 strings
    return data_samples, labels

# ============================================================================
# TRAINING FUNCTIONS (No changes needed)
# ============================================================================
def train_epoch(model, dataloader, criterion, optimizer, device):
    # ... (Keep existing train_epoch logic)
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc='Training')
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    
    return epoch_loss, epoch_acc

def validate(model, dataloader, criterion, device):
    # ... (Keep existing validate logic)
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validating')
        for sequences, labels in pbar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probabilities.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    
    # Calculate precision, recall, f1
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted', zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, precision, recall, f1, cm

def plot_confusion_matrix(cm, epoch, save_path):
    # ... (Keep existing plot_confusion_matrix logic)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['FAKE', 'REAL'],
                yticklabels=['FAKE', 'REAL'])
    plt.title(f'Confusion Matrix - Epoch {epoch}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_training_history(history, save_path):
    # ... (Keep existing plot_training_history logic)
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss plot
    axes[0].plot(history['train_loss'], label='Train Loss', marker='o')
    axes[0].plot(history['val_loss'], label='Val Loss', marker='o')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True)
    
    # Accuracy plot
    axes[1].plot(history['train_acc'], label='Train Acc', marker='o')
    axes[1].plot(history['val_acc'], label='Val Acc', marker='o')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ============================================================================
# MAIN TRAINING FUNCTION (Modified to use new data structure)
# ============================================================================
def main():
    logger.info("="*80)
    logger.info("🚀 VERIFEED MODEL TRAINING")
    logger.info("="*80)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Epochs: {NUM_EPOCHS}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Train/Val Split: {TRAIN_SPLIT}/{VAL_SPLIT}")
    logger.info(f"Target Accuracy: 94-100%")
    logger.info("="*80)
    
    # 1. Load data sequences and labels (data_samples now holds Base64 lists)
    data_samples, labels = load_metadata()
    
    if len(data_samples) == 0:
        logger.error("No valid sequences loaded from metadata.json. Training cannot proceed.")
        return
    
    # 2. Split data (70/30) - Splitting the data_samples (Base64 lists) and labels
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        data_samples, labels, 
        train_size=TRAIN_SPLIT,
        random_state=RANDOM_SEED,
        stratify=labels
    )
    
    logger.info(f"\nDataset Split:")
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Validation samples: {len(val_samples)}")
    
    # 3. Create datasets using the split data lists
    train_dataset = DeepfakeDataset(train_samples, train_labels, transform=train_transforms)
    val_dataset = DeepfakeDataset(val_samples, val_labels, transform=val_transforms)
    
    # 4. Create dataloaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=0, # Changed num_workers to 0 for better Windows compatibility during development
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=0, # Changed num_workers to 0
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    # ... (Rest of the script remains unchanged)
    
    # Initialize model
    logger.info("\nInitializing model...")
    model = DeepfakeDetectionModel(num_classes=2).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'val_precision': [], 'val_recall': [], 'val_f1': []
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    # Training loop
    logger.info("\n" + "="*80)
    logger.info("Starting Training...")
    logger.info("="*80)
    
    for epoch in range(NUM_EPOCHS):
        logger.info(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        logger.info("-" * 80)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc, val_precision, val_recall, val_f1, cm = validate(
            model, val_loader, criterion, DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_acc)
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_precision'].append(val_precision)
        history['val_recall'].append(val_recall)
        history['val_f1'].append(val_f1)
        
        # Log metrics
        logger.info(f"\nResults:")
        logger.info(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        logger.info(f"  Precision: {val_precision:.4f} | Recall: {val_recall:.4f} | F1: {val_f1:.4f}")
        logger.info(f"\n  Confusion Matrix:")
        logger.info(f"    {cm}")
        
        # Save confusion matrix plot
        cm_path = os.path.join(LOGS_DIR, f'confusion_matrix_e{epoch+1}.png')
        plot_confusion_matrix(cm, epoch + 1, cm_path)
        
        # Save checkpoint
        checkpoint_path = os.path.join(CHECKPOINT_DIR, f'checkpoint_e{epoch+1}.pt')
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_acc': train_acc,
            'val_acc': val_acc,
            'history': history
        }, checkpoint_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            
            # Save with accuracy in filename (matches prediction backend naming)
            best_model_path = os.path.join(MODELS_DIR, f'model_acc_{val_acc:.2f}_e{epoch+1}.pt')
            torch.save(model.state_dict(), best_model_path)
            
            # Also save as best_model.pt for easy loading
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))
            
            logger.info(f"\n  ✅ New best model saved! Accuracy: {val_acc:.2f}%")
            
            # Check if we've reached target accuracy
            if val_acc >= 94.0:
                logger.info(f"\n  🎯 TARGET ACCURACY REACHED: {val_acc:.2f}% >= 94%")
    
    # Final summary
    logger.info("\n" + "="*80)
    logger.info("Training Complete!")
    logger.info("="*80)
    logger.info(f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})")
    logger.info(f"Best model saved at: {MODELS_DIR}/model_acc_{best_val_acc:.2f}_e{best_epoch}.pt")
    
    if best_val_acc >= 94.0:
        logger.info(f"✅ SUCCESS: Achieved target accuracy of 94-100%!")
    else:
        logger.info(f"⚠️  Final accuracy: {best_val_acc:.2f}% (Target: 94-100%)")
        logger.info("  Consider: longer training, more data augmentation, or hyperparameter tuning")
        
    # Plot training history
    history_plot_path = os.path.join(LOGS_DIR, 'training_history.png')
    plot_training_history(history, history_plot_path)
    logger.info(f"\nTraining plots saved to: {LOGS_DIR}/")
    
    # Save training history
    history_json_path = os.path.join(LOGS_DIR, 'training_history.json')
    with open(history_json_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    logger.info("="*80)

if __name__ == '__main__':
    main()