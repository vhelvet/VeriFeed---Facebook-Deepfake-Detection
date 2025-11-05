"""
VERIFEED MODEL TRAINING SCRIPT
Loads data from training_samples.pkl and trains the deepfake detection model
SAVES ALL MODELS with format: model_EPOCH_acc_ACCURACY_frames.pt
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import pickle
import numpy as np
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration (must match backend)
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters
BATCH_SIZE = 4
EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5

# Directories
TRAINING_DATA_DIR = 'training_data'
MODELS_DIR = 'models'
os.makedirs(MODELS_DIR, exist_ok=True)

# Model Architecture (same as backend)
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

# Dataset class
class DeepfakeDataset(Dataset):
    def __init__(self, samples, labels, transform=None):
        """
        Args:
            samples: List of face frame sequences (each is a list of numpy arrays)
            labels: List of labels (0=FAKE, 1=REAL)
            transform: torchvision transforms to apply
        """
        self.samples = samples
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        face_frames = self.samples[idx]
        label = self.labels[idx]
        
        # Transform each frame in the sequence
        transformed_frames = []
        for frame in face_frames:
            if self.transform:
                frame_tensor = self.transform(frame)
                transformed_frames.append(frame_tensor)
        
        # Stack into sequence: (SEQUENCE_LENGTH, C, H, W)
        sequence = torch.stack(transformed_frames)
        
        return sequence, label

# Transforms
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

def load_training_data():
    """Load training data from pickle file"""
    training_file = os.path.join(TRAINING_DATA_DIR, 'training_samples.pkl')
    
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"Training data not found at {training_file}")
    
    logger.info(f"Loading training data from {training_file}")
    
    with open(training_file, 'rb') as f:
        training_data = pickle.load(f)
    
    logger.info(f"Loaded {len(training_data)} samples")
    
    # Extract samples and labels
    samples = []
    labels = []
    
    for sample in training_data:
        face_frames = sample['face_frames']
        label = sample['label']
        
        # Ensure we have exactly SEQUENCE_LENGTH frames for training
        if len(face_frames) != SEQUENCE_LENGTH:
            if len(face_frames) > SEQUENCE_LENGTH:
                # Select evenly distributed frames
                indices = np.linspace(0, len(face_frames) - 1, SEQUENCE_LENGTH, dtype=int)
                selected_faces = [face_frames[i] for i in indices]
            elif len(face_frames) > 0:
                # Pad with last frame if needed (only if some faces were detected)
                selected_faces = face_frames.copy()
                while len(selected_faces) < SEQUENCE_LENGTH:
                    selected_faces.append(selected_faces[-1])
            else:
                # Skip samples with zero faces detected
                continue
        else:
            selected_faces = face_frames
        
        samples.append(selected_faces)
        labels.append(label)
    
    # Print statistics
    real_count = sum(1 for l in labels if l == 1)
    fake_count = len(labels) - real_count
    logger.info(f"Dataset: {len(samples)} total ({real_count} REAL, {fake_count} FAKE)")
    
    return samples, labels

def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
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
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def validate_epoch(model, dataloader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc='Validation'):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = 100 * correct / total
    
    return epoch_loss, epoch_acc

def main():
    """Main training function"""
    print("\n" + "="*70)
    print("🎯 VERIFEED MODEL TRAINING")
    print("="*70)
    print(f"Device: {DEVICE}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"Epochs: {EPOCHS}")
    print(f"Learning Rate: {LEARNING_RATE}")
    print("="*70 + "\n")
    
    # Load data
    try:
        samples, labels = load_training_data()
    except FileNotFoundError as e:
        logger.error(str(e))
        logger.error("Please add training samples using the /training/add_sample endpoint first")
        return
    
    if len(samples) < 10:
        logger.error(f"Not enough samples for training (have {len(samples)}, need at least 10)")
        return
    
    # Split data: 80% train, 20% validation
    train_samples, val_samples, train_labels, val_labels = train_test_split(
        samples, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    logger.info(f"Train set: {len(train_samples)} samples")
    logger.info(f"Validation set: {len(val_samples)} samples")
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_samples, train_labels, transform=train_transforms)
    val_dataset = DeepfakeDataset(val_samples, val_labels, transform=val_transforms)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    # Initialize model
    model = DeepfakeDetectionModel(num_classes=2).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3
    )
    
    # Training loop
    best_val_acc = 0.0
    
    for epoch in range(EPOCHS):
        print(f"\nEpoch {epoch+1}/{EPOCHS}")
        print("-" * 50)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_acc = validate_epoch(model, val_loader, criterion, DEVICE)
        
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Step scheduler
        scheduler.step(val_loss)
        
        # --- NEW MODEL SAVING LOGIC ---
        # 1. Generate the required file name
        # Format: model_XX_acc_YY_frames.pt
        
        # Use a zero-padded epoch number (e.g., 01, 05, 10)
        epoch_str = str(epoch + 1).zfill(2)
        
        # Round accuracy to nearest integer for the filename
        acc_int = int(round(val_acc))
        
        # Construct the file name
        model_filename = f"model_{epoch_str}_acc_{acc_int}_frames.pt"
        model_path = os.path.join(MODELS_DIR, model_filename)

        # 2. Save the model state
        torch.save(model.state_dict(), model_path)
        logger.info(f"💾 Saved model checkpoint: {model_filename}")
        
        # 3. Update best model logic (optional, but good practice)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Also save the best model to a standard name for easy backend loading
            torch.save(model.state_dict(), os.path.join(MODELS_DIR, 'best_model.pt'))
            logger.info("✓ Updated 'best_model.pt'")
    
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"All models saved in the '{MODELS_DIR}/' directory.")
    print("="*70 + "\n")

if __name__ == '__main__':
    main()