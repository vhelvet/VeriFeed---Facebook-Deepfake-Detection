"""
VERIFEED MODEL TRAINING - Advanced Training Pipeline
Target: 94-100% Accuracy with Proper Validation
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import pickle
import os
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import logging
from tqdm import tqdm
import random

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training hyperparameters - OPTIMIZED FOR 500 SAMPLES
BATCH_SIZE = 8  # Increased for faster training with more data
EPOCHS = 100  # More epochs for convergence
LEARNING_RATE = 5e-5  # Lower LR for stability
WEIGHT_DECAY = 1e-4  # Stronger regularization
PATIENCE = 15  # More patience for convergence
MIN_DELTA = 0.0005  # Smaller delta for fine improvements

# Directories
TRAINING_DATA_DIR = 'training_data'
MODELS_DIR = 'models'
LOGS_DIR = 'logs'
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# Model Architecture (Same as backend)
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


# Advanced Data Augmentation
class AdvancedAugmentation:
    """Advanced augmentation for better generalization"""
    def __init__(self):
        self.train_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.2, scale=(0.02, 0.1))
        ])
        
        self.val_transforms = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((IM_SIZE, IM_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD)
        ])
    
    def __call__(self, frame, is_training=True):
        if is_training:
            return self.train_transforms(frame)
        else:
            return self.val_transforms(frame)


# Custom Dataset
class DeepfakeDataset(Dataset):
    def __init__(self, samples, augmentation, is_training=True):
        self.samples = samples
        self.augmentation = augmentation
        self.is_training = is_training
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        face_frames = sample['face_frames']
        label = sample['label']
        
        # Select SEQUENCE_LENGTH frames evenly distributed
        if len(face_frames) >= SEQUENCE_LENGTH:
            indices = np.linspace(0, len(face_frames) - 1, SEQUENCE_LENGTH, dtype=int)
            selected_faces = [face_frames[i] for i in indices]
        else:
            # Pad if needed
            selected_faces = face_frames.copy()
            while len(selected_faces) < SEQUENCE_LENGTH:
                selected_faces.append(selected_faces[-1])
        
        # Apply augmentation
        transformed_frames = []
        for frame in selected_faces[:SEQUENCE_LENGTH]:
            frame_tensor = self.augmentation(frame, is_training=self.is_training)
            transformed_frames.append(frame_tensor)
        
        sequence = torch.stack(transformed_frames)
        return sequence, label


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=0.25, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.CrossEntropyLoss(reduction='none')(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EarlyStopping:
    """Early stopping to prevent overfitting"""
    def __init__(self, patience=10, min_delta=0.001, mode='max'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False
        
        if self.mode == 'max':
            if score > self.best_score + self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        else:
            if score < self.best_score - self.min_delta:
                self.best_score = score
                self.counter = 0
            else:
                self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            return True
        
        return False


def load_training_data():
    """Load and prepare training data"""
    training_file = os.path.join(TRAINING_DATA_DIR, 'training_samples.pkl')
    
    if not os.path.exists(training_file):
        raise FileNotFoundError(f"No training data found at {training_file}")
    
    with open(training_file, 'rb') as f:
        training_data = pickle.load(f)
    
    logger.info(f"Loaded {len(training_data)} samples")
    
    # Count classes
    real_count = sum(1 for s in training_data if s['label'] == 1)
    fake_count = len(training_data) - real_count
    
    logger.info(f"Real samples: {real_count}, Fake samples: {fake_count}")
    
    if len(training_data) < 20:
        raise ValueError(f"Not enough training data. Have {len(training_data)}, need at least 20")
    
    return training_data, real_count, fake_count


def split_data(training_data, test_size=0.20, val_size=0.15):
    """Split data into train/val/test sets with stratification
    For 500 samples: ~325 train, ~75 val, ~100 test
    """
    labels = [s['label'] for s in training_data]
    
    # First split: train+val vs test
    train_val_data, test_data = train_test_split(
        training_data, test_size=test_size, stratify=labels, random_state=42
    )
    
    # Second split: train vs val
    train_val_labels = [s['label'] for s in train_val_data]
    val_ratio = val_size / (1 - test_size)
    train_data, val_data = train_test_split(
        train_val_data, test_size=val_ratio, stratify=train_val_labels, random_state=42
    )
    
    # Log class distribution
    train_real = sum(1 for s in train_data if s['label'] == 1)
    train_fake = len(train_data) - train_real
    val_real = sum(1 for s in val_data if s['label'] == 1)
    val_fake = len(val_data) - val_real
    test_real = sum(1 for s in test_data if s['label'] == 1)
    test_fake = len(test_data) - test_real
    
    logger.info(f"Data split - Train: {len(train_data)} (Real: {train_real}, Fake: {train_fake})")
    logger.info(f"           - Val: {len(val_data)} (Real: {val_real}, Fake: {val_fake})")
    logger.info(f"           - Test: {len(test_data)} (Real: {test_real}, Fake: {test_fake})")
    
    return train_data, val_data, test_data


def calculate_class_weights(real_count, fake_count):
    """Calculate class weights for imbalanced data"""
    total = real_count + fake_count
    weight_fake = total / (2 * fake_count)
    weight_real = total / (2 * real_count)
    return torch.tensor([weight_fake, weight_real], dtype=torch.float32).to(DEVICE)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
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
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Validating"):
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    
    # Calculate detailed metrics
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_labels, all_preds, average='weighted'
    )
    
    cm = confusion_matrix(all_labels, all_preds)
    
    return epoch_loss, epoch_acc, precision, recall, f1, cm


def save_checkpoint(model, optimizer, epoch, val_acc, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
    }, path)
    logger.info(f"Checkpoint saved: {path}")


def train_model():
    """Main training function"""
    logger.info("="*70)
    logger.info("STARTING TRAINING")
    logger.info("="*70)
    logger.info(f"Device: {DEVICE}")
    
    # Load data
    training_data, real_count, fake_count = load_training_data()
    
    # Split data
    train_data, val_data, test_data = split_data(training_data)
    
    # Create datasets
    augmentation = AdvancedAugmentation()
    train_dataset = DeepfakeDataset(train_data, augmentation, is_training=True)
    val_dataset = DeepfakeDataset(val_data, augmentation, is_training=False)
    test_dataset = DeepfakeDataset(test_data, augmentation, is_training=False)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)
    
    logger.info(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Initialize model
    model = DeepfakeDetectionModel(num_classes=2).to(DEVICE)
    
    # Calculate class weights
    class_weights = calculate_class_weights(real_count, fake_count)
    logger.info(f"Class weights - Fake: {class_weights[0]:.4f}, Real: {class_weights[1]:.4f}")
    
    # Loss function with class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # Alternative: Use Focal Loss for better handling of hard examples
    # criterion = FocalLoss(alpha=0.25, gamma=2.0)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=PATIENCE, min_delta=MIN_DELTA, mode='max')
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'lr': []
    }
    
    best_val_acc = 0.0
    best_model_path = os.path.join(MODELS_DIR, 'best_model.pt')  # Keep for backend compatibility
    training_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING STARTED")
    logger.info("="*70 + "\n")
    
    # Training loop
    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        logger.info("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, DEVICE)
        
        # Validate
        val_loss, val_acc, precision, recall, f1, cm = validate(model, val_loader, criterion, DEVICE)
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        logger.info(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        logger.info(f"Confusion Matrix:\n{cm}")
        
        # Save best model with accuracy in filename
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Save with accuracy in filename
            acc_filename = f'model_valacc_{val_acc:.2f}_epoch{epoch+1}_{training_timestamp}.pt'
            acc_model_path = os.path.join(MODELS_DIR, acc_filename)
            torch.save(model.state_dict(), acc_model_path)
            
            # Also save as best_model.pt for backend compatibility
            torch.save(model.state_dict(), best_model_path)
            
            logger.info(f"✓ New best model saved!")
            logger.info(f"  - {acc_filename}")
            logger.info(f"  - best_model.pt (for backend)")
        
        # Early stopping check
        if early_stopping(val_acc):
            logger.info(f"\nEarly stopping triggered after {epoch+1} epochs")
            break
    
    # Final evaluation on test set
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION ON TEST SET")
    logger.info("="*70)
    
    model.load_state_dict(torch.load(best_model_path))
    test_loss, test_acc, test_precision, test_recall, test_f1, test_cm = validate(
        model, test_loader, criterion, DEVICE
    )
    
    logger.info(f"\nTest Accuracy: {test_acc:.2f}%")
    logger.info(f"Test Precision: {test_precision:.4f}")
    logger.info(f"Test Recall: {test_recall:.4f}")
    logger.info(f"Test F1 Score: {test_f1:.4f}")
    logger.info(f"Test Confusion Matrix:\n{test_cm}")
    
    # Save final results
    results = {
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'test_cm': test_cm.tolist(),
        'history': history,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = os.path.join(LOGS_DIR, f'training_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pkl')
    with open(results_path, 'wb') as f:
        pickle.dump(results, f)
    
    logger.info(f"\nResults saved to: {results_path}")
    logger.info(f"Best model saved to: {best_model_path}")
    
    logger.info("\n" + "="*70)
    logger.info("TRAINING COMPLETED")
    logger.info("="*70)
    
    return model, results


if __name__ == '__main__':
    try:
        model, results = train_model()
        
        print("\n" + "="*70)
        print("🎉 TRAINING SUMMARY")
        print("="*70)
        print(f"Best Validation Accuracy: {results['best_val_acc']:.2f}%")
        print(f"Final Test Accuracy: {results['test_acc']:.2f}%")
        print(f"Test Precision: {results['test_precision']:.4f}")
        print(f"Test Recall: {results['test_recall']:.4f}")
        print(f"Test F1 Score: {results['test_f1']:.4f}")
        print("="*70)
        
        if results['test_acc'] >= 94:
            print("✓ TARGET ACCURACY ACHIEVED! (≥94%)")
        else:
            print(f"✗ Target not reached. Need {94 - results['test_acc']:.2f}% more accuracy.")
            print("\nTips to improve:")
            print("- Collect more diverse training data (50+ samples per class)")
            print("- Ensure balanced dataset (equal REAL and FAKE samples)")
            print("- Try different augmentation strategies")
            print("- Increase training epochs if not converged")
        
        print("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)