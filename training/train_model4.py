"""
VeriFeed Advanced Training Script - Target Accuracy: 94-100%
70% Training / 30% Testing Split
Features: Advanced augmentation, learning rate scheduling, early stopping, gradient clipping
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms, models
import numpy as np
import json
import base64
import cv2
import logging
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configuration
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Training Hyperparameters
BATCH_SIZE = 8
EPOCHS = 40
LEARNING_RATE = 0.0001
WEIGHT_DECAY = 1e-5
PATIENCE = 10
MIN_DELTA = 0.001

# Paths
METADATA_FILE = Path("training_data/metadata.json")
MODELS_DIR = Path("models4")
MODELS_DIR.mkdir(exist_ok=True)

# Advanced Data Augmentation
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
    transforms.RandomPerspective(distortion_scale=0.2, p=0.3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
    transforms.RandomErasing(p=0.3, scale=(0.02, 0.15))
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

# Model Architecture with Improved Design
class ImprovedDeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=2,
                 hidden_dim=2048, bidirectional=True, dropout=0.5):
        super(ImprovedDeepfakeDetectionModel, self).__init__()
        
        # ResNeXt50 backbone with pretrained weights
        model = models.resnext50_32x4d(weights='IMAGENET1K_V2')
        self.model = nn.Sequential(*list(model.children())[:-2])
        
        # Freeze early layers for better generalization
        for param in list(self.model.parameters())[:100]:
            param.requires_grad = False
        
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        
        # Bidirectional LSTM with multiple layers
        self.lstm = nn.LSTM(
            latent_dim, 
            hidden_dim, 
            lstm_layers,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0,
            batch_first=False
        )
        
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        # Advanced classifier head
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(lstm_output_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.5),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(dropout * 0.3),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, x):
        batch_size, seq_length, c, h, w = x.shape
        
        # Process through CNN
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, -1)
        
        # Process through LSTM
        x = x.permute(1, 0, 2)  # (seq, batch, features)
        x_lstm, _ = self.lstm(x)
        x_lstm = x_lstm[-1]  # Take last output
        
        # Classification
        out = self.classifier(x_lstm)
        return out


# Custom Dataset
class DeepfakeDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        label = sample['label']
        frames_b64 = sample['frames']
        
        # Decode frames
        frames = []
        for frame_b64 in frames_b64:
            try:
                img_data = base64.b64decode(frame_b64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
            except Exception as e:
                logger.error(f"Error decoding frame: {e}")
                # Use black frame as fallback
                frames.append(np.zeros((IM_SIZE, IM_SIZE, 3), dtype=np.uint8))
        
        # Apply transforms
        if self.transform:
            frames = [self.transform(frame) for frame in frames]
        
        sequence = torch.stack(frames)
        return sequence, label


def load_training_data():
    """Load and split training data"""
    logger.info("Loading training data...")
    
    if not METADATA_FILE.exists():
        raise FileNotFoundError(f"Metadata file not found: {METADATA_FILE}")
    
    with open(METADATA_FILE, 'r') as f:
        metadata = json.load(f)
    
    samples = metadata['samples']
    
    if len(samples) == 0:
        raise ValueError("No training samples found!")
    
    logger.info(f"Total samples: {len(samples)}")
    logger.info(f"Real samples: {metadata['real_samples']}")
    logger.info(f"Fake samples: {metadata['fake_samples']}")
    
    # Split into train and test (70/30)
    train_samples, test_samples = train_test_split(
        samples, 
        test_size=0.3, 
        random_state=42,
        stratify=[s['label'] for s in samples]
    )
    
    logger.info(f"Training samples: {len(train_samples)}")
    logger.info(f"Testing samples: {len(test_samples)}")
    
    return train_samples, test_samples, metadata


def get_class_weights(samples):
    """Calculate class weights for balanced training"""
    labels = [s['label'] for s in samples]
    class_counts = np.bincount(labels)
    total = len(labels)
    weights = total / (len(class_counts) * class_counts)
    return torch.FloatTensor(weights).to(DEVICE)


def train_epoch(model, dataloader, criterion, optimizer, scaler, device):
    """Train for one epoch with mixed precision"""
    model.train()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for sequences, labels in pbar:
        sequences = sequences.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        # Mixed precision training
        with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
            outputs = model(sequences)
            loss = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        
        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        scaler.step(optimizer)
        scaler.update()
        
        running_loss += loss.item()
        
        _, predicted = torch.max(outputs.data, 1)
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
        pbar = tqdm(dataloader, desc="Validation")
        for sequences, labels in pbar:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
    
    epoch_loss = running_loss / len(dataloader)
    epoch_acc = accuracy_score(all_labels, all_preds) * 100
    precision = precision_score(all_labels, all_preds, average='weighted') * 100
    recall = recall_score(all_labels, all_preds, average='weighted') * 100
    f1 = f1_score(all_labels, all_preds, average='weighted') * 100
    
    return epoch_loss, epoch_acc, precision, recall, f1, all_labels, all_preds


def plot_training_history(history, save_path):
    """Plot and save training history"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train Loss')
    axes[0, 0].plot(history['val_loss'], label='Val Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train Acc')
    axes[0, 1].plot(history['val_acc'], label='Val Acc')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    
    # Precision, Recall, F1
    axes[1, 0].plot(history['precision'], label='Precision')
    axes[1, 0].plot(history['recall'], label='Recall')
    axes[1, 0].plot(history['f1'], label='F1-Score')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Score (%)')
    axes[1, 0].set_title('Validation Metrics')
    axes[1, 0].legend()
    axes[1, 0].grid(True)
    
    # Learning Rate
    axes[1, 1].plot(history['lr'], label='Learning Rate')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].set_yscale('log')
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrix(labels, preds, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Fake', 'Real'],
                yticklabels=['Fake', 'Real'])
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def main():
    logger.info("="*70)
    logger.info("🚀 VeriFeed Advanced Training - Target Accuracy: 94-100%")
    logger.info("="*70)
    logger.info(f"Device: {DEVICE}")
    logger.info(f"Batch Size: {BATCH_SIZE}")
    logger.info(f"Learning Rate: {LEARNING_RATE}")
    logger.info(f"Epochs: {EPOCHS}")
    logger.info(f"Train/Test Split: 70/30")
    logger.info("="*70)
    
    # Load data
    train_samples, test_samples, metadata = load_training_data()
    
    # Create datasets
    train_dataset = DeepfakeDataset(train_samples, transform=train_transforms)
    test_dataset = DeepfakeDataset(test_samples, transform=val_transforms)
    
    # Calculate class weights for balanced training
    class_weights = get_class_weights(train_samples)
    logger.info(f"Class weights: {class_weights}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        persistent_workers=True
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=False,
        num_workers=4,
        pin_memory=True if DEVICE.type == 'cuda' else False,
        persistent_workers=True
    )
    
    # Initialize model
    model = ImprovedDeepfakeDetectionModel(
        num_classes=2,
        lstm_layers=2,
        bidirectional=True,
        dropout=0.5
    ).to(DEVICE)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=LEARNING_RATE, 
        weight_decay=WEIGHT_DECAY
    )
    
    # Learning rate scheduler - ReduceLROnPlateau for adaptive learning
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='max', 
        factor=0.5, 
        patience=5, 
        min_lr=1e-7
    )
    
    # Mixed precision training
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE.type == 'cuda'))
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': [],
        'precision': [], 'recall': [], 'f1': [],
        'lr': []
    }
    
    # Early stopping
    best_acc = 0.0
    best_model_path = None
    patience_counter = 0
    
    logger.info("\n" + "="*70)
    logger.info("Starting Training...")
    logger.info("="*70 + "\n")
    
    for epoch in range(EPOCHS):
        logger.info(f"\nEpoch {epoch+1}/{EPOCHS}")
        logger.info("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, DEVICE
        )
        
        # Validate
        val_loss, val_acc, precision, recall, f1, val_labels, val_preds = validate(
            model, test_loader, criterion, DEVICE
        )
        
        # Update learning rate
        scheduler.step(val_acc)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['precision'].append(precision)
        history['recall'].append(recall)
        history['f1'].append(f1)
        history['lr'].append(current_lr)
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        logger.info(f"Precision: {precision:.2f}% | Recall: {recall:.2f}% | F1: {f1:.2f}%")
        logger.info(f"Learning Rate: {current_lr:.2e}")
        
        # Save best model
        if val_acc > best_acc + MIN_DELTA:
            best_acc = val_acc
            patience_counter = 0
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            model_filename = f"model_acc_{val_acc:.2f}_epoch{epoch+1}_{timestamp}.pt"
            model_path = MODELS_DIR / model_filename
            
            torch.save(model.state_dict(), model_path)
            logger.info(f"✓ New best model saved: {model_filename}")
            
            best_model_path = model_path
            
            # Save confusion matrix for best model
            cm_path = MODELS_DIR / f"confusion_matrix_{val_acc:.2f}.png"
            plot_confusion_matrix(val_labels, val_preds, cm_path)
            
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{PATIENCE}")
        
        # Early stopping
        if patience_counter >= PATIENCE:
            logger.info(f"\n⚠️ Early stopping triggered after {epoch+1} epochs")
            break
    
    # Save training history plot
    history_path = MODELS_DIR / "training_history.png"
    plot_training_history(history, history_path)
    
    # Final evaluation on best model
    logger.info("\n" + "="*70)
    logger.info("FINAL EVALUATION ON BEST MODEL")
    logger.info("="*70)
    
    if best_model_path:
        model.load_state_dict(torch.load(best_model_path))
        final_loss, final_acc, final_prec, final_rec, final_f1, final_labels, final_preds = validate(
            model, test_loader, criterion, DEVICE
        )
        
        logger.info(f"Best Model: {best_model_path.name}")
        logger.info(f"Final Test Accuracy: {final_acc:.2f}%")
        logger.info(f"Final Precision: {final_prec:.2f}%")
        logger.info(f"Final Recall: {final_rec:.2f}%")
        logger.info(f"Final F1-Score: {final_f1:.2f}%")
        
        # Detailed classification report
        cm = confusion_matrix(final_labels, final_preds)
        logger.info("\nConfusion Matrix:")
        logger.info(f"{'':>10} {'Pred Fake':>12} {'Pred Real':>12}")
        logger.info(f"{'True Fake':<10} {cm[0][0]:>12} {cm[0][1]:>12}")
        logger.info(f"{'True Real':<10} {cm[1][0]:>12} {cm[1][1]:>12}")
        
        if final_acc >= 94:
            logger.info("\n🎉 TARGET ACCURACY ACHIEVED! 🎉")
        else:
            logger.info(f"\n⚠️ Target accuracy not reached. Current: {final_acc:.2f}%")
            logger.info("Consider: More training data, longer training, or hyperparameter tuning")
    
    logger.info("\n" + "="*70)
    logger.info("Training Complete!")
    logger.info("="*70)


if __name__ == "__main__":
    main()