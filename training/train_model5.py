import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms, models
import numpy as np
import cv2
import json
import base64
from pathlib import Path
import logging
import os
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import time

# --- ALIGNED CONFIGURATION FROM BACKEND ---
TRAINING_DATA_DIR = Path("training_data")
METADATA_FILE = TRAINING_DATA_DIR / "metadata.json"
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# ------------------------------------------

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Model Architecture (Copied from Prediction Backend) ---
class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1,
                 hidden_dim=2048, bidirectional=False):
        super(DeepfakeDetectionModel, self).__init__()
        # Using ResNext50 as the feature extractor
        model = models.resnext50_32x4d(weights='DEFAULT')
        # Remove the final classification layer and average pooling layer
        self.model = nn.Sequential(*list(model.children())[:-2]) 
        
        # LSTM layer for temporal sequence modeling
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, 
                            bidirectional=bidirectional, batch_first=True)
        self.dp = nn.Dropout(0.4)
        # Final linear layer for classification
        self.linear1 = nn.Linear(hidden_dim if not bidirectional else hidden_dim*2, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, C, H, W) -> (B, S, 3, 112, 112)
        batch_size, seq_length, c, h, w = x.shape
        # Reshape to treat all frames as a single batch for the feature extractor
        x = x.view(batch_size * seq_length, c, h, w)
        
        # Feature Extraction
        fmap = self.model(x)
        x = self.avgpool(fmap)
        
        # Reshape back to (batch_size, seq_length, feature_size) for LSTM
        x = x.view(batch_size, seq_length, -1)
        
        # LSTM processing
        x_lstm, _ = self.lstm(x, None)
        # Take the output from the last time step
        x_lstm = x_lstm[:, -1, :] 
        
        x_lstm = self.dp(x_lstm)
        out = self.linear1(x_lstm)
        return out


# --- Custom Dataset and Preprocessing ---
def decode_base64_to_frame(b64_frame: str) -> np.ndarray | None:
    """Decode a base64 string to an OpenCV BGR image array."""
    try:
        if ',' in b64_frame:
            b64_frame = b64_frame.split(',', 1)[1]
        nparr = np.frombuffer(base64.b64decode(b64_frame), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is not None:
            # Convert BGR to RGB for torchvision compatibility
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame
    except Exception as e:
        logger.warning(f"Error decoding base64 frame: {e}")
        return None

class DeepfakeDataset(Dataset):
    def __init__(self, metadata_path, transform=None):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        self.samples = metadata.get('samples', [])
        self.transform = transform
        
        # Filter samples to ensure sequence length is correct (sanity check)
        self.samples = [s for s in self.samples if len(s.get('frames', [])) == SEQUENCE_LENGTH]
        logger.info(f"Loaded {len(self.samples)} valid samples for training.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        label = sample_info['label'] # 0 for fake, 1 for real
        frames_b64 = sample_info['frames']
        
        # Load and transform each frame in the sequence
        sequence_tensors = []
        for b64 in frames_b64:
            frame_rgb = decode_base64_to_frame(b64)
            if frame_rgb is None:
                # Fallback: use a black image if decoding fails
                frame_rgb = np.zeros((IM_SIZE, IM_SIZE, 3), dtype=np.uint8) 
            
            # Apply transformations
            if self.transform:
                frame_tensor = self.transform(frame_rgb)
                sequence_tensors.append(frame_tensor)
        
        # Stack all tensors to form the sequence tensor (S, C, H, W)
        sequence = torch.stack(sequence_tensors)
        
        # Convert label to tensor
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return sequence, label_tensor

# --- Transformations (Data Augmentation for Training) ---
train_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])

val_transforms = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IM_SIZE, IM_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD)
])


# --- Training and Evaluation Functions ---
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    
    for sequences, labels in tqdm(dataloader, desc="Training"):
        sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(sequences)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_samples += labels.size(0)
        correct_predictions += (predicted == labels).sum().item()
        
    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct_predictions / total_samples
    return avg_loss, accuracy

def evaluate_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
        for sequences, labels in tqdm(dataloader, desc="Evaluating"):
            sequences, labels = sequences.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_predictions) * 100
    
    return avg_loss, accuracy, all_labels, all_predictions


# --- Main Training Script ---
def run_training(epochs=20, batch_size=16, learning_rate=1e-4, train_split_ratio=0.7):
    logger.info(f"--- Deepfake Detection Model Training ({'CUDA' if DEVICE.type == 'cuda' else 'CPU'}) ---")
    logger.info(f"Configuration: Epochs={epochs}, Batch Size={batch_size}, LR={learning_rate}")

    # 1. Load Data
    if not METADATA_FILE.exists():
        logger.error(f"Metadata file not found at: {METADATA_FILE}. Please run the data collection backend first.")
        return

    full_dataset = DeepfakeDataset(METADATA_FILE, transform=train_transforms)
    test_dataset = DeepfakeDataset(METADATA_FILE, transform=val_transforms)

    if len(full_dataset) == 0:
        logger.error("No valid samples found in metadata.json. Training cannot proceed.")
        return

    # 2. Split Data (70% Train, 30% Test)
    train_size = int(train_split_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size
    
    # We use indices to ensure consistency in splits across the two datasets (train/val transforms)
    train_indices, test_indices = random_split(range(len(full_dataset)), [train_size, test_size])
    
    # Create the final split datasets using the indices
    train_set = torch.utils.data.Subset(full_dataset, train_indices.indices)
    test_set = torch.utils.data.Subset(test_dataset, test_indices.indices)
    
    logger.info(f"Dataset Split: Train Samples={len(train_set)}, Test Samples={len(test_set)}")

    # 3. Create DataLoaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=os.cpu_count() // 2 if os.cpu_count() else 0)

    # 4. Initialize Model, Loss, and Optimizer
    model = DeepfakeDetectionModel(num_classes=2).to(DEVICE)
    # The weights for ResNeXt are pre-trained on ImageNet.
    # We can unfreeze the layers as we train a sequence model on top.
    
    # Loss function (Cross-Entropy for classification)
    criterion = nn.CrossEntropyLoss() 
    # Optimizer (Adam is a good general choice)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    # Scheduler to reduce LR when validation loss plateaus (a key technique)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)

    # Create directory to save models
    save_dir = Path("models5")
    save_dir.mkdir(exist_ok=True)
    best_accuracy = 0.0

    # 5. Training Loop
    logger.info("Starting training loop...")
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Evaluate on Test set
        test_loss, test_acc, _, _ = evaluate_model(model, test_loader, criterion)
        
        # Step the scheduler
        scheduler.step(test_loss)

        epoch_time = time.time() - start_time
        logger.info(f"Epoch {epoch+1}/{epochs}: Train Loss={train_loss:.4f}, Train Acc={train_acc:.2f}% | "
                    f"Test Loss={test_loss:.4f}, Test Acc={test_acc:.2f}% (Time: {epoch_time:.2f}s)")
        
        
        # 6. Save Best Model
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            model_save_path = save_dir / f"model_acc_{test_acc:.2f}_e{epoch+1}.pt"
            torch.save(model.state_dict(), model_save_path)
            logger.info(f"🏆 NEW BEST MODEL saved with Accuracy: {best_accuracy:.2f}% at {model_save_path}")
            
    # 7. Final Evaluation and Report
    logger.info("\n--- Final Evaluation on Test Set ---")
    final_test_loss, final_test_acc, all_labels, all_predictions = evaluate_model(model, test_loader, criterion)
    
    logger.info(f"Final Test Accuracy: {final_test_acc:.2f}%")
    
    if final_test_acc >= 94.0:
        logger.info("✅ SUCCESS: Target accuracy (>= 94.0%) achieved!")
    else:
        logger.warning(f"⚠️ TARGET NOT MET: Final accuracy is {final_test_acc:.2f}%. Try more data or epochs.")

    # Detailed report
    report = classification_report(all_labels, all_predictions, target_names=['Fake (0)', 'Real (1)'], digits=4)
    logger.info("\nClassification Report:\n" + report)
    logger.info("Confusion Matrix:\n" + str(confusion_matrix(all_labels, all_predictions)))


if __name__ == '__main__':
    # Adjust hyper-parameters here for optimal performance to reach 94%+
    # A larger batch size might speed up training on GPU.
    # More epochs or a lower learning rate might be needed for convergence.
    run_training(
        epochs=20,           # Increased epochs for better convergence
        batch_size=32,
        learning_rate=1e-4,  
        train_split_ratio=0.7 
    )