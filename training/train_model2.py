import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import numpy as np
import cv2
import pickle
import os
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from datetime import datetime

# --- 1. Configuration (MUST MATCH BACKEND) ---
SEQUENCE_LENGTH = 20
IM_SIZE = 112
MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

TRAINING_DATA_DIR = 'training_data'
MODELS_DIR = 'models'
TRAINING_FILE = os.path.join(TRAINING_DATA_DIR, 'training_samples.pkl')
os.makedirs(MODELS_DIR, exist_ok=True)

# Training Hyperparameters
BATCH_SIZE = 8  # Adjust based on your GPU VRAM
NUM_EPOCHS = 30 
LEARNING_RATE = 1e-4 
PATIENCE = 5 # Early stopping patience

# --- 2. Model Architecture (Exact Copy) ---
class DeepfakeDetectionModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, 
                 hidden_dim=2048, bidirectional=False):
        super(DeepfakeDetectionModel, self).__init__()
        # Load ResNeXt-50 with default ImageNet weights
        model = models.resnext50_32x4d(weights='DEFAULT') 
        # Use all layers except the last pooling and linear layers for feature extraction
        self.model = nn.Sequential(*list(model.children())[:-2]) 
        
        self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, 
                            bidirectional=bidirectional, batch_first=True)
        self.dp = nn.Dropout(0.4)
        
        # Output layer
        self.linear1 = nn.Linear(hidden_dim if not bidirectional else hidden_dim*2, num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x shape: (batch_size, seq_length, C, H, W)
        batch_size, seq_length, c, h, w = x.shape
        
        # Reshape to treat all frames as a single batch for CNN
        x = x.view(batch_size * seq_length, c, h, w)
        
        # 1. Spatial Feature Extraction (ResNeXt)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        
        # Reshape to (batch_size, seq_length, latent_dim) for LSTM
        x = x.view(batch_size, seq_length, -1)
        
        # 2. Temporal Feature Analysis (LSTM)
        x_lstm, _ = self.lstm(x, None)
        
        # Use the feature vector from the last frame in the sequence
        x_lstm = x_lstm[:, -1, :]
        
        x_lstm = self.dp(x_lstm)
        
        # 3. Final Classification
        out = self.linear1(x_lstm)
        return out

# --- 3. Custom Dataset ---
class DeepfakeDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        face_frames = sample['face_frames']
        label = sample['label']
        
        # Pad or truncate face frames to match SEQUENCE_LENGTH
        if len(face_frames) > SEQUENCE_LENGTH:
            # Select evenly distributed frames (same logic as in the backend)
            indices = np.linspace(0, len(face_frames) - 1, SEQUENCE_LENGTH, dtype=int)
            face_frames = [face_frames[i] for i in indices]
        elif len(face_frames) < SEQUENCE_LENGTH:
            # Pad by repeating the last frame
            last_frame = face_frames[-1] if face_frames else np.zeros((IM_SIZE, IM_SIZE, 3), dtype=np.uint8)
            while len(face_frames) < SEQUENCE_LENGTH:
                face_frames.append(last_frame)
        
        # Apply transformation to each frame
        sequence_tensors = []
        for frame in face_frames:
            # frame is a BGR OpenCV array (from pickle), convert to RGB for transform
            if frame.size > 0:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
                sequence_tensors.append(self.transform(frame_rgb))
            else:
                 # Handle empty frames if they somehow slipped through
                 dummy_frame = np.zeros((IM_SIZE, IM_SIZE, 3), dtype=np.uint8)
                 sequence_tensors.append(self.transform(dummy_frame))
            
        # Stack all tensors to form the sequence tensor
        sequence = torch.stack(sequence_tensors)
        
        return sequence, torch.tensor(label, dtype=torch.long)

# --- 4. Training and Evaluation Functions ---
def train_epoch(model, dataloader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    for inputs, labels in tqdm(dataloader, desc="Training"):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct_predictions += torch.sum(preds == labels.data).item()
        total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

def evaluate_model(model, dataloader, criterion):
    model.eval()
    running_loss = 0.0
    correct_predictions = 0
    total_samples = 0
    
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc="Validation"):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct_predictions += torch.sum(preds == labels.data).item()
            total_samples += inputs.size(0)

    epoch_loss = running_loss / total_samples
    epoch_acc = correct_predictions / total_samples
    return epoch_loss, epoch_acc

# --- 5. Main Execution Block ---
def main():
    print(f"Device: {DEVICE}")

    # --- Data Loading and Splitting ---
    if not os.path.exists(TRAINING_FILE):
        print(f"Error: Training data file not found at {TRAINING_FILE}. Please collect data first.")
        return

    try:
        with open(TRAINING_FILE, 'rb') as f:
            all_data = pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle file: {e}")
        return
    
    # Filter for the first 250 real and 250 fake samples
    real_samples = [s for s in all_data if s['label'] == 1]
    fake_samples = [s for s in all_data if s['label'] == 0]
    
    real_samples = real_samples[:250]
    fake_samples = fake_samples[:250]
    
    if len(real_samples) < 250 or len(fake_samples) < 250:
         print(f"Warning: Found {len(real_samples)} REAL and {len(fake_samples)} FAKE samples. Training with available data.")

    combined_data = real_samples + fake_samples
    
    if not combined_data:
        print("Error: No valid training data found after filtering.")
        return

    # Split data into training and validation sets (80% train, 20% validation)
    # Stratify ensures equal proportions of real/fake in both splits
    train_data, val_data = train_test_split(
        combined_data, 
        test_size=0.2, 
        random_state=42, 
        shuffle=True, 
        stratify=[s['label'] for s in combined_data]
    )

    print(f"Total samples used: {len(combined_data)}")
    print(f"Training samples: {len(train_data)}")
    print(f"Validation samples: {len(val_data)}")

    # --- Transforms and Dataloaders ---
    # NOTE: Add data augmentation here for better results!
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        # Add a light random augmentation here to fight overfitting
        # transforms.RandomRotation(5),
        # transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    val_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IM_SIZE, IM_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD)
    ])

    train_dataset = DeepfakeDataset(train_data, transform=train_transforms)
    val_dataset = DeepfakeDataset(val_data, transform=val_transforms)

    # Note: Increase num_workers > 0 for faster data loading on systems with multiple CPU cores
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # --- Model, Loss, and Optimizer Setup ---
    model = DeepfakeDetectionModel().to(DEVICE)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    # --- Training Loop ---
    best_val_acc = 0.0
    epochs_no_improve = 0
    start_time = datetime.now()

    print("\nStarting Training with Checkpointing...")
    for epoch in range(NUM_EPOCHS):
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer)
        
        # Validation phase
        val_loss, val_acc = evaluate_model(model, val_loader, criterion)
        
        val_acc_percent = val_acc * 100
        train_acc_percent = train_acc * 100
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {train_loss:.4f} | Train Acc: {train_acc_percent:.2f}% | Val Loss: {val_loss:.4f} | **Val Acc: {val_acc_percent:.2f}%**")

        # Checkpoint and Save Logic (Saving all improving models by accuracy)
        if val_acc > best_val_acc:
            
            # 1. Update best accuracy and reset patience
            best_val_acc = val_acc
            epochs_no_improve = 0

            # 2. Define the accuracy-based filename (e.g., model_acc_95.52.pt)
            accuracy_filename = f'model_acc_{val_acc_percent:.2f}_e{epoch+1}.pt'
            checkpoint_save_path = os.path.join(MODELS_DIR, accuracy_filename)
            
            # 3. Save the model state dict to the unique path
            torch.save(model.state_dict(), checkpoint_save_path)

            # 4. Also save an alias as 'best_model.pt' for the highest one
            alias_save_path = os.path.join(MODELS_DIR, 'best_model.pt')
            torch.save(model.state_dict(), alias_save_path)

            print(f"  -> Model saved! Filename: **{accuracy_filename}**. New best accuracy: {best_val_acc*100:.2f}%")
        
        else:
            epochs_no_improve += 1
            print(f"  -> Validation accuracy did not improve. Patience: {epochs_no_improve}/{PATIENCE}")
        
        # Early Stopping check
        if epochs_no_improve >= PATIENCE:
            print(f"\nEarly stopping triggered after {epoch+1} epochs.")
            break

    end_time = datetime.now()
    print(f"\nTraining Complete in {(end_time - start_time).total_seconds():.2f} seconds.")
    print(f"Final Best Validation Accuracy: {best_val_acc*100:.2f}%")

if __name__ == '__main__':
    main()