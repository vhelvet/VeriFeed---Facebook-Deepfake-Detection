# Deepfake Video Detection

A standalone Python package for detecting deepfake videos using ResNext50 CNN + LSTM architecture.

## Features

- Deepfake detection using transfer learning with ResNext50
- Temporal analysis with LSTM for video sequences
- Face extraction and preprocessing
- Model training and prediction
- Confidence scoring for predictions

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Training
```python
from deepfake_detection.train import train_model
from deepfake_detection.data_loader import VideoDataset

# Prepare your dataset
train_dataset = VideoDataset(video_paths, labels, sequence_length=20)
train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# Train the model
model = train_model(train_loader, num_epochs=20, learning_rate=1e-5)
```

### Prediction
```python
from deepfake_detection.predict import predict_video
from deepfake_detection.model import Model

# Load trained model
model = Model(num_classes=2)
model.load_state_dict(torch.load('path/to/model.pt'))

# Predict on a video
result, confidence = predict_video('path/to/video.mp4', model, sequence_length=20)
print(f"Prediction: {'REAL' if result == 1 else 'FAKE'}, Confidence: {confidence:.2f}%")
```

## Model Architecture

- **Feature Extraction**: ResNext50-32x4d (pre-trained)
- **Temporal Analysis**: LSTM with 2048 hidden units
- **Classification**: Linear layer with dropout

## Requirements

- Python 3.7+
- PyTorch 1.8+
- OpenCV
- face_recognition
- NumPy
- Matplotlib

## Dataset

The model was trained on:
- FaceForensics++
- Celeb-DF
- Deepfake Detection Challenge (DFDC) datasets

## Performance

| Sequence Length | Accuracy |
|----------------|----------|
| 10 frames      | 84.21%   |
| 20 frames      | 87.79%   |
| 40 frames      | 89.35%   |
| 60 frames      | 90.59%   |
| 80 frames      | 91.50%   |
| 100 frames     | 93.59%   |

## License

GPL v3
