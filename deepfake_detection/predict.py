import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from torch import nn
from torchvision import transforms
from .data_loader import ValidationDataset

def predict_video(video_path, model, sequence_length=20, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predict whether a video is real or fake
    
    Args:
        video_path (str): Path to the video file
        model: Trained deepfake detection model
        sequence_length (int): Number of frames to use for prediction
        device (str): Device to run inference on ('cuda' or 'cpu')
    
    Returns:
        tuple: (prediction, confidence) where prediction is 0 for FAKE, 1 for REAL
    """
    # Set up transforms
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    # Create dataset and loader
    dataset = ValidationDataset([video_path], sequence_length=sequence_length, transform=train_transforms)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Softmax for confidence calculation
    sm = nn.Softmax(dim=1)
    
    with torch.no_grad():
        # Get the video data
        video_data = dataset[0].to(device)
        
        # Make prediction
        _, logits = model(video_data)
        logits = sm(logits)
        
        # Get prediction and confidence
        _, prediction = torch.max(logits, 1)
        confidence = logits[:, int(prediction.item())].item() * 100
        
        return int(prediction.item()), confidence

def predict_batch(video_paths, model, sequence_length=20, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Predict multiple videos in batch
    
    Args:
        video_paths (list): List of video file paths
        model: Trained deepfake detection model
        sequence_length (int): Number of frames to use for prediction
        device (str): Device to run inference on
    
    Returns:
        list: List of tuples (prediction, confidence) for each video
    """
    results = []
    for video_path in video_paths:
        result = predict_video(video_path, model, sequence_length, device)
        results.append(result)
    return results

def generate_heatmap(video_path, model, frame_index=0, sequence_length=20, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """
    Generate heatmap visualization for a video frame
    
    Args:
        video_path (str): Path to the video file
        model: Trained deepfake detection model
        frame_index (int): Index of frame to generate heatmap for
        sequence_length (int): Number of frames
        device (str): Device to run inference on
    
    Returns:
        numpy.ndarray: Heatmap image
    """
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    
    # Inverse normalization for visualization
    inv_normalize = transforms.Normalize(
        mean=-1 * np.divide(mean, std),
        std=np.divide([1, 1, 1], std)
    )
    
    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    
    dataset = ValidationDataset([video_path], sequence_length=sequence_length, transform=train_transforms)
    model = model.to(device)
    model.eval()
    sm = nn.Softmax(dim=1)
    
    with torch.no_grad():
        video_data = dataset[0].to(device)
        fmap, logits = model(video_data)
        logits = sm(logits)
        
        # Get weights from the linear layer
        weight_softmax = model.linear1.weight.detach().cpu().numpy()
        idx = np.argmax(logits.detach().cpu().numpy())
        
        # Generate heatmap
        bz, nc, h, w = fmap.shape
        out = np.dot(fmap[frame_index].detach().cpu().numpy().reshape((nc, h * w)).T, weight_softmax[idx, :].T)
        predict = out.reshape(h, w)
        predict = predict - np.min(predict)
        predict_img = predict / np.max(predict)
        predict_img = np.uint8(255 * predict_img)
        
        # Resize and apply colormap
        out = cv2.resize(predict_img, (im_size, im_size))
        heatmap = cv2.applyColorMap(out, cv2.COLORMAP_JET)
        
        # Convert tensor back to image
        image = video_data[0, -1, :, :, :].to("cpu").clone().detach()
        image = image.squeeze()
        image = inv_normalize(image)
        image = image.numpy()
        image = image.transpose(1, 2, 0)
        image = image.clip(0, 1)
        
        # Combine heatmap with original image
        result = heatmap * 0.5 + image * 0.8 * 255
        result = np.clip(result, 0, 255).astype(np.uint8)
        
        return result

def validate_video(video_path, transform, sequence_length=20):
    """
    Validate if a video is corrupted and can be processed
    
    Args:
        video_path (str): Path to the video file
        transform: Transform to apply to frames
        sequence_length (int): Number of frames to check
    
    Returns:
        bool: True if video is valid, False otherwise
    """
    try:
        count = sequence_length
        frames = []
        
        vidObj = cv2.VideoCapture(video_path)
        success = 1
        frame_count = 0
        
        while success and frame_count < count:
            success, image = vidObj.read()
            if success:
                frames.append(transform(image))
                frame_count += 1
        
        vidObj.release()
        
        if len(frames) < count:
            return False
        
        frames = torch.stack(frames)
        frames = frames[:count]
        return True
        
    except Exception as e:
        print(f"Error validating video {video_path}: {e}")
        return False
