import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import face_recognition
from torchvision import transforms

class VideoDataset(Dataset):
    def __init__(self, video_names, labels=None, sequence_length=60, transform=None):
        self.video_names = video_names
        self.labels = labels
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)

        for i, frame in enumerate(self.frame_extract(video_path)):
            if self.labels is not None:
                faces = face_recognition.face_locations(frame)
                try:
                    top, right, bottom, left = faces[0]
                    frame = frame[top:bottom, left:right, :]
                except:
                    pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break

        frames = torch.stack(frames)
        frames = frames[:self.count]

        if self.labels is not None:
            temp_video = video_path.split('/')[-1]
            label = self.labels.iloc[(self.labels.loc[self.labels["file"] == temp_video].index.values[0]), 1]
            label = 0 if label == 'FAKE' else 1
            return frames, label
        else:
            return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

class ValidationDataset(Dataset):
    def __init__(self, video_names, sequence_length=60, transform=None):
        self.video_names = video_names
        self.transform = transform
        self.count = sequence_length

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        video_path = self.video_names[idx]
        frames = []
        a = int(100 / self.count)
        first_frame = np.random.randint(0, a)

        for i, frame in enumerate(self.frame_extract(video_path)):
            faces = face_recognition.face_locations(frame)
            try:
                top, right, bottom, left = faces[0]
                frame = frame[top:bottom, left:right, :]
            except:
                pass
            frames.append(self.transform(frame))
            if len(frames) == self.count:
                break

        frames = torch.stack(frames)
        frames = frames[:self.count]
        return frames.unsqueeze(0)

    def frame_extract(self, path):
        vidObj = cv2.VideoCapture(path)
        success = 1
        while success:
            success, image = vidObj.read()
            if success:
                yield image

def get_transforms():
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((im_size, im_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])

    return train_transforms, test_transforms
