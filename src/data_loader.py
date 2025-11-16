"""
Data loader for UCF101 skeleton dataset
Handles loading and preprocessing of skeleton annotations
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
import os


class UCF101SkeletonDataset(Dataset):
    """
    Dataset class for UCF101 skeleton data
    """
    
    def __init__(self, annotations, split_videos, normalize=True, max_frames=300):
        """
        Args:
            annotations: List of annotation dictionaries
            split_videos: Set of video identifiers for this split
            normalize: Whether to normalize keypoint coordinates
            max_frames: Maximum number of frames to pad/truncate to
        """
        self.max_frames = max_frames
        self.normalize = normalize
        
        # Filter annotations by split
        self.annotations = [
            ann for ann in annotations 
            if ann['frame_dir'] in split_videos
        ]
        
        # Normalize keypoints if requested
        if normalize:
            self._normalize_keypoints()
        
        print(f"Loaded {len(self.annotations)} samples")
    
    def _normalize_keypoints(self):
        """Normalize keypoint coordinates to [0, 1] range"""
        for ann in self.annotations:
            keypoint = ann['keypoint'].astype(np.float32)
            img_shape = ann['img_shape']  # (height, width)
            
            # Normalize by image dimensions
            if len(keypoint.shape) == 4:  # (M, T, V, C)
                keypoint[:, :, :, 0] = keypoint[:, :, :, 0] / img_shape[1]  # width
                keypoint[:, :, :, 1] = keypoint[:, :, :, 1] / img_shape[0]  # height
            
            ann['keypoint'] = keypoint
    
    def _pad_or_truncate(self, keypoint, keypoint_score):
        """
        Pad or truncate sequence to max_frames
        Args:
            keypoint: (M, T, V, C) array
            keypoint_score: (M, T, V) array
        Returns:
            Padded/truncated arrays
        """
        M, T, V, C = keypoint.shape
        T_current = T
        
        if T_current > self.max_frames:
            # Truncate
            keypoint = keypoint[:, :self.max_frames, :, :]
            keypoint_score = keypoint_score[:, :self.max_frames, :]
        elif T_current < self.max_frames:
            # Pad with zeros
            pad_length = self.max_frames - T_current
            pad_keypoint = np.zeros((M, pad_length, V, C), dtype=keypoint.dtype)
            pad_score = np.zeros((M, pad_length, V), dtype=keypoint_score.dtype)
            
            keypoint = np.concatenate([keypoint, pad_keypoint], axis=1)
            keypoint_score = np.concatenate([keypoint_score, pad_score], axis=1)
        
        return keypoint, keypoint_score
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations[idx]
        
        keypoint = ann['keypoint'].astype(np.float32)
        keypoint_score = ann['keypoint_score'].astype(np.float32)
        label = ann['label']
        
        # Handle multiple persons: take the first one (or average if needed)
        M = keypoint.shape[0]
        if M > 1:
            # Use the person with highest average confidence
            avg_scores = np.mean(keypoint_score, axis=(1, 2))  # (M,)
            best_person_idx = np.argmax(avg_scores)
            keypoint = keypoint[best_person_idx:best_person_idx+1]
            keypoint_score = keypoint_score[best_person_idx:best_person_idx+1]
        
        # Pad or truncate to max_frames
        keypoint, keypoint_score = self._pad_or_truncate(keypoint, keypoint_score)
        
        # Convert to (T, V, C) format (single person)
        keypoint = keypoint[0]  # (T, V, C)
        keypoint_score = keypoint_score[0]  # (T, V)
        
        # Combine coordinates with scores as input features
        # Option 1: Use only coordinates (x, y)
        # Option 2: Concatenate coordinates with scores
        # We'll use both: (T, V, 3) where last dim is [x, y, score]
        features = np.concatenate([
            keypoint,  # (T, V, 2)
            keypoint_score[:, :, np.newaxis]  # (T, V, 1)
        ], axis=-1)  # (T, V, 3)
        
        # Convert to torch tensors
        features = torch.FloatTensor(features)
        label = torch.LongTensor([label])[0]
        
        return features, label


def load_ucf101_data(data_path, split_name='train1'):
    """
    Load UCF101 skeleton data from pickle file
    
    Args:
        data_path: Path to pickle file
        split_name: Name of split to use (train1, train2, train3, test1, test2, test3)
    
    Returns:
        train_dataset, test_dataset, num_classes
    """
    print(f"Loading data from {data_path}...")
    
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get splits
    train_split_name = split_name if 'train' in split_name else 'train1'
    test_split_name = split_name.replace('train', 'test') if 'train' in split_name else 'test1'
    
    train_videos = set(data['split'][train_split_name])
    test_videos = set(data['split'][test_split_name])
    
    annotations = data['annotations']
    
    # Create datasets
    train_dataset = UCF101SkeletonDataset(
        annotations, 
        train_videos, 
        normalize=True
    )
    
    test_dataset = UCF101SkeletonDataset(
        annotations,
        test_videos,
        normalize=True
    )
    
    # Count unique classes
    all_labels = [ann['label'] for ann in annotations]
    num_classes = len(set(all_labels))
    
    print(f"Number of classes: {num_classes}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    return train_dataset, test_dataset, num_classes


def get_data_loaders(train_dataset, test_dataset, batch_size=32, num_workers=4):
    """
    Create DataLoaders for train and test sets
    
    Args:
        train_dataset: Training dataset
        test_dataset: Test dataset
        batch_size: Batch size
        num_workers: Number of worker processes
    
    Returns:
        train_loader, test_loader
    """
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader


if __name__ == '__main__':
    # Test data loading
    data_path = '../data/UCF101 Module 2 Deep Learning.pkl'
    train_dataset, test_dataset, num_classes = load_ucf101_data(data_path, 'train1')
    
    train_loader, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size=8)
    
    # Test a batch
    for features, labels in train_loader:
        print(f"Batch shape: {features.shape}")
        print(f"Labels shape: {labels.shape}")
        print(f"Sample label: {labels[0].item()}")
        break

