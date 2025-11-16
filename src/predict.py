"""
Prediction script for UCF101 action recognition models
"""

import argparse
import torch
import pickle
import numpy as np
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import UCF101SkeletonDataset
from src.models import BaselineLSTM, STGCN


def load_model(model_type, checkpoint_path, num_classes, device):
    """Load trained model from checkpoint"""
    # Create model
    if model_type == 'baseline':
        model = BaselineLSTM(
            num_classes=num_classes,
            input_dim=3,
            hidden_dim=128,
            num_layers=2,
            dropout=0.5,
            bidirectional=True
        )
    elif model_type == 'st_gcn':
        model = STGCN(
            num_classes=num_classes,
            in_channels=3,
            graph_layout='coco',
            num_stages=4,
            hidden_channels=[64, 64, 128, 256],
            dropout=0.5
        )
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    return model, checkpoint


def predict_single_sample(model, features, device, top_k=5):
    """
    Predict action for a single sample
    
    Args:
        model: Trained model
        features: Input features (T, V, C) numpy array
        device: Device to run inference on
        top_k: Number of top predictions to return
    
    Returns:
        top_k_predictions: List of (class_idx, probability) tuples
    """
    # Convert to tensor
    if isinstance(features, np.ndarray):
        features = torch.FloatTensor(features)
    
    # Add batch dimension: (T, V, C) -> (1, T, V, C)
    if len(features.shape) == 3:
        features = features.unsqueeze(0)
    
    features = features.to(device)
    
    # Predict
    with torch.no_grad():
        outputs = model(features)
        probs = torch.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probs, k=top_k, dim=1)
    
    # Convert to numpy
    top_probs = top_probs.cpu().numpy()[0]
    top_indices = top_indices.cpu().numpy()[0]
    
    # Return as list of tuples
    predictions = [(int(idx), float(prob)) for idx, prob in zip(top_indices, top_probs)]
    
    return predictions


def predict_from_pickle(model, model_type, checkpoint_path, data_path, 
                       split_name='test1', device='cpu', num_samples=None):
    """
    Predict on samples from pickle file
    
    Args:
        model: Trained model
        model_type: Type of model ('baseline' or 'st_gcn')
        checkpoint_path: Path to checkpoint (used to get num_classes)
        data_path: Path to pickle file
        split_name: Split to use for prediction
        device: Device to run inference on
        num_samples: Number of samples to predict (None for all)
    
    Returns:
        predictions: List of predictions
    """
    # Load data
    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    
    # Get split
    if 'train' in split_name:
        split_videos = set(data['split'][split_name])
    else:
        split_videos = set(data['split'][split_name])
    
    # Filter annotations
    annotations = [
        ann for ann in data['annotations']
        if ann['frame_dir'] in split_videos
    ]
    
    # Limit samples if requested
    if num_samples is not None:
        annotations = annotations[:num_samples]
    
    # Load model to get num_classes
    checkpoint = torch.load(checkpoint_path, map_location=device)
    num_classes = checkpoint.get('num_classes', 101)
    
    # Create dataset (for normalization)
    train_split_name = split_name.replace('test', 'train') if 'test' in split_name else split_name
    train_videos = set(data['split'][train_split_name])
    dataset = UCF101SkeletonDataset(
        data['annotations'],
        train_videos,  # Use train split for normalization parameters
        normalize=True
    )
    
    predictions = []
    
    print(f"Predicting on {len(annotations)} samples...")
    for ann in tqdm(annotations):
        # Preprocess sample
        keypoint = ann['keypoint'].astype(np.float32)
        keypoint_score = ann['keypoint_score'].astype(np.float32)
        
        # Handle multiple persons
        M = keypoint.shape[0]
        if M > 1:
            avg_scores = np.mean(keypoint_score, axis=(1, 2))
            best_person_idx = np.argmax(avg_scores)
            keypoint = keypoint[best_person_idx:best_person_idx+1]
            keypoint_score = keypoint_score[best_person_idx:best_person_idx+1]
        
        # Pad or truncate
        max_frames = 300
        T = keypoint.shape[1]
        if T > max_frames:
            keypoint = keypoint[:, :max_frames, :, :]
            keypoint_score = keypoint_score[:, :max_frames, :]
        elif T < max_frames:
            pad_length = max_frames - T
            pad_keypoint = np.zeros((1, pad_length, keypoint.shape[2], keypoint.shape[3]), 
                                   dtype=keypoint.dtype)
            pad_score = np.zeros((1, pad_length, keypoint_score.shape[2]), 
                               dtype=keypoint_score.dtype)
            keypoint = np.concatenate([keypoint, pad_keypoint], axis=1)
            keypoint_score = np.concatenate([keypoint_score, pad_score], axis=1)
        
        # Normalize
        img_shape = ann['img_shape']
        keypoint = keypoint.astype(np.float32)
        keypoint[:, :, :, 0] = keypoint[:, :, :, 0] / img_shape[1]
        keypoint[:, :, :, 1] = keypoint[:, :, :, 1] / img_shape[0]
        
        # Prepare features
        keypoint = keypoint[0]  # (T, V, C)
        keypoint_score = keypoint_score[0]  # (T, V)
        features = np.concatenate([
            keypoint,
            keypoint_score[:, :, np.newaxis]
        ], axis=-1)  # (T, V, 3)
        
        # Predict
        pred = predict_single_sample(model, features, device, top_k=5)
        
        predictions.append({
            'frame_dir': ann['frame_dir'],
            'true_label': ann['label'],
            'predictions': pred,
            'top_prediction': pred[0]
        })
    
    return predictions


def print_predictions(predictions, num_samples=10):
    """Print prediction results"""
    print("\n" + "="*80)
    print("PREDICTION RESULTS")
    print("="*80)
    
    correct = 0
    for i, pred in enumerate(predictions[:num_samples]):
        true_label = pred['true_label']
        top_pred_label, top_pred_prob = pred['top_prediction']
        is_correct = (true_label == top_pred_label)
        
        if is_correct:
            correct += 1
        
        status = "✓" if is_correct else "✗"
        print(f"\n{status} Sample {i+1}: {pred['frame_dir']}")
        print(f"  True Label: {true_label}")
        print(f"  Predicted: {top_pred_label} (confidence: {top_pred_prob*100:.2f}%)")
        print(f"  Top 3 predictions:")
        for j, (label, prob) in enumerate(pred['predictions'][:3]):
            print(f"    {j+1}. Class {label}: {prob*100:.2f}%")
    
    if len(predictions) > num_samples:
        print(f"\n... (showing first {num_samples} of {len(predictions)} samples)")
    
    # Calculate accuracy
    total_correct = sum(1 for p in predictions if p['true_label'] == p['top_prediction'][0])
    accuracy = total_correct / len(predictions) * 100
    print(f"\n" + "="*80)
    print(f"Overall Accuracy: {accuracy:.2f}% ({total_correct}/{len(predictions)})")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(description='Predict action recognition')
    parser.add_argument('--model', type=str, required=True,
                       choices=['baseline', 'st_gcn'],
                       help='Model to use for prediction')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str,
                       default='data/UCF101 Module 2 Deep Learning.pkl',
                       help='Path to dataset pickle file')
    parser.add_argument('--split', type=str, default='test1',
                       choices=['test1', 'test2', 'test3', 'train1', 'train2', 'train3'],
                       help='Dataset split to use')
    parser.add_argument('--num_samples', type=int, default=None,
                       help='Number of samples to predict (None for all)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save predictions (optional)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    num_classes = checkpoint.get('num_classes', 101)
    
    print(f"Loading {args.model} model...")
    model, checkpoint = load_model(args.model, args.checkpoint, num_classes, device)
    print(f"Model loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Predict
    predictions = predict_from_pickle(
        model=model,
        model_type=args.model,
        checkpoint_path=args.checkpoint,
        data_path=args.data_path,
        split_name=args.split,
        device=device,
        num_samples=args.num_samples
    )
    
    # Print results
    print_predictions(predictions, num_samples=min(10, len(predictions)))
    
    # Save predictions if requested
    if args.output:
        with open(args.output, 'wb') as f:
            pickle.dump(predictions, f)
        print(f"\nPredictions saved to {args.output}")


if __name__ == '__main__':
    main()

