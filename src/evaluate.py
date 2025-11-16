"""
Evaluation script for UCF101 action recognition models
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_ucf101_data, get_data_loaders
from src.models import BaselineLSTM, STGCN


def evaluate_model(model, test_loader, device, num_classes=101, class_names=None):
    """Evaluate model on test set"""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc='Evaluating')
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Get predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
            running_loss += loss.item()
    
    # Calculate metrics
    accuracy = accuracy_score(all_labels, all_preds)
    avg_loss = running_loss / len(test_loader)
    
    # Classification report
    if class_names is None:
        class_names = [f'Class {i}' for i in range(num_classes)]
    
    report = classification_report(
        all_labels, all_preds,
        target_names=class_names[:num_classes],
        output_dict=True,
        zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    return {
        'accuracy': accuracy,
        'loss': avg_loss,
        'predictions': all_preds,
        'labels': all_labels,
        'probabilities': all_probs,
        'classification_report': report,
        'confusion_matrix': cm
    }


def plot_confusion_matrix(cm, class_names, save_path, num_classes=101, top_n=20):
    """Plot confusion matrix"""
    # For large number of classes, show top N classes by frequency
    if num_classes > top_n:
        # Calculate class frequencies
        class_counts = cm.sum(axis=1)
        top_indices = np.argsort(class_counts)[-top_n:]
        
        # Select top N classes
        cm_subset = cm[np.ix_(top_indices, top_indices)]
        class_names_subset = [class_names[i] for i in top_indices]
        title = f'Confusion Matrix (Top {top_n} Classes)'
    else:
        cm_subset = cm
        class_names_subset = class_names
        title = 'Confusion Matrix'
    
    # Normalize confusion matrix
    cm_normalized = cm_subset.astype('float') / (cm_subset.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=class_names_subset, yticklabels=class_names_subset,
                cbar_kws={'label': 'Normalized Frequency'})
    plt.title(title, fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_path}")


def print_evaluation_results(results):
    """Print evaluation results"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Accuracy: {results['accuracy']*100:.2f}%")
    print(f"Average Loss: {results['loss']:.4f}")
    print("\n" + "-"*60)
    
    # Classification report
    report = results['classification_report']
    print("\nClassification Report:")
    print(f"  Macro Avg Precision: {report['macro avg']['precision']:.4f}")
    print(f"  Macro Avg Recall: {report['macro avg']['recall']:.4f}")
    print(f"  Macro Avg F1-Score: {report['macro avg']['f1-score']:.4f}")
    print(f"  Weighted Avg Precision: {report['weighted avg']['precision']:.4f}")
    print(f"  Weighted Avg Recall: {report['weighted avg']['recall']:.4f}")
    print(f"  Weighted Avg F1-Score: {report['weighted avg']['f1-score']:.4f}")
    
    print("\n" + "="*60)


def main():
    parser = argparse.ArgumentParser(description='Evaluate action recognition model')
    parser.add_argument('--model', type=str, required=True,
                       choices=['baseline', 'st_gcn'],
                       help='Model to evaluate')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str,
                       default='data/UCF101 Module 2 Deep Learning.pkl',
                       help='Path to dataset pickle file')
    parser.add_argument('--split', type=str, default='train1',
                       choices=['train1', 'train2', 'train3'],
                       help='Dataset split to use')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_dataset, test_dataset, num_classes = load_ucf101_data(
        args.data_path, args.split
    )
    
    _, test_loader = get_data_loaders(
        train_dataset, test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    # Create model
    print(f"Creating {args.model} model...")
    if args.model == 'baseline':
        model = BaselineLSTM(
            num_classes=num_classes,
            input_dim=3,
            hidden_dim=128,
            num_layers=2,
            dropout=0.5,
            bidirectional=True
        )
    elif args.model == 'st_gcn':
        model = STGCN(
            num_classes=num_classes,
            in_channels=3,
            graph_layout='coco',
            num_stages=4,
            hidden_channels=[64, 64, 128, 256],
            dropout=0.5
        )
    
    # Load checkpoint - try to find it if path doesn't exist
    checkpoint_path = args.checkpoint
    if not os.path.exists(checkpoint_path):
        # Try to find checkpoint with model class name
        model_class_name = model.__class__.__name__
        split_dir = f"{args.model}_{args.split.replace('train', 'split')}"
        possible_paths = [
            os.path.join('results', split_dir, f'{model_class_name}_best.pth'),
            os.path.join('results', split_dir, f'{args.model}_best.pth'),
            os.path.join('results', split_dir, 'best.pth'),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                checkpoint_path = path
                print(f"Found checkpoint at: {checkpoint_path}")
                break
        else:
            raise FileNotFoundError(
                f"Checkpoint not found at {args.checkpoint}. "
                f"Tried: {possible_paths}. "
                f"Please train the model first or specify the correct checkpoint path."
            )
    
    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Checkpoint loaded from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc' in checkpoint:
        print(f"Validation accuracy from training: {checkpoint['val_acc']:.2f}%")
    
    # Evaluate
    print("\nEvaluating model...")
    results = evaluate_model(model, test_loader, device, num_classes=num_classes)
    
    # Print results
    print_evaluation_results(results)
    
    # Save results
    output_dir = os.path.join(args.output_dir, f"{args.model}_{args.split.replace('train', 'split')}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Save classification report
    import json
    report_path = os.path.join(output_dir, 'classification_report.json')
    with open(report_path, 'w') as f:
        json.dump(results['classification_report'], f, indent=2)
    
    # Plot confusion matrix
    cm_path = os.path.join(output_dir, 'confusion_matrix.png')
    class_names = [f'Class {i}' for i in range(num_classes)]
    plot_confusion_matrix(results['confusion_matrix'], class_names, cm_path, num_classes)
    
    # Save predictions
    predictions_path = os.path.join(output_dir, 'predictions.npz')
    np.savez(predictions_path,
             predictions=results['predictions'],
             labels=results['labels'],
             probabilities=results['probabilities'])
    
    print(f"\nEvaluation results saved to {output_dir}")


if __name__ == '__main__':
    main()

