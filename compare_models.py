"""
Script to compare baseline and ST-GCN models
"""

import argparse
import os
import sys
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.evaluate import evaluate_model, print_evaluation_results
from src.data_loader import load_ucf101_data, get_data_loaders
from src.models import BaselineLSTM, STGCN
import torch


def find_checkpoint(checkpoint_path, model_type, split):
    """Find checkpoint file, trying multiple possible names"""
    if os.path.exists(checkpoint_path):
        return checkpoint_path
    
    # Try to find checkpoint with model class name
    split_dir = f"{model_type}_{split.replace('train', 'split')}"
    
    # Model class names
    model_class_names = {
        'baseline': 'BaselineLSTM',
        'st_gcn': 'STGCN'
    }
    model_class_name = model_class_names.get(model_type, model_type)
    
    possible_paths = [
        os.path.join('results', split_dir, f'{model_class_name}_best.pth'),
        os.path.join('results', split_dir, f'{model_type}_best.pth'),
        os.path.join('results', split_dir, 'best.pth'),
        checkpoint_path,  # Try original path again
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Found checkpoint at: {path}")
            return path
    
    raise FileNotFoundError(
        f"Checkpoint not found at {checkpoint_path}. "
        f"Tried: {possible_paths}. "
        f"Please train the model first or specify the correct checkpoint path."
    )


def compare_models(baseline_checkpoint, stgcn_checkpoint, data_path, 
                   split='train1', device='auto', output_dir='results/comparison'):
    """Compare baseline and ST-GCN models"""
    
    # Device setup
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    train_dataset, test_dataset, num_classes = load_ucf101_data(data_path, split)
    _, test_loader = get_data_loaders(train_dataset, test_dataset, batch_size=32)
    
    results = {}
    
    # Evaluate Baseline
    print("\n" + "="*60)
    print("EVALUATING BASELINE MODEL")
    print("="*60)
    baseline_model = BaselineLSTM(
        num_classes=num_classes,
        input_dim=3,
        hidden_dim=128,
        num_layers=2,
        dropout=0.5,
        bidirectional=True
    )
    # Find checkpoint automatically if needed
    baseline_checkpoint_path = find_checkpoint(baseline_checkpoint, 'baseline', split)
    baseline_checkpoint_data = torch.load(baseline_checkpoint_path, map_location=device)
    baseline_model.load_state_dict(baseline_checkpoint_data['model_state_dict'])
    baseline_model = baseline_model.to(device)
    
    baseline_results = evaluate_model(baseline_model, test_loader, device, num_classes)
    results['baseline'] = baseline_results
    print_evaluation_results(baseline_results)
    
    # Evaluate ST-GCN
    print("\n" + "="*60)
    print("EVALUATING ST-GCN MODEL")
    print("="*60)
    stgcn_model = STGCN(
        num_classes=num_classes,
        in_channels=3,
        graph_layout='coco',
        num_stages=4,
        hidden_channels=[64, 64, 128, 256],
        dropout=0.5
    )
    # Find checkpoint automatically if needed
    stgcn_checkpoint_path = find_checkpoint(stgcn_checkpoint, 'st_gcn', split)
    stgcn_checkpoint_data = torch.load(stgcn_checkpoint_path, map_location=device)
    stgcn_model.load_state_dict(stgcn_checkpoint_data['model_state_dict'])
    stgcn_model = stgcn_model.to(device)
    
    stgcn_results = evaluate_model(stgcn_model, test_loader, device, num_classes)
    results['st_gcn'] = stgcn_results
    print_evaluation_results(stgcn_results)
    
    # Comparison
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    
    baseline_acc = baseline_results['accuracy'] * 100
    stgcn_acc = stgcn_results['accuracy'] * 100
    improvement = stgcn_acc - baseline_acc
    
    print(f"Baseline Accuracy: {baseline_acc:.2f}%")
    print(f"ST-GCN Accuracy: {stgcn_acc:.2f}%")
    print(f"Improvement: {improvement:.2f}% ({improvement/baseline_acc*100:.1f}% relative)")
    
    baseline_f1 = baseline_results['classification_report']['weighted avg']['f1-score']
    stgcn_f1 = stgcn_results['classification_report']['weighted avg']['f1-score']
    
    print(f"\nBaseline F1-Score (weighted): {baseline_f1:.4f}")
    print(f"ST-GCN F1-Score (weighted): {stgcn_f1:.4f}")
    print(f"Improvement: {stgcn_f1 - baseline_f1:.4f}")
    
    # Model sizes
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    stgcn_params = sum(p.numel() for p in stgcn_model.parameters())
    
    print(f"\nBaseline Parameters: {baseline_params:,}")
    print(f"ST-GCN Parameters: {stgcn_params:,}")
    print(f"Ratio: {stgcn_params/baseline_params:.2f}x")
    
    # Save comparison results
    os.makedirs(output_dir, exist_ok=True)
    
    comparison_data = {
        'baseline': {
            'accuracy': baseline_acc,
            'f1_score': baseline_f1,
            'parameters': baseline_params,
            'loss': baseline_results['loss']
        },
        'st_gcn': {
            'accuracy': stgcn_acc,
            'f1_score': stgcn_f1,
            'parameters': stgcn_params,
            'loss': stgcn_results['loss']
        },
        'improvement': {
            'accuracy': improvement,
            'accuracy_relative': improvement/baseline_acc*100,
            'f1_score': stgcn_f1 - baseline_f1
        }
    }
    
    with open(os.path.join(output_dir, 'comparison.json'), 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    # Plot comparison
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    models = ['Baseline\n(LSTM)', 'ST-GCN']
    accuracies = [baseline_acc, stgcn_acc]
    colors = ['#3498db', '#2ecc71']
    
    bars = axes[0].bar(models, accuracies, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[0].set_ylabel('Accuracy (%)', fontsize=12)
    axes[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    axes[0].set_ylim([0, 100])
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width()/2., height,
                    f'{acc:.2f}%',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # F1-Score comparison
    f1_scores = [baseline_f1, stgcn_f1]
    bars = axes[1].bar(models, f1_scores, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
    axes[1].set_ylabel('F1-Score (Weighted)', fontsize=12)
    axes[1].set_title('Model F1-Score Comparison', fontsize=14, fontweight='bold')
    axes[1].grid(axis='y', alpha=0.3)
    axes[1].set_ylim([0, 1])
    
    # Add value labels on bars
    for bar, f1 in zip(bars, f1_scores):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height,
                    f'{f1:.4f}',
                    ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'model_comparison.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison results saved to {output_dir}")
    print("="*60)
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Compare baseline and ST-GCN models')
    parser.add_argument('--baseline_checkpoint', type=str, required=True,
                       help='Path to baseline model checkpoint')
    parser.add_argument('--stgcn_checkpoint', type=str, required=True,
                       help='Path to ST-GCN model checkpoint')
    parser.add_argument('--data_path', type=str,
                       default='data/UCF101 Module 2 Deep Learning.pkl',
                       help='Path to dataset pickle file')
    parser.add_argument('--split', type=str, default='train1',
                       choices=['train1', 'train2', 'train3'],
                       help='Dataset split to use')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='results/comparison',
                       help='Directory to save comparison results')
    
    args = parser.parse_args()
    
    compare_models(
        baseline_checkpoint=args.baseline_checkpoint,
        stgcn_checkpoint=args.stgcn_checkpoint,
        data_path=args.data_path,
        split=args.split,
        device=args.device,
        output_dir=args.output_dir
    )


if __name__ == '__main__':
    main()

