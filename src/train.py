"""
Training script for UCF101 action recognition models
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import sys
from tqdm import tqdm
import numpy as np
from datetime import datetime

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_loader import load_ucf101_data, get_data_loaders
from src.models import BaselineLSTM, STGCN


def train_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
    for batch_idx, (features, labels) in enumerate(pbar):
        features = features.to(device)
        labels = labels.to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': running_loss / (batch_idx + 1),
            'acc': 100. * correct / total
        })
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device, epoch):
    """Validate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]')
        for features, labels in pbar:
            features = features.to(device)
            labels = labels.to(device)
            
            # Forward pass
            outputs = model(features)
            loss = criterion(outputs, labels)
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': running_loss / len(val_loader),
                'acc': 100. * correct / total
            })
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = 100. * correct / total
    
    return epoch_loss, epoch_acc, all_preds, all_labels


def train(model, train_loader, val_loader, num_epochs, device, 
          learning_rate=0.001, weight_decay=1e-4, checkpoint_dir='results'):
    """Train model"""
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Create checkpoint directory
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # TensorBoard writer
    log_dir = os.path.join(checkpoint_dir, 'tensorboard_logs')
    writer = SummaryWriter(log_dir)
    
    # Training history
    best_val_acc = 0.0
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []
    
    model_name = model.__class__.__name__
    print(f"\nTraining {model_name}")
    print(f"Device: {device}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, 
                                           optimizer, device, epoch)
        
        # Validate
        val_loss, val_acc, _, _ = validate(model, val_loader, criterion, device, epoch)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Save history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # TensorBoard logging
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Accuracy/Train', train_acc, epoch)
        writer.add_scalar('Accuracy/Val', val_acc, epoch)
        writer.add_scalar('LearningRate', optimizer.param_groups[0]['lr'], epoch)
        
        # Print epoch results
        print(f"Epoch [{epoch+1}/{num_epochs}]")
        print(f"  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_best.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
            }, checkpoint_path)
            print(f"  ✓ Saved best model (Val Acc: {val_acc:.2f}%)")
        
        print("-" * 60)
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'{model_name}_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'train_acc': train_acc,
            }, checkpoint_path)
    
    writer.close()
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc
    }
    
    import pickle
    history_path = os.path.join(checkpoint_dir, f'{model_name}_history.pkl')
    with open(history_path, 'wb') as f:
        pickle.dump(history, f)
    
    print(f"\nTraining completed!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    
    return history


def main():
    parser = argparse.ArgumentParser(description='Train action recognition model')
    parser.add_argument('--model', type=str, default='baseline', 
                       choices=['baseline', 'st_gcn'],
                       help='Model to train')
    parser.add_argument('--data_path', type=str, 
                       default='data/UCF101 Module 2 Deep Learning.pkl',
                       help='Path to dataset pickle file')
    parser.add_argument('--split', type=str, default='train1',
                       choices=['train1', 'train2', 'train3'],
                       help='Dataset split to use')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loader workers')
    parser.add_argument('--checkpoint_dir', type=str, default='results',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use (cuda/cpu/auto)')
    
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
    
    train_loader, test_loader = get_data_loaders(
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
    
    model = model.to(device)
    
    # Create checkpoint directory with model and split name
    checkpoint_dir = os.path.join(
        args.checkpoint_dir, 
        f"{args.model}_{args.split.replace('train', 'split')}"
    )
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Train
    history = train(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        num_epochs=args.epochs,
        device=device,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_dir=checkpoint_dir
    )
    
    print("\nTraining history saved!")


if __name__ == '__main__':
    main()

