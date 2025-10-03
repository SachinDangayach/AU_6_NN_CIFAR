#!/usr/bin/env python3
"""
CIFAR-10 Training Script

This script trains a CNN model on CIFAR-10 dataset with advanced convolutional techniques.
Uses the modularized components for clean, maintainable code.

Requirements:
- Target accuracy: 85%
- Total parameters: < 200k
- Receptive field: > 44
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
from torchsummary import summary

# Import our modularized components
from models.cifar10_net import CIFAR10Net
from models.trainer import Trainer
from models.tester import Tester
from data.dataset import CIFAR10DataLoader
from utils.visualization import TrainerVisualizer
from config import Config


def setup_device():
    """Setup device (GPU/CPU) for training"""
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")
    if use_cuda:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def create_model(dropout_value=0.1, num_classes=10):
    """Create and initialize the CIFAR-10 model"""
    model = CIFAR10Net(dropout_value=dropout_value, num_classes=num_classes)
    print(f"Model created with {model.count_parameters():,} parameters")
    return model


def setup_data(norm_mean, norm_std, batch_size=128):
    """Setup data loaders with albumentations"""
    print("Setting up data loading...")
    
    data_loader = CIFAR10DataLoader(
        data_path='./data',
        batch_size=batch_size,
        num_workers=4 if torch.cuda.is_available() else 1
    )
    
    train_loader, test_loader, classes, norm_params = data_loader.get_data()
    
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    print(f"Batch size: {batch_size}")
    print(f"Classes: {classes}")
    
    return train_loader, test_loader, classes, norm_params


def setup_optimizer(model, lr=0.1, momentum=0.9, weight_decay=1e-4):
    """Setup optimizer and scheduler"""
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    return optimizer, scheduler


def train_model(model, device, train_loader, test_loader, optimizer, scheduler, epochs=20):
    """Train the model and return training metrics"""
    print(f"\nStarting training for {epochs} epochs...")
    
    trainer = Trainer()
    tester = Tester()
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    best_test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
        print("-" * 50)
        
        # Training
        train_results = trainer.train(model, device, train_loader, optimizer, epoch)
        train_losses.extend(train_results['losses'])
        train_accuracies.extend(train_results['accuracies'])
        
        # Testing
        test_results = tester.test(model, device, test_loader, epoch)
        test_losses.append(test_results['loss'])
        test_accuracies.append(test_results['accuracy'])
        
        # Step learning rate
        scheduler.step()
        
        # Track best accuracy
        current_test_acc = test_results['accuracy']
        if current_test_acc > best_test_acc:
            best_test_acc = current_test_acc
            best_epoch = epoch + 1
        
        print(f"Current Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
    
    print(f"\nTraining completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")
    
    return train_losses, train_accuracies, test_losses, test_accuracies


def save_model_and_results(model, train_losses, train_accuracies, test_losses, test_accuracies, 
                          classes, norm_mean, norm_std, output_dir="outputs"):
    """Save trained model and results"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, "cifar10_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': CIFAR10Net,
        'dropout_value': 0.1,
        'classes': classes,
        'norm_mean': norm_mean,
        'norm_std': norm_std
    }, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save training history
    history_path = os.path.join(output_dir, "training_history.pth")
    torch.save({
        'train_losses': train_losses,
        'train_accuracies': train_accuracies,
        'test_losses': test_losses,
        'test_accuracies': test_accuracies
    }, history_path)
    print(f"Training history saved to: {history_path}")
    
    return model_path, history_path


def main():
    parser = argparse.ArgumentParser(description='Train CIFAR-10 CNN with Advanced Convolutions')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout value')
    parser.add_argument('--output_dir', type=str, default='outputs', help='Output directory')
    parser.add_argument('--show_model_summary', action='store_true', help='Show model summary')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("CIFAR-10 CNN Training with Advanced Convolutions")
    print("=" * 60)
    
    # Setup device
    device = setup_device()
    
    # Get normalization statistics
    print("\nCalculating dataset normalization statistics...")
    data_loader = CIFAR10DataLoader()
    norm_mean, norm_std = data_loader.get_cifar10_stats()
    
    # Create model
    print("\nCreating model...")
    model = create_model(dropout_value=args.dropout)
    model = model.to(device)
    
    if args.show_model_summary:
        print("\nModel Summary:")
        summary(model, input_size=(3, 32, 32))
    
    # Setup data
    train_loader, test_loader, classes, norm_params = setup_data(norm_mean, norm_std, args.batch_size)
    
    # Setup optimizer
    optimizer, scheduler = setup_optimizer(model, args.lr, args.momentum, args.weight_decay)
    
    # Train model
    train_losses, train_accuracies, test_losses, test_accuracies = train_model(
        model, device, train_loader, test_loader, optimizer, scheduler, args.epochs
    )
    
    # Generate results and visualizations
    print("\nGenerating results and visualizations...")
    
    visualizer = TrainerVisualizer()
    tester = Tester()
    
    # Plot training curves
    visualizer.plot_training_curves(train_losses, train_accuracies, test_losses, test_accuracies, 
                                   save_path=os.path.join(args.output_dir, "training_curves.png"))
    
    # Model prediction validation
    visualizer.validate_model_predictions(model, test_loader, classes, device, 
                                          undo_normalization=norm_params)
    
    # Generate classification report
    report = tester.generate_classification_report(model, test_loader, classes, device)
    
    # Show misclassified images
    visualizer.show_misclassified_images(model, classes, test_loader, 
                                        undo_normalization=norm_params)
    
    # Save model and results
    model_path, history_path = save_model_and_results(
        model, train_losses, train_accuracies, test_losses, test_accuracies,
        classes, norm_mean, norm_std, args.output_dir
    )
    
    # Final summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Final Training Accuracy: {train_accuracies[-1]:.2f}%")
    print(f"Final Test Accuracy: {test_accuracies[-1]:.2f}%")
    print(f"Total Parameters: {model.count_parameters():,}")
    print(f"Target Accuracy Achieved: {'✓' if test_accuracies[-1] >= 85.0 else '✗'}")
    print(f"Parameter Count Below 200k: {'✓' if model.count_parameters() < 200000 else '✗'}")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()