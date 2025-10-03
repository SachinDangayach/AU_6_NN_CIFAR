#!/usr/bin/env python3
"""
CIFAR-10 Training Script (Command Line)

Streamlined command-line interface for training CIFAR-10 CNN with advanced convolutions.
This script uses the modularized components for clean, maintainable code.

Usage:
    python cifar10_trainer.py
    python cifar10_trainer.py --epochs 30 --batch_size 256
"""

import argparse
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR

# Import our modularized components
from models.cifar10_net import CIFAR10Net
from models.trainer import Trainer
from models.tester import Tester
from data.dataset import CIFAR10DataLoader
from utils.visualization import TrainerVisualizer
from config import Config


def setup_device():
    """Setup device for training"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device


def create_model(config):
    """Create CIFAR-10 model"""
    model = CIFAR10Net(dropout_value=config.MODEL_DROPOUT)
    print(f"Model created with {model.count_parameters():,} parameters")
    return model


def setup_data(config):
    """Setup data loaders"""
    print("Setting up data pipeline...")
    
    data_loader = CIFAR10DataLoader(
        data_path=config.TRAIN_DATA_PATH,
        batch_size=config.DEFAULT_BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    
    train_loader, test_loader, classes, norm_params = data_loader.get_data()
    
    print(f"Dataset: {len(train_loader.dataset):,} train, {len(test_loader.dataset):,} test")
    print(f"Classes: {classes}")
    
    return train_loader, test_loader, classes, norm_params


def setup_optimizer(model, config):
    """Setup optimizer and scheduler"""
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.DEFAULT_LEARNING_RATE,
        momentum=config.DEFAULT_MOMENTUM,
        weight_decay=config.DEFAULT_WEIGHT_DECAY
    )
    
    scheduler = StepLR(
        optimizer,
        step_size=config.LR_SCHEDULER_STEP_SIZE,
        gamma=config.LR_SCHEDULER_GAMMA
    )
    
    return optimizer, scheduler


def train_model(model, device, train_loader, test_loader, optimizer, scheduler, 
                trainer, tester, config):
    """Main training loop"""
    print(f"\nStarting training for {config.DEFAULT_EPOCHS} epochs...")
    print("=" * 60)
    
    train_losses = []
    train_accuracies = []
    test_losses = []
    test_accuracies = []
    
    best_test_acc = 0.0
    best_epoch = 0
    
    for epoch in range(config.DEFAULT_EPOCHS):
        print(f"\n📅 EPOCH {epoch + 1}/{config.DEFAULT_EPOCHS}")
        print("-" * 40)
        
        # Training
        train_results = trainer.train(model, device, train_loader, optimizer, epoch)
        train_losses.extend(train_results['losses'])
        train_accuracies.extend(train_results['accuracies'])
        
        # Testing
        test_results = tester.test(model, device, test_loader, epoch)
        test_losses.append(test_results['loss'])
        test_accuracies.append(test_results['accuracy'])
        
        # Learning rate scheduling
        scheduler.step()
        
        # Track best accuracy
        current_test_acc = test_results['accuracy']
        if current_test_acc > best_test_acc:
            best_test_acc = current_test_acc
            best_epoch = epoch + 1
        
        # Print summary
        print(f"📈 Results:")
        print(f"  Train Acc: {train_results['final_accuracy']:.2f}%")
        print(f"  Test Acc: {current_test_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        print(f"  Best: {best_test_acc:.2f}% (Epoch {best_epoch})")
    
    print(f"\n🎉 Training completed!")
    print(f"Best test accuracy: {best_test_acc:.2f}% at epoch {best_epoch}")
    
    return train_losses, train_accuracies, test_losses, test_accuracies, best_test_acc


def main():
    parser = argparse.ArgumentParser(
        description='Train CIFAR-10 CNN with Advanced Convolutions',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size for training')
    parser.add_argument('--lr', type=float, default=0.1,
                       help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9,
                       help='SGD momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout probability')
    parser.add_argument('--output_dir', type=str, default='runs',
                       help='Output directory for results')
    parser.add_argument('--show_config', action='store_true',
                       help='Show configuration and exit')
    
    args = parser.parse_args()
    
    # Create custom config from args
    config = Config()
    config.DEFAULT_EPOCHS = args.epochs
    config.DEFAULT_BATCH_SIZE = args.batch_size
    config.DEFAULT_LEARNING_RATE = args.lr
    config.DEFAULT_MOMENTUM = args.momentum
    config.DEFAULT_WEIGHT_DECAY = args.weight_decay
    config.MODEL_DROPOUT = args.dropout
    config.DEFAULT_OUTPUT_DIR = args.output_dir
    
    print("🚀 CIFAR-10 CNN with Advanced Convolutions")
    print("=" * 60)
    
    if args.show_config:
        config.print_config()
        return
    
    # Setup
    device = setup_device()
    model = create_model(config)
    train_loader, test_loader, classes, norm_params = setup_data(config)
    optimizer, scheduler = setup_optimizer(model, config)
    
    # Initialize training components
    trainer = Trainer()
    tester = Tester()
    visualizer = TrainerVisualizer()
    
    # Train model
    training_history = train_model(
        model, device, train_loader, test_loader, 
        optimizer, scheduler, trainer, tester, config
    )
    
    train_losses, train_accuracies, test_losses, test_accuracies, best_acc = training_history
    
    # Visualize results
    print("\nCreating visualizations...")
    os.makedirs(args.output_dir, exist_ok=True)
    
    visualizer.plot_training_curves(
        train_losses, train_accuracies, test_losses, test_accuracies,
        save_path=os.path.join(args.output_dir, "training_curves.png")
    )
    
    # Model validation
    visualizer.validate_model_predictions(
        model, test_loader, classes, device,
        undo_normalization=norm_params, num_samples=15
    )
    
    # Classification report
    report = tester.generate_classification_report(model, test_loader, classes, device)
    
    # Misclassified images
    visualizer.show_misclassified_images(
        model, classes, test_loader,
        undo_normalization=norm_params, num_of_images=20
    )
    
    # Save model
    model_path = os.path.join(args.output_dir, "cifar10_model.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_class': CIFAR10Net,
        'config': config.__dict__,
        'classes': classes,
        'norm_params': norm_params,
        'final_accuracy': report['accuracy'],
        'training_history': training_history[:-1]  # Exclude best_acc
    }, model_path)
    
    # Final summary
    print("\n" + "=" * 60)
    print("🏆 TRAINING SUMMARY")
    print("=" * 60)
    print(f"Final Test Accuracy: {best_acc:.2f}%")
    print(f"Model Parameters: {model.count_parameters():,}")
    print(f"Target Accuracy Achieved: {'✅' if best_acc >= 85.0 else '❌'}")
    print(f"Param Limit Met: {'✅' if model.count_parameters() < 200000 else '❌'}")
    print(f"Model saved to: {model_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
