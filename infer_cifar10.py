#!/usr/bin/env python3
"""
CIFAR-10 Model Inference Script

This script loads a trained model and performs inference on test images,
providing detailed analysis and validation capabilities.

Usage:
    python infer_cifar10.py --model_path outputs/cifar10_model.pth
"""

import argparse
import os
import torch
import numpy as np
from PIL import Image
import sys

# Add project modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.session7_model import CIFAR10Net
from dataset.session7_dataset import get_transforms, get_datasets
from utils.session7_utils import (
    show_misclassified_images, 
    validate_model_predictions, 
    generate_classification_report,
    plot_training_curves
)


def load_trained_model(model_path, device='cpu'):
    """
    Load a trained model from checkpoint
    
    Args:
        model_path: Path to saved model checkpoint
        device: Device to load model on
        
    Returns:
        model: Loaded and initialized model
        checkpoint_info: Dictionary with checkpoint metadata
    """
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Create model instance
    model = CIFAR10Net(
        dropout_value=checkpoint.get('dropout_value', 0.1),
        num_classes=len(checkpoint.get('classes', []) or 10)
    )
    
    # Load state dictionary
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded successfully!")
    print(f"Parameters: {model.count_parameters():,}")
    
    return model, checkpoint


def setup_data_for_inference(norm_mean, norm_std):
    """Setup test data for inference"""
    print("Setting up test data...")
    
    # Get test transforms (no augmentation)
    _, test_transform = get_transforms(norm_mean, norm_std)
    
    # Get test dataset
    _, test_set = get_datasets(test_transform, test_transform)
    
    # Create test loader
    test_loader = torch.utils.data.DataLoader(
        test_set, 
        batch_size=128, 
        shuffle=False, 
        num_workers=2
    )
    
    return test_loader, test_set.classes


def perform_inference(model, test_loader, classes, device, checkpoint_info):
    """Perform comprehensive inference and analysis"""
    print("\nPerforming inference and analysis...")
    print("=" * 50)
    
    # Validate model predictions on random samples
    print("\n1. Random Sample Validation:")
    validate_model_predictions(model, test_loader, classes, device, num_samples=10)
    
    # Generate classification report
    print("\n2. Classification Report:")
    report = generate_classification_report(model, test_loader, classes, device)
    
    # Show misclassified images
    print("\n3. Misclassified Images Analysis:")
    undo_normalization = None
    if 'norm_mean' in checkpoint_info and 'norm_std' in checkpoint_info:
        undo_normalization = (checkpoint_info['norm_mean'], checkpoint_info['norm_std'])
    
    show_misclassified_images(
        model, classes, test_loader, 
        num_of_images=20, 
        undo_normalization=undo_normalization
    )
    
    return report


def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 Model Inference')
    parser.add_argument('--model_path', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device for inference')
    parser.add_argument('--num_samples', type=int, default=10,
                       help='Number of random samples to validate')
    
    args = parser.parse_args()
    
    # Setup device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Inference device: {device}")
    
    try:
        # Load model
        model, checkpoint_info = load_trained_model(args.model_path, device)
        
        # Setup data
        norm_mean = checkpoint_info.get('norm_mean', (0.49, 0.48, 0.45))
        norm_std = checkpoint_info.get('norm_std', (0.25, 0.24, 0.26))
        test_loader, classes = setup_data_for_inference(norm_mean, norm_std)
        
        print(f"Dataset: {len(test_loader)} batches, {len(test_loader.dataset)} samples")
        print(f"Classes: {classes}")
        
        # Perform inference
        report = perform_inference(model, test_loader, classes, device, checkpoint_info)
        
        # Print summary
        print("\n" + "=" * 60)
        print("INFERENCE SUMMARY")
        print("=" * 60)
        print(f"Overall Accuracy: {report['accuracy']:.2%}")
        print(f"Model Parameters: {model.count_parameters():,}")
        
        # Check if targets were met
        print(f"\nTarget Requirements:")
        print(f"85%+ Accuracy: {'✓' if report['accuracy'] >= 0.85 else '✗'}")
        print(f"<200k Parameters: {'✓' if model.count_parameters() < 200000 else '✗'}")
        
    except Exception as e:
        print(f"Error during inference: {str(e)}")
        return 1
    
    print("\nInference completed successfully!")
    return 0


if __name__ == "__main__":
    exit(main())