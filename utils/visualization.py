# Visualization and Analysis Utilities
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
import seaborn as sns


class TrainerVisualizer:
    """Visualization utilities for CIFAR-10 training"""
    
    def __init__(self):
        plt.style.use('default')
        sns.set_palette("husl")
    
    def plot_training_curves(self, train_losses, train_accuracies, test_losses, test_accuracies, 
                           title="CIFAR-10 Training Results", save_path=None):
        """
        Plot comprehensive training and validation curves
        
        Args:
            train_losses: List of training losses per batch
            train_accuracies: List of training accuracies per batch
            test_losses: List of test losses per epoch
            test_accuracies: List of test accuracies per epoch
            title: Plot title
            save_path: Optional path to save the plot
        """
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(title, fontsize=16, fontweight='bold')
        
        # Training Loss
        axs[0, 0].plot(train_losses, alpha=0.7, linewidth=1)
        axs[0, 0].set_title("Training Loss", fontsize=14, fontweight='bold')
        axs[0, 0].set_xlabel("Batch")
        axs[0, 0].set_ylabel("Loss")
        axs[0, 0].grid(True, alpha=0.3)
        
        # Training Accuracy
        axs[1, 0].plot(train_accuracies, 'b-', alpha=0.7, linewidth=1)
        axs[1, 0].set_title("Training Accuracy", fontsize=14, fontweight='bold')
        axs[1, 0].set_xlabel("Batch")
        axs[1, 0].set_ylabel("Accuracy (%)")
        axs[1, 0].grid(True, alpha=0.3)
        
        # Test Loss
        axs[0, 1].plot(test_losses, 'r-', linewidth=2, marker='o')
        axs[0, 1].set_title("Test Loss", fontsize=14, fontweight='bold')
        axs[0, 1].set_xlabel("Epoch")
        axs[0, 1].set_ylabel("Loss")
        axs[0, 1].grid(True, alpha=0.3)
        
        # Test Accuracy
        axs[1, 1].plot(test_accuracies, 'r-', linewidth=2, marker='o')
        axs[1, 1].set_title("Test Accuracy", fontsize=14, fontweight='bold')
        axs[1, 1].set_xlabel("Epoch")
        axs[1, 1].set_ylabel("Accuracy (%)")
        axs[1, 1].grid(True, alpha=0.3)
        
        # Add final values to plots
        if train_losses:
            axs[0, 0].text(0.02, 0.98, f'Final: {train_losses[-1]:.4f}', 
                          transform=axs[0, 0].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='wheat', alpha=0.8))
        
        if test_accuracies:
            axs[1, 1].text(0.02, 0.98, f'Final: {test_accuracies[-1]:.2f}%', 
                          transform=axs[1, 1].transAxes, verticalalignment='top',
                          bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen', alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def show_sample_images(self, data_loader, classes, undo_normalization=None, num_images=12):
        """
        Display sample images from dataset
        
        Args:
            data_loader: Data loader
            classes: List of class names
            undo_normalization: Tuple of (mean, std) for denormalization
            num_images: Number of images to show
        """
        itr = iter(data_loader)
        images, labels = next(itr)
        
        if num_images > len(images):
            num_images = len(images)
        
        rows = (num_images + 3) // 4
        plt.figure(figsize=(15, 4 * rows))
        
        for i in range(num_images):
            plt.subplot(rows, 4, i + 1)
            
            img = images[i]
            if undo_normalization:
                mean = torch.tensor(undo_normalization['mean']).view(3, 1, 1)
                std = torch.tensor(undo_normalization['std']).view(3, 1, 1)
                img = img * std + mean
            
            img = torch.clamp(img, 0, 1)
            plt.imshow(img.permute(1, 2, 0))
            plt.title(classes[labels[i]], fontsize=12)
            plt.axis('off')
        
        plt.suptitle('Sample Images from Dataset', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
    
    def show_misclassified_images(self, model, classes, test_loader, num_of_images=20, 
                                 undo_normalization=None):
        """
        Display misclassified images with improved visualization
        
        Args:
            model: Trained PyTorch model
            classes: List of class names
            test_loader: Test data loader
            num_of_images: Number of misclassified images to display
            undo_normalization: Tuple of (mean, std) for denormalization
        """
        model.eval()
        
        misclassified_data = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                
                for i in range(len(images)):
                    if predictions[i] != labels[i]:
                        misclassified_data.append({
                            'image': images[i],
                            'true_label': labels[i],
                            'predicted_label': predictions[i].item(),
                            'confidence': torch.softmax(outputs[i], dim=0).max().item()
                        })
                        
                        if len(misclassified_data) >= num_of_images:
                            break
                
                if len(misclassified_data) >= num_of_images:
                    break
        
        if not misclassified_data:
            print("No misclassified images found!")
            return
        
        print(f"Found {len(misclassified_data)} misclassified images. Showing {min(num_of_images, len(misclassified_data))} samples.")
        
        cols = min(4, len(misclassified_data))
        rows = (len(misclassified_data) + cols - 1) // cols
        
        plt.figure(figsize=(15, 4 * rows))
        plt.suptitle('Misclassified Images Analysis', fontsize=16, fontweight='bold')
        
        for i, data in enumerate(misclassified_data[:num_of_images]):
            plt.subplot(rows, cols, i + 1)
            
            img = data['image']
            if undo_normalization:
                mean = torch.tensor(undo_normalization['mean']).view(3, 1, 1)
                std = torch.tensor(undo_normalization['std']).view(3, 1, 1)
                img = img * std + mean
            
            img = torch.clamp(img, 0, 1)
            plt.imshow(img.permute(1, 2, 0))
            
            plt.title(f'True: {classes[data["true_label"]]}\\nPred: {classes[data["predicted_label"]]}\\nConf: {data["confidence"]:.2f}', 
                     fontsize=10, color='red')
            plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def validate_model_predictions(self, model, test_loader, classes, device, 
                                  num_samples=10, undo_normalization=None):
        """
        Validate model predictions on random samples
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            classes: List of class names
            device: Device to run inference on
            num_samples: Number of samples to validate
            undo_normalization: Tuple of (mean, std) for denormalization
        """
        model.eval()
        
        # Get random samples
        total_samples = len(test_loader.dataset)
        sample_indices = torch.randperm(total_samples)[:num_samples]
        
        cols = 3
        rows = (num_samples + cols - 1) // cols
        
        plt.figure(figsize=(15, 4 * rows))
        plt.suptitle('Model Prediction Validation', fontsize=16, fontweight='bold')
        
        with torch.no_grad():
            for i, idx in enumerate(sample_indices):
                # Get sample from dataset
                image, true_label = test_loader.dataset[idx]
                
                # Add batch dimension and move to device
                image_batch = image.unsqueeze(0).to(device)
                
                # Get prediction
                output = model(image_batch)
                predicted_label = output.argmax(dim=1).item()
                confidence = torch.softmax(output, dim=1).max().item()
                
                # Plot result
                plt.subplot(rows, cols, i + 1)
                
                # Denormalize image for display
                img_display = image
                if undo_normalization:
                    mean = torch.tensor(undo_normalization['mean']).view(3, 1, 1)
                    std = torch.tensor(undo_normalization['std']).view(3, 1, 1)
                    img_display = img_display * std + mean
                
                img_display = torch.clamp(img_display, 0, 1)
                plt.imshow(img_display.permute(1, 2, 0))
                
                correct = predicted_label == true_label
                color = 'green' if correct else 'red'
                title = f'True: {classes[true_label]}\\nPred: {classes[predicted_label]}\\nConf: {confidence:.2f}'
                
                plt.title(title, fontsize=10, color=color, fontweight='bold')
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    def plot_confusion_matrix(self, report, classes, save_path=None):
        """
        Plot confusion matrix
        
        Args:
            report: Classification report from tester
            classes: List of class names
            save_path: Optional path to save the plot
        """
        cm = confusion_matrix(report['labels'], report['predictions'])
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.xlabel('Predicted', fontsize=12)
        plt.ylabel('Actual', fontsize=12)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Confusion matrix saved to {save_path}")
        
        plt.tight_layout()
        plt.show()


# Legacy functions for backward compatibility
def valid_accuracy_loss_plots(train_losses, train_acc, test_losses, test_acc):
    """Legacy function name for backward compatibility"""
    visualizer = TrainerVisualizer()
    visualizer.plot_training_curves(train_losses, train_acc, test_losses, test_acc)


def show_sample_images(data_loader, classes, mean=0.5, std=0.25, num_of_images=10, is_norm=True):
    """Legacy function for showing sample images"""
    visualizer = TrainerVisualizer()
    undo_normalization = None
    if is_norm:
        undo_normalization = {'mean': (mean,) * 3, 'std': (std,) * 3}
    visualizer.show_sample_images(data_loader, classes, undo_normalization, num_of_images)


def show_misclassified_images(model, classes, test_loader, num_of_images=20):
    """Legacy function for showing misclassified images"""
    visualizer = TrainerVisualizer()
    visualizer.show_misclassified_images(model, classes, test_loader, num_of_images)
