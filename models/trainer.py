# Training Functions for CIFAR-10
import torch
import torch.nn.functional as F
from tqdm import tqdm


class Trainer:
    """Training utilities for CIFAR-10 model"""
    
    def __init__(self):
        pass
    
    def train(self, model, device, train_loader, optimizer, epoch):
        """
        Train the model for one epoch
        
        Args:
            model: PyTorch model
            device: Training device
            train_loader: Training data loader
            optimizer: Optimizer
            epoch: Current epoch number
            
        Returns:
            dict: Training results including losses and accuracies
        """
        model.train()
        
        losses = []
        accuracies = []
        correct = 0
        processed = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}')
        
        for batch_idx, (data, target) in enumerate(pbar):
            data, target = data.to(device), target.to(device)
            
            # Zero gradients
            optimizer.zero_grad()
            
            # Forward pass
            output = model(data)
            
            # Calculate loss
            loss = F.nll_loss(output, target)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Track statistics
            losses.append(loss.item())
            
            # Calculate accuracy
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            processed += len(data)
            
            accuracy = 100. * correct / processed
            accuracies.append(accuracy)
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.6f}',
                'Acc': f'{accuracy:.2f}%'
            })
        
        final_accuracy = 100. * correct / processed
        avg_loss = sum(losses) / len(losses)
        
        print(f"\nTraining Epoch {epoch+1} Complete:")
        print(f"  Average Loss: {avg_loss:.6f}")
        print(f"  Accuracy: {final_accuracy:.2f}%\n")
        
        return {
            'losses': losses,
            'accuracies': accuracies,
            'final_accuracy': final_accuracy,
            'average_loss': avg_loss
        }


# Legacy function for backward compatibility
def train(model, device, train_loader, optimizer, epoch, train_losses, train_acc):
    """Legacy training function"""
    trainer = Trainer()
    results = trainer.train(model, device, train_loader, optimizer, epoch)
    train_losses.extend(results['losses'])
    train_acc.extend(results['accuracies'])
