# Testing Functions for CIFAR-10
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, confusion_matrix


class Tester:
    """Testing utilities for CIFAR-10 model"""
    
    def __init__(self):
        pass
    
    def test(self, model, device, test_loader, epoch=None):
        """
        Test the model
        
        Args:
            model: PyTorch model
            device: Testing device
            test_loader: Test data loader
            epoch: Current epoch number (optional)
            
        Returns:
            dict: Testing results
        """
        model.eval()
        test_loss = 0
        correct = 0
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                
                # Sum up batch loss
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                
                # Get predictions
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        
        # Calculate metrics
        test_loss /= len(test_loader.dataset)
        accuracy = 100. * correct / len(test_loader.dataset)
        
        # Print results
        if epoch is not None:
            print(f"\nTest Epoch {epoch+1}:")
        else:
            print("\nModel Testing:")
        
        print(f"  Average Loss: {test_loss:.6f}")
        print(f"  Accuracy: {accuracy:.2f}% ({correct}/{len(test_loader.dataset)})\n")
        
        return {
            'loss': test_loss,
            'accuracy': accuracy,
            'correct': correct,
            'total': len(test_loader.dataset)
        }
    
    def generate_classification_report(self, model, test_loader, classes, device):
        """
        Generate comprehensive classification report
        
        Args:
            model: Trained PyTorch model
            test_loader: Test data loader
            classes: List of class names
            device: Device to run inference on
            
        Returns:
            dict: Classification report with metrics
        """
        model.eval()
        
        all_predictions = []
        all_labels = []
        
        print("Generating predictions for all test samples...")
        with torch.no_grad():
            for images, labels in test_loader:
                images = images.to(device)
                outputs = model(images)
                predictions = outputs.argmax(dim=1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        # Calculate overall accuracy
        accuracy = sum(1 for p, l in zip(all_predictions, all_labels) if p == l) / len(all_predictions)
        
        print(f"\nOverall Test Accuracy: {accuracy:.2%}")
        print("\nClassification Report:")
        print(classification_report(all_labels, all_predictions, 
                                 target_names=classes, 
                                 digits=3))
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'confusion_matrix': confusion_matrix(all_labels, all_predictions)
        }
    
    def predict_single_image(self, model, image, device, classes=None):
        """
        Make prediction on a single image
        
        Args:
            model: Trained PyTorch model
            image: Preprocessed image tensor
            device: Device to run inference on
            classes: List of class names (optional)
            
        Returns:
            dict: Prediction results
        """
        model.eval()
        
        with torch.no_grad():
            image_batch = image.unsqueeze(0).to(device)
            output = model(image_batch)
            prediction = output.argmax(dim=1).item()
            confidence = torch.softmax(output, dim=1).max().item()
            
            result = {
                'prediction': prediction,
                'confidence': confidence
            }
            
            if classes:
                result['class_name'] = classes[prediction]
                
            return result


# Legacy function for backward compatibility
def test(model, device, test_loader, test_losses, test_acc):
    """Legacy testing function"""
    tester = Tester()
    results = tester.test(model, device, test_loader)
    test_losses.append(results['loss'])
    test_acc.append(results['accuracy'])
